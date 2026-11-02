"""第四阶段验收脚本：FastAPI 双通道接入 + HITL 审批接口端到端测试。

用例一览
--------
============================  ================================================
用例                           验证点
============================  ================================================
1. 健康检查                    MySQL / 本地 Qdrant / 检查点三件套探活
2. 通道 A 全链路               报修 → WAITING_APPROVAL → 审批通过 → DISPATCHED
                               + 库存核销 + 报告 + trace 头透传
3. 审批幂等                    重复审批不重复扣库存，返回 replayed=True
4. 审批驳回                    工单 CLOSED、库存不变、报告体现驳回
5. 无需换件（E-500）           直接 COMPLETED，不产生工单
6. 通道 B Webhook              JSON 解包、上报城市兜底、同样挂起等审批
7. 参数与状态校验              422 / 404 / 409 三类错误语义
8. 控制台查询                  工单列表、备件库存、筛选与分页
9. CORS 与 Trace ID 透传      预检放行、请求体/请求头两种 trace 来源
============================  ================================================

运行::

    python scripts/test_api.py             # 全部用例（结束自动清理业务数据）
    python scripts/test_api.py --keep      # 保留工单与库存变动，便于人工核对
    python scripts/test_api.py --only 2,7  # 只跑指定用例

说明
----
* 用 ``fastapi.testclient.TestClient`` 走真实 ASGI 栈（含 lifespan、中间件、
  依赖注入），但不占用网络端口，适合本地/CI 反复执行；
* 除 ``--keep`` 外，脚本结束时会删除本轮生成的工单并还原库存基线。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# 保证从任意工作目录执行都能 import 到 src / scripts / main
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
for _path in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import text  # noqa: E402

import main as app_entry  # noqa: E402
from src.api.schemas import is_valid_trace_id  # noqa: E402
from src.api.routes import API_PREFIX  # noqa: E402
from src.core.db import session_scope  # noqa: E402
from src.core.logger import logger  # noqa: E402

# ---------------------------------------------------------------------------
# 测试常量
# ---------------------------------------------------------------------------
TEST_DEVICE_SN: str = "AGV-SZ-003"
TEST_LOCATION: str = "苏州"
TEST_PART: str = "型号A电磁阀"
TEST_WAREHOUSE: str = "苏州中心仓"
RAW_E201: str = "苏州厂区的 AGV-SZ-003 小车夹爪漏气合不上，报 E-201"
RAW_E500: str = "AGV-SZ-003 急停按钮被误拍下，面板报 E-500，请按手册指导复位"
UNKNOWN_SN: str = "AGV-XX-999"

#: 用例之间需要复用的上下文（trace_id / order_no 等）
_CTX: dict[str, Any] = {}
#: 本轮写入数据库的工单与库存基线，用于结束时回滚
_CREATED_ORDERS: list[str] = []
_STOCK_BASELINE: dict[tuple[str, str], int] = {}


# ---------------------------------------------------------------------------
# 数据库辅助
# ---------------------------------------------------------------------------
def _fetch_order(order_no: str) -> dict[str, Any] | None:
    with session_scope() as db:
        row = (
            db.execute(
                text(
                    "select order_no, device_sn, suggested_part, allocated_warehouse, "
                    "approval_status, approver, dispatch_status, trace_id "
                    "from work_orders where order_no = :order_no"
                ).bindparams(order_no=order_no)
            )
            .mappings()
            .first()
        )
    return dict(row) if row else None


def _fetch_stock(part_name: str, warehouse: str) -> int | None:
    with session_scope() as db:
        value = db.execute(
            text(
                "select stock_quantity from spare_parts "
                "where part_name = :part_name and warehouse_name = :warehouse_name"
            ).bindparams(part_name=part_name, warehouse_name=warehouse)
        ).scalar()
    return int(value) if value is not None else None


def _count_orders_by_trace(trace_id: str) -> int:
    with session_scope() as db:
        count = db.execute(
            text("select count(*) from work_orders where trace_id = :trace_id").bindparams(
                trace_id=trace_id
            )
        ).scalar()
    return int(count or 0)


def _count_orders() -> int:
    """统计 work_orders 总行数，用于校验列表接口返回的 ``total`` 是否与库一致。"""
    with session_scope() as db:
        count = db.execute(text("select count(*) from work_orders")).scalar()
    return int(count or 0)


def _track_order(order_no: str | None) -> None:
    if order_no and order_no not in _CREATED_ORDERS:
        _CREATED_ORDERS.append(order_no)


def _track_stock(part_name: str, warehouse: str) -> None:
    key: tuple[str, str] = (part_name, warehouse)
    if key in _STOCK_BASELINE:
        return
    current: int | None = _fetch_stock(part_name, warehouse)
    if current is not None:
        _STOCK_BASELINE[key] = current


def _cleanup(keep: bool) -> None:
    """删除本轮生成的工单并还原库存。"""
    if keep:
        logger.warning("--keep 已开启，保留工单 {} 与库存变动", _CREATED_ORDERS)
        return
    if not _CREATED_ORDERS and not _STOCK_BASELINE:
        return
    try:
        with session_scope() as db:
            for order_no in _CREATED_ORDERS:
                db.execute(
                    text("delete from work_orders where order_no = :order_no").bindparams(
                        order_no=order_no
                    )
                )
            for (part_name, warehouse), quantity in _STOCK_BASELINE.items():
                db.execute(
                    text(
                        "update spare_parts set stock_quantity = :quantity, updated_at = now() "
                        "where part_name = :part_name and warehouse_name = :warehouse"
                    ).bindparams(
                        quantity=quantity, part_name=part_name, warehouse=warehouse
                    )
                )
        logger.info("已清理测试工单 {}，库存已还原 {}", _CREATED_ORDERS, list(_STOCK_BASELINE.items()))
    except Exception as exc:  # noqa: BLE001 - 清理失败不应掩盖测试结论
        logger.error("清理测试数据失败（请手工核对）：{}", exc)


# ---------------------------------------------------------------------------
# 断言辅助
# ---------------------------------------------------------------------------
def _failed(checks: dict[str, bool]) -> list[str]:
    """返回不通过的断言名列表。"""
    return [name for name, ok in checks.items() if not ok]


def _json_text(payload: Any) -> str:
    """把响应体压成一行便于失败时排查。"""
    import json

    return json.dumps(payload, ensure_ascii=False)[:400]


# ---------------------------------------------------------------------------
# 用例 1：健康检查
# ---------------------------------------------------------------------------
def test_health(client: TestClient) -> tuple[bool, str]:
    """GET /health：三大依赖全部可用。"""
    logger.info("【用例 1】GET /health 健康检查")
    started: float = time.perf_counter()
    resp = client.get("/health")
    body: dict[str, Any] = resp.json()
    elapsed: float = (time.perf_counter() - started) * 1000

    checks: dict[str, bool] = {
        "http_200": resp.status_code == 200,
        "status_ok": body.get("status") == "ok",
        "mysql_ok": bool(body.get("mysql", {}).get("ok")),
        "qdrant_ok": bool(body.get("qdrant", {}).get("ok")),
        "checkpointer_ok": bool(body.get("checkpointer", {}).get("ok")),
        "version": body.get("version") == app_entry.settings.APP_VERSION,
        "trace_header": is_valid_trace_id(resp.headers.get("x-trace-id")),
    }
    failed: list[str] = _failed(checks)
    if failed:
        return False, f"{failed} | body={_json_text(body)}"
    return True, (
        f"mysql/qdrant/checkpointer 均 ok | qdrant={body['qdrant']['detail']} | "
        f"{elapsed:.0f}ms"
    )


# ---------------------------------------------------------------------------
# 用例 2：通道 A 全链路（报修 → 挂起 → 审批通过 → 派工）
# ---------------------------------------------------------------------------
def test_chat_full_flow(client: TestClient) -> tuple[bool, str]:
    """POST /diagnose/chat 挂起 → POST /work-orders/approve 通过并派工。"""
    logger.info("【用例 2】通道 A 全链路：报修 → 挂起 → 审批通过 → 派工")
    started: float = time.perf_counter()

    resp = client.post(f"{API_PREFIX}/diagnose/chat", json={"query": RAW_E201})
    if resp.status_code != 200:
        return False, f"报修接口返回 {resp.status_code}：{_json_text(resp.json())}"
    body: dict[str, Any] = resp.json()
    order_no: str | None = body.get("order_no")
    _track_order(order_no)
    _track_stock(TEST_PART, TEST_WAREHOUSE)
    stock_before: int | None = _fetch_stock(TEST_PART, TEST_WAREHOUSE)

    chat_checks: dict[str, bool] = {
        "status_waiting": body.get("status") == "WAITING_APPROVAL",
        "channel": body.get("channel") == "human_chat",
        "device_sn": body.get("device_sn") == TEST_DEVICE_SN,
        "location": body.get("device_location") == TEST_LOCATION,
        "error_code": body.get("error_code") == "E-201",
        "required_part": body.get("required_part") == TEST_PART,
        "warehouse": body.get("selected_warehouse") == TEST_WAREHOUSE,
        "order_no": bool(order_no),
        "approval_pending": body.get("approval_status") == "PENDING",
        "approval_payload": bool((body.get("approval_payload") or {}).get("order_no")),
        "sop_guide": bool(body.get("sop_guide")),
        "trace_header_matches": resp.headers.get("x-trace-id") == body.get("trace_id"),
        "trace_generated": is_valid_trace_id(body.get("trace_id")),
    }
    failed_chat: list[str] = _failed(chat_checks)
    if failed_chat:
        return False, f"挂起阶段校验失败：{failed_chat} | body={_json_text(body)}"

    draft: dict[str, Any] | None = _fetch_order(str(order_no))
    draft_checks: dict[str, bool] = {
        "db_pending": bool(draft and draft.get("approval_status") == "PENDING"),
        "db_init": bool(draft and draft.get("dispatch_status") == "INIT"),
        "db_trace": bool(draft and draft.get("trace_id") == body.get("trace_id")),
        "db_warehouse": bool(draft and draft.get("allocated_warehouse") == TEST_WAREHOUSE),
        "stock_not_locked": stock_before is not None
        and _fetch_stock(TEST_PART, TEST_WAREHOUSE) == stock_before,
    }
    failed_draft: list[str] = _failed(draft_checks)
    if failed_draft:
        return False, f"工单草稿校验失败：{failed_draft} | row={draft}"

    approve = client.post(
        f"{API_PREFIX}/work-orders/approve",
        json={
            "trace_id": body["trace_id"],
            "order_no": order_no,
            "approval_status": "APPROVED",
            "approver": "张主管",
        },
    )
    if approve.status_code != 200:
        return False, f"审批接口返回 {approve.status_code}：{_json_text(approve.json())}"
    approved: dict[str, Any] = approve.json()
    stock_after: int | None = _fetch_stock(TEST_PART, TEST_WAREHOUSE)
    row: dict[str, Any] | None = _fetch_order(str(order_no))
    report: str = str(approved.get("final_response") or "")

    approve_checks: dict[str, bool] = {
        "approved": approved.get("approval_status") == "APPROVED",
        "approver": approved.get("approver") == "张主管",
        "dispatched": approved.get("dispatch_status") == "DISPATCHED",
        "stock_deducted_flag": approved.get("stock_deducted") is True,
        "stock_db": stock_before is not None
        and stock_after is not None
        and stock_after == stock_before - 1,
        "not_replayed": approved.get("replayed") is False,
        "db_approved": bool(row and row.get("approval_status") == "APPROVED"),
        "db_dispatched": bool(row and row.get("dispatch_status") == "DISPATCHED"),
        "db_approver": bool(row and row.get("approver") == "张主管"),
        "report_header": "# 机器人售后服务处理报告" in report,
        "report_order": str(order_no) in report,
        "report_warehouse": TEST_WAREHOUSE in report,
        "report_dispatch": "DISPATCHED" in report,
    }
    failed_approve: list[str] = _failed(approve_checks)
    if failed_approve:
        return False, f"审批通过阶段校验失败：{failed_approve} | body={_json_text(approved)}"

    _CTX["trace_id"] = body["trace_id"]
    _CTX["order_no"] = order_no
    _CTX["stock_after_approve"] = stock_after
    elapsed: float = (time.perf_counter() - started) * 1000
    return True, (
        f"trace_id={body['trace_id']} order_no={order_no} 库存 {stock_before}→{stock_after} "
        f"审批=APPROVED/DISPATCHED 端到端 {elapsed:.0f}ms"
    )


# ---------------------------------------------------------------------------
# 用例 3：审批幂等
# ---------------------------------------------------------------------------
def test_approve_idempotent(client: TestClient) -> tuple[bool, str]:
    """重复审批同一工单：返回 replayed=True 且不重复扣库存。"""
    logger.info("【用例 3】重复审批（幂等性）")
    trace_id: str | None = _CTX.get("trace_id")
    order_no: str | None = _CTX.get("order_no")
    if not trace_id or not order_no:
        return False, "依赖用例 2 的上下文缺失"

    stock_before: int | None = _fetch_stock(TEST_PART, TEST_WAREHOUSE)
    resp = client.post(
        f"{API_PREFIX}/work-orders/approve",
        json={
            "trace_id": trace_id,
            "order_no": order_no,
            "approval_status": "APPROVED",
            "approver": "张主管",
        },
    )
    if resp.status_code != 200:
        return False, f"重复审批返回 {resp.status_code}：{_json_text(resp.json())}"
    body: dict[str, Any] = resp.json()
    stock_after: int | None = _fetch_stock(TEST_PART, TEST_WAREHOUSE)

    checks: dict[str, bool] = {
        "replayed": body.get("replayed") is True,
        "still_dispatched": body.get("dispatch_status") == "DISPATCHED",
        "no_double_deduct": body.get("stock_deducted") is False,
        "stock_unchanged": stock_before is not None and stock_after == stock_before,
        "reports_kept": bool(body.get("final_response")),
    }
    failed: list[str] = _failed(checks)
    if failed:
        return False, f"{failed} | body={_json_text(body)}"
    return True, f"replayed=True 库存保持 {stock_after} 件（未重复核销）"


# ---------------------------------------------------------------------------
# 用例 4：审批驳回
# ---------------------------------------------------------------------------
def test_approve_rejected(client: TestClient) -> tuple[bool, str]:
    """驳回分支：工单 CLOSED、库存不变、报告体现驳回。"""
    logger.info("【用例 4】审批驳回分支")
    chat = client.post(f"{API_PREFIX}/diagnose/chat", json={"query": RAW_E201})
    if chat.status_code != 200:
        return False, f"报修接口返回 {chat.status_code}：{_json_text(chat.json())}"
    chat_body: dict[str, Any] = chat.json()
    order_no: str | None = chat_body.get("order_no")
    _track_order(order_no)
    _track_stock(TEST_PART, TEST_WAREHOUSE)
    if not order_no:
        return False, "预期挂起并生成工单，实际未生成"

    stock_before: int | None = _fetch_stock(TEST_PART, TEST_WAREHOUSE)
    approve = client.post(
        f"{API_PREFIX}/work-orders/approve",
        json={
            "trace_id": chat_body["trace_id"],
            "order_no": order_no,
            "approval_status": "REJECTED",
            "approver": "李经理",
        },
    )
    if approve.status_code != 200:
        return False, f"审批接口返回 {approve.status_code}：{_json_text(approve.json())}"
    body: dict[str, Any] = approve.json()
    row: dict[str, Any] | None = _fetch_order(str(order_no))
    stock_after: int | None = _fetch_stock(TEST_PART, TEST_WAREHOUSE)
    report: str = str(body.get("final_response") or "")

    checks: dict[str, bool] = {
        "rejected": body.get("approval_status") == "REJECTED",
        "closed": body.get("dispatch_status") == "CLOSED",
        "no_deduct": body.get("stock_deducted") is False,
        "stock_unchanged": stock_before is not None and stock_after == stock_before,
        "db_rejected": bool(row and row.get("approval_status") == "REJECTED"),
        "db_closed": bool(row and row.get("dispatch_status") == "CLOSED"),
        "db_approver": bool(row and row.get("approver") == "李经理"),
        "report_reject": "驳回" in report and "CLOSED" in report,
    }
    failed: list[str] = _failed(checks)
    if failed:
        return False, f"{failed} | body={_json_text(body)}"
    return True, f"order_no={order_no} 审批=REJECTED/CLOSED 库存保持 {stock_after} 件"


# ---------------------------------------------------------------------------
# 用例 5：无需换件（E-500）
# ---------------------------------------------------------------------------
def test_chat_no_part(client: TestClient) -> tuple[bool, str]:
    """E-500 急停复位：直接 COMPLETED，不产生工单、不进入审批。"""
    logger.info("【用例 5】E-500 无需换件")
    resp = client.post(f"{API_PREFIX}/diagnose/chat", json={"query": RAW_E500})
    if resp.status_code != 200:
        return False, f"报修接口返回 {resp.status_code}：{_json_text(resp.json())}"
    body: dict[str, Any] = resp.json()
    report: str = str(body.get("final_response") or "")
    trace_id: str = str(body.get("trace_id") or "")

    checks: dict[str, bool] = {
        "completed": body.get("status") == "COMPLETED",
        "no_order": not body.get("order_no"),
        "no_part": not body.get("required_part"),
        "no_payload": body.get("approval_payload") is None,
        "no_db_order": _count_orders_by_trace(trace_id) == 0,
        "report_ready": "# 机器人售后服务处理报告" in report,
        "report_no_part": "无需更换备件" in report,
        "report_sop": "复位" in report,
    }
    failed: list[str] = _failed(checks)
    if failed:
        return False, f"{failed} | body={_json_text(body)}"
    _CTX["no_part_trace"] = trace_id
    return True, f"trace_id={trace_id} status=COMPLETED 无工单，报告 {len(report)} 字"


# ---------------------------------------------------------------------------
# 用例 6：通道 B Webhook
# ---------------------------------------------------------------------------
def test_webhook_channel(client: TestClient) -> tuple[bool, str]:
    """Webhook 告警：JSON 解包、上报城市兜底、挂起后按驳回收尾。"""
    logger.info("【用例 6】通道 B：设备 Webhook 告警")
    payload: dict[str, Any] = {
        "device_sn": UNKNOWN_SN,  # 台账中不存在 → 验证 location 兜底
        "location": "苏州",
        "error_code": "E-201",
        "symptom": "夹爪持续漏气、无法闭合，面板常亮 E-201",
    }
    resp = client.post(f"{API_PREFIX}/alert/webhook", json=payload)
    if resp.status_code != 200:
        return False, f"告警接口返回 {resp.status_code}：{_json_text(resp.json())}"
    body: dict[str, Any] = resp.json()
    order_no: str | None = body.get("order_no")
    _track_order(order_no)
    _track_stock(TEST_PART, TEST_WAREHOUSE)

    checks: dict[str, bool] = {
        "channel": body.get("channel") == "webhook_alert",
        "waiting": body.get("status") == "WAITING_APPROVAL",
        "sn": body.get("device_sn") == UNKNOWN_SN,
        "code": body.get("error_code") == "E-201",
        "location_fallback": body.get("device_location") == "苏州",
        "symptom_from_message": "E-201" in str(body.get("symptom")),
        "part": body.get("required_part") == TEST_PART,
        "warehouse": body.get("selected_warehouse") == TEST_WAREHOUSE,
        "order_no": bool(order_no),
    }
    failed: list[str] = _failed(checks)
    if failed:
        return False, f"{failed} | body={_json_text(body)}"

    # 收尾：驳回，避免留下待审批工单
    approve = client.post(
        f"{API_PREFIX}/work-orders/approve",
        json={
            "trace_id": body["trace_id"],
            "order_no": order_no,
            "approval_status": "REJECTED",
            "approver": "李经理",
        },
    )
    closed: bool = approve.status_code == 200 and approve.json().get("dispatch_status") == "CLOSED"
    if not closed:
        return False, f"Webhook 工单收尾失败：{_json_text(approve.json())}"
    return True, (
        f"order_no={order_no} 上报城市兜底=苏州、就近仓={TEST_WAREHOUSE}、已驳回收尾"
    )


# ---------------------------------------------------------------------------
# 用例 7：参数与状态校验
# ---------------------------------------------------------------------------
def test_validation_errors(client: TestClient) -> tuple[bool, str]:
    """422 / 404 / 409 三类错误语义。"""
    logger.info("【用例 7】参数与状态校验")
    trace_id: str | None = _CTX.get("trace_id")
    order_no: str | None = _CTX.get("order_no")
    no_part_trace: str | None = _CTX.get("no_part_trace")

    blank_query = client.post(f"{API_PREFIX}/diagnose/chat", json={"query": "   "})
    missing_query = client.post(f"{API_PREFIX}/diagnose/chat", json={})
    bad_decision = client.post(
        f"{API_PREFIX}/work-orders/approve",
        json={
            "trace_id": trace_id or "TR-FALLBACK-01",
            "order_no": order_no or "WO-X",
            "approval_status": "MAYBE",
        },
    )
    unknown_trace = client.post(
        f"{API_PREFIX}/work-orders/approve",
        json={
            "trace_id": "TR-NOT-EXIST-0001",
            "order_no": "WO-000000-0000",
            "approval_status": "APPROVED",
        },
    )
    mismatch = client.post(
        f"{API_PREFIX}/work-orders/approve",
        json={
            "trace_id": trace_id or "TR-FALLBACK-01",
            "order_no": "WO-000000-0000",
            "approval_status": "APPROVED",
        },
    )
    no_order_flow = client.post(
        f"{API_PREFIX}/work-orders/approve",
        json={
            "trace_id": no_part_trace or "TR-FALLBACK-02",
            "order_no": "WO-000000-0000",
            "approval_status": "APPROVED",
        },
    )
    bad_webhook = client.post(
        f"{API_PREFIX}/alert/webhook", json={"device_sn": "AGV-SZ-003", "error_code": "E-201"}
    )

    checks: dict[str, bool] = {
        "blank_query_422": blank_query.status_code == 422,
        "missing_query_422": missing_query.status_code == 422,
        "bad_decision_422": bad_decision.status_code == 422,
        "missing_webhook_symptom_422": bad_webhook.status_code == 422,
        "unknown_trace_404": unknown_trace.status_code == 404,
        "mismatch_409": mismatch.status_code == 409,
        "no_order_409": no_order_flow.status_code == 409,
        "error_shape": isinstance(unknown_trace.json().get("detail"), str),
    }
    failed: list[str] = _failed(checks)
    if failed:
        return False, (
            f"{failed} | blank={blank_query.status_code} missing={missing_query.status_code} "
            f"decision={bad_decision.status_code} unknown={unknown_trace.status_code} "
            f"mismatch={mismatch.status_code} no_order={no_order_flow.status_code} "
            f"webhook={bad_webhook.status_code}"
        )
    return True, "422（参数）/ 404（线程不存在）/ 409（工单不匹配、无需审批）语义正确"


# ---------------------------------------------------------------------------
# 用例 8：控制台查询接口
# ---------------------------------------------------------------------------
def test_query_endpoints(client: TestClient) -> tuple[bool, str]:
    """工单列表与备件库存查询（含筛选与分页）。"""
    logger.info("【用例 8】控制台查询：工单 / 备件库存")
    orders = client.get(f"{API_PREFIX}/work-orders", params={"limit": 3})
    if orders.status_code != 200:
        return False, f"工单列表返回 {orders.status_code}：{_json_text(orders.json())}"
    order_body: dict[str, Any] = orders.json()

    approved = client.get(
        f"{API_PREFIX}/work-orders", params={"limit": 2, "approval_status": "APPROVED"}
    )
    by_trace = client.get(
        f"{API_PREFIX}/work-orders", params={"trace_id": _CTX.get("trace_id", "TR-NONE")}
    )
    spare = client.get(f"{API_PREFIX}/spare-parts", params={"part_name": TEST_PART, "limit": 10})
    if spare.status_code != 200:
        return False, f"备件库存返回 {spare.status_code}：{_json_text(spare.json())}"
    spare_body: dict[str, Any] = spare.json()

    order_checks: dict[str, bool] = {
        "orders_200": orders.status_code == 200,
        "limit_respected": len(order_body.get("items", [])) == 3,
        # total 必须与库中真实行数一致；不写死具体条数，因为 mock 数据规模由
        # scripts/seed_mock_data.py 的 --orders 决定，硬编码阈值会让换种子就变红
        "total_matches_db": int(order_body.get("total") or 0) == _count_orders(),
        "total_gt_limit": int(order_body.get("total") or 0) > 3,
        "fields": all(
            {"order_no", "device_sn", "approval_status", "dispatch_status", "created_at"}
            <= set(item)
            for item in order_body.get("items", [])
        ),
        "created_at_formatted": all(
            isinstance(item.get("created_at"), str) for item in order_body.get("items", [])
        ),
        "approved_filter": approved.status_code == 200
        and all(i.get("approval_status") == "APPROVED" for i in approved.json().get("items", [])),
        "trace_filter": by_trace.status_code == 200
        and by_trace.json().get("total") == 1
        and by_trace.json()["items"][0]["order_no"] == _CTX.get("order_no"),
    }
    spare_checks: dict[str, bool] = {
        "spare_200": spare.status_code == 200,
        "has_part": any(item.get("warehouse_name") == TEST_WAREHOUSE for item in spare_body.get("items", [])),
        "price_float": all(
            isinstance(item.get("unit_price"), (int, float))
            for item in spare_body.get("items", [])
            if item.get("unit_price") is not None
        ),
        "has_quantity": all(
            isinstance(item.get("stock_quantity"), int) for item in spare_body.get("items", [])
        ),
        "total_ge_2": int(spare_body.get("total") or 0) >= 2,
    }
    failed: list[str] = _failed({**order_checks, **spare_checks})
    if failed:
        return False, (
            f"{failed} | orders={_json_text(order_body)} | spare={_json_text(spare_body)}"
        )
    return True, (
        f"工单总数 {order_body.get('total')}（返回 {len(order_body.get('items', []))} 条）、"
        f"备件 {spare_body.get('total')} 行、筛选与时间格式化正常"
    )


# ---------------------------------------------------------------------------
# 用例 9：CORS 与 Trace ID 透传
# ---------------------------------------------------------------------------
def test_cors_and_trace(client: TestClient) -> tuple[bool, str]:
    """CORS 预检放行 + trace_id 的三种来源（生成 / 请求头 / 请求体优先）。"""
    logger.info("【用例 9】CORS 与 Trace ID 透传")
    preflight = client.options(
        f"{API_PREFIX}/diagnose/chat",
        headers={
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,x-trace-id",
        },
    )

    header_trace = "TR-HEADER-TEST-0001"
    via_header = client.post(
        f"{API_PREFIX}/diagnose/chat",
        json={"query": RAW_E500},
        headers={"X-Trace-Id": header_trace, "Origin": "http://localhost:8501"},
    )
    body_trace = "TR-BODY-TEST-0001"
    via_body = client.post(
        f"{API_PREFIX}/diagnose/chat",
        json={"query": RAW_E500, "trace_id": body_trace},
        headers={"X-Trace-Id": "TR-HEADER-IGNORED-02"},
    )
    invalid_header = client.post(
        f"{API_PREFIX}/diagnose/chat",
        json={"query": RAW_E500},
        headers={"X-Trace-Id": "bad id with spaces"},
    )

    header_body: dict[str, Any] = via_header.json()
    body_body: dict[str, Any] = via_body.json()
    invalid_body: dict[str, Any] = invalid_header.json()

    checks: dict[str, bool] = {
        "preflight_200": preflight.status_code == 200,
        "allow_origin": preflight.headers.get("access-control-allow-origin")
        in ("*", "http://localhost:8501"),
        "allow_trace_header": "x-trace-id"
        in str(preflight.headers.get("access-control-allow-headers", "")).lower(),
        "expose_trace_header": "x-trace-id"
        in str(via_header.headers.get("access-control-expose-headers", "")).lower(),
        "header_trace_used": header_body.get("trace_id") == header_trace,
        "header_echoed": via_header.headers.get("x-trace-id") == header_trace,
        "body_beats_header": body_body.get("trace_id") == body_trace,
        "body_trace_echoed": via_body.headers.get("x-trace-id") == body_trace,
        "invalid_header_replaced": invalid_body.get("trace_id") != "bad id with spaces"
        and is_valid_trace_id(invalid_body.get("trace_id")),
        "completed": body_body.get("status") == "COMPLETED",
    }
    failed: list[str] = _failed(checks)
    if failed:
        return False, (
            f"{failed} | preflight={preflight.status_code} "
            f"allow={preflight.headers.get('access-control-allow-origin')} "
            f"expose={via_header.headers.get('access-control-expose-headers')} "
            f"header_trace={header_body.get('trace_id')} body_trace={body_body.get('trace_id')} "
            f"invalid={invalid_body.get('trace_id')}"
        )
    return True, (
        f"CORS 预检放行；trace 三来源正确（头={header_body.get('trace_id')}，"
        f"体优先={body_body.get('trace_id')}，非法头已替换）"
    )


# ---------------------------------------------------------------------------
# 汇总与主流程
# ---------------------------------------------------------------------------
def _report(results: list[tuple[str, bool, str]]) -> int:
    """打印汇总表并返回进程退出码。"""
    logger.info("=" * 78)
    logger.info("测试汇总")
    logger.info("-" * 78)
    for name, ok, detail in results:
        if ok:
            logger.success("✓ {:<38} {}", name, detail)
        else:
            logger.error("✗ {:<38} {}", name, detail)
    passed: int = sum(1 for _, ok, _ in results if ok)
    logger.info("-" * 78)
    logger.info("通过 {}/{}", passed, len(results))
    logger.info("=" * 78)
    return 0 if passed == len(results) else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="FastAPI 接入层端到端测试")
    parser.add_argument("--keep", action="store_true", help="保留生成的工单与库存变动")
    parser.add_argument("--only", type=str, default="", help="只跑指定用例编号，逗号分隔")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """运行全部接口用例。

    Returns:
        int: 0 表示全部通过，1 表示存在失败，2 表示环境不满足。
    """
    args: argparse.Namespace = parse_args(argv)
    logger.info("=" * 78)
    logger.info("Service Copilot 接入层端到端测试 | base_url=TestClient")
    logger.info("=" * 78)

    # 知识库缺失时自动引导灌库（复用第二阶段脚本）
    from test_tools import ensure_knowledge_base  # noqa: PLC0415

    ensure_knowledge_base()

    cases: tuple[tuple[str, Callable[[TestClient], tuple[bool, str]]], ...] = (
        ("1. 健康检查", test_health),
        ("2. 通道 A 全链路（挂起→通过→派工）", test_chat_full_flow),
        ("3. 审批幂等", test_approve_idempotent),
        ("4. 审批驳回", test_approve_rejected),
        ("5. 无需换件（E-500）", test_chat_no_part),
        ("6. 通道 B Webhook 告警", test_webhook_channel),
        ("7. 参数与状态校验", test_validation_errors),
        ("8. 控制台查询接口", test_query_endpoints),
        ("9. CORS 与 Trace 透传", test_cors_and_trace),
    )
    selected: set[str] = {item.strip() for item in args.only.split(",") if item.strip()}

    results: list[tuple[str, bool, str]] = []
    # TestClient 作为上下文管理器会执行 lifespan（启动探活 / 退出清理）
    with TestClient(app_entry.app) as client:
        try:
            for name, case in cases:
                if selected and name.split(".")[0] not in selected:
                    continue
                try:
                    ok, detail = case(client)
                except Exception as exc:  # noqa: BLE001 - 汇总所有异常
                    logger.exception("用例 {} 抛出异常", name)
                    ok, detail = False, f"异常：{type(exc).__name__}: {exc}"
                results.append((name, ok, detail))
        finally:
            _cleanup(keep=bool(args.keep))

    if not results:
        logger.error("没有匹配的用例（--only={}）", args.only)
        return 2
    return _report(results)


if __name__ == "__main__":
    raise SystemExit(main())
