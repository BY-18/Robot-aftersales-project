"""第三阶段验收脚本：LangGraph「排障与备件调度」状态机端到端测试。

覆盖的两条业务通道与三种收尾分支：

===========================  ==============================================
用例                         验证点
===========================  ==============================================
1. 图结构与路由自检          6 个节点、条件边路由、缺货跳过审批
2. E-201 全链路              挂起（HITL）→ 审批通过 → 派工 + 扣库存 + 报告
3. E-201 审批驳回            工单 CLOSED、库存不变、报告体现驳回
4. E-500 无需换件            直接出报告，不生成工单、不进审批
5. Webhook 通道              JSON 报文本地解包（不调用 LLM）
6. 跨进程挂起 / 恢复         checkpoints.sqlite 真实落盘，两个独立进程接力
===========================  ==============================================

运行::

    python scripts/test_graph.py                     # 全部用例（跑完自动清理业务数据）
    python scripts/test_graph.py --keep               # 保留工单与库存变动，便于人工核对
    python scripts/test_graph.py --only 2,6           # 只跑指定用例
    python scripts/test_graph.py --worker start --thread TR-X --raw "..."    # 子进程：跑到挂起
    python scripts/test_graph.py --worker resume --thread TR-X --decision APPROVED

说明
----
* 用例 2 在挂起后用**新建的 SqliteSaver 连接**恢复，等价于"重启进程后继续"，可证明
  状态确实落在 ``checkpoints.sqlite``；用例 6 用两个独立子进程做同等的强验证。
* 用例 6 会先把父进程的本地 Qdrant 客户端关掉：Qdrant 本地文件模式是进程独占锁，
  不释放的话子进程打不开 ``./qdrant_local_data``。
* 除 ``--keep`` 外，脚本结束时会删除本轮生成的工单，并把库存改回运行前的数值。
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# 保证从任意工作目录执行都能 import 到 src 包与 scripts 包
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
SCRIPT_PATH: Path = Path(__file__).resolve()
for _path in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from langgraph.types import Command  # noqa: E402
from sqlalchemy import text  # noqa: E402

from src.agents.graph import NODE_NAMES, build_graph, make_config  # noqa: E402
from src.agents.nodes import (  # noqa: E402
    should_check_stock,
    should_request_approval,
    stock_check_node,
)
from src.core.config import settings  # noqa: E402
from src.core.db import session_scope, verify_mysql_connection  # noqa: E402
from src.core.logger import logger, new_trace_id, trace_context  # noqa: E402
from src.tools.rag_tool import close_qdrant_client  # noqa: E402

# ---------------------------------------------------------------------------
# 测试常量
# ---------------------------------------------------------------------------
TEST_DEVICE_SN: str = "AGV-SZ-003"
TEST_LOCATION: str = "苏州"
TEST_PART: str = "型号A电磁阀"
TEST_WAREHOUSE: str = "苏州中心仓"
RAW_E201: str = "苏州装配线 AGV-SZ-003 报 E-201，夹爪不动，气缸一直持续漏气"
RAW_E500: str = "AGV-SZ-003 急停按钮被人员误拍下，控制面板报 E-500，需要现场复位"
WEBHOOK_E201: str = json.dumps(
    {
        "device_sn": "AGV-SZ-003",
        "error_code": "E-201",
        "message": "夹爪气缸持续漏气，面板报错 E-201",
        "severity": "HIGH",
    },
    ensure_ascii=False,
)

#: 子进程通信标记：worker 只把结果写成这一行 JSON（日志走 stderr，不会污染 stdout）
_MARKER: str = "__TEST_RESULT__"

# 本轮测试确实写入数据库的工单与库存基线，用于结束时回滚
_CREATED_ORDERS: list[str] = []
_STOCK_BASELINE: dict[tuple[str, str], int] = {}
_OPEN_CONNS: list[sqlite3.Connection] = []


# ---------------------------------------------------------------------------
# 数据库小工具（测试专用，不进入生产代码路径）
# ---------------------------------------------------------------------------
def _fetch_order(order_no: str) -> dict[str, Any] | None:
    """按工单号读取审批 / 派工状态。"""
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
    """读取某备件在某仓库的当前库存；行不存在时返回 ``None``。"""
    with session_scope() as db:
        value = db.execute(
            text(
                "select stock_quantity from spare_parts "
                "where part_name = :part_name and warehouse_name = :warehouse_name"
            ).bindparams(part_name=part_name, warehouse_name=warehouse)
        ).scalar()
    return int(value) if value is not None else None


def _count_orders_by_trace(trace_id: str) -> int:
    """统计某条 trace 下生成的工单数量（用于验证"没有生成工单"）。"""
    with session_scope() as db:
        count = db.execute(
            text("select count(*) from work_orders where trace_id = :trace_id").bindparams(
                trace_id=trace_id
            )
        ).scalar()
    return int(count or 0)


def _track_order(order_no: str | None) -> None:
    """登记本轮创建的工单号，便于结束时清理。"""
    if order_no and order_no not in _CREATED_ORDERS:
        _CREATED_ORDERS.append(order_no)


def _track_stock(part_name: str, warehouse: str) -> None:
    """登记库存基线（只记录第一次见到的值，即运行前的原始数量）。"""
    key: tuple[str, str] = (part_name, warehouse)
    if key in _STOCK_BASELINE:
        return
    current: int | None = _fetch_stock(part_name, warehouse)
    if current is not None:
        _STOCK_BASELINE[key] = current


# ---------------------------------------------------------------------------
# 图运行辅助
# ---------------------------------------------------------------------------
def _new_saver() -> Any:
    """新建一个独立的 SqliteSaver 连接（等价于"另一个进程重新打开检查点文件"）。"""
    from langgraph.checkpoint.sqlite import SqliteSaver

    path: Path = settings.checkpoint_db_abspath
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    _OPEN_CONNS.append(conn)
    return SqliteSaver(conn)


def _fresh_graph() -> Any:
    """用全新检查点连接编译图，验证状态来自磁盘而非进程内存。"""
    return build_graph(_new_saver())


def _start(trace_id: str, raw_input: str, channel: str = "human_chat") -> dict[str, Any]:
    """从零跑一条工单，直到审批挂起（或直接跑完）。"""
    app: Any = _fresh_graph()
    with trace_context(trace_id):
        return app.invoke(
            {"trace_id": trace_id, "channel": channel, "raw_input": raw_input},
            make_config(trace_id),
        )


def _resume(
    trace_id: str,
    status: str,
    approver: str,
) -> dict[str, Any]:
    """用**全新进程/连接**恢复挂起的工作流，返回最终状态。"""
    app: Any = _fresh_graph()
    with trace_context(trace_id):
        return app.invoke(
            Command(resume={"status": status, "approver": approver}),
            make_config(trace_id),
        )


def _interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    """取出 ``interrupt()`` 抛出的审批载荷。"""
    raw = result.get("__interrupt__")
    if not raw:
        return None
    first = raw[0]
    value = getattr(first, "value", None)
    return value if isinstance(value, dict) else None


# ---------------------------------------------------------------------------
# 用例 1：图结构、条件路由与缺货分支
# ---------------------------------------------------------------------------
def test_graph_shape() -> tuple[bool, str]:
    """校验节点注册、条件边路由与「全库缺货 → 跳过审批」分支。"""
    logger.info("【用例 1】图结构与条件路由自检")

    app: Any = _fresh_graph()
    graph_view = app.get_graph()
    node_names: set[str] = set(graph_view.nodes)
    missing: list[str] = [name for name in NODE_NAMES if name not in node_names]
    edge_pairs: set[tuple[str, str]] = {(edge.source, edge.target) for edge in graph_view.edges}
    required_edges: set[tuple[str, str]] = {
        ("parse_fault", "rag_diagnose"),
        ("rag_diagnose", "stock_check"),
        ("rag_diagnose", "generate_report"),
        ("stock_check", "human_approval"),
        ("stock_check", "generate_report"),
        ("human_approval", "dispatch_order"),
        ("dispatch_order", "generate_report"),
        ("generate_report", "__end__"),
    }
    missing_edges: list[tuple[str, str]] = sorted(required_edges - edge_pairs)
    if missing or missing_edges:
        return False, f"缺少节点={missing} 缺少边={missing_edges}"

    # 路由函数本身（无需跑 LLM）
    route_no_part: str = should_check_stock({"required_part": None})
    route_part: str = should_check_stock({"required_part": TEST_PART})
    route_no_order: str = should_request_approval({"order_no": None})
    route_order: str = should_request_approval({"order_no": "WO-TEST-0001"})

    # 缺货分支：用一个不存在的备件走真实节点，确认不落工单、直接出报告
    sold_out_state: dict[str, Any] = {
        "trace_id": new_trace_id("TR-GRAPH-SOLD"),
        "device_sn": TEST_DEVICE_SN,
        "device_location": TEST_LOCATION,
        "error_code": "E-888",
        "symptom": "缺货分支自检",
        "required_part": "不存在的备件-XYZ-999",
    }
    with trace_context(str(sold_out_state["trace_id"])):
        update: dict[str, Any] = stock_check_node(sold_out_state)  # type: ignore[arg-type]
    sold_out_ok: bool = not update.get("order_no") and should_request_approval(
        {**sold_out_state, **update}  # type: ignore[misc]
    ) == "generate_report"

    ok: bool = (
        route_no_part == "generate_report"
        and route_part == "stock_check"
        and route_no_order == "generate_report"
        and route_order == "human_approval"
        and sold_out_ok
        and _count_orders_by_trace(str(sold_out_state["trace_id"])) == 0
    )
    return ok, (
        f"节点 {len(NODE_NAMES)} 个 / 边 {len(edge_pairs)} 条 | "
        f"无件→{route_no_part} 有件→{route_part} 缺货跳过审批={sold_out_ok}"
    )


# ---------------------------------------------------------------------------
# 用例 2：E-201 全链路（挂起 → 审批通过 → 派工 + 扣库存）
# ---------------------------------------------------------------------------
def test_hitl_approve() -> tuple[bool, str]:
    """验证 HITL 挂起、跨连接恢复、工单派工与库存核销。"""
    trace_id: str = new_trace_id("TR-GRAPH-OK")
    logger.info("【用例 2】E-201 全链路 | trace_id={}", trace_id)
    started: float = time.perf_counter()

    paused: dict[str, Any] = _start(trace_id, RAW_E201)
    payload: dict[str, Any] | None = _interrupt_payload(paused)
    if payload is None:
        return False, f"期望在 human_approval 挂起，实际未挂起：{list(paused)}"

    order_no: str | None = paused.get("order_no")
    _track_order(order_no)
    _track_stock(TEST_PART, TEST_WAREHOUSE)
    stock_before: int | None = _fetch_stock(TEST_PART, TEST_WAREHOUSE)

    paused_checks: dict[str, bool] = {
        "extract_sn": paused.get("device_sn") == TEST_DEVICE_SN,
        "extract_code": paused.get("error_code") == "E-201",
        "device_location": paused.get("device_location") == TEST_LOCATION,
        "required_part": paused.get("required_part") == TEST_PART,
        "preferred_warehouse": payload.get("selected_warehouse") == TEST_WAREHOUSE,
        "order_no": bool(order_no),
        "approval_pending": paused.get("approval_status") == "PENDING",
        "interrupt_payload": payload.get("type") == "approval_required",
    }
    failed_pause: list[str] = [k for k, v in paused_checks.items() if not v]
    if failed_pause:
        return False, f"挂起阶段校验失败：{failed_pause} | state={paused.get('error_code')}"

    draft: dict[str, Any] | None = _fetch_order(str(order_no))
    draft_ok: bool = bool(
        draft
        and draft.get("approval_status") == "PENDING"
        and draft.get("dispatch_status") == "INIT"
        and draft.get("allocated_warehouse") == TEST_WAREHOUSE
        and draft.get("trace_id") == trace_id
    )
    stock_locked_ok: bool = stock_before is not None and (
        _fetch_stock(TEST_PART, TEST_WAREHOUSE) == stock_before
    )

    # 用全新连接恢复：等价于"进程重启后接着审批"
    final: dict[str, Any] = _resume(trace_id, "APPROVED", "张主管")
    row: dict[str, Any] | None = _fetch_order(str(order_no))
    stock_after: int | None = _fetch_stock(TEST_PART, TEST_WAREHOUSE)
    report: str = str(final.get("final_response") or "")

    approved_checks: dict[str, bool] = {
        "state_approved": final.get("approval_status") == "APPROVED",
        "state_approver": final.get("approver") == "张主管",
        "db_approved": bool(row and row.get("approval_status") == "APPROVED"),
        "db_dispatched": bool(row and row.get("dispatch_status") == "DISPATCHED"),
        "db_approver": bool(row and row.get("approver") == "张主管"),
        "stock_deducted": stock_before is not None
        and stock_after is not None
        and stock_after == stock_before - 1,
        "report_has_report": "# 机器人售后服务处理报告" in report,
        "report_has_order": str(order_no) in report,
        "report_has_warehouse": TEST_WAREHOUSE in report,
        "report_has_dispatch": "DISPATCHED" in report,
    }
    failed_after: list[str] = [k for k, v in approved_checks.items() if not v]
    if failed_after:
        return False, f"审批通过阶段校验失败：{failed_after} | row={row}"

    elapsed: float = (time.perf_counter() - started) * 1000
    return True, (
        f"order_no={order_no} 仓库={TEST_WAREHOUSE} 库存 {stock_before}→{stock_after} "
        f"审批=APPROVED/DISPATCHED 耗时={elapsed:.1f}ms（草稿未锁库存={stock_locked_ok}）"
    )


# ---------------------------------------------------------------------------
# 用例 3：审批驳回
# ---------------------------------------------------------------------------
def test_hitl_reject() -> tuple[bool, str]:
    """验证驳回分支：工单 CLOSED、库存不变、报告体现驳回。"""
    trace_id: str = new_trace_id("TR-GRAPH-NO")
    logger.info("【用例 3】E-201 审批驳回 | trace_id={}", trace_id)

    paused: dict[str, Any] = _start(trace_id, RAW_E201)
    order_no: str | None = paused.get("order_no")
    if not order_no:
        return False, "未生成工单，无法验证驳回分支"
    _track_order(order_no)
    _track_stock(TEST_PART, TEST_WAREHOUSE)
    stock_before: int | None = _fetch_stock(TEST_PART, TEST_WAREHOUSE)

    final: dict[str, Any] = _resume(trace_id, "REJECTED", "李经理")
    row: dict[str, Any] | None = _fetch_order(str(order_no))
    stock_after: int | None = _fetch_stock(TEST_PART, TEST_WAREHOUSE)
    report: str = str(final.get("final_response") or "")

    checks: dict[str, bool] = {
        "state_rejected": final.get("approval_status") == "REJECTED",
        "state_approver": final.get("approver") == "李经理",
        "db_rejected": bool(row and row.get("approval_status") == "REJECTED"),
        "db_closed": bool(row and row.get("dispatch_status") == "CLOSED"),
        "db_approver": bool(row and row.get("approver") == "李经理"),
        "stock_unchanged": stock_before is not None
        and stock_after is not None
        and stock_after == stock_before,
        "report_mentions_reject": "驳回" in report and "CLOSED" in report,
    }
    failed: list[str] = [k for k, v in checks.items() if not v]
    if failed:
        return False, f"驳回分支校验失败：{failed} | row={row}"
    return True, f"order_no={order_no} 审批=REJECTED/CLOSED 库存保持 {stock_after} 件"


# ---------------------------------------------------------------------------
# 用例 4：E-500 急停复位（无需换件）
# ---------------------------------------------------------------------------
def test_no_part_path() -> tuple[bool, str]:
    """验证「无需换件」分支：不进库存核验，直接出排障指南。"""
    trace_id: str = new_trace_id("TR-GRAPH-RST")
    logger.info("【用例 4】E-500 无需换件 | trace_id={}", trace_id)

    result: dict[str, Any] = _start(trace_id, RAW_E500)
    report: str = str(result.get("final_response") or "")

    checks: dict[str, bool] = {
        "no_interrupt": "__interrupt__" not in result,
        "code": result.get("error_code") == "E-500",
        "no_required_part": not result.get("required_part"),
        "no_order_no": not result.get("order_no"),
        "no_db_order": _count_orders_by_trace(trace_id) == 0,
        "report_ready": "# 机器人售后服务处理报告" in report,
        "report_no_part": "无需更换备件" in report,
        "report_sop": "复位" in report,
    }
    failed: list[str] = [k for k, v in checks.items() if not v]
    if failed:
        return False, f"无需换件分支校验失败：{failed}"
    return True, (
        f"error_code=E-500 required_part=None order_no=None 直接出报告（{len(report)} 字）"
    )


# ---------------------------------------------------------------------------
# 用例 5：Webhook 告警通道
# ---------------------------------------------------------------------------
def test_webhook_channel() -> tuple[bool, str]:
    """验证边缘设备 JSON 报文走本地解包，不经 LLM 也不会解析错字段。"""
    trace_id: str = new_trace_id("TR-GRAPH-WH")
    logger.info("【用例 5】Webhook 通道 | trace_id={}", trace_id)

    paused: dict[str, Any] = _start(trace_id, WEBHOOK_E201, channel="webhook_alert")
    payload: dict[str, Any] | None = _interrupt_payload(paused)
    order_no: str | None = paused.get("order_no")
    _track_order(order_no)
    _track_stock(TEST_PART, TEST_WAREHOUSE)

    checks: dict[str, bool] = {
        "sn": paused.get("device_sn") == TEST_DEVICE_SN,
        "code": paused.get("error_code") == "E-201",
        "symptom_from_message": paused.get("symptom") == "夹爪气缸持续漏气，面板报错 E-201",
        "channel_kept": paused.get("channel") == "webhook_alert",
        "paused": payload is not None and bool(order_no),
        "part": paused.get("required_part") == TEST_PART,
    }
    if not all(checks.values()):
        failed: list[str] = [k for k, v in checks.items() if not v]
        return False, f"Webhook 分支校验失败：{failed}"

    # 收尾：驳回，避免留下待审批工单（--keep 时照样走完整闭环，不影响数据正确性）
    _resume(trace_id, "REJECTED", "李经理")
    row: dict[str, Any] | None = _fetch_order(str(order_no))
    ok: bool = bool(row and row.get("dispatch_status") == "CLOSED")
    return ok, f"order_no={order_no} 报文解包正确、已按驳回收尾"


# ---------------------------------------------------------------------------
# 用例 6：跨进程挂起 / 恢复
# ---------------------------------------------------------------------------
def _spawn_worker(args: list[str]) -> tuple[dict[str, Any], subprocess.CompletedProcess[str]]:
    """启动一个 worker 子进程并解析它输出的结果行。"""
    command: list[str] = [sys.executable, "-X", "utf8", str(SCRIPT_PATH), "--worker", *args]
    env: dict[str, str] = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    proc: subprocess.CompletedProcess[str] = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=300,
        check=False,
    )
    markers: list[str] = [
        line[len(_MARKER) :] for line in proc.stdout.splitlines() if line.startswith(_MARKER)
    ]
    if not markers:
        raise RuntimeError(
            f"worker {args[0]} 未返回结果（exit={proc.returncode}）| "
            f"stdout 尾部={proc.stdout[-500:]!r} | stderr 尾部={proc.stderr[-800:]!r}"
        )
    return json.loads(markers[-1]), proc


def test_cross_process() -> tuple[bool, str]:
    """两个独立进程接力：进程 A 挂起，进程 B 从 checkpoints.sqlite 恢复并派工。"""
    trace_id: str = new_trace_id("TR-GRAPH-XP")
    logger.info("【用例 6】跨进程挂起 / 恢复 | trace_id={}", trace_id)

    # Qdrant 本地文件模式是进程独占锁，必须先把父进程的客户端关掉
    close_qdrant_client()

    started: dict[str, Any]
    proc_start: subprocess.CompletedProcess[str]
    started, proc_start = _spawn_worker(
        ["start", "--thread", trace_id, "--raw", RAW_E201]
    )
    order_no: str | None = started.get("order_no")
    if not started.get("paused") or not order_no:
        return False, f"子进程未在审批处挂起：{started} | stderr={proc_start.stderr[-400:]}"
    _track_order(order_no)
    _track_stock(TEST_PART, TEST_WAREHOUSE)
    stock_before: int | None = _fetch_stock(TEST_PART, TEST_WAREHOUSE)

    finished: dict[str, Any]
    proc_resume: subprocess.CompletedProcess[str]
    finished, proc_resume = _spawn_worker(
        [
            "resume",
            "--thread",
            trace_id,
            "--decision",
            "APPROVED",
            "--approver",
            "王工",
        ]
    )
    row: dict[str, Any] | None = _fetch_order(str(order_no))
    stock_after: int | None = _fetch_stock(TEST_PART, TEST_WAREHOUSE)

    checks: dict[str, bool] = {
        "distinct_start_pid": started.get("process_id") != os.getpid(),
        "distinct_resume_pid": finished.get("process_id")
        not in (os.getpid(), started.get("process_id")),
        "state_approved": finished.get("approval_status") == "APPROVED",
        "db_dispatched": bool(row and row.get("dispatch_status") == "DISPATCHED"),
        "db_approver": bool(row and row.get("approver") == "王工"),
        "stock_deducted": stock_before is not None
        and stock_after is not None
        and stock_after == stock_before - 1,
    }
    failed: list[str] = [k for k, v in checks.items() if not v]
    if failed:
        return False, (
            f"跨进程校验失败：{failed} | start={started} resume={finished} | "
            f"stderr={proc_resume.stderr[-400:]}"
        )
    return True, (
        f"PID {started.get('process_id')} 挂起 → PID {finished.get('process_id')} 恢复 | "
        f"order_no={order_no} 库存 {stock_before}→{stock_after}"
    )


# ---------------------------------------------------------------------------
# worker 模式（供用例 6 的子进程调用）
# ---------------------------------------------------------------------------
def _run_worker(args: argparse.Namespace) -> int:
    """子进程入口：start 只跑到挂起，resume 从检查点接着跑完。"""
    trace_id: str = str(args.thread).strip()
    if not trace_id:
        logger.error("worker 需要 --thread 参数")
        return 2

    app: Any = build_graph()
    with trace_context(trace_id):
        if args.worker == "start":
            result: dict[str, Any] = app.invoke(
                {"trace_id": trace_id, "channel": "human_chat", "raw_input": args.raw},
                make_config(trace_id),
            )
            snapshot = app.get_state(make_config(trace_id))
            payload: dict[str, Any] = {
                "process_id": os.getpid(),
                "paused": "__interrupt__" in result,
                "order_no": result.get("order_no"),
                "next_nodes": list(snapshot.next),
                "approval_status": result.get("approval_status"),
            }
        else:
            result = app.invoke(
                Command(resume={"status": args.decision, "approver": args.approver}),
                make_config(trace_id),
            )
            payload = {
                "process_id": os.getpid(),
                "order_no": result.get("order_no"),
                "approval_status": result.get("approval_status"),
                "approver": result.get("approver"),
                "report_head": str(result.get("final_response") or "")[:60],
            }

    # 结果行必须独占 stdout，日志走 stderr，父进程只认这一行
    print(_MARKER + json.dumps(payload, ensure_ascii=False))
    return 0


# ---------------------------------------------------------------------------
# 清理与汇总
# ---------------------------------------------------------------------------
def _cleanup(keep: bool) -> None:
    """删除本轮生成的工单并还原库存。"""
    if keep:
        if _CREATED_ORDERS or _STOCK_BASELINE:
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
        logger.info(
            "已清理测试工单 {}，库存已还原 {}", _CREATED_ORDERS, list(_STOCK_BASELINE.items())
        )
    except Exception as exc:  # noqa: BLE001 - 清理失败不应掩盖测试结论
        logger.error("清理测试数据失败（请手工核对）：{}", exc)


def _close_connections() -> None:
    """关闭脚本自建的全部 SQLite 检查点连接。"""
    for conn in _OPEN_CONNS:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            pass
    _OPEN_CONNS.clear()


def _report(results: list[tuple[str, bool, str]]) -> int:
    """打印汇总表并返回进程退出码。"""
    logger.info("=" * 78)
    logger.info("测试汇总")
    logger.info("-" * 78)
    for name, ok, detail in results:
        if ok:
            logger.success("✓ {:<40} {}", name, detail)
        else:
            logger.error("✗ {:<40} {}", name, detail)
    passed: int = sum(1 for _, ok, _ in results if ok)
    logger.info("-" * 78)
    logger.info("通过 {}/{}", passed, len(results))
    logger.info("=" * 78)
    return 0 if passed == len(results) else 1


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="LangGraph 排障与备件调度状态机端到端测试")
    parser.add_argument("--keep", action="store_true", help="保留生成的工单与库存变动")
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="只跑指定用例编号，逗号分隔（例如 2,6）",
    )
    parser.add_argument(
        "--worker",
        choices=("start", "resume"),
        default=None,
        help="子进程模式：start 跑到审批挂起 / resume 恢复执行（供跨进程用例调用）",
    )
    parser.add_argument("--thread", type=str, default="", help="worker 模式的 thread_id（= trace_id）")
    parser.add_argument("--raw", type=str, default=RAW_E201, help="worker start 的报修文本")
    parser.add_argument(
        "--decision", type=str, default="APPROVED", choices=("APPROVED", "REJECTED")
    )
    parser.add_argument("--approver", type=str, default="张主管")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """执行全部测试用例。

    Returns:
        int: 0 表示全部通过，1 表示存在失败，2 表示环境不满足。
    """
    args: argparse.Namespace = parse_args(argv)

    # worker 模式只跑一条图，不进入完整测试
    if args.worker:
        return _run_worker(args)

    with trace_context():
        logger.info("=" * 78)
        logger.info("LangGraph 排障与备件调度状态机 · 端到端测试")
        logger.info("项目根目录  : {}", PROJECT_ROOT)
        logger.info("MySQL       : {}", settings.mysql_url_masked)
        logger.info("检查点文件  : {}", settings.checkpoint_db_abspath)
        logger.info("Qdrant 目录 : {}", settings.qdrant_local_abspath)
        logger.info("=" * 78)

        if not verify_mysql_connection():
            logger.error("MySQL 不可用，测试终止")
            return 2

        # 知识库缺失时自动引导灌库（复用第二阶段脚本，避免重复实现）
        from test_tools import ensure_knowledge_base  # noqa: PLC0415

        ensure_knowledge_base()

        cases: tuple[tuple[str, Callable[[], tuple[bool, str]]], ...] = (
            ("1. 图结构与条件路由", test_graph_shape),
            ("2. E-201 全链路（挂起→通过→派工）", test_hitl_approve),
            ("3. E-201 审批驳回", test_hitl_reject),
            ("4. E-500 无需换件直接出报告", test_no_part_path),
            ("5. Webhook 告警通道", test_webhook_channel),
            ("6. 跨进程挂起 / 恢复", test_cross_process),
        )
        selected: set[str] = {
            item.strip() for item in args.only.split(",") if item.strip()
        }

        results: list[tuple[str, bool, str]] = []
        try:
            for name, case in cases:
                if selected and name.split(".")[0] not in selected:
                    continue
                try:
                    ok, detail = case()
                except Exception as exc:  # noqa: BLE001 - 测试脚本需要汇总所有异常
                    logger.exception("用例 {} 抛出异常", name)
                    ok, detail = False, f"异常：{type(exc).__name__}: {exc}"
                results.append((name, ok, detail))
        finally:
            _cleanup(keep=bool(args.keep))
            _close_connections()
            close_qdrant_client()

        if not results:
            logger.error("没有匹配的用例（--only={}）", args.only)
            return 2
        return _report(results)


if __name__ == "__main__":
    raise SystemExit(main())
