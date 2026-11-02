"""备件库存 / 设备档案 / 工单草稿工具集（安全参数化 Text2SQL）。

安全设计（对应需求 FR-3.2 / FR-3.3）
------------------------------------
* **零字符串拼接**：所有 SQL 均为固定模板 + SQLAlchemy ``text()`` + ``.bindparams()``，
  用户输入只作为**绑定参数**传入，永远不参与 SQL 文本拼装，从机制上杜绝注入；
* **只读优先**：查询类工具只发 ``select``；唯一写操作 ``create_work_order_draft``
  只插入工单草稿行（``approval_status='PENDING'``），不触碰库存表，库存核销留到审批通过后；
* **白名单字段**：返回列在 SQL 中显式列举，避免 ``select *`` 把敏感列带出去。

可观测性
--------
每次调用都会用 Loguru 记录：入参 → 实际执行的 SQL（单行化）与绑定参数 → 耗时 → 返回条数，
日志自动携带当前上下文的 ``trace_id``（见 ``src/core/logger.py``）。
"""

from __future__ import annotations

import time
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from langchain_core.tools import tool
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from src.core.db import session_scope
from src.core.logger import get_trace_id, logger

__all__ = [
    "query_device_info",
    "query_spare_part_stock",
    "create_work_order_draft",
    "SQL_TOOLS",
]

# ---------------------------------------------------------------------------
# 固定 SQL 模板（全部使用绑定参数，严禁拼接）
# ---------------------------------------------------------------------------
_DEVICE_SQL: str = (
    "select device_sn, model_name, category, location, workshop, status "
    "from robot_devices where device_sn = :sn"
)

_STOCK_SQL: str = (
    "select part_code, part_name, spec_model, stock_quantity, warehouse_name, "
    "warehouse_location, bin_location, unit_price "
    "from spare_parts where part_name like :part_name and stock_quantity > 0"
)

_INSERT_ORDER_SQL: str = (
    "insert into work_orders (order_no, device_sn, error_code, symptom, suggested_part, "
    "allocated_warehouse, approval_status, dispatch_status, trace_id, created_at, updated_at) "
    "values (:order_no, :device_sn, :error_code, :symptom, :suggested_part, "
    ":allocated_warehouse, :approval_status, :dispatch_status, :trace_id, now(), now())"
)

_SELECT_ORDER_SQL: str = (
    "select id, order_no, device_sn, error_code, suggested_part, allocated_warehouse, "
    "approval_status, dispatch_status, trace_id, created_at "
    "from work_orders where order_no = :order_no"
)

_LATEST_ORDER_SQL: str = (
    "select order_no from work_orders where order_no like :prefix order by order_no desc limit 1"
)


# ---------------------------------------------------------------------------
# 通用辅助
# ---------------------------------------------------------------------------
def _jsonable(value: Any) -> Any:
    """把数据库原生类型转换为可 JSON 序列化的类型。

    LLM 工具链最终要序列化成文本/JSON，``Decimal`` 与 ``datetime`` 无法直接序列化，
    因此在工具边界统一转换。

    Args:
        value: 任意数据库字段值。

    Returns:
        Any: 可序列化的值（Decimal→float，datetime→字符串）。
    """
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.strftime("%Y-%m-%d %H:%M:%S") if isinstance(value, datetime) else value.isoformat()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _one_line(sql: str) -> str:
    """把多行 SQL 压成单行，便于日志阅读与检索。"""
    return " ".join(sql.split())


def _fetch_all(sql: str, params: dict[str, Any], label: str) -> list[dict[str, Any]]:
    """执行只读查询并返回字典列表。

    Args:
        sql: 含命名占位符的 SQL 模板。
        params: 绑定参数字典。
        label: 日志标签，用于区分是哪个工具在查。

    Returns:
        list[dict[str, Any]]: 查询结果（已做 JSON 安全转换）。

    Raises:
        Exception: 数据库异常向上抛出（由工具入口统一记录错误日志）。
    """
    started: float = time.perf_counter()
    logger.debug("[sql][{}] SQL={} | params={}", label, _one_line(sql), params)

    with session_scope() as db:
        rows: list[dict[str, Any]] = [
            dict(row) for row in db.execute(text(sql).bindparams(**params)).mappings().all()
        ]

    elapsed_ms: float = (time.perf_counter() - started) * 1000
    logger.info(
        "[sql][{}] 执行完成 | 返回 {} 行 | 耗时 {:.1f} ms | params={}",
        label,
        len(rows),
        elapsed_ms,
        params,
    )
    return [_jsonable(row) for row in rows]


# ---------------------------------------------------------------------------
# 工具 2.1：设备档案查询
# ---------------------------------------------------------------------------
@tool
def query_device_info(device_sn: str) -> dict[str, Any]:
    """按设备序列号查询机器人档案（型号、类别、所在城市与车间、运行状态）。

    排障流水线的第一步：``location`` 决定后续就近仓库调拨的优先级。

    Args:
        device_sn: 设备序列号，例如 ``AGV-SZ-003``。

    Returns:
        dict: 命中时包含 device_sn / model_name / category / location / workshop / status；
        未命中时返回 ``{"found": False, ...}``。
    """
    started: float = time.perf_counter()
    sn: str = (device_sn or "").strip()
    logger.info("[tool] query_device_info 入参 device_sn={!r} trace_id={}", sn, get_trace_id())

    if not sn:
        logger.warning("[tool] query_device_info 收到空 device_sn，直接返回未找到")
        return {"found": False, "device_sn": "", "message": "device_sn 不能为空"}

    try:
        rows: list[dict[str, Any]] = _fetch_all(_DEVICE_SQL, {"sn": sn}, "query_device_info")
    except Exception as exc:  # noqa: BLE001 - 工具对 LLM 必须返回可读结果而非裸异常
        logger.error("[tool] query_device_info 查询失败 device_sn={} err={}", sn, exc)
        return {"found": False, "device_sn": sn, "message": f"数据库查询失败：{exc}"}

    if not rows:
        logger.warning("[tool] query_device_info 未找到设备 device_sn={}", sn)
        return {
            "found": False,
            "device_sn": sn,
            "message": f"设备台账中不存在序列号 {sn}，请核对铭牌 SN",
        }

    result: dict[str, Any] = {"found": True, **rows[0]}
    logger.info(
        "[tool] query_device_info 命中 device_sn={} location={} 耗时={:.1f}ms",
        result.get("device_sn"),
        result.get("location"),
        (time.perf_counter() - started) * 1000,
    )
    return result


# ---------------------------------------------------------------------------
# 工具 2.2：备件库存查询（多仓就近排序）
# ---------------------------------------------------------------------------
def _is_preferred(row: dict[str, Any], preferred_location: str | None) -> bool:
    """判断某行库存是否属于「就近」仓库。

    Args:
        row: ``spare_parts`` 的一行。
        preferred_location: 设备所在城市，例如 ``苏州``。

    Returns:
        bool: 仓库名称或仓库城市包含该地点时返回 ``True``。
    """
    if not preferred_location:
        return False
    target: str = preferred_location.strip()
    return target in str(row.get("warehouse_location") or "") or target in str(
        row.get("warehouse_name") or ""
    )


@tool
def query_spare_part_stock(part_name: str, preferred_location: str | None = None) -> list[dict[str, Any]]:
    """查询某备件在各仓库的实时可用库存，并按「就近调拨」原则排序。

    Args:
        part_name: 备件标准名称（支持模糊匹配），例如 ``型号A电磁阀``。
        preferred_location: 设备所在城市（如 ``苏州``），用于把同城仓库排到最前面。

    Returns:
        list[dict]: 每行含 part_code / part_name / spec_model / stock_quantity /
        warehouse_name / warehouse_location / bin_location / unit_price，
        以及 ``is_preferred_location`` 就近标记；无可用库存时返回空列表。
    """
    started: float = time.perf_counter()
    name: str = (part_name or "").strip()
    location: str | None = (preferred_location or "").strip() or None
    logger.info(
        "[tool] query_spare_part_stock 入参 part_name={!r} preferred_location={!r} trace_id={}",
        name,
        location,
        get_trace_id(),
    )

    if not name:
        logger.warning("[tool] query_spare_part_stock 收到空备件名，返回空列表")
        return []

    try:
        rows: list[dict[str, Any]] = _fetch_all(
            _STOCK_SQL, {"part_name": f"%{name}%"}, "query_spare_part_stock"
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("[tool] query_spare_part_stock 查询失败 part_name={} err={}", name, exc)
        return []

    # 就近优先：同城仓库置顶；同城/异城内部均按库存量降序，避免先推荐零星散库
    for row in rows:
        row["is_preferred_location"] = _is_preferred(row, location)
    rows.sort(key=lambda r: (not r["is_preferred_location"], -int(r.get("stock_quantity") or 0)))

    total_qty: int = sum(int(r.get("stock_quantity") or 0) for r in rows)
    logger.info(
        "[tool] query_spare_part_stock 完成 part_name={} 仓库数={} 总库存={} 首选仓={} 耗时={:.1f}ms",
        name,
        len(rows),
        total_qty,
        rows[0].get("warehouse_name") if rows else "无",
        (time.perf_counter() - started) * 1000,
    )
    if not rows:
        logger.warning("[tool] query_spare_part_stock 未找到可用库存 part_name={} location={}", name, location)
    return rows


# ---------------------------------------------------------------------------
# 工具 2.3：工单草稿落库
# ---------------------------------------------------------------------------
def _next_order_no(db: Any, prefix: str) -> str:
    """生成下一个工单号：``WO-YYYYMM-<4 位流水>``。

    Args:
        db: 当前数据库会话。
        prefix: 前缀，形如 ``WO-202610-``。

    Returns:
        str: 递增后的工单号；当月无历史工单时从 ``0001`` 开始。
    """
    row = db.execute(text(_LATEST_ORDER_SQL).bindparams(prefix=f"{prefix}%")).first()
    sequence: int = 1
    if row and row[0]:
        tail: str = str(row[0]).rsplit("-", 1)[-1]
        if tail.isdigit():
            sequence = int(tail) + 1
    return f"{prefix}{sequence:04d}"


@tool
def create_work_order_draft(
    device_sn: str,
    error_code: str,
    symptom: str,
    suggested_part: str,
    allocated_warehouse: str,
    trace_id: str,
) -> dict[str, Any]:
    """把排障结论与备件建议落库为「待审批」工单草稿（HITL 前置动作）。

    只写草稿、不锁库存：``approval_status='PENDING'``、``dispatch_status='INIT'``，
    等主管在审批终端确认后才进入派工与库存核销。

    Args:
        device_sn: 设备序列号。
        error_code: 故障码，例如 ``E-201``。
        symptom: 故障现象描述（可为空字符串）。
        suggested_part: 建议更换的备件标准名称。
        allocated_warehouse: 建议调拨的仓库名称（如 ``苏州中心仓``）。
        trace_id: 全链路追踪号，写入工单便于回溯。

    Returns:
        dict: 成功时含 ``success=True`` 与 order_no / order_id / approval_status /
        dispatch_status / created_at；失败时含 ``success=False`` 与 message。
    """
    started: float = time.perf_counter()
    sn: str = (device_sn or "").strip()
    code: str = (error_code or "").strip().upper()
    current_trace: str = (trace_id or "").strip() or get_trace_id()
    logger.info(
        "[tool] create_work_order_draft 入参 device_sn={!r} error_code={!r} part={!r} "
        "warehouse={!r} trace_id={}",
        sn,
        code,
        suggested_part,
        allocated_warehouse,
        current_trace,
    )

    if not sn:
        return {"success": False, "message": "device_sn 不能为空，工单未创建"}

    prefix: str = f"WO-{datetime.now():%Y%m}-"
    params_base: dict[str, Any] = {
        "device_sn": sn,
        "error_code": code or None,
        "symptom": (symptom or "").strip() or None,
        "suggested_part": (suggested_part or "").strip() or None,
        "allocated_warehouse": (allocated_warehouse or "").strip() or None,
        "approval_status": "PENDING",
        "dispatch_status": "INIT",
        "trace_id": current_trace,
    }

    # 工单号唯一约束冲突（并发插入同一流水号）时重试，最多 5 次
    for attempt in range(1, 6):
        try:
            with session_scope() as db:
                order_no: str = _next_order_no(db, prefix)
                db.execute(
                    text(_INSERT_ORDER_SQL).bindparams(order_no=order_no, **params_base)
                )
                saved = (
                    db.execute(text(_SELECT_ORDER_SQL).bindparams(order_no=order_no))
                    .mappings()
                    .first()
                )
                result: dict[str, Any] = _jsonable(dict(saved)) if saved else {}
        except IntegrityError as exc:
            logger.warning(
                "[sql][create_work_order_draft] 工单号冲突，第 {} 次重试：{}", attempt, exc
            )
            continue
        except Exception as exc:  # noqa: BLE001
            logger.error("[tool] create_work_order_draft 落库失败 device_sn={} err={}", sn, exc)
            return {"success": False, "device_sn": sn, "message": f"工单落库失败：{exc}"}

        elapsed_ms: float = (time.perf_counter() - started) * 1000
        logger.success(
            "[tool] create_work_order_draft 成功 order_no={} id={} trace_id={} 耗时={:.1f}ms",
            result.get("order_no"),
            result.get("id"),
            result.get("trace_id"),
            elapsed_ms,
        )
        return {
            "success": True,
            "order_id": result.get("id"),
            "order_no": result.get("order_no"),
            "device_sn": result.get("device_sn", sn),
            "error_code": result.get("error_code", code),
            "suggested_part": result.get("suggested_part"),
            "allocated_warehouse": result.get("allocated_warehouse"),
            "approval_status": result.get("approval_status", "PENDING"),
            "dispatch_status": result.get("dispatch_status", "INIT"),
            "trace_id": result.get("trace_id", current_trace),
            "created_at": result.get("created_at"),
            "message": "工单草稿已写入，等待主管审批（PENDING）",
        }

    logger.error("[tool] create_work_order_draft 连续 5 次工单号冲突，放弃创建")
    return {"success": False, "device_sn": sn, "message": "工单号生成冲突，请稍后重试"}


#: 便于 LangGraph 节点一次性绑定全部 SQL 工具
SQL_TOOLS: list[Any] = [query_device_info, query_spare_part_stock, create_work_order_draft]
