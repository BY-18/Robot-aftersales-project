"""接入层核心路由：双通道诊断、HITL 审批恢复、控制台查询与健康检查。

链路约定
--------
* 每个请求都会拿到一个 ``trace_id``（请求体 > ``X-Trace-Id`` 请求头 > 服务端生成），
  它同时是 **日志上下文** 与 LangGraph 的 ``thread_id``，因此「报修 → 挂起 → 审批 →
  派工」的日志可以用一个 ID 串起来；
* 诊断接口是**同步阻塞**实现（DeepSeek / PyMySQL / 本地 Qdrant 都是同步客户端），
  因此用 ``def`` 而非 ``async def``，由 FastAPI 丢到线程池执行，不阻塞事件循环；
* 审批接口走 LangGraph 官方两段式：``update_state`` 落审批结论 → ``invoke(None)``
  从断点续跑，最终由 ``dispatch_order_node`` 完成库存核销与结案。
"""

from __future__ import annotations

import json
import time
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, Request, status
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from src.agents.graph import get_checkpointer, get_graph, make_config
from src.api.schemas import (
    ApprovalDecision,
    ChatDiagnoseRequest,
    ComponentHealth,
    DiagnoseResponse,
    ErrorResponse,
    HealthResponse,
    SparePartItem,
    SparePartListResponse,
    StockItem,
    WebhookAlertRequest,
    WorkOrderApproveRequest,
    WorkOrderApproveResponse,
    WorkOrderItem,
    WorkOrderListResponse,
)
from src.core.config import settings
from src.core.db import session_scope, verify_mysql_connection
from src.core.logger import get_trace_id, logger, new_trace_id, trace_context
from src.tools.rag_tool import MANUAL_COLLECTION_NAME, knowledge_base_health

__all__ = ["router", "health_router", "resolve_trace_id"]

#: 业务路由前缀（版本化，便于后续 v2 平滑上线）
API_PREFIX: str = "/api/v1"

router: APIRouter = APIRouter(prefix=API_PREFIX)
#: 健康检查挂在根路径（``GET /health``），不进 /api/v1 版本前缀
health_router: APIRouter = APIRouter()


# ---------------------------------------------------------------------------
# 通用辅助
# ---------------------------------------------------------------------------
def resolve_trace_id(request: Request, explicit: str | None = None) -> str:
    """确定本次请求的 ``trace_id``。

    优先级：请求体显式值 > 中间件从 ``X-Trace-Id`` 头解析的值 > 现场生成。

    Args:
        request: 当前请求对象（中间件会写入 ``request.state.trace_id``）。
        explicit: 请求体里传入的 ``trace_id``（可为 ``None``）。

    Returns:
        str: 最终生效的追踪号，同时作为 LangGraph 的 ``thread_id``。
    """
    candidate: str = (explicit or "").strip()
    if not candidate:
        candidate = str(getattr(request.state, "trace_id", "") or "").strip()
    if not candidate:
        current: str = get_trace_id()
        candidate = current if current and current != "-" else new_trace_id()

    # 回写到 request.state：响应头的 X-Trace-Id 要反映真正生效的值
    request.state.trace_id = candidate
    return candidate


def _jsonable(value: Any) -> Any:
    """把数据库原生类型转换为可 JSON 化的值（Decimal / datetime / date）。"""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, date):
        return value.isoformat()
    return value


def _format_dt(value: Any) -> Optional[str]:
    """把时间列格式化为 ``YYYY-MM-DD HH:MM:SS`` 字符串。"""
    formatted: Any = _jsonable(value)
    return str(formatted) if formatted is not None else None


def _interrupt_payload(result: dict[str, Any]) -> Optional[dict[str, Any]]:
    """从图执行结果中取出 ``interrupt()`` 的审批载荷。"""
    raw: Any = result.get("__interrupt__")
    if not raw:
        return None
    value: Any = getattr(raw[0], "value", None)
    return value if isinstance(value, dict) else None


def _run_workflow(
    trace_id: str,
    channel: str,
    raw_input: str,
) -> tuple[dict[str, Any], Optional[dict[str, Any]]]:
    """执行一次完整的状态机（跑到挂起或结束）。

    Args:
        trace_id: 本次链路号，同时作 ``thread_id``。
        channel: ``human_chat`` 或 ``webhook_alert``。
        raw_input: 自然语言报修文本 / Webhook JSON 报文。

    Returns:
        tuple[dict, dict | None]: ``(最终状态, 审批载荷)``；未挂起时载荷为 ``None``。

    Raises:
        HTTPException: 图执行异常时返回 500，避免把内部堆栈抛给调用方。
    """
    graph = get_graph()
    config = make_config(trace_id)
    initial_state: dict[str, Any] = {
        "trace_id": trace_id,
        "channel": channel,
        "raw_input": raw_input,
    }
    try:
        with trace_context(trace_id):
            result: dict[str, Any] = graph.invoke(initial_state, config)
    except Exception as exc:  # noqa: BLE001 - 对外统一为 500，细节进日志
        logger.exception("[api] 状态机执行失败 trace_id={} channel={}", trace_id, channel)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"状态机执行失败：{type(exc).__name__}: {exc}",
        ) from exc
    return result, _interrupt_payload(result)


def _to_diagnose_response(
    result: dict[str, Any],
    trace_id: str,
    channel: str,
    approval_payload: Optional[dict[str, Any]],
    elapsed_ms: float,
) -> DiagnoseResponse:
    """把 LangGraph 状态映射为对外响应。"""
    paused: bool = approval_payload is not None
    return DiagnoseResponse(
        trace_id=trace_id,
        channel=channel,  # type: ignore[arg-type]  # 仅 human_chat / webhook_alert
        status="WAITING_APPROVAL" if paused else "COMPLETED",
        device_sn=result.get("device_sn"),
        device_location=result.get("device_location"),
        error_code=result.get("error_code"),
        symptom=result.get("symptom"),
        sop_guide=result.get("sop_guide"),
        required_part=result.get("required_part"),
        stock_info=[StockItem(**row) for row in (result.get("stock_info") or [])],
        selected_warehouse=result.get("selected_warehouse"),
        order_no=result.get("order_no"),
        approval_status=result.get("approval_status"),
        approval_payload=approval_payload,
        final_response=result.get("final_response"),
        elapsed_ms=round(elapsed_ms, 1),
    )


def _fetch_order_row(order_no: str) -> Optional[dict[str, Any]]:
    """读取工单当前状态（审批 / 派工 / 备件 / 仓库）。"""
    with session_scope() as db:
        row = (
            db.execute(
                text(
                    "select order_no, approval_status, approver, dispatch_status, "
                    "suggested_part, allocated_warehouse from work_orders "
                    "where order_no = :order_no"
                ).bindparams(order_no=order_no)
            )
            .mappings()
            .first()
        )
    return dict(row) if row else None


def _fetch_stock(part_name: str | None, warehouse: str | None) -> Optional[int]:
    """读取某备件在某仓库的当前库存。"""
    if not part_name or not warehouse:
        return None
    with session_scope() as db:
        value = db.execute(
            text(
                "select stock_quantity from spare_parts "
                "where part_name = :part_name and warehouse_name = :warehouse_name"
            ).bindparams(part_name=part_name, warehouse_name=warehouse)
        ).scalar()
    return int(value) if value is not None else None


# ---------------------------------------------------------------------------
# 通道 A：人工自然语言报修
# ---------------------------------------------------------------------------
@router.post(
    "/diagnose/chat",
    response_model=DiagnoseResponse,
    tags=["诊断接入"],
    summary="通道 A：人工报修诊断",
    responses={
        500: {"model": ErrorResponse, "description": "状态机执行失败"},
    },
)
def diagnose_chat(payload: ChatDiagnoseRequest, request: Request) -> DiagnoseResponse:
    """接收工程师的自然语言报修，触发排障 → 备件 → 审批状态机。

    若需要更换备件，工作流会停在人工审批节点并返回 ``status="WAITING_APPROVAL"``
    （含工单号与推荐仓库）；无需换件则直接返回 ``status="COMPLETED"`` 与完整报告。

    Args:
        payload: 报修请求体（``query`` 必填，``trace_id`` 可选）。
        request: FastAPI 请求对象（用于读取中间件写入的 trace_id）。

    Returns:
        DiagnoseResponse: 诊断结果；挂起时附带审批载荷。
    """
    started: float = time.perf_counter()
    trace_id: str = resolve_trace_id(request, payload.trace_id)
    logger.info(
        "[api] POST {}diagnose/chat | trace_id={} query={!r}",
        API_PREFIX,
        trace_id,
        payload.query[:120],
    )

    result, approval_payload = _run_workflow(trace_id, "human_chat", payload.query)
    response: DiagnoseResponse = _to_diagnose_response(
        result, trace_id, "human_chat", approval_payload, (time.perf_counter() - started) * 1000
    )
    logger.success(
        "[api] diagnose/chat 完成 | trace_id={} status={} order_no={} 耗时={}ms",
        trace_id,
        response.status,
        response.order_no,
        response.elapsed_ms,
    )
    return response


# ---------------------------------------------------------------------------
# 通道 B：设备 IoT 告警
# ---------------------------------------------------------------------------
@router.post(
    "/alert/webhook",
    response_model=DiagnoseResponse,
    tags=["诊断接入"],
    summary="通道 B：设备 Webhook 告警",
    responses={
        500: {"model": ErrorResponse, "description": "状态机执行失败"},
    },
)
def alert_webhook(payload: WebhookAlertRequest, request: Request) -> DiagnoseResponse:
    """接收边缘设备自动上报的告警，走与通道 A 完全相同的状态机。

    请求体先被序列化为 JSON 报文交给 ``parse_fault_node``：该节点识别出
    ``channel="webhook_alert"`` 且报文是结构化 JSON 时**不调用 LLM**，直接取字段，
    保证 Webhook 入口的解析耗时稳定在毫秒级。

    Args:
        payload: 设备告警报文。
        request: FastAPI 请求对象。

    Returns:
        DiagnoseResponse: 诊断结果；需要换件时同样挂起等待审批。
    """
    started: float = time.perf_counter()
    trace_id: str = resolve_trace_id(request)
    alert: dict[str, Any] = {
        "device_sn": payload.device_sn,
        "error_code": payload.error_code,
        "symptom": payload.symptom,
    }
    if payload.location:
        alert["location"] = payload.location
    raw_input: str = json.dumps(alert, ensure_ascii=False)
    logger.info(
        "[api] POST {}alert/webhook | trace_id={} device_sn={} error_code={}",
        API_PREFIX,
        trace_id,
        payload.device_sn,
        payload.error_code,
    )

    result, approval_payload = _run_workflow(trace_id, "webhook_alert", raw_input)
    response: DiagnoseResponse = _to_diagnose_response(
        result, trace_id, "webhook_alert", approval_payload, (time.perf_counter() - started) * 1000
    )
    logger.success(
        "[api] alert/webhook 完成 | trace_id={} status={} order_no={} 耗时={}ms",
        trace_id,
        response.status,
        response.order_no,
        response.elapsed_ms,
    )
    return response


# ---------------------------------------------------------------------------
# HITL：人工审批恢复
# ---------------------------------------------------------------------------
@router.post(
    "/work-orders/approve",
    response_model=WorkOrderApproveResponse,
    tags=["人工审批"],
    summary="HITL 审批恢复（断点续跑）",
    responses={
        404: {"model": ErrorResponse, "description": "trace_id 对应的工作流不存在"},
        409: {"model": ErrorResponse, "description": "工单号不匹配或该线程无可审批工单"},
        500: {"model": ErrorResponse, "description": "断点恢复失败"},
    },
)
def approve_work_order(
    payload: WorkOrderApproveRequest, request: Request
) -> WorkOrderApproveResponse:
    """对挂起中的工单给出审批结论，并从检查点续跑到结案。

    执行顺序（LangGraph 官方两段式）：

    1. ``get_state`` 校验线程存在、工单号一致、且确实停在审批节点；
    2. ``update_state`` 把 ``approval_status`` / ``approver`` 写进检查点；
    3. ``invoke(None, config)`` 从断点续跑，``human_approval_node`` 检测到已决状态
       后放行，``dispatch_order_node`` 完成库存核销与派工，最后生成报告。

    重复审批是幂等的：已结案（``next`` 为空）时直接回放当前数据库状态，不会二次扣库存。

    Args:
        payload: 审批请求体（``trace_id`` / ``order_no`` / ``approval_status`` / ``approver``）。
        request: FastAPI 请求对象。

    Returns:
        WorkOrderApproveResponse: 审批与派工结果。

    Raises:
        HTTPException: 404 线程不存在；409 工单不匹配或无工单；500 恢复失败。
    """
    started: float = time.perf_counter()
    trace_id: str = payload.trace_id
    request.state.trace_id = trace_id
    graph = get_graph()
    config = make_config(trace_id)

    snapshot = graph.get_state(config)
    values: dict[str, Any] = dict(snapshot.values or {})
    if not values:
        logger.warning("[api] 审批失败：trace_id={} 无对应工作流", trace_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"未找到 trace_id={trace_id} 对应的工作流，请确认报修请求是否成功",
        )

    state_order: Optional[str] = values.get("order_no")
    if not state_order:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="该工作流未生成工单（本故障无需换件），无需审批",
        )
    if state_order != payload.order_no:
        logger.warning(
            "[api] 审批失败：工单号不匹配 | trace_id={} 期望={} 收到={}",
            trace_id,
            state_order,
            payload.order_no,
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"工单号不匹配：线程 {trace_id} 对应工单为 {state_order}，请求为 {payload.order_no}",
        )

    # ---- 幂等：工作流已跑完（next 为空）→ 直接回放当前状态，不重复扣库存 ----
    if not snapshot.next:
        row: Optional[dict[str, Any]] = _fetch_order_row(state_order)
        stored: str = str(
            (row or {}).get("approval_status") or values.get("approval_status") or ""
        ).upper()
        # 只回放已决状态；万一状态是 PENDING（历史脏数据）就按本次请求的结论展示
        current_status: str = stored if stored in ("APPROVED", "REJECTED") else payload.approval_status
        logger.warning(
            "[api] 重复审批（幂等回放）| trace_id={} order_no={} 当前状态={}",
            trace_id,
            state_order,
            current_status,
        )
        return WorkOrderApproveResponse(
            trace_id=trace_id,
            order_no=state_order,
            approval_status=current_status,  # type: ignore[arg-type]  # 取值已被收缩到两态
            approver=str((row or {}).get("approver") or values.get("approver") or payload.approver),
            dispatch_status=(row or {}).get("dispatch_status"),
            stock_deducted=False,
            stock_quantity_after=_fetch_stock(
                values.get("required_part") or (row or {}).get("suggested_part"),
                values.get("selected_warehouse") or (row or {}).get("allocated_warehouse"),
            ),
            final_response=values.get("final_response"),
            elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
            replayed=True,
        )

    part: Optional[str] = values.get("required_part")
    warehouse: Optional[str] = values.get("selected_warehouse")
    stock_before: Optional[int] = _fetch_stock(part, warehouse)
    logger.info(
        "[api] POST {}work-orders/approve | trace_id={} order_no={} decision={} approver={}",
        API_PREFIX,
        trace_id,
        state_order,
        payload.approval_status,
        payload.approver,
    )

    try:
        with trace_context(trace_id):
            # 第 1 步：把审批结论写进检查点（审批终端可立即看到最新状态）
            graph.update_state(
                config,
                {
                    "approval_status": payload.approval_status,
                    "approver": payload.approver,
                },
            )
            # 第 2 步：从断点恢复执行（human_approval_node 已决 → 放行 → 派工 → 报告）
            result: dict[str, Any] = graph.invoke(None, config)
    except Exception as exc:  # noqa: BLE001
        logger.exception("[api] 审批恢复失败 trace_id={} order_no={}", trace_id, state_order)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"断点恢复失败：{type(exc).__name__}: {exc}",
        ) from exc

    if "__interrupt__" in result:
        # 兜底：节点未识别已决状态时会原地打转，这里明确报错而不是假装成功
        logger.error("[api] 审批后工作流仍停在审批节点 trace_id={} order_no={}", trace_id, state_order)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="工作流仍停在审批节点，恢复未生效（请检查 human_approval_node 实现）",
        )

    row = _fetch_order_row(state_order) or {}
    stock_after: Optional[int] = _fetch_stock(
        part or row.get("suggested_part"), warehouse or row.get("allocated_warehouse")
    )
    stock_deducted: bool = (
        payload.approval_status == "APPROVED"
        and stock_before is not None
        and stock_after is not None
        and stock_after == stock_before - 1
    )
    response = WorkOrderApproveResponse(
        trace_id=trace_id,
        order_no=state_order,
        approval_status=payload.approval_status,
        approver=payload.approver,
        dispatch_status=row.get("dispatch_status"),
        stock_deducted=stock_deducted,
        stock_quantity_after=stock_after,
        final_response=result.get("final_response"),
        elapsed_ms=round((time.perf_counter() - started) * 1000, 1),
    )
    logger.success(
        "[api] 审批完成 | trace_id={} order_no={} dispatch_status={} 库存 {}→{} 耗时={}ms",
        trace_id,
        state_order,
        response.dispatch_status,
        stock_before,
        stock_after,
        response.elapsed_ms,
    )
    return response


# ---------------------------------------------------------------------------
# 控制台查询：工单列表 / 备件库存
# ---------------------------------------------------------------------------
#: 工单列表：筛选条件用 ``:param is null or ...`` 的固定模板实现，仍然零字符串拼接
_WORK_ORDERS_WHERE: str = (
    " from work_orders "
    "where (:approval_status is null or approval_status = :approval_status) "
    "  and (:trace_id is null or trace_id = :trace_id)"
)
_WORK_ORDERS_SQL: str = (
    "select order_no, device_sn, error_code, symptom, suggested_part, allocated_warehouse, "
    "approval_status, approver, dispatch_status, trace_id, created_at, updated_at"
    + _WORK_ORDERS_WHERE
    + " order by id desc limit :limit"
)
_WORK_ORDERS_COUNT_SQL: str = "select count(*)" + _WORK_ORDERS_WHERE

#: 备件库存：只列有货行；``part_name`` 为模糊筛选（参数化绑定，无注入面）
_SPARE_PARTS_WHERE: str = " from spare_parts where stock_quantity > 0 and (:pattern is null or part_name like :pattern)"
_SPARE_PARTS_SQL: str = (
    "select part_code, part_name, spec_model, stock_quantity, warehouse_name, "
    "warehouse_location, bin_location, unit_price, updated_at"
    + _SPARE_PARTS_WHERE
    + " order by part_name asc, stock_quantity desc, warehouse_name asc limit :limit"
)
_SPARE_PARTS_COUNT_SQL: str = "select count(*)" + _SPARE_PARTS_WHERE


@router.get(
    "/work-orders",
    response_model=WorkOrderListResponse,
    tags=["数据查询"],
    summary="最新工单列表",
)
def list_work_orders(
    limit: int = Query(default=20, ge=1, le=200, description="返回条数上限"),
    approval_status: Optional[ApprovalDecision] = Query(
        default=None, description="按审批状态筛选：PENDING / APPROVED / REJECTED"
    ),
    trace_id: Optional[str] = Query(
        default=None, min_length=1, max_length=64, description="按链路号筛选（查某次报修的工单）"
    ),
) -> WorkOrderListResponse:
    """按倒序返回最新工单，供前端控制台展示审批与派工状态。

    Args:
        limit: 返回条数上限（1~200）。
        approval_status: 可选审批状态筛选。
        trace_id: 可选链路号筛选。

    Returns:
        WorkOrderListResponse: 总条数 + 工单列表。

    Raises:
        HTTPException: 数据库不可用时返回 503。
    """
    params: dict[str, Any] = {
        "approval_status": (approval_status or None),
        "trace_id": (trace_id or "").strip() or None,
    }
    try:
        with session_scope() as db:
            rows = (
                db.execute(
                    text(_WORK_ORDERS_SQL).bindparams(limit=limit, **params)
                )
                .mappings()
                .all()
            )
            total: int = int(
                db.execute(text(_WORK_ORDERS_COUNT_SQL).bindparams(**params)).scalar() or 0
            )
    except SQLAlchemyError as exc:  # 数据库类故障 → 503；其余异常按 500 上抛
        logger.exception("[api] 查询工单列表失败")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"数据库不可用：{type(exc).__name__}",
        ) from exc

    items: list[WorkOrderItem] = [
        WorkOrderItem(
            order_no=str(row.get("order_no")),
            device_sn=str(row.get("device_sn")),
            error_code=row.get("error_code"),
            symptom=row.get("symptom"),
            suggested_part=row.get("suggested_part"),
            allocated_warehouse=row.get("allocated_warehouse"),
            approval_status=row.get("approval_status"),
            approver=row.get("approver"),
            dispatch_status=row.get("dispatch_status"),
            trace_id=row.get("trace_id"),
            created_at=_format_dt(row.get("created_at")),
            updated_at=_format_dt(row.get("updated_at")),
        )
        for row in rows
    ]
    logger.info("[api] 工单列表 | 返回 {} / 共 {} 条 | 筛选={}", len(items), total, params)
    return WorkOrderListResponse(total=total, limit=limit, items=items)


@router.get(
    "/spare-parts",
    response_model=SparePartListResponse,
    tags=["数据查询"],
    summary="备件实时库存",
)
def list_spare_parts(
    part_name: Optional[str] = Query(
        default=None, min_length=1, max_length=64, description="备件名称（模糊匹配）"
    ),
    limit: int = Query(default=50, ge=1, le=200, description="返回条数上限"),
) -> SparePartListResponse:
    """返回有货备件的实时库存，按「备件名称 + 库存量倒序」排列。

    Args:
        part_name: 可选名称模糊筛选，例如 ``型号A电磁阀``。
        limit: 返回条数上限（1~200）。

    Returns:
        SparePartListResponse: 总条数 + 库存列表。

    Raises:
        HTTPException: 数据库不可用时返回 503。
    """
    keyword: str = (part_name or "").strip()
    params: dict[str, Any] = {"pattern": f"%{keyword}%" if keyword else None}
    try:
        with session_scope() as db:
            rows = (
                db.execute(text(_SPARE_PARTS_SQL).bindparams(limit=limit, **params))
                .mappings()
                .all()
            )
            total: int = int(
                db.execute(text(_SPARE_PARTS_COUNT_SQL).bindparams(**params)).scalar() or 0
            )
    except SQLAlchemyError as exc:
        logger.exception("[api] 查询备件库存失败")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"数据库不可用：{type(exc).__name__}",
        ) from exc

    items: list[SparePartItem] = [
        SparePartItem(
            part_code=row.get("part_code"),
            part_name=row.get("part_name"),
            spec_model=row.get("spec_model"),
            stock_quantity=int(row.get("stock_quantity") or 0),
            warehouse_name=row.get("warehouse_name"),
            warehouse_location=row.get("warehouse_location"),
            bin_location=row.get("bin_location"),
            unit_price=_jsonable(row.get("unit_price")),
            updated_at=_format_dt(row.get("updated_at")),
        )
        for row in rows
    ]
    logger.info("[api] 备件库存 | 返回 {} / 共 {} 条 | 筛选={}", len(items), total, params)
    return SparePartListResponse(total=total, items=items)


# ---------------------------------------------------------------------------
# 运维：健康检查
# ---------------------------------------------------------------------------
@health_router.get(
    "/health",
    response_model=HealthResponse,
    tags=["运维监控"],
    summary="健康检查（MySQL / Qdrant / 检查点）",
)
def health() -> HealthResponse:
    """探测三件关键依赖：MySQL、本地 Qdrant 知识库、LangGraph 检查点存储。

    Returns:
        HealthResponse: 任一组件异常时 ``status="degraded"``（HTTP 仍为 200，
        便于监控系统读取明细；如需 503 语义，可在网关层按 ``status`` 字段判断）。
    """
    mysql_ok: bool = verify_mysql_connection()
    mysql: ComponentHealth = ComponentHealth(
        ok=mysql_ok,
        detail=settings.mysql_url_masked if mysql_ok else "SELECT 1 失败，请查看服务日志",
    )

    try:
        exists, points = knowledge_base_health()
        qdrant: ComponentHealth = ComponentHealth(
            ok=exists,
            detail=(
                f"集合 {MANUAL_COLLECTION_NAME} 可用，点位 {points} 条"
                if exists
                else f"集合 {MANUAL_COLLECTION_NAME} 缺失，请执行 scripts/init_qdrant_data.py"
            ),
        )
    except Exception as exc:  # noqa: BLE001 - 健康检查不允许抛异常
        qdrant = ComponentHealth(ok=False, detail=f"本地 Qdrant 不可用：{type(exc).__name__}: {exc}")

    try:
        saver = get_checkpointer()
        checkpoint_path = settings.checkpoint_db_abspath
        checkpointer: ComponentHealth = ComponentHealth(
            ok=checkpoint_path.exists(),
            detail=f"{type(saver).__name__} → {checkpoint_path}",
        )
    except Exception as exc:  # noqa: BLE001
        checkpointer = ComponentHealth(
            ok=False, detail=f"检查点后端不可用：{type(exc).__name__}: {exc}"
        )

    healthy: bool = mysql.ok and qdrant.ok and checkpointer.ok
    response = HealthResponse(
        status="ok" if healthy else "degraded",
        version=settings.APP_VERSION,
        app_env=settings.APP_ENV,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        mysql=mysql,
        qdrant=qdrant,
        checkpointer=checkpointer,
    )
    if not healthy:
        logger.warning(
            "[api] 健康检查降级 | mysql={} qdrant={} checkpointer={}",
            mysql.ok,
            qdrant.ok,
            checkpointer.ok,
        )
    return response
