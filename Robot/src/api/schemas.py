"""接入层请求 / 响应 Schema（Pydantic v2）。

设计约定
--------
* **入口即清洗**：设备 SN、故障码统一大写，文本去首尾空白。避免同一条数据在
  状态机里出现 ``agv-sz-003`` 与 ``AGV-SZ-003`` 两种写法，把 Qdrant 的
  ``error_code`` 精确过滤和 MySQL 的 ``device_sn`` 等值查询一起打偏；
* **trace_id 双入口**：请求体显式传入的 ``trace_id`` 优先，其次取 ``X-Trace-Id``
  请求头，最后由服务端生成。无论哪条路径，它都会成为 LangGraph 的 ``thread_id``；
* **响应字段与 ``AgentState`` 同名**：前端控制台拿到即可直接渲染，无需二次映射。
"""

from __future__ import annotations

import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "ChannelType",
    "WorkflowStatus",
    "ApprovalDecision",
    "is_valid_trace_id",
    "ChatDiagnoseRequest",
    "WebhookAlertRequest",
    "WorkOrderApproveRequest",
    "StockItem",
    "DiagnoseResponse",
    "WorkOrderApproveResponse",
    "WorkOrderItem",
    "WorkOrderListResponse",
    "SparePartItem",
    "SparePartListResponse",
    "ComponentHealth",
    "HealthResponse",
    "ErrorResponse",
]

#: 接入通道：人工自然语言报修 / 边缘设备 Webhook 告警
ChannelType = Literal["human_chat", "webhook_alert"]

#: 工作流结束方式：挂起等审批 / 已完结
WorkflowStatus = Literal["WAITING_APPROVAL", "COMPLETED"]

#: 审批结论（只允许这两种，避免脏值写进工单）
ApprovalDecision = Literal["APPROVED", "REJECTED"]

#: trace_id 允许的字符集：只放行可安全写日志、可作 thread_id 的字符
_TRACE_ID_PATTERN: str = r"^[A-Za-z0-9._:-]{4,64}$"
_TRACE_ID_RE: re.Pattern[str] = re.compile(_TRACE_ID_PATTERN)


def is_valid_trace_id(value: Optional[str]) -> bool:
    """判断外部传入的追踪号是否可直接采用。

    追踪号会被写进日志与 LangGraph 的 ``thread_id``，因此不允许换行、空格等
    可能导致日志注入的字符；不合规时调用方应改为服务端生成。

    Args:
        value: 候选追踪号（通常来自 ``X-Trace-Id`` 请求头）。

    Returns:
        bool: 合规返回 ``True``。
    """
    cleaned: str = (value or "").strip()
    return bool(cleaned) and _TRACE_ID_RE.match(cleaned) is not None


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------
class ChatDiagnoseRequest(BaseModel):
    """通道 A：工程师口语化报修。"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "苏州厂区的 AGV-SZ-003 小车夹爪漏气合不上，报 E-201",
                "trace_id": "TR-20261029-0001",
            }
        }
    )

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="现场故障的自然语言描述（设备 SN、故障码、现象可混写）",
    )
    trace_id: Optional[str] = Field(
        default=None,
        pattern=_TRACE_ID_PATTERN,
        description="全链路追踪号；留空则由服务端生成（同时作为 LangGraph thread_id）",
    )

    @field_validator("query")
    @classmethod
    def _strip_query(cls, value: str) -> str:
        cleaned: str = value.strip()
        if not cleaned:
            raise ValueError("query 不能为空白字符串")
        return cleaned

    @field_validator("trace_id")
    @classmethod
    def _strip_trace_id(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned: str = value.strip()
        return cleaned or None


class WebhookAlertRequest(BaseModel):
    """通道 B：联网设备主动上报的告警报文。"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "device_sn": "AGV-SZ-003",
                "location": "苏州",
                "error_code": "E-201",
                "symptom": "夹爪持续漏气、无法闭合，面板常亮 E-201 报警",
            }
        }
    )

    device_sn: str = Field(..., min_length=1, max_length=64, description="设备序列号")
    location: Optional[str] = Field(
        default=None,
        max_length=64,
        description="设备上报的城市；台账里查不到该 SN 时作为就近调拨的兜底依据",
    )
    error_code: str = Field(..., min_length=1, max_length=32, description="故障码，如 E-201")
    symptom: str = Field(..., min_length=1, max_length=2000, description="告警现象描述")

    @field_validator("device_sn")
    @classmethod
    def _upper_sn(cls, value: str) -> str:
        cleaned: str = value.strip().upper()
        if not cleaned:
            raise ValueError("device_sn 不能为空白字符串")
        return cleaned

    @field_validator("error_code")
    @classmethod
    def _upper_code(cls, value: str) -> str:
        cleaned: str = value.strip().upper()
        if not cleaned:
            raise ValueError("error_code 不能为空白字符串")
        return cleaned

    @field_validator("symptom")
    @classmethod
    def _strip_symptom(cls, value: str) -> str:
        cleaned: str = value.strip()
        if not cleaned:
            raise ValueError("symptom 不能为空白字符串")
        return cleaned

    @field_validator("location")
    @classmethod
    def _strip_location(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned: str = value.strip()
        return cleaned or None


class WorkOrderApproveRequest(BaseModel):
    """HITL 人工审批请求：对挂起中的工单给出结论。"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "trace_id": "TR-20261029-0001",
                "order_no": "WO-202610-0301",
                "approval_status": "APPROVED",
                "approver": "张主管",
            }
        }
    )

    trace_id: str = Field(
        ...,
        min_length=4,
        max_length=64,
        pattern=_TRACE_ID_PATTERN,
        description="挂起中的工作流 thread_id（即首次报修时返回的 trace_id）",
    )
    order_no: str = Field(
        ..., min_length=1, max_length=64, description="待审批工单号，须与该线程一致"
    )
    approval_status: ApprovalDecision = Field(..., description="APPROVED 通过 / REJECTED 驳回")
    approver: str = Field(default="张主管", min_length=1, max_length=64, description="审批人")

    @field_validator("approval_status", mode="before")
    @classmethod
    def _upper_status(cls, value: Any) -> Any:
        """容忍前端传小写 ``approved``；非字符串值原样交给 Literal 校验报错。"""
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("order_no")
    @classmethod
    def _strip_order_no(cls, value: str) -> str:
        cleaned: str = value.strip()
        if not cleaned:
            raise ValueError("order_no 不能为空白字符串")
        return cleaned

    @field_validator("approver")
    @classmethod
    def _strip_approver(cls, value: str) -> str:
        cleaned: str = value.strip()
        if not cleaned:
            raise ValueError("approver 不能为空白字符串")
        return cleaned


# ---------------------------------------------------------------------------
# 业务数据模型
# ---------------------------------------------------------------------------
class StockItem(BaseModel):
    """一条备件库存记录（``spare_parts`` 行）。"""

    part_code: Optional[str] = Field(default=None, description="备件编码")
    part_name: Optional[str] = Field(default=None, description="备件标准名称")
    spec_model: Optional[str] = Field(default=None, description="规格型号")
    stock_quantity: int = Field(default=0, description="可用库存数量")
    warehouse_name: Optional[str] = Field(default=None, description="仓库名称")
    warehouse_location: Optional[str] = Field(default=None, description="仓库所在城市")
    bin_location: Optional[str] = Field(default=None, description="货位")
    unit_price: Optional[float] = Field(default=None, description="单价")
    is_preferred_location: bool = Field(default=False, description="是否与设备同城（就近优选）")


class WorkOrderItem(BaseModel):
    """一条工单记录（``work_orders`` 行）。"""

    order_no: str
    device_sn: str
    error_code: Optional[str] = None
    symptom: Optional[str] = None
    suggested_part: Optional[str] = None
    allocated_warehouse: Optional[str] = None
    approval_status: Optional[str] = None
    approver: Optional[str] = None
    dispatch_status: Optional[str] = None
    trace_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class SparePartItem(StockItem):
    """备件库存展示行（含最后更新时间，便于控制台显示数据新鲜度）。"""

    updated_at: Optional[str] = None


# ---------------------------------------------------------------------------
# 响应模型
# ---------------------------------------------------------------------------
class DiagnoseResponse(BaseModel):
    """双通道诊断接口的统一响应。

    ``status`` 为 ``WAITING_APPROVAL`` 时表示工作流已停在人工审批节点，
    此时 ``order_no`` / ``approval_payload`` 有值，前端可据此渲染审批卡片。
    """

    trace_id: str = Field(..., description="本次请求的全链路追踪号（= LangGraph thread_id）")
    channel: ChannelType = Field(..., description="接入通道")
    status: WorkflowStatus = Field(..., description="WAITING_APPROVAL 挂起等审批 / COMPLETED 已完结")
    device_sn: Optional[str] = None
    device_location: Optional[str] = None
    error_code: Optional[str] = None
    symptom: Optional[str] = None
    sop_guide: Optional[str] = Field(default=None, description="手册检索出的排障 SOP 原文")
    required_part: Optional[str] = Field(default=None, description="需更换备件；为空表示无需换件")
    stock_info: list[StockItem] = Field(default_factory=list, description="可用库存（就近优先排序）")
    selected_warehouse: Optional[str] = Field(default=None, description="推荐调拨仓库")
    order_no: Optional[str] = Field(default=None, description="生成的工单号（待审批草稿）")
    approval_status: Optional[str] = Field(default=None, description="PENDING / APPROVED / REJECTED")
    approval_payload: Optional[dict[str, Any]] = Field(
        default=None, description="挂起时的审批提示载荷（工单、备件、仓库、库存）"
    )
    final_response: Optional[str] = Field(default=None, description="完整售后处理报告（Markdown）")
    elapsed_ms: float = Field(default=0.0, description="本次处理耗时（毫秒）")


class WorkOrderApproveResponse(BaseModel):
    """审批恢复接口的响应。"""

    trace_id: str
    order_no: str
    approval_status: ApprovalDecision
    approver: str
    dispatch_status: Optional[str] = Field(default=None, description="DISPATCHED 已派工 / CLOSED 已关闭")
    stock_deducted: bool = Field(default=False, description="本次是否核销了 1 件库存")
    stock_quantity_after: Optional[int] = Field(default=None, description="核销后该仓库存数量")
    final_response: Optional[str] = Field(default=None, description="恢复后生成的完整报告")
    elapsed_ms: float = Field(default=0.0)
    replayed: bool = Field(
        default=False, description="True 表示工作流早已结案，本次为幂等回放（未重复扣库存）"
    )


class WorkOrderListResponse(BaseModel):
    """工单列表（控制台展示）。"""

    total: int = Field(..., description="满足筛选条件的总条数")
    limit: int = Field(..., description="本次返回的最大条数")
    items: list[WorkOrderItem] = Field(default_factory=list)


class SparePartListResponse(BaseModel):
    """备件库存列表（控制台展示）。"""

    total: int = Field(..., description="满足筛选条件的总条数")
    items: list[SparePartItem] = Field(default_factory=list)


class ComponentHealth(BaseModel):
    """单个依赖组件的健康状态。"""

    ok: bool
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    """``GET /health`` 响应：MySQL、本地 Qdrant 与检查点存储三件套。"""

    status: Literal["ok", "degraded"]
    version: str
    app_env: str
    timestamp: str
    mysql: ComponentHealth
    qdrant: ComponentHealth
    checkpointer: ComponentHealth


class ErrorResponse(BaseModel):
    """统一错误响应（FastAPI ``HTTPException`` 的默认结构）。"""

    detail: str
