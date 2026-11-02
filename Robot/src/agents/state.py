"""LangGraph 多智能体全局状态定义。

``AgentState`` 是贯穿整条排障流水线的**唯一共享上下文**：所有节点都只读取/写入
这里的字段，节点之间不直接传参，因此新增字段时只要在 TypedDict 里声明即可。

字段分组（与业务流程对应）：

* 接入与识别：``channel`` / ``raw_input`` / ``device_sn`` / ``device_location`` /
  ``error_code`` / ``symptom``
* 诊断与备件：``sop_guide`` / ``required_part``
* 库存与工单：``stock_info`` / ``selected_warehouse`` / ``order_no``
* 人在回路：``approval_status`` / ``approver``
* 输出：``final_response``
"""

from __future__ import annotations

from typing import Literal, Optional, TypedDict

__all__ = ["AgentState", "ChannelType", "ApprovalStatus"]


#: 接入通道：人工口语化报修 / 边缘 Webhook 告警
ChannelType = Literal["human_chat", "webhook_alert"]

#: 审批状态机取值
ApprovalStatus = Literal["PENDING", "APPROVED", "REJECTED"]


class AgentState(TypedDict, total=False):
    """排障与备件调度工作流的全局状态。

    使用 ``total=False``：所有字段都是「可选」的，节点只需返回自己负责的增量字段，
    LangGraph 会把返回的字典合并进全局状态（而不是要求每个节点回填全部字段）。
    """

    # ---------------------- 追踪与接入 ----------------------
    trace_id: str
    """全链路追踪 ID，同时作为 LangGraph 的 ``thread_id`` 用于断点续跑。"""

    channel: ChannelType
    """接入通道：``human_chat``（工程师口语报修）或 ``webhook_alert``（设备自动上报）。"""

    raw_input: str
    """用户原始自然语言描述或 Webhook 告警文本。"""

    # ---------------------- 故障解析结果 ----------------------
    device_sn: Optional[str]
    """机器人序列号，例如 ``AGV-SZ-003``。"""

    device_location: Optional[str]
    """设备所在城市，例如 ``苏州``；决定备件就近调拨的优先级。"""

    error_code: Optional[str]
    """故障错误码，例如 ``E-201``。"""

    symptom: Optional[str]
    """故障现象描述（口语化）。"""

    # ---------------------- 诊断结果 ----------------------
    sop_guide: Optional[str]
    """RAG 检索出的排障步骤（含手册章节与原文引用）。"""

    required_part: Optional[str]
    """需要更换的备件标准名称；``None`` 表示仅需复位/调试，无需备件。"""

    # ---------------------- 库存与工单 ----------------------
    stock_info: Optional[list[dict]]
    """可用备件库存列表（已按就近原则排序）。"""

    selected_warehouse: Optional[str]
    """推荐调拨的仓库名称。"""

    order_no: Optional[str]
    """生成的工单号，例如 ``WO-202610-0312``。"""

    # ---------------------- 人在回路 ----------------------
    approval_status: Optional[ApprovalStatus]
    """审批状态：``PENDING`` 挂起等待 / ``APPROVED`` 通过 / ``REJECTED`` 驳回。"""

    approver: Optional[str]
    """审批人姓名。"""

    # ---------------------- 最终输出 ----------------------
    final_response: Optional[str]
    """最终汇总输出的排障与调度报告（Markdown 文本）。"""
