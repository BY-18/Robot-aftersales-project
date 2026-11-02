"""LangGraph 工作流节点实现（排障 → 备件 → 审批 → 派工 → 报告）。

节点职责一览
------------
=========================  ==========================================
节点                        做的事
=========================  ==========================================
``parse_fault_node``        LLM 抽取 SN / 故障码 / 现象，并查设备所在城市
``rag_diagnose_node``       Qdrant 手册检索，产出 SOP 与所需备件
``stock_check_node``        就近多仓查库存 + 落库 PENDING 工单草稿
``human_approval_node``     HITL 挂起（``interrupt``），等外部审批信号
``dispatch_order_node``     写入审批结果；通过则核销库存并置派工
``generate_report_node``    汇总生成对客报告
=========================  ==========================================

设计约定
--------
* **节点只返回增量字段**：LangGraph 会把返回的 dict 合并进 ``AgentState``，
  因此不需要在每个节点回填全部字段。
* **工具一律走 LangChain 工具对象**：``query_device_info.invoke({...})``，
  这样单元测试与后续 Agent 的 Tool Calling 共用同一条代码路径。
* **LLM 抽取失败必须能兜底**：DeepSeek 超时/欠费/返回空时退化为正则抽取，
  保证车间断网时工作流仍能跑通（对应需求里的私有化降级要求）。
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Literal

from langgraph.types import interrupt
from pydantic import BaseModel, Field
from sqlalchemy import text

from src.agents.state import AgentState
from src.core.db import session_scope
from src.core.llm import get_llm
from src.core.logger import get_trace_id, logger
from src.tools.rag_tool import fetch_repair_hits, format_hits
from src.tools.sql_tool import (
    create_work_order_draft,
    query_device_info,
    query_spare_part_stock,
)

__all__ = [
    "parse_fault_node",
    "rag_diagnose_node",
    "stock_check_node",
    "human_approval_node",
    "dispatch_order_node",
    "generate_report_node",
    "should_check_stock",
    "should_request_approval",
]

#: 工单派工状态：审批通过 → 已派工；驳回 → 关闭
_DISPATCH_STATUS_APPROVED: str = "DISPATCHED"
_DISPATCH_STATUS_REJECTED: str = "CLOSED"

#: 「已决」审批状态。HITL 接口会先 ``update_state`` 写入结论再 ``invoke(None)`` 续跑，
#: 此时节点必须能识别出"已经批过了"，否则会再次 ``interrupt()`` 原地打转。
_DECIDED_STATUSES: frozenset[str] = frozenset({"APPROVED", "REJECTED"})

#: 审批人缺省值（外部只给了结论、没给审批人时使用）
_APPROVER_UNKNOWN: str = "未记录"


# ---------------------------------------------------------------------------
# 通用辅助
# ---------------------------------------------------------------------------
def _invoke_tool(tool_obj: Any, payload: dict[str, Any]) -> Any:
    """调用 LangChain 工具并兼容「返回对象 / 返回 JSON 字符串」两种形态。

    Args:
        tool_obj: ``@tool`` 装饰后的工具对象。
        payload: 工具入参。

    Returns:
        Any: 解析后的 Python 对象。
    """
    result: Any = tool_obj.invoke(payload)
    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return result
    return result


# ---------------------------------------------------------------------------
# 节点 1：故障解析与设备定位
# ---------------------------------------------------------------------------
class _FaultSlots(BaseModel):
    """LLM 结构化抽取的结果模型（用于 ``with_structured_output``）。"""

    device_sn: str | None = Field(
        default=None, description="机器人设备序列号，例如 AGV-SZ-003；没有则返回 null"
    )
    error_code: str | None = Field(
        default=None, description="故障错误码，例如 E-201；没有则返回 null"
    )
    symptom: str | None = Field(
        default=None, description="故障现象的一句话描述，保留部位与现象，不要编造"
    )


_EXTRACT_PROMPT: str = (
    "你是工业机器人售后排障系统的信息抽取模块。请从下面的现场报修文本中抽取字段，"
    "不要输出任何解释。\n"
    "输出必须是 json 对象，字段定义如下：\n"
    '  "device_sn": 设备序列号，形如 AGV-SZ-003 / ARM-X6-002；没有则 null；\n'
    '  "error_code": 故障码，形如 E-201 / E-104，保持大写；没有则 null；\n'
    '  "symptom": 一句话概括故障部位与现象；没有则 null。\n'
    "规则：\n"
    "1. 设备的型号前缀不属于故障码：AGV-SZ-003 里的 SZ-003 不是故障码；\n"
    "2. 文本中没有的信息一律返回 null，严禁编造。\n\n"
    "报修文本：\n{raw_input}"
)

#: 设备 SN 形态：AGV-SZ-003 / ARM-X6-002 / L-500-001
_SN_PATTERN: re.Pattern[str] = re.compile(r"\b([A-Z]{2,6}(?:-[A-Z0-9]{1,6})?-\d{2,6})\b")
#: 故障码形态：E-201 / E104。后顾断言排除「前接字母数字或连字符」的情况，
#: 否则会从 AGV-SZ-003 里抠出假故障码 SZ-003；中文紧邻（如「报E-201」）仍可命中。
_CODE_PATTERN: re.Pattern[str] = re.compile(r"(?<![A-Za-z0-9-])([A-Z]{1,3})-?(\d{2,4})\b")

#: 结构化输出失败时的二次尝试顺序。
#: DeepSeek 目前**不支持** langchain 默认的 ``json_schema`` 严格模式（会返回
#: ``400 This response_format type is unavailable now``），因此必须显式指定
#: ``json_mode``（走 ``response_format={"type": "json_object"}``）或函数调用方式。
_STRUCTURED_METHODS: tuple[str, ...] = ("json_mode", "function_calling")

_NULL_WORDS: frozenset[str] = frozenset({"", "-", "N/A", "NA", "NONE", "NULL", "UNKNOWN"})


def _clean_code(value: Any, *, is_sn: bool) -> str | None:
    """清洗 SN / 故障码：去空格、转大写、修补 ``E201`` → ``E-201``。

    Args:
        value: 模型或正则给出的原始值。
        is_sn: 为 ``True`` 时按设备 SN 处理（不做故障码补连字符的修正）。

    Returns:
        str | None: 清洗后的值；空值 / 模型自造的 "null" 文案返回 ``None``。
    """
    raw: str = re.sub(r"\s+", "", str(value or "")).upper()
    if raw in _NULL_WORDS:
        return None
    if not is_sn:
        raw = re.sub(r"^([A-Z]{1,4})(\d{2,4})$", r"\1-\2", raw)
    return raw or None


def _normalize_slots(slots: dict[str, Any]) -> dict[str, Any]:
    """规范化 LLM 抽出的槽位，并修正「把 SN 尾巴当故障码」这类常见错误。

    Args:
        slots: ``{"device_sn": ..., "error_code": ..., "symptom": ...}``。

    Returns:
        dict[str, Any]: 清洗后的槽位。
    """
    device_sn: str | None = _clean_code(slots.get("device_sn"), is_sn=True)
    error_code: str | None = _clean_code(slots.get("error_code"), is_sn=False)
    symptom: str | None = str(slots.get("symptom") or "").strip() or None

    # AGV-SZ-003 的尾部 SZ-003 被误当故障码时丢弃，避免污染手册检索过滤条件
    if device_sn and error_code and device_sn.endswith(error_code):
        logger.debug(
            "[node:parse_fault] 故障码 {} 疑似取自设备 SN {} 的尾部，已丢弃", error_code, device_sn
        )
        error_code = None
    return {"device_sn": device_sn, "error_code": error_code, "symptom": symptom}


def _load_json_object(text: str) -> dict[str, Any] | None:
    """从模型回复中抠出 JSON 对象（兼容 ```json 代码块与前后闲聊）。

    Args:
        text: 模型返回的原始文本。

    Returns:
        dict[str, Any] | None: 解析出的对象；无法解析时为 ``None``。
    """
    cleaned: str = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.S)
    if fenced:
        cleaned = fenced.group(1).strip()
    start: int = cleaned.find("{")
    end: int = cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data: Any = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _llm_slots(raw_input: str) -> dict[str, Any]:
    """用 DeepSeek 抽取槽位（三级尝试，任一层成功即返回）。

    1. ``json_mode``：``response_format={"type": "json_object"}``，DeepSeek 支持；
    2. ``function_calling``：工具调用式结构化输出；
    3. 纯文本 + 手工解析 JSON：应对连 ``response_format`` 都不支持的私有化端点。

    Args:
        raw_input: 原始报修文本。

    Returns:
        dict[str, Any]: 规范化后的槽位。

    Raises:
        Exception: 上述三级全部失败时向上抛出，由 :func:`_extract_slots` 降级为正则。
    """
    llm: Any = get_llm()
    prompt: str = _EXTRACT_PROMPT.format(raw_input=raw_input)
    failures: list[str] = []

    for method in _STRUCTURED_METHODS:
        try:
            parsed: Any = llm.with_structured_output(_FaultSlots, method=method).invoke(prompt)
            if isinstance(parsed, _FaultSlots):
                slots: dict[str, Any] = parsed.model_dump()
            elif isinstance(parsed, dict):
                slots = _FaultSlots(**parsed).model_dump()
            else:  # pragma: no cover - 防御性分支
                raise ValueError(f"结构化输出类型异常：{type(parsed)}")
            logger.debug("[node:parse_fault] 槽位抽取成功（method={}）", method)
            return _normalize_slots(slots)
        except Exception as exc:  # noqa: BLE001 - 逐级降级
            failures.append(f"{method}: {exc}")
            logger.debug("[node:parse_fault] 抽取方式 {} 不可用：{}", method, exc)

    # 兜底：直接读模型文本里的 JSON（部分端点连 response_format 都不识别）
    message: Any = llm.invoke(prompt)
    data: dict[str, Any] | None = _load_json_object(str(getattr(message, "content", "") or ""))
    if data is None:
        raise RuntimeError("DeepSeek 未返回可解析的 JSON；已尝试：" + " | ".join(failures))
    logger.debug("[node:parse_fault] 槽位抽取成功（method=plain_text）")
    return _normalize_slots(_FaultSlots(**data).model_dump())


def _regex_slots(raw_input: str) -> dict[str, Any]:
    """正则兜底抽取（DeepSeek 不可用时使用）。

    车间断网、API 欠费、模型超时都会走到这里：虽然不如 LLM 灵活，
    但设备 SN 与故障码格式高度固定，正则足够可靠。

    先把命中的 SN 从文本中抹掉再找故障码，否则 ``AGV-SZ-003`` 会被抠出假故障码
    ``SZ-003``，进而让手册检索的错误码过滤条件失效。

    Args:
        raw_input: 原始报修文本。

    Returns:
        dict: 含 device_sn / error_code / symptom（缺失为 None）。
    """
    text: str = raw_input.upper()
    sn_match: re.Match[str] | None = _SN_PATTERN.search(text)
    device_sn: str | None = sn_match.group(1) if sn_match else None

    remainder: str = re.sub(re.escape(device_sn), " ", text) if device_sn else text
    code_match: re.Match[str] | None = _CODE_PATTERN.search(remainder)
    error_code: str | None = (
        f"{code_match.group(1)}-{code_match.group(2)}" if code_match else None
    )
    return {
        "device_sn": device_sn,
        "error_code": error_code,
        "symptom": raw_input.strip() or None,
    }


def _extract_slots(raw_input: str) -> tuple[dict[str, Any], str]:
    """优先用 DeepSeek 抽取槽位，失败则退化为正则。

    Args:
        raw_input: 原始报修文本。

    Returns:
        tuple[dict, str]: ``(槽位字典, 抽取来源)``，来源为 ``deepseek`` 或 ``regex``。
    """
    try:
        slots: dict[str, Any] = _llm_slots(raw_input)
        if not any(slots.get(k) for k in ("device_sn", "error_code", "symptom")):
            raise ValueError("LLM 抽取结果全为空")
        return slots, "deepseek"
    except Exception as exc:  # noqa: BLE001 - 任何 LLM 故障都必须降级
        logger.warning("[node:parse_fault] DeepSeek 抽取失败，退化为正则抽取：{}", exc)
        return _regex_slots(raw_input), "regex"


def _slots_from_webhook(raw_input: str) -> dict[str, Any] | None:
    """通道 B：边缘设备 Webhook 报文是结构化 JSON，直接取字段，无需 LLM。

    Args:
        raw_input: JSON 字符串。

    Returns:
        dict[str, Any] | None: 解析成功返回槽位，非 JSON 或缺少关键字段返回 None。
    """
    try:
        payload: Any = json.loads(raw_input)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None

    device_sn = payload.get("device_sn") or payload.get("sn") or payload.get("deviceSn")
    error_code = payload.get("error_code") or payload.get("code") or payload.get("errorCode")
    symptom = payload.get("symptom") or payload.get("message") or payload.get("msg")
    if not device_sn and not error_code:
        return None
    slots: dict[str, Any] = _normalize_slots(
        {"device_sn": device_sn, "error_code": error_code, "symptom": symptom}
    )
    # 设备上报的城市：台账里查不到该 SN 时作为兜底，仍可参与备件就近调拨排序
    reported: str = str(payload.get("location") or payload.get("city") or "").strip()
    slots["device_location"] = reported or None
    return slots


def parse_fault_node(state: AgentState) -> dict[str, Any]:
    """解析报修输入并定位设备。

    流程：Webhook 报文直接取字段 → 否则 DeepSeek 抽取（失败退化正则）→
    按 ``device_sn`` 查设备档案拿到所在城市 ``device_location``。

    Args:
        state: 全局状态（需含 ``raw_input``）。

    Returns:
        dict: 增量状态（device_sn / error_code / symptom / device_location）。
    """
    started: float = time.perf_counter()
    raw_input: str = (state.get("raw_input") or "").strip()
    channel: str = state.get("channel") or "human_chat"
    logger.info(
        "[node:parse_fault] 开始 | channel={} trace_id={} raw_input={!r}",
        channel,
        state.get("trace_id") or get_trace_id(),
        raw_input[:120],
    )

    slots: dict[str, Any] | None = None
    source: str = "webhook_json"
    if channel == "webhook_alert":
        slots = _slots_from_webhook(raw_input)
        if slots is None:
            logger.info("[node:parse_fault] Webhook 报文非结构化，转为常规抽取")

    if slots is None:
        slots, source = _extract_slots(raw_input)

    device_sn: str | None = slots.get("device_sn")
    error_code: str | None = slots.get("error_code")
    symptom: str | None = slots.get("symptom") or raw_input or None
    # 仅 webhook 通道会带：设备自报城市，作为台账缺失时的兜底
    reported_location: str | None = slots.get("device_location")

    # 设备定位：只有拿到 SN 才能查城市，失败也不阻断主流程
    device_location: str | None = reported_location or None
    if device_sn:
        device: Any = _invoke_tool(query_device_info, {"device_sn": device_sn})
        if isinstance(device, dict) and device.get("found"):
            device_location = device.get("location") or reported_location
            logger.info(
                "[node:parse_fault] 设备命中 device_sn={} model={} location={} workshop={}",
                device_sn,
                device.get("model_name"),
                device_location,
                device.get("workshop"),
            )
        else:
            logger.warning(
                "[node:parse_fault] 设备台账未找到 SN={}，改用上报城市={}（无则按无优先仓处理）",
                device_sn,
                reported_location or "无",
            )
    else:
        logger.warning("[node:parse_fault] 未能从输入中识别设备 SN")

    elapsed_ms: float = (time.perf_counter() - started) * 1000
    logger.success(
        "[node:parse_fault] 完成 | 来源={} SN={} code={} location={} 耗时={:.1f}ms",
        source,
        device_sn,
        error_code,
        device_location,
        elapsed_ms,
    )
    return {
        "device_sn": device_sn,
        "error_code": error_code,
        "symptom": symptom,
        "device_location": device_location,
    }


# ---------------------------------------------------------------------------
# 节点 2：知识库排障诊断
# ---------------------------------------------------------------------------
def rag_diagnose_node(state: AgentState) -> dict[str, Any]:
    """检索原厂手册，产出排障 SOP 与所需备件。

    这里直接调用 ``fetch_repair_hits``（``search_repair_manual`` 工具的核心实现），
    因为节点需要 ``required_part`` 这种结构化字段：读字典比正则解析自己生成的
    文本更可靠，也避免同一次检索重复打两次 embedding 请求。

    Args:
        state: 全局状态（需含 symptom / error_code）。

    Returns:
        dict: 增量状态（sop_guide / required_part）。
    """
    started: float = time.perf_counter()
    query: str = (state.get("symptom") or state.get("raw_input") or "").strip()
    error_code: str | None = state.get("error_code")
    logger.info("[node:rag_diagnose] 开始 | query={!r} error_code={}", query[:120], error_code)

    try:
        hits, mode = fetch_repair_hits(query=query, error_code=error_code)
    except Exception as exc:  # noqa: BLE001 - 知识库未初始化等情况不应炸掉整个工作流
        logger.error("[node:rag_diagnose] 手册检索失败：{}", exc)
        return {
            "sop_guide": f"手册检索失败：{exc}",
            "required_part": None,
        }

    elapsed_ms: float = (time.perf_counter() - started) * 1000
    sop_guide: str = format_hits(hits, query, error_code, mode, elapsed_ms)
    required_part: str | None = hits[0].get("required_part") if hits else None

    logger.success(
        "[node:rag_diagnose] 完成 | 命中={} 模式={} 备件={} 耗时={:.1f}ms",
        len(hits),
        mode,
        required_part or "无需换件",
        elapsed_ms,
    )
    if not hits:
        logger.warning("[node:rag_diagnose] 手册未命中，将按「无需换件」路径生成报告")

    return {"sop_guide": sop_guide, "required_part": required_part}


def should_check_stock(state: AgentState) -> Literal["stock_check", "generate_report"]:
    """条件路由：需要换件才去查库存，否则直接出报告。

    Args:
        state: 全局状态。

    Returns:
        str: 下一节点名。
    """
    required_part: str | None = state.get("required_part")
    if required_part:
        logger.info("[route] required_part={} → stock_check", required_part)
        return "stock_check"
    logger.info("[route] 无需更换备件 → generate_report")
    return "generate_report"


# ---------------------------------------------------------------------------
# 节点 3：库存核验与工单草稿
# ---------------------------------------------------------------------------
def stock_check_node(state: AgentState) -> dict[str, Any]:
    """按就近原则选仓并生成 PENDING 工单草稿。

    排序由 SQL 工具负责：同城仓库优先，其次按库存量降序。
    本地无货时会自然落到异地总仓；全库无货则跳过工单，报告里给出缺货预警。

    Args:
        state: 全局状态（需含 required_part / device_location）。

    Returns:
        dict: 增量状态（stock_info / selected_warehouse / order_no / approval_status）。
    """
    started: float = time.perf_counter()
    part: str | None = state.get("required_part")
    location: str | None = state.get("device_location")
    trace_id: str = state.get("trace_id") or get_trace_id()
    logger.info(
        "[node:stock_check] 开始 | part={!r} preferred_location={!r} trace_id={}",
        part,
        location,
        trace_id,
    )

    rows: Any = _invoke_tool(
        query_spare_part_stock, {"part_name": part or "", "preferred_location": location}
    )
    stock_info: list[dict[str, Any]] = rows if isinstance(rows, list) else []

    if not stock_info:
        logger.warning("[node:stock_check] {} 在全部仓库均无可用库存，跳过工单生成", part)
        return {"stock_info": [], "selected_warehouse": None, "order_no": None}

    selected: dict[str, Any] = stock_info[0]
    warehouse: str = str(selected.get("warehouse_name") or "")
    logger.info(
        "[node:stock_check] 选中仓库 {} | 库存 {} | 货位 {} | 就近={} | 备选仓数={}",
        warehouse,
        selected.get("stock_quantity"),
        selected.get("bin_location"),
        selected.get("is_preferred_location"),
        len(stock_info) - 1,
    )

    draft: Any = _invoke_tool(
        create_work_order_draft,
        {
            "device_sn": state.get("device_sn") or "",
            "error_code": state.get("error_code") or "",
            "symptom": state.get("symptom") or "",
            "suggested_part": part,
            "allocated_warehouse": warehouse,
            "trace_id": trace_id,
        },
    )
    if not isinstance(draft, dict) or not draft.get("success"):
        logger.error("[node:stock_check] 工单草稿创建失败：{}", draft)
        return {
            "stock_info": stock_info,
            "selected_warehouse": warehouse,
            "order_no": None,
        }

    order_no: str = str(draft.get("order_no"))
    logger.success(
        "[node:stock_check] 完成 | 工单 {} 已挂起待审批 | 耗时={:.1f}ms",
        order_no,
        (time.perf_counter() - started) * 1000,
    )
    return {
        "stock_info": stock_info,
        "selected_warehouse": warehouse,
        "order_no": order_no,
        "approval_status": "PENDING",
    }


def should_request_approval(state: AgentState) -> Literal["human_approval", "generate_report"]:
    """条件路由：只有真正生成了工单才需要主管审批。

    Args:
        state: 全局状态。

    Returns:
        str: 下一节点名。
    """
    if state.get("order_no"):
        logger.info("[route] order_no={} → human_approval", state.get("order_no"))
        return "human_approval"
    logger.info("[route] 未生成工单（缺货/无备件）→ generate_report")
    return "generate_report"


# ---------------------------------------------------------------------------
# 节点 4：人工审批（HITL 中断）
# ---------------------------------------------------------------------------
def _coerce_decision(payload: Any) -> tuple[str, str]:
    """把外部审批信号归一化为 ``(approval_status, approver)``。

    兼容两种写法：``"APPROVED"`` 字符串，或 ``{"status": "APPROVED", "approver": "张主管"}``。

    Args:
        payload: ``Command(resume=...)`` 传入的恢复值。

    Returns:
        tuple[str, str]: 审批状态与审批人（未知状态按驳回处理）。
    """
    if isinstance(payload, str):
        status: str = payload.strip().upper()
        approver: str = _APPROVER_UNKNOWN
    elif isinstance(payload, dict):
        status = str(payload.get("status") or payload.get("approval_status") or "").strip().upper()
        approver = str(
            payload.get("approver") or payload.get("operator") or _APPROVER_UNKNOWN
        ).strip()
    else:
        status, approver = "", _APPROVER_UNKNOWN

    if status not in _DECIDED_STATUSES:
        logger.warning("[node:human_approval] 非法的审批信号 {!r}，按 REJECTED 处理", payload)
        status = "REJECTED"
    return status, approver or _APPROVER_UNKNOWN


def _decided_from_state(state: AgentState) -> tuple[str, str] | None:
    """读取状态中**已写入**的审批结论。

    HITL 审批接口的标准做法是先 ``app.update_state(config, {...})`` 落审批结论，再
    ``app.invoke(None, config)`` 从断点续跑；此时本节点会被重新执行，若仍无脑
    ``interrupt()``，工作流会原地打转（LangGraph 只在收到 ``Command(resume=...)``
    时才把 resume 载荷回填给 ``interrupt()``）。因此这里先查状态，已决则直接放行。

    Args:
        state: 全局状态。

    Returns:
        tuple[str, str] | None: ``(approval_status, approver)``；未决时为 ``None``。
    """
    status: str = str(state.get("approval_status") or "").strip().upper()
    if status not in _DECIDED_STATUSES:
        return None
    approver: str = str(state.get("approver") or "").strip() or _APPROVER_UNKNOWN
    return status, approver


def human_approval_node(state: AgentState) -> dict[str, Any]:
    """HITL 中断节点：挂起工作流，等主管在审批终端给出结论。

    两种恢复方式都支持：

    1. **update_state + invoke(None)**（REST 审批接口使用）：先把结论写进状态，
       再让图从断点续跑，本节点检测到已决状态后直接放行；
    2. **Command(resume=...)**：``interrupt()`` 的返回值就是 resume 载荷。

    Args:
        state: 全局状态。

    Returns:
        dict: 增量状态（approval_status / approver）。
    """
    order_no: str | None = state.get("order_no")

    decided: tuple[str, str] | None = _decided_from_state(state)
    if decided is not None:
        status, approver = decided
        logger.info(
            "[node:human_approval] 状态已含审批结论，跳过中断直接续跑 | order_no={} "
            "status={} approver={}",
            order_no,
            status,
            approver,
        )
        return {"approval_status": status, "approver": approver}

    logger.warning(
        "[node:human_approval] 工作流挂起，等待主管审批 | order_no={} trace_id={}",
        order_no,
        state.get("trace_id"),
    )

    payload: dict[str, Any] = {
        "type": "approval_required",
        "order_no": order_no,
        "trace_id": state.get("trace_id"),
        "device_sn": state.get("device_sn"),
        "error_code": state.get("error_code"),
        "required_part": state.get("required_part"),
        "selected_warehouse": state.get("selected_warehouse"),
        "stock_info": state.get("stock_info"),
        "question": "是否批准该备件调拨并派工？请回复 APPROVED 或 REJECTED",
    }
    decision: Any = interrupt(payload)

    status, approver = _coerce_decision(decision)
    logger.success(
        "[node:human_approval] 审批完成 | order_no={} status={} approver={}",
        order_no,
        status,
        approver,
    )
    return {"approval_status": status, "approver": approver}


# ---------------------------------------------------------------------------
# 节点 5：工单派工与库存核销
# ---------------------------------------------------------------------------
_ORDER_STATUS_SQL: str = (
    "select order_no, approval_status, dispatch_status, allocated_warehouse, suggested_part "
    "from work_orders where order_no = :order_no for update"
)

_UPDATE_ORDER_SQL: str = (
    "update work_orders set approval_status = :approval_status, approver = :approver, "
    "dispatch_status = :dispatch_status, updated_at = now() where order_no = :order_no"
)

_DEDUCT_STOCK_SQL: str = (
    "update spare_parts set stock_quantity = stock_quantity - 1, updated_at = now() "
    "where part_name = :part_name and warehouse_name = :warehouse_name "
    "and stock_quantity > 0"
)


def dispatch_order_node(state: AgentState) -> dict[str, Any]:
    """按审批结论写入工单终态，并在通过时核销 1 件库存。

    幂等性：若工单已是 ``DISPATCHED`` 则直接跳过，避免重复恢复导致重复扣库存。

    Args:
        state: 全局状态（需含 order_no / approval_status / approver）。

    Returns:
        dict: 增量状态（approval_status / approver，供报告节点使用）。
    """
    started: float = time.perf_counter()
    order_no: str | None = state.get("order_no")
    status: str = (state.get("approval_status") or "REJECTED").upper()
    approver: str = state.get("approver") or _APPROVER_UNKNOWN
    logger.info(
        "[node:dispatch_order] 开始 | order_no={} status={} approver={}", order_no, status, approver
    )

    if not order_no:
        logger.error("[node:dispatch_order] 缺少 order_no，跳过派工")
        return {"approval_status": status, "approver": approver}

    dispatch_status: str = (
        _DISPATCH_STATUS_APPROVED if status == "APPROVED" else _DISPATCH_STATUS_REJECTED
    )

    try:
        with session_scope() as db:
            row = (
                db.execute(text(_ORDER_STATUS_SQL).bindparams(order_no=order_no))
                .mappings()
                .first()
            )
            if row is None:
                logger.error("[node:dispatch_order] 工单 {} 不存在，跳过派工", order_no)
                return {"approval_status": status, "approver": approver}

            if str(row.get("dispatch_status") or "") == _DISPATCH_STATUS_APPROVED:
                logger.warning(
                    "[node:dispatch_order] 工单 {} 已派工（幂等跳过，不重复扣库存）", order_no
                )
                return {
                    "approval_status": str(row.get("approval_status") or status),
                    "approver": str(row.get("approver") or approver),
                }

            db.execute(
                text(_UPDATE_ORDER_SQL).bindparams(
                    approval_status=status,
                    approver=approver,
                    dispatch_status=dispatch_status,
                    order_no=order_no,
                )
            )

            if status == "APPROVED":
                part_name: str = str(row.get("suggested_part") or "")
                warehouse: str = str(row.get("allocated_warehouse") or "")
                deducted = db.execute(
                    text(_DEDUCT_STOCK_SQL).bindparams(
                        part_name=part_name, warehouse_name=warehouse
                    )
                )
                if deducted.rowcount == 1:
                    logger.success(
                        "[node:dispatch_order] 库存核销成功 | {} @ {}-1",
                        part_name,
                        warehouse,
                    )
                else:
                    logger.error(
                        "[node:dispatch_order] 库存核销失败（库存已耗尽？）| {} @ {}",
                        part_name,
                        warehouse,
                    )
    except Exception as exc:  # noqa: BLE001
        logger.error("[node:dispatch_order] 数据库操作失败 order_no={} err={}", order_no, exc)
        return {"approval_status": status, "approver": approver}

    logger.success(
        "[node:dispatch_order] 完成 | order_no={} dispatch_status={} 耗时={:.1f}ms",
        order_no,
        dispatch_status,
        (time.perf_counter() - started) * 1000,
    )
    return {"approval_status": status, "approver": approver}


# ---------------------------------------------------------------------------
# 节点 6：报告生成
# ---------------------------------------------------------------------------
def _sop_summary(sop_guide: str | None, limit: int = 600) -> str:
    """截取 SOP 关键片段用于报告正文，避免报告过长。"""
    if not sop_guide:
        return "（无手册检索结果）"
    cleaned: str = sop_guide.strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "……（完整内容见 sop_guide 字段）"


def generate_report_node(state: AgentState) -> dict[str, Any]:
    """汇总全流程信息，生成对客的售后处理报告。

    报告是确定性拼接（不调用 LLM），既省 token 又保证同一输入输出稳定，
    便于回归测试与审计。

    Args:
        state: 全局状态（含之前所有节点产出的字段）。

    Returns:
        dict: ``{"final_response": 报告文本}``。
    """
    trace_id: str = state.get("trace_id") or "-"
    device_sn: str | None = state.get("device_sn")
    location: str | None = state.get("device_location")
    error_code: str | None = state.get("error_code")
    symptom: str | None = state.get("symptom")
    required_part: str | None = state.get("required_part")
    stock_info = state.get("stock_info") or []
    warehouse: str | None = state.get("selected_warehouse")
    order_no: str | None = state.get("order_no")
    approval_status: str | None = state.get("approval_status")
    approver: str | None = state.get("approver")

    lines: list[str] = [
        "# 机器人售后服务处理报告",
        "",
        f"- 追踪号：{trace_id}",
        f"- 接入通道：{state.get('channel') or 'human_chat'}",
        "",
        "## 一、设备信息",
        f"- 设备 SN：{device_sn or '未识别'}",
        f"- 所在城市：{location or '未知'}",
        "",
        "## 二、故障诊断",
        f"- 故障码：{error_code or '未识别'}",
        f"- 现象描述：{symptom or '未提供'}",
        "",
        "### 手册排障依据",
        _sop_summary(state.get("sop_guide")),
        "",
        "## 三、备件与库存",
    ]

    if not required_part:
        lines += [
            "- 结论：本故障无需更换备件，按上述 SOP 执行复位/调试即可。",
        ]
    else:
        lines.append(f"- 需更换备件：{required_part}")
        if stock_info:
            lines.append(f"- 推荐调拨仓库：{warehouse}（共 {len(stock_info)} 个仓库有货）")
            lines.append("")
            lines.append("| 仓库 | 城市 | 可调数量 | 货位 | 就近 |")
            lines.append("|---|---|---|---|---|")
            for row in stock_info:
                lines.append(
                    f"| {row.get('warehouse_name')} | {row.get('warehouse_location')} "
                    f"| {row.get('stock_quantity')} | {row.get('bin_location')} "
                    f"| {'是' if row.get('is_preferred_location') else '否'} |"
                )
        else:
            lines.append("- ⚠️ 全部仓库均无可用库存，需启动紧急采购/跨区调拨流程。")

    lines += ["", "## 四、工单状态"]
    if order_no:
        lines += [
            f"- 工单号：{order_no}",
            f"- 审批状态：{approval_status or 'PENDING'}",
            f"- 审批人：{approver or '待审批'}",
        ]
        if approval_status == "APPROVED":
            lines.append("- 派工结果：已派工（DISPATCHED），库存已核销 1 件。")
        elif approval_status == "REJECTED":
            lines.append("- 派工结果：审批驳回，工单关闭（CLOSED），库存未变动。")
        else:
            lines.append("- 派工结果：等待主管审批中。")
    else:
        lines.append("- 本次未生成工单。")

    lines += ["", "## 五、后续建议"]
    if required_part and stock_info and approval_status == "APPROVED":
        lines.append(
            f"1. 由库管员按货位 {stock_info[0].get('bin_location')} 完成出库与打包；"
            f"2. 现场工程师按 SOP 更换 {required_part} 后执行复位标定；"
            "3. 试运行 5 个循环确认故障消除。"
        )
    elif required_part and not stock_info:
        lines.append("1. 备件缺货，请同步采购与调度；2. 可先用同型号备件替换验证故障点。")
    elif not required_part:
        lines.append("1. 按 SOP 完成复位/调试；2. 若故障复现，补充故障码后重新发起诊断。")
    else:
        lines.append("1. 等待主管审批结果；2. 审批通过后由库管员按货位出库。")

    report: str = "\n".join(lines)
    logger.success("[node:generate_report] 报告已生成 | order_no={} 字数={}", order_no, len(report))
    return {"final_response": report}
