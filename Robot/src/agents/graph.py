"""LangGraph 状态图编译与检查点持久化（阶段三：排障与备件调度状态机）。

图结构
------
::

    START
      │
      ▼
    parse_fault          故障解析 + 设备定位（DeepSeek / 正则兜底）
      │
      ▼
    rag_diagnose         Qdrant 手册检索，产出 SOP 与所需备件
      │
      ├── required_part 为空（如 E-500 急停复位）──────────────┐
      ▼                                                       │
    stock_check          多仓就近查库存 + 生成 PENDING 工单     │
      │                                                       │
      ├── 未生成工单（全库缺货）────────────────────────────────┤
      ▼                                                       │
    human_approval       interrupt() 挂起，等待人工审批          │
      │                                                       │
      ▼                                                       │
    dispatch_order       写审批结果 + 核销库存 + 置派工          │
      │                                                       │
      ▼                                                       ▼
    generate_report ─────────────────────────────────────────┘
      │
      ▼
     END

人在回路（HITL）
----------------
``human_approval`` 节点内部调用 ``interrupt()`` 抛出自定义中断信号，LangGraph 会把
当时的完整状态写入检查点并暂停。外部世界通过同一个 ``thread_id``（即 ``trace_id``）
用 ``Command(resume=...)`` 恢复：

.. code-block:: python

    app.invoke(initial_state, make_config(trace_id))          # 跑到审批处挂起
    app.invoke(Command(resume={"status": "APPROVED", "approver": "张主管"}),
               make_config(trace_id))                          # 恢复并跑完

不使用 ``compile(interrupt_before=["dispatch_order"])`` 的原因：静态断点只能"停在
派工前"，审批结论仍需外部再写回状态；``interrupt()`` 能在挂起时把待审批工单信息
（工单号 / 备件 / 仓库 / 库存）作为 payload 直接推给审批终端，恢复值也天然是审批结论。

检查点
------
优先使用 ``langgraph-checkpoint-sqlite`` 的 ``SqliteSaver``（落盘到项目根的
``checkpoints.sqlite``，进程重启后仍能恢复挂起的工单）；若该包缺失，自动回退到
内置 ``MemorySaver``（仅进程内有效）并在日志中告警。
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.agents.nodes import (
    dispatch_order_node,
    generate_report_node,
    human_approval_node,
    parse_fault_node,
    rag_diagnose_node,
    should_check_stock,
    should_request_approval,
    stock_check_node,
)
from src.agents.state import AgentState
from src.core.config import settings
from src.core.logger import logger

__all__ = [
    "build_graph",
    "get_graph",
    "get_checkpointer",
    "make_config",
    "NODE_NAMES",
]

#: 图中全部节点名（顺序与业务流一致，供测试与前端画布使用）
NODE_NAMES: tuple[str, ...] = (
    "parse_fault",
    "rag_diagnose",
    "stock_check",
    "human_approval",
    "dispatch_order",
    "generate_report",
)

_checkpointer: Optional[BaseCheckpointSaver] = None
_compiled: Any = None


def _create_sqlite_checkpointer() -> BaseCheckpointSaver:
    """构造落盘检查点后端；缺少依赖时回退内存。

    Returns:
        BaseCheckpointSaver: ``SqliteSaver``（正常）或 ``MemorySaver``（依赖缺失）。
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as exc:  # pragma: no cover - 依赖缺失的降级路径
        logger.warning(
            "未安装 langgraph-checkpoint-sqlite（{}），回退到 MemorySaver："
            "进程重启后挂起的工单将无法恢复，生产环境请 pip install langgraph-checkpoint-sqlite",
            exc,
        )
        return MemorySaver()

    db_path: Path = settings.checkpoint_db_abspath
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # check_same_thread=False：FastAPI 的多线程 worker 会复用同一连接
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    saver = SqliteSaver(conn)
    logger.info("检查点后端：SqliteSaver → {}", db_path)
    return saver


def get_checkpointer(force_memory: bool = False) -> BaseCheckpointSaver:
    """获取检查点持久化后端（进程内单例）。

    Args:
        force_memory: 为 ``True`` 时**临时**返回一个全新的 ``MemorySaver``
            （测试用），不会污染进程级单例，后续调用仍返回落盘后端。

    Returns:
        BaseCheckpointSaver: ``SqliteSaver`` 或 ``MemorySaver`` 实例。
    """
    global _checkpointer
    if force_memory:
        logger.warning("检查点后端：MemorySaver（本次强制内存模式，进程重启后挂起状态会丢失）")
        return MemorySaver()

    if _checkpointer is None:
        _checkpointer = _create_sqlite_checkpointer()
    return _checkpointer


def build_graph(checkpointer: Optional[BaseCheckpointSaver] = None) -> Any:
    """编译排障与备件调度状态图。

    Args:
        checkpointer: 检查点后端；``None`` 时使用 :func:`get_checkpointer` 的进程单例。
            传入**新建**的 ``SqliteSaver`` 即可验证"状态确实落在磁盘上"（跨进程恢复）。

    Returns:
        CompiledStateGraph: 可直接 ``invoke(state)`` / ``invoke(Command(resume=...))`` 的图。
    """
    builder: StateGraph = StateGraph(AgentState)

    # ---- 注册节点 ----
    builder.add_node("parse_fault", parse_fault_node)
    builder.add_node("rag_diagnose", rag_diagnose_node)
    builder.add_node("stock_check", stock_check_node)
    builder.add_node("human_approval", human_approval_node)
    builder.add_node("dispatch_order", dispatch_order_node)
    builder.add_node("generate_report", generate_report_node)

    # ---- 主干边 ----
    builder.add_edge(START, "parse_fault")
    builder.add_edge("parse_fault", "rag_diagnose")

    # ---- 条件边 1：是否需要换件（E-500 急停复位这类无需备件，直接出报告）----
    builder.add_conditional_edges(
        "rag_diagnose",
        should_check_stock,
        {"stock_check": "stock_check", "generate_report": "generate_report"},
    )

    # ---- 条件边 2：是否真的生成了工单（全库缺货时没有可审批对象，直接出报告）----
    builder.add_conditional_edges(
        "stock_check",
        should_request_approval,
        {"human_approval": "human_approval", "generate_report": "generate_report"},
    )

    # ---- 审批 → 派工 → 报告 ----
    builder.add_edge("human_approval", "dispatch_order")
    builder.add_edge("dispatch_order", "generate_report")
    builder.add_edge("generate_report", END)

    app: Any = builder.compile(checkpointer=checkpointer or get_checkpointer())
    logger.info("状态图编译完成：{}", " → ".join(NODE_NAMES))
    return app


def get_graph() -> Any:
    """获取进程内单例图（FastAPI 复用；测试请直接调 :func:`build_graph`）。"""
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled


def make_config(thread_id: str) -> dict[str, Any]:
    """构造 LangGraph 运行配置。

    ``thread_id`` 同时充当业务上的 ``trace_id``，因此挂起与恢复用的是同一个线程号。

    Args:
        thread_id: 线程/追踪 ID。

    Returns:
        dict: 形如 ``{"configurable": {"thread_id": ...}}`` 的配置。
    """
    return {"configurable": {"thread_id": thread_id}}
