"""原厂维修手册 RAG 检索工具（Qdrant 本地文件模式）。

工具能力
--------
``search_repair_manual(query, error_code=None)``：把现场口语化故障描述转成 1024 维向量，
在本地 Qdrant 集合 ``robot_manual_sop`` 中检索 Top-K 排障 SOP 切片，输出结构化的文本
（故障码 / 排障步骤 / 建议更换备件）。

三级容错（保证没有 Embedding Key 时开发流程也不会卡死）
------------------------------------------------------
1. **带 error_code 的向量检索**：Payload ``error_code`` 精确过滤 + 语义相似度；
2. **全库语义检索 + 重排**：过滤未命中时放开过滤重新召回，并对故障码相同的切片加权；
3. **Payload 关键字兜底**：Embedding 接口不可用（无 Key / 网络不通 / 401）时，
   直接遍历本地集合，用「故障码精确匹配 + 中文 2-gram 关键词重合度」打分排序。

并发与锁
--------
Qdrant 本地模式是**单进程独占**的文件存储，且客户端并非线程安全，
因此这里用进程级 ``RLock`` 串行化所有读写操作，并用单例复用同一个客户端。
"""

from __future__ import annotations

import atexit
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Sequence

from langchain_core.tools import tool
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from src.core.config import MANUAL_COLLECTION_NAME, VECTOR_SIZE, settings
from src.core.llm import get_embedding_model
from src.core.logger import get_trace_id, logger

__all__ = [
    "search_repair_manual",
    "fetch_repair_hits",
    "format_hits",
    "hit_to_dict",
    "get_qdrant_client",
    "close_qdrant_client",
    "knowledge_base_health",
    "MANUAL_COLLECTION_NAME",
]

#: 默认召回条数（需求规定 Top-K = 2）
DEFAULT_TOP_K: int = 2
#: 兜底扫描时的最大点位数量，防止知识库变大后遍历过慢
FALLBACK_SCAN_LIMIT: int = 512

_client_lock: threading.RLock = threading.RLock()
_client: QdrantClient | None = None


# ---------------------------------------------------------------------------
# Qdrant 客户端单例
# ---------------------------------------------------------------------------
def get_qdrant_client() -> QdrantClient:
    """获取（并复用）本地 Qdrant 客户端。

    Returns:
        QdrantClient: 指向 ``./qdrant_local_data`` 的本地模式客户端。
    """
    global _client
    with _client_lock:
        if _client is None:
            path: str = str(settings.qdrant_local_abspath)
            _client = QdrantClient(path=path)
            logger.debug("[rag] 已打开本地 Qdrant：{}", path)
        return _client


def close_qdrant_client() -> None:
    """关闭并释放本地 Qdrant 客户端（进程退出前调用，及时释放目录锁）。"""
    global _client
    with _client_lock:
        if _client is not None:
            _client.close()
            _client = None
            logger.debug("[rag] 本地 Qdrant 客户端已关闭")


def _close_on_exit() -> None:
    """进程退出时兜底关闭客户端。

    Qdrant 本地客户端在 ``__del__`` 里会重新 ``import portalocker``，
    而解释器关闭阶段导入会失败并抛出 ``ImportError``（虽然只是噪音）。
    在 atexit 阶段（Python 仍健康）主动关闭可彻底避免该告警，
    同时确保文件锁及时释放。
    """
    try:
        close_qdrant_client()
    except Exception:  # noqa: BLE001 - 退出阶段不允许再抛异常
        pass


atexit.register(_close_on_exit)


def _collection_exists(client: QdrantClient) -> bool:
    """判断知识库集合是否已创建（兼容不同版本客户端 API）。"""
    try:
        return bool(client.collection_exists(collection_name=MANUAL_COLLECTION_NAME))
    except Exception:  # noqa: BLE001 - 老版本没有该方法
        try:
            client.get_collection(collection_name=MANUAL_COLLECTION_NAME)
            return True
        except Exception:  # noqa: BLE001
            return False


def knowledge_base_health() -> tuple[bool, int]:
    """探测本地知识库是否可用（供 ``GET /health`` 使用）。

    与检索路径共用同一个客户端单例和进程级锁：Qdrant 本地模式的文件存储并非
    线程安全，健康检查若绕过锁并发读，可能和检索请求互相踩踏。

    Returns:
        tuple[bool, int]: ``(集合是否存在, 点位数量)``；集合缺失或读取失败时数量为 0。
    """
    client: QdrantClient = get_qdrant_client()
    with _client_lock:
        if not _collection_exists(client):
            return False, 0
        try:
            count = client.count(collection_name=MANUAL_COLLECTION_NAME, exact=True).count
        except Exception as exc:  # noqa: BLE001 - 健康检查不允许抛异常
            logger.warning("[rag] 知识库点位统计失败：{}", exc)
            return True, 0
    return True, int(count or 0)


# ---------------------------------------------------------------------------
# 检索结果的内部表示
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class _Hit:
    """一条命中的手册切片。"""

    point_id: int | str
    score: float
    payload: dict[str, Any]

    @property
    def error_code(self) -> str:
        return str(self.payload.get("error_code", "-"))

    @property
    def content(self) -> str:
        """排障步骤原文。优先取 ``content``，兼容阶段一写入的 ``chunk_text``。"""
        return str(self.payload.get("content") or self.payload.get("chunk_text") or "")

    @property
    def required_part(self) -> str:
        """建议更换的备件标准名称（展示用，无换件需求时给出文案）。"""
        return self.required_part_value or "无（仅需复位/调试）"

    @property
    def required_part_value(self) -> str | None:
        """建议更换的备件名（结构化场景用）：无需换件时返回 ``None``。

        状态机据此判断是否进入库存核验分支，因此不能返回「无…」这类展示文案。
        """
        raw: Any = self.payload.get("required_part") or self.payload.get("target_part")
        if not raw:
            return None
        cleaned: str = str(raw).strip()
        return cleaned or None

    @property
    def section_title(self) -> str:
        return str(self.payload.get("section_title") or "未标注章节")

    @property
    def embedding_source(self) -> str:
        """向量来源标记：``pseudo`` 表示离线兜底生成的伪向量。"""
        return str(self.payload.get("embedding_source") or "cloud")


# ---------------------------------------------------------------------------
# 文本工具
# ---------------------------------------------------------------------------
def _normalize_error_code(error_code: str | None) -> str | None:
    """规范化故障码：去空格并转大写（库内统一为 ``E-201`` 形式）。"""
    if not error_code:
        return None
    cleaned: str = error_code.strip().upper()
    return cleaned or None


def _tokenize(text: str) -> set[str]:
    """把文本切成检索词集合：英文/数字整词 + 中文 2-gram。

    中文没有空格分词，直接按整句匹配几乎命中不了（例如用户说「夹爪不动」，
    手册写「夹爪无开合动作」），因此拆成字符级 2-gram 求重合度。

    Args:
        text: 原始文本。

    Returns:
        set[str]: 检索词集合。
    """
    lowered: str = text.lower()
    tokens: set[str] = set(re.findall(r"[a-z0-9][a-z0-9\-_]*", lowered))
    for chunk in re.findall(r"[\u4e00-\u9fff]+", text):
        if len(chunk) == 1:
            tokens.add(chunk)
        else:
            tokens.update(chunk[i : i + 2] for i in range(len(chunk) - 1))
    return tokens


# ---------------------------------------------------------------------------
# 检索实现
# ---------------------------------------------------------------------------
def _run_query(
    client: QdrantClient,
    vector: Sequence[float],
    limit: int,
    query_filter: qm.Filter | None,
) -> list[_Hit]:
    """执行一次向量检索（兼容新旧两套客户端 API）。"""
    if hasattr(client, "query_points"):
        response = client.query_points(
            collection_name=MANUAL_COLLECTION_NAME,
            query=list(vector),
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        raw_points = response.points
    else:  # pragma: no cover - 兼容 1.9 及更早期版本
        raw_points = client.search(
            collection_name=MANUAL_COLLECTION_NAME,
            query_vector=list(vector),
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
    return [
        _Hit(point_id=p.id, score=float(p.score), payload=dict(p.payload or {}))
        for p in raw_points
    ]


def _vector_search(
    client: QdrantClient,
    query: str,
    error_code: str | None,
    top_k: int,
) -> list[_Hit]:
    """向量检索主路径：优先按故障码过滤，未命中则全库召回并重排。

    Args:
        client: 本地 Qdrant 客户端。
        query: 用户查询文本。
        error_code: 已规范化的故障码（可为 ``None``）。
        top_k: 召回条数。

    Returns:
        list[_Hit]: 命中的手册切片（按相关度降序）。

    Raises:
        RuntimeError: Embedding 接口不可用或维度不符（由上层退化为关键字兜底）。
    """
    # 未配置（或仍是占位符）时直接抛错，交由上层走关键字兜底，避免白发一次注定失败的请求
    if not settings.has_embedding_key:
        raise RuntimeError("EMBEDDING_API_KEY 未配置或仍是占位符，跳过向量检索")

    text_for_embedding: str = query or (error_code or "")
    vector: list[float] = get_embedding_model().embed_query(text_for_embedding)
    if len(vector) != VECTOR_SIZE:
        raise RuntimeError(
            f"Embedding 维度 {len(vector)} 与集合维度 {VECTOR_SIZE} 不一致，无法检索"
        )

    if error_code:
        # 1) 故障码硬过滤：只在该故障码相关的切片里做语义排序
        code_filter = qm.Filter(
            must=[qm.FieldCondition(key="error_code", match=qm.MatchValue(value=error_code))]
        )
        filtered: list[_Hit] = _run_query(client, vector, top_k, code_filter)
        if filtered:
            logger.debug("[rag] 故障码过滤检索命中 {} 条（error_code={}）", len(filtered), error_code)
            return filtered
        logger.info("[rag] 故障码 {} 精确过滤无命中，放开过滤走全库语义检索", error_code)

    # 2) 全库语义召回 + 故障码重排加权
    candidates: list[_Hit] = _run_query(client, vector, max(top_k * 3, 6), None)
    if error_code:
        for hit in candidates:
            if hit.error_code.upper() == error_code:
                # 微小加权即可改变邻近同分结果的顺序，不掩盖真实语义分数差异
                hit.score = min(1.0, hit.score + 0.05)
    candidates.sort(key=lambda h: h.score, reverse=True)
    return candidates[:top_k]


def _keyword_fallback(
    client: QdrantClient,
    query: str,
    error_code: str | None,
    top_k: int,
) -> list[_Hit]:
    """无 Embedding 能力时的兜底检索：遍历 Payload 做故障码与关键词打分。

    Args:
        client: 本地 Qdrant 客户端。
        query: 用户查询文本。
        error_code: 已规范化的故障码（可为 ``None``）。
        top_k: 返回条数。

    Returns:
        list[_Hit]: 按打分降序的切片；``score`` 为 0~1 的归一化相关度。
    """
    points, _ = client.scroll(
        collection_name=MANUAL_COLLECTION_NAME,
        limit=FALLBACK_SCAN_LIMIT,
        with_payload=True,
        with_vectors=False,
    )

    query_tokens: set[str] = _tokenize(query)
    scored: list[_Hit] = []
    for point in points:
        payload: dict[str, Any] = dict(point.payload or {})
        hit = _Hit(point_id=point.id, score=0.0, payload=payload)

        score: float = 0.0
        hit_code: str = hit.error_code.upper()
        if error_code and hit_code == error_code:
            # 故障码是强信号，一旦精确命中直接置顶
            score += 10.0
        elif error_code and error_code in hit_code:
            score += 5.0

        doc_tokens: set[str] = _tokenize(f"{hit.section_title} {hit.content}")
        overlap: int = len(query_tokens & doc_tokens)
        if query_tokens:
            score += 5.0 * overlap / len(query_tokens)

        if score <= 0:
            continue
        # 归一化到 0~1，语义与向量分数含义接近，便于上层统一排序
        hit.score = round(min(score / 15.0, 1.0), 4)
        scored.append(hit)

    scored.sort(key=lambda h: h.score, reverse=True)
    logger.debug(
        "[rag] 关键字兜底扫描 {} 条点位，命中 {} 条（query_tokens={}）",
        len(points),
        len(scored),
        len(query_tokens),
    )
    return scored[:top_k]


# ---------------------------------------------------------------------------
# 结果格式化（对外统一使用 dict，便于状态机直接消费）
# ---------------------------------------------------------------------------
def hit_to_dict(hit: _Hit) -> dict[str, Any]:
    """把内部命中对象转换为普通字典（可 JSON 序列化，可直接塞进 AgentState）。

    Args:
        hit: 内部 ``_Hit`` 实例。

    Returns:
        dict: 含 error_code / section_title / content / required_part / score 等字段。
    """
    return {
        "point_id": hit.point_id,
        "score": round(hit.score, 4),
        "error_code": hit.error_code,
        "section_title": hit.section_title,
        "content": hit.content,
        "required_part": hit.required_part_value,
        "required_part_display": hit.required_part,
        "embedding_source": hit.embedding_source,
    }


def format_hits(
    hits: list[dict[str, Any]],
    query: str,
    error_code: str | None,
    mode: str,
    elapsed_ms: float,
) -> str:
    """把结构化命中列表渲染成便于大模型阅读的纯文本。

    Args:
        hits: ``hit_to_dict`` 产出的字典列表。
        query: 原始查询文本。
        error_code: 规范化后的故障码。
        mode: 检索模式描述。
        elapsed_ms: 检索耗时（毫秒）。

    Returns:
        str: 供 LLM / 报告直接使用的文本。
    """
    header: str = (
        f"【原厂维修手册检索结果】\n"
        f"- 查询内容：{query or '（空）'}\n"
        f"- 故障码过滤：{error_code or '无'}\n"
        f"- 检索模式：{mode}\n"
        f"- 召回条数：{len(hits)}（Top-K）\n"
        f"- 耗时：{elapsed_ms:.1f} ms\n"
    )
    if not hits:
        return (
            header
            + "\n未检索到匹配的手册切片。建议：补充更具体的故障码（如 E-201），"
            "或用「部位 + 现象」描述，例如「夹爪不动 气缸漏气」。"
        )

    blocks: list[str] = []
    for index, hit in enumerate(hits, start=1):
        pseudo_note: str = (
            "\n- 注意：该向量为离线兜底伪向量，语义分数仅供参考"
            if hit.get("embedding_source") == "pseudo"
            else ""
        )
        blocks.append(
            f"\n[命中 {index}] 故障码：{hit.get('error_code', '-')} | 相关度：{hit.get('score', 0):.4f}\n"
            f"- 章节：{hit.get('section_title', '未标注章节')}\n"
            f"- 建议更换备件(required_part)：{hit.get('required_part_display', '无（仅需复位/调试）')}\n"
            f"- 排障步骤(content)：{hit.get('content', '')}{pseudo_note}"
        )
    return header + "\n" + "\n".join(blocks)


def fetch_repair_hits(
    query: str,
    error_code: str | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> tuple[list[dict[str, Any]], str]:
    """执行一次手册检索，返回「结构化命中 + 检索模式」。

    这是 ``search_repair_manual`` 工具与 LangGraph 节点共用的核心入口：
    节点需要 ``required_part`` 这类结构化字段，直接读字典比正则解析自己生成的
    文本可靠得多，也不会为同一次检索多打一次 embedding 请求。

    Args:
        query: 故障现象描述。
        error_code: 故障码（可空）。
        top_k: 召回条数。

    Returns:
        tuple[list[dict], str]: ``(命中列表, 检索模式)``；知识库未创建时抛出 RuntimeError。

    Raises:
        RuntimeError: 本地集合尚未创建。
    """
    normalized_code: str | None = _normalize_error_code(error_code)
    client: QdrantClient = get_qdrant_client()

    if not _collection_exists(client):
        raise RuntimeError(
            f"知识库集合 {MANUAL_COLLECTION_NAME} 尚未创建，请先执行 "
            "python scripts/init_qdrant_data.py --offline（或配置 Key 后 --recreate）"
        )

    hits: list[_Hit]
    mode: str
    try:
        with _client_lock:
            hits = _vector_search(client, query, normalized_code, top_k)
        mode = "向量检索（云端 Embedding）"
    except Exception as exc:  # noqa: BLE001 - 任何向量链路故障都必须降级而不是抛出
        logger.warning("[rag] 向量检索不可用，退化为 Payload 关键字/故障码匹配：{}", exc)
        with _client_lock:
            hits = _keyword_fallback(client, query, normalized_code, top_k)
        mode = "关键字兜底（Embedding 不可用）"

    return [hit_to_dict(hit) for hit in hits], mode


# ---------------------------------------------------------------------------
# LangChain 工具入口
# ---------------------------------------------------------------------------
@tool
def search_repair_manual(query: str, error_code: str | None = None) -> str:
    """检索原厂维修手册 SOP 切片，返回排障步骤与所需备件。

    适用于「某故障怎么修、要换什么件」类问题。若已知错误码（如 E-201）请一并传入，
    可显著提升召回准确率；无错误码时系统会退化为语义检索 / 关键字匹配。

    Args:
        query: 现场故障的口语化描述，例如「夹爪不动 气缸漏气」。
        error_code: 故障错误码，例如 ``E-201``；未知时留空。

    Returns:
        str: 结构化的检索结果文本，包含故障码、章节、排障步骤与建议更换备件。
    """
    started: float = time.perf_counter()
    trace_id: str = get_trace_id()
    logger.info(
        "[rag] search_repair_manual 入参 query={!r} error_code={!r} trace_id={}",
        query,
        error_code,
        trace_id,
    )

    try:
        hits, mode = fetch_repair_hits(query, error_code)
    except RuntimeError as exc:
        logger.error("[rag] 检索失败：{}", exc)
        return (
            f"【原厂维修手册检索结果】\n{exc}"
        )

    elapsed_ms: float = (time.perf_counter() - started) * 1000
    result: str = format_hits(hits, query, _normalize_error_code(error_code), mode, elapsed_ms)
    logger.info(
        "[rag] 检索完成 | trace_id={} | 模式={} | 命中={} | 耗时={:.1f}ms | Top1={}",
        trace_id,
        mode,
        len(hits),
        elapsed_ms,
        f"{hits[0].get('error_code')}/{hits[0].get('required_part_display')}" if hits else "无",
    )
    return result
