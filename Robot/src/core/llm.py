"""大模型与向量模型工厂（全部走云端 HTTP API，零本地模型）。

本模块只做两件事：

1. ``get_llm()``            —— 获取 DeepSeek 对话模型（``deepseek-chat``）；
2. ``get_embedding_model()`` —— 获取云端 Embedding 模型（默认 bge-m3，1024 维）。

两个工厂均带 ``lru_cache``，进程内只构造一次，避免重复建连。

关键说明：``OpenAIEmbeddings`` 默认会把文本先在**本地**用 tiktoken 切成 token 数组
再发送，很多国产 OpenAI 兼容端点并不支持这种入参，会直接返回 400。
因此这里固定设置 ``check_embedding_ctx_length=False``，让它发送原始字符串。
"""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from loguru import logger

from src.core.config import VECTOR_SIZE, is_usable_api_key, settings

__all__ = [
    "get_llm",
    "get_deepseek_llm",
    "get_embedding_model",
    "probe_embedding_dimension",
]


def _ensure_api_key(api_key: str, env_name: str) -> None:
    """校验 API Key 是否可用（非空 + 纯 ASCII）。

    为什么需要 ASCII 校验：Key 会被塞进 HTTP 请求头，而请求头按 ASCII 编码；
    若把 ``sk-请替换为你的Key`` 这类中文占位符直接写进 .env，openai SDK 会在
    本地抛出 ``UnicodeEncodeError``，报错信息与「Key 无效」毫无关系，极易误导排查。

    Args:
        api_key: 待校验的 Key。
        env_name: 对应的环境变量名，用于报错提示。

    Raises:
        RuntimeError: Key 为空或含非 ASCII 字符。
    """
    if not api_key.strip():
        logger.error("未配置 {}：请在 src/core/.env 中填写真实密钥后重试", env_name)
        raise RuntimeError(f"{env_name} 缺失，无法初始化模型客户端")

    if not is_usable_api_key(api_key):
        logger.error(
            "{} 仍是占位符或含非法字符（当前值前 12 位：{}…），请填入真实密钥",
            env_name,
            api_key[:12],
        )
        raise RuntimeError(f"{env_name} 不是可用的密钥（疑似占位符或含非 ASCII 字符）")


@lru_cache(maxsize=1)
def get_llm(
    temperature: float | None = None,
    model: str | None = None,
) -> ChatOpenAI:
    """获取 DeepSeek 聊天模型实例（OpenAI 兼容协议）。

    Args:
        temperature: 采样温度；``None`` 时使用 ``.env`` 中的 ``DEEPSEEK_TEMPERATURE``（默认 0）。
        model: 模型名；``None`` 时使用 ``.env`` 中的 ``DEEPSEEK_MODEL``（默认 ``deepseek-chat``）。

    Returns:
        ChatOpenAI: 已配置 base_url / api_key 的 LangChain 聊天模型对象。

    Raises:
        RuntimeError: 未配置 ``DEEPSEEK_API_KEY`` 或 Key 非法。
    """
    _ensure_api_key(settings.DEEPSEEK_API_KEY, "DEEPSEEK_API_KEY")

    llm = ChatOpenAI(
        model=model or settings.DEEPSEEK_MODEL,
        api_key=settings.DEEPSEEK_API_KEY,  # type: ignore[arg-type]  # 允许 str
        base_url=settings.DEEPSEEK_BASE_URL,
        temperature=(settings.DEEPSEEK_TEMPERATURE if temperature is None else temperature),
        timeout=settings.LLM_TIMEOUT,
        max_retries=settings.LLM_MAX_RETRIES,
        streaming=False,
    )
    logger.debug(
        "DeepSeek 模型已初始化：{} @ {}", settings.DEEPSEEK_MODEL, settings.DEEPSEEK_BASE_URL
    )
    return llm


#: 语义化别名：排障/库存节点读取更直观。
get_deepseek_llm = get_llm


@lru_cache(maxsize=1)
def get_embedding_model() -> OpenAIEmbeddings:
    """获取云端 Embedding 模型实例。

    完全通过 HTTP 调用远端推理服务，**不会**加载任何本地模型、**不会**下载权重。

    Returns:
        OpenAIEmbeddings: 已配置 base_url / api_key 的向量模型对象。

    Raises:
        RuntimeError: 未配置 ``EMBEDDING_API_KEY`` 或 Key 非法。
    """
    _ensure_api_key(settings.EMBEDDING_API_KEY, "EMBEDDING_API_KEY")

    extra_kwargs: dict[str, object] = {}
    # 仅当模型/服务商支持输出维度裁剪时才传 dimensions，
    # 否则部分端点（如 bge-m3）会因未知参数直接报 400。
    if settings.EMBEDDING_DIMENSIONS is not None:
        extra_kwargs["dimensions"] = settings.EMBEDDING_DIMENSIONS

    embeddings = OpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        api_key=settings.EMBEDDING_API_KEY,  # type: ignore[arg-type]  # 允许 str
        base_url=settings.EMBEDDING_BASE_URL,
        # 关键：直接发送原始文本，避免本地 tiktoken 分词数组导致兼容端点 400
        check_embedding_ctx_length=False,
        # 单批条数，部分服务商限制 10 条/次
        chunk_size=settings.EMBEDDING_BATCH_SIZE,
        **extra_kwargs,
    )
    logger.debug(
        "云端 Embedding 模型已初始化：{} @ {}",
        settings.EMBEDDING_MODEL,
        settings.EMBEDDING_BASE_URL,
    )
    return embeddings


def probe_embedding_dimension(expected: int = VECTOR_SIZE) -> int:
    """探活云端 Embedding 接口，并校验返回向量维度。

    在灌库脚本启动时调用：既能验证 Key / 网络是否可用，也能防止
    「模型维度与 Qdrant Collection 维度不一致」这种隐蔽错误。

    Args:
        expected: 期望维度，默认取全局常量 ``VECTOR_SIZE``（1024）。

    Returns:
        int: 服务端实际返回的向量维度。

    Raises:
        RuntimeError: 网络不可达、Key 无效、模型名/地址错误或维度不一致。
    """
    logger.info(
        "正在探活云端 Embedding：{} @ {}", settings.EMBEDDING_MODEL, settings.EMBEDDING_BASE_URL
    )
    try:
        vector: list[float] = get_embedding_model().embed_query("维度探活：机器人售后排障")
    except Exception as exc:  # noqa: BLE001 - 需要把底层异常翻译成人话
        detail: str = str(exc)
        hint: str = "请检查网络连通性与服务商状态"
        if "401" in detail or "403" in detail or "invalid_api_key" in detail:
            hint = "认证失败：EMBEDDING_API_KEY 无效或不匹配该服务商"
        elif "404" in detail or "not found" in detail.lower():
            hint = "接口不存在：请核对 EMBEDDING_BASE_URL 与 EMBEDDING_MODEL 是否匹配该服务商"
        elif "400" in detail:
            hint = (
                "请求被拒绝：常见原因是模型名为空/不支持，或服务端不接受 dimensions 参数"
                "（可在 .env 中注释掉 EMBEDDING_DIMENSIONS）"
            )
        elif "timeout" in detail.lower() or "timed out" in detail.lower():
            hint = "请求超时：请确认内网/代理是否放通该域名"
        logger.error(
            "云端 Embedding 探活失败：{} | 服务地址={} | {}", hint, settings.EMBEDDING_BASE_URL, detail
        )
        raise RuntimeError(f"Embedding 接口探活失败：{hint}") from exc

    dimension: int = len(vector)
    if dimension != expected:
        logger.error(
            "Embedding 维度不匹配：模型 {} 返回 {} 维，但 Qdrant Collection 期望 {} 维。"
            "请在 .env 中更换模型或调整 VECTOR_SIZE",
            settings.EMBEDDING_MODEL,
            dimension,
            expected,
        )
        raise RuntimeError(f"Embedding 维度不匹配：{dimension} != {expected}")
    logger.success("Embedding 探活通过：模型={} 维度={}", settings.EMBEDDING_MODEL, dimension)
    return dimension


if __name__ == "__main__":  # pragma: no cover - 手工自检入口
    probe_embedding_dimension()
