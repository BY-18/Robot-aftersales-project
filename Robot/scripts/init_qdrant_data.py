"""Qdrant 本地知识库初始化脚本（第一阶段 RAG 打桩）。

用途
----
在**纯本地文件模式**下（``QdrantClient(path=...)``，不启动 Qdrant 服务端、不用 Docker）
创建 ``robot_manual_sop`` 集合，并写入 3 条模拟的「原厂维修手册 SOP」切片：

* 数据 1：故障码 ``E-201`` → 建议更换「型号A电磁阀」
* 数据 2：故障码 ``E-104`` → 建议更换「400W交流伺服电机」
* 数据 3：故障码 ``E-500`` → 安全回路急停，复位即可，**无需更换备件**
  （对应状态机 ``should_check_stock`` 的「无需换件 → 直接出报告」分支）

向量全部由**云端 Embedding API** 生成（见 ``src/core/llm.py``），
脚本内不存在任何本地模型加载逻辑，不会触发 PyTorch / transformers 下载。

运行
----
在项目根目录执行::

    # 首次创建（集合已存在则直接追加/覆盖同 ID 记录）
    python scripts/init_qdrant_data.py

    # 强制重建集合（改了向量维度或想清空重灌时使用）
    python scripts/init_qdrant_data.py --recreate

    # 灌库后顺便跑一次相似度自检
    python scripts/init_qdrant_data.py --recreate --query "夹爪不动 气缸漏气 E-201"

    # 尚未配置 EMBEDDING_API_KEY 时，用离线伪向量打通链路（无真实语义能力）
    python scripts/init_qdrant_data.py --offline --recreate

注意
----
Qdrant 本地模式会对数据目录加**进程级独占锁**：同一时刻只能有一个进程打开
``./qdrant_local_data``。若 FastAPI 服务正在运行并持有该目录，本脚本会报
"Storage folder is already accessed"，此时请先停掉服务再执行。
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 保证脚本在任意工作目录下执行时都能 import 到 src 包
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.http import models as qdrant_models  # noqa: E402

from src.core.config import (  # noqa: E402
    MANUAL_COLLECTION_NAME,
    VECTOR_SIZE,
    settings,
)
from src.core.llm import get_embedding_model, probe_embedding_dimension  # noqa: E402
from src.core.logger import logger  # noqa: E402

# ---------------------------------------------------------------------------
# 模拟知识库数据：真实项目中这里会由 PDF/Markdown 手册分块（Chunking）产生
# ---------------------------------------------------------------------------
MOCK_SOP_DOCUMENTS: list[dict[str, Any]] = [
    {
        "doc_id": "manual-v1-e201",
        "error_code": "E-201",
        "model_code": "ARM-X6",
        "section_title": "第 4 章 气动夹爪系统 / 4.3 E-201 夹爪无动作",
        "target_part": "型号A电磁阀",
        "fault_type": "replace_part",
        "chunk_text": (
            "故障现象：机械臂上电后夹爪无开合动作，示教器屏幕报错 E-201，"
            "贴近气缸可听到持续漏气声。"
            "排查步骤：1) 检查气源压力是否低于 0.45 MPa；2) 断开气路，用肥皂水"
            "涂抹电磁阀进出口接头，确认阀体内部密封圈老化导致内漏；"
            "3) 万用表测量电磁阀线圈阻值，正常应为 24~28 Ω，若开路则线圈烧毁。"
            "处置结论：判定为电磁阀密封失效，需更换备件「型号A电磁阀」，"
            "更换后执行夹爪零点标定并试运行 5 个循环。"
        ),
    },
    {
        "doc_id": "manual-v1-e104",
        "error_code": "E-104",
        "model_code": "AGV-T3",
        "section_title": "第 6 章 行走驱动系统 / 6.2 E-104 伺服过载停机",
        "target_part": "400W交流伺服电机",
        "fault_type": "replace_part",
        "chunk_text": (
            "故障现象：AGV 小车行走中突然停机，红色故障灯长亮，日志报 E-104 伺服过载。"
            "排查步骤：1) 手动推动车体确认无机械卡滞；2) 读取驱动器报警码，"
            "若为过流且伴随电机外壳温度超过 80 ℃，判定电机绕组匝间短路；"
            "3) 测量三相绕组阻值偏差是否超过 5%，偏差过大即确认电机损坏。"
            "处置结论：需更换备件「400W交流伺服电机」，更换后重新执行驱动器参数"
            "自学习与行走里程标定。"
        ),
    },
    {
        "doc_id": "manual-v1-e500",
        "error_code": "E-500",
        "model_code": "L-500",
        "section_title": "第 9 章 安全回路 / 9.1 E-500 急停触发停机",
        # target_part 显式为 None：本故障无需换件，状态机据此走「直接出报告」分支
        "target_part": None,
        "fault_type": "reset_only",
        "chunk_text": (
            "故障现象：设备急停按钮被拍下或安全回路被断开后整车停机，"
            "控制面板报错 E-500，驱动使能丢失，无法下发运动指令。"
            "排查步骤：1) 确认设备运动范围内无人员，检查安全门、安全光栅、防撞条是否复位；"
            "2) 顺时针旋出急停按钮，确认按钮指示灯熄灭；"
            "3) 在控制面板执行「复位 / 清除报警」，观察安全回路指示灯是否转绿；"
            "4) 重新使能驱动并低速点动，确认行走与夹爪动作恢复正常。"
            "处置结论：E-500 属安全回路保护动作，复位并清除报警即可恢复，"
            "无需更换任何备件；若复位后 5 分钟内再次触发 E-500，"
            "需检查安全回路接线与光栅对射是否被遮挡。"
        ),
    },
]


def resolve_qdrant_path() -> Path:
    """把配置中的相对路径 ``./qdrant_local_data`` 解析为绝对路径。

    Returns:
        Path: 数据目录绝对路径（不保证已存在，Qdrant 会自动创建）。
    """
    # 统一由 config 锚定到项目根目录，避免受进程工作目录影响
    return settings.qdrant_local_abspath


def collection_state(
    client: QdrantClient, collection: str
) -> tuple[bool, int | None]:
    """探测集合是否存在，并尽力读取其向量维度。

    兼容差异：不同版本的 qdrant-client 中 ``info.config.params.vectors`` 既可能是
    单个 ``VectorParams``，也可能是「命名向量」的 ``dict``；集合不存在时
    ``get_collection`` 会直接抛异常。这里统一收敛为 ``(exists, size)`` 二元组。

    Args:
        client: 已打开的本地 Qdrant 客户端。
        collection: 集合名称。

    Returns:
        tuple[bool, int | None]: ``(是否存在, 向量维度)``；维度读不到时为 ``None``。
    """
    try:
        info = client.get_collection(collection_name=collection)
    except Exception:  # noqa: BLE001 - 集合不存在时各版本抛出的异常类型不统一
        return False, None

    vectors_cfg = info.config.params.vectors
    if isinstance(vectors_cfg, dict):
        # 命名向量场景：取第一个向量的维度用于校验
        first = next(iter(vectors_cfg.values()), None)
        return True, getattr(first, "size", None)
    return True, getattr(vectors_cfg, "size", None)


def ensure_collection(client: QdrantClient, recreate: bool = False) -> None:
    """创建 ``robot_manual_sop`` 集合，维度 1024、距离度量 Cosine。

    Args:
        client: 已打开的本地 Qdrant 客户端。
        recreate: 为 ``True`` 时先删除同名集合再重建（用于清空重灌）。

    Raises:
        RuntimeError: 集合已存在但向量维度与 ``VECTOR_SIZE`` 不一致。
    """
    collection: str = MANUAL_COLLECTION_NAME
    exists, current_size = collection_state(client, collection)

    if exists and recreate:
        logger.warning("集合 {} 已存在，按 --recreate 参数执行删除重建", collection)
        client.delete_collection(collection_name=collection)
        exists, current_size = False, None

    if exists:
        if current_size is not None and current_size != VECTOR_SIZE:
            logger.error(
                "集合 {} 已存在但维度为 {}，与期望的 {} 不一致；"
                "请执行 python scripts/init_qdrant_data.py --recreate",
                collection,
                current_size,
                VECTOR_SIZE,
            )
            raise RuntimeError(f"集合 {collection} 向量维度不匹配")
        logger.info("集合 {} 已存在（维度 {}），跳过创建", collection, current_size)
        return

    client.create_collection(
        collection_name=collection,
        vectors_config=qdrant_models.VectorParams(
            size=VECTOR_SIZE,
            distance=qdrant_models.Distance.COSINE,
        ),
    )
    logger.success(
        "已创建集合 {}：size={} distance=COSINE", collection, VECTOR_SIZE
    )


def build_points(vectors: list[list[float]], source: str = "cloud") -> list[qdrant_models.PointStruct]:
    """把模拟文档与对应向量组装为 Qdrant 点位。

    Args:
        vectors: 云端 Embedding 返回的向量列表，顺序需与 ``MOCK_SOP_DOCUMENTS`` 一致。
        source: 向量来源标记，``cloud`` 表示真实云端 Embedding，``pseudo`` 表示离线伪向量。

    Returns:
        list[PointStruct]: 可直接 upsert 的点位列表。

    Raises:
        ValueError: 向量条数与文档条数不一致。
    """
    if len(vectors) != len(MOCK_SOP_DOCUMENTS):
        raise ValueError(
            f"向量条数({len(vectors)}) 与文档条数({len(MOCK_SOP_DOCUMENTS)}) 不一致"
        )

    points: list[qdrant_models.PointStruct] = []
    for index, (doc, vector) in enumerate(zip(MOCK_SOP_DOCUMENTS, vectors), start=1):
        payload: dict[str, Any] = {
            "doc_id": doc["doc_id"],
            "error_code": doc["error_code"],
            "model_code": doc["model_code"],
            "section_title": doc["section_title"],
            "target_part": doc["target_part"],
            "fault_type": doc["fault_type"],
            "chunk_text": doc["chunk_text"],
            "embedding_source": source,
        }
        points.append(
            qdrant_models.PointStruct(
                id=index,  # 固定整数 ID，便于重复执行时幂等覆盖
                vector=vector,
                payload=payload,
            )
        )
    return points


class _OfflineHashEmbeddings:
    """离线兜底「伪 Embedding」：基于文本哈希的确定性向量。

    仅用于**没有配置 EMBEDDING_API_KEY 时打通链路**：它不含任何语义能力，
    检索只能依赖 ``error_code`` 精确过滤或关键字匹配（RAG 工具自带该兜底）。
    配置真实 Key 后请重新执行 ``python scripts/init_qdrant_data.py --recreate`` 覆盖。
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量生成伪向量。"""
        return [_pseudo_vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """单条生成伪向量。"""
        return _pseudo_vector(text)


def _pseudo_vector(text: str, size: int = VECTOR_SIZE) -> list[float]:
    """由文本哈希派生一个确定性的单位向量（同一文本永远得到同一向量）。"""
    seed: int = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
    rng: random.Random = random.Random(seed)
    raw: list[float] = [rng.uniform(-1.0, 1.0) for _ in range(size)]
    norm: float = math.sqrt(sum(value * value for value in raw)) or 1.0
    return [value / norm for value in raw]


def ingest(client: QdrantClient, embedder: Any, source: str = "cloud") -> int:
    """向量化并写入模拟 SOP 数据。

    Args:
        client: 已打开的本地 Qdrant 客户端。
        embedder: 具备 ``embed_documents`` 的对象（云端 Embedding 或离线伪向量器）。
        source: 向量来源标记，写入 Payload 便于区分真假向量。

    Returns:
        int: 集合内累计点位数量。
    """
    logger.info("开始向量化 {} 条模拟手册切片……", len(MOCK_SOP_DOCUMENTS))
    texts: list[str] = [doc["chunk_text"] for doc in MOCK_SOP_DOCUMENTS]

    # 云端模式下是一次 HTTP 请求，离线模式下是纯本地哈希计算，均不加载本地模型
    vectors: list[list[float]] = embedder.embed_documents(texts)
    logger.success("向量化完成，共 {} 条，单条维度 {}", len(vectors), len(vectors[0]))

    points: list[qdrant_models.PointStruct] = build_points(vectors, source=source)
    client.upsert(collection_name=MANUAL_COLLECTION_NAME, points=points, wait=True)

    total: int = client.count(collection_name=MANUAL_COLLECTION_NAME, exact=True).count
    logger.success("已写入集合 {}，当前点位总数：{}", MANUAL_COLLECTION_NAME, total)
    return total


def run_demo_query(client: QdrantClient, query: str, embedder: Any) -> None:
    """灌库后执行一次相似度检索，用于肉眼确认数据可用。

    Args:
        client: 已打开的本地 Qdrant 客户端。
        query: 自检用的自然语言故障描述。
        embedder: 与灌库阶段一致的向量化器。
    """
    logger.info("相似度自检查询：{}", query)
    query_vector: list[float] = embedder.embed_query(query)

    # qdrant-client 1.10+ 使用 query_points，1.9 及更早版本只有 search，这里做兼容
    if hasattr(client, "query_points"):
        hits = client.query_points(
            collection_name=MANUAL_COLLECTION_NAME,
            query=query_vector,
            limit=3,
            with_payload=True,
        ).points
    else:  # pragma: no cover - 兼容旧版客户端
        hits = client.search(
            collection_name=MANUAL_COLLECTION_NAME,
            query_vector=query_vector,
            limit=3,
            with_payload=True,
        )

    if not hits:
        logger.warning("检索无命中，请确认灌库是否成功")
        return

    for rank, hit in enumerate(hits, start=1):
        payload: dict[str, Any] = hit.payload or {}
        logger.info(
            "Top{} | score={:.4f} | error_code={} | target_part={} | {}",
            rank,
            hit.score,
            payload.get("error_code", "-"),
            payload.get("target_part", "-"),
            payload.get("section_title", "-"),
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="初始化 Qdrant 本地知识库（robot_manual_sop）",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="删除并重建集合（清空后重新灌库）",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="",
        help="灌库完成后执行一次相似度自检查询（传入自然语言故障描述）",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help=(
            "离线兜底模式：不调用云端 Embedding，改用文本哈希派生的确定性伪向量灌库，"
            "用于尚未配置 EMBEDDING_API_KEY 时打通链路（无真实语义检索能力）"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """脚本主流程。

    Returns:
        int: 进程退出码，0 表示成功，1 表示失败，130 表示用户中断。
    """
    args: argparse.Namespace = parse_args(argv)
    qdrant_path: Path = resolve_qdrant_path()
    logger.info("Qdrant 本地数据目录：{}", qdrant_path)
    logger.info("目标集合：{}（{} 维，Cosine）", MANUAL_COLLECTION_NAME, VECTOR_SIZE)

    client: QdrantClient | None = None
    try:
        if args.offline:
            logger.warning(
                "离线兜底模式：跳过云端 Embedding 探活，使用确定性伪向量灌库；"
                "配置真实 EMBEDDING_API_KEY 后请重新执行 --recreate 覆盖"
            )
            embedder: Any = _OfflineHashEmbeddings()
            source: str = "pseudo"
        else:
            # 先探活云端 Embedding，避免灌库到一半才发现 Key/维度有问题
            probe_embedding_dimension(expected=VECTOR_SIZE)
            embedder = get_embedding_model()
            source = "cloud"

        client = QdrantClient(path=str(qdrant_path))
        ensure_collection(client, recreate=bool(args.recreate))
        ingest(client, embedder, source=source)
        if args.query.strip():
            run_demo_query(client, args.query.strip(), embedder)
    except KeyboardInterrupt:  # pragma: no cover - 交互式中断
        logger.warning("已手动中断")
        return 130
    except Exception as exc:  # noqa: BLE001 - 命令行脚本需要友好的收尾提示
        logger.error("初始化失败，已终止：{}", exc)
        logger.opt(exception=exc).debug("异常堆栈（调试用）")
        return 1
    finally:
        # 本地模式必须显式关闭，否则进程退出后目录锁可能延迟释放
        if client is not None:
            client.close()

    logger.success("Qdrant 知识库初始化完成 ✔")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
