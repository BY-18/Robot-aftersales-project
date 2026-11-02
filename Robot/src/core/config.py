"""全局配置中心（Service Copilot）。

本模块使用 ``pydantic-settings`` 从 ``.env`` 文件读取并校验所有运行期配置，
集中管理三类外部依赖的参数：

1. **MySQL**：设备档案、备件库存、工单流转等结构化业务数据（同步 ``pymysql`` 驱动）。
2. **DeepSeek**：排障推理与备件抽取的主大模型（OpenAI 兼容协议）。
3. **Embedding API**：云端向量化服务（OpenAI 兼容协议，1024 维）。

设计约定（与项目「零本地大模型」原则一致）：

* 本项目**不做任何本地模型推理**，向量化一律走云端 HTTP 接口；
* Qdrant 使用官方客户端的本地文件模式，数据目录与 LangGraph 检查点文件路径
  按需求**硬编码为相对路径**，不通过环境变量注入，避免不同机器路径漂移。

使用方式::

    from src.core.config import settings

    print(settings.mysql_url)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from urllib.parse import quote_plus

from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------
# config.py 位于 <项目根>/src/core/config.py，因此向上一级两级即项目根目录。
CORE_DIR: Path = Path(__file__).resolve().parent
PROJECT_ROOT: Path = CORE_DIR.parent.parent

# ---------------------------------------------------------------------------
# 硬编码的本地持久化路径（按需求固定，不走 .env）
# ---------------------------------------------------------------------------
#: Qdrant 本地文件模式的数据目录（相对路径，相对于进程启动时的工作目录）。
#: 注意：本地模式下同一目录同一时间只能被**一个进程**打开，服务与脚本不要同时持有。
QDRANT_LOCAL_DATA_PATH: str = "./qdrant_local_data"

#: LangGraph 状态检查点文件（SqliteSaver），用于人在回路（HITL）断点续跑。
CHECKPOINT_DB_PATH: str = "checkpoints.sqlite"

#: 知识库向量维度。需与云端 Embedding 模型的实际输出维度一致（本项目固定 1024）。
VECTOR_SIZE: int = 1024

#: Qdrant 知识库集合名称（原厂维修手册 SOP 切片）。
MANUAL_COLLECTION_NAME: str = "robot_manual_sop"

#: 常见占位符标记：出现在 Key 里就说明用户还没填真实密钥。
_PLACEHOLDER_MARKERS: tuple[str, ...] = ("replace", "your", "请替换", "changeme", "xxx")


def is_usable_api_key(value: str) -> bool:
    """判断 .env 中的 API Key 是否真的可用。

    三类情况都视为「未配置」：

    1. 空字符串；
    2. 含非 ASCII 字符（例如中文占位符，塞进 HTTP 头会直接抛 UnicodeEncodeError）；
    3. 含 ``REPLACE`` / ``your`` / ``请替换`` 等占位符标记。

    这样可以避免「填了占位符就以为配好了，结果每次调用都白发一次注定失败的请求」。

    Args:
        value: 原始配置值。

    Returns:
        bool: 可用返回 ``True``。
    """
    cleaned: str = (value or "").strip()
    if not cleaned:
        return False
    lowered: str = cleaned.lower()
    if any(marker in lowered for marker in _PLACEHOLDER_MARKERS):
        return False
    try:
        cleaned.encode("ascii")
    except UnicodeEncodeError:
        return False
    return True


class Settings(BaseSettings):
    """应用配置对象。

    读取顺序（后者覆盖前者）：

    1. 类中定义的默认值；
    2. ``src/core/.env``；
    3. ``<项目根>/.env``；
    4. 真实环境变量（优先级最高）。

    字段均为大写，与 ``.env`` 中的键名一一对应（大小写不敏感）。
    """

    model_config = SettingsConfigDict(
        env_file=(CORE_DIR / ".env", PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # .env 中出现未声明变量时忽略，而不是抛错
    )

    # ---------------------- MySQL ----------------------
    MYSQL_HOST: str = "127.0.0.1"
    MYSQL_PORT: int = 3306
    MYSQL_USER: str = "root"
    #: 数据库密码**不写默认值**，一律由 .env 或环境变量注入，避免随仓库泄漏。
    MYSQL_PASSWORD: str = ""
    MYSQL_DB: str = "service_copilot"
    MYSQL_CHARSET: str = "utf8mb4"

    # 连接池参数：本地单机私有化部署，池子无需过大
    MYSQL_POOL_SIZE: int = 5
    MYSQL_MAX_OVERFLOW: int = 10
    MYSQL_POOL_RECYCLE: int = 3600  # 秒；避免 MySQL 8 默认 8 小时空闲断连
    MYSQL_ECHO: bool = False  # True 时 SQLAlchemy 会打印全部 SQL，调试用

    # ---------------------- DeepSeek 大模型 ----------------------
    # 密钥统一放在 src/core/.env（已被 .gitignore 忽略），源码里**不写任何默认 Key**。
    # 留空时 has_deepseek_key 为 False，槽位抽取会自动降级为正则匹配（见 nodes.py）。
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"
    DEEPSEEK_MODEL: str = "deepseek-chat"
    DEEPSEEK_TEMPERATURE: float = 0.0  # 排障场景要求稳定可复现，温度取 0
    LLM_TIMEOUT: int = 60
    LLM_MAX_RETRIES: int = 2

    # ---------------------- 云端 Embedding ----------------------
    #: 留空时 has_embedding_key 为 False，RAG 检索会自动降级为本地关键词打分（见 rag_tool.py）。
    EMBEDDING_API_KEY: str = ""
    EMBEDDING_BASE_URL: str = "https://maas.qianwenaiapi.com/compatible-mode/v1"
    #: 默认 bge-m3：原生 1024 维，与 VECTOR_SIZE 对齐，且无需传 dimensions 参数。
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    #: 若所用模型支持「输出维度裁剪」（如 text-embedding-3-*、text-embedding-v3），
    #: 可在 .env 中显式指定为 1024；留空表示不传该参数。
    EMBEDDING_DIMENSIONS: int | None = None
    #: 批量向量化时的单批条数，部分服务商限制为 10 条/次
    EMBEDDING_BATCH_SIZE: int = 10

    # ---------------------- 应用 ----------------------
    APP_ENV: str = "dev"
    LOG_LEVEL: str = "INFO"
    #: 应用版本号，随阶段推进递增（在 ``GET /health`` 中回报）。
    APP_VERSION: str = "0.4.0"
    #: CORS 允许来源，逗号分隔；``*`` 表示放开（本地控制台/演示用）。
    #: 生产环境请改为具体域名，例如 ``https://copilot.example.com``。
    CORS_ALLOW_ORIGINS: str = "*"

    # ------------------------------------------------------------------
    # 派生属性
    # ------------------------------------------------------------------
    @property
    def mysql_url(self) -> str:
        """返回 SQLAlchemy 可直接使用的同步 MySQL 连接串。

        密码经过 ``quote_plus`` 转义，避免密码中出现 ``@``、``/``、``:`` 等
        特殊字符时把 DSN 解析错位。

        Returns:
            str: 形如 ``mysql+pymysql://root:***@127.0.0.1:3306/service_copilot?charset=utf8mb4``
        """
        password: str = quote_plus(self.MYSQL_PASSWORD)
        return (
            f"mysql+pymysql://{self.MYSQL_USER}:{password}"
            f"@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DB}"
            f"?charset={self.MYSQL_CHARSET}"
        )

    @property
    def mysql_url_masked(self) -> str:
        """返回密码打码后的连接串，仅用于日志输出，防止密钥泄漏。"""
        return (
            f"mysql+pymysql://{self.MYSQL_USER}:***"
            f"@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DB}"
        )

    @property
    def qdrant_local_path(self) -> str:
        """Qdrant 本地文件模式数据目录（硬编码相对路径）。"""
        return QDRANT_LOCAL_DATA_PATH

    @property
    def qdrant_local_abspath(self) -> Path:
        """Qdrant 数据目录的绝对路径。

        相对路径是相对于**进程工作目录**的，服务用 uvicorn 从别的目录启动时就会
        指向错误位置，因此统一在这里锚定到项目根目录。
        """
        raw: Path = Path(QDRANT_LOCAL_DATA_PATH)
        return raw if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()

    @property
    def checkpoint_db_path(self) -> str:
        """LangGraph 检查点（SQLite）文件路径（硬编码相对路径）。"""
        return CHECKPOINT_DB_PATH

    @property
    def checkpoint_db_abspath(self) -> Path:
        """LangGraph 检查点文件的绝对路径。"""
        raw: Path = Path(CHECKPOINT_DB_PATH)
        return raw if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()

    @property
    def vector_size(self) -> int:
        """向量维度（固定 1024）。"""
        return VECTOR_SIZE

    @property
    def has_deepseek_key(self) -> bool:
        """是否已配置可用的 DeepSeek API Key（占位符视为未配置）。"""
        return is_usable_api_key(self.DEEPSEEK_API_KEY)

    @property
    def cors_origins(self) -> list[str]:
        """把逗号分隔的 CORS 配置解析为来源列表。

        Returns:
            list[str]: 来源列表；配置为空时回退为 ``["*"]``（放开但关闭凭证）。
        """
        raw: str = (self.CORS_ALLOW_ORIGINS or "").strip()
        if not raw or raw == "*":
            return ["*"]
        return [item.strip() for item in raw.split(",") if item.strip()]

    @property
    def has_embedding_key(self) -> bool:
        """是否已配置可用的云端 Embedding API Key（占位符视为未配置）。"""
        return is_usable_api_key(self.EMBEDDING_API_KEY)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """获取全局唯一的配置实例（带缓存，进程内只解析一次 .env）。"""
    return Settings()


#: 模块级单例，便于 `from src.core.config import settings` 直接使用。
settings: Settings = get_settings()


if __name__ == "__main__":  # pragma: no cover - 手工自检入口
    from loguru import logger

    logger.info("项目根目录      : {}", PROJECT_ROOT)
    logger.info("MySQL 连接串    : {}", settings.mysql_url_masked)
    logger.info("DeepSeek 模型   : {} @ {}", settings.DEEPSEEK_MODEL, settings.DEEPSEEK_BASE_URL)
    logger.info("DeepSeek Key    : {}", "已配置" if settings.has_deepseek_key else "缺失")
    logger.info("Embedding 模型  : {} @ {}", settings.EMBEDDING_MODEL, settings.EMBEDDING_BASE_URL)
    logger.info("Embedding Key   : {}", "已配置" if settings.has_embedding_key else "缺失")
    logger.info("Qdrant 数据目录 : {}", settings.qdrant_local_path)
    logger.info("检查点文件      : {}", settings.checkpoint_db_path)
    logger.info("向量维度        : {}", settings.vector_size)
