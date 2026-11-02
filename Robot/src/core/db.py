"""MySQL 数据库访问层（SQLAlchemy 2.x + PyMySQL 同步驱动）。

职责：

* 基于 ``src.core.config.settings.mysql_url`` 构建全局 ``Engine``；
* 暴露线程安全的 ``SessionLocal`` 会话工厂；
* 提供 FastAPI 依赖注入用的 ``get_db()`` 与脚本用的 ``session_scope()``；
* 提供 ``verify_mysql_connection()``，在服务/脚本启动时显式探测连通性，
  连接失败时输出明确的 Loguru 报错，而不是让异常在业务深处才炸。

安全约定（对应需求 FR-3.3）：业务用的 MySQL 账号应为只读账号；
本项目所有 Text2SQL 查询一律通过 SQLAlchemy ``text()`` + ``bindparams`` 参数化绑定，
详见 ``src/tools/text2sql_tool.py``（后续阶段实现）。
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.core.config import settings

__all__ = [
    "engine",
    "SessionLocal",
    "get_db",
    "session_scope",
    "verify_mysql_connection",
    "dispose_engine",
]


def _build_engine() -> Engine:
    """构建 SQLAlchemy Engine。

    Returns:
        Engine: 已配置连接池与探活策略的引擎实例。

    Raises:
        RuntimeError: 连接串非法或驱动未安装（如缺少 ``pymysql`` / ``cryptography``）。
    """
    url: str = settings.mysql_url
    try:
        engine_obj: Engine = create_engine(
            url,
            echo=settings.MYSQL_ECHO,
            # 从连接池取出连接前先 ping 一次，避免拿到已被 MySQL 服务端回收的死连接
            pool_pre_ping=True,
            pool_size=settings.MYSQL_POOL_SIZE,
            max_overflow=settings.MYSQL_MAX_OVERFLOW,
            pool_recycle=settings.MYSQL_POOL_RECYCLE,
            future=True,
            # 连接超时，避免网络不通时线程长时间挂死
            connect_args={"connect_timeout": 5},
        )
    except Exception as exc:  # noqa: BLE001 - 此处需要兜底并给出可读报错
        logger.error(
            "构建 MySQL Engine 失败：{} | 目标={} | 请检查 .env 中的 "
            "MYSQL_HOST / MYSQL_PORT / MYSQL_USER / MYSQL_PASSWORD / MYSQL_DB，"
            "以及是否已安装 pymysql、cryptography",
            exc,
            settings.mysql_url_masked,
        )
        raise RuntimeError(f"MySQL Engine 初始化失败：{exc}") from exc

    logger.debug("MySQL Engine 已创建：{}", settings.mysql_url_masked)
    return engine_obj


#: 全局唯一 Engine（进程内复用连接池）
engine: Engine = _build_engine()

#: 会话工厂。expire_on_commit=False 便于提交后继续读取 ORM 对象属性。
SessionLocal: sessionmaker[Session] = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


def verify_mysql_connection() -> bool:
    """显式探测 MySQL 连通性（执行 ``SELECT 1``）。

    适合在 FastAPI 启动事件或初始化脚本开头调用。

    Returns:
        bool: 连接可用返回 ``True``；失败返回 ``False`` 并已记录错误日志。
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001 - 探活需要兜底
        logger.error(
            "MySQL 连接失败：{} | 目标={} | 请确认 MySQL 服务已启动、库 service_copilot "
            "已创建、账号密码正确，且该账号具备 SELECT 权限",
            exc,
            settings.mysql_url_masked,
        )
        return False

    logger.success(
        "MySQL 连接正常：{} / {}",
        settings.mysql_url_masked,
        settings.MYSQL_DB,
    )
    return True


def get_db() -> Generator[Session, None, None]:
    """FastAPI 依赖注入用的会话生成器。

    Yields:
        Session: 单个请求生命周期内的数据库会话；异常时自动回滚。
    """
    db: Session = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """脚本/工具函数用的会话上下文管理器（自动提交、异常回滚、最终关闭）。

    Examples:
        >>> with session_scope() as db:            # doctest: +SKIP
        ...     rows = db.execute(text("SELECT 1")).all()
    """
    db: Session = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def dispose_engine() -> None:
    """释放连接池中的全部连接（服务优雅退出时调用）。"""
    engine.dispose()
    logger.debug("MySQL 连接池已释放")


if __name__ == "__main__":  # pragma: no cover - 手工连通性自检入口
    raise SystemExit(0 if verify_mysql_connection() else 1)
