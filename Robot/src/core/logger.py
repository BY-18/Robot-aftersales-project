"""统一日志底座：Loguru + Trace ID 上下文穿透 + 文件落盘。

日志去哪了
----------
一次 ``import src.core.logger`` 就会装配三个 Handler（``configure_logging()`` 幂等，
重复导入/重复调用不会叠加）：

============================  ==========  ==============  ====================
目标                          级别        轮转 / 保留      用途
============================  ==========  ==============  ====================
``stderr``（控制台）           ``INFO``    -               开发调试、容器 stdout 采集
``logs/app.log``               ``INFO``    10 MB / 15 天   业务全量轨迹（含 ERROR）
``logs/error.log``             ``ERROR``   10 MB / 30 天   异常单独成卷，便于告警与事后定位
============================  ==========  ==============  ====================

``logs/`` 位于项目根目录，随首次配置自动创建（已被 ``.gitignore`` 忽略）。
文件格式不带 ANSI 颜色，字段顺序与控制台一致并保留 ``trace_id`` 与
``模块:函数:行号``，可直接 grep / 交给 Filebeat、Loki 等采集器。

设计要点
--------
1. **Trace ID 用 ContextVar 存储**：同一个请求/工作流内，所有模块打的日志都自动
   带上同一个 ``trace_id``，无需层层传参（FastAPI 中间件里 ``set_trace_id()`` 一次即可）。
2. **patcher 注入 extra**：Loguru 的格式化串里写 ``{extra[trace_id]}`` 时，若该字段
   不存在会直接报错，因此用 ``logger.configure(patcher=...)`` 在每条日志渲染前兜底填充。
3. **幂等配置**：``configure_logging()`` 重复调用不会叠加 handler。
4. **日志不能拖垮服务**：``logs/`` 不可写（如只读容器）时降级为仅控制台输出并告警，
   而不是让整个应用起不来。

用法::

    from src.core.logger import logger, trace_context

    with trace_context() as tid:            # 也可 trace_context("TR-xxx") 指定
        logger.info("开始检索")              # 控制台 + logs/app.log 都会带上 TR-xxx
"""

from __future__ import annotations

import sys
import uuid
from contextlib import contextmanager
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any, Iterator

from loguru import logger

from src.core.config import PROJECT_ROOT, settings

__all__ = [
    "logger",
    "new_trace_id",
    "get_trace_id",
    "set_trace_id",
    "trace_context",
    "configure_logging",
    "LOG_DIR",
    "APP_LOG_PATH",
    "ERROR_LOG_PATH",
]

#: 默认占位符：不在任何 trace 上下文中时显示 "-"
_TRACE_ID_DEFAULT: str = "-"

_trace_id_var: ContextVar[str] = ContextVar("trace_id", default=_TRACE_ID_DEFAULT)

# ---------------------------------------------------------------------------
# 文件落盘配置
# ---------------------------------------------------------------------------
#: 日志目录（项目根/logs），首次配置时自动创建
LOG_DIR: Path = PROJECT_ROOT / "logs"

#: 业务全量日志：工具调用 / SQL 执行 / API 请求 / Agent 状态扭转都在这里
APP_LOG_PATH: Path = LOG_DIR / "app.log"

#: 异常日志：只收 ERROR 及以上，便于单独接告警
ERROR_LOG_PATH: Path = LOG_DIR / "error.log"

#: 单文件轮转阈值
_ROTATION: str = "10 MB"
#: 业务日志保留窗口
_APP_RETENTION: str = "15 days"
#: 异常日志保留窗口（排障追溯期更长，留 30 天）
_ERROR_RETENTION: str = "30 days"
#: 业务日志级别
_APP_LEVEL: str = "INFO"
#: 异常日志级别
_ERROR_LEVEL: str = "ERROR"

#: 控制台输出格式：时间 | 级别 | trace_id | 模块:函数:行号 - 消息
CONSOLE_FORMAT: str = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <7}</level> | "
    "<cyan>{extra[trace_id]}</cyan> | "
    "<magenta>{name}</magenta>:<magenta>{function}</magenta>:<magenta>{line}</magenta>"
    " - <level>{message}</level>"
)

#: 文件输出格式：与控制台同构，但去掉 ANSI 颜色，保证日志文件可被 grep / 采集器解析
FILE_FORMAT: str = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
    "{level: <7} | "
    "{extra[trace_id]} | "
    "{name}:{function}:{line} - {message}"
)

_configured: bool = False


def new_trace_id(prefix: str = "TR") -> str:
    """生成一个新的全链路追踪 ID。

    Args:
        prefix: ID 前缀，默认 ``TR``。

    Returns:
        str: 形如 ``TR-9f2c1b7d4e60`` 的追踪号。
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def get_trace_id() -> str:
    """读取当前上下文的 Trace ID；未设置时返回 ``-``。"""
    return _trace_id_var.get()


def set_trace_id(trace_id: str | None = None) -> str:
    """设置当前上下文的 Trace ID。

    Args:
        trace_id: 指定的追踪号；传 ``None`` 时自动生成。

    Returns:
        str: 实际生效的 Trace ID。
    """
    tid: str = trace_id or new_trace_id()
    _trace_id_var.set(tid)
    return tid


@contextmanager
def trace_context(trace_id: str | None = None) -> Iterator[str]:
    """Trace ID 上下文管理器，退出后自动还原为上一层上下文。

    Args:
        trace_id: 指定追踪号；``None`` 时自动生成。

    Yields:
        str: 本次上下文生效的 Trace ID。
    """
    tid: str = trace_id or new_trace_id()
    token: Token[str] = _trace_id_var.set(tid)
    try:
        yield tid
    finally:
        _trace_id_var.reset(token)


def _inject_trace_id(record: dict[str, Any]) -> None:
    """Loguru patcher：保证每条日志的 extra 中都有 trace_id。"""
    record["extra"].setdefault("trace_id", _trace_id_var.get())


def _add_file_handler(path: Path, level: str, retention: str) -> bool:
    """挂载一个文件 Handler（自动创建目录）。

    目录不可写时**不让服务启动失败**：只告警并继续用控制台输出。这在只读容器、
    挂载盘权限不对的场景下很常见，日志系统不该成为可用性的单点。

    Args:
        path: 日志文件绝对路径。
        level: 该 Handler 接收的最低级别。
        retention: 保留策略（如 ``15 days``）。

    Returns:
        bool: 挂载成功返回 ``True``。
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            str(path),
            level=level,
            format=FILE_FORMAT,
            encoding="utf-8",
            rotation=_ROTATION,
            retention=retention,
            colorize=False,  # 文件里绝不能带 ANSI 转义符，否则 grep / 采集器全是乱码
            enqueue=True,  # 异步写：FastAPI 线程池并发打日志时不会阻塞业务线程
            backtrace=False,
            diagnose=False,
        )
        return True
    except OSError as exc:  # 只读容器 / 权限不足 / 磁盘满
        logger.warning("日志目录不可写，已降级为仅控制台输出 | {} | {}", path, exc)
        return False


def configure_logging(
    level: str | None = None,
    force: bool = False,
    log_to_file: bool = True,
) -> None:
    """初始化日志输出：控制台 + 两个文件 Handler（幂等）。

    Args:
        level: 控制台日志级别，默认取 ``settings.LOG_LEVEL``。
        force: 为 ``True`` 时强制重新配置（测试或改配置后重载用）。
        log_to_file: 是否同时落盘到 ``logs/``；测试里可传 ``False`` 避免写文件。
    """
    global _configured
    if _configured and not force:
        return

    logger.remove()

    # patcher 要在**任何**日志产生之前装好：下面挂 handler 时的告警/汇总日志本身就带
    # ``{extra[trace_id]}`` 占位，缺字段会渲染失败，所以不能等到最后再 configure。
    logger.configure(patcher=_inject_trace_id)

    # ---- Handler 1：控制台（带颜色，容器 stdout 采集）----
    logger.add(
        sys.stderr,
        level=(level or settings.LOG_LEVEL).upper(),
        format=CONSOLE_FORMAT,
        colorize=True,
        backtrace=False,
        diagnose=False,
    )

    # ---- Handler 2 / 3：文件落盘（app.log 全量、error.log 仅异常）----
    if log_to_file:
        app_ok: bool = _add_file_handler(APP_LOG_PATH, _APP_LEVEL, _APP_RETENTION)
        error_ok: bool = _add_file_handler(ERROR_LOG_PATH, _ERROR_LEVEL, _ERROR_RETENTION)
        if app_ok and error_ok:
            logger.info("日志落盘已启用 | 业务 {} | 异常 {}", APP_LOG_PATH, ERROR_LOG_PATH)

    _configured = True


# 导入即生效，保证脚本/服务使用同一套格式与落盘策略
configure_logging()
