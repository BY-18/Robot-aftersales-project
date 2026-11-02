"""Service Copilot 服务入口：FastAPI 应用装配、静态大屏挂载与全链路 Trace ID 中间件。

职责
----
1. **创建应用**：标题 / 版本 / OpenAPI 元数据统一取自 ``src.core.config.settings``；
2. **CORS**：允许本地 Streamlit 控制台等前端跨域调用（来源可在 ``.env`` 中收紧）；
3. **Trace ID 中间件**：读取 ``X-Trace-Id`` 请求头（缺失或不合规则现场生成），
   写入 ``request.state`` 与 Loguru 的 ContextVar，并在响应头回写 ``X-Trace-Id``，
   保证「一次报修 → 一次审批 → 一次派工」的日志能被同一个 ID 串起来；
4. **挂载路由**：``/api/v1`` 业务接口 + 根路径 ``/health`` 健康检查；
5. **挂载静态大屏**：``src/static`` 挂到 ``/``，浏览器直接打开根路径即可体验全闭环
   （挂载顺序在路由之后，确保 ``/api/v1/*`` 与 ``/health`` 不会被静态资源抢占）；
6. **生命周期**：启动时探测 MySQL、退出时释放连接池与本地 Qdrant 文件锁。

本地启动::

    uvicorn main:app --host 127.0.0.1 --port 8000 --reload
    python main.py --port 8000          # 等价写法（内部调用 uvicorn.run）

访问地址::

    http://127.0.0.1:8000/              # 排障与备件调度控制台（静态大屏）
    http://127.0.0.1:8000/docs          # Swagger UI
    http://127.0.0.1:8000/openapi.json  # OpenAPI 描述
"""

from __future__ import annotations

import argparse
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Awaitable, Callable

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from src.api.routes import health_router, router
from src.api.schemas import is_valid_trace_id
from src.core.config import settings
from src.core.db import dispose_engine, verify_mysql_connection
from src.core.logger import logger, new_trace_id, trace_context
from src.tools.rag_tool import close_qdrant_client

__all__ = ["app", "create_app", "TraceIdMiddleware", "TRACE_HEADER", "STATIC_DIR"]

#: 全链路追踪请求 / 响应头名称
TRACE_HEADER: str = "X-Trace-Id"

#: 静态控制台目录（单文件大屏 index.html 所在处）
STATIC_DIR: Path = Path(__file__).resolve().parent / "src" / "static"

#: OpenAPI 标签排序（让 Swagger UI 的接口分组与业务流程一致）
_OPENAPI_TAGS: list[dict[str, str]] = [
    {"name": "诊断接入", "description": "通道 A 人工报修与通道 B 设备 Webhook 告警"},
    {"name": "人工审批", "description": "HITL 审批恢复：断点续跑、库存核销与派工"},
    {"name": "数据查询", "description": "前端控制台使用的工单与备件库存查询"},
    {"name": "运维监控", "description": "依赖组件健康检查"},
]


class TraceIdMiddleware(BaseHTTPMiddleware):
    """为每个请求建立 trace_id 上下文，并在响应头回写。"""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """解析或生成 trace_id，绑定日志上下文后放行请求。

        Args:
            request: 当前请求。
            call_next: 下游 ASGI 应用调用入口。

        Returns:
            Response: 下游响应（已附加 ``X-Trace-Id`` 头）。
        """
        incoming: str = (request.headers.get(TRACE_HEADER) or "").strip()
        trace_id: str = incoming if is_valid_trace_id(incoming) else new_trace_id()
        if incoming and trace_id != incoming:
            logger.debug("[api] 请求头 {!r} 不合规，已改用服务端生成的 trace_id", incoming)

        request.state.trace_id = trace_id
        with trace_context(trace_id):
            response: Response = await call_next(request)
            # 端点可能用请求体里的 trace_id 覆盖（见 routes.resolve_trace_id），以最终值为准
            effective: str = str(getattr(request.state, "trace_id", "") or trace_id)
            response.headers[TRACE_HEADER] = effective
            return response


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """应用生命周期：启动探活 + 退出清理。

    Args:
        _app: FastAPI 实例（框架注入，此处不需要使用）。

    Yields:
        None: 服务运行期。
    """
    logger.info("=" * 74)
    logger.info(
        "Service Copilot API 启动 | env={} version={}", settings.APP_ENV, settings.APP_VERSION
    )
    logger.info("MySQL      : {}", settings.mysql_url_masked)
    logger.info("Qdrant     : {}", settings.qdrant_local_abspath)
    logger.info("检查点文件 : {}", settings.checkpoint_db_abspath)
    logger.info("CORS 来源  : {}", settings.cors_origins)
    logger.info("控制台     : http://127.0.0.1:8000/  ← 静态大屏（src/static/index.html）")
    logger.info("=" * 74)

    # 启动探活只记录日志，不阻断启动：MySQL 掉线时服务仍可提供 /health 与报错信息
    verify_mysql_connection()

    try:
        yield
    finally:
        close_qdrant_client()
        dispose_engine()
        logger.info("Service Copilot API 已退出：连接池与本地 Qdrant 文件锁已释放")


def create_app() -> FastAPI:
    """创建并装配 FastAPI 应用。

    Returns:
        FastAPI: 已配置 CORS、Trace ID 中间件与全部路由的应用实例。
    """
    application: FastAPI = FastAPI(
        title="Service Copilot · 机器人售后排障与备件调度",
        description=(
            "双通道接入（人工报修 / 设备 Webhook）+ LangGraph 排障状态机 + "
            "人在回路审批（HITL）+ 备件库存调度的一体化售后中台。\n\n"
            f"所有接口都会返回 `{TRACE_HEADER}` 响应头，其值即全链路 `trace_id`"
            "（同时是 LangGraph 的 `thread_id`）。"
        ),
        version=settings.APP_VERSION,
        openapi_tags=_OPENAPI_TAGS,
        lifespan=lifespan,
    )

    # CORS：本地控制台 / 演示页直接跨域调用；生产环境把 CORS_ALLOW_ORIGINS 收窄为具体域名
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,  # 未使用 Cookie 鉴权，配合 allow_origins=["*"] 才是合法组合
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[TRACE_HEADER],
    )
    application.add_middleware(TraceIdMiddleware)

    application.include_router(router)
    application.include_router(health_router)

    # 静态大屏必须**最后**挂载：Starlette 按注册顺序匹配路由，挂到 "/" 的 Mount 会吃掉
    # 所有未匹配路径，若先挂它，后面的 /api/v1/* 与 /health 就再也命不中了。
    # html=True 让 GET / 直接返回 index.html，无需额外写重定向路由。
    if STATIC_DIR.is_dir():
        application.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="console")
        logger.debug("静态大屏已挂载：/ → {}", STATIC_DIR)
    else:  # 目录被删或部署时漏打包，不影响 API 可用性
        logger.warning("静态大屏目录不存在，跳过挂载：{}", STATIC_DIR)

    return application


#: 模块级应用实例（``uvicorn main:app`` 依赖该名称）
app: FastAPI = create_app()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析 ``python main.py`` 的启动参数。"""
    parser = argparse.ArgumentParser(description="Service Copilot FastAPI 服务")
    parser.add_argument(
        "--host",
        default=os.getenv("API_HOST", "127.0.0.1"),
        help="监听地址（默认 127.0.0.1，容器内请用 0.0.0.0）",
    )
    parser.add_argument("--port", type=int, default=int(os.getenv("API_PORT", "8000")))
    parser.add_argument("--reload", action="store_true", help="开发模式热重载")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """命令行启动入口（``python main.py``）。

    Returns:
        int: 进程退出码。
    """
    args: argparse.Namespace = parse_args(argv)
    import uvicorn  # 局部导入：仅命令行启动时才需要

    logger.info("以命令行方式启动 uvicorn：http://{}:{}", args.host, args.port)
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        reload=bool(args.reload),
        log_config=None,  # 交给 Loguru，避免 uvicorn 默认 logging 与 Trace ID 格式打架
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
