"""一键验收脚本：环境预检 → 三套功能测试 → 真实服务冒烟 → 数据卫生。

为什么要单独写这个脚本
----------------------
``test_tools.py`` / ``test_graph.py`` / ``test_api.py`` 各自只验证一层，而
"项目能不能跑通"是一个**组合命题**：依赖装没装、Key 配没配、MySQL/知识库是否就绪、
三套测试能否同时通过、真实 HTTP 服务能否起来、跑完有没有污染业务数据。
本脚本把这些串成一条可复现的验收流水线，任何一环失败都会让退出码非 0。

检查项与含义
------------
===========================  ==================================================
步骤                          证明什么
===========================  ==================================================
1. 运行时预检                  Python 版本、13 个关键依赖、API Key、本地数据目录
2. 数据底座                    MySQL 连通、三张表行数、库存基线快照
3. 功能测试（三套）            工具层 4 项 / 状态机 6 项 / 接口层 9 项
4. 真实服务冒烟                ``python main.py`` 起 uvicorn → /health → 双通道
                               → 审批闭环 → 控制台查询，并记录端到端耗时
5. 数据卫生                    订单数与库存是否回到第 2 步的基线
===========================  ==================================================

运行::

    python scripts/verify_all.py               # 完整验收（约 2 分钟）
    python scripts/verify_all.py --skip-suites # 跳过功能测试，只做冒烟
    python scripts/verify_all.py --skip-live   # 跳过真实服务，只跑三套测试

退出码：0 全部通过（允许 WARN）；1 存在 FAIL；2 环境不满足无法开始。
"""

from __future__ import annotations

import argparse
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
for _path in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from src.core.logger import logger  # noqa: E402

#: 关键依赖（缺任何一个都应视为环境未就绪）
_REQUIRED_PACKAGES: tuple[str, ...] = (
    "fastapi",
    "uvicorn",
    "httpx",
    "langgraph",
    "langgraph-checkpoint-sqlite",
    "langchain-openai",
    "sqlalchemy",
    "pymysql",
    "cryptography",
    "qdrant-client",
    "pydantic",
    "pydantic-settings",
    "loguru",
)

#: 三套功能测试：命令 → 期望通过的用例数
_SUITES: tuple[tuple[str, int], ...] = (
    ("scripts/test_tools.py --with-order", 4),
    ("scripts/test_graph.py", 6),
    ("scripts/test_api.py", 9),
)

#: 计划书验收口径：报修到工单挂起的端到端目标（毫秒）
TARGET_E2E_MS: float = 5000.0

PASS: str = "PASS"
WARN: str = "WARN"
FAIL: str = "FAIL"

#: 验收结果矩阵：(步骤, 名称, 状态, 明细, 耗时秒)
_RESULTS: list[tuple[str, str, str, str, float]] = []
#: 环境信息（打印在报告头部）
_ENV_INFO: dict[str, str] = {}


# ---------------------------------------------------------------------------
# 结果记录
# ---------------------------------------------------------------------------
def _record(step: str, name: str, status: str, detail: str, seconds: float) -> None:
    """记录一条检查结果并即时打印（便于长流程观察进度）。"""
    _RESULTS.append((step, name, status, detail, seconds))
    icon: str = {PASS: "✓", WARN: "!", FAIL: "✗"}[status]
    logger.info("{} [{}] {:<32} {}", icon, status, name, detail)


def _timed(step: str, name: str, func: Callable[[], tuple[str, str, Any]]) -> tuple[str, Any]:
    """执行一个检查函数并记录耗时。

    Args:
        step: 步骤编号（如 ``3.测试``）。
        name: 检查项名称。
        func: 返回 ``(状态, 明细, 载荷)`` 的检查函数。

    Returns:
        tuple[str, Any]: ``(状态, 载荷)``；载荷供后续步骤复用（如基线快照）。
    """
    started: float = time.perf_counter()
    try:
        status, detail, payload = func()
    except Exception as exc:  # noqa: BLE001 - 单步失败不应中断整条流水线
        logger.exception("检查步骤 {} 抛出异常", name)
        status, detail, payload = FAIL, f"异常：{type(exc).__name__}: {exc}", None
    _record(step, name, status, detail, time.perf_counter() - started)
    return status, payload


def _child_env() -> dict[str, str]:
    """构造子进程环境变量：继承父进程 + 强制 UTF-8。

    Returns:
        dict[str, str]: 可直接传给 ``subprocess`` 的环境变量字典。
    """
    return {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}


# ---------------------------------------------------------------------------
# 步骤 1：运行时预检
# ---------------------------------------------------------------------------
def check_runtime() -> tuple[str, str, None]:
    """Python 版本、关键依赖、API Key、本地数据目录。"""
    from importlib.metadata import PackageNotFoundError, version

    py = sys.version_info
    _ENV_INFO["python"] = f"{py.major}.{py.minor}.{py.micro}"
    logger.info("Python {} | 解释器 {}", _ENV_INFO["python"], sys.executable)

    missing: list[str] = []
    for package in _REQUIRED_PACKAGES:
        try:
            _ENV_INFO[package] = version(package)
        except PackageNotFoundError:
            missing.append(package)
    if missing:
        return FAIL, f"缺少依赖：{', '.join(missing)}", None

    from src.core.config import settings

    _ENV_INFO["deepseek"] = f"{settings.DEEPSEEK_MODEL} @ {settings.DEEPSEEK_BASE_URL}"
    _ENV_INFO["embedding"] = f"{settings.EMBEDDING_MODEL} @ {settings.EMBEDDING_BASE_URL}"

    key_state: str = (
        f"DeepSeek={'已配置' if settings.has_deepseek_key else '缺失'} "
        f"Embedding={'已配置' if settings.has_embedding_key else '缺失'}"
    )
    if not (settings.has_deepseek_key and settings.has_embedding_key):
        # 缺 Key 不会让链路崩（有正则 / 关键字兜底），但抽取与检索质量会明显下降
        return WARN, f"{key_state}（将走降级链路）| {len(_REQUIRED_PACKAGES)} 个依赖就绪", None

    qdrant_dir: Path = settings.qdrant_local_abspath
    if not qdrant_dir.exists():
        return WARN, f"{key_state} | 本地 Qdrant 目录不存在（首次检索会自动创建）", None

    return PASS, f"Python {py.major}.{py.minor} | 依赖 {len(_REQUIRED_PACKAGES)} 个就绪 | {key_state}", None


# ---------------------------------------------------------------------------
# 步骤 2：数据底座与基线快照
# ---------------------------------------------------------------------------
def check_data() -> tuple[str, str, dict[str, Any] | None]:
    """MySQL 连通性、三张表行数、库存基线。"""
    from sqlalchemy import text

    from src.core.db import session_scope, verify_mysql_connection

    if not verify_mysql_connection():
        return FAIL, "MySQL 连接失败（检查 .env 与数据库服务）", None

    with session_scope() as db:
        counts: dict[str, int] = {
            table: int(db.execute(text(f"select count(*) from {table}")).scalar() or 0)
            for table in ("robot_devices", "spare_parts", "work_orders")
        }
        stock_rows = (
            db.execute(
                text(
                    "select warehouse_name, stock_quantity from spare_parts "
                    "where part_name = :part_name"
                ).bindparams(part_name="型号A电磁阀")
            )
            .mappings()
            .all()
        )
    stock_baseline: dict[str, int] = {
        str(row["warehouse_name"]): int(row["stock_quantity"]) for row in stock_rows
    }
    baseline: dict[str, Any] = {"counts": counts, "stock": stock_baseline}

    empty: list[str] = [name for name, count in counts.items() if count == 0]
    if empty:
        return FAIL, f"数据表为空：{', '.join(empty)}（请先导入种子数据）", baseline
    if not stock_baseline:
        return FAIL, "备件「型号A电磁阀」无库存记录，验收用例无法进行", baseline

    return (
        PASS,
        f"设备 {counts['robot_devices']} / 备件 {counts['spare_parts']} / 工单 "
        f"{counts['work_orders']} 行 | 型号A电磁阀库存 {stock_baseline}",
        baseline,
    )


# ---------------------------------------------------------------------------
# 步骤 3：三套功能测试
# ---------------------------------------------------------------------------
def _run_script(command: str) -> tuple[int, str]:
    """在项目根目录执行一个测试脚本。

    Args:
        command: 形如 ``scripts/test_api.py --keep`` 的命令字符串。

    Returns:
        tuple[int, str]: ``(退出码, stderr 尾部)``；日志走 stderr，故用它做断言与排错。
    """
    proc: subprocess.CompletedProcess[str] = subprocess.run(
        [sys.executable, "-X", "utf8", *command.split()],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
        check=False,
        env=_child_env(),
    )
    return proc.returncode, (proc.stderr or "")[-3000:]


def check_suites() -> tuple[str, str, None]:
    """依次执行工具层 / 状态机 / 接口层三套测试。"""
    summary: list[str] = []
    problems: list[str] = []
    for command, expected in _SUITES:
        script: str = command.split()[0]
        label: str = script.replace("scripts/", "").replace(".py", "")
        code, tail = _run_script(command)
        match = re.findall(r"通过\s+(\d+)/(\d+)", tail)
        passed, total = (int(match[-1][0]), int(match[-1][1])) if match else (-1, -1)
        if code != 0 or passed != expected:
            problems.append(f"{label} exit={code} {passed}/{total}（期望 {expected}）")
        summary.append(f"{label} {passed}/{total}")

    if problems:
        return FAIL, "；".join(problems), None
    return PASS, " | ".join(summary), None


# ---------------------------------------------------------------------------
# 步骤 4：真实服务冒烟
# ---------------------------------------------------------------------------
def _free_port() -> int:
    """申请一个空闲端口。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_health(client: Any, timeout: float = 90.0) -> dict[str, Any] | None:
    """轮询 ``/health`` 直到服务就绪。

    Args:
        client: ``httpx.Client`` 实例。
        timeout: 最长等待秒数。

    Returns:
        dict[str, Any] | None: 健康检查响应体；超时返回 ``None``。
    """
    deadline: float = time.time() + timeout
    while time.time() < deadline:
        try:
            response = client.get("/health", timeout=3.0)
            if response.status_code == 200:
                return response.json()
        except Exception:  # noqa: BLE001 - 服务尚未监听时连接会被拒绝
            time.sleep(1.0)
    return None


def check_live_service() -> tuple[str, str, None]:
    """真实 uvicorn 进程：健康检查 → 双通道 → 审批闭环 → 控制台查询。"""
    import httpx
    from sqlalchemy import text

    from src.core.db import session_scope

    port: int = _free_port()
    log_path: Path = Path(tempfile.gettempdir()) / f"verify_uvicorn_{port}.log"
    log_file = log_path.open("w", encoding="utf-8")
    proc: subprocess.Popen[bytes] = subprocess.Popen(
        [
            sys.executable,
            "-X",
            "utf8",
            str(PROJECT_ROOT / "main.py"),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=str(PROJECT_ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=_child_env(),
    )

    order_no: str | None = None
    warehouse: str = "苏州中心仓"
    stock_before: int | None = None
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=120.0) as client:
            health: dict[str, Any] | None = _wait_health(client)
            if health is None:
                tail: str = log_path.read_text(encoding="utf-8", errors="replace")[-1500:]
                return FAIL, f"服务未在 90s 内就绪 | 日志尾部：{tail}", None
            if health.get("status") != "ok":
                return FAIL, f"/health 降级：{health}", None

            # ---- 通道 B（无需换件，零写入）----
            t0: float = time.perf_counter()
            webhook = client.post(
                "/api/v1/alert/webhook",
                json={
                    "device_sn": "AGV-SZ-003",
                    "location": "苏州",
                    "error_code": "E-500",
                    "symptom": "急停被误拍下，面板报 E-500，需要复位",
                },
            )
            webhook_ms: float = (time.perf_counter() - t0) * 1000

            # ---- 通道 A（会挂起并建单）----
            t0 = time.perf_counter()
            chat = client.post(
                "/api/v1/diagnose/chat",
                json={"query": "苏州厂区的 AGV-SZ-003 小车夹爪漏气合不上，报 E-201"},
            )
            chat_ms: float = (time.perf_counter() - t0) * 1000

            problems: list[str] = []
            if webhook.status_code != 200 or webhook.json().get("status") != "COMPLETED":
                problems.append(f"通道B 异常：{webhook.status_code} {webhook.text[:200]}")
            if chat.status_code != 200 or chat.json().get("status") != "WAITING_APPROVAL":
                problems.append(f"通道A 异常：{chat.status_code} {chat.text[:200]}")
            if problems:
                return FAIL, "；".join(problems), None

            chat_body: dict[str, Any] = chat.json()
            order_no = chat_body.get("order_no")
            warehouse = str(chat_body.get("selected_warehouse") or warehouse)

            with session_scope() as db:
                raw = db.execute(
                    text(
                        "select stock_quantity from spare_parts "
                        "where part_name = :p and warehouse_name = :w"
                    ).bindparams(p="型号A电磁阀", w=warehouse)
                ).scalar()
            stock_before = int(raw) if raw is not None else None

            t0 = time.perf_counter()
            approve = client.post(
                "/api/v1/work-orders/approve",
                json={
                    "trace_id": chat_body["trace_id"],
                    "order_no": order_no,
                    "approval_status": "APPROVED",
                    "approver": "张主管",
                },
            )
            approve_ms: float = (time.perf_counter() - t0) * 1000
            approved: dict[str, Any] = approve.json()

            if approve.status_code != 200 or approved.get("dispatch_status") != "DISPATCHED":
                problems.append(f"审批失败：{approve.status_code} {approve.text[:200]}")
            if approved.get("stock_deducted") is not True:
                problems.append("审批通过但未核销库存")

            orders = client.get("/api/v1/work-orders", params={"limit": 3})
            parts = client.get("/api/v1/spare-parts", params={"limit": 5})
            missing = client.post(
                "/api/v1/work-orders/approve",
                json={
                    "trace_id": "TR-VERIFY-NOT-EXIST",
                    "order_no": "WO-000000-0000",
                    "approval_status": "APPROVED",
                },
            )
            if orders.status_code != 200 or parts.status_code != 200:
                problems.append(
                    f"查询接口异常：orders={orders.status_code} parts={parts.status_code}"
                )
            if missing.status_code != 404:
                problems.append(f"未知 trace 应为 404，实际 {missing.status_code}")
            if problems:
                return FAIL, "；".join(problems), None

            status: str = PASS
            detail: str = (
                f"端口 {port} | 通道B {webhook_ms:.0f}ms、通道A 挂起 {chat_ms:.0f}ms、"
                f"审批续跑 {approve_ms:.0f}ms | {order_no} → DISPATCHED"
            )
            if chat_ms > TARGET_E2E_MS:
                status = WARN
                detail += f"（超 {TARGET_E2E_MS / 1000:.0f}s 目标）"
            return status, detail, None
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        log_file.close()

        try:
            log_text: str = log_path.read_text(encoding="utf-8", errors="replace")
            if "Traceback (most recent call last)" in log_text:
                logger.error("服务端日志出现异常堆栈，详见 {}", log_path)
        except Exception:  # noqa: BLE001
            pass

        if order_no:
            try:
                with session_scope() as db:
                    db.execute(
                        text("delete from work_orders where order_no = :order_no").bindparams(
                            order_no=order_no
                        )
                    )
                    if stock_before is not None:
                        db.execute(
                            text(
                                "update spare_parts set stock_quantity = :q, updated_at = now() "
                                "where part_name = :p and warehouse_name = :w"
                            ).bindparams(q=stock_before, p="型号A电磁阀", w=warehouse)
                        )
                logger.info("冒烟数据已清理：工单 {}，库存还原 {}", order_no, stock_before)
            except Exception as exc:  # noqa: BLE001
                logger.error("清理冒烟数据失败（请手工核对）：{}", exc)


# ---------------------------------------------------------------------------
# 步骤 5：数据卫生
# ---------------------------------------------------------------------------
def check_hygiene(baseline: dict[str, Any] | None) -> tuple[str, str, None]:
    """对比验收前后的工单数与库存，确认验收可重复执行、不污染数据。"""
    if not baseline:
        return WARN, "缺少基线快照，跳过数据卫生检查", None

    from sqlalchemy import text

    from src.core.db import session_scope

    with session_scope() as db:
        orders_now: int = int(db.execute(text("select count(*) from work_orders")).scalar() or 0)
        stock_now: dict[str, int] = {
            str(row["warehouse_name"]): int(row["stock_quantity"])
            for row in db.execute(
                text(
                    "select warehouse_name, stock_quantity from spare_parts "
                    "where part_name = :part_name"
                ).bindparams(part_name="型号A电磁阀")
            )
            .mappings()
            .all()
        }

    expected_orders: int = int(baseline["counts"]["work_orders"])
    expected_stock: dict[str, int] = dict(baseline["stock"])
    problems: list[str] = []
    if orders_now != expected_orders:
        problems.append(f"工单数 {expected_orders}→{orders_now}")
    if stock_now != expected_stock:
        problems.append(f"库存 {expected_stock}→{stock_now}")
    if problems:
        return FAIL, f"数据未回到基线：{'；'.join(problems)}", None
    return PASS, f"工单 {orders_now} 行、库存 {stock_now} 与基线一致", None


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------
def _print_matrix(started_at: datetime) -> bool:
    """打印结果矩阵。

    Returns:
        bool: 是否全部通过（无 FAIL）。
    """
    width: int = 100
    print()
    print("=" * width)
    print(f"Service Copilot 验收报告   {started_at:%Y-%m-%d %H:%M:%S}")
    print("-" * width)
    if _ENV_INFO:
        print(
            f"环境：Python {_ENV_INFO.get('python', '?')} | "
            f"langgraph {_ENV_INFO.get('langgraph', '?')} | "
            f"fastapi {_ENV_INFO.get('fastapi', '?')} | "
            f"sqlalchemy {_ENV_INFO.get('sqlalchemy', '?')} | "
            f"qdrant-client {_ENV_INFO.get('qdrant-client', '?')}"
        )
        print(f"模型：{_ENV_INFO.get('deepseek', '?')}")
    print("=" * width)
    print(f"{'步骤 / 检查项':<40}{'状态':<8}{'耗时':>8}  明细")
    print("-" * width)
    for step, name, status, detail, seconds in _RESULTS:
        label: str = f"{step} {name}"
        print(f"{label[:38]:<40}{status:<8}{seconds:>7.1f}s  {detail}")
    print("-" * width)
    passed: int = sum(1 for _, _, status, _, _ in _RESULTS if status == PASS)
    warned: int = sum(1 for _, _, status, _, _ in _RESULTS if status == WARN)
    failed: int = sum(1 for _, _, status, _, _ in _RESULTS if status == FAIL)
    print(f"共 {len(_RESULTS)} 项：PASS {passed} / WARN {warned} / FAIL {failed}")
    if failed:
        print("结论：✗ 未通过，请按上表 FAIL 行逐项排查")
    elif warned:
        print("结论：✓ 功能可用（存在降级项，见 WARN 行）")
    else:
        print("结论：✓ 全链路跑通")
    print("=" * width)
    print()
    return failed == 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Service Copilot 一键验收")
    parser.add_argument("--skip-suites", action="store_true", help="跳过三套功能测试")
    parser.add_argument("--skip-live", action="store_true", help="跳过真实服务冒烟")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """执行验收流水线。

    Returns:
        int: 0 通过（含 WARN），1 存在失败，2 环境不满足。
    """
    args: argparse.Namespace = parse_args(argv)
    started_at: datetime = datetime.now()
    logger.info("开始验收：{}", PROJECT_ROOT)

    runtime_status, _ = _timed("1.预检", "运行时与依赖", check_runtime)
    if runtime_status == FAIL:
        _print_matrix(started_at)
        return 2

    data_status, baseline = _timed("2.底座", "MySQL 与基线快照", check_data)
    if data_status == FAIL:
        _print_matrix(started_at)
        return 2

    if not args.skip_suites:
        _timed("3.测试", "工具层 / 状态机 / 接口层", check_suites)
    if not args.skip_live:
        _timed("4.冒烟", "真实服务（uvicorn）", check_live_service)
    _timed("5.卫生", "业务数据回滚核对", lambda: check_hygiene(baseline))

    ok: bool = _print_matrix(started_at)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
