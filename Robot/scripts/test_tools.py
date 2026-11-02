"""工具层联调测试脚本（第二阶段验收用）。

按顺序验证三件事：

1. ``query_device_info("AGV-SZ-003")``      → 设备位于 **苏州**；
2. ``search_repair_manual("夹爪不动 气缸漏气", "E-201")`` → 命中 **型号A电磁阀**；
3. ``query_spare_part_stock("型号A电磁阀", "苏州")``      → 首选 **苏州中心仓**，库存 3。

运行::

    python scripts/test_tools.py                  # 跑 1~3
    python scripts/test_tools.py --with-order     # 额外验证工单草稿落库（验证后自动删除）
    python scripts/test_tools.py --with-order --keep-order   # 保留生成的工单

说明
----
* 脚本开头会自动检查 Qdrant 集合是否存在：缺失时用**离线伪向量**自动引导灌库
  （若已配置真实 ``EMBEDDING_API_KEY`` 则改用云端 Embedding 重建），保证测试不卡死；
* 所有工具调用都在同一个 ``trace_context`` 内执行，日志中的 trace_id 全程一致。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 保证从任意工作目录执行都能 import 到 src 包
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qdrant_client import QdrantClient  # noqa: E402
from sqlalchemy import text  # noqa: E402

from src.core.config import MANUAL_COLLECTION_NAME, settings  # noqa: E402
from src.core.db import session_scope  # noqa: E402
from src.core.logger import logger, trace_context  # noqa: E402
from src.tools.rag_tool import close_qdrant_client, search_repair_manual  # noqa: E402
from src.tools.sql_tool import (  # noqa: E402
    create_work_order_draft,
    query_device_info,
    query_spare_part_stock,
)

#: 测试用例使用的固定设备与备件
TEST_DEVICE_SN: str = "AGV-SZ-003"
TEST_ERROR_CODE: str = "E-201"
TEST_SYMPTOM: str = "夹爪不动，气缸有持续漏气声，屏幕报错 E-201"
TEST_PART_NAME: str = "型号A电磁阀"
TEST_LOCATION: str = "苏州"
EXPECTED_WAREHOUSE: str = "苏州中心仓"
EXPECTED_STOCK_QTY: int = 3


# ---------------------------------------------------------------------------
# 通用辅助
# ---------------------------------------------------------------------------
def _invoke(tool_obj: Any, payload: dict[str, Any]) -> Any:
    """统一调用入口：兼容返回原始对象或被序列化为 JSON 字符串的情况。

    Args:
        tool_obj: ``@tool`` 装饰后的工具对象。
        payload: 工具入参字典。

    Returns:
        Any: 解析后的结果（dict / list / str）。
    """
    result: Any = tool_obj.invoke(payload)
    if isinstance(result, str):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return result
    return result


def _collection_ready() -> bool:
    """检查本地 Qdrant 集合是否已创建。"""
    client = QdrantClient(path=str(settings.qdrant_local_abspath))
    try:
        return bool(client.collection_exists(collection_name=MANUAL_COLLECTION_NAME))
    finally:
        client.close()


def ensure_knowledge_base() -> None:
    """知识库缺失时自动引导灌库，避免测试因缺少集合而失败。

    已配置真实 Embedding Key → 走云端向量；否则走 ``--offline`` 伪向量兜底。
    """
    if _collection_ready():
        logger.info("知识库集合 {} 已存在，跳过引导", MANUAL_COLLECTION_NAME)
        return

    use_offline: bool = not settings.has_embedding_key
    logger.warning(
        "未发现知识库集合 {}，正在自动初始化（模式：{}）……",
        MANUAL_COLLECTION_NAME,
        "离线伪向量兜底" if use_offline else "云端 Embedding",
    )
    cmd: list[str] = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "init_qdrant_data.py"),
        "--recreate",
    ]
    if use_offline:
        cmd.append("--offline")
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)


def _report(results: list[tuple[str, bool, str]]) -> int:
    """打印汇总表并返回进程退出码。"""
    logger.info("=" * 78)
    logger.info("测试汇总")
    logger.info("-" * 78)
    for name, ok, detail in results:
        if ok:
            logger.success("✓ {:<34} {}", name, detail)
        else:
            logger.error("✗ {:<34} {}", name, detail)
    passed: int = sum(1 for _, ok, _ in results if ok)
    logger.info("-" * 78)
    logger.info("通过 {}/{}", passed, len(results))
    logger.info("=" * 78)
    return 0 if passed == len(results) else 1


# ---------------------------------------------------------------------------
# 测试用例
# ---------------------------------------------------------------------------
def test_device_info() -> tuple[bool, str]:
    """用例 1：查询设备位置信息，预期苏州。"""
    logger.info("【用例 1】查询设备 {} 的档案信息", TEST_DEVICE_SN)
    started: float = time.perf_counter()
    device: dict[str, Any] = _invoke(query_device_info, {"device_sn": TEST_DEVICE_SN})
    elapsed_ms: float = (time.perf_counter() - started) * 1000

    if not isinstance(device, dict) or not device.get("found"):
        return False, f"未查到设备：{device}"
    logger.info(
        "用例 1 返回：device_sn={} model={} location={} workshop={} status={}（{:.1f}ms）",
        device.get("device_sn"),
        device.get("model_name"),
        device.get("location"),
        device.get("workshop"),
        device.get("status"),
        elapsed_ms,
    )
    ok: bool = device.get("location") == TEST_LOCATION
    return ok, f"location={device.get('location')}（期望 {TEST_LOCATION}）"


def test_rag_manual() -> tuple[bool, str]:
    """用例 2：检索 E-201 手册，预期命中型号A电磁阀。"""
    logger.info("【用例 2】检索故障码 {} 的维修手册（query={!r}）", TEST_ERROR_CODE, "夹爪不动 气缸漏气")
    started: float = time.perf_counter()
    manual: str = _invoke(
        search_repair_manual, {"query": "夹爪不动 气缸漏气", "error_code": TEST_ERROR_CODE}
    )
    elapsed_ms: float = (time.perf_counter() - started) * 1000

    logger.info("用例 2 返回原文（{:.1f}ms）：\n{}", elapsed_ms, manual)
    hit_code: bool = TEST_ERROR_CODE in manual
    hit_part: bool = TEST_PART_NAME in manual
    ok: bool = hit_code and hit_part
    return ok, (
        f"命中故障码={hit_code} 命中备件={hit_part}（期望 {TEST_ERROR_CODE} / {TEST_PART_NAME}）"
    )


def test_spare_part_stock() -> tuple[bool, str]:
    """用例 3：查询苏州库存，预期苏州中心仓 3 件且排首位。"""
    logger.info(
        "【用例 3】查询备件 {!r} 在 {!r} 的库存", TEST_PART_NAME, TEST_LOCATION
    )
    started: float = time.perf_counter()
    rows: Any = _invoke(
        query_spare_part_stock,
        {"part_name": TEST_PART_NAME, "preferred_location": TEST_LOCATION},
    )
    elapsed_ms: float = (time.perf_counter() - started) * 1000

    if not isinstance(rows, list) or not rows:
        return False, f"未查到库存：{rows}"

    for row in rows:
        logger.info(
            "用例 3 库存行：{} | {} | 库存={} | 货位={} | 就近={}",
            row.get("warehouse_name"),
            row.get("warehouse_location"),
            row.get("stock_quantity"),
            row.get("bin_location"),
            row.get("is_preferred_location"),
        )

    top: dict[str, Any] = rows[0]
    ok_top: bool = top.get("warehouse_name") == EXPECTED_WAREHOUSE
    ok_qty: bool = int(top.get("stock_quantity") or 0) == EXPECTED_STOCK_QTY
    ok: bool = ok_top and ok_qty
    return ok, (
        f"首选仓={top.get('warehouse_name')}/{top.get('stock_quantity')} 件"
        f"（期望 {EXPECTED_WAREHOUSE}/{EXPECTED_STOCK_QTY} 件，耗时 {elapsed_ms:.1f}ms）"
    )


def test_create_order(keep: bool = False) -> tuple[bool, str]:
    """用例 4（可选）：工单草稿落库，默认验证后删除，避免污染业务数据。"""
    logger.info("【用例 4】写入工单草稿（验证审批/派工初始状态）")
    started: float = time.perf_counter()
    draft: dict[str, Any] = _invoke(
        create_work_order_draft,
        {
            "device_sn": TEST_DEVICE_SN,
            "error_code": TEST_ERROR_CODE,
            "symptom": TEST_SYMPTOM,
            "suggested_part": TEST_PART_NAME,
            "allocated_warehouse": EXPECTED_WAREHOUSE,
            "trace_id": "",
        },
    )
    elapsed_ms: float = (time.perf_counter() - started) * 1000

    if not isinstance(draft, dict) or not draft.get("success"):
        return False, f"工单写入失败：{draft}"

    order_no: str = str(draft.get("order_no"))
    logger.info(
        "用例 4 工单已创建：order_no={} status={}/{} trace_id={}（{:.1f}ms）",
        order_no,
        draft.get("approval_status"),
        draft.get("dispatch_status"),
        draft.get("trace_id"),
        elapsed_ms,
    )

    ok: bool = (
        draft.get("approval_status") == "PENDING"
        and draft.get("dispatch_status") == "INIT"
    )

    if keep:
        logger.warning("已保留测试工单 {}，如需清理请手动删除该行", order_no)
    else:
        # 仅测试用途：清理本次写入的草稿，避免反复执行污染 work_orders
        with session_scope() as db:
            db.execute(
                text("delete from work_orders where order_no = :order_no").bindparams(
                    order_no=order_no
                )
            )
        logger.info("测试工单 {} 已清理", order_no)

    return ok, f"order_no={order_no} approval={draft.get('approval_status')}/dispatch={draft.get('dispatch_status')}"


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Service Copilot 工具层联调测试")
    parser.add_argument(
        "--with-order",
        action="store_true",
        help="额外执行工单草稿落库用例（默认写完即删）",
    )
    parser.add_argument(
        "--keep-order",
        action="store_true",
        help="配合 --with-order 使用，保留生成的测试工单不删除",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """执行全部测试用例。

    Returns:
        int: 0 表示全部通过，1 表示存在失败。
    """
    args: argparse.Namespace = parse_args(argv)

    with trace_context() as tid:
        logger.info("=" * 78)
        logger.info("Service Copilot 工具层联调测试 | trace_id={}", tid)
        logger.info("项目根目录 : {}", PROJECT_ROOT)
        logger.info("MySQL      : {}", settings.mysql_url_masked)
        logger.info("Qdrant 目录 : {}", settings.qdrant_local_abspath)
        logger.info(
            "Embedding  : {}（Key {}）",
            settings.EMBEDDING_MODEL,
            "已配置" if settings.has_embedding_key else "未配置 → 走关键字兜底",
        )
        logger.info("=" * 78)

        ensure_knowledge_base()

        results: list[tuple[str, bool, str]] = []
        for name, case in (
            (f"1. 设备档案 {TEST_DEVICE_SN}", test_device_info),
            (f"2. 手册检索 {TEST_ERROR_CODE}", test_rag_manual),
            (f"3. 库存查询 {TEST_PART_NAME}", test_spare_part_stock),
        ):
            try:
                ok, detail = case()
            except Exception as exc:  # noqa: BLE001 - 测试脚本需要汇总所有异常
                logger.exception("用例 {} 抛出异常", name)
                ok, detail = False, f"异常：{exc}"
            results.append((name, ok, detail))

        if args.with_order:
            try:
                ok, detail = test_create_order(keep=args.keep_order)
            except Exception as exc:  # noqa: BLE001
                logger.exception("用例 4 抛出异常")
                ok, detail = False, f"异常：{exc}"
            results.append(("4. 工单草稿落库", ok, detail))

        close_qdrant_client()
        return _report(results)


if __name__ == "__main__":
    raise SystemExit(main())
