"""模拟数据生成脚本：设备档案 / 备件分仓库存 / 历史维修工单。

为什么需要这个脚本
------------------
``sql/init_schema.sql`` 只建空表，后端一启动虽然 ``/health`` 是 ok，但所有业务接口
都会返回空结果；而「环境可复现」要求同一份脚本在任何机器上都能刷出结构一致、
且业务语义自洽的演示数据。本脚本用 Faker 生成接近真实业务形态的数据，
并在结尾对 7 条业务约束逐项断言，避免出现「数据刷进去了但跑不通流程」的假成功。

数据口径（默认参数）
--------------------
============================  ==========================================
表                            生成量
============================  ==========================================
``robot_devices``             100 台，分布苏州 / 上海 / 深圳 / 广州 / 成都 / 武汉
``spare_parts``               500 **种**备件（504 行：4 种核心易损件各占 2 个仓库）
``work_orders``               200 条历史工单，覆盖 APPROVED / REJECTED / PENDING 三态
============================  ==========================================

基线数据（**不可随意改动，多个验收用例依赖它**）
------------------------------------------------
* ``AGV-SZ-003`` 是苏州的设备（``scripts/test_api.py`` / ``test_graph.py`` / ``verify_all.py`` 都用它）；
* ``型号A电磁阀`` 在 ``苏州中心仓`` 库存 3 件、在 ``上海总仓`` 库存 15 件
  （``scripts/test_tools.py`` 断言首选仓为苏州中心仓且库存恰为 3）；
* ``400W交流伺服电机`` / ``360度安全激光雷达`` / ``绝对值多圈编码器`` 与知识库里的
  E-104 / E-305 / E-402 排障章节对应，必须存在于库存表。

自洽约束（脚本结尾逐条断言，共 12 项，任一不满足则退出码非 0）
--------------------------------------------------------------
1. 工单 ``device_sn`` 必须存在于设备台账；
2. 需换件的工单，``suggested_part`` 必须存在于库存表（``spare_parts.part_name``）；
3. 工单的 ``allocated_warehouse`` 必须是该备件**确实有货**的仓库（出库仓不能凭空捏造）；
4. ``approval_status`` 与 ``dispatch_status`` 必须成对：
   ``PENDING/INIT``、``APPROVED/DISPATCHED``、``REJECTED/CLOSED``；
5. 只有非 ``PENDING`` 的工单才有 ``approver``；
6. ``created_at <= updated_at``，且工单时间落在**历史月份**——当月工单号段留给运行时
   ``create_work_order_draft`` 从 ``WO-YYYYMM-0001`` 开始自增，避免撞号；
7. 基线库存与设备与上表一致。

除上述 7 条语义约束外，还会核对 3 张表的行数/种类数是否与命令行参数一致，
因此日志里会打印「全部 12 项自洽校验通过」。

运行
----
在项目根目录执行（会先清空 3 张业务表，属于破坏性操作）::

    python scripts/seed_mock_data.py                    # 默认 100 / 500 / 200
    python scripts/seed_mock_data.py --orders 50        # 只刷 50 条工单
    python scripts/seed_mock_data.py --seed 42          # 换一个随机种子
    python scripts/seed_mock_data.py --skip-verify      # 跳过结尾断言

约定：同一 ``--seed`` 必然刷出同一份数据（``random`` 与 ``Faker`` 双种子固定），
便于复现某个 bug 或对比两次验收的性能数据。
"""

from __future__ import annotations

import argparse
import random
import sys
import uuid
from collections import Counter
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# 保证脚本在任意工作目录下执行时都能 import 到 src 包（与 init_qdrant_data.py 一致）
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from faker import Faker
except ImportError:  # pragma: no cover - 依赖缺失时给出可执行的修复建议
    print(
        "缺少依赖 Faker，请先安装：\n"
        "    uv pip install --python .venv\\Scripts\\python.exe 'Faker>=25.0.0'\n"
        "    # 或：pip install 'Faker>=25.0.0'",
        file=sys.stderr,
    )
    raise SystemExit(2)

from sqlalchemy import text  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from src.core.db import session_scope, verify_mysql_connection  # noqa: E402
from src.core.logger import logger  # noqa: E402

__all__ = ["main", "build_devices", "build_parts", "build_work_orders"]

# ===========================================================================
# 一、常量：城市 / 型号 / 仓库 / 备件家族
# ===========================================================================
#: 城市 → 设备 SN 中的厂区代码（SN 形态遵循 ``[A-Z]{2,6}-[A-Z0-9]{1,6}-\\d{2,6}``，
#: 与 src/agents/nodes.py::_SN_PATTERN 保持一致，否则正则抽取会漏掉设备号）
_CITY_SN_CODE: dict[str, str] = {
    "苏州": "SZ",
    "上海": "SH",
    "深圳": "SZX",
    "广州": "GZ",
    "成都": "CD",
    "武汉": "WH",
}

#: 各城市的厂区/行政区域（用于拼接车间名，保证「城市 ↔ 区域」不串味）
_CITY_DISTRICTS: dict[str, tuple[str, ...]] = {
    "苏州": ("工业园区", "吴中区", "高新区", "相城区"),
    "上海": ("临港新片区", "松江区", "嘉定区", "金山区"),
    "深圳": ("宝安区", "龙岗区", "坪山区"),
    "广州": ("南沙区", "黄埔区", "番禺区"),
    "成都": ("双流区", "龙泉驿区", "温江区"),
    "武汉": ("江夏区", "汉阳区", "东西湖区"),
}

#: 城市权重（展开成 100 个样本池；默认 100 台时苏州会略多一台，因为基线设备固定为苏州）
_CITY_WEIGHTS: dict[str, int] = {"苏州": 26, "上海": 20, "广州": 14, "成都": 14, "武汉": 14, "深圳": 12}

#: 机器人型号 → (品类, SN 前缀)。
#: 型号与品类**必须对应**（旧数据里存在「L-500 搬运小车被标成机械臂」的脏数据，
#: 本脚本按型号派生品类，保证业务自洽）。
_MODELS: tuple[tuple[str, str, str], ...] = (
    ("L-500搬运小车", "AGV", "AGV"),
    ("L-1000重型潜伏车", "AGV", "AGV"),
    ("M-6R六轴工业机械臂", "机械臂", "ARM"),
    ("SCARA四轴高速臂", "机械臂", "ARM"),
    ("H-100复合机器人", "复合机器人", "AMR"),
)

#: 设备状态分布（权重）：绝大多数正常，少量告警与维护中
_DEVICE_STATUS_CHOICES: tuple[str, ...] = ("NORMAL", "ERROR", "MAINTAIN")
_DEVICE_STATUS_WEIGHTS: tuple[int, ...] = (88, 8, 4)

#: 车间名模板（``{n}`` 由 Faker 填充序号）
_WORKSHOP_TEMPLATES: tuple[str, ...] = (
    "{city}{district}第{n}智能制造车间",
    "{city}{district}{kind}车间",
    "{city}{district}洁净室无尘仓",
    "{city}{district}立体库{n}号巷道",
)
_WORKSHOP_KINDS: tuple[str, ...] = ("装配", "焊装", "喷涂", "总装", "机加工", "检测")

#: 仓库定义：(仓库名, 所在城市, 货位前缀)。
#: 苏州只有裸货位（如 ``A-02-04``），其余大区仓带城市前缀（如 ``SH-B-11``），
#: 与 plan.md 第四章的多级仓（本地仓 / 大区中心仓）设计一致。
_WAREHOUSES: tuple[tuple[str, str, str], ...] = (
    ("苏州中心仓", "苏州", ""),
    ("上海总仓", "上海", "SH-"),
    ("广州华南仓", "广州", "GZ-"),
    ("武汉华中仓", "武汉", "WH-"),
    ("成都西南仓", "成都", "CD-"),
)

#: 货位分区字母
_BIN_ZONES: str = "ABCD"

# ---------------------------------------------------------------------------
# 基线数据：被测试脚本硬编码引用，改动前请先全仓搜索
# ---------------------------------------------------------------------------
#: 基线设备：苏州的 AGV-SZ-003（test_api / test_graph / verify_all 的固定用例设备）
_BASELINE_DEVICE: dict[str, Any] = {
    "device_sn": "AGV-SZ-003",
    "model_name": "L-500搬运小车",
    "category": "AGV",
    "location": "苏州",
    "workshop": "苏州工业园区第3智能制造车间",
    "status": "NORMAL",
}

#: 核心易损件：每种备件**固定分布在两个仓库**，既满足基线断言，
#: 又能让「多仓就近调拨」这条业务分支在演示时有真实数据可挑。
#: 元组含义：物料编码 / 品名 / 规格 / 仓库 / 库存 / 单价
_CORE_PARTS: tuple[tuple[str, str, str, str, int, str], ...] = (
    ("P-VALVE-0001", "型号A电磁阀", "SMC-SY5120-5G-C6", "苏州中心仓", 3, "450.00"),
    ("P-VALVE-0002", "型号A电磁阀", "SMC-SY5120-5G-C6", "上海总仓", 15, "450.00"),
    ("P-MOTOR-0003", "400W交流伺服电机", "PANASONIC-MSMD042", "苏州中心仓", 2, "2100.00"),
    ("P-MOTOR-0004", "400W交流伺服电机", "PANASONIC-MSMD042", "上海总仓", 9, "2100.00"),
    ("P-LIDAR-0005", "360度安全激光雷达", "SICK-S300-PRO", "苏州中心仓", 4, "8800.00"),
    ("P-LIDAR-0006", "360度安全激光雷达", "SICK-S300-PRO", "广州华南仓", 6, "8800.00"),
    ("P-ENCOD-0007", "绝对值多圈编码器", "HEIDENHAIN-ROQ425", "武汉华中仓", 5, "1650.00"),
    ("P-ENCOD-0008", "绝对值多圈编码器", "HEIDENHAIN-ROQ425", "成都西南仓", 3, "1650.00"),
)

#: 故障码 → (对应备件品名, 权重)。E-500 是安全回路急停，复位即可，**无需备件**，
#: 因此备件为 ``None``，用来覆盖状态机「不查库存、直接出报告」的分支。
_ERROR_CODE_TO_PART: dict[str, tuple[str | None, int]] = {
    "E-201": ("型号A电磁阀", 30),
    "E-104": ("400W交流伺服电机", 22),
    "E-305": ("360度安全激光雷达", 14),
    "E-402": ("绝对值多圈编码器", 14),
    "E-500": (None, 20),
}

#: 故障现象模板（与故障码语义绑定，避免「E-500 却描述成漏气」这种不自洽）
_SYMPTOM_TEMPLATES: dict[str, tuple[str, ...]] = {
    "E-201": (
        "夹爪无开合动作，贴近气缸可听到持续漏气声，示教器报 E-201",
        "上电后夹爪不动作，气缸进出口接头有明显漏气，面板报 E-201",
        "夹爪合不拢且动作无力，气路压力正常，屏幕报 E-201",
    ),
    "E-104": (
        "行走中突然停机，红色故障灯长亮，日志报 E-104 伺服过载",
        "小车加速段抖动后憋停，驱动器报 E-104，电机外壳烫手",
        "满载爬坡时失去动力，控制面板报 E-104 伺服过载保护",
    ),
    "E-305": (
        "安全激光雷达频繁误报，车体在空旷通道无规律急停",
        "避障区域提示重新标定，面板报 E-305 雷达通讯异常",
        "运行中不停触发避障停机，检查发现雷达视窗有积灰",
    ),
    "E-402": (
        "定位漂移，回原点后坐标偏差明显，报 E-402 编码器异常",
        "关节零点丢失，多次回零失败，面板报 E-402",
        "行走里程数据跳变，编码器反馈值不连续，报 E-402",
    ),
    "E-500": (
        "急停按钮被人员误拍下，控制面板报 E-500，需要现场复位",
        "安全门被打开导致安全回路断开，整车停机报 E-500",
        "光栅被物料遮挡触发急停保护，面板报 E-500",
    ),
}

#: 工单状态组合：(审批状态, 派单状态, 权重)
_ORDER_STATUS_CHOICES: tuple[tuple[str, str, int], ...] = (
    ("PENDING", "INIT", 34),
    ("APPROVED", "DISPATCHED", 41),
    ("REJECTED", "CLOSED", 25),
)

# ===========================================================================
# 二、SQL 模板（全部具名绑定参数，不做字符串拼接）
# ===========================================================================
_INSERT_DEVICE_SQL: str = (
    "insert into robot_devices (device_sn, model_name, category, location, workshop, status, created_at) "
    "values (:device_sn, :model_name, :category, :location, :workshop, :status, :created_at)"
)

_INSERT_PART_SQL: str = (
    "insert into spare_parts (part_code, part_name, spec_model, stock_quantity, warehouse_name, "
    "warehouse_location, bin_location, unit_price, updated_at) "
    "values (:part_code, :part_name, :spec_model, :stock_quantity, :warehouse_name, "
    ":warehouse_location, :bin_location, :unit_price, :updated_at)"
)

_INSERT_ORDER_SQL: str = (
    "insert into work_orders (order_no, device_sn, error_code, symptom, suggested_part, "
    "allocated_warehouse, approval_status, approver, dispatch_status, trace_id, created_at, updated_at) "
    "values (:order_no, :device_sn, :error_code, :symptom, :suggested_part, "
    ":allocated_warehouse, :approval_status, :approver, :dispatch_status, :trace_id, :created_at, :updated_at)"
)

#: 按依赖逆序清空：工单 → 库存 → 设备
_DELETE_SQL: tuple[str, ...] = (
    "delete from work_orders",
    "delete from spare_parts",
    "delete from robot_devices",
)

# ===========================================================================
# 三、数据构造
# ===========================================================================
def _city_pool(count: int) -> list[str]:
    """按权重展开城市池并补齐到 ``count`` 个。

    Args:
        count: 需要的设备台数。

    Returns:
        list[str]: 长度为 ``count`` 的城市序列（调用方负责 shuffle）。
    """
    weighted: list[str] = [city for city, weight in _CITY_WEIGHTS.items() for _ in range(weight)]
    return [weighted[i % len(weighted)] for i in range(count)]


def _make_workshop(city: str, faker: Faker) -> str:
    """拼一个「城市 + 区域 + 车间形态」的车间名。

    Args:
        city: 城市名，需在 ``_CITY_DISTRICTS`` 中。
        faker: 已固定种子的 Faker 实例。

    Returns:
        str: 形如 ``苏州工业园区第3智能制造车间`` 的车间名。
    """
    district: str = faker.random_element(_CITY_DISTRICTS[city])
    template: str = faker.random_element(_WORKSHOP_TEMPLATES)
    return template.format(
        city=city,
        district=district,
        kind=faker.random_element(_WORKSHOP_KINDS),
        n=faker.random_int(min=1, max=12),
    )


def build_devices(count: int, faker: Faker, rng: random.Random) -> list[dict[str, Any]]:
    """构造设备档案列表（含基线设备 AGV-SZ-003）。

    Args:
        count: 设备总数（≥1）。
        faker: 已固定种子的 Faker 实例。
        rng: 已固定种子的随机数发生器。

    Returns:
        list[dict]: 可直接批量插入 ``robot_devices`` 的行字典。
    """
    now: datetime = datetime.now()
    pool: list[str] = _city_pool(count)
    rng.shuffle(pool)

    # 基线设备必须最先存在：后面的 SN 生成要避开它
    devices: list[dict[str, Any]] = [
        {**_BASELINE_DEVICE, "created_at": now - timedelta(days=rng.randint(120, 700))}
    ]
    used_sn: set[str] = {_BASELINE_DEVICE["device_sn"]}
    sequence: Counter[tuple[str, str]] = Counter()

    for index in range(count - 1):
        model_name, category, sn_prefix = _MODELS[index % len(_MODELS)]
        city: str = pool[index]
        city_code: str = _CITY_SN_CODE[city]

        # 同一 (前缀, 城市码) 内顺序递增，得到 AGV-SZ-001 / ARM-SH-012 这样的 SN
        key: tuple[str, str] = (sn_prefix, city_code)
        while True:
            sequence[key] += 1
            device_sn: str = f"{sn_prefix}-{city_code}-{sequence[key]:03d}"
            if device_sn not in used_sn:
                break
        used_sn.add(device_sn)

        devices.append(
            {
                "device_sn": device_sn,
                "model_name": model_name,
                "category": category,
                "location": city,
                "workshop": _make_workshop(city, faker),
                "status": rng.choices(_DEVICE_STATUS_CHOICES, weights=_DEVICE_STATUS_WEIGHTS, k=1)[0],
                "created_at": now - timedelta(days=rng.randint(1, 730), hours=rng.randint(0, 23)),
            }
        )

    logger.info(
        "[seed] 设备构造完成 | {} 台 | 分布={}",
        len(devices),
        dict(Counter(device["location"] for device in devices)),
    )
    return devices


# ---------------------------------------------------------------------------
# 备件：4 个核心易损件固定 + 其余按「家族 × 规格」批量展开
# ---------------------------------------------------------------------------
def _name_valve(k: int) -> str:
    """电磁阀家族第 ``k`` 个变体名（k=0 即基线品名 ``型号A电磁阀``，会被核心件占用而跳过）。"""
    letter: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[k % 26]
    suffix: str = "" if k < 26 else f"-{k // 26 + 1}代"
    return f"型号{letter}电磁阀{suffix}"


def _name_motor(k: int) -> str:
    """交流伺服电机家族第 ``k`` 个变体名（k=1 即基线品名 ``400W交流伺服电机``）。"""
    powers: tuple[int, ...] = (200, 400, 750, 1000, 1500, 2000, 50, 100, 3000, 5000)
    suffix: str = "" if k < 10 else f"-{k // 10 + 1}号"
    return f"{powers[k % 10]}W交流伺服电机{suffix}"


def _pool_builder(desc: str, specs: tuple[str, ...]) -> Callable[[int], str]:
    """为「描述 + 规格候选」型家族生成取名函数。

    命名规则：``{描述}-{规格}``；规格用尽后追加 ``-N批次``，保证同一家族内名称唯一
    （与旧数据里 ``超声波避障传感器-批次7`` 的写法保持一致）。

    Args:
        desc: 品名描述，例如 ``谐波减速器``。
        specs: 规格候选，例如 ``("14型", "17型")``。

    Returns:
        Callable[[int], str]: 输入家族内第 k 个变体序号，返回唯一的品名。
    """

    def build(k: int) -> str:
        spec: str = specs[k % len(specs)]
        batch: str = "" if k < len(specs) else f"-{k // len(specs) + 1}批次"
        return f"{desc}-{spec}{batch}"

    return build


#: 其余备件家族：(物料编码缩写, 品名描述, 规格候选, 供应商前缀, 参考单价)
_PART_FAMILIES: tuple[tuple[str, str, tuple[str, ...], str, str], ...] = (
    ("LIDAR", "360度安全激光雷达", ("16线", "32线", "64线", "128线"), "SICK-S300", "8800.00"),
    ("ENCOD", "绝对值多圈编码器", ("17位", "19位", "21位", "25位"), "HEIDENHAIN-ROQ", "1650.00"),
    ("REDUC", "谐波减速器", ("14型", "17型", "20型", "25型", "32型", "40型"), "HARMONIC-CSF", "3200.00"),
    ("RVRED", "RV减速器", ("10E", "20E", "40E", "80E", "160E"), "NABTESCO-RV", "5400.00"),
    ("GUIDE", "直线导轨", ("300mm", "500mm", "800mm", "1200mm", "1500mm"), "HIWIN-HGH", "980.00"),
    ("SCREW", "滚珠丝杠", ("400mm", "600mm", "900mm", "1200mm"), "THK-BNK", "1450.00"),
    ("BELT", "同步带", ("HTD-3M", "HTD-5M", "HTD-8M", "GT2-6mm"), "GATES-HT", "180.00"),
    ("GRIP", "气动夹爪", ("行程20mm", "行程40mm", "行程60mm", "行程80mm"), "SMC-MHL", "1250.00"),
    ("SUCK", "真空吸盘", ("直径20mm", "直径30mm", "直径40mm", "直径60mm"), "SMC-ZP", "95.00"),
    ("CYL", "气缸", ("Φ32x100", "Φ40x150", "Φ50x200", "Φ63x250"), "SMC-CDQ", "620.00"),
    ("FORCE", "六维力传感器", ("量程100N", "量程200N", "量程500N"), "ATI-OMEGA", "46000.00"),
    ("PHOTO", "光电开关", ("漫反射", "对射式", "回归反射"), "OMRON-E3Z", "260.00"),
    ("PROXY", "接近开关", ("M8", "M12", "M18", "M30"), "OMRON-E2E", "150.00"),
    ("SERVODRIVE", "伺服驱动器", ("0.4kW", "0.75kW", "1.5kW", "2.0kW", "3.0kW"), "YASKAWA-SGDV", "4800.00"),
    ("CABLE", "编码器线缆", ("3m", "5m", "10m", "15m"), "IGUS-CF", "320.00"),
    ("CHAIN", "拖链", ("宽度25mm", "宽度38mm", "宽度50mm", "宽度75mm"), "IGUS-E2", "410.00"),
    ("PLANET", "行星减速机", ("减速比5", "减速比10", "减速比20", "减速比50"), "NEUGART-PLE", "2900.00"),
    ("SAFETY", "安全继电器", ("单通道", "双通道", "三通道"), "PILZ-PNOZ", "1680.00"),
    ("BATT", "锂电电池组", ("24V20Ah", "24V40Ah", "48V30Ah", "48V50Ah"), "PANASONIC-LC", "6500.00"),
    ("CHARGER", "充电桩触点模块", ("标准型", "重载型", "防尘型"), "ABB-CP", "1900.00"),
    ("WHEEL", "聚氨酯驱动轮", ("直径125mm", "直径150mm", "直径200mm"), "BLICKLE-PU", "760.00"),
    ("CASTER", "万向脚轮", ("直径75mm", "直径100mm", "直径125mm"), "BLICKLE-L", "230.00"),
    ("CAMERA", "工业相机", ("2MP", "5MP", "12MP", "20MP"), "BASLER-ACA", "5200.00"),
    ("FACE", "末端法兰盘", ("ISO50", "ISO63", "ISO80"), "SCHUNK-FL", "1100.00"),
    ("BRAKE", "伺服抱闸", ("24V", "48V", "110V"), "KEB-BR", "2700.00"),
)

#: 家族轮转表：(物料编码缩写, 参考单价, 供应商前缀, 取名函数)。
#: 电磁阀与伺服电机单独排在最前，保证它们的规格序列与知识库/基线品名对齐。
_NAME_BUILDERS: tuple[tuple[str, str, str, Callable[[int], str]], ...] = (
    (
        "VALVE",
        "450.00",
        "SMC-SY",
        _name_valve,
    ),
    (
        "MOTOR",
        "2100.00",
        "PANASONIC-MSMD",
        _name_motor,
    ),
    *(
        (abbr, price, vendor, _pool_builder(desc, specs))
        for abbr, desc, specs, vendor, price in _PART_FAMILIES
    ),
)


def _make_bin_location(warehouse_prefix: str, faker: Faker) -> str:
    """生成货位号：``A-02-04`` / ``SH-B-11`` 风格（分区-排-位）。"""
    zone: str = faker.random_element(_BIN_ZONES)
    return f"{warehouse_prefix}{zone}-{faker.random_int(min=1, max=20):02d}-{faker.random_int(min=1, max=20):02d}"


def _stock_quantity(rng: random.Random) -> int:
    """按「少量缺货 + 小批应急 + 常备库存」的分布生成库存量。"""
    bucket: int = rng.choices((0, 1, 2), weights=(8, 22, 70), k=1)[0]
    if bucket == 0:
        return 0
    if bucket == 1:
        return rng.randint(1, 5)
    return rng.randint(6, 49)


def _price(base: str, rng: random.Random) -> Decimal:
    """在参考单价上下浮动 ±15%，并保留两位小数。"""
    factor: float = rng.uniform(0.85, 1.15)
    return (Decimal(base) * Decimal(f"{factor:.4f}")).quantize(Decimal("0.01"))


def build_parts(
    part_names: int, faker: Faker, rng: random.Random
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """构造备件库存行。

    Args:
        part_names: 目标准备件**种类数**（≥4，含 4 个核心易损件）。
        faker: 已固定种子的 Faker 实例。
        rng: 已固定种子的随机数发生器。

    Returns:
        tuple[list, list]: ``(库存行列表, 供工单引用的候选池)``。
        候选池只含 ``stock_quantity > 0`` 的行，用于保证工单指派的仓库确实有货。
    """
    now: datetime = datetime.now()
    rows: list[dict[str, Any]] = []
    warehouse_index: dict[str, str] = {name: location for name, location, _ in _WAREHOUSES}

    # ---- 1) 核心易损件：固定编码 / 仓库 / 库存（基线断言依赖）----
    for code, name, spec, warehouse, qty, price in _CORE_PARTS:
        _, _, bin_prefix = next(item for item in _WAREHOUSES if item[0] == warehouse)
        rows.append(
            {
                "part_code": code,
                "part_name": name,
                "spec_model": spec,
                "stock_quantity": qty,
                "warehouse_name": warehouse,
                "warehouse_location": warehouse_index[warehouse],
                "bin_location": _make_bin_location(bin_prefix, faker),
                "unit_price": Decimal(price),
                "updated_at": now - timedelta(days=rng.randint(0, 30)),
            }
        )
    used_names: set[str] = {name for _, name, _, _, _, _ in _CORE_PARTS}
    used_codes: set[str] = {code for code, _, _, _, _, _ in _CORE_PARTS}

    # ---- 2) 其余备件：家族轮转 + 规格展开 ----
    generated: int = 0
    cursor: int = 0
    code_sequence: int = 100  # 避开核心易损件的 0001~0008 号段
    while len(used_names) < part_names:
        abbr, base_price, vendor, build_name = _NAME_BUILDERS[cursor % len(_NAME_BUILDERS)]
        variant: int = cursor // len(_NAME_BUILDERS)
        cursor += 1

        name: str = build_name(variant)
        # 与核心易损件重名（型号A电磁阀 / 400W交流伺服电机 等）时跳过该变体
        if name in used_names:
            continue
        used_names.add(name)

        code_sequence += 1
        part_code: str = f"P-{abbr}-{code_sequence:04d}"
        while part_code in used_codes:  # 理论不会触发，兜底防御
            code_sequence += 1
            part_code = f"P-{abbr}-{code_sequence:04d}"
        used_codes.add(part_code)

        warehouse, location, bin_prefix = _WAREHOUSES[(generated + 2) % len(_WAREHOUSES)]
        generated += 1
        rows.append(
            {
                "part_code": part_code,
                "part_name": name,
                # 规格型号：供应商前缀 + Faker 数字 + 两位字母，形态接近真实物料号
                "spec_model": f"{vendor}-{faker.numerify('####')}{faker.lexify('??').upper()}",
                "stock_quantity": _stock_quantity(rng),
                "warehouse_name": warehouse,
                "warehouse_location": location,
                "bin_location": _make_bin_location(bin_prefix, faker),
                "unit_price": _price(base_price, rng),
                "updated_at": now - timedelta(days=rng.randint(0, 30)),
            }
        )

    # ---- 3) 候选池：只保留有货行，供工单「仓库必须真有货」的自洽约束使用 ----
    candidates: list[dict[str, Any]] = [
        {"part_name": row["part_name"], "warehouse_name": row["warehouse_name"],
         "stock_quantity": row["stock_quantity"]}
        for row in rows
        if row["stock_quantity"] > 0
    ]

    logger.info(
        "[seed] 备件构造完成 | {} 种 / {} 行 | 有货行 {} | 库存分布={}",
        len(used_names),
        len(rows),
        len(candidates),
        dict(Counter(row["warehouse_name"] for row in rows)),
    )
    return rows, candidates


def _month_start(reference: datetime, months_ago: int) -> datetime:
    """返回 ``reference`` 往前推 ``months_ago`` 个月的那个月的 1 号 00:00。"""
    year: int = reference.year
    month: int = reference.month - months_ago
    while month <= 0:
        month += 12
        year -= 1
    return datetime(year, month, 1)


def build_work_orders(
    count: int,
    devices: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    faker: Faker,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """构造历史工单（时间落在过去 12 个月，不占用当月工单号段）。

    Args:
        count: 工单条数。
        devices: 已构造的设备列表（用于自洽地引用 ``device_sn``）。
        candidates: 有货的 (备件, 仓库) 组合池。
        faker: 已固定种子的 Faker 实例。
        rng: 已固定种子的随机数发生器。

    Returns:
        list[dict]: 可直接批量插入 ``work_orders`` 的行字典。
    """
    now: datetime = datetime.now()
    error_codes: list[str] = list(_ERROR_CODE_TO_PART)
    error_weights: list[int] = [weight for _, weight in _ERROR_CODE_TO_PART.values()]

    #: 备件名 → 有货的仓库列表（按库存量做权重，库存多的仓更容易被选中）
    warehouse_pool: dict[str, list[dict[str, Any]]] = {}
    for row in candidates:
        warehouse_pool.setdefault(str(row["part_name"]), []).append(row)

    orders: list[dict[str, Any]] = []
    for _ in range(count):
        device: dict[str, Any] = rng.choice(devices)
        error_code: str = rng.choices(error_codes, weights=error_weights, k=1)[0]
        part_name, _ = _ERROR_CODE_TO_PART[error_code]
        city: str = str(device["location"])

        allocated_warehouse: str | None = None
        if part_name:
            rows = warehouse_pool.get(part_name) or []
            if rows:
                allocated_warehouse = str(
                    rng.choices(
                        rows, weights=[int(row["stock_quantity"]) for row in rows], k=1
                    )[0]["warehouse_name"]
                )

        approval_status, dispatch_status, _ = rng.choices(
            _ORDER_STATUS_CHOICES, weights=[w for _, _, w in _ORDER_STATUS_CHOICES], k=1
        )[0]
        created_at: datetime = faker.date_time_between(
            start_date=_month_start(now, 12), end_date=_month_start(now, 1)
        )
        updated_at: datetime = created_at + timedelta(hours=rng.randint(1, 72))

        orders.append(
            {
                "order_no": "",  # 稍后按「月份 + 月内流水」统一编号
                "device_sn": device["device_sn"],
                "error_code": error_code,
                # 现象模板里带上城市与 SN，看起来像现场真实上报
                "symptom": f"{city}厂区 {device['device_sn']}：{faker.random_element(_SYMPTOM_TEMPLATES[error_code])}",
                "suggested_part": part_name,
                "allocated_warehouse": allocated_warehouse,
                "approval_status": approval_status,
                "approver": None if approval_status == "PENDING" else f"{faker.last_name()}主管",
                "dispatch_status": dispatch_status,
                "trace_id": f"TR-{uuid.uuid4().hex[:12]}",
                "created_at": created_at,
                "updated_at": updated_at,
            }
        )

    # ---- 编号：按时间正序，每月从 0001 起流水，形如 WO-202609-0001 ----
    orders.sort(key=lambda order: order["created_at"])
    monthly: Counter[str] = Counter()
    for order in orders:
        month_key: str = order["created_at"].strftime("%Y%m")
        monthly[month_key] += 1
        order["order_no"] = f"WO-{month_key}-{monthly[month_key]:04d}"

    logger.info(
        "[seed] 工单构造完成 | {} 条 | 月份={} | 状态={}",
        len(orders),
        sorted(monthly),
        dict(Counter(order["approval_status"] for order in orders)),
    )
    return orders


# ===========================================================================
# 四、落库与校验
# ===========================================================================
def _bulk_insert(db: Session, sql: str, rows: list[dict[str, Any]], label: str, chunk: int = 200) -> None:
    """分批执行批量插入。

    Args:
        db: 数据库会话。
        sql: 含具名绑定参数的 INSERT 模板。
        rows: 待插入的行字典列表。
        label: 日志标签。
        chunk: 单批条数（避免超大 SQL 语句超过 max_allowed_packet）。
    """
    for start in range(0, len(rows), chunk):
        batch: list[dict[str, Any]] = rows[start : start + chunk]
        db.execute(text(sql), batch)
        logger.debug("[seed] {} 已写入 {}/{}", label, start + len(batch), len(rows))


def _scalar(db: Session, sql: str, **params: Any) -> int:
    """执行计数类查询并返回整数结果。"""
    value: Any = db.execute(text(sql).bindparams(**params)).scalar()
    return int(value or 0)


def verify_consistency(db: Session, expected: dict[str, int]) -> bool:
    """逐条校验业务自洽约束与数据口径（共 12 项），并打印明细。

    Args:
        db: 数据库会话。
        expected: 期望的行数/种类数，例如 ``{"devices": 100, "part_names": 500, "orders": 200}``。

    Returns:
        bool: 全部通过返回 ``True``。
    """
    checks: list[tuple[str, int, int, str]] = [
        # (检查项, 实际值, 期望值, 失败说明)
        ("设备台账行数", _scalar(db, "select count(*) from robot_devices"), expected["devices"], "与 --devices 不一致"),
        (
            "备件种类数",
            _scalar(db, "select count(distinct part_name) from spare_parts"),
            expected["part_names"],
            "与 --parts 不一致",
        ),
        ("工单行数", _scalar(db, "select count(*) from work_orders"), expected["orders"], "与 --orders 不一致"),
        # 约束 1：工单引用的设备必须存在
        (
            "工单设备SN全部在台账",
            _scalar(
                db,
                "select count(*) from work_orders o "
                "left join robot_devices d on d.device_sn = o.device_sn where d.id is null",
            ),
            0,
            "存在孤儿工单（device_sn 不在设备台账）",
        ),
        # 约束 2：需换件的工单，备件品名必须在库存表中
        (
            "建议备件均在库存表",
            _scalar(
                db,
                "select count(*) from work_orders o where o.suggested_part is not null "
                "and not exists (select 1 from spare_parts p where p.part_name = o.suggested_part)",
            ),
            0,
            "存在引用了不存在备件的工单",
        ),
        # 约束 3：出库仓必须真的存有该备件
        (
            "出库仓确实有该备件",
            _scalar(
                db,
                "select count(*) from work_orders o where o.suggested_part is not null "
                "and not exists (select 1 from spare_parts p where p.part_name = o.suggested_part "
                "and p.warehouse_name = o.allocated_warehouse)",
            ),
            0,
            "存在「出库仓无此备件」的工单",
        ),
        # 约束 4：审批状态与派单状态成对
        (
            "审批/派单状态成对",
            _scalar(
                db,
                "select count(*) from work_orders where "
                "(approval_status = 'PENDING' and dispatch_status <> 'INIT') or "
                "(approval_status = 'APPROVED' and dispatch_status <> 'DISPATCHED') or "
                "(approval_status = 'REJECTED' and dispatch_status <> 'CLOSED')",
            ),
            0,
            "存在非法状态组合",
        ),
        # 约束 5：只有已决工单才有审批人
        (
            "审批人与状态匹配",
            _scalar(
                db,
                "select count(*) from work_orders where "
                "(approval_status = 'PENDING' and approver is not null) or "
                "(approval_status <> 'PENDING' and approver is null)",
            ),
            0,
            "PENDING 工单不该有审批人，已决工单必须有",
        ),
        # 约束 6：时间方向正确
        (
            "created_at 不晚于 updated_at",
            _scalar(db, "select count(*) from work_orders where updated_at < created_at"),
            0,
            "存在时间倒流的工单",
        ),
    ]

    # 约束 6 续：当月工单号段必须为空，留给运行时自增
    current_prefix: str = f"{datetime.now():%Y%m}"
    checks.append(
        (
            "当月工单号段为空",
            _scalar(db, "select count(*) from work_orders where order_no like :pattern", pattern=f"WO-{current_prefix}-%"),
            0,
            f"当月 WO-{current_prefix}-* 已被占用，运行时建单会与之撞号",
        )
    )

    # 约束 7：基线数据（苏州中心仓 3 件 / 上海总仓 15 件）
    baseline_rows: list[tuple[str, int]] = [
        (str(row[0]), int(row[1]))
        for row in db.execute(
            text(
                "select warehouse_name, stock_quantity from spare_parts "
                "where part_name = :part_name"
            ).bindparams(part_name="型号A电磁阀")
        ).all()
    ]
    baseline_expected: dict[str, int] = {"苏州中心仓": 3, "上海总仓": 15}
    actual_stock: dict[str, int] = dict(baseline_rows)
    # 逐仓比对（而非只比对已有行），这样「仓库整行缺失」也会被判为不通过
    baseline_mismatch: int = sum(
        1 for warehouse, qty in baseline_expected.items() if actual_stock.get(warehouse) != qty
    )
    checks.append(
        (
            "型号A电磁阀基线库存",
            baseline_mismatch,
            0,
            f"实际={actual_stock}，期望={baseline_expected}",
        )
    )
    # 基线设备必须存在且位于苏州
    checks.append(
        (
            "基线设备 AGV-SZ-003 位于苏州",
            _scalar(
                db,
                "select count(*) from robot_devices where device_sn = :sn and location = :loc",
                sn="AGV-SZ-003",
                loc="苏州",
            ),
            1,
            "测试用例设备缺失或城市不符",
        )
    )

    failures: list[str] = []
    for name, actual, want, hint in checks:
        if actual == want:
            logger.success("[verify] ✓ {:<22} {}", name, actual)
        else:
            logger.error("[verify] ✗ {:<22} 实际={} 期望={}（{}）", name, actual, want, hint)
            failures.append(name)

    if failures:
        logger.error("[verify] 自洽校验未通过：{}", "、".join(failures))
        return False
    logger.success("[verify] 全部 {} 项自洽校验通过", len(checks))
    return True


def seed(
    devices_count: int,
    part_names: int,
    orders_count: int,
    seed_value: int,
    skip_verify: bool,
) -> int:
    """执行「清空 → 生成 → 落库 → 校验」全过程。

    Returns:
        int: 进程退出码（0 成功，1 校验失败，2 环境不满足）。
    """
    if part_names < len({name for _, name, _, _, _, _ in _CORE_PARTS}):
        logger.error("[seed] --parts 至少为 {}（核心易损件种类数）", len(_CORE_PARTS))
        return 2
    if devices_count < 1 or orders_count < 0:
        logger.error("[seed] 参数非法：devices 需 ≥1，orders 需 ≥0")
        return 2

    if not verify_mysql_connection():
        logger.error("[seed] MySQL 不可用，请先确认服务已启动并执行过 sql/init_schema.sql")
        return 2

    # 双种子固定：random 控制挑选，Faker 控制文本，保证同 seed 同数据
    random.seed(seed_value)
    Faker.seed(seed_value)
    rng: random.Random = random.Random(seed_value)
    faker: Faker = Faker("zh_CN")

    devices: list[dict[str, Any]] = build_devices(devices_count, faker, rng)
    parts, candidates = build_parts(part_names, faker, rng)
    orders: list[dict[str, Any]] = build_work_orders(orders_count, devices, candidates, faker, rng)

    started: datetime = datetime.now()
    with session_scope() as db:
        for sql in _DELETE_SQL:
            db.execute(text(sql))
        logger.info("[seed] 已清空 3 张业务表，开始写入…")

        _bulk_insert(db, _INSERT_DEVICE_SQL, devices, "robot_devices")
        _bulk_insert(db, _INSERT_PART_SQL, parts, "spare_parts")
        _bulk_insert(db, _INSERT_ORDER_SQL, orders, "work_orders")
        db.flush()

        logger.success(
            "[seed] 写入完成 | 设备 {} 行 / 备件 {} 行（{} 种）/ 工单 {} 行 | 耗时 {:.1f}s",
            len(devices),
            len(parts),
            part_names,
            len(orders),
            (datetime.now() - started).total_seconds(),
        )

        if skip_verify:
            logger.warning("[seed] 已按 --skip-verify 跳过自洽校验")
            return 0

        ok: bool = verify_consistency(
            db, {"devices": devices_count, "part_names": part_names, "orders": orders_count}
        )

    next_order_no: str = f"WO-{datetime.now():%Y%m}-0001"
    logger.info("[seed] 运行时新建工单号将从 {} 开始", next_order_no)
    return 0 if ok else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Service Copilot 模拟数据生成（会清空 3 张业务表，请谨慎执行）"
    )
    parser.add_argument("--devices", type=int, default=100, help="设备台数（默认 100）")
    parser.add_argument("--parts", type=int, default=500, help="备件种类数（默认 500）")
    parser.add_argument("--orders", type=int, default=200, help="历史工单条数（默认 200）")
    parser.add_argument("--seed", type=int, default=20261030, help="随机种子（默认 20261030，同种子同数据）")
    parser.add_argument("--skip-verify", action="store_true", help="跳过结尾的 7 项自洽校验")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """命令行入口。

    Returns:
        int: 进程退出码。
    """
    args: argparse.Namespace = parse_args(argv)
    logger.info(
        "[seed] 开始刷数据 | devices={} parts={} orders={} seed={}",
        args.devices,
        args.parts,
        args.orders,
        args.seed,
    )
    return seed(
        devices_count=args.devices,
        part_names=args.parts,
        orders_count=args.orders,
        seed_value=args.seed,
        skip_verify=bool(args.skip_verify),
    )


if __name__ == "__main__":
    raise SystemExit(main())
