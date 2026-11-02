# Service Copilot · 机器人售后排障与备件智能调度智能体

> 一个把「现场报修 → 手册排障 → 多仓备件调度 → 人工审批 → 派工结案」打通成一个闭环的售后中台 Demo。
> 双通道接入 + LangGraph 状态机 + 人在回路（HITL）审批 + 本地向量库 RAG + 安全 Text2SQL。

![Python](https://img.shields.io/badge/Python-3.14+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-状态机-1C3C3C)
![Qdrant](https://img.shields.io/badge/Qdrant-本地文件模式-DC244C)
![MySQL](https://img.shields.io/badge/MySQL-8.0-4479A1?logo=mysql&logoColor=white)

---

## 目录

- [一、项目简介](#一项目简介)
- [二、核心特性](#二核心特性)
- [三、系统架构](#三系统架构)
- [四、技术栈](#四技术栈)
- [五、目录结构](#五目录结构)
- [六、快速开始](#六快速开始)
- [七、接口一览](#七接口一览)
- [八、配置说明](#八配置说明)
- [九、安全设计](#九安全设计)
- [十、测试与验收](#十测试与验收)
- [十一、可视化控制台](#十一可视化控制台)
- [十二、常见问题](#十二常见问题)
- [十三、Roadmap](#十三roadmap)

---

## 一、项目简介

售后工程师在现场排查机器人故障时，通常要在 **维修手册、ERP 库存、工单系统** 之间来回切换：
查故障码要翻手册，判断换什么件靠经验，查库存要问仓库，领用高价备件还要走主管审批。
一次「报修 → 备件到手」动辄几小时，而其中真正需要人判断的只有「批不批」这一步。

**Service Copilot** 把这个过程做成了自动化流水线：

| 传统方式 | 本项目 |
| --- | --- |
| 工程师口头描述故障，人工翻手册 | LLM 抽取设备 SN / 故障码 / 现象 → Qdrant 语义检索原厂 SOP |
| 靠经验判断换哪个件 | 从手册切片中直接产出「建议更换备件」标准名称 |
| 打电话问仓库有没有货 | 多仓就近查库存，自动推荐同城仓并落库 `PENDING` 工单草稿 |
| 高价备件出库走纸质审批 | 工作流 `interrupt()` 挂起，控制台弹审批卡片，一键批准 / 驳回 |
| 审批后手工改库存、手工派工 | `Command(resume=...)` 续跑，自动核销 1 件库存 + 工单置 `DISPATCHED` |

一次完整的报修只会有 **一次人工介入**（审批），其余环节全部交给状态机。

---

## 二、核心特性

| 特性 | 说明 |
| --- | --- |
| **双通道接入** | 通道 A `POST /diagnose/chat` 接工程师口语化报修；通道 B `POST /alert/webhook` 接设备主动上报的 JSON 告警。两条通道汇入**同一个**状态机，出口结果完全一致。 |
| **RAG 排障推理** | 现场描述 → 1024 维向量 → 本地 Qdrant 集合 `robot_manual_sop` 检索 Top-K 手册切片，输出带章节引用的 SOP 与所需备件。 |
| **三级容错检索** | ① 故障码 payload 过滤 + 语义检索 → ② 放开过滤全库召回 + 故障码加权重排 → ③ Embedding 接口不可用时退化为「故障码精确匹配 + 中文 2-gram 重合度」纯本地打分。**没有 Embedding Key 也能跑通全流程。** |
| **HITL 人在回路** | 审批节点调用 `interrupt()` 挂起，完整状态写入 SQLite 检查点；外部凭同一个 `trace_id`（即 LangGraph `thread_id`）恢复续跑。进程重启后挂起工单仍可恢复。 |
| **零字符串拼接 SQL** | 所有 SQL 都是固定模板 + `text()` + `bindparams()`，用户输入永远只是绑定参数；配合 MySQL 只读账号形成**双重阻断**。 |
| **全链路 Trace ID** | `X-Trace-Id` 请求头贯穿 HTTP 层、LangGraph 节点、工具层与 Loguru 日志；`trace_id` 同时就是检查点的 `thread_id`，一次报修的所有日志、工单、审批都能被同一个 ID 串起来。 |
| **审批幂等** | 重复提交同一工单审批不会重复扣库存，返回 `replayed=true`，避免前端连点造成库存错账。 |
| **无需 Docker / Redis / 本地模型** | 向量库用 Qdrant 官方客户端的**本地文件模式**，状态落 SQLite，推理全部走云端 HTTP API。整个项目 `pip install` 即可起。 |
| **自检式测试脚本** | 每套脚本都会自己造数据、自己断言、自己清理现场（工单与库存回到基线），可以反复运行。 |

---

## 三、系统架构

### 3.1 分层视图

```
┌──────────────────────────────────────────────────────────────────────┐
│  接入层  main.py · FastAPI                                           │
│  ┌───────────────────────────┐   ┌──────────────────────────────┐   │
│  │ 通道 A  POST /diagnose/chat│   │ 通道 B  POST /alert/webhook  │   │
│  │ 工程师口语化报修            │   │ 设备 JSON 告警               │   │
│  └─────────────┬─────────────┘   └──────────────┬───────────────┘   │
│                └──────────────┬──────────────────┘                   │
│         TraceIdMiddleware（X-Trace-Id 生成 / 透传 / 回写）           │
│         CORSMiddleware · 静态大屏挂载 · /health 探活                  │
└───────────────────────────────┬──────────────────────────────────────┘
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  编排层  src/agents/  LangGraph 状态机                                │
│   parse_fault → rag_diagnose → stock_check → human_approval          │
│                       └─(无需换件)──────────┐   → dispatch_order      │
│                                             └──→ generate_report → END│
│   检查点：SqliteSaver（checkpoints.sqlite），缺失时回退 MemorySaver   │
└───────┬───────────────────┬──────────────────────┬───────────────────┘
        ▼                   ▼                      ▼
┌───────────────┐   ┌───────────────┐     ┌──────────────────────┐
│  LLM 工厂     │   │  RAG 工具     │     │  SQL 工具（只读）     │
│  src/core/llm │   │ src/tools/    │     │ src/tools/sql_tool   │
│  DeepSeek     │   │ rag_tool.py   │     │ 设备档案 / 多仓库存   │
│  + Embedding  │   │ 三级容错检索   │     │ 工单草稿（PENDING）   │
└───────────────┘   └───────┬───────┘     └──────────┬───────────┘
                            ▼                        ▼
                    ┌───────────────┐        ┌───────────────┐
                    │ Qdrant 本地   │        │  MySQL 8.0    │
                    │ qdrant_local_ │        │ service_      │
                    │ data/（文件） │        │ copilot       │
                    └───────────────┘        └───────────────┘
```

### 3.2 LangGraph 工作流

```
                 START
                   │
                   ▼
      ┌──────────────────────────┐
      │ parse_fault              │  LLM 抽取 SN / 故障码 / 现象
      │                          │  查台账补全所在城市；LLM 失败降级正则
      └────────────┬─────────────┘
                   ▼
      ┌──────────────────────────┐
      │ rag_diagnose             │  Qdrant 手册检索 → SOP + 所需备件
      └────────────┬─────────────┘
                   │
        ┌──────────┴───────────┐
        │ required_part 为空？  │
        │（如 E-500 急停复位）  │
        └──────┬──────────┬────┘
           是  │          │ 否
               │          ▼
               │  ┌──────────────────────────┐
               │  │ stock_check              │ 多仓就近查库存
               │  │                          │ 落库 PENDING 工单草稿
               │  └────────────┬─────────────┘
               │      ┌────────┴────────┐
               │      │ 是否生成工单？   │   全库缺货 → 直接出报告
               │      └────┬───────┬────┘
               │       否  │       │ 是
               │           │       ▼
               │           │  ┌──────────────────────────┐
               │           │  │ human_approval  ◆HITL    │ interrupt() 挂起
               │           │  │ 推送审批载荷，等外部信号  │ 状态写入检查点
               │           │  └────────────┬─────────────┘
               │           │               ▼
               │           │  ┌──────────────────────────┐
               │           │  │ dispatch_order           │ 写审批结论
               │           │  │                          │ 批准→核销库存+DISPATCHED
               │           │  │                          │ 驳回→CLOSED，库存不动
               │           │  └────────────┬─────────────┘
               ▼           ▼               ▼
      ┌──────────────────────────────────────────┐
      │ generate_report   汇总 Markdown 对客报告  │
      └───────────────────┬──────────────────────┘
                          ▼
                         END
```

**为什么用 `interrupt()` 而不是 `compile(interrupt_before=[...])`？**
静态断点只能"停在派工之前"，审批结论还得外部另找地方写回状态；`interrupt()` 能在挂起时
把待审批工单信息（工单号 / 备件 / 仓库 / 当前库存）作为 payload 直接推给审批终端，
恢复时传入的值天然就是审批结论，链路更短、状态更一致。

---

## 四、技术栈

| 层次 | 选型 | 备注 |
| --- | --- | --- |
| Web 框架 | FastAPI + Uvicorn | 自带 OpenAPI / Swagger UI |
| 智能体编排 | LangGraph | `StateGraph` + 条件边 + 检查点 |
| 检查点持久化 | `langgraph-checkpoint-sqlite` | 落 `checkpoints.sqlite`，跨进程可恢复 |
| 大模型 | DeepSeek `deepseek-chat` | 经 `langchain-openai` 走 OpenAI 兼容协议，`temperature=0` |
| 向量模型 | 云端 Embedding（默认 `BAAI/bge-m3`，1024 维） | 同样走 OpenAI 兼容协议 |
| 向量库 | Qdrant 官方客户端**本地文件模式** | 无需启动服务端、无需 Docker |
| 关系库 | MySQL 8.0 + SQLAlchemy 2.x + PyMySQL | 设备 / 备件 / 工单三张表 |
| 配置 | pydantic-settings | 多层 `.env` + 环境变量覆盖 |
| 日志 | Loguru | 三通道（控制台 / app.log / error.log）+ ContextVar 注入 trace_id |
| 前端 | 原生 HTML + Tailwind CDN + 原生 JS 单文件 | 无需 Node 构建 |

> **设计底线：全栈云端推理，零本地模型。**
> 项目刻意不引入 `torch` / `transformers` / `sentence-transformers` 等任何本地推理框架，
> 避免数百 MB ~ 数 GB 的模型下载，保证任何一台能装 Python 的机器都能起服务。

---

## 五、目录结构

```text
Robot/
├── main.py                     # 服务入口：FastAPI 装配、CORS、Trace 中间件、静态大屏挂载
├── requirements.txt            # 依赖清单
├── pyproject.toml              # 项目元数据
├── code_artifact.md            # 需求文档（FR/NFR、场景、路线图）
├── plan.md                     # 项目计划（架构、WBS、验收标准、风险矩阵）
│
├── src/
│   ├── api/
│   │   ├── routes.py           # 6 个接口实现（诊断 / Webhook / 审批 / 查询 / 健康检查）
│   │   └── schemas.py          # Pydantic 请求响应模型与字段校验
│   ├── agents/
│   │   ├── graph.py            # 状态图编译 + 检查点后端（SqliteSaver / MemorySaver）
│   │   ├── nodes.py            # 6 个节点实现 + 条件路由函数
│   │   └── state.py            # AgentState（贯穿全流程的唯一共享上下文）
│   ├── core/
│   │   ├── config.py           # pydantic-settings 配置中心（MySQL / LLM / Embedding）
│   │   ├── db.py               # SQLAlchemy Engine / 会话 / 连通性探测
│   │   ├── llm.py              # DeepSeek 与 Embedding 工厂（lru_cache 单例）
│   │   ├── logger.py           # Loguru + trace_id ContextVar + 文件轮转
│   │   ├── .env.example        # 配置模板（复制为 .env 后填 Key）
│   │   └── .env                # 本地真实密钥（已 gitignore，不入库）
│   ├── tools/
│   │   ├── rag_tool.py         # 手册检索（三级容错 + 进程级 RLock + 单例客户端）
│   │   └── sql_tool.py         # 安全 Text2SQL：设备查询 / 多仓库存 / 工单草稿
│   ├── static/
│   │   └── index.html          # 单文件可视化控制台（大屏）
│   ├── mcp_servers/            # 预留：MCP 工具总线（规划中）
│   └── skills/                 # 预留：Skill 技能层（规划中）
│
├── scripts/
│   ├── init_qdrant_data.py     # 初始化本地知识库（3 条模拟 SOP 切片）
│   ├── seed_mock_data.py       # 生成模拟业务数据（100 设备 / 500 种备件 / 200 工单）
│   ├── test_tools.py           # 工具层联调（设备 → RAG → 库存 → 工单草稿）
│   ├── test_graph.py           # 状态机端到端（含跨进程挂起恢复）
│   ├── test_api.py             # 接口层端到端（双通道 + 审批 + 幂等 + CORS）
│   ├── verify_all.py           # 一键验收：环境预检 → 三套测试 → 真服务冒烟 → 数据卫生
│   └── ui_smoke_test.mjs       # 用 Edge CDP 驱动控制台，截图验证「弹层 → 批准 → 库存 −1」
│
├── sql/
│   └── init_schema.sql         # 建库建表建索引 + 只读账号权限隔离
│
├── logs/                       # 运行日志（自动创建，已 gitignore）
├── qdrant_local_data/          # Qdrant 本地数据（已 gitignore）
├── checkpoints.sqlite          # LangGraph 检查点（已 gitignore）
└── tests/                      # 预留：pytest 单元测试
```

---

## 六、快速开始

### 6.1 环境要求

| 组件 | 版本 | 是否必需 |
| --- | --- | --- |
| Python | 3.14+（开发与验证环境：3.14.8） | ✅ 必需 |
| MySQL | 8.0.x | ✅ 必需 |
| DeepSeek API Key | — | ⭕ 可选（缺失时降级为正则抽取） |
| Embedding API Key | — | ⭕ 可选（缺失时检索降级为本地关键词打分） |
| Node.js | 18+ | ⭕ 仅跑 `ui_smoke_test.mjs` 需要 |
| Docker / Redis | — | ❌ **不需要** |

### 6.2 安装依赖

```bash
git clone https://github.com/BY-18/service-copilot.git
cd service-copilot

# 创建虚拟环境
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 6.3 配置密钥

```bash
# Windows
copy src\core\.env.example src\core\.env
# macOS / Linux
cp src/core/.env.example src/core/.env
```

编辑 `src/core/.env`，至少填好 MySQL 连接信息：

```ini
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=你的密码          # 不要提交到仓库
MYSQL_DB=service_copilot

DEEPSEEK_API_KEY=sk-xxxxxxxx     # 可选，不填则走正则兜底
EMBEDDING_API_KEY=sk-xxxxxxxx    # 可选，不填则走关键词兜底
EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
EMBEDDING_MODEL=BAAI/bge-m3
```

> `.env` 已被 `.gitignore` 忽略；`config.py` 也会把含 `REPLACE` / `your` / `请替换` /
> 中文等占位符的 Key 视作「未配置」，避免填了占位符还以为配好了、每次白打一次注定失败的请求。

### 6.4 初始化数据库

```bash
mysql -h 127.0.0.1 -P 3306 -u root -p < sql/init_schema.sql
```

> ⚠️ 该脚本会 `DROP TABLE` 后重建 3 张业务表，已有数据请先 `mysqldump` 备份。
> 脚本同时会创建只读账号 `copilot_agent`（仅 3 张表的 `SELECT` 权限），用于安全隔离演示。

### 6.5 灌入模拟数据与知识库

```bash
# 模拟业务数据：100 台设备 / 500 种备件（504 行，核心易损件分仓各占 1 行）/ 200 条历史工单
python scripts/seed_mock_data.py

# 初始化本地向量知识库（3 条模拟原厂 SOP 切片）
python scripts/init_qdrant_data.py --recreate

# 没有 Embedding Key 时，可用离线伪向量打通链路
python scripts/init_qdrant_data.py --offline --recreate
```

> 基线数据是**不可随意改动**的：`AGV-SZ-003` 必须在苏州、`型号A电磁阀` 必须在「苏州中心仓」有 3 件、
> 在「上海总仓」有 15 件 —— 后续三套测试脚本的断言都依赖这些数值。脚本结束时会做 12 项自洽校验，
> 任一不满足即退出码 1，避免出现「数据灌进去了但跑不通流程」的假成功。

### 6.6 启动服务

```bash
# 方式一：命令行启动（内部调 uvicorn.run）
python main.py --port 8000

# 方式二：uvicorn 直接启动（开发热重载）
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

启动后访问：

| 地址 | 内容 |
| --- | --- |
| <http://127.0.0.1:8000/> | 可视化控制台（静态大屏） |
| <http://127.0.0.1:8000/docs> | Swagger UI（可直接试调） |
| <http://127.0.0.1:8000/openapi.json> | OpenAPI 描述 |
| <http://127.0.0.1:8000/health> | 三件套依赖探活 |

### 6.7 跑一遍完整闭环（curl）

```bash
# ① 通道 A：工程师口语化报修 → 工作流会停在审批节点
curl -X POST http://127.0.0.1:8000/api/v1/diagnose/chat \
  -H "Content-Type: application/json" \
  -H "X-Trace-Id: TR-DEMO-0001" \
  -d "{\"query\": \"苏州厂区的 AGV-SZ-003 小车夹爪漏气合不上，报 E-201\"}"

# 响应：status = "WAITING_APPROVAL"，同时返回 order_no 与审批载荷

# ② 主管审批通过 → 从检查点续跑，核销 1 件库存并派工
curl -X POST http://127.0.0.1:8000/api/v1/work-orders/approve \
  -H "Content-Type: application/json" \
  -d "{\"trace_id\": \"TR-DEMO-0001\", \"order_no\": \"WO-202610-0001\", \"approval_status\": \"APPROVED\", \"approver\": \"张主管\"}"

# ③ 查工单与库存
curl "http://127.0.0.1:8000/api/v1/work-orders?limit=5"
curl "http://127.0.0.1:8000/api/v1/spare-parts?part_name=型号A电磁阀"
```

通道 B（设备告警）不调用 LLM 做槽位抽取，直接解析 JSON 报文字段：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/alert/webhook \
  -H "Content-Type: application/json" \
  -d "{\"device_sn\":\"AGV-SZ-003\",\"location\":\"苏州\",\"error_code\":\"E-201\",\"symptom\":\"夹爪持续漏气、无法闭合\"}"
```

---

## 七、接口一览

统一前缀 `/api/v1`（`/health` 除外）。所有响应都会带回 `X-Trace-Id` 头。

| 方法 | 路径 | 说明 | 关键返回 |
| --- | --- | --- | --- |
| `POST` | `/api/v1/diagnose/chat` | 通道 A：自然语言报修 | `status`（`WAITING_APPROVAL` / `COMPLETED`）、`sop_guide`、`order_no`、`approval_payload` |
| `POST` | `/api/v1/alert/webhook` | 通道 B：设备 JSON 告警 | 同上 |
| `POST` | `/api/v1/work-orders/approve` | HITL 审批恢复 | `dispatch_status`、`stock_deducted`、`stock_quantity_after`、`replayed` |
| `GET` | `/api/v1/work-orders` | 工单列表 | 支持 `limit` / `approval_status` / `trace_id` 筛选 |
| `GET` | `/api/v1/spare-parts` | 备件实时库存 | 支持 `part_name` 模糊筛选、`limit` |
| `GET` | `/health` | 依赖探活 | `mysql` / `qdrant` / `checkpointer` 三组件状态 |

**错误语义**

| 状态码 | 含义 |
| --- | --- |
| `422` | 参数校验失败（字段缺失 / 超长 / 格式非法） |
| `404` | 工单或挂起的工作流不存在 |
| `409` | 状态冲突（该 `trace_id` 没有处于挂起状态的审批任务） |
| `503` | MySQL 等依赖不可用 |

**`X-Trace-Id` 规则**：请求头带合法值就用它，否则服务端现场生成；`trace_id` 同时作为
LangGraph 的 `thread_id`，也是 `work_orders.trace_id` 列的值 —— 一次报修的日志、工单、
审批记录都可以用同一个 ID 反查。

---

## 八、配置说明

所有配置集中在 `src/core/config.py`，读取优先级（后者覆盖前者）：
**类默认值 → `src/core/.env` → `<项目根>/.env` → 真实环境变量**。

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MYSQL_HOST` / `MYSQL_PORT` | `127.0.0.1` / `3306` | MySQL 地址 |
| `MYSQL_USER` / `MYSQL_PASSWORD` | `root` / *(请自行填写)* | 连接账号 |
| `MYSQL_DB` | `service_copilot` | 业务库名 |
| `MYSQL_POOL_SIZE` / `MYSQL_MAX_OVERFLOW` | `5` / `10` | 连接池大小（本地单机够用） |
| `DEEPSEEK_API_KEY` | 空 | 不填则槽位抽取降级为正则 |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` | OpenAI 兼容端点 |
| `DEEPSEEK_MODEL` | `deepseek-chat` | 对话模型 |
| `DEEPSEEK_TEMPERATURE` | `0.0` | 排障场景要求稳定可复现 |
| `EMBEDDING_API_KEY` | 空 | 不填则检索降级为本地关键词打分 |
| `EMBEDDING_BASE_URL` | `https://api.siliconflow.cn/v1` | 向量服务端点 |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | 原生 1024 维，与集合维度对齐 |
| `EMBEDDING_DIMENSIONS` | 空 | 仅服务商支持输出维度裁剪时填写（如 `1024`） |
| `EMBEDDING_BATCH_SIZE` | `10` | 批量向量化单批条数（部分服务商限制 10 条/次） |
| `APP_ENV` / `LOG_LEVEL` | `dev` / `INFO` | 运行环境与日志级别 |
| `CORS_ALLOW_ORIGINS` | `*` | 逗号分隔；生产环境请收窄为具体域名 |

**硬编码的相对路径**（按需求固定，不走 `.env`，避免跨机器路径漂移）：

| 常量 | 值 | 用途 |
| --- | --- | --- |
| `QDRANT_LOCAL_DATA_PATH` | `./qdrant_local_data` | 本地向量库目录 |
| `CHECKPOINT_DB_PATH` | `checkpoints.sqlite` | LangGraph 检查点文件 |
| `VECTOR_SIZE` | `1024` | 向量维度 |
| `MANUAL_COLLECTION_NAME` | `robot_manual_sop` | 知识库集合名 |

---

## 九、安全设计

### 9.1 SQL 注入的三重防线

```
用户自然语言
     │
     ▼
① LLM 只抽取「槽位」（设备 SN / 故障码），不生成 SQL 文本
     │
     ▼
② 固定 SQL 模板 + SQLAlchemy text() + bindparams()
   → 用户输入永远只是绑定参数，从不参与 SQL 文本拼装
     │
     ▼
③ MySQL 最小权限账号 copilot_agent
   → 只有 3 张表的 SELECT；即使模型幻觉出 DELETE / DROP，也会被 MySQL 直接拒绝
     │
     ▼
④ 唯一的设计内写路径（审批核销）由代码里写死的模板 SQL 执行，不由模型生成
```

数据库脚本 `sql/init_schema.sql` 里刻意保留了一段**反证测试**注释：
用只读账号执意写库，应当报 `ERROR 1142 command denied`。

### 9.2 密钥管理

- 真实 Key 只放在 `src/core/.env`（已 `.gitignore`），源码与文档中不出现；
- `.env.example` 只保留空值模板；
- 日志输出连接串时会打码（`mysql_url_masked`），不打印密码；
- `config.py` 会拦截占位符与含非 ASCII 字符的 Key，避免把中文占位符塞进 HTTP 头导致
  `UnicodeEncodeError`（该报错信息与「Key 无效」毫无关系，极易误导排查）。

> ⚠️ **如果你 Fork 了这个项目：请务必更换为你自己的 API Key，不要复用仓库中的任何凭据。**

### 9.3 HITL 与幂等

高价备件出库必须经过人工审批：工作流在 `human_approval` 节点 `interrupt()` 挂起，
状态写入 `checkpoints.sqlite`。审批接口会先检查工单当前状态：

- 已决（`APPROVED` / `REJECTED`）→ 直接返回幂等回放结果，**不重复扣库存**；
- 重复点击、网络重试、并发提交都不会造成库存错账。

---

## 十、测试与验收

四套脚本，逐层递进，每套都能独立运行并自带清理：

| 脚本 | 覆盖内容 | 运行 |
| --- | --- | --- |
| `scripts/test_tools.py` | 工具层 4 项：设备定位 → RAG 命中 → 就近选仓 → 工单草稿落库 | `python scripts/test_tools.py` |
| `scripts/test_graph.py` | 状态机 6 项：图结构 / E-201 全链路 / 审批驳回 / E-500 无需换件 / Webhook 通道 / **跨进程挂起恢复** | `python scripts/test_graph.py` |
| `scripts/test_api.py` | 接口层 9 项：健康检查 / 双通道 / 审批幂等 / 驳回 / 参数校验 / CORS 与 Trace 透传 | `python scripts/test_api.py` |
| `scripts/verify_all.py` | 一键验收：环境预检 → 三套测试 → 真实服务冒烟 → 数据卫生 | `python scripts/verify_all.py` |

常用参数：

```bash
python scripts/test_graph.py --only 2,6     # 只跑指定用例
python scripts/test_graph.py --keep         # 保留工单与库存变动，便于人工核对
python scripts/verify_all.py --skip-live    # 跳过真实服务冒烟，只跑三套功能测试
python scripts/verify_all.py --skip-suites  # 跳过功能测试，只做冒烟
```

**跨进程恢复**是这套测试里最有价值的一条：用例 2 在挂起后用一个**新建的** `SqliteSaver`
连接恢复执行，等价于"重启进程后继续"；用例 6 更进一步开两个独立子进程做同等强度的验证。

```bash
# 手工验证跨进程挂起 / 恢复
python scripts/test_graph.py --worker start  --thread TR-X --raw "苏州 AGV-SZ-003 报 E-201 夹爪漏气"
python scripts/test_graph.py --worker resume --thread TR-X --decision APPROVED
```

> 注意：Qdrant 本地文件模式是**进程独占锁**，服务运行期间脚本打不开 `./qdrant_local_data`。
> 跨进程用例会主动关闭父进程的 Qdrant 客户端以避免冲突。

---

## 十一、可视化控制台

启动服务后访问 <http://127.0.0.1:8000/>，单文件大屏（`src/static/index.html`）包含四个区域：

| 区域 | 内容 |
| --- | --- |
| **接入面板** | 「模拟人工报修 E-201」一键演示；也可手输自然语言或提交 Webhook 报文 |
| **智能体执行轨迹** | 实时时间线：槽位抽取 → 手册检索 → 备件判定 → 库存调度 → HITL 挂起 |
| **ERP 备件实时库存** | 备件名称 / 物料编码 / 仓库货位 / 可用库存 / 单价 / 更新时间；审批通过后库存数字会闪烁 −1 |
| **工单看板** | 工单号 / 设备 SN / 故障码 / 建议备件 / 出库仓 / 审批态 / 派工态 / 创建时间 |
| **HITL 审批弹层** | 挂起时自动弹出，展示备件、仓库、当前库存；批准 → 核销 1 件 + `DISPATCHED`，驳回 → `CLOSED` 且库存不动 |

控制台的端到端 UI 验证（需要本机 Edge 与 Node 18+）：

```bash
node scripts/ui_smoke_test.mjs
# 产出截图：$TEMP/copilot-hitl-modal.png、$TEMP/copilot-approved.png
```

---

## 十二、常见问题

<details>
<summary><b>MySQL 连不上，服务还能起来吗？</b></summary>

能。启动探活只记录日志、不阻断启动 —— MySQL 掉线时 `/health` 仍可访问并回报
`status=degraded` 与具体错误，业务接口返回 `503`。这样部署时的排查链路不会被启动失败掩盖。
</details>

<details>
<summary><b>报错 "Qdrant local mode storage folder is already accessed"</b></summary>

Qdrant 本地文件模式是**单进程独占**的。服务运行期间，脚本无法同时打开
`./qdrant_local_data`。请先停掉服务再跑脚本，或反之。
</details>

<details>
<summary><b>没有 API Key 能跑吗？</b></summary>

能，且是三处降级：

1. 无 `DEEPSEEK_API_KEY` → 槽位抽取退化为正则匹配（SN / `E-xxx` 故障码）；
2. 无 `EMBEDDING_API_KEY` → 检索退化为「故障码精确匹配 + 中文 2-gram 重合度」本地打分；
3. 知识库没初始化 → `init_qdrant_data.py --offline` 用伪向量灌库，打通链路但无语义能力。

这保证了车间断网 / 演示现场没网时，工作流仍然跑得通。
</details>

<details>
<summary><b>进程重启后，挂起的审批工单还在吗？</b></summary>

在。只要装了 `langgraph-checkpoint-sqlite`，状态就落在 `checkpoints.sqlite`，凭原
`trace_id` 即可恢复。若该包缺失，会自动回退 `MemorySaver` 并打印警告 ——
此时进程重启后挂起状态会丢失。
</details>

<details>
<summary><b>Windows 控制台中文乱码</b></summary>

PowerShell 需显式指定 UTF-8 读取文件，例如 `Get-Content -Encoding UTF8 file.py`；
或先执行 `[Console]::OutputEncoding = [Text.Encoding]::UTF8`。
</details>

<details>
<summary><b>端口被占用 / 换端口</b></summary>

`python main.py --port 8001`，或用环境变量 `API_HOST` / `API_PORT`。
</details>

---

## 十三、Roadmap

- [x] 阶段一：基础设施打桩（MySQL 建表 + Qdrant 灌库 + 配置中心 + 日志底座）
- [x] 阶段二：工具层（安全 Text2SQL + RAG 检索 + 工单草稿）
- [x] 阶段三：LangGraph 状态机（HITL 挂起 / 恢复 / 跨进程检查点）
- [x] 阶段四：FastAPI 双通道接入 + 审批接口 + 静态控制台
- [x] 阶段五：一键验收脚本 + 端到端 UI 冒烟
- [ ] MCP 工具总线（`src/mcp_servers/`）：把工具层标准化为 MCP Server
- [ ] Skill 技能层（`src/skills/`）：把「排障 SOP」沉淀为可复用技能包
- [ ] Redis 分布式锁：多实例部署下的库存核销并发控制
- [ ] pytest 单元测试（`tests/`）接入 CI
- [ ] 鉴权与多租户（JWT + RBAC）

---

## 关于参考文档

- [`code_artifact.md`](./code_artifact.md) —— 详细需求文档（用户画像、FR/NFR、实施路线图、风险矩阵）
- [`plan.md`](./plan.md) —— 项目计划（分层架构、WBS、验收标准、依赖清单）

---

## 许可证

本项目目前**未添加开源许可证文件**。如需开源，建议补充
[MIT License](https://choosealicense.com/licenses/mit/) 的 `LICENSE` 文件后再公开发布。

---

<div align="center">
<sub>Service Copilot · 让售后工程师只做那个「批不批」的决定</sub>
</div>
