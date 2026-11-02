# 机器人售后排障与备件智能调度智能体（Service Copilot）—— 项目计划文档

> 本文档基于《机器人售后排障与备件智能调度智能体-需求文档 V1.0》制定，聚焦系统工程实施规划、技术规格设计与任务分解，不包含可执行业务代码。

| 项目 | 内容 |
|---|---|
| 项目代号 | Service Copilot |
| 关系型数据库 | MySQL 8.0 @ 172.16.50.78:3306 (库名: service_copilot) |
| 向量知识库 | Qdrant @ 172.16.50.78:6333 |
| 状态与会话缓存 | Redis 7.0 @ 172.16.50.78:6379 |
| 核心编排框架 | LangGraph + LangChain + MCP (Model Context Protocol) |
| 安全执行底座 | SQLAlchemy 参数化绑定中间件 (防注入) |
| 并发与状态持久化 | Redis Checkpointer + Redis 分布式锁 + Celery 异步调度 |

---

## 一、项目目标与范围

### 1.1 项目目标
构建覆盖工业机械臂、仓储 AGV 等机器人集群运行维护的**智能排障与备件调度中枢**。系统打通“双通道故障接入（人工口语化报修 + 边缘 Webhook 告警）”，通过 LangGraph 编排多智能体协同，结合 Qdrant 向量知识库（非结构化手册）与受控 Text2SQL（结构化 ERP 库存），实现自动排障诊断、备件就近调拨推演与“人在回路 (HITL)”工单流转闭环，覆盖需求规格说明书中 FR-1 ~ FR-5 全部功能。

### 1.2 核心交付能力
1. **多智能体编排**：基于 LangGraph 的 `StateGraph` 实现 `Router Agent`、`Diagnose Agent`、`Inventory Agent` 及 `HITL Coordinator` 的有序状态流转与条件跳转。
2. **双通道故障接入**：支持一线售后工程师口语化报修（槽位提取与消歧）与边缘硬件 Webhook 高频结构化报文解析。
3. **安全数据底座与 Text2SQL 防护**：通过 SQLAlchemy `text()` 与 `.bindparams()` 强制参数化绑定，配合 MySQL 只读连接账号，杜绝大模型 SQL 注入。
4. **混合 RAG 知识检索**：在 Qdrant 建立原厂维修手册向量库，实现故障代码硬匹配与故障现象语义向量的混合召回，并从中抽取标准备件品名。
5. **人在回路（HITL）与状态持久化**：利用 Redis Checkpointer 实现工单生成后的状态机断点挂起（Suspend），经由主管审批后无缝唤醒并执行最终派工。
6. **本地化与私有化交付**：基于 Docker Compose 实现全栈容器化单机部署，适配车间内网隔离与企业私有云环境。

### 1.3 实施范围（对齐需求文档路线图）
- **阶段一（最小闭环与基础设施打桩）**：MySQL/Redis/Qdrant 容器化拉起 + 数据表初始化与假数据注入 + 双通道接入校验 + 基础 FastAPI 框架骨架。
- **阶段二（核心智能体与安全攻坚）**：LangGraph 状态机编排 + Qdrant 混合检索 RAG + SQLAlchemy 防注入 Text2SQL + Redis 状态断点与审批流打通。
- **阶段三（前端看板与打桩联动验证）**：Streamlit 交互式控制台 + `mock_robot_alert.py` 边缘打桩模拟脚本 + 端到端测试与观测报告归档。

---

## 二、总体技术架构

### 2.1 分层架构

```
┌─────────────────────────────────────────────────────────────────┐
│  交互与网关层：Streamlit 控制台 | FastAPI 异步 Webhook | 企业移动端 │
├─────────────────────────────────────────────────────────────────┤
│  智能体编排层（LangGraph StateGraph）                           │
│   ┌──────────────┐     ┌──────────────┐     ┌────────────────┐  │
│   │ Router Agent │──▶  │Diagnose Agent│──▶  │Inventory Agent │  │
│   │ 双通道统一分发│     │ RAG 手册检索 │     │受控 Text2SQL   │  │
│   └──────────────┘     └──────────────┘     └────────────────┘  │
│          ▲                    ▲                     ▲           │
│          └──────────── MCP 工具总线与 Skill 能力层 ──┘           │
│    (ERP库存/MySQL/Qdrant/日志提取/通知审批 均以标准化工具暴露)      │
├─────────────────────────────────────────────────────────────────┤
│  人在回路层（HITL）：Redis Checkpoint 挂起 ◀─▶ 主管审批看板唤醒   │
├─────────────────────────────────────────────────────────────────┤
│  并发与控制层：Redis 分布式锁 + Celery 异步任务 + Loguru Trace ID │
├─────────────────────────────────────────────────────────────────┤
│  数据与存储层：MySQL service_copilot | Qdrant 知识库 | Redis 7.0 │
├─────────────────────────────────────────────────────────────────┤
│  边缘设备层：边缘机器人 Webhook | 本地 MCP Server (日志直连)      │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 LangGraph 多智能体工作流设计

系统采用 LangGraph 的 `StateGraph` 进行带环状态机编排，共享全局强类型的 `AgentState` 上下文。

**全局共享状态数据结构（定义）：**
- `trace_id`: 全链路排查追踪 UUID
- `channel`: 触发通道类型（`MANUAL` 或 `WEBHOOK`）
- `device_sn`: 机器人硬件序列号
- `error_code`: 故障错误代码（如 `E-201`）
- `symptom`: 故障现象口语化描述
- `location`: 设备所属厂区与就近大区（如 `苏州`）
- `need_replace`: 是否需更换硬件备件（布尔值）
- `part_name`: 提取出的标准备件品名
- `stock_info`: 查库返回的仓位、余量与交期对象
- `approval_status`: 审批状态（`PENDING`, `APPROVED`, `REJECTED`）
- `work_order_draft`: 工单草稿内容

**工作流节点与流转设计：**

| 节点 | 职责 | 输入状态 | 输出状态 | 路由条件 |
|---|---|---|---|---|
| `router_node`（路由Agent） | 解析输入，对齐通道 A/B 报文并抽取实体槽位 | 原始报文 / 文本 | 核心基础状态字段 | 核心槽位完整 → diagnose；缺失严重 → clarify |
| `clarify_node`（消歧Agent） | 向人工追问缺失的故障关键信息 | 部分状态字段 | 追问 Prompt / 补充文本 | 补充完成 → router；用户取消 → END |
| `diagnose_node`（排障Agent） | 执行 Qdrant 混合检索，抽取更换备件品名 | error_code, symptom | need_replace, part_name | need_replace=True → inventory；False → generate_draft |
| `inventory_node`（库存Agent） | 执行安全参数化 Text2SQL 查询就近库存余量 | part_name, location | stock_info | 查询完毕 → generate_draft |
| `generate_draft_node` | 整合信息生成工单草稿并配置断点 | 全量状态上下文 | work_order_draft | 触发 `interrupt_before` 挂起进入 Redis |
| `approval_callback_node` | 接收主管审批操作，唤醒工作流 | approval_status | final_result | APPROVED → dispatch；REJECTED → END |
| `dispatch_node`（派工节点） | 调拨核销锁定库存，触发派单通知 | final_result | dispatch_status | 派发完成 → END |

### 2.3 MCP 工具总线设计

采用标准化 MCP (Model Context Protocol) 将底层数据存取与边缘设备交互能力抽象为独立工具服务，使 Agent 与物理层完全解耦。

| MCP Server | 暴露工具 (Tools) | 业务功能用途 |
|---|---|---|
| `mcp-mysql-inventory` | `query_stock_by_part` / `query_device_meta` / `create_work_order` | 执行受控只读参数化 SQL 查询与审批后的工单落库 |
| `mcp-qdrant-sop` | `search_sop_by_code` / `hybrid_search_manual` | 向量与关键词混合检索原厂维修手册切片 |
| `mcp-edge-logger` | `fetch_robot_can_dump` / `read_local_firmware_log` | 通过安全只读协议读取断网车间本地设备的日志文件 |
| `mcp-notification` | `send_approval_card` / `notify_warehouse_dispatch` | 触发 Web 审批看板弹窗、企业微信/钉钉工单卡片推送 |

### 2.4 Skill 技能层设计

Skill 负责将特定算法、实体抽取与业务风控逻辑模块化封装，供 Agent 节点复用。

| Skill 标识 | 业务能力描述 | 主要调用节点 |
|---|---|---|
| `skill-intent-extraction` | 自然语言实体抽取（从口语长文本提取 SN、错误码、厂区） | `router_node` |
| `skill-part-extraction` | 从 RAG 召回的非结构化维修手册中识别并规整备件标准品名 | `diagnose_node` |
| `skill-sql-sanitizer` | SQL 语句安全校验，强制进行 SQLAlchemy 参数化绑定和语法白名单审查 | `inventory_node` |
| `skill-multi-warehouse-rank` | 多仓就近调度算法：优先本地厂区仓，次级调度大区中心仓与总仓 | `inventory_node` |
| `skill-hitl-snapshot` | 状态机上下文序列化与反序列化，对接 Redis 检查点持久化 | `generate_draft_node` |

---

## 三、高并发与分布式锁设计

### 3.1 核心并发与冲突场景分析

| 场景 | 并发风险 | 保护对象与策略 |
|---|---|---|
| 机器人高频报警上报 | 单台设备短时间内因同一故障连续上报几十次 Webhook，导致工作流重复创建 | 设备报警风暴去重锁（基于设备 SN + 错误代码） |
| 热门备件并发锁定 | 多个并发工单同时扣减同一种紧俏备件库存，导致负库存超卖 | 备件料号排他锁（基于 part_name） |
| 主管并发重复审批 | 审批卡片被多次点击或网络重试，引发工单重复派发 | 工单审批幂等锁（基于 work_order_id） |
| Redis Checkpoint 写入 | 状态流转时多节点并发刷写检查点导致上下文覆盖 | 线程会话锁（基于 thread_id） |

### 3.2 分布式锁规范（Redis）

**技术方案**：基于 Redis 7.0 实现具备原子性校验的可重入分布式锁。

**分布式锁 Key 设计规范：**

| 锁类型 | Key 模板 | 建议 TTL | 持锁主体 |
|---|---|---|---|
| 设备告警去重锁 | `lock:alert:device:{device_sn}:{error_code}` | 60s（防高频抖动） | FastAPI Webhook 入口中间件 |
| 备件调拨锁 | `lock:inventory:part:{part_code}` | 10s（事务结束即释放） | `inventory_node` 校验与核销逻辑 |
| 工单审批锁 | `lock:approval:order:{order_id}` | 15s | `approval_callback_node` |
| 状态机实例锁 | `lock:graph:thread:{thread_id}` | 5min（与节点执行上限一致）| LangGraph 工作流驱动引擎 |

**锁机制技术要点：**
1. **加锁原子性**：统一执行 `SET key unique_token NX PX ttl`，其中 `unique_token` 采用全局 UUID + Trace ID 组合。
2. **安全释放**：使用 Lua 脚本先校验 `unique_token` 再执行 `DEL`，彻底防止因任务执行时间过长导致锁自动过期后误删其他线程持有的锁。
3. **退避与重试**：获取锁失败时执行指数退避重试（间隔 50ms、100ms、200ms，最多重试 3 次），仍未获取则进入异常队列，不引发线程级永久阻塞。

### 3.3 异步任务与调度控制
- **异步处理**：利用 Celery 异步队列接管大模型长文本推理与耗时通知推送任务，保障 FastAPI 网关接口的高吞吐能力。
- **连接复用**：
  - MySQL：配置 SQLAlchemy 异步连接池（`pool_size=20`, `max_overflow=10`, `pool_recycle=3600`）。
  - Qdrant：全局复用 `AsyncQdrantClient` 单例。
  - Redis：构建 `ConnectionPool`（`max_connections=50`），兼顾 Checkpointer 与分布式锁通信。

---

## 四、MySQL 数据库脚本计划（service_copilot @ 172.16.50.78:3306）

> 部署环境：`172.16.50.78:3306`，业务库名 `service_copilot`
> 字符集：`utf8mb4`，排序规则：`utf8mb4_unicode_ci`，存储引擎：`InnoDB`

### 4.1 脚本清单计划

| 脚本文件路径 | 脚本用途 |
|---|---|
| `sql/00_init_database.sql` | 创建数据库 `service_copilot`、指定字符集并创建只读/读写专有用户 |
| `sql/01_ddl_tables.sql` | 创建全部业务表、审计表及外键关系 |
| `sql/02_init_seed_data.sql` | 录入 50+ 条虚拟设备档案、多厂区多仓备件库存、标准物料字典 |
| `sql/03_indexes.sql` | 针对高频 Text2SQL 字段与工单追踪字段补充二级索引 |
| `sql/04_cleanup.sql` | 自动化环境重置脚本（按依赖逆序清空表结构） |

### 4.2 核心业务表结构规划

#### 4.2.1 设备档案与物料主数据
| 表名 | 关键字段定义 | 业务说明 |
|---|---|---|
| `robot_devices` | id, device_sn (唯一), model_code, factory_location, workshop_id, status, created_at | 机器人资产档案台账表 |
| `spare_parts_master` | id, part_code (唯一), part_name, category, unit, min_safety_stock, is_critical | 备件物料标准字典主表 |
| `error_code_mapping` | id, error_code, model_code, default_description, category | 故障代码字典基准表 |

#### 4.2.2 仓库与实时库存
| 表名 | 关键字段定义 | 业务说明 |
|---|---|---|
| `warehouses` | id, warehouse_code, name, region, priority, is_central | 仓库主表（本地仓、大区中心仓） |
| `spare_parts_inventory` | id, part_id, warehouse_id, stock_qty, locked_qty, bin_location, updated_at | 备件分仓实时库存表（Text2SQL 核心表） |
| `inventory_transactions` | id, order_id, part_id, warehouse_id, change_qty, type, created_at | 库存调拨与冻结审计流水表 |

#### 4.2.3 工单与人在回路流转
| 表名 | 关键字段定义 | 业务说明 |
|---|---|---|
| `work_orders` | id, order_no, device_sn, error_code, part_id, req_qty, status, approver, created_at | 售后维修工单主表 |
| `order_approval_records` | id, order_id, action, comments, operator, operated_at | 主管审批历史轨迹留痕表 |
| `agent_diagnosis_logs` | id, trace_id, input_channel, raw_input, sop_citation, generated_sql, status | Agent 推理与诊断详情审计表 |

### 4.3 关键索引设计
- `robot_devices(device_sn, factory_location)`
- `spare_parts_inventory(part_id, warehouse_id)`
- `spare_parts_inventory(stock_qty)` —— 支撑高效 Text2SQL 数值范围检索
- `work_orders(device_sn, status, created_at)`
- `agent_diagnosis_logs(trace_id)`

---

## 五、Qdrant 向量数据库规划（172.16.50.78:6333）

> 连接端点：`172.16.50.78:6333`

### 5.1 脚本清单计划
| 脚本文件路径 | 脚本用途 |
|---|---|
| `qdrant/01_create_collections.py` | 自动化创建全部 Collection 并指定向量维度与距离度量 |
| `qdrant/02_ingest_manuals.py` | 解析 PDF/Markdown 维修手册，分块并向量化灌库 |
| `qdrant/03_clear_collections.py` | 清理并重置 Collection（用于重测） |

### 5.2 Collection 规格设计

| Collection 名称 | 向量维度 | 距离算法 | Payload 关联字段 | 业务检索用途 |
|---|---|---|---|---|
| `robot_repair_manuals` | 1024 | Cosine | model_code, error_code, section_title, target_part, chunk_text | 原厂维修手册 SOP 语义检索（核心知识库） |
| `history_troubleshooting` | 1024 | Cosine | device_sn, error_code, resolution, verified_part, feedback_score | 历史成功排障案例库（经验反哺召回） |
| `symptom_glossary` | 1024 | Dot | canonical_term, synonyms, category | 现场口语化故障特征与专业术语对齐词库 |

### 5.3 混合检索策略与 Payload 过滤
- 采用 **Hybrid Search** 策略：
  1. **硬约束过滤**：通过 Qdrant Payload 筛选 `model_code` 与精确匹配的 `error_code`。
  2. **向量语义检索**：针对现场工程师描述的 `symptom` 计算余弦相似度。
  3. **倒排融合**：使用 RRF (Reciprocal Rank Fusion) 算法综合两路召回结果，输出置信度最高的 Top-3 切片。

---

## 六、目录结构与工程交付物计划

```
ServiceCopilot/
├── docs/
│   ├── service_copilot_prd.md                     (详细需求规格说明书)
│   └── service_copilot_plan.md                    (本文档：实施规划)
├── sql/                                           (MySQL 初始化脚本)
│   ├── 00_init_database.sql
│   ├── 01_ddl_tables.sql
│   ├── 02_init_seed_data.sql
│   ├── 03_indexes.sql
│   └── 04_cleanup.sql
├── qdrant/                                        (Qdrant 向量库脚本)
│   ├── 01_create_collections.py
│   ├── 02_ingest_manuals.py
│   └── 03_clear_collections.py
├── data/                                          (原始资料资产)
│   ├── manuals/                                   (PDF 原厂维修排障手册)
│   └── mock_logs/                                 (模拟设备运行与报警日志)
├── src/                                           (核心业务源代码)
│   ├── api/                                       (接入层)
│   │   ├── dependencies.py                        (鉴权与只读数据源依赖注入)
│   │   ├── router_webhook.py                      (通道 B：设备 Webhook 入口)
│   │   └── router_manual.py                       (通道 A：前端会话接口)
│   ├── agents/                                    (LangGraph 多智能体编排)
│   │   ├── state.py                               (AgentState 强类型状态定义)
│   │   ├── router_agent.py                        (双通道解析分发节点)
│   │   ├── diagnose_agent.py                      (RAG 手册推理节点)
│   │   ├── inventory_agent.py                     (受控 Text2SQL 节点)
│   │   ├── hitl_coordinator.py                    (人在回路中断与审批处理)
│   │   └── graph.py                               (StateGraph 状态机总编译)
│   ├── mcp_servers/                               (MCP 标准化工具总线)
│   │   ├── mcp_mysql_server.py
│   │   ├── mcp_qdrant_server.py
│   │   ├── mcp_edge_logger.py
│   │   └── mcp_notification.py
│   ├── skills/                                    (独立能力封装层)
│   │   ├── skill_intent_extract.py
│   │   ├── skill_part_extract.py
│   │   ├── skill_sql_sanitizer.py                 (SQLAlchemy 参数化绑定防注入)
│   │   └── skill_warehouse_rank.py
│   ├── core/                                      (系统基础设施底座)
│   │   ├── config.py                              (环境变量与配置管理)
│   │   ├── database.py                            (SQLAlchemy 异步连接池)
│   │   ├── redis_client.py                        (Redis 连接池与分布式锁封装)
│   │   ├── qdrant_client.py                       (Qdrant 异步连接单例)
│   │   └── logger.py                              (Loguru 注入 Trace ID 中间件)
│   └── main.py                                    (FastAPI 启动总入口)
├── ui/                                            (极简控制台前端)
│   └── app_streamlit.py                           (Streamlit 人工报修与主管审批看板)
├── tests/                                         (测试与模拟套件)
│   ├── mock_robot_alert.py                        (通道 B 边缘告警模拟上报脚本)
│   ├── test_sql_injection.py                      (SQL 防注入渗透攻击测试用例)
│   └── test_hitl_workflow.py                      (工单断点挂起与唤醒自动化测试)
├── docker-compose.yml                             (单机一键启动编排文件)
├── requirements.txt                               (依赖清单)
└── README.md                                      (系统部署与演示说明)
```

---

## 七、任务分解（WBS）

| 编号 | 任务阶段 | 任务模块与描述 | 输出目录 | 阶段 | 前置依赖 |
|---|---|---|---|---|---|
| T1 | 基础设施 | 编写 `docker-compose.yml`，本地拉起 MySQL, Redis, Qdrant 集群 | 根目录 | 一 | - |
| T2 | 数据底座 | 编写 MySQL 建库、建表、索引与测试假数据脚本 | sql/ | 一 | T1 |
| T3 | 数据底座 | 编写 Qdrant Collection 创建与维修手册分块向量化灌库脚本 | qdrant/ | 一 | T1 |
| T4 | 框架基础 | 构建基础设施层（配置中心、连接池、Loguru Trace ID 拦截器） | src/core | 一 | T2, T3 |
| T5 | 接入设计 | 开发通道 B Webhook 接口与通道 A 基础接入路由 | src/api | 一 | T4 |
| T6 | 并发控制 | 实现基于 Redis 的原子分布式锁与防重试机制 | src/core | 二 | T4 |
| T7 | 工具协议 | 封装标准 MCP MySQL、Qdrant 与通知服务工具集 | src/mcp_servers | 二 | T2, T3 |
| T8 | 技能沉淀 | 实现 `skill-sql-sanitizer`（SQLAlchemy 参数化防注入核心拦截） | src/skills | 二 | T7 |
| T9 | 状态定义 | 编写 `AgentState` 强类型定义与上下文数据校验器 | src/agents | 二 | T5 |
| T10 | 节点开发 | 实现 `Router Agent`（通道对齐）与 `Diagnose Agent`（RAG 备件提取） | src/agents | 二 | T8, T9 |
| T11 | 节点开发 | 实现 `Inventory Agent`（带防注入的 Text2SQL 查库逻辑） | src/agents | 二 | T8, T10 |
| T12 | HITL 编排 | 接入 Redis Checkpointer，实现工单草稿挂起与审批回调唤醒 | src/agents | 二 | T6, T11 |
| T13 | 编排装配 | 组合编译完整 LangGraph StateGraph，完成核心路由连通 | src/agents | 二 | T10, T11, T12 |
| T14 | 前端实现 | 使用 Streamlit 快速搭建操作员提问端与主管一键审批控制台 | ui/ | 三 | T13 |
| T15 | 边缘模拟 | 编写 `mock_robot_alert.py` 本地打桩脚本，高频模拟设备 Webhook 告警 | tests/ | 三 | T5 |
| T16 | 安全渗透 | 编写 SQL 注入针对性自动化攻击测试，验证安全拦截能力 | tests/ | 三 | T8 |
| T17 | 全链路调测 | 串联“双通道触发 $\to$ RAG 手册检索 $\to$ Text2SQL 查库 $\to$ 审批派发”端到端测试 | tests/ | 三 | T14, T15 |
| T18 | 交付归档 | 产出系统测试评估报告，编写运行部署文档 | docs/ | 三 | T17 |

---

## 八、验收标准（对齐需求与实施规范）

| 指标类别 | 验收项与具体指标 | 验证方法与手段 |
|---|---|---|
| **接入吞吐性能** | 通道 B 边缘机器 Webhook 响应解析耗时 $\le 100\text{ ms}$ | 使用测试脚本发起 50 次连续 Webhook 压测，统计 P99 耗时 |
| **检索效率** | Qdrant 原厂维修手册 Top-3 召回耗时 $\le 500\text{ ms}$ | 记录排障 Agent 检索阶段平均耗时统计 |
| **端到端时延** | 从故障接入到生成工单挂起草稿总耗时 $\le 5\text{ s}$ | 端到端自动化脚本运行日志 Trace ID 打点耗时统计 |
| **业务准确率** | 手册排障 SOP 召回准确率 $\ge 95\%$，备件抽取准确率 $\ge 92\%$ | 基于 50 组已知标准答案的历史故障测试集进行回归评测 |
| **安全防护要求** | 针对 SQL 注入攻击、非 SELECT 语句执行拦截率达到 $100\%$ | 执行 `test_sql_injection.py` 模拟恶意注入语句，断言报错与拦截记录 |
| **流程高可用性** | LangGraph 审批断点恢复成功率达到 $100\%$ | 在工单挂起阶段重启服务容器，重启后验证审批唤醒续跑能力 |
| **可解释性** | 诊断结论 $100\%$ 包含维修手册原文段落引用（Citation） | 抽检自动生成工单内容中是否有明确的章节依据标注 |

---

## 九、依赖环境与第三方清单

| 组件/库名称 | 建议版本 | 在系统中的用途与定位 |
|---|---|---|
| **Python** | 3.11 或 3.12 | 基础运行语言环境 |
| **FastAPI** | $\ge 0.110.0$ | 异步 Web 框架，承载通道 A/B 流量与管理接口 |
| **LangGraph** | $\ge 0.2.0$ | 多智能体有状态编排、条件跳转与 HITL 状态机控制 |
| **LangChain** | $\ge 0.2.0$ | Prompt 管理、文档切块与输出标准化解析 |
| **SQLAlchemy** | $\ge 2.0.0$ | 异步数据库 ORM 及底层参数化查询安全驱动 |
| **qdrant-client** | $\ge 1.9.0$ | Qdrant 向量数据库通信客户端 |
| **redis-py** | $\ge 5.0.0$ | Redis 异步通信、分布式锁实现与 Checkpointer 持久化 |
| **mcp** | 最新标准版 | Model Context Protocol 工具总线标准化集成 |
| **Loguru** | $\ge 0.7.0$ | 结构化日志输出与基于 ContextVar 的 Trace ID 追踪 |
| **Streamlit** | $\ge 1.35.0$ | 极简交互控制台与主管审批看板前端框架 |
| **Pydantic** | $\ge 2.7.0$ | 请求报文强类型校验与 AgentState 模式约束 |

---

## 十、实施风险管理矩阵

| 风险描述 | 潜在系统影响 | 预置缓解与应对策略 |
|---|---|---|
| **大模型产生 SQL 幻觉或越权** | 访问未经授权的数据表，甚至破坏数据库表结构 | 严格实行连接池只读权限限制；在 `skill-sql-sanitizer` 强制采用 SQLAlchemy 参数化绑定 |
| **原厂手册备件名称不标准** | 大模型抽取出的备件名在 ERP 无法精确匹配 | 在入库前建立备件物料标准同义词词典；支持 Text2SQL 阶段使用标准品名模糊关联辅助校准 |
| **高并发下工单重复审批** | 导致库存重复冻结或重复发件出库 | 采用基于 `order_id` 的 Redis 分布式排他锁，并在写入端设计唯一幂等约束 |
| **工业车间现场断网** | 云端公有 LLM API 无法直连访问 | 系统各模块采用标准化解耦接口设计，底层支持切换为本地部署的 Ollama/vLLM 私有化开源大模型 |
| **状态机中断等待超时** | 主管多日未审批，导致任务在 Redis 中积压僵死 | 设置超时告警策略（如挂起超过 24 小时自动流转至兜底人工客服待办列表） |

---

> 备注：本文档为技术实施标准蓝图，后续工程开发（Vibe Coding / 敏捷编码）将全面遵照本规划文件中的目录分层与接口规范推进落地。