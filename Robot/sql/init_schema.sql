-- =============================================================================
-- Service Copilot · 机器人售后排障与备件智能调度智能体
-- 数据库初始化脚本：库 / 表 / 索引 / 账号权限
--
-- 目标实例   : MySQL 8.0.x（本地 127.0.0.1:3306 或 plan.md 约定的 172.16.50.78:3306）
-- 业务库名   : service_copilot
-- 执行账号   : root（建库建表建账号都需要该权限）
--
-- 执行方式:
--     mysql -h 127.0.0.1 -P 3306 -u root -p < sql/init_schema.sql
--     或在 MySQL Workbench / Navicat 中整体执行本文件
--
-- ⚠️ 破坏性提示
--     本脚本会 DROP 掉 3 张业务表后重建，用于「从零可复现」地拉起环境。
--     执行完请立刻运行 python scripts/seed_mock_data.py 灌入模拟数据，
--     否则后端 /health 仍会 ok，但所有业务接口都会查不到数据。
--     已有数据的环境请先备份：
--         mysqldump -u root -p service_copilot > backup.sql
--
-- 与后端代码的对应关系（改字段前请先看这几处）
--     robot_devices  ← src/tools/sql_tool.py::query_device_info
--     spare_parts    ← src/tools/sql_tool.py::query_spare_part_stock
--                     src/agents/nodes.py::dispatch_order_node（审批通过后 UPDATE 扣减）
--     work_orders    ← src/tools/sql_tool.py::create_work_order_draft（INSERT 草稿）
--                     src/api/routes.py::list_work_orders / approve_work_order
-- =============================================================================

SET NAMES utf8mb4;

-- -----------------------------------------------------------------------------
-- 一、数据库
-- -----------------------------------------------------------------------------
-- 字符集 utf8mb4 + unicode_ci：需存中文车间名与故障现象，且与 plan.md 第四章一致。
CREATE DATABASE IF NOT EXISTS `service_copilot`
    DEFAULT CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE `service_copilot`;

-- 重建（按依赖逆序：工单 → 库存 → 设备）
DROP TABLE IF EXISTS `work_orders`;
DROP TABLE IF EXISTS `spare_parts`;
DROP TABLE IF EXISTS `robot_devices`;

-- -----------------------------------------------------------------------------
-- 二、表结构
-- -----------------------------------------------------------------------------
-- 2.1 机器人设备档案表
--     device_sn 是全局唯一标识，Webhook 报文与人工报修都靠它定位设备与所在城市；
--     location 决定备件「就近调拨」的优先级（见 query_spare_part_stock 的排序）。
CREATE TABLE `robot_devices` (
    `id`         int          NOT NULL AUTO_INCREMENT,
    `device_sn`  varchar(64)  NOT NULL                COMMENT '设备序列号',
    `model_name` varchar(64)  NOT NULL                COMMENT '机器人型号',
    `category`   varchar(32)  NOT NULL                COMMENT '品类(AGV/机械臂/复合机器人)',
    `location`   varchar(64)  NOT NULL                COMMENT '所在厂区/城市',
    `workshop`   varchar(64)  NOT NULL                COMMENT '车间编号',
    `status`     varchar(32)  DEFAULT 'NORMAL'        COMMENT '当前状态(NORMAL/ERROR/MAINTAIN)',
    `created_at` datetime     DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    -- 设备 SN 索引：UNIQUE 会自带 B+Tree 索引，SN 精确查询（query_device_info）直接命中
    UNIQUE KEY `uk_device_sn` (`device_sn`),
    -- 厂区 + 状态：支撑「某厂区有哪些故障设备」这类看板统计，避免全表扫描
    KEY `idx_device_location` (`location`, `status`),
    KEY `idx_device_model` (`model_name`)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci
  COMMENT = '机器人设备档案表';

-- 2.2 备件分仓实时库存表
--     一行 = 「某备件在某仓库」的库存快照，因此 part_code（物料编码）全局唯一，
--     而 part_name（标准品名）允许重复出现（同一品名分布在多个仓库）。
CREATE TABLE `spare_parts` (
    `id`                 int           NOT NULL AUTO_INCREMENT,
    `part_code`          varchar(64)   NOT NULL              COMMENT '备件物料编码',
    `part_name`          varchar(128)  NOT NULL              COMMENT '备件标准名称',
    `spec_model`         varchar(128)  DEFAULT NULL          COMMENT '规格型号',
    `stock_quantity`     int           NOT NULL DEFAULT '0'  COMMENT '当前库存量',
    `warehouse_name`     varchar(64)   NOT NULL              COMMENT '所属仓库',
    `warehouse_location` varchar(64)   NOT NULL              COMMENT '所在城市/厂区',
    `bin_location`       varchar(64)   DEFAULT NULL          COMMENT '货位号',
    `unit_price`         decimal(10,2) NOT NULL              COMMENT '单价',
    `updated_at`         datetime      DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    UNIQUE KEY `uk_part_code` (`part_code`),
    -- 备件名索引：控制台按品名查询 / 排障后回查同品名库存
    KEY `idx_part_name` (`part_name`),
    -- 品名 + 仓库复合索引：审批通过后的库存扣减
    --   UPDATE spare_parts SET ... WHERE part_name = ? AND warehouse_name = ?
    --   走该索引可精确到行，避免对 2000+ 行做全表扫描
    KEY `idx_part_warehouse` (`part_name`, `warehouse_name`),
    -- 低库存预警 / 缺货筛选：WHERE stock_quantity > 0 / ORDER BY stock_quantity
    KEY `idx_part_stock` (`stock_quantity`)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci
  COMMENT = '备件实时库存表';

-- 2.3 售后维修工单表（HITL 审批与派工的落库对象）
--     approval_status 与 dispatch_status 必须成对出现：
--       PENDING  / INIT       → 草稿已生成，等待主管审批
--       APPROVED / DISPATCHED → 审批通过，库存已核销
--       REJECTED / CLOSED     → 审批驳回，库存不变
CREATE TABLE `work_orders` (
    `id`                 int          NOT NULL AUTO_INCREMENT,
    `order_no`           varchar(64)  NOT NULL                COMMENT '工单编号',
    `device_sn`          varchar(64)  NOT NULL                COMMENT '故障设备SN',
    `error_code`         varchar(32)  DEFAULT NULL            COMMENT '故障代码',
    `symptom`            text                                  COMMENT '故障现象描述',
    `suggested_part`     varchar(128) DEFAULT NULL            COMMENT '推荐更换备件',
    `allocated_warehouse` varchar(64) DEFAULT NULL            COMMENT '出库仓库',
    `approval_status`    varchar(32)  DEFAULT 'PENDING'       COMMENT '审批状态(PENDING/APPROVED/REJECTED)',
    `approver`           varchar(64)  DEFAULT NULL            COMMENT '审批人',
    `dispatch_status`    varchar(32)  DEFAULT 'INIT'          COMMENT '派单状态(INIT/DISPATCHED/CLOSED)',
    `trace_id`           varchar(64)  DEFAULT NULL            COMMENT '全链路追踪ID',
    `created_at`         datetime     DEFAULT CURRENT_TIMESTAMP,
    `updated_at`         datetime     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (`id`),
    -- 工单号唯一：create_work_order_draft 靠它做并发插入冲突重试（IntegrityError）
    UNIQUE KEY `uk_order_no` (`order_no`),
    -- trace_id 与 LangGraph thread_id 同值，审批接口按它定位挂起的工作流
    KEY `idx_order_trace` (`trace_id`),
    -- 设备维度的历史工单回溯：WHERE device_sn = ? ORDER BY created_at DESC
    KEY `idx_order_device` (`device_sn`, `created_at`),
    -- 审批看板按状态倒序分页：WHERE approval_status = ? ORDER BY id DESC
    KEY `idx_order_status` (`approval_status`, `created_at`)
) ENGINE = InnoDB
  DEFAULT CHARSET = utf8mb4
  COLLATE = utf8mb4_0900_ai_ci
  COMMENT = '售后维修工单表';

-- =============================================================================
-- 三、账号与权限隔离（对应需求 FR-3.2 / FR-3.3 与 plan.md 第十章风险缓解）
-- =============================================================================
-- 设计意图
-- ---------
-- Text2SQL 场景的最大风险是大模型生成越权语句（DROP TABLE / UPDATE 全表 / 导出敏感表）。
-- 仅靠「参数化绑定」只能挡住注入，挡不住「大模型自己写出一条合法的写语句」，
-- 因此再加一道**物理权限**防线，形成双重阻断：
--
--   ① 参数化绑定（SQLAlchemy text() + bindparams，见 src/tools/sql_tool.py）
--        → 用户输入永远只是绑定参数，不参与 SQL 文本拼装，杜绝注入；
--   ② 数据库最小权限（本段脚本）
--        → 日常查询走 copilot_agent：只有 3 张表的 SELECT。
--          即使大模型幻觉出 DELETE / UPDATE / DROP / 建表语句，
--          MySQL 也会直接以 "command denied" 拒绝，破坏不可能发生。
--
-- 为什么审批扣减仍用 root
-- ----------------------
-- 「主管审批通过 → 核销库存 + 派工」是**唯一**的设计内写入路径，且它不由大模型生成 SQL，
-- 而是代码里写死的固定模板（src/agents/nodes.py::dispatch_order_node 的 _DEDUCT_STOCK_SQL），
-- 因此保留 root（或高权限账号）跑这条路径是可以接受的权衡，优先保证审批闭环不被打断。
--
-- 生产环境加固建议（本次未启用，避免影响现有演示链路）
--     CREATE USER 'copilot_approver'@'%' IDENTIFIED BY '...';
--     GRANT SELECT, UPDATE ON `service_copilot`.`spare_parts`   TO 'copilot_approver'@'%';
--     GRANT SELECT, UPDATE ON `service_copilot`.`work_orders`    TO 'copilot_approver'@'%';
--   然后给审批路径单独建一个 SQLAlchemy Engine，与只读 Engine 分离，
--   这样 root 就可以彻底退出业务链路。
-- =============================================================================

-- 3.1 日常查询账号：只读，且只读这 3 张表
-- 密码请自行替换为强口令后再执行（此账号仅用于权限隔离演示，应用代码不直接使用它）
CREATE USER IF NOT EXISTS 'copilot_agent'@'%' IDENTIFIED BY 'REPLACE_WITH_STRONG_PASSWORD';
-- 重跑本脚本时把密码重置回已知值，并清空历史授权，保证权限配置幂等且不残留
ALTER USER 'copilot_agent'@'%' IDENTIFIED BY 'REPLACE_WITH_STRONG_PASSWORD';
REVOKE ALL PRIVILEGES, GRANT OPTION FROM 'copilot_agent'@'%';

GRANT SELECT ON `service_copilot`.`robot_devices` TO 'copilot_agent'@'%';
GRANT SELECT ON `service_copilot`.`spare_parts`   TO 'copilot_agent'@'%';
GRANT SELECT ON `service_copilot`.`work_orders`   TO 'copilot_agent'@'%';

-- 3.2 审批写入账号：沿用已有 root
--     root 的权限由 MySQL 实例自身维护（ALL PRIVILEGES），本脚本**刻意不做任何变更**，
--     仅在此说明它承担的职责：审批通过后，copilot_agent 无权执行的两条写语句
--         UPDATE work_orders  SET approval_status/approver/dispatch_status ...
--         UPDATE spare_parts  SET stock_quantity = stock_quantity - 1 ...
--     由 root 连接执行（对应 .env 中的 MYSQL_USER=root）。

FLUSH PRIVILEGES;

-- -----------------------------------------------------------------------------
-- 四、权限自检（需要时手工执行，确认隔离生效）
-- -----------------------------------------------------------------------------
-- 期望：只出现 3 条 "GRANT SELECT ON `service_copilot`.`xxx` TO ..."，没有任何写权限
-- SHOW GRANTS FOR 'copilot_agent'@'%';
--
-- 反证测试（用只读账号执意写库，应报 ERROR 1142 command denied）：
--     DELETE FROM service_copilot.work_orders WHERE id = 1;
