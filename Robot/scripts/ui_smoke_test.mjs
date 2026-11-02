/**
 * 用 Edge DevTools Protocol 驱动控制台，验证「点按钮 → 弹审批卡 → 批准 → 库存扣减」整条链路。
 * 只用 Node 内置能力（fetch / WebSocket），无需安装任何依赖。
 *
 * 产出：
 *   $TEMP/copilot-hitl-modal.png   审批卡片弹出时的界面
 *   $TEMP/copilot-approved.png     批准后库存扣减的界面
 */
import { spawn } from 'node:child_process';
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';

const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const PORT = 9333;
const URL = 'http://127.0.0.1:8000/';
const OUT_DIR = os.tmpdir();
const profile = path.join(OUT_DIR, 'edge-cdp-profile');

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// 每次跑都用全新 profile：避免浏览器磁盘缓存命中旧的 index.html
await fs.rm(profile, { recursive: true, force: true });

const edge = spawn(
  EDGE,
  [
    '--headless=new',
    '--disable-gpu',
    '--hide-scrollbars',
    '--no-first-run',
    `--remote-debugging-port=${PORT}`,
    `--user-data-dir=${profile}`,
    '--window-size=1680,1000',
    'about:blank',
  ],
  { stdio: 'ignore', detached: false },
);

async function pageWsUrl() {
  for (let i = 0; i < 60; i += 1) {
    try {
      const res = await fetch(`http://127.0.0.1:${PORT}/json/list`);
      const tabs = await res.json();
      const page = tabs.find((t) => t.type === 'page');
      if (page && page.webSocketDebuggerUrl) return page.webSocketDebuggerUrl;
    } catch {
      /* devtools 还没起来 */
    }
    await sleep(300);
  }
  throw new Error('DevTools 端口未就绪');
}

function makeCdp(ws) {
  let seq = 0;
  const pending = new Map();
  ws.addEventListener('message', (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.id && pending.has(msg.id)) {
      const { resolve, reject } = pending.get(msg.id);
      pending.delete(msg.id);
      if (msg.error) reject(new Error(JSON.stringify(msg.error)));
      else resolve(msg.result);
    }
  });
  return (method, params = {}) =>
    new Promise((resolve, reject) => {
      const id = ++seq;
      pending.set(id, { resolve, reject });
      ws.send(JSON.stringify({ id, method, params }));
    });
}

const results = [];
function check(name, ok, detail) {
  results.push({ name, ok, detail });
  console.log(`  ${ok ? '✓' : '✗'} ${name}${detail ? ' — ' + detail : ''}`);
}

const wsUrl = await pageWsUrl();
const ws = new WebSocket(wsUrl);
await new Promise((resolve, reject) => {
  ws.addEventListener('open', resolve, { once: true });
  ws.addEventListener('error', reject, { once: true });
});
const send = makeCdp(ws);

await send('Page.enable');
await send('Runtime.enable');
await send('Page.navigate', { url: URL });
await sleep(6000); // 等页面首屏 + 首轮 api 拉取完成

async function evaluate(expression) {
  const out = await send('Runtime.evaluate', {
    expression,
    returnByValue: true,
    awaitPromise: true,
  });
  if (out.exceptionDetails) {
    throw new Error(out.exceptionDetails.exception?.description || 'JS 执行异常');
  }
  return out.result.value;
}

async function screenshot(name) {
  const shot = await send('Page.captureScreenshot', { format: 'png' });
  const file = path.join(OUT_DIR, name);
  await fs.writeFile(file, Buffer.from(shot.data, 'base64'));
  return file;
}

async function waitFor(expression, timeoutMs = 90000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await evaluate(expression)) return true;
    await sleep(400);
  }
  return false;
}

// ---------------- 1. 首屏检查 ----------------
const boot = await evaluate(`(() => ({
  title: document.title,
  tailwind: typeof window.tailwind,
  partMeta: document.getElementById('partMeta').textContent,
  orderMeta: document.getElementById('orderMeta').textContent,
  partsRows: document.querySelectorAll('#partsBody tr').length,
  firstPartNames: Array.from(document.querySelectorAll('#partsBody tr td:first-child')).slice(0, 6).map(t => t.textContent),
  healthOk: Array.from(document.querySelectorAll('[data-health] .dot')).map(d => d.className.includes('dot--ok')),
}))()`);

console.log('\n[1] 首屏渲染');
check('页面标题正确', boot.title.includes('Service Copilot'), boot.title);
check('Tailwind CDN 已加载', boot.tailwind !== 'undefined', 'typeof tailwind=' + boot.tailwind);
check('备件表已渲染', boot.partsRows > 0, `行数=${boot.partsRows}，meta=${boot.partMeta}`);
check(
  '核心备件「型号A电磁阀」在首屏可见',
  boot.firstPartNames.includes('型号A电磁阀'),
  '前 6 行=' + JSON.stringify(boot.firstPartNames.slice(0, 6)),
);
check('工单表已渲染', parseInt(boot.orderMeta, 10) > 0, 'meta=' + boot.orderMeta);
check('三项依赖健康（MySQL/Qdrant/Checkpoint）', boot.healthOk.length === 3 && boot.healthOk.every(Boolean), JSON.stringify(boot.healthOk));

// 记录审批前库存
const stockBefore = await evaluate(`(() => {
  const rows = Array.from(document.querySelectorAll('#partsBody tr'));
  const row = rows.find(r => r.children[0].textContent === '型号A电磁阀' && r.children[2].textContent.includes('苏州中心仓'));
  return row ? Number(row.children[3].textContent.trim().split(/\\s+/)[0]) : null;
})()`);
check('审批前 苏州中心仓 型号A电磁阀库存 = 3', stockBefore === 3, '实际=' + stockBefore);

// ---------------- 2. 点击通道 A（E-201） ----------------
console.log('\n[2] 点击【模拟人工报修 E-201】');
await evaluate(`document.getElementById('btnE201').click()`);

const modalOpen = await waitFor(`document.getElementById('approvalOverlay').classList.contains('is-open')`);
check('WAITING_APPROVAL 后弹出 HITL 审批卡片', modalOpen);

const modal = await evaluate(`(() => ({
  orderNo: document.getElementById('moOrder').textContent,
  part: document.getElementById('moPart').textContent,
  warehouse: document.getElementById('moWarehouse').textContent,
  stock: document.getElementById('moStock').textContent,
  trace: document.getElementById('moTrace').textContent,
  steps: Array.from(document.querySelectorAll('#timeline .tl-item .mb-1')).map(n => n.textContent),
  approveBtn: document.getElementById('btnApprove').textContent.trim(),
  rejectBtn: document.getElementById('btnReject').textContent.trim(),
}))()`);

console.log('   工单号=' + modal.orderNo + ' 备件=' + modal.part + ' 仓库=' + modal.warehouse + ' 库存=' + modal.stock);
console.log('   推理流步骤：' + modal.steps.join(' → '));
check('审批卡展示工单号', /^WO-\d{6}-\d{4}$/.test(modal.orderNo), modal.orderNo);
check('审批卡展示申请备件', modal.part === '型号A电磁阀', modal.part);
check('审批卡展示推荐仓库', modal.warehouse.includes('苏州中心仓'), modal.warehouse);
check('审批卡展示扣减预览', /3\s*件/.test(modal.stock), modal.stock);
check('推理流含 6 个阶段', modal.steps.length >= 6, modal.steps.length + ' 步');
check('存在【批准并扣减库存】按钮', modal.approveBtn.includes('批准并扣减库存'), modal.approveBtn);
check('存在【驳回】按钮', modal.rejectBtn.includes('驳回'), modal.rejectBtn);

const shotModal = await screenshot('copilot-hitl-modal.png');
console.log('   截图：' + shotModal);

// ---------------- 3. 点击批准 ----------------
console.log('\n[3] 点击【批准并扣减库存】');
await evaluate(`document.getElementById('btnApprove').click()`);
const done = await waitFor(`!document.getElementById('moDone').classList.contains('hidden')`);
check('审批返回并渲染结果态', done);

const after = await evaluate(`(() => {
  const rows = Array.from(document.querySelectorAll('#partsBody tr'));
  const row = rows.find(r => r.children[0].textContent === '型号A电磁阀' && r.children[2].textContent.includes('苏州中心仓'));
  const orderRow = Array.from(document.querySelectorAll('#ordersBody tr')).find(r => r.textContent.includes(document.getElementById('moOrder').textContent));
  return {
    stockNow: row ? Number(row.children[3].textContent.trim().split(/\\s+/)[0]) : null,
    flashClass: row ? row.className : '',
    badge: document.getElementById('moDoneBadge').textContent,
    detail: document.getElementById('moDoneDetail').textContent,
    orderBadges: orderRow ? Array.from(orderRow.querySelectorAll('.badge')).map(b => b.textContent) : [],
    lastStep: (document.querySelector('#timeline .tl-item:last-child .mb-1') || {}).textContent || '',
    partMeta: document.getElementById('partMeta').textContent,
  };
})()`);

console.log('   派工徽章=' + after.badge);
console.log('   结果文案=' + after.detail);
console.log('   工单表徽章=' + JSON.stringify(after.orderBadges));
check('派工状态 = DISPATCHED', after.badge === 'DISPATCHED', after.badge);
check('结果文案包含库存核销信息', /核销/.test(after.detail), after.detail);
check('库存精准扣减 3 → 2', after.stockNow === 2, '实际=' + after.stockNow);
check('变动行触发闪烁高亮', /flash-down/.test(after.flashClass), after.flashClass);
check('工单表同步为 APPROVED / DISPATCHED', JSON.stringify(after.orderBadges) === JSON.stringify(['APPROVED', 'DISPATCHED']), JSON.stringify(after.orderBadges));
check('推理流追加审批闭环步骤', after.lastStep.includes('人工审批闭环') || after.lastStep.includes('结案'), after.lastStep);

const shotApproved = await screenshot('copilot-approved.png');
console.log('   截图：' + shotApproved);

// ---------------- 4. 通道 B（E-500）无需审批 ----------------
console.log('\n[4] 关闭弹窗后点击【模拟设备 Webhook 告警 E-500】');
await evaluate(`document.getElementById('modalClose').click()`);
await sleep(300);
await evaluate(`document.getElementById('btnE500').click()`);
const e500 = await waitFor(`document.getElementById('tlMeta').textContent.includes('已完成')`);
const e500State = await evaluate(`(() => ({
  meta: document.getElementById('tlMeta').textContent,
  modalOpen: document.getElementById('approvalOverlay').classList.contains('is-open'),
  steps: Array.from(document.querySelectorAll('#timeline .tl-item .mb-1')).map(n => n.textContent),
}))()`);
check('E-500 直接完成、不弹审批卡', e500 && !e500State.modalOpen, 'meta=' + e500State.meta);
check('E-500 推理流无工单步骤', e500State.steps.some((s) => s.includes('无需换件')), JSON.stringify(e500State.steps));

// ---------------- 汇总 ----------------
const shotFinal = await screenshot('copilot-e500.png');
console.log('   截图：' + shotFinal);

ws.close();
edge.kill();

const failed = results.filter((r) => !r.ok);
console.log(`\n通过 ${results.length - failed.length}/${results.length}`);
if (failed.length) process.exitCode = 1;
