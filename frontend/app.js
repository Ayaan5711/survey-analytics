'use strict';

// ── Chart store (XSS-safe onclick) ────────────────────────────────────────
const _charts = new Map();
let _chartSeq = 0;

function _storeChart(title, src) {
  const id = ++_chartSeq;
  _charts.set(id, { title, src });
  return id;
}

function openCanvasById(id) {
  const c = _charts.get(id);
  if (c) openCanvas(c.title, c.src);
}

function pinChartById(id) {
  const c = _charts.get(id);
  if (c) pinChart(c.src, c.title);
}

// ── State ──────────────────────────────────────────────────────────────────
const state = {
  sessions: [],
  activeSessionId: null,
  activeTab: 'chat',      // 'chat' | 'dashboard'
  history: [],            // [{role, content}, ...]
  uploadedFile: null,     // File object pending sheet selection
  uploadedFileSheets: [], // sheet names if multi-sheet Excel
};

// ── API helpers ────────────────────────────────────────────────────────────
async function apiGet(path) {
  const r = await fetch(`/api${path}`);
  if (!r.ok) throw new Error(`GET ${path} → ${r.status}`);
  return r.json();
}

async function apiPost(path, body) {
  const r = await fetch(`/api${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.detail || `POST ${path} → ${r.status}`);
  }
  return r.json();
}

async function apiPostForm(path, formData) {
  const r = await fetch(`/api${path}`, { method: 'POST', body: formData });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    // Return the raw response so callers can check status
    const detail = typeof err.detail === 'object' ? err.detail : { message: err.detail };
    const e = new Error(detail.message || `POST ${path} → ${r.status}`);
    e.status = r.status;
    e.detail = detail;
    throw e;
  }
  return r.json();
}

async function apiDelete(path) {
  const r = await fetch(`/api${path}`, { method: 'DELETE' });
  if (!r.ok) throw new Error(`DELETE ${path} → ${r.status}`);
  return r.json();
}

// ── DOM refs ───────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const $uploadOverlay  = () => $('upload-overlay');
const $sheetPicker    = () => $('sheet-picker');
const $sessionList    = () => $('session-list');
const $messages       = () => $('messages');
const $chatInput      = () => $('chat-input');
const $btnSend        = () => $('btn-send');
const $dashboardArea  = () => $('dashboard-area');
const $canvasPanel    = () => $('canvas-panel');
const $canvasTitle    = () => $('canvas-title');
const $canvasImg      = () => $('canvas-img');

// ── Bootstrap ──────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  buildShell();
  await loadSessions();
  bindUpload();
  bindChat();
  bindCanvas();
  bindTabs();
});

function buildShell() {
  document.body.innerHTML = `
<div id="sidebar">
  <div id="sidebar-header">
    Sessions
    <button id="btn-new-session" onclick="showUploadOverlay()">+ New</button>
  </div>
  <div id="session-list"></div>
</div>
<div id="main">
  <div id="tab-bar">
    <button class="tab-btn active" data-tab="chat" onclick="switchTab('chat')">Chat</button>
    <button class="tab-btn" data-tab="dashboard" onclick="switchTab('dashboard')">Dashboard</button>
  </div>
  <div id="chat-area">
    <div id="messages"></div>
    <div id="chat-input-area">
      <textarea id="chat-input" placeholder="Ask a question about your data…" rows="1"></textarea>
      <button id="btn-send" onclick="sendMessage()">Send</button>
    </div>
  </div>
  <div id="dashboard-area" class="hidden"></div>
</div>
<div id="canvas-panel">
  <div id="canvas-header">
    <span id="canvas-title">Chart</span>
    <button id="btn-close-canvas" onclick="closeCanvas()">×</button>
  </div>
  <div id="canvas-body"><img id="canvas-img" src="" alt=""></div>
  <div id="canvas-actions">
    <button onclick="downloadCanvas()">Download PNG</button>
    <button onclick="pinCanvas()">Pin to Dashboard</button>
  </div>
</div>
<div id="upload-overlay">
  <h1>Survey Analytics</h1>
  <p>Upload a CSV or Excel file to begin.</p>
  <div id="drop-zone">
    <span>Drag & drop your file here</span>
    <button onclick="$('file-input').click()">Browse</button>
    <input type="file" id="file-input" accept=".csv,.xlsx,.xls">
  </div>
</div>
<div id="sheet-picker">
  <div id="sheet-picker-box">
    <h2>Select a sheet</h2>
    <div id="sheet-list"></div>
  </div>
</div>`;
}

// ── Session management ─────────────────────────────────────────────────────
async function loadSessions() {
  try {
    state.sessions = await apiGet('/sessions');
  } catch { state.sessions = []; }
  renderSessionList();
  if (state.sessions.length === 0) {
    showUploadOverlay();
  } else if (!state.activeSessionId) {
    await selectSession(state.sessions[0].session_id);
  }
}

function renderSessionList() {
  const el = $sessionList();
  if (!el) return;
  el.innerHTML = state.sessions.map(s => `
    <div class="session-item ${s.session_id === state.activeSessionId ? 'active' : ''}"
         onclick="selectSession('${s.session_id}')">
      <div class="session-name">${esc(s.filename)}</div>
      <div class="session-meta">${s.row_count} rows · ${fmtDate(s.uploaded_at)}</div>
    </div>`).join('');
}

async function selectSession(id) {
  state.activeSessionId = id;
  state.history = [];
  renderSessionList();
  hideUploadOverlay();
  switchTab(state.activeTab);
  if (state.activeTab === 'chat') {
    await loadMessages();
  } else {
    await loadDashboard();
  }
}

// ── Upload ─────────────────────────────────────────────────────────────────
function showUploadOverlay() { $uploadOverlay().style.display = 'flex'; }
function hideUploadOverlay() { $uploadOverlay().style.display = 'none'; }

function bindUpload() {
  const dz = $('drop-zone');
  const fi = $('file-input');
  if (!dz || !fi) return;

  dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('drag-over'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('drag-over'));
  dz.addEventListener('drop', e => {
    e.preventDefault(); dz.classList.remove('drag-over');
    const f = e.dataTransfer.files[0];
    if (f) uploadFile(f);
  });
  fi.addEventListener('change', () => { if (fi.files[0]) uploadFile(fi.files[0]); fi.value = ''; });
}

async function uploadFile(file) {
  const dz = $('drop-zone');
  dz.innerHTML = '<div class="spinner"></div><span>Uploading…</span>';

  const fd = new FormData();
  fd.append('file', file);
  try {
    const data = await apiPostForm('/upload', fd);
    await loadSessions();
    await selectSession(data.session_id);
    // Show dashboard auto-generated content first
    switchTab('dashboard');
  } catch (err) {
    if (err.status === 409 && err.detail?.sheets) {
      state.uploadedFile = file;
      state.uploadedFileSheets = err.detail.sheets;
      showSheetPicker(err.detail.sheets);
    } else {
      dz.innerHTML = `<span style="color:red">${esc(err.message)}</span><button onclick="resetDropZone()">Try again</button>`;
    }
  }
}

function resetDropZone() {
  const dz = $('drop-zone');
  dz.innerHTML = `<span>Drag & drop your file here</span>
    <button onclick="$('file-input').click()">Browse</button>
    <input type="file" id="file-input" accept=".csv,.xlsx,.xls">`;
  $('file-input').addEventListener('change', () => {
    const fi = $('file-input');
    if (fi.files[0]) uploadFile(fi.files[0]);
    fi.value = '';
  });
}

function showSheetPicker(sheets) {
  const picker = $sheetPicker();
  const list = $('sheet-list');
  list.innerHTML = sheets.map(s =>
    `<div class="sheet-option" onclick="selectSheet('${esc(s)}')">${esc(s)}</div>`
  ).join('');
  picker.classList.add('visible');
}

async function selectSheet(sheetName) {
  $sheetPicker().classList.remove('visible');
  const fd = new FormData();
  fd.append('file', state.uploadedFile);
  try {
    const data = await apiPostForm(`/upload/sheet?sheet_name=${encodeURIComponent(sheetName)}`, fd);
    await loadSessions();
    await selectSession(data.session_id);
    switchTab('dashboard');
  } catch (err) {
    alert(`Upload failed: ${err.message}`);
  }
}

// ── Chat ───────────────────────────────────────────────────────────────────
function bindChat() {
  const input = $chatInput();
  if (!input) return;
  input.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = `${Math.min(input.scrollHeight, 160)}px`;
  });
}

async function loadMessages() {
  if (!state.activeSessionId) return;
  const msgs = $messages();
  if (!msgs) return;
  msgs.innerHTML = '';
  state.history = [];
  try {
    const data = await apiGet(`/sessions/${state.activeSessionId}/messages`);
    for (const m of data) {
      appendMessage(m.role, m.content, null, m.follow_ups || [], m.caveats || []);
      state.history.push({ role: m.role, content: m.content });
    }
  } catch { /* first session, no messages */ }
}

async function sendMessage() {
  const input = $chatInput();
  const msg = input.value.trim();
  if (!msg || !state.activeSessionId) return;
  input.value = '';
  input.style.height = 'auto';
  $btnSend().disabled = true;

  appendMessage('user', msg, null, [], []);
  state.history.push({ role: 'user', content: msg });

  // Thinking indicator
  const thinkId = `think-${Date.now()}`;
  $messages().insertAdjacentHTML('beforeend',
    `<div id="${thinkId}" class="message assistant">
       <div class="message-bubble"><span class="spinner"></span> Analysing…</div>
     </div>`);
  scrollMessages();

  try {
    const data = await apiPost(`/sessions/${state.activeSessionId}/chat`, {
      message: msg,
      history: state.history.slice(-20),
    });
    document.getElementById(thinkId)?.remove();
    appendMessage('assistant', data.content, data.chart,
                  data.follow_ups || [], data.caveats || [], data.generated_code);
    state.history.push({ role: 'assistant', content: data.content });
  } catch (err) {
    document.getElementById(thinkId)?.remove();
    appendMessage('assistant', `Error: ${err.message}`, null, [], []);
  }
  $btnSend().disabled = false;
  scrollMessages();
}

function appendMessage(role, content, chart, followUps, caveats, code) {
  const msgs = $messages();
  if (!msgs) return;

  let chartHtml = '';
  if (chart?.png_b64) {
    const src = `data:image/png;base64,${chart.png_b64}`;
    const title = chart.title || 'Chart';
    const cid = _storeChart(title, src);
    chartHtml = `
      <div class="message-chart" data-chart-id="${cid}">
        <img src="${src}" alt="${esc(title)}">
      </div>
      <div class="chart-actions">
        <button onclick="openCanvasById(${cid})">Expand</button>
        <button onclick="downloadPng('${src}', '${esc(title)}')">Download</button>
        <button onclick="pinChartById(${cid})">Pin</button>
      </div>`;
  }

  const codeHtml = code ? `
    <button class="show-code-btn" onclick="toggleCode(this)">Show code</button>
    <pre class="code-block">${esc(code)}</pre>` : '';

  const caveatHtml = caveats.length
    ? `<div class="caveats">⚠ ${caveats.map(esc).join(' · ')}</div>` : '';

  const fuHtml = followUps.length
    ? `<div class="follow-ups">${followUps.map(f =>
        `<button class="follow-up-chip" onclick="useFollowUp(this)">${esc(f)}</button>`
      ).join('')}</div>` : '';

  msgs.insertAdjacentHTML('beforeend', `
    <div class="message ${role}">
      <div class="message-bubble">${nl2br(esc(content))}</div>
      ${chartHtml}
      ${codeHtml}
      ${caveatHtml}
      ${fuHtml}
    </div>`);
  scrollMessages();
}

function scrollMessages() {
  const msgs = $messages();
  if (msgs) msgs.scrollTop = msgs.scrollHeight;
}

function toggleCode(btn) {
  const block = btn.nextElementSibling;
  block.classList.toggle('visible');
  btn.textContent = block.classList.contains('visible') ? 'Hide code' : 'Show code';
}

function useFollowUp(btn) {
  const input = $chatInput();
  if (input) { input.value = btn.textContent; input.focus(); }
}

// ── Dashboard Tab ──────────────────────────────────────────────────────────
async function loadDashboard() {
  const area = $dashboardArea();
  if (!area || !state.activeSessionId) return;
  area.innerHTML = '<div class="spinner" style="margin:40px auto;display:block"></div>';
  try {
    const [dash, ins] = await Promise.all([
      apiGet(`/sessions/${state.activeSessionId}/dashboard`),
      apiGet(`/sessions/${state.activeSessionId}/insights`).catch(() => []),
    ]);
    renderDashboard(area, dash, ins);
  } catch (err) {
    area.innerHTML = `<p style="padding:20px;color:red">Failed to load dashboard: ${esc(err.message)}</p>`;
  }
}

function renderDashboard(area, dash, insights) {
  const chartsHtml = (dash.charts || []).map(c => {
    const src = `data:image/png;base64,${c.png_b64}`;
    const cid = _storeChart(c.title, src);
    return `
    <div class="dashboard-chart-card">
      <div class="dashboard-chart-title">${esc(c.title)}</div>
      <img src="${src}" alt="${esc(c.title)}"
           onclick="openCanvasById(${cid})" style="cursor:pointer">
    </div>`;
  }).join('');

  const insightsHtml = insights.length ? `
    <div class="insights-section">
      <h2>Key Findings</h2>
      ${insights.map(i => `
        <div class="insight-card">
          <div class="insight-title">${i.rank}. ${esc(i.title)}</div>
          <div class="insight-summary">${esc(i.summary)}</div>
        </div>`).join('')}
    </div>` : '';

  area.innerHTML = `
    <div class="stats-cards">
      <div class="stat-card"><div class="label">Rows</div><div class="value">${dash.row_count}</div></div>
      <div class="stat-card"><div class="label">Columns</div><div class="value">${dash.col_count}</div></div>
      <div class="stat-card"><div class="label">File</div><div class="value" style="font-size:14px">${esc(dash.filename)}</div></div>
    </div>
    ${dash.narrative ? `<p style="font-size:14px;line-height:1.7;margin-bottom:20px;max-width:700px">${esc(dash.narrative)}</p>` : ''}
    <div class="dashboard-charts">${chartsHtml}</div>
    ${insightsHtml}
    <button class="export-btn" onclick="exportPdf()">Export PDF Report</button>`;
}

async function exportPdf() {
  if (!state.activeSessionId) return;
  const btn = document.querySelector('.export-btn');
  if (btn) btn.textContent = 'Generating…';
  try {
    const r = await fetch(`/api/sessions/${state.activeSessionId}/export/pdf`);
    if (!r.ok) throw new Error(`Export failed: ${r.status}`);
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = `report_${state.activeSessionId}.pdf`;
    a.click();
    URL.revokeObjectURL(url);
  } catch (err) { alert(err.message); }
  finally { if (btn) btn.textContent = 'Export PDF Report'; }
}

// ── Canvas panel ───────────────────────────────────────────────────────────
let _canvasCurrentSrc = '';
let _canvasCurrentTitle = '';

function bindCanvas() {}

function openCanvas(title, src) {
  _canvasCurrentSrc = src;
  _canvasCurrentTitle = title;
  $canvasTitle().textContent = title;
  $canvasImg().src = src;
  $canvasPanel().classList.add('open');
}

function closeCanvas() { $canvasPanel().classList.remove('open'); }

function downloadCanvas() {
  if (!_canvasCurrentSrc) return;
  downloadPng(_canvasCurrentSrc, _canvasCurrentTitle);
}

function downloadPng(src, title) {
  const a = document.createElement('a');
  a.href = src;
  a.download = `${title.replace(/\s+/g, '_')}.png`;
  a.click();
}

async function pinCanvas() {
  if (!state.activeSessionId || !_canvasCurrentSrc) return;
  await pinChart(_canvasCurrentSrc, _canvasCurrentTitle);
}

async function pinChart(src, title) {
  if (!state.activeSessionId) return;
  const png_b64 = src.replace('data:image/png;base64,', '');
  try {
    await apiPost(`/sessions/${state.activeSessionId}/pins`, { png_b64, title });
    alert(`Pinned: ${title}`);
  } catch (err) { alert(`Pin failed: ${err.message}`); }
}

// ── Tabs ───────────────────────────────────────────────────────────────────
function bindTabs() {}

function switchTab(tab) {
  state.activeTab = tab;
  const chatArea = $('chat-area');
  const dashArea = $dashboardArea();
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
  if (tab === 'chat') {
    if (chatArea) chatArea.style.display = 'flex';
    if (dashArea) dashArea.classList.add('hidden');
    if (state.activeSessionId) loadMessages();
  } else {
    if (chatArea) chatArea.style.display = 'none';
    if (dashArea) dashArea.classList.remove('hidden');
    if (state.activeSessionId) loadDashboard();
  }
}

// ── Utilities ──────────────────────────────────────────────────────────────
function esc(str) {
  return String(str ?? '')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
function nl2br(str) { return str.replace(/\n/g, '<br>'); }
function fmtDate(iso) {
  if (!iso) return '';
  try { return new Date(iso).toLocaleDateString(); } catch { return ''; }
}
