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
  compareSessionId: null, // second session for comparison mode
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
    <button class="tab-btn" data-tab="dashboard" onclick="switchTab('dashboard')">Dashboard</button>
    <button class="tab-btn active" data-tab="chat" onclick="switchTab('chat')">Chat</button>
  </div>
  <div id="chat-area">
    <div id="compare-bar" class="hidden"></div>
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
  <button id="btn-close-upload" onclick="closeUploadOverlay()" title="Back">×</button>
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
  renderCompareBar();
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
  // Clear a stale comparison if it points at the newly-selected session.
  if (state.compareSessionId === id) state.compareSessionId = null;
  renderSessionList();
  renderCompareBar();
  hideUploadOverlay();
  switchTab(state.activeTab);
  if (state.activeTab === 'chat') {
    await loadMessages();
  } else {
    await loadDashboard();
  }
}

// ── Comparison mode ──────────────────────────────────────────────────────────
function renderCompareBar() {
  const bar = $('compare-bar');
  if (!bar) return;
  const others = state.sessions.filter(s => s.session_id !== state.activeSessionId);
  if (!state.activeSessionId || others.length === 0) {
    bar.classList.add('hidden');
    return;
  }
  bar.classList.remove('hidden');
  const opts = ['<option value="">Compare against…</option>']
    .concat(others.map(s =>
      `<option value="${s.session_id}" ${s.session_id === state.compareSessionId ? 'selected' : ''}>${esc(s.filename)}</option>`))
    .join('');
  bar.innerHTML = `
    <span class="compare-label">Compare</span>
    <select id="compare-select" onchange="setCompare(this.value)">${opts}</select>
    ${state.compareSessionId ? `<button onclick="viewDiff()">View diff</button>
      <button class="compare-clear" onclick="setCompare('')">Clear</button>` : ''}`;
}

function setCompare(id) {
  state.compareSessionId = id || null;
  renderCompareBar();
}

async function viewDiff() {
  if (!state.activeSessionId || !state.compareSessionId) return;
  try {
    const res = await apiPost('/compare', {
      base_session_id: state.activeSessionId,
      compare_session_id: state.compareSessionId,
    });
    showDiffModal(res.diff);
  } catch (err) { alert(`Compare failed: ${err.message}`); }
}

function showDiffModal(diff) {
  const numeric = (diff.column_diffs || []).filter(d => 'mean_delta' in d);
  const numRows = numeric.map(d =>
    `<tr><td>${esc(d.column)}</td><td>${esc(d.base_mean)}</td><td>${esc(d.compare_mean)}</td>
     <td class="${d.mean_delta >= 0 ? 'pos' : 'neg'}">${d.mean_delta >= 0 ? '+' : ''}${esc(d.mean_delta)}</td></tr>`).join('');
  const numTable = numeric.length ? `
    <table class="data-table"><thead><tr><th>Column</th><th>Base mean</th><th>Compare mean</th><th>Δ</th></tr></thead>
    <tbody>${numRows}</tbody></table>` : '<p class="diff-empty">No shared numeric columns.</p>';

  const modal = document.createElement('div');
  modal.className = 'modal-overlay visible';
  modal.innerHTML = `
    <div class="modal-box">
      <div class="modal-head">
        <h2>${esc(diff.base_filename)} vs ${esc(diff.compare_filename)}</h2>
        <button onclick="this.closest('.modal-overlay').remove()">×</button>
      </div>
      <div class="modal-body">
        <div class="stats-cards">
          <div class="stat-card"><div class="label">Base rows</div><div class="value">${diff.base_row_count}</div></div>
          <div class="stat-card"><div class="label">Compare rows</div><div class="value">${diff.compare_row_count}</div></div>
          <div class="stat-card"><div class="label">Row Δ</div><div class="value">${diff.row_count_delta >= 0 ? '+' : ''}${diff.row_count_delta}</div></div>
        </div>
        <h2>Numeric changes</h2>
        ${numTable}
        ${(diff.only_in_base || []).length ? `<p class="diff-note">Only in base: ${diff.only_in_base.map(esc).join(', ')}</p>` : ''}
        ${(diff.only_in_compare || []).length ? `<p class="diff-note">Only in compare: ${diff.only_in_compare.map(esc).join(', ')}</p>` : ''}
      </div>
    </div>`;
  document.body.appendChild(modal);
}

// ── Upload ─────────────────────────────────────────────────────────────────
function showUploadOverlay() {
  resetDropZone();                                   // clear any stale "Uploading…" spinner
  const closeBtn = $('btn-close-upload');
  // Only allow closing the overlay if there's an existing session to go back to.
  if (closeBtn) closeBtn.style.display = state.activeSessionId ? 'block' : 'none';
  $uploadOverlay().style.display = 'flex';
}
function hideUploadOverlay() { $uploadOverlay().style.display = 'none'; }
function closeUploadOverlay() { if (state.activeSessionId) hideUploadOverlay(); }

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
  dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('drag-over'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('drag-over'));
  dz.addEventListener('drop', e => {
    e.preventDefault(); dz.classList.remove('drag-over');
    const f = e.dataTransfer.files[0];
    if (f) uploadFile(f);
  });
}

function showSheetPicker(sheets) {
  const picker = $sheetPicker();
  const list = $('sheet-list');
  list.innerHTML = sheets.map((s, i) => {
    _sheetNames[i] = s;
    return `<div class="sheet-option" onclick="selectSheet(_sheetNames[${i}])">${esc(s)}</div>`;
  }).join('');
  picker.classList.add('visible');
}
const _sheetNames = [];

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
      appendMessage(m.role, m.content, m.chart || null,
                    m.follow_ups || [], m.caveats || [], m.generated_code || null, m.table || null);
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
      compare_session_id: state.compareSessionId,
    });
    document.getElementById(thinkId)?.remove();
    appendMessage('assistant', data.content, data.chart,
                  data.follow_ups || [], data.caveats || [], data.generated_code, data.table);
    state.history.push({ role: 'assistant', content: data.content });
  } catch (err) {
    document.getElementById(thinkId)?.remove();
    appendMessage('assistant', `Error: ${err.message}`, null, [], []);
  }
  $btnSend().disabled = false;
  scrollMessages();
}

function renderTable(table) {
  // table is {rows: [...]|{...}, title} — render the deterministic facts as a grid.
  if (!table) return '';
  let rows = table.rows;
  if (!rows) return '';
  // Normalise dict-of-values into rows of {key, value}.
  if (!Array.isArray(rows)) {
    rows = Object.entries(rows).map(([k, v]) => ({ metric: k, value: v }));
  }
  if (!rows.length || typeof rows[0] !== 'object') return '';
  const cols = Object.keys(rows[0]);
  const head = cols.map(c => `<th>${esc(c)}</th>`).join('');
  const body = rows.slice(0, 50).map(r =>
    `<tr>${cols.map(c => `<td>${esc(r[c])}</td>`).join('')}</tr>`).join('');
  return `
    <div class="data-table-wrap">
      <table class="data-table">
        <thead><tr>${head}</tr></thead>
        <tbody>${body}</tbody>
      </table>
    </div>`;
}

function appendMessage(role, content, chart, followUps, caveats, code, table) {
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

  const tableHtml = renderTable(table);

  msgs.insertAdjacentHTML('beforeend', `
    <div class="message ${role}">
      <div class="message-bubble">${nl2br(esc(content))}</div>
      ${chartHtml}
      ${tableHtml}
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
    const [dash, ins, pins] = await Promise.all([
      apiGet(`/sessions/${state.activeSessionId}/dashboard`),
      apiGet(`/sessions/${state.activeSessionId}/insights`).catch(() => []),
      apiGet(`/sessions/${state.activeSessionId}/pins`).catch(() => []),
    ]);
    renderDashboard(area, dash, ins, pins);
  } catch (err) {
    area.innerHTML = `<p style="padding:20px;color:red">Failed to load dashboard: ${esc(err.message)}</p>`;
  }
}

function renderDashboard(area, dash, insights, pins) {
  pins = pins || [];
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

  const pinnedHtml = pins.filter(p => p.png_b64).map(p => {
    const src = `data:image/png;base64,${p.png_b64}`;
    const cid = _storeChart(p.title, src);
    return `
    <div class="dashboard-chart-card">
      <div class="dashboard-chart-title">${esc(p.title || 'Pinned chart')}</div>
      <img src="${src}" alt="${esc(p.title || '')}"
           onclick="openCanvasById(${cid})" style="cursor:pointer">
      <button onclick="unpinChart('${p.pin_id}')" style="margin-top:6px">Unpin</button>
    </div>`;
  }).join('');
  const pinnedSection = pinnedHtml
    ? `<h2 style="margin-top:24px">Pinned charts</h2><div class="dashboard-charts">${pinnedHtml}</div>`
    : '';

  const insightsHtml = insights.length ? `
    <div class="insights-section">
      <h2>Key Findings</h2>
      ${insights.map(i => `
        <div class="insight-card">
          <div class="insight-title">${i.rank}. ${esc(i.title)}</div>
          <div class="insight-summary">${esc(i.summary)}</div>
        </div>`).join('')}
    </div>` : '';

  // ── Data quality (from quality_flags already in the response — no extra calls) ──
  const q = dash.quality_flags || {};
  const qItems = [];
  if (q.duplicate_rows) qItems.push(`${q.duplicate_rows} duplicate row${q.duplicate_rows > 1 ? 's' : ''}`);
  if ((q.mostly_empty_columns || []).length) qItems.push(`${q.mostly_empty_columns.length} mostly-empty column${q.mostly_empty_columns.length > 1 ? 's' : ''}`);
  if ((q.constant_columns || []).length) qItems.push(`${q.constant_columns.length} constant column${q.constant_columns.length > 1 ? 's' : ''}`);
  if ((q.fuzzy_category_issues || []).length) qItems.push(`${q.fuzzy_category_issues.length} column${q.fuzzy_category_issues.length > 1 ? 's' : ''} with near-duplicate values`);
  const qualityHtml = `
    <div class="quality-card ${qItems.length ? 'has-issues' : 'clean'}">
      <span class="quality-title">${qItems.length ? '⚠ Data quality' : '✓ Data quality'}</span>
      <span class="quality-body">${qItems.length ? qItems.join(' · ') : 'No issues detected'}</span>
    </div>`;

  // ── Column overview (from column_summary — collapsible to avoid clutter) ──
  const cs = dash.column_summary || {};
  const colRows = Object.entries(cs).map(([name, c]) => `
    <tr><td>${esc(name)}</td><td>${esc(c.dtype)}</td>
        <td>${Math.round(c.missing_pct)}%</td><td>${esc(c.n_unique)}</td></tr>`).join('');
  const columnsHtml = Object.keys(cs).length ? `
    <details class="columns-details">
      <summary>Column overview (${Object.keys(cs).length})</summary>
      <div class="data-table-wrap">
        <table class="data-table">
          <thead><tr><th>Column</th><th>Type</th><th>Missing</th><th>Unique</th></tr></thead>
          <tbody>${colRows}</tbody>
        </table>
      </div>
    </details>` : '';

  area.innerHTML = `
    <div class="stats-cards">
      <div class="stat-card"><div class="label">Rows</div><div class="value">${dash.row_count}</div></div>
      <div class="stat-card"><div class="label">Columns</div><div class="value">${dash.col_count}</div></div>
      <div class="stat-card"><div class="label">File</div><div class="value" style="font-size:14px">${esc(dash.filename)}</div></div>
    </div>
    ${qualityHtml}
    ${dash.narrative ? `<p class="dash-narrative">${esc(dash.narrative)}</p>` : ''}
    <div class="dashboard-charts">${chartsHtml}</div>
    ${columnsHtml}
    ${pinnedSection}
    ${insightsHtml}
    <button class="export-btn" onclick="exportPdf()">Export PDF Report</button>`;
}

async function unpinChart(pinId) {
  if (!state.activeSessionId) return;
  try {
    await apiDelete(`/sessions/${state.activeSessionId}/pins/${pinId}`);
    await loadDashboard();
  } catch (err) { alert(`Unpin failed: ${err.message}`); }
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
