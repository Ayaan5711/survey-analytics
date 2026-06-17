# Survey Analytics — Plan 4: Frontend (Vanilla JS/HTML/CSS)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the black & white minimal browser UI — upload flow, live chat stream, inline charts, dashboard tab with insights, session sidebar, and PDF export button — served as static files from the FastAPI app.

**Architecture:** Single-page vanilla JS app (no build step, no npm). FastAPI serves the `frontend/` directory as static files at `/`. The JS communicates with the existing `/api/` endpoints. Layout: sidebar (session list) + main area (chat or dashboard tab) + collapsible canvas panel (expanded chart view).

**Tech Stack:** HTML5, vanilla JS (ES2022, no framework), CSS custom properties for theming, `fetch()` API, Base64 image rendering. FastAPI `StaticFiles` mount + `aiofiles` for file serving.

## Global Constraints

- No npm, no build step, no TypeScript — pure `.html`, `.js`, `.css`
- Black & white palette only: `#000`, `#fff`, `#111`, `#eee`, `#555`, `#ccc`
- `aiofiles>=23.0` must be added to `pyproject.toml` for `StaticFiles`
- All API calls go to `/api/` (same origin, no CORS needed)
- Tested manually in browser; automated test only verifies static files are served
- Run tests from `backend/`: `pytest tests/ -v`

---

## File Map

```text
backend/
  app/
    main.py                   — mount StaticFiles at "/" (modify)
  pyproject.toml              — add aiofiles>=23.0 (modify)
frontend/
  index.html                  — shell: sidebar + main area + canvas panel
  styles.css                  — black & white minimal theme; CSS grid layout
  app.js                      — all JS: state, API calls, DOM rendering
tests/
  test_static.py              — smoke: GET / returns 200; GET /app.js returns 200
```

---

## Task 1: Static File Setup

**Files:**
- Modify: `backend/pyproject.toml`
- Modify: `backend/app/main.py`
- Create: `frontend/index.html` (shell only, no content yet)
- Create: `frontend/styles.css` (empty)
- Create: `frontend/app.js` (empty)
- Test: `backend/tests/test_static.py`

- [ ] **Step 1: Add `aiofiles>=23.0` to `pyproject.toml`**

```toml
dependencies = [
    ...existing deps...,
    "fpdf2>=2.7",
    "aiofiles>=23.0",
]
```

- [ ] **Step 2: Install**

```bash
cd backend && pip install -e ".[dev]"
python -c "import aiofiles; print('aiofiles ok')"
```

- [ ] **Step 3: Create minimal `frontend/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Survey Analytics</title>
  <link rel="stylesheet" href="/styles.css">
</head>
<body>
  <div id="app">Loading...</div>
  <script src="/app.js"></script>
</body>
</html>
```

- [ ] **Step 4: Create empty `frontend/styles.css`**

```css
/* styles loaded */
```

- [ ] **Step 5: Create empty `frontend/app.js`**

```js
// app loaded
console.log('Survey Analytics JS loaded');
```

- [ ] **Step 6: Modify `backend/app/main.py`** to mount static files

```python
from __future__ import annotations
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.config import settings
from app.db.database import create_tables
from app.api import upload, sessions, dashboard, chat, insights, compare, export

_FRONTEND = Path(__file__).parent.parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    create_tables()
    yield


app = FastAPI(title="Survey Analytics", lifespan=lifespan)
app.include_router(upload.router, prefix="/api")
app.include_router(sessions.router, prefix="/api")
app.include_router(dashboard.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(insights.router, prefix="/api")
app.include_router(compare.router, prefix="/api")
app.include_router(export.router, prefix="/api")

if _FRONTEND.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND), html=True), name="frontend")
```

- [ ] **Step 7: Write failing test**

```python
# backend/tests/test_static.py
from __future__ import annotations
import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def static_client():
    with TestClient(app) as c:
        yield c


def test_root_returns_html(static_client):
    r = static_client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_app_js_served(static_client):
    r = static_client.get("/app.js")
    assert r.status_code == 200


def test_styles_css_served(static_client):
    r = static_client.get("/styles.css")
    assert r.status_code == 200
```

- [ ] **Step 8: Run test**

```bash
cd backend && pytest tests/test_static.py -v
# Expected: all PASS
```

- [ ] **Step 9: Commit**

```bash
git add backend/pyproject.toml backend/app/main.py frontend/ backend/tests/test_static.py
git commit -m "feat: static file serving — FastAPI mounts frontend/ at /"
```

---

## Task 2: Layout + Theme (CSS)

**Files:**
- Modify: `frontend/styles.css`

- [ ] **Step 1: Write the full CSS**

```css
/* frontend/styles.css */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --black: #000;
  --white: #fff;
  --gray-100: #f5f5f5;
  --gray-200: #eee;
  --gray-400: #ccc;
  --gray-600: #555;
  --font: "Inter", "Helvetica Neue", Helvetica, Arial, sans-serif;
  --mono: "JetBrains Mono", "Fira Code", "Courier New", monospace;
  --sidebar-w: 220px;
  --canvas-w: 380px;
}

body {
  font-family: var(--font);
  background: var(--white);
  color: var(--black);
  height: 100dvh;
  overflow: hidden;
  display: grid;
  grid-template-columns: var(--sidebar-w) 1fr;
  grid-template-rows: 1fr;
}

/* ── Sidebar ── */
#sidebar {
  background: var(--gray-100);
  border-right: 1px solid var(--gray-200);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
#sidebar-header {
  padding: 16px 14px 10px;
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  border-bottom: 1px solid var(--gray-200);
  display: flex;
  justify-content: space-between;
  align-items: center;
}
#btn-new-session {
  background: var(--black);
  color: var(--white);
  border: none;
  padding: 4px 10px;
  font-size: 11px;
  cursor: pointer;
  font-family: var(--font);
}
#btn-new-session:hover { background: var(--gray-600); }
#session-list { flex: 1; overflow-y: auto; padding: 8px 0; }
.session-item {
  padding: 9px 14px;
  cursor: pointer;
  font-size: 12px;
  border-bottom: 1px solid var(--gray-200);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  color: var(--gray-600);
}
.session-item:hover, .session-item.active {
  background: var(--gray-200);
  color: var(--black);
}
.session-item .session-name { font-weight: 600; font-size: 12px; color: var(--black); }
.session-item .session-meta { font-size: 10px; color: var(--gray-600); }

/* ── Main area ── */
#main {
  display: grid;
  grid-template-rows: auto 1fr;
  overflow: hidden;
  position: relative;
}
#tab-bar {
  display: flex;
  border-bottom: 1px solid var(--gray-200);
  background: var(--white);
}
.tab-btn {
  padding: 10px 20px;
  background: none;
  border: none;
  border-bottom: 2px solid transparent;
  font-size: 13px;
  font-family: var(--font);
  cursor: pointer;
  color: var(--gray-600);
}
.tab-btn.active { border-bottom-color: var(--black); color: var(--black); font-weight: 600; }

/* ── Chat Tab ── */
#chat-area {
  display: flex;
  flex-direction: column;
  height: 100%;
  overflow: hidden;
}
#messages {
  flex: 1;
  overflow-y: auto;
  padding: 20px 24px;
  display: flex;
  flex-direction: column;
  gap: 20px;
}
.message { max-width: 80%; }
.message.user { align-self: flex-end; }
.message.assistant { align-self: flex-start; }
.message-bubble {
  padding: 12px 16px;
  font-size: 14px;
  line-height: 1.6;
  border: 1px solid var(--gray-200);
}
.message.user .message-bubble {
  background: var(--black);
  color: var(--white);
  border-color: var(--black);
}
.message.assistant .message-bubble { background: var(--gray-100); }
.message-chart { margin-top: 10px; cursor: pointer; }
.message-chart img { max-width: 100%; border: 1px solid var(--gray-200); display: block; }
.chart-actions { display: flex; gap: 8px; margin-top: 6px; }
.chart-actions button {
  font-size: 11px; padding: 3px 10px;
  border: 1px solid var(--gray-400);
  background: var(--white); cursor: pointer; font-family: var(--font);
}
.chart-actions button:hover { background: var(--gray-200); }
.follow-ups { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
.follow-up-chip {
  padding: 5px 12px; border: 1px solid var(--gray-400);
  font-size: 12px; cursor: pointer; background: var(--white); font-family: var(--font);
}
.follow-up-chip:hover { background: var(--gray-200); }
.caveats { font-size: 11px; color: var(--gray-600); margin-top: 6px; font-style: italic; }
.show-code-btn {
  font-size: 11px; padding: 3px 10px; margin-top: 6px;
  border: 1px solid var(--gray-400); background: var(--white);
  cursor: pointer; font-family: var(--font);
}
.code-block {
  display: none; margin-top: 8px;
  background: var(--gray-100); border: 1px solid var(--gray-200);
  padding: 12px; font-family: var(--mono); font-size: 11px;
  white-space: pre-wrap; word-break: break-all; overflow-x: auto;
}
.code-block.visible { display: block; }
#chat-input-area {
  padding: 14px 24px;
  border-top: 1px solid var(--gray-200);
  display: flex;
  gap: 10px;
  background: var(--white);
}
#chat-input {
  flex: 1; padding: 10px 14px; border: 1px solid var(--gray-400);
  font-size: 14px; font-family: var(--font); resize: none; min-height: 44px;
  max-height: 160px; overflow-y: auto;
}
#chat-input:focus { outline: 2px solid var(--black); border-color: transparent; }
#btn-send {
  background: var(--black); color: var(--white); border: none;
  padding: 10px 20px; font-size: 14px; cursor: pointer; font-family: var(--font);
  align-self: flex-end;
}
#btn-send:hover { background: var(--gray-600); }
#btn-send:disabled { background: var(--gray-400); cursor: not-allowed; }

/* ── Dashboard Tab ── */
#dashboard-area { overflow-y: auto; padding: 24px; }
.stats-cards { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; }
.stat-card {
  border: 1px solid var(--gray-200); padding: 16px 20px; min-width: 140px;
}
.stat-card .label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--gray-600); }
.stat-card .value { font-size: 28px; font-weight: 700; margin-top: 4px; }
.dashboard-charts { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 16px; }
.dashboard-chart-card { border: 1px solid var(--gray-200); padding: 12px; }
.dashboard-chart-card img { width: 100%; display: block; }
.dashboard-chart-title { font-size: 12px; font-weight: 600; margin-bottom: 8px; }
.insights-section { margin-top: 28px; }
.insights-section h2 { font-size: 14px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 14px; border-bottom: 1px solid var(--gray-200); padding-bottom: 6px; }
.insight-card { border: 1px solid var(--gray-200); padding: 14px 16px; margin-bottom: 10px; cursor: pointer; }
.insight-card:hover { background: var(--gray-100); }
.insight-title { font-size: 13px; font-weight: 600; }
.insight-summary { font-size: 12px; color: var(--gray-600); margin-top: 4px; line-height: 1.5; }
.export-btn {
  margin-top: 24px; padding: 10px 20px; background: var(--black);
  color: var(--white); border: none; font-size: 13px; cursor: pointer; font-family: var(--font);
}
.export-btn:hover { background: var(--gray-600); }

/* ── Canvas Panel ── */
#canvas-panel {
  position: fixed; top: 0; right: 0;
  width: var(--canvas-w); height: 100dvh;
  background: var(--white);
  border-left: 1px solid var(--gray-200);
  display: flex; flex-direction: column;
  transform: translateX(100%);
  transition: transform 0.2s ease;
  z-index: 100;
}
#canvas-panel.open { transform: translateX(0); }
#canvas-header {
  padding: 14px 16px; border-bottom: 1px solid var(--gray-200);
  display: flex; justify-content: space-between; align-items: center;
}
#canvas-title { font-size: 13px; font-weight: 600; }
#btn-close-canvas { background: none; border: none; font-size: 20px; cursor: pointer; line-height: 1; }
#canvas-body { flex: 1; overflow-y: auto; padding: 16px; }
#canvas-img { width: 100%; display: block; border: 1px solid var(--gray-200); }
#canvas-actions { padding: 12px 16px; border-top: 1px solid var(--gray-200); display: flex; gap: 8px; }
#canvas-actions button {
  flex: 1; padding: 8px; border: 1px solid var(--gray-400);
  background: var(--white); font-size: 12px; cursor: pointer; font-family: var(--font);
}
#canvas-actions button:hover { background: var(--gray-200); }

/* ── Upload overlay ── */
#upload-overlay {
  position: fixed; inset: 0; background: rgba(255,255,255,0.95);
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  z-index: 200;
}
#upload-overlay h1 { font-size: 28px; font-weight: 700; margin-bottom: 8px; }
#upload-overlay p { font-size: 14px; color: var(--gray-600); margin-bottom: 32px; }
#drop-zone {
  width: 400px; height: 160px;
  border: 2px dashed var(--gray-400);
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  gap: 12px; cursor: pointer; transition: border-color 0.15s;
}
#drop-zone.drag-over { border-color: var(--black); }
#drop-zone span { font-size: 13px; color: var(--gray-600); }
#file-input { display: none; }
#drop-zone button {
  padding: 8px 20px; background: var(--black); color: var(--white);
  border: none; font-size: 13px; cursor: pointer; font-family: var(--font);
}

/* ── Sheet picker overlay ── */
#sheet-picker {
  display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.5);
  align-items: center; justify-content: center; z-index: 300;
}
#sheet-picker.visible { display: flex; }
#sheet-picker-box {
  background: var(--white); padding: 28px; width: 360px;
}
#sheet-picker-box h2 { font-size: 16px; margin-bottom: 16px; }
.sheet-option {
  padding: 10px 14px; border: 1px solid var(--gray-200); margin-bottom: 8px;
  cursor: pointer; font-size: 13px;
}
.sheet-option:hover { background: var(--gray-100); }

/* ── Loading spinner ── */
.spinner {
  display: inline-block; width: 16px; height: 16px;
  border: 2px solid var(--gray-400); border-top-color: var(--black);
  border-radius: 50%; animation: spin 0.6s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

.hidden { display: none !important; }
```

- [ ] **Step 2: Commit**

```bash
git add frontend/styles.css
git commit -m "feat: CSS theme — black & white layout, sidebar, chat, dashboard, canvas panel"
```

---

## Task 3: Full Application JavaScript

**Files:**
- Modify: `frontend/app.js`

- [ ] **Step 1: Write the complete `frontend/app.js`**

```js
// frontend/app.js
'use strict';

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
    chartHtml = `
      <div class="message-chart" onclick="openCanvas('${title}', '${src}')">
        <img src="${src}" alt="${esc(title)}">
      </div>
      <div class="chart-actions">
        <button onclick="openCanvas('${esc(title)}', '${src}')">Expand</button>
        <button onclick="downloadPng('${src}', '${esc(title)}')">Download</button>
        <button onclick="pinChart('${src}', '${esc(title)}')">Pin</button>
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
  const chartsHtml = (dash.charts || []).map(c => `
    <div class="dashboard-chart-card">
      <div class="dashboard-chart-title">${esc(c.title)}</div>
      <img src="data:image/png;base64,${c.png_b64}" alt="${esc(c.title)}"
           onclick="openCanvas('${esc(c.title)}', 'data:image/png;base64,${c.png_b64}')" style="cursor:pointer">
    </div>`).join('');

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
```

- [ ] **Step 2: Commit**

```bash
git add frontend/app.js
git commit -m "feat: full frontend JS — upload, chat, dashboard, canvas panel, session sidebar"
```

---

## Task 4: Full HTML Shell

**Files:**
- Modify: `frontend/index.html`

The JS calls `buildShell()` to inject the DOM dynamically, so `index.html` stays minimal. The current shell is already correct from Task 1. No changes needed — verify it loads correctly.

- [ ] **Step 1: Verify the server serves the app**

```bash
cd backend
uvicorn app.main:app --reload
```

Open [http://localhost:8000](http://localhost:8000) in a browser.

Expected:
- Upload overlay appears (black & white)
- Drag & drop a CSV → upload triggers → chat view appears with auto-dashboard message
- Typing in chat input and pressing Enter → sends message → response appears with chart

- [ ] **Step 2: Run static file tests**

```bash
cd backend && pytest tests/test_static.py -v
# Expected: all PASS
```

- [ ] **Step 3: Run full test suite**

```bash
pytest -v
# Expected: all tests PASS
```

- [ ] **Step 4: Commit**

```bash
git add frontend/index.html
git commit -m "feat: Plan 4 complete — full frontend wired to backend API"
```

---

## Self-Review

**Spec coverage:**
- ✅ Black & white minimal theme
- ✅ Upload flow: drag & drop + browse, multi-sheet Excel picker (409 → sheet picker modal)
- ✅ Auto-dashboard as first message (dashboard tab loads immediately after upload)
- ✅ Chat stream UI: messages, inline charts, follow-up chips, caveats, Show code toggle
- ✅ Canvas panel: expand chart, download PNG, pin to dashboard
- ✅ Dashboard tab: stats cards, charts grid, insight feed, narrative
- ✅ Session sidebar: session list, New button
- ✅ PDF export button on dashboard tab
- ✅ StaticFiles served from FastAPI — no separate server needed
- ⚠️ Comparison mode UI not implemented — the `/compare` API exists from Plan 3 but the UI to select a second session is not built. Can be added as a follow-up.
- ⚠️ Plotly charts — API returns `plotly_json` when LLM generates Plotly code (Tier-2), but the UI currently only handles `png_b64`. To support Plotly, add `<script src="https://cdn.plot.ly/plotly-2.30.0.min.js">` to `index.html` and handle `chart.plotly_json` in `appendMessage()`.
