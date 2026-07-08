/* SURVEY INSIGHT AGENT — Frontend */

const API_BASE = "/TextAnalyticsWebUtility-WS/ConvInsightServlet";

// --------- STATE ------------------------------------------------------------------------------------------------------------------------------
let sessionId      = null;
let activeSheet    = null;
let allColumns     = [];       // full column_info array from upload
let chatHistory    = [];       // {role, text} for export
let colsExpanded   = false;
const COL_PREVIEW  = 12;

// --------- DOM REFS ------------------------------------------------------------------------------------------------------------------------
const excelFile    = document.getElementById("excelFile");
const uploadBtn    = document.getElementById("uploadBtn");
const dropZone     = document.getElementById("dropZone");
const uploadSection= document.getElementById("uploadSection");
const fileCard     = document.getElementById("fileCard");
const fcName       = document.getElementById("fcName");
const fcMeta       = document.getElementById("fcMeta");
const reUpBtn      = document.getElementById("reUpBtn");
const statsGrid    = document.getElementById("statsGrid");
const sheetBlock   = document.getElementById("sheetBlock");
const sheetTabRow  = document.getElementById("sheetTabRow");
const colBlock     = document.getElementById("colBlock");
const colList      = document.getElementById("colList");
const colBadge     = document.getElementById("colBadge");
const colShowMore  = document.getElementById("colShowMore");
const insightBlock = document.getElementById("insightBlock");
const insightContent = document.getElementById("insightContent");
const insightSkel  = document.getElementById("insightSkel");
const dataTypeBadge= document.getElementById("dataTypeBadge");
const chatLog      = document.getElementById("chatLog");
const emptyState   = document.getElementById("emptyState");
const sheetCtx     = document.getElementById("sheetCtx");
const exportBtn    = document.getElementById("exportBtn");
const clearBtn     = document.getElementById("clearBtn");
const suggBar      = document.getElementById("suggBar");
const suggChips    = document.getElementById("suggChips");
const chatForm     = document.getElementById("chatForm");
const msgInput     = document.getElementById("msgInput");
const sendBtn      = document.getElementById("sendBtn");
const attachBtn    = document.getElementById("attachBtn");
const statusMsg    = document.getElementById("statusMsg");
const mobTabs      = document.getElementById("mobTabs");
const sidePanel    = document.getElementById("sidePanel");
const chatPanel    = document.getElementById("chatPanel");

// --------- HELPERS ---------------------------------------------------------------------------------------------------------------------------
function escHtml(v) {
  return String(v)
    .replace(/&/g,"&amp;").replace(/</g,"&lt;")
    .replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/'/g,"&#39;");
}

function fmtInline(text) {
  const e = escHtml(text);
  return e
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g,     "<em>$1</em>")
    .replace(/`(.+?)`/g,       "<code>$1</code>");
}

function numFmt(n) {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (n >= 1_000)     return (n / 1_000).toFixed(1) + "K";
  return String(n);
}

function setLoading(btn, on) {
  btn.disabled = on;
  btn.classList.toggle("is-loading", on);
}

function show(el) { el.hidden = false; }
function hide(el) { el.hidden = true; }

// --------- MARKDOWN RENDERER ---------------------------------------------------------------------------------------------
function parseMarkdownTable(lines, start) {
  const rows = [];
  let i = start;
  while (i < lines.length && lines[i].trim().startsWith("|")) {
    rows.push(lines[i].trim());
    i++;
  }
  if (rows.length < 2 || !rows[1].includes("---")) return null;

  const cells = l => l.replace(/^\|/,"").replace(/\|$/,"")
                       .split("|").map(c => c.trim());
  const headers = cells(rows[0]);
  const body    = rows.slice(2).map(cells);

  let html = '<div class="table-wrap"><table class="rpt-table"><thead><tr>';
  headers.forEach(h => { html += `<th>${fmtInline(h)}</th>`; });
  html += "</tr></thead><tbody>";
  body.forEach(row => {
    html += "<tr>";
    row.forEach(c => { html += `<td>${fmtInline(c)}</td>`; });
    html += "</tr>";
  });
  html += "</tbody></table></div>";
  return { html, nextIdx: i };
}

function renderMd(text) {
  const lines = text.split("\n");
  let i = 0, html = '<div class="md-body">';

  while (i < lines.length) {
    const raw  = lines[i];
    const line = raw.trim();
    if (!line) { i++; continue; }

    const tbl = parseMarkdownTable(lines, i);
    if (tbl) { html += tbl.html; i = tbl.nextIdx; continue; }

    if (line.startsWith("### ")) { html += `<h3>${fmtInline(line.slice(4))}</h3>`; i++; continue; }
    if (line.startsWith("## "))  { html += `<h2>${fmtInline(line.slice(3))}</h2>`; i++; continue; }
    if (line.startsWith("# "))   { html += `<h1>${fmtInline(line.slice(2))}</h1>`; i++; continue; }

    if (line.startsWith("- ") || line.startsWith("* ")) {
      html += "<ul>";
      while (i < lines.length && (lines[i].trim().startsWith("- ") || lines[i].trim().startsWith("* "))) {
        html += `<li>${fmtInline(lines[i].trim().slice(2))}</li>`;
        i++;
      }
      html += "</ul>";
      continue;
    }

    html += `<p>${fmtInline(line)}</p>`;
    i++;
  }
  html += "</div>";
  return html;
}

// --------- CHART GALLERY ---------------------------------------------------------------------------------------------------------
function openLightbox(src, title) {
  const ov = document.createElement("div");
  ov.className = "lightbox";
  ov.innerHTML =
    `<div class="lb-inner"><div class="lb-head"><span>${escHtml(title)}</span>` +
    `<button class="lb-close" aria-label="Close">×</button></div>` +
    `<div class="lb-body"><img src="${src}" alt="${escHtml(title)}"></div></div>`;
  const close = () => ov.remove();
  ov.addEventListener("click", e => { if (e.target === ov || e.target.classList.contains("lb-close")) close(); });
  document.addEventListener("keydown", function esc(e){ if(e.key==="Escape"){ close(); document.removeEventListener("keydown", esc);} });
  // simple wheel zoom on the image
  const img = ov.querySelector("img");
  let scale = 1;
  ov.querySelector(".lb-body").addEventListener("wheel", e => {
    e.preventDefault();
    scale = Math.min(6, Math.max(1, scale * (e.deltaY < 0 ? 1.15 : 1/1.15)));
    img.style.transform = `scale(${scale})`;
  }, { passive: false });
  document.body.appendChild(ov);
}

// Pinned charts for the PDF report ("canvas"), like the original survey app's pin-to-report.
let pinnedCharts = [];

function chartKey(c) { return (c.title || "") + "|" + (c.image_base64 || "").slice(0, 40); }

function updatePinBadge() {
  const badge = document.getElementById("pinBadge");
  if (!badge) return;
  badge.textContent = pinnedCharts.length;
  badge.hidden = pinnedCharts.length === 0;
}

function togglePin(c, btn) {
  const key = chartKey(c);
  const idx = pinnedCharts.findIndex(p => chartKey(p) === key);
  if (idx >= 0) {
    pinnedCharts.splice(idx, 1);
    btn.classList.remove("pinned");
    btn.textContent = "📌 Pin";
  } else {
    pinnedCharts.push({ title: c.title || "Chart", mime_type: c.mime_type || "image/png", image_base64: c.image_base64 });
    btn.classList.add("pinned");
    btn.textContent = "📌 Pinned";
  }
  updatePinBadge();
}

function buildChartGallery(charts) {
  if (!Array.isArray(charts) || !charts.length) return null;
  const gallery = document.createElement("div");
  gallery.className = "chart-gallery";
  charts.forEach(c => {
    if (!c?.image_base64) return;
    const fig  = document.createElement("figure");
    fig.className = "chart-card";
    const src  = `data:${c.mime_type || "image/png"};base64,${c.image_base64}`;
    const img  = document.createElement("img");
    img.src    = src;
    img.alt    = c.title ? `Chart: ${c.title}` : "Generated chart";
    img.loading= "lazy";
    img.style.cursor = "zoom-in";
    img.title  = "Click to zoom";
    img.addEventListener("click", () => openLightbox(src, c.title || "Chart"));
    fig.appendChild(img);

    const bar = document.createElement("div");
    bar.className = "chart-actions";

    const pin = document.createElement("button");
    pin.type = "button";
    pin.className = "chart-dl chart-pin";
    const alreadyPinned = pinnedCharts.some(p => chartKey(p) === chartKey(c));
    if (alreadyPinned) pin.classList.add("pinned");
    pin.textContent = alreadyPinned ? "📌 Pinned" : "📌 Pin";
    pin.title = "Pin this chart into the PDF report";
    pin.addEventListener("click", () => togglePin(c, pin));
    bar.appendChild(pin);

    const dl = document.createElement("a");
    dl.href = src;
    dl.download = ((c.title || "chart").replace(/\s+/g, "_")) + ".png";
    dl.textContent = "⤓ Download";
    dl.className = "chart-dl";
    bar.appendChild(dl);
    fig.appendChild(bar);

    if (c.title) {
      const cap = document.createElement("figcaption");
      cap.textContent = c.title;
      fig.appendChild(cap);
    }
    gallery.appendChild(fig);
  });
  return gallery.children.length ? gallery : null;
}

// --------- CHAT MESSAGES ---------------------------------------------------------------------------------------------------------
function hideEmptyState() {
  if (emptyState) emptyState.style.display = "none";
}

function appendMsg(kind, text, charts = [], extraClass = "") {
  hideEmptyState();
  const div = document.createElement("div");
  div.className = `chat-message msg-${kind}${extraClass ? " " + extraClass : ""}`;

  if (kind === "assistant" || kind === "auto-insight") {
    div.innerHTML = renderMd(text);
    const g = buildChartGallery(charts);
    if (g) div.appendChild(g);
  } else {
    div.textContent = text;
  }

  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;
  chatHistory.push({ role: kind, text });
  return div;
}

function showThinking(label = "Thinking") {
  hideEmptyState();
  const phases = [
    "Reading your question…",
    "Inspecting data schema…",
    "Running analysis…",
    "Drafting insights…",
  ];
  const div = document.createElement("div");
  div.className = "chat-message msg-assistant msg-thinking";
  div.innerHTML = `
    <div class="thinking-head">
      <span class="spinner" aria-hidden="true"></span>
      <strong>${escHtml(label)}</strong>
      <span class="thinking-elapsed">0s</span>
    </div>
    <p class="thinking-step">${phases[0]}</p>`;
  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;

  const stepEl = div.querySelector(".thinking-step");
  const timeEl = div.querySelector(".thinking-elapsed");
  let pi = 0;
  const t0 = Date.now();
  const iv = setInterval(() => {
    pi = (pi + 1) % phases.length;
    stepEl.textContent = phases[pi];
    timeEl.textContent = `${Math.floor((Date.now() - t0) / 1000)}s`;
    chatLog.scrollTop = chatLog.scrollHeight;
  }, 1400);

  return { stop: () => { clearInterval(iv); div.remove(); } };
}

// --------- UPLOAD LOGIC ------------------------------------------------------------------------------------------------------------
uploadBtn.addEventListener("click", () => excelFile.click());
attachBtn.addEventListener("click", () => excelFile.click());
reUpBtn  .addEventListener("click", () => excelFile.click());
excelFile.addEventListener("change", () => { if (excelFile.files.length) doUpload(excelFile.files); });

// Drag & drop
dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("drag-over"); });
dropZone.addEventListener("dragleave", ()  => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  if (e.dataTransfer.files.length) doUpload(e.dataTransfer.files);
});
dropZone.addEventListener("click", e => {
  if (e.target !== uploadBtn) excelFile.click();
});

async function doUpload(fileList) {
  const files = Array.from(fileList || []);
  if (!files.length) return;
  statusMsg.textContent = files.length > 1 ? `Uploading ${files.length} files…` : "Uploading…";
  setLoading(attachBtn, true);

  const fd = new FormData();
  files.forEach(f => fd.append("files", f));   // multi-file: field name "files"

  try {
    const res = await fetch(API_BASE, { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Upload failed");

    sessionId   = data.session_id;
    activeSheet = data.active_sheet;
    allColumns  = data.column_info || [];
    try { localStorage.setItem("convinsight_session", sessionId); } catch (e) {}

    populateDashboard(data);
    statusMsg.textContent = `Session: ${data.session_id.slice(0, 8)}…`;
    show(exportBtn); show(clearBtn);
    switchMobTab("chat");

    const fileLabel = (data.files && data.files.length > 1)
      ? `${data.files.length} files combined` : data.filename;
    appendMsg("system",
      `✅ Loaded: ${fileLabel}  |  ${data.shape[0].toLocaleString()} rows × ${data.shape[1]} cols`
    );
    if (data.skipped_files && data.skipped_files.length) {
      appendMsg("system",
        `⚠ ${data.skipped_files.length} file(s) had a different structure and were NOT combined: ` +
        data.skipped_files.map(escHtml).join(", ")
      );
    }

    // Kick off background AI analysis
    loadAutoInsights();

  } catch (err) {
    statusMsg.textContent = `Upload error: ${err.message}`;
  } finally {
    setLoading(attachBtn, false);
  }
}

// --------- DASHBOARD POPULATION ------------------------------------------------------------------------------------
function populateDashboard(data) {
  // File card
  fcName.textContent = data.filename;
  fcMeta.textContent = `${data.shape[0].toLocaleString()} rows · ${data.shape[1]} cols · ${data.sheet_names.length} sheet(s)`;
  hide(uploadSection);
  show(fileCard);

  // Stats
  document.getElementById("stRows")   .textContent = numFmt(data.shape[0]);
  document.getElementById("stCols")   .textContent = numFmt(data.shape[1]);
  document.getElementById("stSheets") .textContent = data.sheet_names.length;
  document.getElementById("stMissing").textContent = `${data.overall_missing_pct ?? "?"}%`;
  show(statsGrid);
  show(document.getElementById("dashActions"));

  // Sheet tabs
  sheetTabRow.innerHTML = "";
  data.sheet_names.forEach(name => {
    const btn = document.createElement("button");
    btn.className = "sheet-tab" + (name === data.active_sheet ? " active" : "");
    btn.textContent = name;
    btn.title = name;
    btn.addEventListener("click", () => switchSheet(name));
    sheetTabRow.appendChild(btn);
  });
  show(sheetBlock);

  // Column list
  renderColumnList(data.column_info || []);

  // Insight block (skeleton shown while async loads)
  show(insightBlock);
}

function renderColumnList(cols) {
  if (!cols.length) { hide(colBlock); return; }
  allColumns = cols;
  colBadge.textContent = cols.length;
  colList.innerHTML = "";

  const visible = colsExpanded ? cols : cols.slice(0, COL_PREVIEW);

  visible.forEach(c => {
    const li = document.createElement("li");
    li.className = "col-item";

    const typeMap = { numeric: ["#", "type-numeric"], text: ["T", "type-text"],
                      datetime: ["D", "type-datetime"], boolean: ["B", "type-boolean"] };
    const [sym, cls] = typeMap[c.type] || ["?", "type-other"];

    const badge   = `<span class="col-type-badge ${cls}" title="${escHtml(c.dtype)}">${sym}</span>`;
    const missing = c.missing_pct > 0
      ? `<span class="col-missing has-missing">${c.missing_pct}%</span>`
      : `<span class="col-missing">0%</span>`;

    li.innerHTML = `${badge}<span class="col-name" title="${escHtml(c.name)}">${escHtml(c.name)}</span>${missing}`;
    colList.appendChild(li);
  });

  if (cols.length > COL_PREVIEW) {
    show(colShowMore);
    colShowMore.textContent = colsExpanded
      ? "Show fewer ▴"
      : `Show all ${cols.length} columns ▾`;
  } else {
    hide(colShowMore);
  }
  show(colBlock);
}

colShowMore.addEventListener("click", () => {
  colsExpanded = !colsExpanded;
  renderColumnList(allColumns);
});

function switchSheet(name) {
  activeSheet = name;
  sheetCtx.textContent = `Sheet: ${name}`;
  show(sheetCtx);
  // Update active tab styling
  sheetTabRow.querySelectorAll(".sheet-tab").forEach(b => {
    b.classList.toggle("active", b.textContent === name);
  });
}

// --------- AUTO-INSIGHTS ------------------------------------------------------------------------------------------------------------
async function loadAutoInsights() {
  if (!sessionId) return;
  const thinking = showThinking("Analyzing your data");

  try {
    const res = await fetch(API_BASE, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: "autoInsights", session_id: sessionId }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Analysis failed");

    thinking.stop();

    // Update data type badge
    if (data.data_type) {
      dataTypeBadge.textContent = data.data_type;
      show(dataTypeBadge);
    }

    // Render compact insight summary in sidebar
    renderInsightSummary(data);

    // Render rich auto-insight message in chat
    appendMsg("auto-insight",
      `### 🔍 Automatic Analysis\n${data.answer}`,
      data.charts || []
    );

    // Suggested questions
    if (Array.isArray(data.suggested_questions) && data.suggested_questions.length) {
      renderSuggestedQuestions(data.suggested_questions);
    }

  } catch (err) {
    thinking.stop();
    appendMsg("system", `⚠️ Auto-analysis failed: ${err.message}`);
    // Still hide skeleton
    if (insightSkel) insightSkel.style.display = "none";
  }
}

function renderInsightSummary(data) {
  if (!insightContent) return;

  // Extract bullet points from the answer
  const bullets = [];
  const lines   = (data.answer || "").split("\n");
  for (const line of lines) {
    const t = line.trim();
    if ((t.startsWith("- ") || t.startsWith("* ")) && t.length > 4) {
      bullets.push(t.slice(2).trim());
      if (bullets.length >= 4) break;
    }
  }

  let html = "";
  if (data.data_type) {
    html += `<span class="insight-data-type">${escHtml(data.data_type)}</span>`;
  }
  if (bullets.length) {
    html += '<ul class="insight-bullets">';
    bullets.forEach(b => { html += `<li class="insight-bullet">${escHtml(b)}</li>`; });
    html += "</ul>";
  } else {
    // Show first 160 chars of answer
    const preview = (data.answer || "").replace(/#+\s*/g,"").trim().slice(0, 160);
    html += `<p style="font-size:.78rem;color:var(--ink-soft);margin:0">${escHtml(preview)}…</p>`;
  }

  insightContent.innerHTML = html;
}

function renderSuggestedQuestions(questions) {
  if (!questions.length) return;
  suggChips.innerHTML = "";
  questions.forEach(q => {
    const chip = document.createElement("button");
    chip.className = "sugg-chip";
    chip.textContent = q;
    chip.addEventListener("click", () => {
      msgInput.value = q;
      autoResize();
      msgInput.focus();
      // Auto-send after brief delay for UX smoothness
      setTimeout(() => chatForm.dispatchEvent(new Event("submit", { cancelable: true })), 80);
    });
    suggChips.appendChild(chip);
  });
  show(suggBar);
}

// --------- CHAT FORM ---------------------------------------------------------------------------------------------------------------------
function autoResize() {
  msgInput.style.height = "auto";
  msgInput.style.height = Math.min(msgInput.scrollHeight, 130) + "px";
}

msgInput.addEventListener("input", autoResize);
autoResize();

msgInput.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    chatForm.dispatchEvent(new Event("submit", { cancelable: true }));
  }
});

chatForm.addEventListener("submit", async e => {
  e.preventDefault();
  const question = msgInput.value.trim();
  if (!question) return;

  if (!sessionId) {
    appendMsg("system", "Please upload a survey workbook first.");
    return;
  }

  appendMsg("user", question);
  msgInput.value = "";
  autoResize();
  setLoading(sendBtn, true);
  const thinking = showThinking();

  try {
    const res = await fetch(API_BASE, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: "chat", session_id: sessionId, question }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Request failed");

    thinking.stop();
    appendMsg("assistant", data.answer, data.charts || []);

    // Update active sheet if agent switched
    if (data.active_sheet && data.active_sheet !== activeSheet) {
      switchSheet(data.active_sheet);
    }

  } catch (err) {
    thinking.stop();
    appendMsg("system", `Error: ${err.message}`);
  } finally {
    setLoading(sendBtn, false);
  }
});

// --------- EXPORT CHAT ---------------------------------------------------------------------------------------------------------------
exportBtn.addEventListener("click", () => {
  if (!chatHistory.length) return;
  const lines = chatHistory.map(m => `[${m.role.toUpperCase()}]\n${m.text}`).join("\n\n---\n\n");
  const header = `Survey Insight Chat Export\nSession: ${sessionId || "unknown"}\nExported: ${new Date().toLocaleString()}\n${"=".repeat(50)}\n\n`;
  const blob = new Blob([header + lines], { type: "text/plain;charset=utf-8" });
  const a  = document.createElement("a");
  a.href   = URL.createObjectURL(blob);
  a.download = `chat-export-${Date.now()}.txt`;
  a.click();
  URL.revokeObjectURL(a.href);
});

// --------- CLEAR CHAT ------------------------------------------------------------------------------------------------------------------
clearBtn.addEventListener("click", () => {
  if (!confirm("Clear the entire chat history?")) return;
  chatLog.innerHTML = "";
  chatHistory = [];
  hide(suggBar);
  // Re-show empty state
  const es = document.createElement("div");
  es.className = "empty-state";
  es.innerHTML = `
    <svg viewBox="0 0 64 64" fill="none" width="60" height="60">
      <circle cx="32" cy="32" r="30" fill="#f1f5f9"/>
      <rect x="14" y="22" width="36" height="22" rx="9" fill="#e2e8f0"/>
      <rect x="22" y="30" width="9" height="2.5" rx="1.25" fill="#94a3b8"/>
      <rect x="22" y="35" width="18" height="2.5" rx="1.25" fill="#cbd5e1"/>
    </svg>
    <p class="es-title">Chat cleared</p>
    <p class="es-body">Ask a question to continue the analysis</p>`;
  chatLog.appendChild(es);
});

// --------- MOBILE TABS ---------------------------------------------------------------------------------------------------------------
function switchMobTab(tab) {
  if (!mobTabs) return;
  mobTabs.querySelectorAll(".mob-tab").forEach(b => b.classList.toggle("active", b.dataset.tab === tab));
  sidePanel.classList.toggle("mob-active", tab === "data");
  chatPanel.classList.toggle("mob-active", tab === "chat");
}

mobTabs.querySelectorAll(".mob-tab").forEach(btn => {
  btn.addEventListener("click", () => switchMobTab(btn.dataset.tab));
});

// Default mobile view: data panel first (chat after upload)
switchMobTab("data");

// Resume a previous session after a page refresh (server keeps the data on disk).
try {
  const saved = localStorage.getItem("convinsight_session");
  if (saved) {
    sessionId = saved;
    show(exportBtn); show(clearBtn);
    appendMsg("system", "↩ Resumed your previous session — ask a question, or upload a new file to start over.");
  }
} catch (e) {}

// ─── DASHBOARD + PDF REPORT (both open as full overlays, not squeezed into the sidebar) ──
const dashBtn   = document.getElementById("dashBtn");
const reportBtn = document.getElementById("reportBtn");

function openOverlay(title, bodyHtml) {
  const ov = document.createElement("div");
  ov.className = "lightbox wide";
  ov.innerHTML =
    `<div class="lb-inner"><div class="lb-head"><span>${escHtml(title)}</span>` +
    `<button class="lb-close" aria-label="Close">×</button></div>` +
    `<div class="lb-body">${bodyHtml}</div></div>`;
  const close = () => ov.remove();
  ov.addEventListener("click", e => { if (e.target === ov || e.target.classList.contains("lb-close")) close(); });
  document.addEventListener("keydown", function esc(e){ if(e.key==="Escape"){ close(); document.removeEventListener("keydown", esc);} });
  document.body.appendChild(ov);
  return ov;
}

async function loadDashboardPanel() {
  if (!sessionId) return;
  setLoading(dashBtn, true);
  const ov = openOverlay("Dashboard", '<p class="blk-label">Loading dashboard…</p>');
  try {
    const res = await fetch(API_BASE, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: "dashboard", session_id: sessionId }),
    });
    const d = await res.json();
    if (!res.ok) throw new Error(d.detail || "Dashboard failed");

    const q = d.quality || {};
    const qs = [];
    if (q.duplicate_rows) qs.push(`${q.duplicate_rows} duplicate rows`);
    if ((q.mostly_empty_columns || []).length) qs.push(`${q.mostly_empty_columns.length} mostly-empty cols`);
    if ((q.constant_columns || []).length) qs.push(`${q.constant_columns.length} constant cols`);

    let html = '<div class="dash-stats">';
    html += `<div class="stat-tile"><span class="st-val">${numFmt(d.stats.rows)}</span><span class="st-lbl">Rows</span></div>`;
    html += `<div class="stat-tile"><span class="st-val">${d.stats.columns}</span><span class="st-lbl">Columns</span></div>`;
    html += `<div class="stat-tile"><span class="st-val">${d.stats.files_count}</span><span class="st-lbl">Files</span></div>`;
    html += `<div class="stat-tile warn-tile"><span class="st-val">${d.stats.missing_pct}%</span><span class="st-lbl">Missing</span></div></div>`;
    html += `<div class="dash-quality">${qs.length ? "⚠ " + qs.join(" · ") : "✓ No major quality issues"}</div>`;
    const body = ov.querySelector(".lb-body");
    body.innerHTML = html;
    const g = buildChartGallery(d.charts || []);
    if (g) body.appendChild(g);
  } catch (err) {
    ov.querySelector(".lb-body").innerHTML = `<p style="color:#dc2626">Dashboard error: ${escHtml(err.message)}</p>`;
  } finally {
    setLoading(dashBtn, false);
  }
}

// Report overlay: review/remove pinned charts (from chat or dashboard), then export a PDF
// built from exactly those — the "pin to report" workflow from the original survey app.
// With nothing pinned, it falls back to the standard 3-chart dashboard report.
function renderReportOverlay() {
  let html = pinnedCharts.length
    ? `<p class="blk-label">${pinnedCharts.length} chart(s) pinned for this report</p>`
    : `<p class="blk-label">No charts pinned yet — click "📌 Pin" under any chart in chat or the dashboard. Exporting now will use the standard dashboard report instead.</p>`;
  html += '<div class="chart-gallery report-canvas">';
  pinnedCharts.forEach((c, i) => {
    const src = `data:${c.mime_type || "image/png"};base64,${c.image_base64}`;
    html += `<figure class="chart-card"><img src="${src}" alt="${escHtml(c.title)}">` +
            `<div class="chart-actions"><button type="button" class="chart-dl" data-unpin="${i}">✕ Remove</button></div>` +
            `<figcaption>${escHtml(c.title)}</figcaption></figure>`;
  });
  html += '</div><button id="genReportBtn" class="btn-ghost-sm" style="margin-top:10px">⤓ Download PDF Report</button>';
  return html;
}

function openReportOverlay() {
  const ov = openOverlay("Report", renderReportOverlay());
  wireReportOverlay(ov);
}

function wireReportOverlay(ov) {
  ov.querySelectorAll("[data-unpin]").forEach(btn => {
    btn.addEventListener("click", () => {
      pinnedCharts.splice(Number(btn.dataset.unpin), 1);
      updatePinBadge();
      ov.querySelector(".lb-body").innerHTML = renderReportOverlay();
      wireReportOverlay(ov);
    });
  });
  const genBtn = ov.querySelector("#genReportBtn");
  if (genBtn) genBtn.addEventListener("click", () => exportReport(genBtn));
}

async function exportReport(btn) {
  if (!sessionId) return;
  const target = btn || reportBtn;
  setLoading(target, true);
  try {
    const res = await fetch(API_BASE, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: "report", session_id: sessionId, charts: pinnedCharts }),
    });
    const d = await res.json();
    if (!res.ok) throw new Error(d.detail || "Report failed");
    const bytes = Uint8Array.from(atob(d.pdf_base64), c => c.charCodeAt(0));
    const url = URL.createObjectURL(new Blob([bytes], { type: "application/pdf" }));
    const a = document.createElement("a");
    a.href = url; a.download = d.filename || "survey_report.pdf"; a.click();
    URL.revokeObjectURL(url);
  } catch (err) {
    statusMsg.textContent = `Report error: ${err.message}`;
  } finally {
    setLoading(target, false);
  }
}

if (dashBtn)   dashBtn.addEventListener("click", loadDashboardPanel);
if (reportBtn) reportBtn.addEventListener("click", openReportOverlay);
