# Survey Analytics Tool — Design

**Date:** 2026-06-16
**Status:** Approved for planning

## 1. Overview

An internal tool, separate from the `pulseiq-mvp` (Aegis) hackathon project. A user uploads a
messy CSV/Excel survey export. The backend processes and profiles it, then a chat interface
lets the user ask free-form questions ("give me a pie chart of X by Y", "what's driving low
satisfaction?"). The backend resolves these via an adaptive agent (pre-built analysis tools +
sandboxed LLM-generated code) and returns visualizations/results to a black & white minimal UI.

`pulseiq-mvp`'s survey pack (`app/packs/survey/`, `frontend/survey.*`) is a reference/starting
point to copy patterns from (CSV loading, type detection, tool-registry shape) — this is a new,
separate repository, not an extension of pulseiq-mvp.

## 2. Stack & Repo Layout

- **Backend:** Python/FastAPI.
- **Frontend:** Vanilla JS/HTML/CSS, black & white minimal theme.
- **LLM:** Azure OpenAI. Config-driven deployment names:
  - `AZURE_OPENAI_DEPLOYMENT_PRIMARY` — GPT-4.1-class model (code-gen, synthesis, story reports).
  - `AZURE_OPENAI_DEPLOYMENT_FALLBACK` — smaller/cheaper "mini"-class model (intent routing,
    tool selection, history summarization). Exact deployment names confirmed against the Azure
    OpenAI catalog during implementation setup.
- **Persistence:** SQLite (sessions, chat history, pinned charts, insights, comparisons) + local
  disk (uploaded data as Parquet, generated chart images/PDFs).
- **Sandbox:** subprocess-based execution (`multiprocessing`), targeting Linux deployment for
  `resource`-based CPU/memory/time limits. The limiter is an isolated module so a
  Windows-specific fallback (timeout-only) can be swapped in for local dev if needed.

```text
survey-analytics/
  backend/
    app/
      api/          # FastAPI routes: upload, dashboard, chat, export, sessions, compare
      data/         # CSV/Excel loading, profiling, schema detection, data-quality checks
      llm/          # Azure OpenAI client, prompt templates, agent orchestration
      tools/        # Tier-1 analysis tool registry (segment stats, crosstabs, trends, anomalies, threshold counts)
      sandbox/      # Tier-2 subprocess runner, AST whitelist, resource limits
      reports/      # Story-report composition + PDF rendering
      db/           # SQLite models + migrations
    tests/
  frontend/
    index.html, app.js, styles.css   # chat stream / canvas / dashboard tab UI
```

## 3. Data Pipeline & Profiling

On upload:

1. Parse CSV/Excel (adapt `csv_loader.py` patterns — type detection, messy headers). If an
   Excel file has multiple sheets, prompt the user to pick one.
2. Persist the data as Parquet on disk; build a **dataset profile** (JSON, cached on disk):
   - per-column dtype, missing %, n_unique
   - numeric stats: min/max/mean/median/std
   - categorical: top-N value counts
   - a small representative sample (~20 rows)

   This profile (never raw rows beyond the sample) is what's sent to the LLM in every prompt —
   token usage stays flat from 10K to 500K rows.
3. **Data quality flags**: duplicate rows, near-duplicate category spellings (fuzzy-matched,
   e.g. "Male"/"male"/"M"), columns >90% empty, constant columns. Surfaced as a "Data quality"
   card in the dashboard and feed per-chart caveats (section 6).
4. **Open-text columns**: run theme/sentiment extraction (adapted from pulseiq's
   `extract_open_text_themes`) once per column, cached — top recurring themes + sentiment split.

## 4. Auto-Dashboard

The first message in the chat stream, generated once per session and cached:

- Deterministic cards: row/column counts, missing data, data quality flags.
- 2-3 standard charts (matplotlib) from the profile: key numeric distributions, top categorical
  breakdowns.
- Open-text theme/sentiment summary, if applicable.
- One LLM call: a short narrative highlighting notable patterns from the profile.

## 5. Chat Agent — Adaptive Multi-Tool

**Two execution tiers:**

- **Tier 1 — Tool-calling (local pandas, no LLM cost, near-instant):** a registry of pre-built
  analysis functions extending pulseiq's `SURVEY_TOOL_REGISTRY` pattern — segment comparisons,
  crosstabs, trend analysis, distributions, anomaly/outlier detection, threshold counts. Each
  tool optionally takes a `session_id` to support comparison mode (section 7).
- **Tier 2 — Sandboxed code-gen:** for custom/novel visualizations no tool covers. LLM generates
  pandas/matplotlib/Plotly code, grounded by `get_column_stats(column)` /
  `preview_values(column, n)` calls against the profile. Executed in the sandbox (section 8).
  On execution error, the error + traceback is sent back to the LLM once for a fix-and-retry; a
  second failure returns a friendly error message.

**Orchestration (per turn):**

1. One LLM call (function-calling, fallback model where possible) reads the profile,
   conversation history, and user message, and plans.
2. **Direct questions** ("pie chart of X by Y") resolve in one pass: pick a Tier-1 tool or write
   Tier-2 code, execute, respond.
3. **Open-ended questions** ("what's driving low satisfaction?") run a bounded multi-step loop
   (cap ~4 steps): chain Tier-1 tool calls (free), accumulate findings, then synthesize a
   narrative across multiple charts, escalating to one Tier-2 call only if a custom chart is
   needed to illustrate a finding.
4. Every response includes: a chart, a narrative, generated code (for Tier-2 responses, shown
   via a "Show code" toggle), 2-3 follow-up question suggestions (generated in the same call,
   no extra cost), and caveats (section 6). Chart format: PNG by default; the agent emits
   Plotly JSON instead when the chart has multiple series/dimensions or benefits from
   zoom/hover (e.g. multi-line trends, many-category breakdowns) — a simple rule-of-thumb
   encoded in the code-gen prompt, not a separate decision step.

**Multi-turn state:** stateless per-turn — each turn loads the cached profile/Parquet fresh in
the sandbox; the LLM relies on conversation history (with sliding-window summarization, section
9) to resolve references like "that" or "now break it down further".

**Worst case per question:** 1 grounding/code-gen call + 1 retry = 2 LLM calls (Tier-2). Typical
case: 1 call. Tier-1-only chains add zero LLM calls beyond the single planning/synthesis call(s).

## 6. Per-Chart Caveats

A rules engine (no LLM cost) runs over the profile for every chat response and insight:

- sample size below a threshold for the segment/column(s) involved
- high missingness (% threshold) for involved columns

Produces short caveat strings (e.g. "based on only 12 responses", "this column is 40% missing")
attached to the response.

## 7. Comparison Mode

User selects a second saved session to compare against the current one.

- A `comparisons` record (diff of the two profiles + key segment stats) is computed once and
  cached.
- The agent receives both profiles + the diff summary in context.
- Tier-1 tools accept an optional `session_id` so the agent can query either dataset.
- Supports questions like "what changed since last quarter?".

## 8. Sandbox (Tier 2)

- Subprocess via `multiprocessing`, loads the cached Parquet for the session.
- AST whitelist check before execution: only `pandas`, `numpy`, `matplotlib`, `seaborn`,
  `plotly` imports permitted; no `os`, `sys`, `open`, `__import__`, network access, or dunder
  attribute access.
- `resource`-enforced CPU time + memory limits, plus a hard wall-clock timeout (~15s). Targets
  Linux; the limiter module is isolated so a Windows-dev fallback (timeout-only via
  `multiprocessing.Process.terminate`) can be swapped in.
- Output: a chart (PNG bytes or Plotly figure JSON) and/or a small result table/value.

## 9. Insight Feed

A background task runs after profiling completes:

- Sweeps a fixed set of Tier-1 tools (segment stats × key metrics, anomaly detection, trend
  analysis if a time dimension exists).
- Ranks results by effect size (e.g. largest segment deltas), keeps the top ~5-8.
- One batched LLM call phrases all findings as sentences (not one call per finding).
- Stored in the `insights` table; surfaced as a feed in the UI. Clicking an insight opens its
  chart in the canvas panel.

## 10. Story Report Export

- Gathers pinned charts + insight feed + caveats for the session (and comparison session, if
  active).
- One LLM call composes a narrative structure: executive summary, key findings with embedded
  chart references, caveats.
- Rendered to PDF (e.g. WeasyPrint/reportlab) with embedded chart images.

## 11. Persistence (SQLite)

- `sessions` — id, filename, uploaded_at, row_count, profile_path, data_path
- `chat_messages` — session_id, role, content, generated_code, chart_ref, follow_ups, caveats (JSON)
- `pinned_charts` — session_id, chart_ref, title, source_message_id
- `insights` — session_id, rank, title, summary, supporting_tool_calls (JSON), chart_ref
- `comparisons` — base_session_id, compare_session_id, diff_summary (JSON, cached)

Single user, no auth in v1. Saved sessions are a simple list, no per-user isolation.

## 12. Cost Optimization

- **Model routing:** fallback/mini model for intent routing, Tier-1 tool selection, and history
  summarization; primary GPT-4.1-class model for Tier-2 code-gen, multi-step synthesis, and
  story reports.
- **Caching:** profile, insight feed, and comparison diffs computed once and persisted.
  Identical/near-identical chat questions (hashed against profile version) reuse cached
  code+result.
- **Context budgeting:** conversation history capped with a sliding window; older turns are
  summarized by the fallback model when the window overflows, keeping prompts flat in long
  sessions.

## 13. UI/UX

- **Default view:** single chat stream. The auto-dashboard is the first assistant message
  (stats cards + small charts + open-text summary + narrative). Charts render inline in later
  messages, each with a "Show code" toggle, caveats, and follow-up suggestion chips.
- **Expand to canvas:** any chart can pop into a collapsible right-side panel for a larger
  focused view, with a PNG/Plotly export action.
- **Dashboard tab:** a grid of pinned charts + auto stats + the insight feed (each insight
  clickable to open in canvas). Includes an "Export report (PDF)" action (section 10).
- **Sidebar:** thin session list ("+ New" upload, prior sessions for revisiting/comparison
  selection).
- Theme: black & white, minimal throughout.

## 14. Out of Scope (v1)

- Multi-user accounts/auth.
- PII detection/redaction (considered, deferred — revisit if the tool is used more broadly or
  with externally-sourced survey data).
- Full plan→act→reflect agent loop beyond the bounded ~4-step Tier-1 chain.
- Docker-based sandbox isolation (subprocess + resource limits chosen for v1; Docker remains an
  option if stronger isolation is needed later).
