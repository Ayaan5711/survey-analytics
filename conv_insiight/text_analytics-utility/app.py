from __future__ import annotations

import base64
import io
import logging
import os
import re

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import agent as _agent
from agent import SurveyAnalysisAgent
from session_store import SurveySessionStore

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO),
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("survey_agent.api")

MAX_UPLOAD_MB = 200  # guard against oversized uploads

app = FastAPI(title="Survey Insight Agent API", version="2.0.0")

# FastAPI is an internal service reached only by the Java servlet (localhost);
# the browser never calls it directly. Kept permissive for that local proxy.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

store = SurveySessionStore()

try:
    survey_agent = SurveyAnalysisAgent()
    startup_error = ""
    logger.info("agent_init_ok")
except Exception as exc:  # noqa: BLE001
    survey_agent = None
    startup_error = str(exc)
    logger.exception("agent_init_failed detail=%s", startup_error)


def _column_info(df: pd.DataFrame) -> list[dict]:
    result, total = [], max(len(df), 1)
    for col in df.columns:
        dt = str(df[col].dtype)
        miss = int(df[col].isna().sum())
        if any(t in dt for t in ("int", "float")):
            ctype = "numeric"
        elif "datetime" in dt:
            ctype = "datetime"
        elif "bool" in dt:
            ctype = "boolean"
        else:
            ctype = "text"
        result.append({"name": str(col), "type": ctype, "dtype": dt,
                        "missing": miss, "missing_pct": round(miss / total * 100, 1),
                        "unique": int(df[col].nunique())})
    return result


def _parse_data_type(text: str, columns: list) -> str:
    m = re.search(r"DATA TYPE:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    cl = " ".join(str(c).lower() for c in columns)
    for dtype, words in [
        ("Survey / Feedback", ["survey", "response", "rating", "satisfaction", "feedback", "score", "question"]),
        ("Sales / Financial", ["revenue", "sales", "profit", "invoice", "price", "cost", "amount", "order"]),
        ("HR / Employee", ["employee", "staff", "salary", "department", "hire", "position", "manager"]),
        ("Inventory", ["product", "inventory", "stock", "sku", "quantity", "warehouse", "item"]),
        ("Time-Series", ["date", "timestamp", "month", "year", "period", "daily", "weekly"]),
    ]:
        if any(w in cl for w in words):
            return dtype
    return "General Data"


def _parse_suggested_questions(text: str) -> list[str]:
    block = re.search(r"SUGGESTED QUESTIONS:(.*?)(?:\n\n|\Z)", text, re.IGNORECASE | re.DOTALL)
    src = block.group(1) if block else text
    items = re.findall(r"(?:^|\n)\s*\d+[\.\)]\s*(.+?)(?=\n\s*\d+[\.\)]|\Z)", src, re.DOTALL)
    return [q.strip().replace("\n", " ") for q in items if len(q.strip()) > 10][:5]


class ChatRequest(BaseModel):
    session_id: str
    question: str


class AutoInsightsRequest(BaseModel):
    session_id: str


class SessionRequest(BaseModel):
    session_id: str


class PinnedChart(BaseModel):
    title: str = ""
    mime_type: str = "image/png"
    image_base64: str


class ExportReportRequest(BaseModel):
    session_id: str
    charts: list[PinnedChart] = []  # user-pinned charts from chat/dashboard; falls back to standard charts if empty


# ─── Deterministic dashboard: stats + quality + a few standard charts ───────
def _quality(df: pd.DataFrame) -> dict:
    total = max(len(df), 1)
    dup = int(df.duplicated().sum())
    empty = [str(c) for c in df.columns if df[c].isna().mean() * 100 >= 90]
    const = [str(c) for c in df.columns if df[c].nunique(dropna=True) <= 1]
    return {"duplicate_rows": dup, "mostly_empty_columns": empty, "constant_columns": const}


def _std_charts(df: pd.DataFrame) -> list[dict]:
    """Up to 3 deterministic charts: a categorical breakdown, a numeric histogram, missing %."""
    charts: list[dict] = []
    cat = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    try:
        if cat:
            import json
            charts.append(json.loads(_agent._t_distribution(df, cat[0]))["charts"][0])
        if num:
            import json
            charts.append(json.loads(_agent._t_distribution(df, num[0]))["charts"][0])
    except Exception:
        pass
    miss = {c: round(df[c].isna().mean() * 100, 1) for c in df.columns}
    miss = {k: v for k, v in sorted(miss.items(), key=lambda x: -x[1]) if v > 0}
    if miss:
        cols = list(miss.keys())[:10]
        fig = _agent._bar(cols, [miss[c] for c in cols], "Missing data %", "Column", "Missing %")
        charts.append(_agent._chart(fig, "Missing data %"))
    return charts


def _sample_df(session, limit: int = 50000) -> pd.DataFrame:
    # Benchmarked at ~1M rows: duplicated()/nunique()/missing% all complete in
    # well under a second, so we use the FULL dataset here rather than a head()
    # slice — a head() sample gives misleading stats (missing %, unique counts,
    # quality flags) whenever the file is sorted or grouped by any column.
    return session.df


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "degraded", "reason": startup_error} if startup_error else {"status": "ok"}


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)) -> dict:
    uploads: list[tuple[str, bytes]] = []
    for f in files:
        name = f.filename or "data.xlsx"
        if not name.lower().endswith((".xlsx", ".xlsm", ".xls", ".csv")):
            raise HTTPException(400, f"Unsupported file: {name}. Use CSV or Excel.")
        raw = await f.read()
        if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(400, f"{name} exceeds {MAX_UPLOAD_MB} MB.")
        uploads.append((name, raw))

    try:
        session = store.create_from_uploads(uploads)
    except Exception as exc:  # noqa: BLE001
        logger.exception("upload_parse_failed")
        raise HTTPException(400, f"Unable to read file(s): {exc}") from exc

    # Full dataset, not a head() slice — see _sample_df's note above.
    sample = session.df
    if session.row_count > 2_000_000:
        # Single gunicorn worker: a very large synchronous parse/stat pass can
        # delay other users' concurrent requests. Not a hard limit — just
        # visibility for ops if uploads start trending much larger than 1M rows.
        logger.warning("upload_large_dataset session=%s rows=%s — may briefly delay other requests "
                       "(single worker)", session.session_id, session.row_count)
    if session.skipped_files:
        logger.warning("upload_skipped_files session=%s skipped=%s", session.session_id, session.skipped_files)
    logger.info("upload_success session=%s files=%s rows=%s", session.session_id, session.files, session.row_count)
    return {
        "session_id": session.session_id,
        "filename": session.filename,
        "files": session.files,
        "skipped_files": session.skipped_files,  # schema mismatch — not included in the combined dataset
        "sheet_names": session.sheet_names,
        "active_sheet": session.active_sheet,
        "shape": [int(session.row_count), int(len(session.columns))],
        "preview": store.dataframe_preview(session, max_rows=8),
        "column_info": _column_info(sample),
        "overall_missing_pct": round(float(sample.isna().sum().sum()) / max(sample.size, 1) * 100, 1),
    }


@app.post("/api/chat")
def chat(payload: ChatRequest) -> dict:
    if survey_agent is None:
        raise HTTPException(500, "Agent is not configured.")
    try:
        session = store.get(payload.session_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    try:
        resp = survey_agent.answer(session, payload.question)
    except Exception as exc:  # noqa: BLE001
        logger.exception("chat_failed session=%s", payload.session_id)
        raise HTTPException(500, f"Agent failed: {exc}") from exc

    session.add_message("user", payload.question)
    session.add_message("assistant", str(resp.get("answer", "")))
    return {"answer": resp.get("answer", ""), "charts": resp.get("charts", []),
            "active_sheet": session.active_sheet, "sheet_names": session.sheet_names}


@app.post("/api/auto-insights")
def auto_insights(payload: AutoInsightsRequest) -> dict:
    if survey_agent is None:
        raise HTTPException(500, "Agent not configured.")
    try:
        session = store.get(payload.session_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc

    prompt = (
        "Perform an automatic exploratory analysis of this dataset. Use EXACTLY this format:\n\n"
        "DATA TYPE: [one-line label]\n\nKEY INSIGHTS:\n- [finding]\n- [finding]\n- [finding]\n\n"
        "Now call the tools to create exactly 2 charts that best illustrate the data.\n\n"
        "SUGGESTED QUESTIONS:\n1. [question]\n2. [question]\n3. [question]\n4. [question]\n5. [question]\n\n"
        "Use actual column names and values."
    )
    try:
        resp = survey_agent.answer(session, prompt)
    except Exception as exc:  # noqa: BLE001
        logger.exception("auto_insights_failed session=%s", payload.session_id)
        raise HTTPException(500, f"Agent failed: {exc}") from exc

    answer = str(resp.get("answer", ""))
    suggested = _parse_suggested_questions(answer)
    defaults = [
        f"What is the distribution of '{session.columns[0]}'?" if session.columns else "What are the key distributions?",
        "Show summary statistics for all numeric columns",
        "Are there any data quality issues or missing values?",
        "What are the most common values in categorical columns?",
        "Identify any outliers or anomalies in the data",
    ]
    for d in defaults:
        if len(suggested) >= 5:
            break
        if d not in suggested:
            suggested.append(d)

    return {"answer": answer, "charts": resp.get("charts", []),
            "data_type": _parse_data_type(answer, session.columns),
            "suggested_questions": suggested[:5],
            "active_sheet": session.active_sheet, "sheet_names": session.sheet_names}


@app.post("/api/dashboard")
def dashboard(payload: SessionRequest) -> dict:
    try:
        session = store.get(payload.session_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    df = _sample_df(session)
    missing_pct = round(float(df.isna().sum().sum()) / max(df.size, 1) * 100, 1)
    return {
        "filename": session.filename,
        "files": session.files,
        "stats": {"rows": int(session.row_count), "columns": len(session.columns),
                  "missing_pct": missing_pct, "files_count": len(session.files)},
        "quality": _quality(df),
        "column_info": _column_info(df),
        "charts": _std_charts(df),
    }


@app.post("/api/export-report")
def export_report(payload: ExportReportRequest) -> dict:
    """Build a self-contained PDF report (overview + quality + charts) and return it
    base64. Uses matplotlib's PdfPages (already a dependency) — no new package.

    If the caller pins specific charts from chat/dashboard (payload.charts), the
    report uses exactly those instead of the 3 standard dashboard charts — this
    is what backs the "pin to report" button in the UI."""
    try:
        session = store.get(payload.session_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from PIL import Image

    df = _sample_df(session)
    q = _quality(df)
    charts = [c.dict() for c in payload.charts] if payload.charts else _std_charts(df)
    missing_pct = round(float(df.isna().sum().sum()) / max(df.size, 1) * 100, 1)

    ql = []
    if q["duplicate_rows"]:
        ql.append(f"{q['duplicate_rows']} duplicate rows")
    if q["mostly_empty_columns"]:
        ql.append(f"{len(q['mostly_empty_columns'])} mostly-empty columns")
    if q["constant_columns"]:
        ql.append(f"{len(q['constant_columns'])} constant columns")
    quality_text = "\n".join(f"- {line}" for line in ql) if ql else "No major issues detected."

    buf = io.BytesIO()
    with PdfPages(buf) as pdf_out:
        # Overview + quality title page.
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
        fig.text(0.08, 0.93, "Survey Insight Report", fontsize=20, fontweight="bold")
        fig.text(0.08, 0.88, f"Dataset: {session.filename}", fontsize=11)
        fig.text(0.08, 0.85,
                 f"{session.row_count:,} rows (all files combined)  |  {len(session.columns)} columns  |  "
                 f"{len(session.files)} file(s)  |  {missing_pct}% missing", fontsize=11)
        fig.text(0.08, 0.79, "Data quality", fontsize=13, fontweight="bold")
        fig.text(0.08, 0.76, quality_text, fontsize=11, va="top")
        pdf_out.savefig(fig)
        plt.close(fig)

        # One chart per page.
        for c in charts:
            try:
                img = Image.open(io.BytesIO(base64.b64decode(c["image_base64"])))
                fig = plt.figure(figsize=(8.27, 11.69))
                ax = fig.add_axes((0.05, 0.1, 0.9, 0.8))
                ax.imshow(img)
                ax.axis("off")
                if c.get("title"):
                    ax.set_title(c["title"], fontsize=12)
                pdf_out.savefig(fig)
                plt.close(fig)
            except Exception:
                continue

    data = buf.getvalue()
    return {"filename": f"survey_report_{session.session_id[:8]}.pdf",
            "pdf_base64": base64.b64encode(data).decode()}
