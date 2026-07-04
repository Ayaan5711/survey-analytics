from __future__ import annotations

import logging
import os
import re

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent import SurveyAnalysisAgent
from session_store import SurveySessionStore


load_dotenv()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("survey_agent.api")

app = FastAPI(title="Survey Insight Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

store = SurveySessionStore()

try:
    survey_agent = SurveyAnalysisAgent()
except RuntimeError as exc:
    survey_agent = None
    startup_error = str(exc)
    logger.exception("agent_init_failed detail=%s", startup_error)
else:
    startup_error = ""
    logger.info("agent_init_ok")


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────

def _get_column_info(df: pd.DataFrame) -> list[dict]:
    """Return per-column metadata for the frontend column explorer."""
    result: list[dict] = []
    total = max(len(df), 1)
    for col in df.columns:
        dtype_str = str(df[col].dtype)
        missing = int(df[col].isna().sum())
        unique = int(df[col].nunique())
        if any(t in dtype_str for t in ("int", "float")):
            col_type = "numeric"
        elif "datetime" in dtype_str:
            col_type = "datetime"
        elif "bool" in dtype_str:
            col_type = "boolean"
        else:
            col_type = "text"
        result.append({
            "name": str(col),
            "type": col_type,
            "dtype": dtype_str,
            "missing": missing,
            "missing_pct": round(missing / total * 100, 1),
            "unique": unique,
        })
    return result


def _parse_data_type(text: str, columns: list) -> str:
    m = re.search(r"DATA TYPE:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    cols_lower = " ".join(str(c).lower() for c in columns)
    for dtype, words in [
        ("Survey / Feedback",  ["survey", "response", "rating", "satisfaction", "feedback", "score", "question"]),
        ("Sales / Financial",  ["revenue", "sales", "profit", "invoice", "price", "cost", "amount", "order"]),
        ("HR / Employee",      ["employee", "staff", "salary", "department", "hire", "position", "manager"]),
        ("Inventory",          ["product", "inventory", "stock", "sku", "quantity", "warehouse", "item"]),
        ("Time-Series",        ["date", "timestamp", "month", "year", "period", "daily", "weekly"]),
    ]:
        if any(w in cols_lower for w in words):
            return dtype
    return "General Data"


def _parse_suggested_questions(text: str) -> list[str]:
    questions: list[str] = []
    block = re.search(r"SUGGESTED QUESTIONS:(.*?)(?:\n\n|\Z)", text, re.IGNORECASE | re.DOTALL)
    source = block.group(1) if block else text
    items = re.findall(r"(?:^|\n)\s*\d+[\.\)]\s*(.+?)(?=\n\s*\d+[\.\)]|\Z)", source, re.DOTALL)
    questions = [q.strip().replace("\n", " ") for q in items if len(q.strip()) > 10]
    return questions[:5]


# ─────────────────────────────────────────
# Models
# ─────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    question: str


class AutoInsightsRequest(BaseModel):
    session_id: str


@app.get("/api/health")
def health() -> dict[str, str]:
    if startup_error:
        return {"status": "degraded", "reason": startup_error}
    return {"status": "ok"}


@app.post("/api/upload")
async def upload_excel(file: UploadFile = File(...)) -> dict:
    filename = file.filename or "survey.xlsx"
    logger.info("upload_request filename=%s", filename)
    extension = os.path.splitext(filename)[1].lower()
    if extension not in {".xlsx", ".xlsm", ".xls"}:
        logger.warning("upload_rejected filename=%s reason=invalid_extension", filename)
        raise HTTPException(status_code=400, detail="Please upload an Excel file.")

    raw = await file.read()
    try:
        session = store.create_from_excel_bytes(filename, raw)
    except Exception as exc:
        logger.exception("upload_parse_failed filename=%s", filename)
        raise HTTPException(status_code=400, detail=f"Unable to parse Excel file: {exc}") from exc

    active_df = session.dataframes[session.active_sheet]
    column_info = _get_column_info(active_df)
    total_cells = max(active_df.shape[0] * active_df.shape[1], 1)
    missing_total = int(active_df.isna().sum().sum())
    overall_missing_pct = round(missing_total / total_cells * 100, 1)
    logger.info(
        "upload_success session_id=%s active_sheet=%s shape=%sx%s",
        session.session_id,
        session.active_sheet,
        active_df.shape[0],
        active_df.shape[1],
    )
    return {
        "session_id": session.session_id,
        "filename": session.filename,
        "sheet_names": session.sheet_names,
        "active_sheet": session.active_sheet,
        "shape": [int(active_df.shape[0]), int(active_df.shape[1])],
        "preview": store.dataframe_preview(active_df, max_rows=8),
        "column_info": column_info,
        "overall_missing_pct": overall_missing_pct,
    }


@app.post("/api/chat")
def chat(payload: ChatRequest) -> dict:
    logger.info("chat_request session_id=%s", payload.session_id)
    if survey_agent is None:
        logger.error("chat_rejected reason=agent_not_configured")
        raise HTTPException(
            status_code=500,
            detail=(
                "Agent is not configured. Set AZURE_OPENAI_DEPLOYMENT, "
                "AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_API_KEY."
            ),
        )

    try:
        session = store.get(payload.session_id)
    except KeyError as exc:
        logger.warning("chat_rejected session_id=%s reason=unknown_session", payload.session_id)
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    try:
        agent_response = survey_agent.answer(session, payload.question)
    except Exception as exc:
        logger.exception("chat_failed session_id=%s", payload.session_id)
        raise HTTPException(status_code=500, detail=f"Agent failed: {exc}") from exc

    answer = str(agent_response.get("answer", ""))
    charts = agent_response.get("charts", [])
    if not isinstance(charts, list):
        charts = []

    session.chat_history.append({"role": "user", "content": payload.question})
    session.chat_history.append({"role": "assistant", "content": answer})
    logger.info("chat_success session_id=%s", payload.session_id)
    return {
        "answer": answer,
        "charts": charts,
        "active_sheet": session.active_sheet,
        "sheet_names": session.sheet_names,
    }


@app.post("/api/auto-insights")
def auto_insights(payload: AutoInsightsRequest) -> dict:
    """Automatically analyze the uploaded dataset and return insights, charts and suggested questions."""
    logger.info("auto_insights_request session_id=%s", payload.session_id)
    if survey_agent is None:
        raise HTTPException(status_code=500, detail="Agent not configured.")

    try:
        session = store.get(payload.session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    prompt = (
        "Perform an automatic exploratory analysis of this dataset. Use EXACTLY this output format:\n\n"
        "DATA TYPE: [one-line label, e.g. 'Survey Responses', 'Sales Report', 'HR Records']\n\n"
        "KEY INSIGHTS:\n"
        "- [specific finding using real column names and values]\n"
        "- [specific finding]\n"
        "- [specific finding]\n\n"
        "Now call run_python to create exactly 2 charts that best illustrate the data "
        "(use bar/pie/histogram as appropriate, with clear titles and axis labels).\n\n"
        "SUGGESTED QUESTIONS:\n"
        "1. [actionable question specific to the column names in this dataset]\n"
        "2. [question]\n"
        "3. [question]\n"
        "4. [question]\n"
        "5. [question]\n\n"
        "Be concise and use actual column names and data values in your response."
    )

    try:
        agent_response = survey_agent.answer(session, prompt)
    except Exception as exc:
        logger.exception("auto_insights_failed session_id=%s", payload.session_id)
        raise HTTPException(status_code=500, detail=f"Agent failed: {exc}") from exc

    answer = str(agent_response.get("answer", ""))
    charts = agent_response.get("charts", [])
    if not isinstance(charts, list):
        charts = []

    active_df = session.dataframes[session.active_sheet]
    cols = list(active_df.columns)
    data_type = _parse_data_type(answer, cols)
    suggested_questions = _parse_suggested_questions(answer)

    # Fill up to 5 with sensible defaults if parsing came up short
    defaults = [
        f"What is the distribution of '{cols[0]}'?" if cols else "What are the key distributions?",
        "Show summary statistics for all numeric columns",
        "Are there any missing values or data quality issues?",
        "What are the most common values in categorical columns?",
        "Identify any outliers or anomalies in the data",
    ]
    for d in defaults:
        if len(suggested_questions) >= 5:
            break
        if d not in suggested_questions:
            suggested_questions.append(d)

    logger.info(
        "auto_insights_success session_id=%s data_type=%s charts=%s",
        payload.session_id, data_type, len(charts),
    )
    return {
        "answer": answer,
        "charts": charts,
        "data_type": data_type,
        "suggested_questions": suggested_questions[:5],
        "active_sheet": session.active_sheet,
        "sheet_names": session.sheet_names,
    }
