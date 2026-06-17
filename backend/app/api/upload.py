from __future__ import annotations
import logging
import os
from dataclasses import asdict
from pathlib import Path
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session
from app.data.loader import get_excel_sheet_names, load_file
from app.data.profiler import build_profile
from app.data.quality import check_quality
from app.dashboard.generator import generate_dashboard
from app.db.database import get_db
from app.db.models import Session as SessionModel
from app.llm.client import llm

logger = logging.getLogger(__name__)
router = APIRouter()


def _data_dir() -> Path:
    """Read DATA_DIR at call time so monkeypatched env vars work in tests."""
    return Path(os.getenv("DATA_DIR", "./data/sessions"))


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    background_tasks: BackgroundTasks = BackgroundTasks(),
) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="File is empty")

    lower = (file.filename or "").lower()
    if not (lower.endswith(".csv") or lower.endswith(".xlsx") or lower.endswith(".xls")):
        raise HTTPException(status_code=400, detail="Unsupported file type — upload .csv, .xlsx, or .xls")

    # Multi-sheet Excel gate: if >1 sheet, return 409 with sheet names
    if lower.endswith((".xlsx", ".xls")):
        sheets = get_excel_sheet_names(content)
        if len(sheets) > 1:
            raise HTTPException(
                status_code=409,
                detail={"message": "Excel file has multiple sheets — pick one", "sheets": sheets},
            )

    try:
        df, schema = load_file(content, file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    record = SessionModel(filename=file.filename, row_count=len(df))
    db.add(record)
    db.commit()
    db.refresh(record)

    data_dir = _data_dir()
    try:
        profile = build_profile(df, schema, record.id, file.filename, data_dir)
        quality = check_quality(df, schema)
    except Exception as exc:
        logger.error(f"Profiling failed for session {record.id}: {exc}")
        raise HTTPException(status_code=500, detail="Failed to profile data")

    session_dir = data_dir / record.id
    record.profile_path = str(session_dir / "profile.json")
    record.data_path = str(session_dir / "data.parquet")
    db.commit()

    dashboard = await generate_dashboard(profile, quality, session_dir)

    # Cache narrative to avoid re-calling LLM on subsequent GET /dashboard requests
    (session_dir / "narrative.txt").write_text(dashboard.narrative, encoding="utf-8")

    # Trigger insight generation in the background
    from app.insights.generator import generate_insights as _gen_insights
    background_tasks.add_task(_gen_insights, record.id, profile, record.data_path)

    return {
        "session_id": record.id,
        "filename": file.filename,
        "row_count": profile.row_count,
        "col_count": profile.col_count,
        "columns": list(profile.columns.keys()),
        "open_text_columns": profile.open_text_columns,
        "dashboard": {
            "narrative": dashboard.narrative,
            "charts": dashboard.charts,
            "quality_flags": asdict(dashboard.quality_flags),
            "column_summary": dashboard.column_summary,
        },
    }


@router.post("/upload/sheet")
async def upload_excel_with_sheet(
    file: UploadFile = File(...),
    sheet_name: str = "",
    db: Session = Depends(get_db),
) -> dict:
    """Re-upload the same Excel bytes with a chosen sheet_name (called after 409)."""
    if not sheet_name:
        raise HTTPException(status_code=400, detail="sheet_name is required")
    content = await file.read()
    try:
        df, schema = load_file(content, file.filename or "upload.xlsx", sheet_name=sheet_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    record = SessionModel(filename=f"{file.filename} [{sheet_name}]", row_count=len(df))
    db.add(record)
    db.commit()
    db.refresh(record)

    data_dir = _data_dir()
    profile = build_profile(df, schema, record.id, record.filename, data_dir)
    quality = check_quality(df, schema)
    session_dir = data_dir / record.id
    record.profile_path = str(session_dir / "profile.json")
    record.data_path = str(session_dir / "data.parquet")
    db.commit()

    dashboard = await generate_dashboard(profile, quality, session_dir)

    # Cache narrative to avoid re-calling LLM on subsequent GET /dashboard requests
    (session_dir / "narrative.txt").write_text(dashboard.narrative, encoding="utf-8")

    return {
        "session_id": record.id,
        "filename": record.filename,
        "row_count": profile.row_count,
        "col_count": profile.col_count,
        "columns": list(profile.columns.keys()),
        "open_text_columns": profile.open_text_columns,
        "dashboard": {
            "narrative": dashboard.narrative,
            "charts": dashboard.charts,
            "quality_flags": asdict(dashboard.quality_flags),
            "column_summary": dashboard.column_summary,
        },
    }
