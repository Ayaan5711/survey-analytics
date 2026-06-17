from __future__ import annotations
import base64
import os
from dataclasses import asdict
from pathlib import Path
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import Session as SessionModel
from app.data.profiler import load_profile
from app.data.quality import check_quality
from app.dashboard.charts import deterministic_charts
from app.llm.client import llm
from app.llm.prompts import dashboard_narrative_prompt

router = APIRouter()


@router.get("/sessions/{session_id}/dashboard")
async def get_dashboard(session_id: str, db: Session = Depends(get_db)) -> dict:
    record = db.get(SessionModel, session_id)
    if not record:
        raise HTTPException(status_code=404, detail="Session not found")
    if not record.profile_path or not Path(record.profile_path).exists():
        raise HTTPException(status_code=404, detail="Profile not found — re-upload the file")

    profile = load_profile(Path(record.profile_path))

    df = pd.read_parquet(record.data_path)
    schema = {col: {"type": cp.dtype, "n_unique": cp.n_unique}
              for col, cp in profile.columns.items()}
    quality = check_quality(df, schema)

    session_dir = Path(record.profile_path).parent
    charts = deterministic_charts(profile, session_dir)
    charts_out = [
        {
            "title": c.title,
            "chart_type": c.chart_type,
            "png_b64": base64.b64encode(c.png_bytes).decode(),
            "filename": c.filename,
        }
        for c in charts
    ]

    # Narrative: serve from disk cache to avoid re-calling LLM on every GET
    narrative_path = session_dir / "narrative.txt"
    if narrative_path.exists():
        narrative = narrative_path.read_text()
    else:
        prompt = dashboard_narrative_prompt(profile)
        narrative = await llm.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            use_fallback=False,
            max_tokens=300,
            temperature=0.4,
        )
        narrative_path.write_text(narrative)

    return {
        "session_id": session_id,
        "filename": profile.filename,
        "row_count": profile.row_count,
        "col_count": profile.col_count,
        "narrative": narrative,
        "charts": charts_out,
        "quality_flags": asdict(quality),
        "column_summary": {
            name: {
                "dtype": cp.dtype,
                "missing_pct": cp.missing_pct,
                "n_unique": cp.n_unique,
                "mean": cp.mean,
                "top_values": dict(list(cp.top_values.items())[:5]),
            }
            for name, cp in profile.columns.items()
        },
        "open_text_columns": profile.open_text_columns,
    }
