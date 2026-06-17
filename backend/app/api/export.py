from __future__ import annotations
import logging
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import Session as SessionModel, Insight, PinnedChart
from app.data.profiler import load_profile
from app.reports.composer import compose_report
from app.reports.renderer import render_pdf
from app.llm.client import llm
from app.llm.prompts import dashboard_narrative_prompt

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/sessions/{session_id}/export/pdf")
async def export_pdf(session_id: str, db: Session = Depends(get_db)) -> Response:
    record = db.get(SessionModel, session_id)
    if not record:
        raise HTTPException(status_code=404, detail="Session not found")
    if not record.profile_path or not Path(record.profile_path).exists():
        raise HTTPException(status_code=404, detail="Profile not found")

    profile = load_profile(Path(record.profile_path))
    insights = db.query(Insight).filter_by(session_id=session_id).order_by(Insight.rank).all()
    pins = db.query(PinnedChart).filter_by(session_id=session_id).all()

    pinned_charts = []
    for p in pins:
        png_path = Path(record.profile_path).parent / f"{p.id}.png"
        if png_path.exists():
            pinned_charts.append({"title": p.title or "", "png_bytes": png_path.read_bytes()})

    # Narrative: use cached or call LLM
    narrative_path = Path(record.profile_path).parent / "narrative.txt"
    if narrative_path.exists():
        narrative = narrative_path.read_text()
    else:
        try:
            narrative = await llm.chat_completion(
                messages=[{"role": "user", "content": dashboard_narrative_prompt(profile)}],
                use_fallback=False, max_tokens=300, temperature=0.4,
            )
            narrative_path.write_text(narrative)
        except Exception as exc:
            logger.error("Narrative LLM call failed for export: %s", exc)
            narrative = f"Dataset: {profile.filename}, {profile.row_count} rows."

    report_data = compose_report(profile, insights, pinned_charts)
    try:
        pdf_bytes = render_pdf(report_data, narrative)
    except Exception as exc:
        logger.error("PDF render failed: %s", exc)
        raise HTTPException(status_code=500, detail="PDF generation failed")

    safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in record.filename)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}_report.pdf"'},
    )
