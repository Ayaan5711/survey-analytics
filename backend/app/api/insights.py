from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import Session as SessionModel, Insight

router = APIRouter()


@router.get("/sessions/{session_id}/insights")
def list_insights(session_id: str, db: Session = Depends(get_db)) -> list[dict]:
    record = db.get(SessionModel, session_id)
    if not record:
        raise HTTPException(status_code=404, detail="Session not found")
    rows = db.query(Insight).filter_by(session_id=session_id).order_by(Insight.rank).all()
    return [
        {
            "insight_id": r.id,
            "rank": r.rank,
            "title": r.title,
            "summary": r.summary,
            "supporting_tool_calls": r.supporting_tool_calls,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]
