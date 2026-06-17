from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import Session as SessionModel

router = APIRouter()


@router.get("/sessions")
def list_sessions(db: Session = Depends(get_db)) -> list[dict]:
    rows = db.query(SessionModel).order_by(SessionModel.uploaded_at.desc()).all()
    return [
        {
            "session_id": s.id,
            "filename": s.filename,
            "row_count": s.row_count,
            "uploaded_at": s.uploaded_at.isoformat() if s.uploaded_at else None,
        }
        for s in rows
    ]


@router.get("/sessions/{session_id}")
def get_session(session_id: str, db: Session = Depends(get_db)) -> dict:
    s = db.get(SessionModel, session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": s.id,
        "filename": s.filename,
        "row_count": s.row_count,
        "uploaded_at": s.uploaded_at.isoformat() if s.uploaded_at else None,
        "profile_path": s.profile_path,
    }


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str, db: Session = Depends(get_db)) -> dict:
    s = db.get(SessionModel, session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    db.delete(s)
    db.commit()
    return {"deleted": session_id}
