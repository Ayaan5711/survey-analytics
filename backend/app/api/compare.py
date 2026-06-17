from __future__ import annotations
import dataclasses
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import Session as SessionModel, Comparison
from app.data.profiler import load_profile
from app.compare.engine import compare_profiles

router = APIRouter()


class CompareRequest(BaseModel):
    base_session_id: str
    compare_session_id: str


@router.post("/compare")
def run_comparison(body: CompareRequest, db: Session = Depends(get_db)) -> dict:
    # Return cached if exists
    cached = db.query(Comparison).filter_by(
        base_session_id=body.base_session_id,
        compare_session_id=body.compare_session_id,
    ).first()
    if cached:
        return {"cached": True, "diff": cached.diff_summary}

    base_rec = db.get(SessionModel, body.base_session_id)
    cmp_rec = db.get(SessionModel, body.compare_session_id)
    if not base_rec:
        raise HTTPException(status_code=404, detail="Base session not found")
    if not cmp_rec:
        raise HTTPException(status_code=404, detail="Compare session not found")
    if not base_rec.profile_path or not Path(base_rec.profile_path).exists():
        raise HTTPException(status_code=404, detail="Base profile not found")
    if not cmp_rec.profile_path or not Path(cmp_rec.profile_path).exists():
        raise HTTPException(status_code=404, detail="Compare profile not found")

    base_profile = load_profile(Path(base_rec.profile_path))
    cmp_profile = load_profile(Path(cmp_rec.profile_path))
    diff = compare_profiles(base_profile, cmp_profile)
    diff_dict = dataclasses.asdict(diff)

    record = Comparison(
        base_session_id=body.base_session_id,
        compare_session_id=body.compare_session_id,
        diff_summary=diff_dict,
    )
    db.add(record)
    db.commit()
    return {"cached": False, "diff": diff_dict}
