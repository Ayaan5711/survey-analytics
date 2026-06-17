from __future__ import annotations
import base64
import logging
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import Session as SessionModel, ChatMessage, PinnedChart
from app.data.profiler import load_profile
from app.llm.agent import ChatAgent

logger = logging.getLogger(__name__)
router = APIRouter()
_agent = ChatAgent()


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


class PinRequest(BaseModel):
    png_b64: str
    title: str
    source_message_id: str | None = None


@router.post("/sessions/{session_id}/chat")
async def chat(
    session_id: str,
    body: ChatRequest,
    db: Session = Depends(get_db),
) -> dict:
    record = db.get(SessionModel, session_id)
    if not record:
        raise HTTPException(status_code=404, detail="Session not found")
    if not record.profile_path or not Path(record.profile_path).exists():
        raise HTTPException(status_code=404, detail="Profile not found — re-upload the file")
    if not record.data_path or not Path(record.data_path).exists():
        raise HTTPException(status_code=404, detail="Data file not found — re-upload the file")

    profile = load_profile(Path(record.profile_path))

    safe_history = [
        {"role": m["role"], "content": str(m.get("content", ""))}
        for m in body.history
        if m.get("role") in ("user", "assistant")
    ][-20:]

    try:
        response = await _agent.run(profile, record.data_path, body.message, safe_history)
    except Exception as exc:
        logger.error("Agent error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Analysis failed — try rephrasing your question")

    chart_paths = None
    if response.chart:
        if "png_b64" in response.chart:
            chart_paths = [{"type": "png", "title": response.chart.get("title", "")}]
        elif "plotly_json" in response.chart:
            chart_paths = [{"type": "plotly", "title": response.chart.get("title", "")}]

    msg = ChatMessage(
        session_id=session_id,
        role="assistant",
        content=response.content,
        generated_code=response.generated_code,
        chart_paths=chart_paths,
        follow_ups=response.follow_ups,
        caveats=response.caveats,
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)

    return {
        "message_id": msg.id,
        "role": "assistant",
        "content": response.content,
        "chart": response.chart,
        "generated_code": response.generated_code,
        "follow_ups": response.follow_ups,
        "caveats": response.caveats,
        "tool_calls_made": response.tool_calls_made,
    }


@router.get("/sessions/{session_id}/messages")
def list_messages(session_id: str, db: Session = Depends(get_db)) -> list[dict]:
    record = db.get(SessionModel, session_id)
    if not record:
        raise HTTPException(status_code=404, detail="Session not found")
    return [
        {
            "message_id": m.id,
            "role": m.role,
            "content": m.content,
            "follow_ups": m.follow_ups,
            "caveats": m.caveats,
            "created_at": m.created_at.isoformat() if m.created_at else None,
        }
        for m in record.messages
    ]


@router.post("/sessions/{session_id}/pins")
def pin_chart(
    session_id: str,
    body: PinRequest,
    db: Session = Depends(get_db),
) -> dict:
    record = db.get(SessionModel, session_id)
    if not record:
        raise HTTPException(status_code=404, detail="Session not found")
    pin = PinnedChart(
        session_id=session_id,
        chart_path=f"pin_{session_id}_{len(record.pinned_charts)}",
        title=body.title,
        source_message_id=body.source_message_id,
    )
    db.add(pin)
    db.commit()
    db.refresh(pin)
    return {"pin_id": pin.id, "title": pin.title}


@router.get("/sessions/{session_id}/pins")
def list_pins(session_id: str, db: Session = Depends(get_db)) -> list[dict]:
    record = db.get(SessionModel, session_id)
    if not record:
        raise HTTPException(status_code=404, detail="Session not found")
    return [
        {
            "pin_id": p.id,
            "title": p.title,
            "created_at": p.created_at.isoformat() if p.created_at else None,
        }
        for p in record.pinned_charts
    ]


@router.delete("/sessions/{session_id}/pins/{pin_id}")
def delete_pin(session_id: str, pin_id: str, db: Session = Depends(get_db)) -> dict:
    pin = db.get(PinnedChart, pin_id)
    if not pin or pin.session_id != session_id:
        raise HTTPException(status_code=404, detail="Pin not found")
    db.delete(pin)
    db.commit()
    return {"deleted": pin_id}
