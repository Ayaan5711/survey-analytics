from __future__ import annotations
import base64
import logging
from datetime import datetime
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
    compare_session_id: str | None = None


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

    # Persist the user's message so it survives tab switches / reloads.
    db.add(ChatMessage(session_id=session_id, role="user", content=body.message))
    db.commit()

    # Optional comparison context: feed a textual diff summary to the agent.
    compare_diff = None
    if body.compare_session_id:
        compare_diff = _build_compare_summary(db, session_id, body.compare_session_id)

    try:
        response = await _agent.run(
            profile, record.data_path, body.message, safe_history, compare_diff=compare_diff
        )
    except Exception as exc:
        logger.error("Agent error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Analysis failed — try rephrasing your question")

    # Store the full chart payload (png_b64 / plotly_json + title) so it can be
    # re-rendered when the chat history is reloaded.
    table_payload = {"rows": response.table, "title": response.table_title} if response.table else None

    msg = ChatMessage(
        session_id=session_id,
        role="assistant",
        content=response.content,
        generated_code=response.generated_code,
        chart_paths=response.chart,
        data_table=table_payload,
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
        "table": table_payload,
        "follow_ups": response.follow_ups,
        "caveats": response.caveats,
        "tool_calls_made": response.tool_calls_made,
    }


def _build_compare_summary(db: Session, base_id: str, compare_id: str) -> str | None:
    """Compute (and reuse) a textual diff summary between two sessions for the agent."""
    from app.compare.engine import compare_profiles
    base_rec = db.get(SessionModel, base_id)
    cmp_rec = db.get(SessionModel, compare_id)
    if not base_rec or not cmp_rec or not base_rec.profile_path or not cmp_rec.profile_path:
        return None
    if not Path(base_rec.profile_path).exists() or not Path(cmp_rec.profile_path).exists():
        return None
    base_p = load_profile(Path(base_rec.profile_path))
    cmp_p = load_profile(Path(cmp_rec.profile_path))
    diff = compare_profiles(base_p, cmp_p)
    lines = [
        f"Base '{diff.base_filename}' ({diff.base_row_count} rows) vs "
        f"compare '{diff.compare_filename}' ({diff.compare_row_count} rows); "
        f"row delta {diff.row_count_delta:+d}.",
    ]
    for d in diff.column_diffs:
        if "mean_delta" in d:
            lines.append(f"- {d['column']}: mean {d['base_mean']} → {d['compare_mean']} "
                         f"(Δ {d['mean_delta']:+})")
    if diff.only_in_base:
        lines.append(f"- columns only in base: {', '.join(diff.only_in_base)}")
    if diff.only_in_compare:
        lines.append(f"- columns only in compare: {', '.join(diff.only_in_compare)}")
    return "\n".join(lines)


@router.get("/sessions/{session_id}/messages")
def list_messages(session_id: str, db: Session = Depends(get_db)) -> list[dict]:
    record = db.get(SessionModel, session_id)
    if not record:
        raise HTTPException(status_code=404, detail="Session not found")
    ordered = sorted(record.messages, key=lambda m: m.created_at or datetime.min)
    return [
        {
            "message_id": m.id,
            "role": m.role,
            "content": m.content,
            "chart": m.chart_paths,
            "generated_code": m.generated_code,
            "table": m.data_table,
            "follow_ups": m.follow_ups,
            "caveats": m.caveats,
            "created_at": m.created_at.isoformat() if m.created_at else None,
        }
        for m in ordered
    ]


@router.post("/sessions/{session_id}/pins")
def pin_chart(
    session_id: str,
    body: PinRequest,
    db: Session = Depends(get_db),
) -> dict:
    import base64
    record = db.get(SessionModel, session_id)
    if not record:
        raise HTTPException(status_code=404, detail="Session not found")
    pin = PinnedChart(
        session_id=session_id,
        chart_path="",
        title=body.title,
        source_message_id=body.source_message_id,
    )
    db.add(pin)
    db.flush()  # populate pin.id before writing to disk

    session_dir = Path(record.profile_path).parent
    png_path = session_dir / f"{pin.id}.png"
    png_path.write_bytes(base64.b64decode(body.png_b64))
    pin.chart_path = str(png_path)

    db.commit()
    db.refresh(pin)
    return {"pin_id": pin.id, "title": pin.title}


@router.get("/sessions/{session_id}/pins")
def list_pins(session_id: str, db: Session = Depends(get_db)) -> list[dict]:
    record = db.get(SessionModel, session_id)
    if not record:
        raise HTTPException(status_code=404, detail="Session not found")
    pins = []
    for p in record.pinned_charts:
        png_b64 = None
        if p.chart_path and Path(p.chart_path).exists():
            png_b64 = base64.b64encode(Path(p.chart_path).read_bytes()).decode()
        pins.append({
            "pin_id": p.id,
            "title": p.title,
            "png_b64": png_b64,
            "created_at": p.created_at.isoformat() if p.created_at else None,
        })
    return pins


@router.delete("/sessions/{session_id}/pins/{pin_id}")
def delete_pin(session_id: str, pin_id: str, db: Session = Depends(get_db)) -> dict:
    pin = db.get(PinnedChart, pin_id)
    if not pin or pin.session_id != session_id:
        raise HTTPException(status_code=404, detail="Pin not found")
    db.delete(pin)
    db.commit()
    return {"deleted": pin_id}
