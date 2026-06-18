from __future__ import annotations
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.llm.agent import ChatResponse


def _upload(client, csv_bytes):
    with patch("app.api.upload.llm.chat_completion", new=AsyncMock(return_value="Narrative.")):
        r = client.post("/api/upload", files={"file": ("test.csv", csv_bytes, "text/csv")})
    assert r.status_code == 200
    return r.json()["session_id"]


def _mock_agent_response():
    return ChatResponse(
        role="assistant",
        content="Engineering has the highest satisfaction score of 4.0.",
        chart={"png_b64": "aGVsbG8=", "title": "Satisfaction by Department"},
        generated_code=None,
        follow_ups=["What about Salary?", "Any outliers?", "Compare with HR?"],
        caveats=["Based on 5 responses."],
        tool_calls_made=["segment_stats"],
    )


def test_chat_returns_response(client, sample_csv_bytes):
    sid = _upload(client, sample_csv_bytes)
    with patch("app.api.chat._agent.run", new=AsyncMock(return_value=_mock_agent_response())):
        r = client.post(f"/api/sessions/{sid}/chat", json={"message": "Satisfaction by Department", "history": []})
    assert r.status_code == 200
    data = r.json()
    assert data["role"] == "assistant"
    assert data["content"]
    assert "message_id" in data
    assert isinstance(data["follow_ups"], list)


def test_chat_persists_message(client, sample_csv_bytes):
    sid = _upload(client, sample_csv_bytes)
    with patch("app.api.chat._agent.run", new=AsyncMock(return_value=_mock_agent_response())):
        client.post(f"/api/sessions/{sid}/chat", json={"message": "hello", "history": []})
    r = client.get(f"/api/sessions/{sid}/messages")
    assert r.status_code == 200
    msgs = r.json()
    # Both the user's message and the assistant's reply are now persisted,
    # ordered chronologically (user first).
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"


def test_chat_session_not_found(client):
    r = client.post("/api/sessions/bad-id/chat", json={"message": "hi", "history": []})
    assert r.status_code == 404


def test_pin_chart(client, sample_csv_bytes):
    sid = _upload(client, sample_csv_bytes)
    r = client.post(f"/api/sessions/{sid}/pins", json={
        "png_b64": "aGVsbG8=", "title": "My Chart", "source_message_id": None
    })
    assert r.status_code == 200
    assert "pin_id" in r.json()


def test_list_pins(client, sample_csv_bytes):
    sid = _upload(client, sample_csv_bytes)
    client.post(f"/api/sessions/{sid}/pins", json={"png_b64": "aGVsbG8=", "title": "Chart 1"})
    r = client.get(f"/api/sessions/{sid}/pins")
    assert r.status_code == 200
    assert len(r.json()) == 1


def test_delete_pin(client, sample_csv_bytes):
    sid = _upload(client, sample_csv_bytes)
    r = client.post(f"/api/sessions/{sid}/pins", json={"png_b64": "aGVsbG8=", "title": "To Delete"})
    pin_id = r.json()["pin_id"]
    r2 = client.delete(f"/api/sessions/{sid}/pins/{pin_id}")
    assert r2.status_code == 200
    assert client.get(f"/api/sessions/{sid}/pins").json() == []
