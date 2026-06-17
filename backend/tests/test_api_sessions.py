import pytest
from unittest.mock import AsyncMock, patch


def _upload(client, csv_bytes):
    with patch("app.api.upload.llm.chat_completion",
               new=AsyncMock(return_value="Narrative.")):
        r = client.post("/api/upload",
                        files={"file": ("test.csv", csv_bytes, "text/csv")})
    assert r.status_code == 200
    return r.json()["session_id"]


def test_list_sessions_empty(client):
    r = client.get("/api/sessions")
    assert r.status_code == 200
    assert r.json() == []


def test_list_sessions_after_upload(client, sample_csv_bytes):
    sid = _upload(client, sample_csv_bytes)
    r = client.get("/api/sessions")
    assert r.status_code == 200
    ids = [s["session_id"] for s in r.json()]
    assert sid in ids


def test_get_session_by_id(client, sample_csv_bytes):
    sid = _upload(client, sample_csv_bytes)
    r = client.get(f"/api/sessions/{sid}")
    assert r.status_code == 200
    assert r.json()["session_id"] == sid


def test_get_session_not_found(client):
    r = client.get("/api/sessions/nonexistent")
    assert r.status_code == 404


def test_delete_session(client, sample_csv_bytes):
    sid = _upload(client, sample_csv_bytes)
    r = client.delete(f"/api/sessions/{sid}")
    assert r.status_code == 200
    r2 = client.get(f"/api/sessions/{sid}")
    assert r2.status_code == 404


def test_get_dashboard_returns_narrative(client, sample_csv_bytes):
    sid = _upload(client, sample_csv_bytes)
    r = client.get(f"/api/sessions/{sid}/dashboard")
    assert r.status_code == 200
    data = r.json()
    assert "narrative" in data
    assert "charts" in data
    assert "quality_flags" in data
