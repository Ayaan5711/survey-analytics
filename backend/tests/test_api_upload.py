import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_upload_csv_returns_session(client, sample_csv_bytes):
    with patch("app.api.upload.llm.chat_completion",
               new=AsyncMock(return_value="Narrative text.")):
        r = client.post("/api/upload",
                        files={"file": ("responses.csv", sample_csv_bytes, "text/csv")})
    assert r.status_code == 200
    data = r.json()
    assert "session_id" in data
    assert data["row_count"] == 5
    assert "columns" in data


@pytest.mark.asyncio
async def test_upload_excel_single_sheet(client, sample_excel_bytes):
    with patch("app.api.upload.llm.chat_completion",
               new=AsyncMock(return_value="Narrative.")):
        r = client.post("/api/upload",
                        files={"file": ("data.xlsx", sample_excel_bytes,
                                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
    assert r.status_code == 200
    data = r.json()
    assert data["row_count"] == 3


def test_upload_empty_file_returns_400(client):
    r = client.post("/api/upload",
                    files={"file": ("empty.csv", b"", "text/csv")})
    assert r.status_code == 400


def test_upload_unsupported_format_returns_400(client):
    r = client.post("/api/upload",
                    files={"file": ("data.txt", b"hello", "text/plain")})
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_upload_excel_multi_sheet_returns_sheets(client):
    """Multi-sheet Excel upload returns 409 with sheet list for the client to pick."""
    import io, pandas as pd
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        pd.DataFrame({"A": [1]}).to_excel(w, sheet_name="Sheet1", index=False)
        pd.DataFrame({"B": [2]}).to_excel(w, sheet_name="Sheet2", index=False)
    buf.seek(0)
    r = client.post("/api/upload",
                    files={"file": ("multi.xlsx", buf.read(),
                                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
    assert r.status_code == 409
    body = r.json()
    assert "sheets" in body["detail"]
