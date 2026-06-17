import base64
import pytest
from unittest.mock import AsyncMock, patch
from app.data.loader import load_file
from app.data.profiler import build_profile
from app.data.quality import check_quality
from app.dashboard.generator import generate_dashboard, DashboardResponse


@pytest.mark.asyncio
async def test_generate_dashboard_returns_response(sample_csv_bytes, tmp_path):
    df, schema = load_file(sample_csv_bytes, "responses.csv")
    profile = build_profile(df, schema, "sess-1", "responses.csv", tmp_path)
    quality = check_quality(df, schema)

    with patch("app.dashboard.generator.llm.chat_completion",
               new=AsyncMock(return_value="This dataset has 5 rows and looks interesting.")):
        result = await generate_dashboard(profile, quality, tmp_path / "sess-1")

    assert isinstance(result, DashboardResponse)
    assert result.narrative
    assert isinstance(result.charts, list)
    assert len(result.charts) >= 1


@pytest.mark.asyncio
async def test_dashboard_charts_are_base64(sample_csv_bytes, tmp_path):
    df, schema = load_file(sample_csv_bytes, "responses.csv")
    profile = build_profile(df, schema, "sess-1", "responses.csv", tmp_path)
    quality = check_quality(df, schema)

    with patch("app.dashboard.generator.llm.chat_completion",
               new=AsyncMock(return_value="Narrative.")):
        result = await generate_dashboard(profile, quality, tmp_path / "sess-1")

    for chart in result.charts:
        # Should be valid base64
        base64.b64decode(chart["png_b64"])


@pytest.mark.asyncio
async def test_dashboard_includes_quality_flags(sample_csv_bytes, tmp_path):
    df, schema = load_file(sample_csv_bytes, "responses.csv")
    profile = build_profile(df, schema, "sess-1", "responses.csv", tmp_path)
    quality = check_quality(df, schema)

    with patch("app.dashboard.generator.llm.chat_completion",
               new=AsyncMock(return_value="Narrative.")):
        result = await generate_dashboard(profile, quality, tmp_path / "sess-1")

    assert result.quality_flags is not None
    assert hasattr(result.quality_flags, "duplicate_rows")
