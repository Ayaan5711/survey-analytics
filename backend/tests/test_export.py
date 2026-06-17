from __future__ import annotations
import pytest
from unittest.mock import AsyncMock, patch
from app.data.loader import load_file
from app.data.profiler import build_profile
from app.reports.composer import compose_report, ReportData
from app.reports.renderer import render_pdf


@pytest.fixture
def profile(sample_csv_bytes, tmp_path):
    df, schema = load_file(sample_csv_bytes, "responses.csv")
    return build_profile(df, schema, "rpt-1", "responses.csv", tmp_path)


def test_compose_report_returns_report_data(profile):
    data = compose_report(profile, insights=[], pinned_charts=[])
    assert isinstance(data, ReportData)
    assert data.filename == "responses.csv"
    assert data.row_count == 5


def test_compose_report_with_insights(profile):
    from app.db.models import Insight
    ins = Insight(session_id="rpt-1", rank=1, title="Top Finding",
                  summary="HR is happiest.", supporting_tool_calls=None)
    data = compose_report(profile, insights=[ins], pinned_charts=[])
    assert len(data.insights) == 1
    assert data.insights[0]["title"] == "Top Finding"


def test_render_pdf_returns_bytes(profile):
    data = compose_report(profile, insights=[], pinned_charts=[])
    pdf_bytes = render_pdf(data, narrative="This dataset has 5 rows.")
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes[:4] == b"%PDF"


def test_render_pdf_with_charts(profile, tmp_path):
    from app.dashboard.charts import deterministic_charts
    from pathlib import Path
    session_dir = tmp_path / profile.session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    charts = deterministic_charts(profile, session_dir)
    pinned = [{"title": c.title, "png_bytes": c.png_bytes} for c in charts[:1]]
    data = compose_report(profile, insights=[], pinned_charts=pinned)
    pdf_bytes = render_pdf(data, narrative="Summary here.")
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
