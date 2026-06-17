import pytest
from app.data.loader import load_file
from app.data.profiler import build_profile
from app.dashboard.charts import deterministic_charts, ChartResult


def test_returns_list_of_chart_results(sample_csv_bytes, tmp_path):
    df, schema = load_file(sample_csv_bytes, "responses.csv")
    profile = build_profile(df, schema, "sess-1", "responses.csv", tmp_path)
    charts = deterministic_charts(profile, tmp_path / "sess-1")
    assert isinstance(charts, list)
    assert all(isinstance(c, ChartResult) for c in charts)


def test_chart_results_have_png_bytes(sample_csv_bytes, tmp_path):
    df, schema = load_file(sample_csv_bytes, "responses.csv")
    profile = build_profile(df, schema, "sess-1", "responses.csv", tmp_path)
    charts = deterministic_charts(profile, tmp_path / "sess-1")
    assert len(charts) >= 1
    for chart in charts:
        assert isinstance(chart.png_bytes, bytes)
        assert len(chart.png_bytes) > 0


def test_chart_has_title(sample_csv_bytes, tmp_path):
    df, schema = load_file(sample_csv_bytes, "responses.csv")
    profile = build_profile(df, schema, "sess-1", "responses.csv", tmp_path)
    charts = deterministic_charts(profile, tmp_path / "sess-1")
    for c in charts:
        assert c.title
