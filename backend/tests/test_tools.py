from __future__ import annotations
import pytest
import pandas as pd
from app.tools.registry import dispatch_tool, ToolResult, TOOL_NAMES


@pytest.fixture
def survey_df():
    return pd.DataFrame({
        "Department": ["HR", "HR", "Engineering", "Engineering", "Sales"],
        "Satisfaction": [4, 2, 5, 3, 4],
        "Salary": [50000, 60000, 80000, 75000, 55000],
        "Comments": ["Great", "Bad", "Good", "OK", "Fine"],
    })


def test_tool_names_list():
    assert "segment_stats" in TOOL_NAMES
    assert "distribution" in TOOL_NAMES
    assert "crosstab" in TOOL_NAMES
    assert "anomalies" in TOOL_NAMES
    assert "threshold_count" in TOOL_NAMES


def test_segment_stats_returns_tool_result(survey_df):
    result = dispatch_tool(survey_df, "segment_stats", {"group_col": "Department", "metric_col": "Satisfaction"})
    assert isinstance(result, ToolResult)
    assert result.tool_name == "segment_stats"
    assert isinstance(result.table, list)
    assert len(result.table) > 0
    assert result.summary
    assert isinstance(result.png_bytes, bytes)


def test_segment_stats_table_has_expected_keys(survey_df):
    result = dispatch_tool(survey_df, "segment_stats", {"group_col": "Department", "metric_col": "Satisfaction"})
    row = result.table[0]
    assert "Department" in row
    assert "mean" in row
    assert "count" in row


def test_distribution_categorical(survey_df):
    result = dispatch_tool(survey_df, "distribution", {"column": "Department"})
    assert result.tool_name == "distribution"
    assert isinstance(result.png_bytes, bytes)
    assert result.table


def test_distribution_numeric(survey_df):
    result = dispatch_tool(survey_df, "distribution", {"column": "Satisfaction"})
    assert isinstance(result.png_bytes, bytes)


def test_crosstab(survey_df):
    result = dispatch_tool(survey_df, "crosstab", {"row_col": "Department", "col_col": "Satisfaction"})
    assert result.tool_name == "crosstab"
    assert result.png_bytes
    assert result.table


def test_anomalies(survey_df):
    result = dispatch_tool(survey_df, "anomalies", {"column": "Salary"})
    assert result.tool_name == "anomalies"
    assert "outlier" in result.summary.lower() or "no outlier" in result.summary.lower()


def test_threshold_count(survey_df):
    result = dispatch_tool(survey_df, "threshold_count", {"column": "Satisfaction", "threshold": 3, "operator": "gt"})
    assert result.tool_name == "threshold_count"
    assert isinstance(result.table, dict)
    assert "count" in result.table


def test_unknown_tool_raises():
    df = pd.DataFrame({"A": [1]})
    with pytest.raises(ValueError, match="Unknown tool"):
        dispatch_tool(df, "nonexistent_tool", {})


def test_caveat_on_small_segment(survey_df):
    # Sales has only 1 row — should get a caveat
    result = dispatch_tool(survey_df, "segment_stats", {"group_col": "Department", "metric_col": "Satisfaction"})
    assert result.caveat is not None
