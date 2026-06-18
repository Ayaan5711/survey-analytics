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


@pytest.fixture
def inflation_df():
    return pd.DataFrame({
        "City": ["Delhi", "Delhi", "Mumbai", "Mumbai", "Mumbai", "Pune"],
        "Gender": ["M", "F", "M", "F", "M", "F"],
        "Income": ["Low", "High", "Low", "High", "Low", "High"],
        "FoodExpectation": [
            "Price increase", "No change", "Price increase",
            "Price increase", "Decline", "No change",
        ],
        "Rate1Yr": [16, 8, 18, 20, 5, 10],
    })


def test_rank_groups_by_value_picks_top(inflation_df):
    result = dispatch_tool(inflation_df, "rank_groups_by_value", {
        "group_col": "City", "target_col": "FoodExpectation", "target_value": "Price increase",
    })
    assert result.tool_name == "rank_groups_by_value"
    # Mumbai: 2/3 = 66.7%, Delhi: 1/2 = 50%, Pune: 0/1 = 0% → Mumbai is top
    assert "Mumbai" in result.summary
    assert isinstance(result.table, list)
    assert result.table[0]["City"] == "Mumbai"
    assert result.table[0]["pct"] == pytest.approx(66.7, abs=0.1)


def test_filter_profile_subsets_and_profiles(inflation_df):
    result = dispatch_tool(inflation_df, "filter_profile", {
        "filter_col": "Rate1Yr", "filter_value": "16", "operator": "gte",
    })
    assert result.tool_name == "filter_profile"
    # Rate1Yr >= 16 → 3 of 6 rows
    assert "3 of 6" in result.summary
    cols = {row["column"] for row in result.table}
    assert "City" in cols and "Gender" in cols


def test_list_filtered_values_returns_raw_rows(inflation_df):
    result = dispatch_tool(inflation_df, "list_filtered_values", {
        "filter_col": "Rate1Yr", "filter_value": "16", "operator": "gte",
        "value_cols": ["City", "FoodExpectation"],
    })
    assert result.tool_name == "list_filtered_values"
    assert len(result.table) == 3
    assert set(result.table[0].keys()) == {"City", "FoodExpectation"}


def test_pivot_table_two_dimensions(inflation_df):
    result = dispatch_tool(inflation_df, "pivot_table", {
        "index_col": "City", "column_col": "Income", "value_col": "Rate1Yr",
    })
    assert result.tool_name == "pivot_table"
    assert isinstance(result.table, list) and result.table
    assert isinstance(result.png_bytes, bytes)


def test_pivot_table_counts_when_no_numeric_value(inflation_df):
    result = dispatch_tool(inflation_df, "pivot_table", {
        "index_col": "City", "column_col": "FoodExpectation",
    })
    assert result.png_bytes and result.table


def test_tool_names_list():
    assert "segment_stats" in TOOL_NAMES
    assert "distribution" in TOOL_NAMES
    assert "pie_chart" in TOOL_NAMES
    assert "crosstab" in TOOL_NAMES
    assert "anomalies" in TOOL_NAMES
    assert "threshold_count" in TOOL_NAMES


def test_pie_chart_returns_shares(survey_df):
    result = dispatch_tool(survey_df, "pie_chart", {"column": "Department"})
    assert isinstance(result, ToolResult)
    assert result.tool_name == "pie_chart"
    assert isinstance(result.png_bytes, bytes) and len(result.png_bytes) > 0
    assert isinstance(result.table, list) and result.table
    row = result.table[0]
    assert {"value", "count", "pct"} <= set(row)
    # percentages should sum to ~100
    assert abs(sum(r["pct"] for r in result.table) - 100) < 1.0


def test_pie_chart_groups_into_other(survey_df):
    result = dispatch_tool(survey_df, "pie_chart", {"column": "Department", "top_n": 1})
    labels = [r["value"] for r in result.table]
    assert "Other" in labels


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
