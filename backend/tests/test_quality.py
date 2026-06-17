import io
import pytest
import pandas as pd
from app.data.quality import check_quality, DataQualityFlags


def _df(data: dict) -> pd.DataFrame:
    return pd.DataFrame(data)


def test_detects_duplicate_rows():
    df = _df({"A": [1, 1, 2], "B": ["x", "x", "y"]})
    flags = check_quality(df, {"A": {"type": "numeric"}, "B": {"type": "categorical"}})
    assert flags.duplicate_rows == 1


def test_no_duplicates():
    df = _df({"A": [1, 2, 3]})
    flags = check_quality(df, {"A": {"type": "numeric"}})
    assert flags.duplicate_rows == 0


def test_detects_fuzzy_category_issues():
    df = _df({"Dept": ["HR", "hr", "HR", "Engineering", "engineering"]})
    flags = check_quality(df, {"Dept": {"type": "categorical"}})
    issue_cols = [i["column"] for i in flags.fuzzy_category_issues]
    assert "Dept" in issue_cols


def test_detects_mostly_empty_columns():
    df = _df({"A": [1, None, None, None, None, None, None, None, None, None, None]})
    flags = check_quality(df, {"A": {"type": "numeric"}})
    assert "A" in flags.mostly_empty_columns


def test_detects_constant_columns():
    df = _df({"A": [1, 1, 1, 1], "B": [1, 2, 3, 4]})
    flags = check_quality(df, {"A": {"type": "numeric"}, "B": {"type": "numeric"}})
    assert "A" in flags.constant_columns
    assert "B" not in flags.constant_columns
