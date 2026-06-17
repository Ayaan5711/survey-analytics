from __future__ import annotations
import pytest
from app.data.loader import load_file
from app.data.profiler import build_profile
from app.compare.engine import compare_profiles, ComparisonDiff
import pandas as pd
import io


@pytest.fixture
def profile_a(tmp_path, sample_csv_bytes):
    df, schema = load_file(sample_csv_bytes, "q1.csv")
    return build_profile(df, schema, "cmp-a", "q1.csv", tmp_path)


@pytest.fixture
def profile_b(tmp_path):
    df = pd.DataFrame({
        "Name": ["X", "Y", "Z", "W", "V"],
        "Department": ["HR", "HR", "Engineering", "Sales", "Sales"],
        "Satisfaction": [3, 3, 4, 5, 2],
        "Comments": ["Meh", "OK", "Good", "Great", "Bad"],
        "Salary": [52000, 58000, 85000, 70000, 60000],
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    schema = {c: {"type": "categorical" if c in ("Name", "Department", "Comments") else "numeric", "n_unique": df[c].nunique()} for c in df.columns}
    return build_profile(df, schema, "cmp-b", "q2.csv", tmp_path)


def test_compare_profiles_returns_diff(profile_a, profile_b):
    diff = compare_profiles(profile_a, profile_b)
    assert isinstance(diff, ComparisonDiff)


def test_diff_has_row_count_change(profile_a, profile_b):
    diff = compare_profiles(profile_a, profile_b)
    assert diff.base_row_count == profile_a.row_count
    assert diff.compare_row_count == profile_b.row_count


def test_diff_numeric_delta(profile_a, profile_b):
    diff = compare_profiles(profile_a, profile_b)
    # Satisfaction column exists in both — should have a numeric delta
    sat = next((c for c in diff.column_diffs if c["column"] == "Satisfaction"), None)
    assert sat is not None
    assert "mean_delta" in sat


def test_diff_shared_columns(profile_a, profile_b):
    diff = compare_profiles(profile_a, profile_b)
    shared = {c["column"] for c in diff.column_diffs}
    assert "Satisfaction" in shared
    assert "Salary" in shared


def test_diff_serializable(profile_a, profile_b):
    import json
    diff = compare_profiles(profile_a, profile_b)
    import dataclasses
    serialized = json.dumps(dataclasses.asdict(diff))
    assert serialized
