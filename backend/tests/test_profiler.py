import json
import pytest
from app.data.loader import load_file
from app.data.profiler import build_profile, load_profile, DatasetProfile


def test_build_profile_shape(sample_csv_bytes, tmp_path):
    df, schema = load_file(sample_csv_bytes, "responses.csv")
    profile = build_profile(
        df=df, schema=schema, session_id="sess-1",
        filename="responses.csv", data_dir=tmp_path
    )
    assert isinstance(profile, DatasetProfile)
    assert profile.row_count == 5
    assert profile.col_count == 5
    assert "Satisfaction" in profile.columns
    assert "Department" in profile.columns


def test_numeric_column_has_stats(sample_csv_bytes, tmp_path):
    df, schema = load_file(sample_csv_bytes, "responses.csv")
    profile = build_profile(df, schema, "sess-1", "responses.csv", tmp_path)
    col = profile.columns["Satisfaction"]
    assert col.dtype == "numeric"
    assert col.mean is not None
    assert col.min is not None


def test_categorical_column_has_top_values(sample_csv_bytes, tmp_path):
    df, schema = load_file(sample_csv_bytes, "responses.csv")
    profile = build_profile(df, schema, "sess-1", "responses.csv", tmp_path)
    col = profile.columns["Department"]
    assert col.dtype == "categorical"
    assert isinstance(col.top_values, dict)
    assert len(col.top_values) > 0


def test_profile_persisted_as_parquet_and_json(sample_csv_bytes, tmp_path):
    df, schema = load_file(sample_csv_bytes, "responses.csv")
    profile = build_profile(df, schema, "sess-1", "responses.csv", tmp_path)
    assert (tmp_path / "sess-1" / "data.parquet").exists()
    assert (tmp_path / "sess-1" / "profile.json").exists()


def test_load_profile_round_trip(sample_csv_bytes, tmp_path):
    df, schema = load_file(sample_csv_bytes, "responses.csv")
    original = build_profile(df, schema, "sess-1", "responses.csv", tmp_path)
    loaded = load_profile(tmp_path / "sess-1" / "profile.json")
    assert loaded.row_count == original.row_count
    assert set(loaded.columns.keys()) == set(original.columns.keys())


def test_sample_rows_count(sample_csv_bytes, tmp_path):
    df, schema = load_file(sample_csv_bytes, "responses.csv")
    profile = build_profile(df, schema, "sess-1", "responses.csv", tmp_path)
    assert len(profile.sample_rows) <= 20
