import io
import pytest
import pandas as pd
from app.data.loader import load_file, detect_column_type, get_excel_sheet_names


def test_load_csv_basic(sample_csv_bytes):
    df, schema = load_file(sample_csv_bytes, "responses.csv")
    assert len(df) == 5
    assert "Department" in schema
    assert "Satisfaction" in schema


def test_load_csv_cleans_headers(sample_csv_bytes):
    """Column names should be stripped and normalised."""
    raw = b"  Name  , Score \n Alice, 5\n"
    df, schema = load_file(raw, "messy.csv")
    assert "Name" in df.columns
    assert "Score" in df.columns


def test_detect_numeric_column(sample_csv_bytes):
    df, schema = load_file(sample_csv_bytes, "responses.csv")
    assert schema["Satisfaction"]["type"] == "numeric"
    assert schema["Salary"]["type"] == "numeric"


def test_detect_categorical_column(sample_csv_bytes):
    df, schema = load_file(sample_csv_bytes, "responses.csv")
    assert schema["Department"]["type"] == "categorical"


def test_detect_open_text_column(sample_csv_bytes):
    df, schema = load_file(sample_csv_bytes, "responses.csv")
    assert schema["Comments"]["type"] == "open_text"


def test_get_excel_sheet_names(sample_excel_bytes):
    sheets = get_excel_sheet_names(sample_excel_bytes)
    assert sheets == ["Q1"]


def test_load_excel_single_sheet(sample_excel_bytes):
    df, schema = load_file(sample_excel_bytes, "data.xlsx", sheet_name="Q1")
    assert "Department" in df.columns
    assert len(df) == 3
