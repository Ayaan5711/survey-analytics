from __future__ import annotations
import io
import re
from typing import Any
import numpy as np
import pandas as pd


_OPEN_TEXT_MIN_AVG_LEN = 20
_OPEN_TEXT_MIN_UNIQUE_RATIO = 0.5
_CATEGORICAL_MAX_UNIQUE_RATIO = 0.2
_CATEGORICAL_MAX_UNIQUE_ABS = 50


def detect_column_type(series: pd.Series) -> str:
    """Classify a column as numeric / categorical / open_text / datetime / unknown."""
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    non_null = series.dropna().astype(str)
    # Filter out empty strings before computing avg_len so that blank cells
    # don't artificially deflate the average and misclassify open-text columns.
    non_empty = non_null[non_null.str.strip() != ""]
    if non_empty.empty:
        return "unknown"
    avg_len = non_empty.str.len().mean()
    unique_ratio = non_empty.nunique() / len(non_empty)
    # Long text with many unique values → clearly open_text
    if avg_len >= _OPEN_TEXT_MIN_AVG_LEN and unique_ratio >= _OPEN_TEXT_MIN_UNIQUE_RATIO:
        return "open_text"
    # Use case-normalised values to detect label-like columns: collapsing
    # near-duplicates (e.g. "Engineering" / "engineering") reveals true cardinality.
    normalised = non_empty.str.lower().str.strip()
    norm_unique_ratio = normalised.nunique() / len(normalised)
    norm_n_unique = normalised.nunique()
    # A column is categorical when its normalised unique ratio is low (clear label set),
    # OR when it has actual duplicates after normalisation AND the absolute unique count
    # stays within a reasonable label vocabulary.
    has_duplicates = norm_unique_ratio < 1.0
    if norm_unique_ratio <= _CATEGORICAL_MAX_UNIQUE_RATIO:
        return "categorical"
    if has_duplicates and norm_n_unique <= _CATEGORICAL_MAX_UNIQUE_ABS:
        return "categorical"
    return "open_text"


def _clean_headers(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [re.sub(r"\s+", " ", c).strip() for c in df.columns]
    return df


def get_excel_sheet_names(content: bytes) -> list[str]:
    xl = pd.ExcelFile(io.BytesIO(content))
    return xl.sheet_names


def load_file(
    content: bytes,
    filename: str,
    sheet_name: str | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Parse CSV or Excel bytes -> (DataFrame, schema).

    schema shape: {col_name: {"type": str, "n_unique": int}}
    Raises ValueError for empty files or unsupported formats.
    """
    if not content:
        raise ValueError("File is empty")

    lower = filename.lower()
    if lower.endswith(".csv"):
        try:
            df = pd.read_csv(io.BytesIO(content))
        except Exception as exc:
            raise ValueError(f"Could not parse CSV: {exc}") from exc
    elif lower.endswith((".xlsx", ".xls")):
        try:
            df = pd.read_excel(io.BytesIO(content), sheet_name=sheet_name or 0)
        except Exception as exc:
            raise ValueError(f"Could not parse Excel: {exc}") from exc
    else:
        raise ValueError(f"Unsupported file type: {filename}. Expected .csv, .xlsx, or .xls")

    if df.empty:
        raise ValueError("File contains no data rows")

    df = _clean_headers(df)

    schema: dict[str, dict[str, Any]] = {
        col: {
            "type": detect_column_type(df[col]),
            "n_unique": int(df[col].nunique()),
        }
        for col in df.columns
    }
    return df, schema
