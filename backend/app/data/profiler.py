from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd


@dataclass
class ColumnProfile:
    dtype: str
    missing_pct: float
    n_unique: int
    # numeric only
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    # categorical only
    top_values: dict[str, int] = field(default_factory=dict)
    # all
    sample_values: list[Any] = field(default_factory=list)


@dataclass
class DatasetProfile:
    session_id: str
    filename: str
    row_count: int
    col_count: int
    columns: dict[str, ColumnProfile]
    sample_rows: list[dict]
    open_text_columns: list[str]
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


def _serialize(obj: Any) -> Any:
    """Recursively convert NaN to None for JSON safety, handle dicts and lists."""
    if isinstance(obj, float) and np.isnan(obj):
        return None
    elif isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_serialize(item) for item in obj]
    else:
        return obj


def _profile_column(series: pd.Series, dtype: str) -> ColumnProfile:
    """Profile a single column based on its dtype."""
    missing_count = series.isna().sum()
    missing_pct = (missing_count / len(series)) * 100 if len(series) > 0 else 0
    n_unique = int(series.nunique())
    non_null = series.dropna()

    # Get sample values: 5 non-null values
    sample_values = non_null.head(5).tolist()

    if dtype == "numeric":
        # Numeric stats
        min_val = float(non_null.min()) if len(non_null) > 0 else None
        max_val = float(non_null.max()) if len(non_null) > 0 else None
        mean_val = float(non_null.mean()) if len(non_null) > 0 else None
        median_val = float(non_null.median()) if len(non_null) > 0 else None
        std_val = float(non_null.std()) if len(non_null) > 0 else None

        # Round mean and std to 4 decimal places
        if mean_val is not None and not np.isnan(mean_val):
            mean_val = round(mean_val, 4)
        else:
            mean_val = None

        if std_val is not None and not np.isnan(std_val):
            std_val = round(std_val, 4)
        else:
            std_val = None

        return ColumnProfile(
            dtype=dtype,
            missing_pct=missing_pct,
            n_unique=n_unique,
            min=min_val,
            max=max_val,
            mean=mean_val,
            median=median_val,
            std=std_val,
            sample_values=sample_values,
        )

    elif dtype == "categorical":
        # Categorical: top 10 values as {str: int}
        value_counts = non_null.value_counts().head(10)
        top_values = {str(k): int(v) for k, v in value_counts.items()}

        return ColumnProfile(
            dtype=dtype,
            missing_pct=missing_pct,
            n_unique=n_unique,
            top_values=top_values,
            sample_values=sample_values,
        )

    else:
        # open_text or other
        return ColumnProfile(
            dtype=dtype,
            missing_pct=missing_pct,
            n_unique=n_unique,
            sample_values=sample_values,
        )


def build_profile(
    df: pd.DataFrame,
    schema: dict[str, dict[str, Any]],
    session_id: str,
    filename: str,
    data_dir: Path,
) -> DatasetProfile:
    """Build a DatasetProfile from a DataFrame and schema.

    Saves the data as parquet and profile as JSON in data_dir/session_id/.
    """
    data_dir = Path(data_dir)
    session_dir = data_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # Save DataFrame as parquet
    parquet_path = session_dir / "data.parquet"
    df.to_parquet(parquet_path, index=False)

    # Profile each column
    columns: dict[str, ColumnProfile] = {}
    open_text_columns: list[str] = []

    for col_name in df.columns:
        dtype = schema[col_name]["type"]
        columns[col_name] = _profile_column(df[col_name], dtype)
        if dtype == "open_text":
            open_text_columns.append(col_name)

    # Get sample rows (first 20, replace NaN with None)
    sample_rows = (
        df.head(20)
        .replace({np.nan: None})
        .to_dict(orient="records")
    )

    # Build profile
    profile = DatasetProfile(
        session_id=session_id,
        filename=filename,
        row_count=len(df),
        col_count=len(df.columns),
        columns=columns,
        sample_rows=sample_rows,
        open_text_columns=open_text_columns,
    )

    # Save profile as JSON
    profile_json_path = session_dir / "profile.json"
    profile_dict = asdict(profile)
    # asdict() already converts nested dataclasses, so just serialize
    serialized = _serialize(profile_dict)

    with open(profile_json_path, "w") as f:
        json.dump(serialized, f, indent=2)

    return profile


def load_profile(path: Path) -> DatasetProfile:
    """Load a DatasetProfile from a JSON file."""
    with open(path, "r") as f:
        data = json.load(f)

    # Reconstruct ColumnProfile objects
    columns: dict[str, ColumnProfile] = {}
    for col_name, col_dict in data["columns"].items():
        columns[col_name] = ColumnProfile(**col_dict)

    data["columns"] = columns
    return DatasetProfile(**data)
