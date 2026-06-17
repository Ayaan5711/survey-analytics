from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import pandas as pd
from rapidfuzz import fuzz


_FUZZY_THRESHOLD = 85
_MOSTLY_EMPTY_PCT = 90.0


@dataclass
class DataQualityFlags:
    duplicate_rows: int = 0
    fuzzy_category_issues: list[dict] = field(default_factory=list)
    mostly_empty_columns: list[str] = field(default_factory=list)
    constant_columns: list[str] = field(default_factory=list)


def _find_fuzzy_pairs(values: list[str]) -> list[tuple[str, str]]:
    """Return pairs that look like the same thing with different casing/spacing."""
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[int, int]] = set()
    for i, a in enumerate(values):
        for j, b in enumerate(values):
            if j <= i or (i, j) in seen:
                continue
            if a.lower() == b.lower():
                pairs.append((a, b))
                seen.add((i, j))
            elif fuzz.ratio(a.lower(), b.lower()) >= _FUZZY_THRESHOLD:
                pairs.append((a, b))
                seen.add((i, j))
    return pairs


def check_quality(df: pd.DataFrame, schema: dict[str, dict[str, Any]]) -> DataQualityFlags:
    flags = DataQualityFlags()
    flags.duplicate_rows = int(df.duplicated().sum())

    total = len(df)
    for col, meta in schema.items():
        if col not in df.columns:
            continue
        series = df[col]
        missing_pct = series.isna().sum() / total * 100 if total else 0
        if missing_pct >= _MOSTLY_EMPTY_PCT:
            flags.mostly_empty_columns.append(col)
        if series.nunique() == 1:
            flags.constant_columns.append(col)
        if meta["type"] == "categorical":
            unique_vals = [str(v) for v in series.dropna().unique().tolist()]
            pairs = _find_fuzzy_pairs(unique_vals)
            if pairs:
                flags.fuzzy_category_issues.append({
                    "column": col,
                    "pairs": [{"a": a, "b": b} for a, b in pairs],
                    "suggestion": f"Found {len(pairs)} near-duplicate value(s) — consider standardising."
                })

    return flags
