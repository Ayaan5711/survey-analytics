from __future__ import annotations
from dataclasses import dataclass, field
from app.data.profiler import DatasetProfile


@dataclass
class ComparisonDiff:
    base_session_id: str
    compare_session_id: str
    base_filename: str
    compare_filename: str
    base_row_count: int
    compare_row_count: int
    row_count_delta: int
    shared_columns: list[str]
    only_in_base: list[str]
    only_in_compare: list[str]
    column_diffs: list[dict] = field(default_factory=list)


def compare_profiles(base: DatasetProfile, compare: DatasetProfile) -> ComparisonDiff:
    """Compare two DatasetProfiles and return a ComparisonDiff.

    Args:
        base: The baseline DatasetProfile
        compare: The comparison DatasetProfile

    Returns:
        ComparisonDiff with row/column/numeric/categorical deltas
    """
    base_cols = set(base.columns)
    compare_cols = set(compare.columns)
    shared = sorted(base_cols & compare_cols)
    only_base = sorted(base_cols - compare_cols)
    only_compare = sorted(compare_cols - base_cols)

    col_diffs: list[dict] = []
    for col in shared:
        bc = base.columns[col]
        cc = compare.columns[col]
        entry: dict = {
            "column": col,
            "dtype": bc.dtype,
            "base_missing_pct": bc.missing_pct,
            "compare_missing_pct": cc.missing_pct,
            "missing_pct_delta": round(cc.missing_pct - bc.missing_pct, 2),
            "base_n_unique": bc.n_unique,
            "compare_n_unique": cc.n_unique,
        }
        if bc.dtype == "numeric" and bc.mean is not None and cc.mean is not None:
            entry["base_mean"] = bc.mean
            entry["compare_mean"] = cc.mean
            entry["mean_delta"] = round(cc.mean - bc.mean, 4)
            entry["base_std"] = bc.std
            entry["compare_std"] = cc.std
        if bc.dtype == "categorical":
            entry["base_top_values"] = dict(list(bc.top_values.items())[:5])
            entry["compare_top_values"] = dict(list(cc.top_values.items())[:5])
        col_diffs.append(entry)

    return ComparisonDiff(
        base_session_id=base.session_id,
        compare_session_id=compare.session_id,
        base_filename=base.filename,
        compare_filename=compare.filename,
        base_row_count=base.row_count,
        compare_row_count=compare.row_count,
        row_count_delta=compare.row_count - base.row_count,
        shared_columns=shared,
        only_in_base=only_base,
        only_in_compare=only_compare,
        column_diffs=col_diffs,
    )
