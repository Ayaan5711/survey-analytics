from __future__ import annotations
from dataclasses import dataclass, field
from app.data.profiler import DatasetProfile


@dataclass
class ReportData:
    filename: str
    row_count: int
    col_count: int
    column_summary: list[dict]
    insights: list[dict]
    pinned_charts: list[dict]   # [{title, png_bytes}]


def compose_report(
    profile: DatasetProfile,
    insights: list,
    pinned_charts: list[dict],
) -> ReportData:
    col_summary = [
        {
            "name": name,
            "dtype": cp.dtype,
            "missing_pct": cp.missing_pct,
            "n_unique": cp.n_unique,
            "mean": cp.mean,
            "top_values": dict(list(cp.top_values.items())[:3]),
        }
        for name, cp in profile.columns.items()
    ]

    insights_out = [
        {"rank": ins.rank, "title": ins.title, "summary": ins.summary}
        for ins in insights
    ]

    return ReportData(
        filename=profile.filename,
        row_count=profile.row_count,
        col_count=profile.col_count,
        column_summary=col_summary,
        insights=insights_out,
        pinned_charts=pinned_charts,
    )
