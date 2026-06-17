from __future__ import annotations
import base64
from dataclasses import dataclass
from pathlib import Path
from app.data.profiler import DatasetProfile
from app.data.quality import DataQualityFlags
from app.dashboard.charts import deterministic_charts, ChartResult
from app.llm.client import llm
from app.llm.prompts import dashboard_narrative_prompt


@dataclass
class DashboardResponse:
    session_id: str
    filename: str
    row_count: int
    col_count: int
    narrative: str
    charts: list[dict]              # [{title, chart_type, png_b64, filename}]
    quality_flags: DataQualityFlags
    open_text_columns: list[str]
    column_summary: dict            # {col_name: {dtype, missing_pct, n_unique, mean, top_values}}


async def generate_dashboard(
    profile: DatasetProfile,
    quality: DataQualityFlags,
    session_dir: Path,
) -> DashboardResponse:
    charts: list[ChartResult] = deterministic_charts(profile, session_dir)

    prompt = dashboard_narrative_prompt(profile)
    narrative = await llm.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        use_fallback=False,
        max_tokens=300,
        temperature=0.4,
    )

    charts_out = [
        {
            "title": c.title,
            "chart_type": c.chart_type,
            "png_b64": base64.b64encode(c.png_bytes).decode(),
            "filename": c.filename,
        }
        for c in charts
    ]

    col_summary = {
        name: {
            "dtype": cp.dtype,
            "missing_pct": cp.missing_pct,
            "n_unique": cp.n_unique,
            "mean": cp.mean,
            "top_values": dict(list(cp.top_values.items())[:5]),
        }
        for name, cp in profile.columns.items()
    }

    return DashboardResponse(
        session_id=profile.session_id,
        filename=profile.filename,
        row_count=profile.row_count,
        col_count=profile.col_count,
        narrative=narrative,
        charts=charts_out,
        quality_flags=quality,
        open_text_columns=profile.open_text_columns,
        column_summary=col_summary,
    )
