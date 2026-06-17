from __future__ import annotations
from app.data.profiler import DatasetProfile


def dashboard_narrative_prompt(profile: DatasetProfile) -> str:
    col_summary = []
    for name, col in profile.columns.items():
        if col.dtype == "numeric":
            col_summary.append(
                f"- {name} (numeric): mean={col.mean}, min={col.min}, max={col.max}, "
                f"missing={col.missing_pct}%"
            )
        elif col.dtype == "categorical":
            top = list(col.top_values.items())[:3]
            col_summary.append(
                f"- {name} (categorical): top values = {top}, n_unique={col.n_unique}, "
                f"missing={col.missing_pct}%"
            )
        else:
            col_summary.append(f"- {name} ({col.dtype}): missing={col.missing_pct}%")

    return f"""You are a data analyst. Below is a summary of a survey dataset called "{profile.filename}".

Dataset: {profile.row_count} rows, {profile.col_count} columns.

Column statistics:
{chr(10).join(col_summary)}

Write a short (3-5 sentence) plain-English summary highlighting the most notable patterns,
potential issues, or interesting findings. Be specific — mention column names and numbers.
Do not use bullet points. Do not suggest next steps."""
