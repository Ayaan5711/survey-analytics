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


import json as _json


def _profile_summary(profile: "DatasetProfile") -> str:
    lines = []
    for name, col in profile.columns.items():
        if col.dtype == "numeric":
            lines.append(f"- {name} (numeric): mean={col.mean}, min={col.min}, max={col.max}, missing={col.missing_pct}%")
        elif col.dtype == "categorical":
            top = list(col.top_values.items())[:3]
            lines.append(f"- {name} (categorical): top={top}, n_unique={col.n_unique}, missing={col.missing_pct}%")
        else:
            lines.append(f"- {name} ({col.dtype}): missing={col.missing_pct}%")
    return "\n".join(lines)


def chat_system_prompt(profile: "DatasetProfile") -> str:
    return f"""You are a data analyst assistant for a survey analytics tool.

Dataset: "{profile.filename}" — {profile.row_count} rows, {profile.col_count} columns.

Column statistics:
{_profile_summary(profile)}

Available columns: {", ".join(profile.columns.keys())}

When the user asks a question:
1. Use one of the provided tools to analyse the data.
2. For open-ended questions, chain up to 4 tool calls before synthesising.
3. Use generate_code only for custom/novel analyses no tool covers.
4. Always be specific — mention column names and numbers in your answers.
5. Never fabricate data values."""


def tool_definitions() -> list[dict]:
    return [
        {"type": "function", "function": {
            "name": "segment_stats",
            "description": "Compute count/mean/median/std for a numeric column broken down by a categorical column. Produces a bar chart.",
            "parameters": {"type": "object", "properties": {
                "group_col": {"type": "string", "description": "Categorical column to group by"},
                "metric_col": {"type": "string", "description": "Numeric column to aggregate"},
            }, "required": ["group_col", "metric_col"]},
        }},
        {"type": "function", "function": {
            "name": "distribution",
            "description": "Show distribution of a single column. Bar chart for categorical, histogram for numeric.",
            "parameters": {"type": "object", "properties": {
                "column": {"type": "string"},
            }, "required": ["column"]},
        }},
        {"type": "function", "function": {
            "name": "crosstab",
            "description": "Cross-tabulate two categorical columns. Produces a heatmap.",
            "parameters": {"type": "object", "properties": {
                "row_col": {"type": "string"},
                "col_col": {"type": "string"},
                "normalize": {"type": "boolean", "default": False},
            }, "required": ["row_col", "col_col"]},
        }},
        {"type": "function", "function": {
            "name": "anomalies",
            "description": "Detect outliers in a numeric column using IQR. Produces a box plot.",
            "parameters": {"type": "object", "properties": {
                "column": {"type": "string"},
            }, "required": ["column"]},
        }},
        {"type": "function", "function": {
            "name": "threshold_count",
            "description": "Count rows where a numeric column meets a threshold condition.",
            "parameters": {"type": "object", "properties": {
                "column": {"type": "string"},
                "threshold": {"type": "number"},
                "operator": {"type": "string", "enum": ["gt", "lt", "gte", "lte", "eq"]},
            }, "required": ["column", "threshold", "operator"]},
        }},
        {"type": "function", "function": {
            "name": "generate_code",
            "description": "Generate and execute custom Python pandas/matplotlib code for analyses not covered by other tools. Use only when no other tool fits.",
            "parameters": {"type": "object", "properties": {
                "code": {"type": "string", "description": "Python code with access to df (DataFrame), pd, np, plt, io. Must set result_png (bytes), result_plotly (JSON str), or result_summary (str)."},
            }, "required": ["code"]},
        }},
    ]


def synthesis_prompt(user_message: str, tool_summaries: list[str]) -> str:
    summaries = "\n".join(f"- {s}" for s in tool_summaries)
    return f"""Based on the analysis results below, write a concise answer to the user's question.

User asked: "{user_message}"

Analysis results:
{summaries}

Respond with JSON:
{{
  "narrative": "<2-4 sentence plain English answer, specific with numbers>",
  "follow_ups": ["<question 1>", "<question 2>", "<question 3>"]
}}

Do not add keys beyond narrative and follow_ups."""
