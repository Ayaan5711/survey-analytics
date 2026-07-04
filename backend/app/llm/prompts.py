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
            # List the exact response values so the model passes them verbatim to tools.
            values = list(col.top_values.keys())[:8]
            vals_str = "; ".join(f'"{v}"' for v in values)
            lines.append(f"- {name} (categorical, {col.n_unique} values; exact values: {vals_str}), missing={col.missing_pct}%")
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
1. Use one of the provided tools to analyse the data — strongly prefer a tool
   that produces a chart so the answer is always backed by a visualization.
2. Default to visualizing: for almost every question, call at least one
   chart-producing tool (segment_stats, distribution, crosstab, anomalies,
   threshold_count) rather than answering from the profile alone. Only skip a
   chart when the user explicitly asks for a single number or a yes/no answer.
3. For open-ended questions, chain up to 4 tool calls before synthesising, and
   include the most illustrative chart.
4. Pick the tool that matches the question shape:
   - "pie chart of X" / "share/proportion of X" → pie_chart.
   - "bar chart of X" / "distribution of X" / "how is X spread" → distribution (one column); use segment_stats or crosstab if X is broken down "by Y".
   - "compare <expectations/ratings/responses> by <gender/age/education/income/category>" → compare_expectations_by_segment (pass the exact response option as target_value).
   - "compare X by Y" / "show X by Y" → crosstab (two categoricals) or segment_stats (numeric metric).
   - "show X by A and B" (two groupings) → pivot_table.
   - "which city/state/segment has the highest % of <response>" → rank_groups_by_value.
   - "profile of respondents who <condition>" → filter_profile.
   - "what exact values did respondents who <condition> enter" → list_filtered_values.
   - "how many selected <value/threshold>" → threshold_count (numeric) or distribution (categorical).
5. Use generate_code only for custom charts no built-in tool covers.
6. Always be specific — quote the column names, the exact response values, and the numbers from the tool results. Do not estimate or fabricate values."""


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
            "name": "pie_chart",
            "description": "Show the share/proportion breakdown of a single categorical column as a pie chart. Use this whenever the user asks for a pie chart or a 'share of' / 'proportion of' a category.",
            "parameters": {"type": "object", "properties": {
                "column": {"type": "string", "description": "Categorical column to break down"},
                "top_n": {"type": "integer", "description": "Max slices before grouping the rest into 'Other'", "default": 8},
            }, "required": ["column"]},
        }},
        {"type": "function", "function": {
            "name": "crosstab",
            "description": "Cross-tabulate two categorical columns. Produces a grouped bar chart.",
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
            "name": "compare_expectations_by_segment",
            "description": "Compare a repeated response scale (a block of questions that share the same answer options, e.g. a set of expectation/Likert questions) across a demographic segment such as gender, age, education, income, or category. Auto-detects the scale columns from the data. For each segment it reports respondent count, sample share, and the % selecting target_value for every scale question. Use for 'compare <expectations/ratings/responses> by <segment>'. Continuous numeric segments (e.g. age) are auto-binned into ranges.",
            "parameters": {"type": "object", "properties": {
                "segment_col": {"type": "string", "description": "Demographic column to compare across"},
                "target_value": {"type": "string", "description": "The response option to measure the share of (e.g. the 'expecting a price increase' option). Pass the exact value from the column's listed values. If omitted, the most common response is used."},
            }, "required": ["segment_col"]},
        }},
        {"type": "function", "function": {
            "name": "rank_groups_by_value",
            "description": "Rank groups by the percentage that selected a specific response value, excluding groups below min_n respondents. Use for 'which city/state/segment has the highest % of <response>?'. For 'price increase' questions use the exact value 'Price increase more than current rate'. Returns a ranked table and names the top group.",
            "parameters": {"type": "object", "properties": {
                "group_col": {"type": "string", "description": "Column to group/rank by (e.g. City, State, segment)"},
                "target_col": {"type": "string", "description": "Column holding the response of interest"},
                "target_value": {"type": "string", "description": "The response value to measure the share of, e.g. 'Price increase more than current rate', 'No change in prices', 'Decline in prices'"},
                "min_n": {"type": "integer", "description": "Minimum respondents for a group to be ranked", "default": 5},
                "match_mode": {"type": "string", "enum": ["eq", "contains"], "default": "eq"},
                "top_n": {"type": "integer", "default": 15},
            }, "required": ["group_col", "target_col", "target_value"]},
        }},
        {"type": "function", "function": {
            "name": "filter_profile",
            "description": "Subset respondents matching a condition, then profile that subset across all other columns (top values / means). Use for 'show the profile of respondents who <condition>'.",
            "parameters": {"type": "object", "properties": {
                "filter_col": {"type": "string"},
                "filter_value": {"type": "string", "description": "Value to match (numeric or categorical)"},
                "operator": {"type": "string", "enum": ["eq", "ne", "gt", "lt", "gte", "lte", "contains"], "default": "eq"},
            }, "required": ["filter_col", "filter_value"]},
        }},
        {"type": "function", "function": {
            "name": "list_filtered_values",
            "description": "List the raw values entered in specific columns for respondents matching a condition. Use for 'for respondents who selected X, what exact values were entered in columns A, B, C?'.",
            "parameters": {"type": "object", "properties": {
                "filter_col": {"type": "string"},
                "filter_value": {"type": "string"},
                "value_cols": {"type": "array", "items": {"type": "string"}, "description": "Columns whose raw values to list"},
                "operator": {"type": "string", "enum": ["eq", "ne", "gt", "lt", "gte", "lte", "contains"], "default": "eq"},
                "max_rows": {"type": "integer", "default": 50},
            }, "required": ["filter_col", "filter_value", "value_cols"]},
        }},
        {"type": "function", "function": {
            "name": "pivot_table",
            "description": "Two-dimensional breakdown of one column across two grouping columns (index × column). Means if a numeric value_col is given, otherwise counts. Use for 'show X by A and B'.",
            "parameters": {"type": "object", "properties": {
                "index_col": {"type": "string"},
                "column_col": {"type": "string"},
                "value_col": {"type": "string", "description": "Optional numeric column to average; omit for counts"},
            }, "required": ["index_col", "column_col"]},
        }},
        {"type": "function", "function": {
            "name": "open_text_themes",
            "description": "Extract recurring themes and the sentiment split from a free-text / open-ended column (e.g. comments, remarks, feedback). Use for 'what are people saying about…', 'summarise the comments', 'themes/sentiment in <column>'. Returns a themes bar + sentiment pie.",
            "parameters": {"type": "object", "properties": {
                "column": {"type": "string", "description": "The open-text column to analyse"},
            }, "required": ["column"]},
        }},
        {"type": "function", "function": {
            "name": "generate_code",
            "description": "Generate and execute custom Python pandas/matplotlib code for analyses not covered by other tools. Use only when no other tool fits.",
            "parameters": {"type": "object", "properties": {
                "code": {"type": "string", "description": "Python code with access to df (DataFrame), pd, np, plt, io. Must set result_png (bytes), result_plotly (JSON str), or result_summary (str)."},
            }, "required": ["code"]},
        }},
    ]


def code_gen_prompt(user_message: str, profile: "DatasetProfile", error: str | None = None) -> str:
    """Prompt for the primary model to (re)write a chart's matplotlib code.

    The sandbox exposes df, pd, np, plt, io and captures result_png / result_summary.
    """
    base = f"""You write Python (pandas + matplotlib) for a sandboxed analytics tool.

Dataset: "{profile.filename}" — {profile.row_count} rows.
Columns:
{_profile_summary(profile)}

User request: "{user_message}"

Write code that builds the requested chart. Hard requirements:
- A DataFrame `df` and the modules `pd`, `np`, `plt`, `io` are already available. Do NOT import or read any files.
- Render exactly one matplotlib figure.
- You MUST save it to PNG bytes and assign them to `result_png`, e.g.:
      buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=100, bbox_inches="tight"); result_png = buf.getvalue()
- NEVER call plt.show(). Set `result_summary` to a one-sentence factual description.
- Use the color "#2563eb" for the primary series to match the UI.

Return ONLY the Python code, no markdown fences, no explanation."""
    if error:
        base += f"\n\nThe previous attempt failed with:\n{error}\nFix the code and return the full corrected script."
    return base


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
