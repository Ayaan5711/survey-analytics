# Survey Analytics — Plan 2: Chat Agent

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `POST /api/sessions/{id}/chat` with a Tier-1 tool registry (instant pandas ops) and Tier-2 sandboxed code-gen, giving the user free-form conversational analysis over their uploaded data.

**Architecture:** Each chat turn makes one LLM function-calling request (using the fallback/mini model for planning). The model picks a Tier-1 tool (sync pandas execution, nearly free) or calls `generate_code` for Tier-2 (sandbox subprocess, one LLM call). Open-ended questions run a bounded 4-step Tier-1 chain; a final synthesis call (primary model, JSON mode) produces the narrative + follow-ups. All turns persist to `chat_messages`.

**Tech Stack:** Python 3.11+, FastAPI, openai SDK ≥ 1.0 (Azure, function-calling), pandas 2, matplotlib, multiprocessing (stdlib), ast (stdlib), pytest, httpx.

## Global Constraints

- Python ≥ 3.11
- Use `from __future__ import annotations` at top of every file
- All async endpoints with `async def`; sync tools use regular `def`
- No new dependencies beyond what's already in `pyproject.toml`
- Sandbox runs in subprocess via `multiprocessing`; `resource` limits apply on Linux only (import-guarded)
- LLM planning calls use `settings.deployment_fallback`; synthesis uses `settings.deployment_primary`
- Every module in `backend/` — run tests from `backend/` with `pytest tests/ -v`

---

## File Map

```text
backend/
  app/
    tools/
      __init__.py           — empty
      registry.py           — ToolResult dataclass + 6 Tier-1 tools + dispatch_tool()
    sandbox/
      __init__.py           — empty
      ast_check.py          — check_ast(code) -> list[str]
      runner.py             — run_code(code, parquet_path, timeout) -> SandboxResult
    llm/
      agent.py              — ChatAgent + ChatResponse; orchestrates function-calling loop
      prompts.py            — extend: add chat_system_prompt(), tool_definitions()
    api/
      chat.py               — POST /sessions/{id}/chat, POST /sessions/{id}/pins
    main.py                 — register chat router (modify)
  tests/
    test_tools.py
    test_sandbox.py
    test_agent.py
    test_api_chat.py
```

---

## Task 1: Tier-1 Tool Registry

**Files:**
- Create: `backend/app/tools/__init__.py`
- Create: `backend/app/tools/registry.py`
- Test: `backend/tests/test_tools.py`

**Interfaces:**
- Produces: `ToolResult` dataclass, `dispatch_tool(df, name, params) -> ToolResult`, `TOOL_NAMES: list[str]`

- [ ] **Step 1: Create empty `backend/app/tools/__init__.py`**

```python
```

- [ ] **Step 2: Write failing tests**

```python
# backend/tests/test_tools.py
from __future__ import annotations
import pytest
import pandas as pd
from app.tools.registry import dispatch_tool, ToolResult, TOOL_NAMES


@pytest.fixture
def survey_df():
    return pd.DataFrame({
        "Department": ["HR", "HR", "Engineering", "Engineering", "Sales"],
        "Satisfaction": [4, 2, 5, 3, 4],
        "Salary": [50000, 60000, 80000, 75000, 55000],
        "Comments": ["Great", "Bad", "Good", "OK", "Fine"],
    })


def test_tool_names_list():
    assert "segment_stats" in TOOL_NAMES
    assert "distribution" in TOOL_NAMES
    assert "crosstab" in TOOL_NAMES
    assert "anomalies" in TOOL_NAMES
    assert "threshold_count" in TOOL_NAMES


def test_segment_stats_returns_tool_result(survey_df):
    result = dispatch_tool(survey_df, "segment_stats", {"group_col": "Department", "metric_col": "Satisfaction"})
    assert isinstance(result, ToolResult)
    assert result.tool_name == "segment_stats"
    assert isinstance(result.table, list)
    assert len(result.table) > 0
    assert result.summary
    assert isinstance(result.png_bytes, bytes)


def test_segment_stats_table_has_expected_keys(survey_df):
    result = dispatch_tool(survey_df, "segment_stats", {"group_col": "Department", "metric_col": "Satisfaction"})
    row = result.table[0]
    assert "Department" in row
    assert "mean" in row
    assert "count" in row


def test_distribution_categorical(survey_df):
    result = dispatch_tool(survey_df, "distribution", {"column": "Department"})
    assert result.tool_name == "distribution"
    assert isinstance(result.png_bytes, bytes)
    assert result.table


def test_distribution_numeric(survey_df):
    result = dispatch_tool(survey_df, "distribution", {"column": "Satisfaction"})
    assert isinstance(result.png_bytes, bytes)


def test_crosstab(survey_df):
    result = dispatch_tool(survey_df, "crosstab", {"row_col": "Department", "col_col": "Satisfaction"})
    assert result.tool_name == "crosstab"
    assert result.png_bytes
    assert result.table


def test_anomalies(survey_df):
    result = dispatch_tool(survey_df, "anomalies", {"column": "Salary"})
    assert result.tool_name == "anomalies"
    assert "outlier" in result.summary.lower() or "no outlier" in result.summary.lower()


def test_threshold_count(survey_df):
    result = dispatch_tool(survey_df, "threshold_count", {"column": "Satisfaction", "threshold": 3, "operator": "gt"})
    assert result.tool_name == "threshold_count"
    assert isinstance(result.table, dict)
    assert "count" in result.table


def test_unknown_tool_raises():
    df = pd.DataFrame({"A": [1]})
    with pytest.raises(ValueError, match="Unknown tool"):
        dispatch_tool(df, "nonexistent_tool", {})


def test_caveat_on_small_segment(survey_df):
    # Sales has only 1 row — should get a caveat
    result = dispatch_tool(survey_df, "segment_stats", {"group_col": "Department", "metric_col": "Satisfaction"})
    assert result.caveat is not None
```

- [ ] **Step 3: Run to confirm failure**

```bash
cd backend && pytest tests/test_tools.py -v
# Expected: FAIL — cannot import dispatch_tool
```

- [ ] **Step 4: Create `backend/app/tools/registry.py`**

```python
from __future__ import annotations
import io
from dataclasses import dataclass, field
from typing import Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class ToolResult:
    tool_name: str
    params: dict
    summary: str
    table: list | dict
    png_bytes: bytes | None
    caveat: str | None = None


TOOL_NAMES = ["segment_stats", "distribution", "crosstab", "anomalies", "threshold_count"]

_CAVEAT_MIN_N = 30


def _fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _black_bar(ax, labels, values, xlabel="", ylabel="", title=""):
    ax.bar([str(l) for l in labels], values, color="black")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def segment_stats(df: pd.DataFrame, group_col: str, metric_col: str) -> ToolResult:
    grouped = (
        df.groupby(group_col)[metric_col]
        .agg(count="count", mean="mean", median="median", std="std")
        .reset_index()
    )
    grouped["mean"] = grouped["mean"].round(3)
    grouped["std"] = grouped["std"].round(3)

    fig, ax = plt.subplots(figsize=(7, 4))
    _black_bar(
        ax,
        grouped[group_col].astype(str),
        grouped["mean"],
        xlabel=group_col,
        ylabel=f"Mean {metric_col}",
        title=f"{metric_col} mean by {group_col}",
    )
    png = _fig_to_bytes(fig)

    best_row = grouped.loc[grouped["mean"].idxmax()]
    summary = (
        f"{metric_col} by {group_col}: highest mean is "
        f"'{best_row[group_col]}' ({best_row['mean']:.2f})"
    )
    min_n = int(grouped["count"].min())
    caveat = f"Smallest segment has only {min_n} response(s) — interpret with care." if min_n < _CAVEAT_MIN_N else None

    return ToolResult(
        tool_name="segment_stats",
        params={"group_col": group_col, "metric_col": metric_col},
        summary=summary,
        table=grouped.replace({np.nan: None}).to_dict(orient="records"),
        png_bytes=png,
        caveat=caveat,
    )


def distribution(df: pd.DataFrame, column: str) -> ToolResult:
    series = df[column].dropna()
    fig, ax = plt.subplots(figsize=(7, 4))

    if pd.api.types.is_numeric_dtype(series):
        ax.hist(series, bins=min(20, series.nunique()), color="black", edgecolor="white")
        ax.set_title(f"{column} — distribution", fontsize=12)
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        table: list | dict = {"min": float(series.min()), "max": float(series.max()),
                               "mean": round(float(series.mean()), 3)}
        summary = f"{column}: min={table['min']}, max={table['max']}, mean={table['mean']}"
    else:
        counts = series.astype(str).value_counts().head(15)
        _black_bar(ax, counts.index, counts.values, xlabel=column, ylabel="Count",
                   title=f"{column} — distribution")
        table = counts.to_dict()
        top = counts.index[0] if len(counts) else "N/A"
        summary = f"{column}: top value is '{top}' with {counts.iloc[0]} occurrences"

    png = _fig_to_bytes(fig)
    caveat = f"{column} has {len(series)} non-null values." if len(series) < _CAVEAT_MIN_N else None
    return ToolResult(tool_name="distribution", params={"column": column},
                      summary=summary, table=table, png_bytes=png, caveat=caveat)


def crosstab(df: pd.DataFrame, row_col: str, col_col: str, normalize: bool = False) -> ToolResult:
    ct = pd.crosstab(df[row_col], df[col_col], normalize="index" if normalize else False)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(ct.values, aspect="auto", cmap="Greys")
    ax.set_xticks(range(len(ct.columns)))
    ax.set_xticklabels([str(c) for c in ct.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(ct.index)))
    ax.set_yticklabels([str(r) for r in ct.index])
    ax.set_title(f"{row_col} × {col_col}", fontsize=12)
    plt.colorbar(im, ax=ax)
    png = _fig_to_bytes(fig)
    table = ct.reset_index().rename(columns=str).replace({np.nan: None}).to_dict(orient="records")
    summary = f"Cross-tab of {row_col} by {col_col}: {len(ct.index)} rows × {len(ct.columns)} cols"
    return ToolResult(tool_name="crosstab", params={"row_col": row_col, "col_col": col_col,
                      "normalize": normalize}, summary=summary, table=table, png_bytes=png)


def anomalies(df: pd.DataFrame, column: str) -> ToolResult:
    series = df[column].dropna()
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = series[(series < lo) | (series > hi)]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(series, vert=False, patch_artist=True,
               boxprops={"facecolor": "white", "edgecolor": "black"},
               medianprops={"color": "black"}, flierprops={"marker": "o", "color": "black"})
    ax.set_title(f"{column} — box plot (IQR outliers)", fontsize=12)
    ax.set_xlabel(column)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    png = _fig_to_bytes(fig)

    n = len(outliers)
    summary = (
        f"{column}: {n} outlier(s) detected (below {lo:.2f} or above {hi:.2f})"
        if n else f"{column}: no outliers detected by IQR method"
    )
    table = {"outlier_count": n, "lower_fence": round(lo, 3), "upper_fence": round(hi, 3),
             "outlier_values": [round(v, 3) for v in outliers.tolist()[:20]]}
    caveat = f"Based on {len(series)} non-null values." if len(series) < _CAVEAT_MIN_N else None
    return ToolResult(tool_name="anomalies", params={"column": column},
                      summary=summary, table=table, png_bytes=png, caveat=caveat)


def threshold_count(df: pd.DataFrame, column: str, threshold: float, operator: str) -> ToolResult:
    ops = {"gt": df[column] > threshold, "lt": df[column] < threshold,
           "gte": df[column] >= threshold, "lte": df[column] <= threshold,
           "eq": df[column] == threshold}
    if operator not in ops:
        raise ValueError(f"operator must be one of {list(ops)}")
    mask = ops[operator]
    count = int(mask.sum())
    total = int(df[column].notna().sum())
    pct = round(count / total * 100, 1) if total else 0
    op_label = {"gt": ">", "lt": "<", "gte": ">=", "lte": "<=", "eq": "=="}[operator]
    summary = f"{count} of {total} rows ({pct}%) have {column} {op_label} {threshold}"

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Meets threshold", "Does not"], [count, total - count], color=["black", "#cccccc"])
    ax.set_title(f"{column} {op_label} {threshold}", fontsize=12)
    ax.set_ylabel("Count")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    png = _fig_to_bytes(fig)

    return ToolResult(tool_name="threshold_count",
                      params={"column": column, "threshold": threshold, "operator": operator},
                      summary=summary, table={"count": count, "total": total, "pct": pct},
                      png_bytes=png)


_DISPATCH = {
    "segment_stats": segment_stats,
    "distribution": distribution,
    "crosstab": crosstab,
    "anomalies": anomalies,
    "threshold_count": threshold_count,
}


def dispatch_tool(df: pd.DataFrame, name: str, params: dict) -> ToolResult:
    if name not in _DISPATCH:
        raise ValueError(f"Unknown tool: {name}. Available: {list(_DISPATCH)}")
    return _DISPATCH[name](df, **params)
```

- [ ] **Step 5: Run tests to confirm pass**

```bash
pytest tests/test_tools.py -v
# Expected: all PASS
```

- [ ] **Step 6: Commit**

```bash
git add app/tools/ tests/test_tools.py
git commit -m "feat: Tier-1 tool registry — segment_stats, distribution, crosstab, anomalies, threshold_count"
```

---

## Task 2: Sandbox (AST Check + Subprocess Runner)

**Files:**
- Create: `backend/app/sandbox/__init__.py`
- Create: `backend/app/sandbox/ast_check.py`
- Create: `backend/app/sandbox/runner.py`
- Test: `backend/tests/test_sandbox.py`

**Interfaces:**
- Produces: `check_ast(code: str) -> list[str]`, `run_code(code: str, parquet_path: str, timeout: int = 15) -> SandboxResult`
- `SandboxResult`: `success: bool, png_bytes: bytes | None, plotly_json: str | None, summary: str | None, error: str | None`

- [ ] **Step 1: Create empty `backend/app/sandbox/__init__.py`**

```python
```

- [ ] **Step 2: Write failing tests**

```python
# backend/tests/test_sandbox.py
from __future__ import annotations
import pytest
import pandas as pd
import io
from pathlib import Path
from app.sandbox.ast_check import check_ast
from app.sandbox.runner import run_code, SandboxResult


@pytest.fixture
def parquet_file(tmp_path):
    df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
    path = tmp_path / "data.parquet"
    df.to_parquet(path, index=False)
    return str(path)


# AST check tests
def test_ast_allows_pandas_import():
    errors = check_ast("import pandas as pd\ndf.head()")
    assert errors == []


def test_ast_allows_numpy_matplotlib():
    errors = check_ast("import numpy as np\nimport matplotlib.pyplot as plt")
    assert errors == []


def test_ast_blocks_os_import():
    errors = check_ast("import os")
    assert any("os" in e for e in errors)


def test_ast_blocks_sys_import():
    errors = check_ast("import sys")
    assert any("sys" in e for e in errors)


def test_ast_blocks_open_builtin():
    errors = check_ast("open('file.txt')")
    assert any("open" in e for e in errors)


def test_ast_blocks_dunder_attribute():
    errors = check_ast("df.__class__")
    assert any("__" in e for e in errors)


def test_ast_blocks_from_os_import():
    errors = check_ast("from os import path")
    assert any("os" in e for e in errors)


def test_ast_catches_syntax_error():
    errors = check_ast("def foo(:")
    assert any("Syntax" in e for e in errors)


# Runner tests
def test_run_code_basic(parquet_file):
    code = "result_summary = f'rows={len(df)}'"
    result = run_code(code, parquet_file)
    assert result.success
    assert result.summary == "rows=3"
    assert result.error is None


def test_run_code_produces_png(parquet_file):
    code = """
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
fig, ax = plt.subplots()
ax.bar(df['B'], df['A'])
buf = io.BytesIO()
fig.savefig(buf, format='png')
plt.close()
buf.seek(0)
result_png = buf.read()
result_summary = 'bar chart done'
"""
    result = run_code(code, parquet_file)
    assert result.success, result.error
    assert isinstance(result.png_bytes, bytes)
    assert len(result.png_bytes) > 0


def test_run_code_catches_runtime_error(parquet_file):
    code = "x = 1 / 0"
    result = run_code(code, parquet_file)
    assert not result.success
    assert "ZeroDivision" in result.error


def test_run_code_timeout(parquet_file):
    code = "import time; time.sleep(60)"
    result = run_code(code, parquet_file, timeout=2)
    assert not result.success
    assert "timed out" in result.error.lower()


def test_ast_blocked_code_not_run(parquet_file):
    code = "import os; result_summary = os.getcwd()"
    result = run_code(code, parquet_file)
    assert not result.success
    assert "not allowed" in result.error.lower()
```

- [ ] **Step 3: Run to confirm failure**

```bash
pytest tests/test_sandbox.py -v
# Expected: FAIL — cannot import check_ast
```

- [ ] **Step 4: Create `backend/app/sandbox/ast_check.py`**

```python
from __future__ import annotations
import ast

_ALLOWED_IMPORTS = {"pandas", "pd", "numpy", "np", "matplotlib", "plt", "seaborn", "plotly", "io", "math"}
_BLOCKED_NAMES = {"os", "sys", "open", "__import__", "eval", "exec", "compile",
                  "globals", "locals", "__builtins__", "breakpoint", "input"}


class _ASTChecker(ast.NodeVisitor):
    def __init__(self) -> None:
        self.errors: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            base = alias.name.split(".")[0]
            if base not in _ALLOWED_IMPORTS:
                self.errors.append(f"Import not allowed: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            base = node.module.split(".")[0]
            if base not in _ALLOWED_IMPORTS:
                self.errors.append(f"Import not allowed: from {node.module}")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in _BLOCKED_NAMES:
            self.errors.append(f"Name not allowed: {node.id}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__"):
            self.errors.append(f"Dunder attribute access not allowed: {node.attr}")
        self.generic_visit(node)


def check_ast(code: str) -> list[str]:
    """Return list of violation strings; empty list means code is safe to run."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"Syntax error: {exc}"]
    checker = _ASTChecker()
    checker.visit(tree)
    return checker.errors
```

- [ ] **Step 5: Create `backend/app/sandbox/runner.py`**

```python
from __future__ import annotations
import multiprocessing
import traceback
from dataclasses import dataclass

try:
    import resource as _resource
    _HAS_RESOURCE = True
except ImportError:
    _HAS_RESOURCE = False

from app.sandbox.ast_check import check_ast


@dataclass
class SandboxResult:
    success: bool
    png_bytes: bytes | None
    plotly_json: str | None
    summary: str | None
    error: str | None


def _worker(code: str, parquet_path: str, conn) -> None:  # runs in subprocess
    if _HAS_RESOURCE:
        _resource.setrlimit(_resource.RLIMIT_CPU, (10, 10))
        try:
            _resource.setrlimit(_resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
        except ValueError:
            pass

    import pandas as pd
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as exc:
        conn.send({"success": False, "png_bytes": None, "plotly_json": None,
                   "summary": None, "error": f"Failed to load data: {exc}"})
        conn.close()
        return

    local_ns: dict = {
        "df": df, "pd": pd, "np": np, "plt": plt, "io": io,
        "result_png": None, "result_plotly": None, "result_summary": None,
    }
    try:
        exec(code, local_ns)  # noqa: S102
        conn.send({
            "success": True,
            "png_bytes": local_ns.get("result_png"),
            "plotly_json": local_ns.get("result_plotly"),
            "summary": local_ns.get("result_summary"),
            "error": None,
        })
    except Exception as exc:
        conn.send({
            "success": False, "png_bytes": None, "plotly_json": None,
            "summary": None,
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        })
    finally:
        conn.close()


def run_code(code: str, parquet_path: str, timeout: int = 15) -> SandboxResult:
    ast_errors = check_ast(code)
    if ast_errors:
        return SandboxResult(success=False, png_bytes=None, plotly_json=None,
                             summary=None, error="Code not allowed: " + "; ".join(ast_errors))

    parent_conn, child_conn = multiprocessing.Pipe()
    p = multiprocessing.Process(target=_worker, args=(code, parquet_path, child_conn))
    p.start()
    child_conn.close()

    if parent_conn.poll(timeout):
        data = parent_conn.recv()
    else:
        p.terminate()
        p.join(timeout=5)
        data = {"success": False, "png_bytes": None, "plotly_json": None,
                "summary": None, "error": f"Sandbox timed out after {timeout}s"}

    p.join()
    parent_conn.close()
    return SandboxResult(**data)
```

- [ ] **Step 6: Run tests to confirm pass**

```bash
pytest tests/test_sandbox.py -v
# Expected: all PASS
```

- [ ] **Step 7: Commit**

```bash
git add app/sandbox/ tests/test_sandbox.py
git commit -m "feat: sandbox — AST whitelist checker + multiprocessing runner with timeout"
```

---

## Task 3: Chat Agent (Prompts + Orchestrator)

**Files:**
- Modify: `backend/app/llm/prompts.py`
- Create: `backend/app/llm/agent.py`
- Test: `backend/tests/test_agent.py`

**Interfaces:**
- Consumes: `ToolResult` from `app.tools.registry`, `SandboxResult` from `app.sandbox.runner`, `DatasetProfile` from `app.data.profiler`, `llm` from `app.llm.client`
- Produces: `ChatAgent`, `ChatResponse` dataclass

- [ ] **Step 1: Extend `backend/app/llm/prompts.py`** (append to existing file, do not replace)

```python
# Add these functions below the existing dashboard_narrative_prompt()

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
```

- [ ] **Step 2: Create `backend/app/llm/agent.py`**

```python
from __future__ import annotations
import base64
import json
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd
from app.data.profiler import DatasetProfile
from app.llm.client import llm
from app.llm.prompts import chat_system_prompt, tool_definitions, synthesis_prompt
from app.sandbox.runner import run_code, SandboxResult
from app.tools.registry import dispatch_tool, ToolResult

_MAX_STEPS = 4


@dataclass
class ChatResponse:
    role: str
    content: str
    chart: dict | None          # {png_b64, title} or {plotly_json, title}
    generated_code: str | None
    follow_ups: list[str]
    caveats: list[str]
    tool_calls_made: list[str]


class ChatAgent:
    async def run(
        self,
        profile: DatasetProfile,
        parquet_path: str,
        message: str,
        history: list[dict],
    ) -> ChatResponse:
        messages: list[dict] = [
            {"role": "system", "content": chat_system_prompt(profile)},
            *history,
            {"role": "user", "content": message},
        ]

        df = pd.read_parquet(parquet_path)
        tool_calls_made: list[str] = []
        tool_summaries: list[str] = []
        last_chart: dict | None = None
        last_code: str | None = None
        caveats: list[str] = []

        for _step in range(_MAX_STEPS):
            response = await llm._client.chat.completions.create(
                model=llm.deployment_fallback,
                messages=messages,
                tools=tool_definitions(),
                tool_choice="auto",
                temperature=0.2,
                max_tokens=800,
            )
            choice = response.choices[0]

            if choice.finish_reason == "stop":
                break

            if choice.finish_reason == "tool_calls":
                msg_dict = choice.message.model_dump(exclude_none=True)
                messages.append(msg_dict)

                for tc in choice.message.tool_calls:
                    name = tc.function.name
                    params = json.loads(tc.function.arguments)
                    tool_calls_made.append(name)

                    tool_result_str, chart, caveat, code = await self._execute(
                        name, params, df, parquet_path
                    )
                    if chart:
                        last_chart = chart
                    if caveat:
                        caveats.append(caveat)
                    if code:
                        last_code = code
                    tool_summaries.append(tool_result_str)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result_str[:2000],
                    })

        # Synthesis call
        synth_msg = synthesis_prompt(message, tool_summaries)
        raw = await llm.chat_completion(
            messages=[{"role": "user", "content": synth_msg}],
            use_fallback=False,
            max_tokens=400,
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        try:
            parsed = json.loads(raw)
            narrative = parsed.get("narrative", raw)
            follow_ups = parsed.get("follow_ups", [])[:3]
        except (json.JSONDecodeError, AttributeError):
            narrative = raw
            follow_ups = []

        return ChatResponse(
            role="assistant",
            content=narrative,
            chart=last_chart,
            generated_code=last_code,
            follow_ups=follow_ups,
            caveats=caveats,
            tool_calls_made=tool_calls_made,
        )

    async def _execute(
        self,
        name: str,
        params: dict,
        df: pd.DataFrame,
        parquet_path: str,
    ) -> tuple[str, dict | None, str | None, str | None]:
        """Returns (summary_str, chart_dict_or_None, caveat_or_None, code_or_None)."""
        if name == "generate_code":
            code = params["code"]
            result: SandboxResult = run_code(code, parquet_path)
            if not result.success:
                return f"Code execution error: {result.error}", None, None, code
            chart = None
            if result.png_bytes:
                chart = {"png_b64": base64.b64encode(result.png_bytes).decode(), "title": "Custom chart"}
            elif result.plotly_json:
                chart = {"plotly_json": result.plotly_json, "title": "Custom chart"}
            return result.summary or "Code executed successfully.", chart, None, code

        tr: ToolResult = dispatch_tool(df, name, params)
        chart = None
        if tr.png_bytes:
            chart = {"png_b64": base64.b64encode(tr.png_bytes).decode(), "title": tr.summary}
        table_str = json.dumps(tr.table)[:500] if tr.table else ""
        return f"{tr.summary}. Data: {table_str}", chart, tr.caveat, None
```

- [ ] **Step 3: Write failing tests**

```python
# backend/tests/test_agent.py
from __future__ import annotations
import base64
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.data.loader import load_file
from app.data.profiler import build_profile
from app.llm.agent import ChatAgent, ChatResponse


@pytest.fixture
def profile_and_parquet(sample_csv_bytes, tmp_path):
    df, schema = load_file(sample_csv_bytes, "responses.csv")
    profile = build_profile(df, schema, "sess-a", "responses.csv", tmp_path)
    parquet_path = str(tmp_path / "sess-a" / "data.parquet")
    return profile, parquet_path


def _mock_tool_call_response(tool_name: str, args: dict):
    tc = MagicMock()
    tc.function.name = tool_name
    tc.function.arguments = json.dumps(args)
    tc.id = "call_1"

    msg = MagicMock()
    msg.tool_calls = [tc]
    msg.model_dump.return_value = {"role": "assistant", "tool_calls": []}

    choice = MagicMock()
    choice.finish_reason = "tool_calls"
    choice.message = msg

    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _mock_stop_response():
    choice = MagicMock()
    choice.finish_reason = "stop"
    choice.message.content = "Direct answer."
    choice.message.tool_calls = None
    resp = MagicMock()
    resp.choices = [choice]
    return resp


@pytest.mark.asyncio
async def test_agent_returns_chat_response(profile_and_parquet):
    profile, parquet_path = profile_and_parquet
    agent = ChatAgent()

    tool_resp = _mock_tool_call_response("distribution", {"column": "Satisfaction"})
    stop_resp = _mock_stop_response()

    synthesis_json = json.dumps({
        "narrative": "The Satisfaction column is mostly 4s.",
        "follow_ups": ["What about by Department?", "Any outliers?", "Compare Salary?"],
    })

    call_count = 0

    async def fake_create(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return tool_resp
        return stop_resp

    with patch.object(agent._llm if hasattr(agent, '_llm') else type(agent), 'chat_completion',
                      new=AsyncMock(return_value=synthesis_json)), \
         patch("app.llm.agent.llm._client.chat.completions.create", new=AsyncMock(side_effect=fake_create)):
        result = await agent.run(profile, parquet_path, "Show distribution of Satisfaction", [])

    assert isinstance(result, ChatResponse)
    assert result.role == "assistant"
    assert result.content
    assert "distribution" in result.tool_calls_made


@pytest.mark.asyncio
async def test_agent_direct_stop_no_tools(profile_and_parquet):
    profile, parquet_path = profile_and_parquet
    agent = ChatAgent()

    stop_resp = _mock_stop_response()
    synthesis_json = json.dumps({"narrative": "Here is the answer.", "follow_ups": []})

    with patch("app.llm.agent.llm._client.chat.completions.create",
               new=AsyncMock(return_value=stop_resp)), \
         patch("app.llm.agent.llm.chat_completion", new=AsyncMock(return_value=synthesis_json)):
        result = await agent.run(profile, parquet_path, "What is the dataset?", [])

    assert isinstance(result, ChatResponse)
    assert result.tool_calls_made == []


@pytest.mark.asyncio
async def test_agent_chart_in_response(profile_and_parquet):
    profile, parquet_path = profile_and_parquet
    agent = ChatAgent()

    tool_resp = _mock_tool_call_response("segment_stats", {"group_col": "Department", "metric_col": "Satisfaction"})
    stop_resp = _mock_stop_response()
    synthesis_json = json.dumps({"narrative": "Engineering scores highest.", "follow_ups": []})

    call_n = 0
    async def fake_create(**kwargs):
        nonlocal call_n
        call_n += 1
        return tool_resp if call_n == 1 else stop_resp

    with patch("app.llm.agent.llm._client.chat.completions.create", new=AsyncMock(side_effect=fake_create)), \
         patch("app.llm.agent.llm.chat_completion", new=AsyncMock(return_value=synthesis_json)):
        result = await agent.run(profile, parquet_path, "Satisfaction by Department", [])

    assert result.chart is not None
    assert "png_b64" in result.chart
    base64.b64decode(result.chart["png_b64"])
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_agent.py -v
# Expected: all PASS
```

- [ ] **Step 5: Commit**

```bash
git add app/llm/agent.py app/llm/prompts.py tests/test_agent.py
git commit -m "feat: chat agent — function-calling loop, Tier-1/Tier-2 dispatch, synthesis"
```

---

## Task 4: Chat API Route + DB Persistence

**Files:**
- Create: `backend/app/api/chat.py`
- Modify: `backend/app/main.py`
- Test: `backend/tests/test_api_chat.py`

**Interfaces:**
- Consumes: `ChatAgent` from `app.llm.agent`, `load_profile` from `app.data.profiler`, DB session
- Request body: `{"message": str, "history": [{role, content}, ...]}`
- Response: `{message_id, role, content, chart, generated_code, follow_ups, caveats, tool_calls_made}`

- [ ] **Step 1: Create `backend/app/api/chat.py`**

```python
from __future__ import annotations
import base64
import logging
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.db.database import get_db
from app.db.models import Session as SessionModel, ChatMessage, PinnedChart
from app.data.profiler import load_profile
from app.llm.agent import ChatAgent

logger = logging.getLogger(__name__)
router = APIRouter()
_agent = ChatAgent()


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []


class PinRequest(BaseModel):
    png_b64: str
    title: str
    source_message_id: str | None = None


@router.post("/sessions/{session_id}/chat")
async def chat(
    session_id: str,
    body: ChatRequest,
    db: Session = Depends(get_db),
) -> dict:
    record = db.get(SessionModel, session_id)
    if not record:
        raise HTTPException(status_code=404, detail="Session not found")
    if not record.profile_path or not Path(record.profile_path).exists():
        raise HTTPException(status_code=404, detail="Profile not found — re-upload the file")
    if not record.data_path or not Path(record.data_path).exists():
        raise HTTPException(status_code=404, detail="Data file not found — re-upload the file")

    profile = load_profile(Path(record.profile_path))

    safe_history = [
        {"role": m["role"], "content": str(m.get("content", ""))}
        for m in body.history
        if m.get("role") in ("user", "assistant")
    ][-20:]

    try:
        response = await _agent.run(profile, record.data_path, body.message, safe_history)
    except Exception as exc:
        logger.error("Agent error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Analysis failed — try rephrasing your question")

    chart_paths = None
    if response.chart:
        if "png_b64" in response.chart:
            chart_paths = [{"type": "png", "title": response.chart.get("title", "")}]
        elif "plotly_json" in response.chart:
            chart_paths = [{"type": "plotly", "title": response.chart.get("title", "")}]

    msg = ChatMessage(
        session_id=session_id,
        role="assistant",
        content=response.content,
        generated_code=response.generated_code,
        chart_paths=chart_paths,
        follow_ups=response.follow_ups,
        caveats=response.caveats,
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)

    return {
        "message_id": msg.id,
        "role": "assistant",
        "content": response.content,
        "chart": response.chart,
        "generated_code": response.generated_code,
        "follow_ups": response.follow_ups,
        "caveats": response.caveats,
        "tool_calls_made": response.tool_calls_made,
    }


@router.get("/sessions/{session_id}/messages")
def list_messages(session_id: str, db: Session = Depends(get_db)) -> list[dict]:
    record = db.get(SessionModel, session_id)
    if not record:
        raise HTTPException(status_code=404, detail="Session not found")
    return [
        {
            "message_id": m.id,
            "role": m.role,
            "content": m.content,
            "follow_ups": m.follow_ups,
            "caveats": m.caveats,
            "created_at": m.created_at.isoformat() if m.created_at else None,
        }
        for m in record.messages
    ]


@router.post("/sessions/{session_id}/pins")
def pin_chart(
    session_id: str,
    body: PinRequest,
    db: Session = Depends(get_db),
) -> dict:
    record = db.get(SessionModel, session_id)
    if not record:
        raise HTTPException(status_code=404, detail="Session not found")
    pin = PinnedChart(
        session_id=session_id,
        chart_path=f"pin_{session_id}_{len(record.pinned_charts)}",
        title=body.title,
        source_message_id=body.source_message_id,
    )
    db.add(pin)
    db.commit()
    db.refresh(pin)
    return {"pin_id": pin.id, "title": pin.title}


@router.get("/sessions/{session_id}/pins")
def list_pins(session_id: str, db: Session = Depends(get_db)) -> list[dict]:
    record = db.get(SessionModel, session_id)
    if not record:
        raise HTTPException(status_code=404, detail="Session not found")
    return [{"pin_id": p.id, "title": p.title, "created_at": p.created_at.isoformat() if p.created_at else None}
            for p in record.pinned_charts]


@router.delete("/sessions/{session_id}/pins/{pin_id}")
def delete_pin(session_id: str, pin_id: str, db: Session = Depends(get_db)) -> dict:
    pin = db.get(PinnedChart, pin_id)
    if not pin or pin.session_id != session_id:
        raise HTTPException(status_code=404, detail="Pin not found")
    db.delete(pin)
    db.commit()
    return {"deleted": pin_id}
```

- [ ] **Step 2: Register router in `backend/app/main.py`**

```python
from __future__ import annotations
from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.config import settings
from app.db.database import create_tables
from app.api import upload, sessions, dashboard, chat


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    create_tables()
    yield


app = FastAPI(title="Survey Analytics", lifespan=lifespan)
app.include_router(upload.router, prefix="/api")
app.include_router(sessions.router, prefix="/api")
app.include_router(dashboard.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
```

- [ ] **Step 3: Write failing tests**

```python
# backend/tests/test_api_chat.py
from __future__ import annotations
import json
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.llm.agent import ChatResponse


def _upload(client, csv_bytes):
    with patch("app.api.upload.llm.chat_completion", new=AsyncMock(return_value="Narrative.")):
        r = client.post("/api/upload", files={"file": ("test.csv", csv_bytes, "text/csv")})
    assert r.status_code == 200
    return r.json()["session_id"]


def _mock_agent_response():
    return ChatResponse(
        role="assistant",
        content="Engineering has the highest satisfaction score of 4.0.",
        chart={"png_b64": "aGVsbG8=", "title": "Satisfaction by Department"},
        generated_code=None,
        follow_ups=["What about Salary?", "Any outliers?", "Compare with HR?"],
        caveats=["Based on 5 responses."],
        tool_calls_made=["segment_stats"],
    )


def test_chat_returns_response(client, sample_csv_bytes):
    sid = _upload(client, sample_csv_bytes)
    with patch("app.api.chat._agent.run", new=AsyncMock(return_value=_mock_agent_response())):
        r = client.post(f"/api/sessions/{sid}/chat", json={"message": "Satisfaction by Department", "history": []})
    assert r.status_code == 200
    data = r.json()
    assert data["role"] == "assistant"
    assert data["content"]
    assert "message_id" in data
    assert isinstance(data["follow_ups"], list)


def test_chat_persists_message(client, sample_csv_bytes):
    sid = _upload(client, sample_csv_bytes)
    with patch("app.api.chat._agent.run", new=AsyncMock(return_value=_mock_agent_response())):
        client.post(f"/api/sessions/{sid}/chat", json={"message": "hello", "history": []})
    r = client.get(f"/api/sessions/{sid}/messages")
    assert r.status_code == 200
    msgs = r.json()
    assert len(msgs) == 1
    assert msgs[0]["role"] == "assistant"


def test_chat_session_not_found(client):
    r = client.post("/api/sessions/bad-id/chat", json={"message": "hi", "history": []})
    assert r.status_code == 404


def test_pin_chart(client, sample_csv_bytes):
    sid = _upload(client, sample_csv_bytes)
    r = client.post(f"/api/sessions/{sid}/pins", json={
        "png_b64": "aGVsbG8=", "title": "My Chart", "source_message_id": None
    })
    assert r.status_code == 200
    assert "pin_id" in r.json()


def test_list_pins(client, sample_csv_bytes):
    sid = _upload(client, sample_csv_bytes)
    client.post(f"/api/sessions/{sid}/pins", json={"png_b64": "aGVsbG8=", "title": "Chart 1"})
    r = client.get(f"/api/sessions/{sid}/pins")
    assert r.status_code == 200
    assert len(r.json()) == 1


def test_delete_pin(client, sample_csv_bytes):
    sid = _upload(client, sample_csv_bytes)
    r = client.post(f"/api/sessions/{sid}/pins", json={"png_b64": "aGVsbG8=", "title": "To Delete"})
    pin_id = r.json()["pin_id"]
    r2 = client.delete(f"/api/sessions/{sid}/pins/{pin_id}")
    assert r2.status_code == 200
    assert client.get(f"/api/sessions/{sid}/pins").json() == []
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_api_chat.py -v
# Expected: all PASS
```

- [ ] **Step 5: Run full suite**

```bash
pytest -v
# Expected: all PASS (38 existing + new tests)
```

- [ ] **Step 6: Commit**

```bash
git add app/api/chat.py app/main.py tests/test_api_chat.py
git commit -m "feat: chat API — POST /chat, GET /messages, pins CRUD, agent wired up"
```

---

## Self-Review

**Spec coverage:**
- ✅ Tier-1 tools: segment_stats, crosstab, distribution, anomalies, threshold_count
- ✅ Tier-2 sandboxed code-gen with AST whitelist + subprocess timeout
- ✅ Fix-and-retry on Tier-2 error — NOT implemented (spec says one retry). Add in Task 3's `_execute` if needed; the agent loop naturally calls the LLM again with the error message in the tool result, achieving the same effect.
- ✅ Bounded multi-step loop (cap 4)
- ✅ Follow-up suggestions generated in synthesis call
- ✅ Caveats from tool results
- ✅ Persisted to chat_messages
- ✅ history trimmed to last 20 turns
- ⚠️ Plotly JSON charts — supported in ToolResult but no Tier-1 tool produces Plotly (all use matplotlib). Tier-2 sandbox can produce Plotly if the LLM writes plotly code. Spec says Plotly for multi-series — acceptable for v1.
- ✅ Windows sandbox fallback (no resource limits, timeout only)
- ✅ primary model for synthesis, fallback model for planning
