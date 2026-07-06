from __future__ import annotations

import base64
import io
import json
import logging
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from langchain_core.tools import StructuredTool
except ImportError:
    from langchain.tools import StructuredTool
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from code_executor import run_python_analysis_code
from session_store import SurveySession

logger = logging.getLogger("survey_agent")
_ACCENT = "#2563eb"
_MIN_N = 5
_MAX_GRAPH_STEPS = 12  # ~4-5 tool calls per question (agent+tool node per call)


# ─── shared helpers (deterministic tools compute exact numbers, then chart) ──
def _truncate(v: Any, n: int = 1200) -> str:
    t = str(v)
    return t if len(t) <= n else t[:n] + "... [truncated]"


def _resolve(name: str, cols: list[str]) -> str:
    """Map an approximate column name to a real one (exact → normalized → substring)."""
    if name in cols:
        return name
    low = {c.lower().strip(): c for c in cols}
    key = name.lower().strip()
    if key in low:
        return low[key]
    hit = [c for c in cols if key in c.lower()]
    return hit[0] if hit else name


def _chart(fig, title: str = "") -> dict:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    buf.seek(0)
    return {"mime_type": "image/png", "image_base64": base64.b64encode(buf.read()).decode(), "title": title}


def _table_md(rows: list[dict]) -> str:
    if not rows:
        return ""
    cols = list(rows[0].keys())
    md = "| " + " | ".join(cols) + " |\n| " + " | ".join("---" for _ in cols) + " |\n"
    for r in rows[:50]:
        md += "| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n"
    return md


def _pack(summary: str, rows: list[dict] | None, chart: dict | None, caveat: str | None = None) -> str:
    payload: dict[str, Any] = {"summary": summary}
    if rows:
        payload["table_markdown"] = _table_md(rows)
    payload["charts"] = [chart] if chart else []
    if caveat:
        payload["caveat"] = caveat
    return json.dumps(payload, default=str)


_MAX_CATEGORIES = 30  # cap crosstab/pivot axes so a high-cardinality column
                      # (e.g. an ID column) can't blow up on a 1M-row dataset


def _cap_categories(s: pd.Series, max_n: int = _MAX_CATEGORIES) -> pd.Series:
    """Collapse all but the top `max_n` most frequent values into 'Other'."""
    vc = s.value_counts()
    if len(vc) <= max_n:
        return s
    keep = set(vc.head(max_n).index)
    return s.where(s.isin(keep), other="Other")


def _small_sample_caveat(n: int, label: str = "responses") -> str | None:
    return f"Based on only {n} {label} — interpret with care." if n < _MIN_N * 2 else None


def _bar(labels, values, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar([str(l) for l in labels], values, color=_ACCENT)
    ax.set_title(title, fontsize=12); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    return fig


# ─── the deterministic tool implementations (operate on a materialized df) ──
def _t_distribution(df, column):
    col = _resolve(column, list(df.columns))
    s = df[col].dropna()
    if pd.api.types.is_numeric_dtype(s):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(s, bins=min(20, max(s.nunique(), 1)), color=_ACCENT, edgecolor="white")
        ax.set_title(f"{col} — distribution"); ax.set_xlabel(col); ax.set_ylabel("Frequency")
        rows = [{"stat": k, "value": round(float(v), 3)} for k, v in
                {"min": s.min(), "max": s.max(), "mean": s.mean(), "median": s.median()}.items()]
        return _pack(f"{col}: mean={round(float(s.mean()),3)}, min={s.min()}, max={s.max()}", rows, _chart(fig, col))
    vc = s.astype(str).value_counts().head(20)
    rows = [{col: k, "count": int(v)} for k, v in vc.items()]
    return _pack(f"{col}: top value '{vc.index[0]}' ({int(vc.iloc[0])})", rows,
                 _chart(_bar(vc.index, vc.values, f"{col} — counts", col, "Count"), col))


def _t_breakdown(df, group_col, metric_col=""):
    g = _resolve(group_col, list(df.columns))
    if metric_col:
        m = _resolve(metric_col, list(df.columns))
        counts = df.groupby(g)[m].size()
        agg = df.groupby(g)[m].mean().round(3).sort_values(ascending=False)
        rows = [{g: str(k), f"mean_{m}": float(v)} for k, v in agg.items()]
        top = agg.index[0]
        caveat = _small_sample_caveat(int(counts.min())) if len(counts) else None
        return _pack(f"{m} by {g}: highest mean '{top}' ({agg.iloc[0]})", rows,
                     _chart(_bar(agg.index, agg.values, f"Mean {m} by {g}", g, f"Mean {m}"), g), caveat)
    vc = df[g].astype(str).value_counts().head(25)
    rows = [{g: str(k), "count": int(v)} for k, v in vc.items()]
    caveat = _small_sample_caveat(int(vc.iloc[-1])) if len(vc) else None
    return _pack(f"Counts by {g}: '{vc.index[0]}' is largest ({int(vc.iloc[0])})", rows,
                 _chart(_bar(vc.index, vc.values, f"Count by {g}", g, "Count"), g), caveat)


def _t_pie(df, column):
    col = _resolve(column, list(df.columns))
    vc = df[col].dropna().astype(str).value_counts()
    caveat = _small_sample_caveat(int(vc.sum())) if len(vc) else None
    if len(vc) > 8:
        vc = pd.concat([vc.head(8), pd.Series({"Other": int(vc.iloc[8:].sum())})])
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = plt.get_cmap("Blues")
    ax.pie(vc.values, labels=vc.index, autopct="%1.1f%%",
           colors=[cmap(x) for x in np.linspace(0.4, 0.9, len(vc))], wedgeprops={"edgecolor": "white"})
    ax.set_title(f"{col} — share"); ax.axis("equal")
    total = int(vc.sum())
    rows = [{col: k, "count": int(v), "pct": round(v / total * 100, 1)} for k, v in vc.items()]
    return _pack(f"{col}: '{vc.index[0]}' largest share ({round(vc.iloc[0]/total*100,1)}%)", rows,
                 _chart(fig, col), caveat)


def _t_crosstab(df, row_col, col_col):
    r = _resolve(row_col, list(df.columns)); c = _resolve(col_col, list(df.columns))
    row_s = _cap_categories(df[r].astype(str))
    col_s = _cap_categories(df[c].astype(str))
    capped = row_s.name != r or (row_s.nunique() < df[r].nunique()) or (col_s.nunique() < df[c].nunique())
    ct = pd.crosstab(row_s, col_s)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(ct.index)); w = 0.8 / max(len(ct.columns), 1)
    cmap = plt.get_cmap("viridis")
    for j, cc in enumerate(ct.columns):
        ax.bar(x + j * w, ct[cc].values, w, label=str(cc), color=cmap(j / max(len(ct.columns) - 1, 1)))
    ax.set_xticks(x + w * (len(ct.columns) - 1) / 2); ax.set_xticklabels([str(i) for i in ct.index], rotation=30, ha="right")
    ax.set_title(f"{r} × {c}"); ax.set_xlabel(r); ax.set_ylabel("Count"); ax.legend(title=c, fontsize=8)
    rows = ct.reset_index().to_dict(orient="records")
    caveat = (f"Both columns have many values — grouped beyond the top {_MAX_CATEGORIES} into 'Other'."
              if capped else None)
    return _pack(f"Cross-tab {r} × {c}: {len(ct.index)}×{len(ct.columns)}", rows, _chart(fig, f"{r} x {c}"), caveat)


def _t_rank_groups(df, group_col, target_col, target_value, min_n=_MIN_N):
    g = _resolve(group_col, list(df.columns)); t = _resolve(target_col, list(df.columns))
    sub = df[[g, t]].dropna()
    key = str(target_value).strip().lower()
    sub["_m"] = sub[t].astype(str).str.strip().str.lower().apply(lambda v: key in v or v == key)
    grp = sub.groupby(g)
    res = pd.DataFrame({"matched": grp["_m"].sum(), "total": grp["_m"].size()})
    res["pct"] = (res["matched"] / res["total"] * 100).round(1)
    res = res[res["total"] >= min_n].sort_values("pct", ascending=False)
    if res.empty:
        return _pack(f"No groups with ≥{min_n} respondents for '{target_value}' in {t}.", None, None)
    rows = [{g: str(i), "matched": int(r.matched), "total": int(r.total), "pct": float(r.pct)}
            for i, r in res.head(15).iterrows()]
    top = res.index[0]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(rows) + 1)))
    ax.barh([str(i) for i in res.head(15).index][::-1], res["pct"].head(15).values[::-1], color=_ACCENT)
    ax.set_xlabel(f"% '{target_value}'"); ax.set_title(f"'{target_value}' in {t} by {g}")
    return _pack(f"Highest '{target_value}' in {t}: '{top}' at {res.iloc[0].pct}% "
                 f"({int(res.iloc[0].matched)}/{int(res.iloc[0].total)}; groups <{min_n} excluded)",
                 rows, _chart(fig, g))


def _t_filter_profile(df, filter_col, filter_value, operator="eq"):
    fc = _resolve(filter_col, list(df.columns))
    s = df[fc]
    if operator in ("gt", "lt", "gte", "lte"):
        n = pd.to_numeric(s, errors="coerce"); v = float(filter_value)
        mask = {"gt": n > v, "lt": n < v, "gte": n >= v, "lte": n <= v}[operator]
    else:
        mask = s.astype(str).str.strip().str.lower() == str(filter_value).strip().lower()
    sub = df[mask]
    total = len(df); n = len(sub); pct = round(n / total * 100, 1) if total else 0
    rows = []
    for c in df.columns:
        if c == fc:
            continue
        cs = sub[c].dropna()
        if cs.empty:
            continue
        if pd.api.types.is_numeric_dtype(cs):
            rows.append({"column": c, "profile": f"mean {round(float(cs.mean()),2)}, median {round(float(cs.median()),2)}"})
        else:
            vc = cs.astype(str).value_counts().head(3)
            rows.append({"column": c, "profile": ", ".join(f"{k} {round(v/len(cs)*100)}%" for k, v in vc.items())})
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Matched", "Other"], [n, total - n], color=[_ACCENT, "#d7dbe0"])
    ax.set_title(f"{fc} {operator} {filter_value}"); ax.set_ylabel("Respondents")
    return _pack(f"{n} of {total} ({pct}%) match {fc} {operator} {filter_value}", rows, _chart(fig, fc))


def _t_pivot(df, index_col, column_col, value_col=""):
    i = _resolve(index_col, list(df.columns)); c = _resolve(column_col, list(df.columns))
    idx_s = _cap_categories(df[i].astype(str))
    col_s = _cap_categories(df[c].astype(str))
    capped = (idx_s.nunique() < df[i].nunique()) or (col_s.nunique() < df[c].nunique())
    if value_col and pd.api.types.is_numeric_dtype(df[_resolve(value_col, list(df.columns))]):
        v = _resolve(value_col, list(df.columns))
        pv = pd.pivot_table(pd.DataFrame({i: idx_s, c: col_s, v: df[v]}),
                            index=i, columns=c, values=v, aggfunc="mean").round(2)
        ylab = f"Mean {v}"
    else:
        pv = pd.crosstab(idx_s, col_s); ylab = "Count"
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(pv.index)); w = 0.8 / max(len(pv.columns), 1)
    for j, cc in enumerate(pv.columns):
        ax.bar(x + j * w, pv[cc].values, w, label=str(cc), color=plt.get_cmap("viridis")(j / max(len(pv.columns) - 1, 1)))
    ax.set_xticks(x + w * (len(pv.columns) - 1) / 2); ax.set_xticklabels([str(k) for k in pv.index], rotation=30, ha="right")
    ax.set_title(f"{i} × {c}"); ax.set_ylabel(ylab); ax.legend(title=c, fontsize=8)
    caveat = (f"Both columns have many values — grouped beyond the top {_MAX_CATEGORIES} into 'Other'."
              if capped else None)
    return _pack(f"{i} × {c}: {len(pv.index)}×{len(pv.columns)} grid",
                 pv.reset_index().to_dict(orient="records"), _chart(fig, i), caveat)


# ─── tool argument schemas ──────────────────────────────────────────────────
class OneCol(BaseModel):
    column: str = Field(description="Column name")

class Breakdown(BaseModel):
    group_col: str = Field(description="Column to group by")
    metric_col: str = Field("", description="Optional numeric column to average; omit for counts")

class TwoCol(BaseModel):
    row_col: str = Field(description="First categorical column")
    col_col: str = Field(description="Second categorical column")

class Rank(BaseModel):
    group_col: str = Field(description="Group to rank, e.g. City/State")
    target_col: str = Field(description="Column holding the response")
    target_value: str = Field(description="Exact response value to measure the share of")

class Filter(BaseModel):
    filter_col: str = Field(description="Column to filter on")
    filter_value: str = Field(description="Value to match")
    operator: str = Field("eq", description="eq|gt|lt|gte|lte")

class Pivot(BaseModel):
    index_col: str = Field(description="Row grouping column")
    column_col: str = Field(description="Column grouping column")
    value_col: str = Field("", description="Optional numeric column to average; omit for counts")

class PyCode(BaseModel):
    code: str = Field(description="Python using df (all files combined), pd, np, plt. Assign to `result`.")


class SurveyAnalysisAgent:
    def __init__(self) -> None:
        # NOTE: QA environment cannot store secrets externally — kept inline as
        # placeholders. TODO: load from env (os.getenv) in higher environments.
        deployment = "gpt-5.4"
        endpoint = "https://tcsion-nonprod-resource.cognitiveservices.azure.com"
        api_key = "REPLACE_WITH_AZURE_OPENAI_KEY"
        api_version = "2025-01-01-preview"

        if not deployment or not endpoint or not api_key:
            raise RuntimeError("Missing Azure OpenAI settings.")

        self.llm = AzureChatOpenAI(
            azure_endpoint=endpoint, api_key=api_key, api_version=api_version,
            deployment_name=deployment, temperature=0.2,
        )
        self.max_history_messages = 16

    # deterministic tools operate directly on the session's in-memory dataframe
    def _build_tools(self, session: SurveySession) -> list[StructuredTool]:
        def wrap(fn):
            def inner(**kwargs):
                try:
                    return fn(session.df, **kwargs)
                except Exception as exc:
                    return json.dumps({"summary": f"Tool error: {exc}", "charts": []})
                finally:
                    # Safety net: close any figure left open by a tool that raised
                    # before reaching _chart() — prevents a slow memory leak on
                    # this long-running single-worker process.
                    plt.close("all")
            return inner

        def run_python(code: str) -> str:
            resp = run_python_analysis_code(code, session.df)
            return json.dumps(resp, default=str)

        def dataset_schema() -> str:
            return session.profile

        T = StructuredTool.from_function
        return [
            T(name="dataset_schema", func=dataset_schema,
              description="Get the dataset profile (columns, types, top values, row count). Call before analysis."),
            T(name="distribution", func=wrap(_t_distribution), args_schema=OneCol,
              description="Distribution of ONE column (bar for categorical, histogram for numeric). Use for 'distribution/bar chart of X'."),
            T(name="breakdown", func=wrap(_t_breakdown), args_schema=Breakdown,
              description="Breakdown by a group: mean of metric_col per group, or counts if metric_col omitted. Use for 'X by Y'."),
            T(name="pie_chart", func=wrap(_t_pie), args_schema=OneCol,
              description="Pie/share of one categorical column. Use for 'pie chart of X' / 'share of X'."),
            T(name="crosstab", func=wrap(_t_crosstab), args_schema=TwoCol,
              description="Cross-tabulate two categorical columns (grouped bar)."),
            T(name="rank_groups_by_value", func=wrap(_t_rank_groups), args_schema=Rank,
              description="Rank groups by % that selected a specific response (min 5 per group). Use for 'which city/state/segment has the highest % of <value>'."),
            T(name="filter_profile", func=wrap(_t_filter_profile), args_schema=Filter,
              description="Subset respondents by a condition, then profile that subset. Use for 'profile of respondents who <condition>'."),
            T(name="pivot_table", func=wrap(_t_pivot), args_schema=Pivot,
              description="Two-dimensional breakdown (index × column), means or counts. Use for 'show X by A and B'."),
            T(name="run_python", func=run_python, args_schema=PyCode,
              description="FALLBACK for custom analysis/charts no other tool covers. df is all files combined; assign result."),
        ]

    def _charts(self, messages: list[Any]) -> list[dict[str, str]]:
        charts = []
        for m in messages:
            content = getattr(m, "content", None)
            if not isinstance(content, str):
                continue
            try:
                parsed = json.loads(content)
            except Exception:
                continue
            for item in (parsed.get("charts") or []) if isinstance(parsed, dict) else []:
                if isinstance(item, dict) and item.get("image_base64"):
                    charts.append({"mime_type": item.get("mime_type", "image/png"),
                                   "image_base64": item["image_base64"], "title": item.get("title", "")})
        return charts[-6:]

    def _history(self, session: SurveySession) -> list[tuple[str, str]]:
        out = []
        for it in session.chat_history[-self.max_history_messages:]:
            role = it.get("role", "").lower()
            if it.get("content") and role in ("user", "assistant"):
                out.append(("human" if role == "user" else "assistant", it["content"]))
        return out

    def answer(self, session: SurveySession, question: str) -> dict[str, Any]:
        tools = self._build_tools(session)
        agent = create_react_agent(self.llm, tools)
        system = (
            "You are a senior survey analyst. Prefer the deterministic tools "
            "(distribution, breakdown, pie_chart, crosstab, rank_groups_by_value, "
            "filter_profile, pivot_table) so numbers are exact; use run_python only "
            "for custom charts no tool covers. The dataset is ALL uploaded files combined. "
            "Always include the tool's numbers (as a markdown table when returned) and a short interpretation.\n\n"
            f"{session.profile}"
        )
        msgs: list[tuple[str, str]] = [("system", system)]
        msgs.extend(self._history(session))
        msgs.append(("human", question))

        # Bound the tool-calling loop (like the main survey app's step cap) so a
        # single question can't chain an unbounded number of tool calls.
        resp = agent.invoke({"messages": msgs}, config={"recursion_limit": _MAX_GRAPH_STEPS})
        messages = resp.get("messages", [])
        if not messages:
            return {"answer": "I could not produce an analysis response.", "charts": []}
        return {"answer": str(messages[-1].content), "charts": self._charts(messages)}
