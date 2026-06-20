from __future__ import annotations
import io
import re
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


TOOL_NAMES = ["segment_stats", "distribution", "pie_chart", "crosstab", "anomalies",
              "threshold_count", "rank_groups_by_value", "filter_profile",
              "list_filtered_values", "pivot_table", "compare_expectations_by_segment"]

_CAVEAT_MIN_N = 30

# Shared palette — keeps generated charts consistent with the UI accent.
_ACCENT = "#2563eb"
_ACCENT_LIGHT = "#bfd4fe"
_MUTED = "#d7dbe0"


def _fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _black_bar(ax, labels, values, xlabel="", ylabel="", title=""):
    ax.bar([str(l) for l in labels], values, color=_ACCENT)
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
        ax.hist(series, bins=min(20, series.nunique()), color=_ACCENT, edgecolor="white")
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


def pie_chart(df: pd.DataFrame, column: str, top_n: int = 8) -> ToolResult:
    """Pie/share breakdown of a categorical column's value counts."""
    series = df[column].dropna().astype(str)
    counts = series.value_counts()
    if len(counts) > top_n:
        top = counts.head(top_n)
        other = counts.iloc[top_n:].sum()
        labels = list(top.index) + ["Other"]
        values = list(top.values) + [int(other)]
    else:
        labels = list(counts.index)
        values = list(counts.values)

    cmap = plt.get_cmap("Blues")
    colors = [cmap(v) for v in np.linspace(0.4, 0.9, len(values))]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(values, labels=labels, autopct="%1.1f%%", colors=colors,
           wedgeprops={"edgecolor": "white", "linewidth": 1},
           textprops={"fontsize": 9})
    ax.set_title(f"{column} — share of responses", fontsize=12)
    ax.axis("equal")
    png = _fig_to_bytes(fig)

    total = int(sum(values))
    table = [{"value": l, "count": int(v), "pct": round(v / total * 100, 1) if total else 0}
             for l, v in zip(labels, values)]
    top_label = labels[0] if labels else "N/A"
    summary = (f"{column}: '{top_label}' is the largest share "
               f"({table[0]['pct']}% of {total} responses)" if table else f"{column}: no data")
    caveat = f"{column} has {len(series)} non-null values." if len(series) < _CAVEAT_MIN_N else None
    return ToolResult(tool_name="pie_chart", params={"column": column},
                      summary=summary, table=table, png_bytes=png, caveat=caveat)


def crosstab(df: pd.DataFrame, row_col: str, col_col: str, normalize: bool = False) -> ToolResult:
    ct = pd.crosstab(df[row_col], df[col_col], normalize="index" if normalize else False)
    # Render as a grouped bar chart (far more legible than a grayscale heatmap):
    # one cluster per row category, one bar per column category in greyscale shades.
    fig, ax = plt.subplots(figsize=(9, 5))
    n_cols = len(ct.columns)
    n_rows = len(ct.index)
    x = np.arange(n_rows)
    width = 0.8 / max(n_cols, 1)
    cmap = plt.get_cmap("viridis")
    colors = [cmap(v) for v in np.linspace(0.15, 0.85, max(n_cols, 1))]
    for j, col in enumerate(ct.columns):
        ax.bar(x + j * width, ct[col].values, width,
               label=str(col), color=colors[j], edgecolor="white", linewidth=0.5)
    ax.set_xticks(x + width * (n_cols - 1) / 2)
    ax.set_xticklabels([str(r) for r in ct.index], rotation=30, ha="right")
    ax.set_xlabel(row_col)
    ax.set_ylabel("Proportion" if normalize else "Count")
    ax.set_title(f"{row_col} × {col_col}", fontsize=12)
    ax.legend(title=col_col, fontsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
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
               boxprops={"facecolor": _ACCENT_LIGHT, "edgecolor": _ACCENT},
               medianprops={"color": _ACCENT}, flierprops={"marker": "o", "markeredgecolor": _ACCENT})
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
    ax.bar(["Meets threshold", "Does not"], [count, total - count], color=[_ACCENT, _MUTED])
    ax.set_title(f"{column} {op_label} {threshold}", fontsize=12)
    ax.set_ylabel("Count")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    png = _fig_to_bytes(fig)

    return ToolResult(tool_name="threshold_count",
                      params={"column": column, "threshold": threshold, "operator": operator},
                      summary=summary, table={"count": count, "total": total, "pct": pct},
                      png_bytes=png)


def _norm(value) -> str:
    """Normalize a cell/value for robust matching: remove ALL whitespace and
    lowercase. Handles label quirks like '>=16 %' vs '>=16%' and '8-9%' vs '8-9 %'."""
    return re.sub(r"\s+", "", str(value)).lower()


def _build_mask(series: pd.Series, value, op: str) -> pd.Series:
    """Boolean mask for a column condition. Supports numeric comparisons and
    string equality / substring match, with whitespace-normalized matching."""
    if op in ("gt", "lt", "gte", "lte"):
        num = pd.to_numeric(series, errors="coerce")
        v = float(value)
        return {"gt": num > v, "lt": num < v, "gte": num >= v, "lte": num <= v}[op]
    s = series.map(_norm)
    target = _norm(value)
    if op == "ne":
        return s != target
    if op == "contains":
        return s.str.contains(re.escape(target), na=False)
    # default: normalized equality
    return s == target


# Standard age groups used when a numeric age column is segmented.
_AGE_BINS = [17, 24, 34, 44, 54, 200]
_AGE_LABELS = ["18-24", "25-34", "35-44", "45-54", "55+"]

# Canonical inflation-expectation response labels.
_PRICE_INCREASE_MORE = "Price increase more than current rate"
_EXPECTATION_LABELS = {_norm(x) for x in {
    "Price increase more than current rate",
    "Price increase similar to current rate",
    "Price increase less than current rate",
    "No change in prices",
    "Decline in prices",
}}


def _bin_segment(df: pd.DataFrame, segment_col: str) -> pd.Series:
    """Return the segment series, binning numeric age-like columns into age groups."""
    s = df[segment_col]
    if pd.api.types.is_numeric_dtype(s) and "age" in segment_col.lower():
        return pd.cut(pd.to_numeric(s, errors="coerce"), bins=_AGE_BINS, labels=_AGE_LABELS)
    return s


def rank_groups_by_value(
    df: pd.DataFrame, group_col: str, target_col: str, target_value: str,
    top_n: int = 15, min_n: int = 5, match_mode: str = "eq",
) -> ToolResult:
    """For a specific response value, rank each group by the share that selected it.

    Answers 'which city/state/segment has the highest % of <response>?' deterministically.
    Groups with fewer than `min_n` respondents are excluded from ranking to avoid
    small-sample outliers (matching the survey's stated n>=5 rule). `match_mode` is
    'eq' (exact, normalized) or 'contains' (substring).
    """
    seg = _bin_segment(df, group_col)
    # Denominator = respondents who answered this question (drop rows missing either).
    work = pd.DataFrame({group_col: seg, target_col: df[target_col]}).dropna(subset=[group_col, target_col])
    group_total = work.groupby(group_col, observed=True)[target_col].size()
    match_mask = _build_mask(work[target_col], target_value, "contains" if match_mode == "contains" else "eq")
    match_total = work[match_mask].groupby(group_col, observed=True)[target_col].size()

    res = pd.DataFrame({"matched": match_total, "total": group_total}).fillna(0)
    res["matched"] = res["matched"].astype(int)
    res["total"] = res["total"].astype(int)
    res["pct"] = (res["matched"] / res["total"] * 100).where(res["total"] > 0, 0).round(1)
    ranked = res[res["total"] >= min_n].sort_values("pct", ascending=False)
    if ranked.empty:                       # fall back to all groups if none meet min_n
        ranked = res.sort_values("pct", ascending=False)
    res_top = ranked.head(top_n)

    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(res_top) + 1)))
    ax.barh([str(i) for i in res_top.index][::-1], res_top["pct"].values[::-1], color=_ACCENT)
    ax.set_xlabel(f"% selecting '{target_value}'")
    ax.set_title(f"'{target_value}' in {target_col} by {group_col}", fontsize=12)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    png = _fig_to_bytes(fig)

    table = [
        {group_col: str(idx), "matched": int(r["matched"]), "total": int(r["total"]), "pct": float(r["pct"])}
        for idx, r in res_top.iterrows()
    ]
    if len(ranked) and ranked.iloc[0]["total"] > 0:
        top_group = ranked.index[0]
        summary = (f"Highest share of '{target_value}' in {target_col}: '{top_group}' "
                   f"at {ranked.iloc[0]['pct']}% ({int(ranked.iloc[0]['matched'])}/{int(ranked.iloc[0]['total'])}; "
                   f"groups with <{min_n} respondents excluded)")
    else:
        summary = f"No respondents selected '{target_value}' in {target_col}."
    caveat = (f"Only groups with at least {min_n} respondents were ranked."
              if len(res) > len(ranked) else None)
    return ToolResult(tool_name="rank_groups_by_value",
                      params={"group_col": group_col, "target_col": target_col,
                              "target_value": target_value, "min_n": min_n, "match_mode": match_mode},
                      summary=summary, table=table, png_bytes=png, caveat=caveat)


def filter_profile(
    df: pd.DataFrame, filter_col: str, filter_value: str, operator: str = "eq",
) -> ToolResult:
    """Subset rows matching a condition, then profile the demographic make-up of that subset.

    Answers 'show the profile of respondents who <condition>'.
    """
    mask = _build_mask(df[filter_col], filter_value, operator)
    subset = df[mask]
    n, total = len(subset), len(df)
    pct = round(n / total * 100, 1) if total else 0

    op_label = {"gt": ">", "lt": "<", "gte": ">=", "lte": "<=", "eq": "==", "ne": "!=", "contains": "contains"}.get(operator, operator)

    table: list[dict] = []
    for col in df.columns:
        if col == filter_col:
            continue
        s = subset[col].dropna()
        if s.empty:
            continue
        if pd.api.types.is_numeric_dtype(s):
            breakdown = f"mean {round(float(s.mean()), 2)}, median {round(float(s.median()), 2)}"
        else:
            vc = s.astype(str).value_counts().head(3)
            breakdown = ", ".join(f"{k} {round(v / len(s) * 100)}%" for k, v in vc.items())
        table.append({"column": col, "profile (within matched)": breakdown})

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["Matched", "Other"], [n, total - n], color=[_ACCENT, _MUTED])
    ax.set_title(f"Respondents where {filter_col} {op_label} {filter_value}", fontsize=11)
    ax.set_ylabel("Respondents")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    png = _fig_to_bytes(fig)

    summary = f"{n} of {total} respondents ({pct}%) match {filter_col} {op_label} {filter_value}"
    caveat = f"Only {n} respondents matched — profile may be unreliable." if n < _CAVEAT_MIN_N else None
    return ToolResult(tool_name="filter_profile",
                      params={"filter_col": filter_col, "filter_value": filter_value, "operator": operator},
                      summary=summary, table=table, png_bytes=png, caveat=caveat)


def list_filtered_values(
    df: pd.DataFrame, filter_col: str, filter_value: str, value_cols: list[str],
    operator: str = "eq", max_rows: int = 50,
) -> ToolResult:
    """List the raw values entered in specific columns for rows matching a condition.

    Answers 'for respondents who selected X, what exact values were entered in Q14/Q15/Q16?'.
    """
    if isinstance(value_cols, str):
        value_cols = [value_cols]
    mask = _build_mask(df[filter_col], filter_value, operator)
    subset = df[mask]
    present = [c for c in value_cols if c in df.columns]
    out = subset[present].head(max_rows)
    table = out.replace({np.nan: None}).to_dict(orient="records")
    summary = (f"{len(subset)} respondents match {filter_col}='{filter_value}'; "
               f"showing {len(out)} rows of {', '.join(present)}")
    return ToolResult(tool_name="list_filtered_values",
                      params={"filter_col": filter_col, "filter_value": filter_value, "value_cols": present},
                      summary=summary, table=table, png_bytes=None)


def pivot_table(
    df: pd.DataFrame, index_col: str, column_col: str, value_col: str | None = None,
) -> ToolResult:
    """Two-dimensional breakdown: index_col × column_col. Mean of value_col if numeric,
    otherwise response counts. Answers 'show X by A and B'."""
    if value_col and pd.api.types.is_numeric_dtype(df[value_col]):
        pivot = pd.pivot_table(df, index=index_col, columns=column_col, values=value_col, aggfunc="mean").round(2)
        ylabel = f"Mean {value_col}"
        title = f"{value_col} by {index_col} × {column_col}"
    else:
        pivot = pd.crosstab(df[index_col], df[column_col])
        ylabel = "Count"
        title = f"{index_col} × {column_col} (counts)"

    fig, ax = plt.subplots(figsize=(9, 5))
    n_cols = len(pivot.columns)
    x = np.arange(len(pivot.index))
    width = 0.8 / max(n_cols, 1)
    cmap = plt.get_cmap("viridis")
    colors = [cmap(v) for v in np.linspace(0.15, 0.85, max(n_cols, 1))]
    for j, col in enumerate(pivot.columns):
        ax.bar(x + j * width, pivot[col].values, width, label=str(col),
               color=colors[j], edgecolor="white", linewidth=0.5)
    ax.set_xticks(x + width * (n_cols - 1) / 2)
    ax.set_xticklabels([str(i) for i in pivot.index], rotation=30, ha="right")
    ax.set_xlabel(index_col)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12)
    ax.legend(title=column_col, fontsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    png = _fig_to_bytes(fig)

    table = pivot.reset_index().rename(columns=str).replace({np.nan: None}).to_dict(orient="records")
    summary = f"{title}: {len(pivot.index)} × {len(pivot.columns)} grid"
    return ToolResult(tool_name="pivot_table",
                      params={"index_col": index_col, "column_col": column_col, "value_col": value_col},
                      summary=summary, table=table, png_bytes=png)


def _detect_expectation_columns(df: pd.DataFrame) -> list[str]:
    """Columns whose values are predominantly inflation-expectation labels."""
    cols = []
    for c in df.columns:
        s = df[c].dropna()
        if s.empty:
            continue
        sample = s.astype(str).map(_norm)
        hits = sample.isin(_EXPECTATION_LABELS).mean()
        if hits >= 0.5:
            cols.append(c)
    return cols


def compare_expectations_by_segment(
    df: pd.DataFrame, segment_col: str, target_value: str = _PRICE_INCREASE_MORE,
    expectation_cols: list[str] | None = None,
) -> ToolResult:
    """Compare inflation expectations across a segment (gender, age group, income, …).

    For each segment value, reports respondent count, sample share, and the share
    selecting `target_value` ('price increase more than current rate' by default)
    for every expectation column. Reproduces the Q1-5 multi-metric comparison.
    Numeric age columns are auto-binned into age groups.
    """
    cols = expectation_cols or _detect_expectation_columns(df)
    if not cols:
        raise ValueError("No inflation-expectation columns detected in this dataset.")

    seg = _bin_segment(df, segment_col)
    work = df.copy()
    work["__seg__"] = seg
    work = work.dropna(subset=["__seg__"])
    total_n = len(work)

    rows: list[dict] = []
    for seg_val, g in work.groupby("__seg__", observed=True):
        n = len(g)
        if n == 0:
            continue
        row = {"segment": str(seg_val), "respondents": n,
               "sample_share": round(n / total_n * 100, 1) if total_n else 0}
        for c in cols:
            s = g[c].dropna()
            share = round(_build_mask(s, target_value, "eq").mean() * 100, 1) if len(s) else 0.0
            row[_short_label(c)] = share
        rows.append(row)
    rows.sort(key=lambda r: r["respondents"], reverse=True)

    # Chart: grouped bars of each segment's share for the first few expectation metrics.
    metric_cols = [_short_label(c) for c in cols][:6]
    fig, ax = plt.subplots(figsize=(9, 5))
    seg_labels = [r["segment"] for r in rows]
    x = np.arange(len(seg_labels))
    width = 0.8 / max(len(metric_cols), 1)
    cmap = plt.get_cmap("viridis")
    colors = [cmap(v) for v in np.linspace(0.15, 0.85, max(len(metric_cols), 1))]
    for j, m in enumerate(metric_cols):
        ax.bar(x + j * width, [r.get(m, 0) for r in rows], width, label=m,
               color=colors[j], edgecolor="white", linewidth=0.5)
    ax.set_xticks(x + width * (len(metric_cols) - 1) / 2)
    ax.set_xticklabels(seg_labels, rotation=20, ha="right")
    ax.set_ylabel(f"% '{target_value}'")
    ax.set_title(f"Inflation expectations by {segment_col}", fontsize=12)
    ax.legend(fontsize=7, ncol=2)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    png = _fig_to_bytes(fig)

    summary = (f"Inflation expectations ('{target_value}') by {segment_col} across "
               f"{len(cols)} metrics and {len(rows)} segments.")
    min_n = min((r["respondents"] for r in rows), default=0)
    caveat = f"Smallest segment has only {min_n} respondents." if min_n < _CAVEAT_MIN_N else None
    return ToolResult(tool_name="compare_expectations_by_segment",
                      params={"segment_col": segment_col, "target_value": target_value},
                      summary=summary, table=rows, png_bytes=png, caveat=caveat)


def _short_label(col: str) -> str:
    """Shorten verbose survey headers like 'Q11_2_Food Products(Q11)' → 'Q11 Food Products'."""
    m = re.match(r"(Q\d+)_\d+_(.+?)\(Q\d+\)$", col)
    if m:
        return f"{m.group(1)} {m.group(2)}"
    return col


_DISPATCH = {
    "segment_stats": segment_stats,
    "distribution": distribution,
    "pie_chart": pie_chart,
    "crosstab": crosstab,
    "anomalies": anomalies,
    "threshold_count": threshold_count,
    "rank_groups_by_value": rank_groups_by_value,
    "filter_profile": filter_profile,
    "list_filtered_values": list_filtered_values,
    "pivot_table": pivot_table,
    "compare_expectations_by_segment": compare_expectations_by_segment,
}


def dispatch_tool(df: pd.DataFrame, name: str, params: dict) -> ToolResult:
    if name not in _DISPATCH:
        raise ValueError(f"Unknown tool: {name}. Available: {list(_DISPATCH)}")
    return _DISPATCH[name](df, **params)
