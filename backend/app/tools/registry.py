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


TOOL_NAMES = ["segment_stats", "distribution", "pie_chart", "crosstab", "anomalies", "threshold_count"]

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


_DISPATCH = {
    "segment_stats": segment_stats,
    "distribution": distribution,
    "pie_chart": pie_chart,
    "crosstab": crosstab,
    "anomalies": anomalies,
    "threshold_count": threshold_count,
}


def dispatch_tool(df: pd.DataFrame, name: str, params: dict) -> ToolResult:
    if name not in _DISPATCH:
        raise ValueError(f"Unknown tool: {name}. Available: {list(_DISPATCH)}")
    return _DISPATCH[name](df, **params)
