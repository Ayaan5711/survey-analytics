from __future__ import annotations
import io
from dataclasses import dataclass
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from app.data.profiler import DatasetProfile


@dataclass
class ChartResult:
    title: str
    chart_type: str       # "bar" | "histogram" | "pie"
    png_bytes: bytes
    filename: str


def _save_fig(title: str, chart_type: str, out_dir: Path, idx: int) -> tuple[bytes, str]:
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    png = buf.read()
    fname = f"chart_{idx}_{chart_type}.png"
    (out_dir / fname).write_bytes(png)
    return png, fname


def deterministic_charts(profile: DatasetProfile, out_dir: Path) -> list[ChartResult]:
    """Generate up to 3 standard charts from the profile JSON (no DataFrame needed)."""
    results: list[ChartResult] = []
    idx = 0

    # Chart 1: Bar chart of first categorical column's top_values
    cat_cols = [(c, p) for c, p in profile.columns.items() if p.dtype == "categorical" and p.top_values]
    if cat_cols:
        col_name, col_p = cat_cols[0]
        labels = list(col_p.top_values.keys())
        values = list(col_p.top_values.values())
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(labels[:10], values[:10], color="black")
        ax.set_title(f"{col_name} — response counts", fontsize=12)
        ax.set_xlabel(col_name)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=30)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        png, fname = _save_fig(f"{col_name} counts", "bar", out_dir, idx)
        results.append(ChartResult(title=f"{col_name} — distribution", chart_type="bar",
                                   png_bytes=png, filename=fname))
        idx += 1

    # Chart 2: Histogram of first numeric column's sample_values
    num_cols = [(c, p) for c, p in profile.columns.items() if p.dtype == "numeric"]
    if num_cols:
        col_name, col_p = num_cols[0]
        sample_vals = [v for v in col_p.sample_values if v is not None]
        if sample_vals:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(sample_vals, bins=min(10, len(sample_vals)), color="black", edgecolor="white")
            ax.set_title(f"{col_name} — distribution (sample)", fontsize=12)
            ax.set_xlabel(col_name)
            ax.set_ylabel("Frequency")
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            png, fname = _save_fig(f"{col_name} histogram", "histogram", out_dir, idx)
            results.append(ChartResult(title=f"{col_name} — histogram", chart_type="histogram",
                                       png_bytes=png, filename=fname))
            idx += 1

    # Chart 3: Horizontal bar of missing data % (only if any column has > 0% missing)
    missing = {c: p.missing_pct for c, p in profile.columns.items() if p.missing_pct > 0}
    if missing:
        fig, ax = plt.subplots(figsize=(7, 4))
        cols = list(missing.keys())[:10]
        pcts = [missing[c] for c in cols]
        ax.barh(cols, pcts, color="black")
        ax.set_title("Missing data %", fontsize=12)
        ax.set_xlabel("Missing %")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        png, fname = _save_fig("Missing data", "bar", out_dir, idx)
        results.append(ChartResult(title="Missing data", chart_type="bar",
                                   png_bytes=png, filename=fname))

    return results
