from __future__ import annotations
import json
import logging
import pandas as pd
from sqlalchemy.orm import Session as DBSession
from app.data.profiler import DatasetProfile
from app.db.models import Insight
from app.llm.client import llm
from app.tools.registry import dispatch_tool

logger = logging.getLogger(__name__)
_MAX_INSIGHTS = 8


def _candidate_findings(profile: DatasetProfile, df: pd.DataFrame) -> list[dict]:
    """Run a fixed sweep of Tier-1 tools and return scored findings."""
    findings: list[dict] = []

    cat_cols = [c for c, p in profile.columns.items() if p.dtype == "categorical"]
    num_cols = [c for c, p in profile.columns.items() if p.dtype == "numeric"]

    # Segment stats: categorical × numeric combinations
    for cat in cat_cols[:3]:
        for num in num_cols[:3]:
            try:
                tr = dispatch_tool(df, "segment_stats", {"group_col": cat, "metric_col": num})
                rows = tr.table
                if not rows:
                    continue
                means = [r.get("mean") or 0 for r in rows if r.get("mean") is not None]
                if len(means) < 2:
                    continue
                effect = max(means) - min(means)
                findings.append({"effect": effect, "summary": tr.summary,
                                  "tool": "segment_stats", "params": {"group_col": cat, "metric_col": num}})
            except Exception as exc:
                logger.debug("Insight sweep error %s×%s: %s", cat, num, exc)

    # Anomaly detection for each numeric column
    for num in num_cols[:3]:
        try:
            tr = dispatch_tool(df, "anomalies", {"column": num})
            outlier_count = tr.table.get("outlier_count", 0) if isinstance(tr.table, dict) else 0
            if outlier_count > 0:
                findings.append({"effect": float(outlier_count), "summary": tr.summary,
                                  "tool": "anomalies", "params": {"column": num}})
        except Exception as exc:
            logger.debug("Anomaly sweep error %s: %s", num, exc)

    findings.sort(key=lambda f: f["effect"], reverse=True)
    return findings[:_MAX_INSIGHTS]


async def generate_insights(
    session_id: str,
    profile: DatasetProfile,
    parquet_path: str,
    db: DBSession,
) -> None:
    # Clear existing insights for this session (idempotent re-run)
    db.query(Insight).filter_by(session_id=session_id).delete()
    db.commit()

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as exc:
        logger.error("Could not load parquet for insight generation: %s", exc)
        return

    candidates = _candidate_findings(profile, df)
    if not candidates:
        return

    # One batched LLM call to phrase all findings
    summaries = "\n".join(f"{i+1}. {f['summary']}" for i, f in enumerate(candidates))
    prompt = (
        f"You are a data analyst. Below are raw statistical findings from a survey dataset "
        f'called "{profile.filename}" ({profile.row_count} rows).\n\n'
        f"Raw findings:\n{summaries}\n\n"
        f"For each finding, write a short title (5 words max) and one plain-English sentence summary. "
        f"Respond with a JSON array:\n"
        f'[{{"title": "...", "summary": "..."}}, ...]\n'
        f"Include all {len(candidates)} findings in the same order."
    )

    try:
        raw = await llm.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            use_fallback=False,
            max_tokens=600,
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            # model wrapped array in an object
            parsed = next(iter(parsed.values())) if parsed else []
    except Exception as exc:
        logger.error("Batch insight LLM call failed: %s", exc)
        parsed = [{"title": f"Finding {i+1}", "summary": c["summary"]} for i, c in enumerate(candidates)]

    for rank, (candidate, phrased) in enumerate(zip(candidates, parsed), start=1):
        insight = Insight(
            session_id=session_id,
            rank=rank,
            title=phrased.get("title", f"Finding {rank}"),
            summary=phrased.get("summary", candidate["summary"]),
            supporting_tool_calls=[{"tool": candidate["tool"], "params": candidate["params"]}],
        )
        db.add(insight)

    db.commit()
