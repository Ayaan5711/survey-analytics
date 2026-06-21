from __future__ import annotations
import json
import logging
from pathlib import Path
from app.llm.client import llm

logger = logging.getLogger(__name__)

_MAX_SAMPLE = 120          # cap responses sent to the LLM (cost control)
_MAX_LEN = 300             # truncate very long individual responses


def _prompt(column: str, samples: list[str]) -> str:
    joined = "\n".join(f"- {s[:_MAX_LEN]}" for s in samples)
    return (
        f'Analyse these free-text survey responses from the column "{column}".\n\n'
        f"Responses:\n{joined}\n\n"
        "Identify the recurring themes and the overall sentiment split. "
        "Respond with JSON only:\n"
        '{"themes": [{"theme": "<short label>", "mentions": <int>}], '
        '"sentiment": {"positive": <pct>, "neutral": <pct>, "negative": <pct>}}\n'
        "List at most 6 themes ordered by frequency. Sentiment percentages must sum to ~100."
    )


def _empty() -> dict:
    return {"themes": [], "sentiment": {"positive": 0, "neutral": 0, "negative": 0}, "n": 0}


async def analyze_open_text(values: list[str], column: str, cache_path: Path) -> dict:
    """Extract themes + sentiment for one open-text column. Cached on disk so the
    LLM is only called once per column (per dataset version)."""
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    samples = [v.strip() for v in values if isinstance(v, str) and v.strip()][:_MAX_SAMPLE]
    if not samples:
        result = _empty()
    else:
        try:
            raw = await llm.chat_completion(
                messages=[{"role": "user", "content": _prompt(column, samples)}],
                use_fallback=True,            # cheap model is plenty for this
                max_tokens=500,
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            parsed = json.loads(raw)
            result = {
                "themes": parsed.get("themes", [])[:6],
                "sentiment": parsed.get("sentiment", {"positive": 0, "neutral": 0, "negative": 0}),
                "n": len(samples),
            }
        except Exception as exc:
            logger.error("Open-text analysis failed for %s: %s", column, exc)
            result = _empty()

    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(result), encoding="utf-8")
    except OSError as exc:
        logger.debug("Could not cache open-text result: %s", exc)
    return result
