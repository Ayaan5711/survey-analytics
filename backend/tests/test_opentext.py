from __future__ import annotations
import json
import pytest
from unittest.mock import AsyncMock, patch
from app.opentext.analyzer import analyze_open_text


@pytest.mark.asyncio
async def test_analyze_open_text_parses_and_caches(tmp_path):
    cache = tmp_path / "ot.json"
    fake = json.dumps({
        "themes": [{"theme": "pricing", "mentions": 5}, {"theme": "service", "mentions": 3}],
        "sentiment": {"positive": 50, "neutral": 30, "negative": 20},
    })
    mock = AsyncMock(return_value=fake)
    with patch("app.opentext.analyzer.llm.chat_completion", new=mock):
        result = await analyze_open_text(["good price", "bad service", "ok"], "Comments", cache)
    assert result["themes"][0]["theme"] == "pricing"
    assert result["sentiment"]["positive"] == 50
    assert result["n"] == 3
    assert cache.exists()

    # Second call must hit the cache (no LLM call).
    with patch("app.opentext.analyzer.llm.chat_completion", new=AsyncMock()) as m2:
        again = await analyze_open_text(["x"], "Comments", cache)
    m2.assert_not_awaited()
    assert again["themes"][0]["theme"] == "pricing"


@pytest.mark.asyncio
async def test_analyze_open_text_empty_input(tmp_path):
    result = await analyze_open_text(["", None, "  "], "Comments", tmp_path / "e.json")
    assert result["n"] == 0
    assert result["themes"] == []
