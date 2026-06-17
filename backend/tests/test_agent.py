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

    with patch("app.llm.agent.llm.chat_completion", new=AsyncMock(return_value=synthesis_json)), \
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
