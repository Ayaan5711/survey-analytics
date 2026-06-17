import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.llm.client import AzureOpenAIClient


@pytest.fixture
def mock_client(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
    return AzureOpenAIClient()


@pytest.mark.asyncio
async def test_chat_completion_returns_string(mock_client):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Analysis result"

    with patch.object(mock_client._client.chat.completions, "create",
                      new=AsyncMock(return_value=mock_response)):
        result = await mock_client.chat_completion(
            messages=[{"role": "user", "content": "Summarise this data"}],
            use_fallback=False,
        )
    assert result == "Analysis result"


@pytest.mark.asyncio
async def test_chat_completion_uses_fallback_model(mock_client):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "ok"

    calls = []
    async def fake_create(**kwargs):
        calls.append(kwargs.get("model"))
        return mock_response

    with patch.object(mock_client._client.chat.completions, "create", side_effect=fake_create):
        await mock_client.chat_completion(
            messages=[{"role": "user", "content": "test"}],
            use_fallback=True,
        )
    assert calls[0] == mock_client.deployment_fallback
