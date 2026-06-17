from __future__ import annotations
import os
from openai import AsyncAzureOpenAI
from app.config import settings


class AzureOpenAIClient:
    def __init__(self) -> None:
        # Read from env vars directly to support test fixtures that use monkeypatch.setenv
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT") or settings.azure_openai_endpoint
        api_key = os.getenv("AZURE_OPENAI_API_KEY") or settings.azure_openai_api_key
        api_version = os.getenv("AZURE_OPENAI_API_VERSION") or settings.azure_openai_api_version

        self._client = AsyncAzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )
        self.deployment_primary = os.getenv("AZURE_OPENAI_DEPLOYMENT_PRIMARY") or settings.deployment_primary
        self.deployment_fallback = os.getenv("AZURE_OPENAI_DEPLOYMENT_FALLBACK") or settings.deployment_fallback

    async def chat_completion(
        self,
        messages: list[dict],
        use_fallback: bool = False,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        response_format: dict | None = None,
    ) -> str:
        model = self.deployment_fallback if use_fallback else self.deployment_primary
        kwargs: dict = dict(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if response_format:
            kwargs["response_format"] = response_format
        response = await self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""


# module-level singleton — imported by consumers
llm = AzureOpenAIClient()
