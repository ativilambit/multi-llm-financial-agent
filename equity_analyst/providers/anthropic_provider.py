from __future__ import annotations

import os
import time
from typing import Any, cast

import anthropic

from equity_analyst.providers.base import LLMProvider
from equity_analyst.types import ProviderResponse, ProviderUsage


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, *, model: str = "claude-sonnet-4-6", client: Any | None = None) -> None:
        self._model = model
        self._client = client or anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    async def generate(self, prompt: str, *, enable_web_search: bool = True) -> ProviderResponse:
        start = time.perf_counter()
        create_kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }
        if enable_web_search:
            create_kwargs["tools"] = cast(Any, [{"type": "web_search_20260209", "name": "web_search"}])

        msg = await self._client.messages.create(**create_kwargs)

        text_parts: list[str] = []
        for block in msg.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(block.text)
        text = "".join(text_parts).strip()

        usage = ProviderUsage(
            input_tokens=getattr(msg.usage, "input_tokens", None),
            output_tokens=getattr(msg.usage, "output_tokens", None),
            total_tokens=None,
        )
        latency_s = time.perf_counter() - start
        return ProviderResponse(
            provider_name=self.name,
            model=self._model,
            text=text,
            usage=usage,
            latency_s=latency_s,
            raw=msg,
        )

