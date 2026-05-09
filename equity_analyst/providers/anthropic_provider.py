from __future__ import annotations

import logging
import os
import time
from typing import Any, cast

import anthropic

from equity_analyst.providers.base import LLMProvider
from equity_analyst.types import ProviderResponse, ProviderUsage

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, *, model: str = "claude-sonnet-4-6", client: Any | None = None) -> None:
        self._model = model
        self._client = client or anthropic.AsyncAnthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )

    async def generate(
        self,
        prompt: str,
        *,
        enable_web_search: bool = True,
        max_output_tokens: int | None = None,
    ) -> ProviderResponse:
        start = time.perf_counter()
        max_tokens = max_output_tokens if max_output_tokens is not None else 4096
        create_kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if enable_web_search:
            create_kwargs["tools"] = cast(
                Any, [{"type": "web_search_20260209", "name": "web_search"}]
            )
        logger.debug(
            "Anthropic request shape model=%s web_search=%s prompt_chars=%s max_tokens=%s",
            self._model,
            enable_web_search,
            len(prompt),
            max_tokens,
        )
        logger.info("Calling provider %s", self.name)
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
        logger.info(
            "Completed provider %s model=%s latency_s=%.3f",
            self.name,
            self._model,
            latency_s,
        )
        return ProviderResponse(
            provider_name=self.name,
            model=self._model,
            text=text,
            usage=usage,
            latency_s=latency_s,
            raw=msg,
        )
