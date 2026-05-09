from __future__ import annotations

import logging
import os
import time
from typing import Any

from openai import AsyncOpenAI

from equity_analyst.providers.base import LLMProvider
from equity_analyst.types import ProviderResponse, ProviderUsage

logger = logging.getLogger(__name__)

XAI_BASE_URL = "https://api.x.ai/v1"


class GrokProvider(LLMProvider):
    name = "grok"

    def __init__(self, *, model: str = "grok-4.3", client: Any | None = None) -> None:
        self._model = model
        self._client = client or AsyncOpenAI(
            api_key=os.environ.get("XAI_API_KEY"),
            base_url=XAI_BASE_URL,
        )

    async def generate(self, prompt: str, *, enable_web_search: bool = True) -> ProviderResponse:
        start = time.perf_counter()
        create_kwargs: dict[str, Any] = {"model": self._model, "input": prompt}
        if enable_web_search:
            create_kwargs["tools"] = [{"type": "web_search"}]
        logger.debug(
            "Grok request shape model=%s web_search=%s prompt_chars=%s tool_count=%s",
            self._model,
            enable_web_search,
            len(prompt),
            len(create_kwargs.get("tools", []) or []),
        )
        logger.info("Calling provider %s", self.name)
        resp = await self._client.responses.create(**create_kwargs)

        text_parts: list[str] = []
        for item in getattr(resp, "output", []) or []:
            if getattr(item, "type", None) == "message":
                for c in getattr(item, "content", []) or []:
                    if getattr(c, "type", None) in {"output_text", "text"} and getattr(c, "text", None):
                        text_parts.append(str(c.text))
        text = "\n".join([t for t in text_parts if t]).strip()

        usage_obj = getattr(resp, "usage", None)
        usage = ProviderUsage(
            input_tokens=getattr(usage_obj, "input_tokens", None),
            output_tokens=getattr(usage_obj, "output_tokens", None),
            total_tokens=getattr(usage_obj, "total_tokens", None),
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
            raw=resp,
        )
