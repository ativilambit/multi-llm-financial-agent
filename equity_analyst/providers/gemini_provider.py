from __future__ import annotations

import logging
import os
import time
from typing import Any

from google import genai
from google.genai import types

from equity_analyst.providers.base import LLMProvider
from equity_analyst.types import ProviderResponse, ProviderUsage

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self, *, model: str = "gemini-2.5-flash", client: Any | None = None) -> None:
        self._model = model
        self._client = client or genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    async def generate(self, prompt: str, *, enable_web_search: bool = True) -> ProviderResponse:
        start = time.perf_counter()
        config: types.GenerateContentConfig | None = None
        if enable_web_search:
            config = types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())],
            )
        logger.debug(
            "Gemini request shape model=%s web_search=%s prompt_chars=%s",
            self._model,
            enable_web_search,
            len(prompt),
        )
        logger.info("Calling provider %s", self.name)
        msg = await self._client.aio.models.generate_content(
            model=self._model,
            contents=prompt,
            config=config,
        )

        text = (msg.text or "").strip()
        um = msg.usage_metadata
        usage = ProviderUsage(
            input_tokens=getattr(um, "prompt_token_count", None) if um else None,
            output_tokens=getattr(um, "candidates_token_count", None) if um else None,
            total_tokens=getattr(um, "total_token_count", None) if um else None,
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
