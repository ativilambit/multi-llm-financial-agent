from __future__ import annotations

import hashlib
import logging
import os
import time
from typing import Any

from openai import AsyncOpenAI

from equity_analyst.providers.base import LLMProvider
from equity_analyst.providers.openai_provider import _serialize_responses_request_body_for_debug
from equity_analyst.types import ProviderResponse, ProviderUsage

logger = logging.getLogger(__name__)

XAI_BASE_URL = "https://api.x.ai/v1"


def _prompt_cache_read_tokens(usage_obj: Any) -> int | None:
    """Best-effort read of xAI/Grok prompt cache hits (OpenAI-compatible usage shape)."""
    if usage_obj is None:
        return None
    for attr in ("cached_tokens", "input_tokens_cached"):
        v = getattr(usage_obj, attr, None)
        if v is not None:
            return int(v)
    itd = getattr(usage_obj, "input_tokens_details", None)
    if itd is not None:
        v = getattr(itd, "cached_tokens", None)
        if v is not None:
            return int(v)
    ptd = getattr(usage_obj, "prompt_tokens_details", None)
    if ptd is not None:
        v = getattr(ptd, "cached_tokens", None)
        if v is not None:
            return int(v)
    return None


class GrokProvider(LLMProvider):
    name = "grok"

    def __init__(self, *, model: str = "grok-4.3", client: Any | None = None) -> None:
        self._model = model
        self._client = client or AsyncOpenAI(
            api_key=os.environ.get("XAI_API_KEY"),
            base_url=XAI_BASE_URL,
        )

    async def generate(
        self,
        prompt: str,
        *,
        enable_web_search: bool = True,
        max_output_tokens: int | None = None,
    ) -> ProviderResponse:
        start = time.perf_counter()
        create_kwargs: dict[str, Any] = {"model": self._model, "input": prompt}
        if max_output_tokens is not None:
            create_kwargs["max_output_tokens"] = max_output_tokens
        if enable_web_search:
            create_kwargs["tools"] = [{"type": "web_search"}]
        logger.debug(
            "Grok request shape model=%s web_search=%s prompt_chars=%s tool_count=%s",
            self._model,
            enable_web_search,
            len(prompt),
            len(create_kwargs.get("tools", []) or []),
        )
        if logger.isEnabledFor(logging.DEBUG):
            body_str = _serialize_responses_request_body_for_debug(
                input_payload=prompt,
                tools=create_kwargs.get("tools"),
            )
            body_hash = hashlib.sha256(body_str.encode("utf-8")).hexdigest()[:16]
            logger.debug(
                "Grok request prefix model=%s prefix_chars=200 prefix=%r hash=%s len=%s",
                self._model,
                body_str[:200],
                body_hash,
                len(body_str),
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
        cache_read = _prompt_cache_read_tokens(usage_obj)
        if cache_read is not None:
            in_tok = getattr(usage_obj, "input_tokens", None)
            out_tok = getattr(usage_obj, "output_tokens", None)
            logger.info(
                "Grok cache stats cache_read=%s input=%s output=%s latency_s=%.3f model=%s",
                cache_read,
                in_tok,
                out_tok,
                latency_s,
                self._model,
            )
        return ProviderResponse(
            provider_name=self.name,
            model=self._model,
            text=text,
            usage=usage,
            latency_s=latency_s,
            raw=resp,
        )
