from __future__ import annotations

import hashlib
import logging
import os
import time
from typing import Any, cast

from openai import AsyncOpenAI

from equity_analyst.providers.base import LLMProvider
from equity_analyst.types import ProviderResponse, ProviderUsage

logger = logging.getLogger(__name__)


def _serialize_responses_request_body_for_debug(*, input_text: str, tools: list[Any] | None) -> str:
    """Stable string of user-visible Responses API body fields (for prefix / cache debugging)."""
    lines: list[str] = [f"input: {input_text}"]
    if tools:
        names: list[str] = []
        for t in tools:
            if isinstance(t, dict):
                names.append(str(t.get("type", t)))
            else:
                names.append(str(getattr(t, "type", repr(t))))
        lines.append("tools: " + ", ".join(names))
    return "\n".join(lines)


def _prompt_cache_read_tokens(usage_obj: Any) -> int | None:
    """Best-effort read of provider-reported prompt cache hits from a usage object."""
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


def _text_from_response_output(resp: Any) -> str:
    text_parts: list[str] = []
    for item in getattr(resp, "output", []) or []:
        if getattr(item, "type", None) == "message":
            for c in getattr(item, "content", []) or []:
                if getattr(c, "type", None) in {"output_text", "text"} and getattr(c, "text", None):
                    text_parts.append(str(c.text))
    return "\n".join([t for t in text_parts if t]).strip()


async def _consume_responses_stream(stream: Any) -> tuple[str, Any | None]:
    """Read an AsyncStream of ResponseStreamEvent; return (text, final Response or None)."""
    text_chunks: list[str] = []
    final: Any | None = None
    async for event in stream:
        et = getattr(event, "type", None)
        if et == "response.output_text.delta":
            delta = getattr(event, "delta", None)
            if delta:
                text_chunks.append(str(delta))
        elif et == "response.completed":
            final = getattr(event, "response", None)
    text = "".join(text_chunks).strip()
    if not text and final is not None:
        text = _text_from_response_output(final)
    return text, final


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, *, model: str = "gpt-5.5", client: Any | None = None) -> None:
        self._model = model
        self._client = client or AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    async def generate(
        self,
        prompt: str,
        *,
        enable_web_search: bool = True,
        max_output_tokens: int | None = None,
    ) -> ProviderResponse:
        start = time.perf_counter()
        create_kwargs: dict[str, Any] = {"model": self._model, "input": prompt, "stream": True}
        if max_output_tokens is not None:
            create_kwargs["max_output_tokens"] = max_output_tokens
        if enable_web_search:
            create_kwargs["tools"] = cast(Any, [{"type": "web_search"}])
        logger.debug(
            "OpenAI request shape model=%s web_search=%s prompt_chars=%s tool_count=%s",
            self._model,
            enable_web_search,
            len(prompt),
            len(create_kwargs.get("tools", []) or []),
        )
        if logger.isEnabledFor(logging.DEBUG):
            body_str = _serialize_responses_request_body_for_debug(
                input_text=prompt,
                tools=create_kwargs.get("tools"),
            )
            body_hash = hashlib.sha256(body_str.encode("utf-8")).hexdigest()[:16]
            logger.debug(
                "OpenAI request prefix model=%s prefix_chars=200 prefix=%r hash=%s len=%s",
                self._model,
                body_str[:200],
                body_hash,
                len(body_str),
            )
        logger.info("Calling provider %s", self.name)
        stream = await self._client.responses.create(**create_kwargs)
        text, resp = await _consume_responses_stream(stream)

        usage_obj = getattr(resp, "usage", None) if resp is not None else None
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
                "OpenAI cache stats cache_read=%s input=%s output=%s latency_s=%.3f model=%s",
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
