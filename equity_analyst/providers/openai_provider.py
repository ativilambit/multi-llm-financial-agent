from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Any, cast

from openai import AsyncOpenAI

from equity_analyst.prompt_export import maybe_export_prompt
from equity_analyst.providers.base import LLMProvider
from equity_analyst.types import ProviderResponse, ProviderUsage

logger = logging.getLogger(__name__)

# Combined with the static prefix hash for cache routing (see OpenAI prompt caching guide).
EQUITY_FANOUT_PROMPT_CACHE_KEY = "equity_analyst_fanout_v1"


def _serialize_responses_request_body_for_debug(
    *,
    instructions: str | None,
    input_payload: str | list[dict[str, Any]],
    tools: list[Any] | None,
) -> str:
    """Stable string of user-visible Responses API body fields (for prefix / cache debugging)."""
    lines: list[str] = []
    if instructions:
        lines.append("instructions_head:" + json.dumps(instructions[:240], ensure_ascii=False))
    if isinstance(input_payload, str):
        body_input_repr = f"input: {input_payload}"
    else:
        body_input_repr = "input_json:" + json.dumps(input_payload, ensure_ascii=False, separators=(",", ":"))
    lines.append(body_input_repr)
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
    found: list[int] = []
    for attr in ("cached_tokens", "input_tokens_cached"):
        v = getattr(usage_obj, attr, None)
        if v is not None:
            found.append(int(v))
    itd = getattr(usage_obj, "input_tokens_details", None)
    if itd is not None:
        v = getattr(itd, "cached_tokens", None)
        if v is not None:
            found.append(int(v))
    ptd = getattr(usage_obj, "prompt_tokens_details", None)
    if ptd is not None:
        v = getattr(ptd, "cached_tokens", None)
        if v is not None:
            found.append(int(v))
    return max(found) if found else None


def _responses_input_for_prompt_cache(
    *,
    cacheable_prefix: str,
    user_message_for_cache: str,
) -> tuple[list[dict[str, Any]], str]:
    """Return user-only message list and static instructions for Responses API prompt caching.

    OpenAI's cache uses an exact prefix match on the serialized prompt. The API's
    top-level ``instructions`` field holds the static system persona; ``input`` carries
    only the variable user turn. That keeps the cached prefix identical across
    iterations when the user body grows (e.g. verifier follow-ups), which a single
    ``input`` array ``[system, user]`` did not reliably satisfy for cache hits.
    """
    user_items: list[dict[str, Any]] = [
        {"type": "message", "role": "user", "content": user_message_for_cache},
    ]
    return user_items, cacheable_prefix


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
        cacheable_prefix: str | None = None,
        user_message_for_cache: str | None = None,
    ) -> ProviderResponse:
        start = time.perf_counter()
        use_structured_cache_split = (
            cacheable_prefix is not None
            and user_message_for_cache is not None
            and cacheable_prefix != ""
        )
        instructions: str | None = None
        input_payload: str | list[dict[str, Any]]
        if use_structured_cache_split:
            assert cacheable_prefix is not None and user_message_for_cache is not None
            expected = f"{cacheable_prefix}\n\n{user_message_for_cache}"
            if prompt != expected:
                logger.warning(
                    "OpenAI prompt/cache split mismatch (using structured input from cache fields); "
                    "prompt_len=%s expected_len=%s",
                    len(prompt),
                    len(expected),
                )
            user_items, instructions = _responses_input_for_prompt_cache(
                cacheable_prefix=cacheable_prefix,
                user_message_for_cache=user_message_for_cache,
            )
            input_payload = user_items
        else:
            input_payload = prompt

        create_kwargs: dict[str, Any] = {
            "model": self._model,
            "input": input_payload,
            "stream": True,
        }
        if use_structured_cache_split:
            create_kwargs["instructions"] = instructions
            create_kwargs["prompt_cache_key"] = EQUITY_FANOUT_PROMPT_CACHE_KEY
            create_kwargs["prompt_cache_retention"] = "24h"
        if max_output_tokens is not None:
            create_kwargs["max_output_tokens"] = max_output_tokens
        if enable_web_search:
            create_kwargs["tools"] = cast(Any, [{"type": "web_search"}])
        logger.debug(
            "OpenAI request shape model=%s web_search=%s structured_cache=%s prompt_chars=%s tool_count=%s",
            self._model,
            enable_web_search,
            use_structured_cache_split,
            len(prompt),
            len(create_kwargs.get("tools", []) or []),
        )
        if logger.isEnabledFor(logging.DEBUG):
            if use_structured_cache_split:
                assert user_message_for_cache is not None
                logger.debug(
                    "OpenAI user block head model=%s chars=500 text=%r",
                    self._model,
                    user_message_for_cache[:500],
                )
            body_str = _serialize_responses_request_body_for_debug(
                instructions=instructions,
                input_payload=input_payload,
                tools=create_kwargs.get("tools"),
            )
            instr_stable = (
                hashlib.sha256((instructions or "").encode("utf-8")).hexdigest()[:16]
                if instructions
                else "none"
            )
            body_hash = hashlib.sha256(body_str.encode("utf-8")).hexdigest()[:16]
            logger.debug(
                "OpenAI request prefix model=%s prefix_chars=200 prefix=%r hash=%s "
                "instructions_sha16=%s len=%s",
                self._model,
                body_str[:200],
                body_hash,
                instr_stable,
                len(body_str),
            )
        exp_system = (instructions or "") if use_structured_cache_split else ""
        exp_user = (
            user_message_for_cache
            if use_structured_cache_split and user_message_for_cache is not None
            else (
                input_payload
                if isinstance(input_payload, str)
                else json.dumps(input_payload, ensure_ascii=False)
            )
        )
        await maybe_export_prompt(
            provider=self.name,
            model=self._model,
            system=exp_system,
            user=exp_user,
            config={
                "model": self._model,
                "max_output_tokens": max_output_tokens,
                "web_search": enable_web_search,
                "stream": True,
                "structured_prompt_cache_split": use_structured_cache_split,
                "prompt_cache_key": EQUITY_FANOUT_PROMPT_CACHE_KEY if use_structured_cache_split else None,
            },
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
