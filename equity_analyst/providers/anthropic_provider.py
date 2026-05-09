from __future__ import annotations

import logging
import os
import time
from typing import Any, cast

import anthropic

from equity_analyst.prompt_parts import EQUITY_ANALYST_SYSTEM_PROMPT, ephemeral_cache_control
from equity_analyst.providers.base import LLMProvider
from equity_analyst.types import ProviderResponse, ProviderUsage

logger = logging.getLogger(__name__)

def split_full_prompt_for_cache(full_prompt: str) -> tuple[str, str]:
    """Split rendered equity ``text`` into (system_preamble, user_message)."""
    prefix = f"{EQUITY_ANALYST_SYSTEM_PROMPT}\n\n"
    if full_prompt.startswith(prefix):
        return EQUITY_ANALYST_SYSTEM_PROMPT, full_prompt[len(prefix) :]
    return "", full_prompt


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    # GA Opus (higher input-token rate limits than Sonnet on typical tiers). IDs:
    # https://docs.anthropic.com/en/docs/about-claude/models/model-ids-and-versions
    def __init__(self, *, model: str = "claude-opus-4-7", client: Any | None = None) -> None:
        self._model = model
        self._client = client or anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    async def generate(
        self,
        prompt: str,
        *,
        enable_web_search: bool = True,
        max_output_tokens: int | None = None,
        prompt_cache_enabled: bool = True,
        user_message_for_cache: str | None = None,
        force_tool_use: bool = True,
    ) -> ProviderResponse:
        start = time.perf_counter()
        max_tokens = max_output_tokens if max_output_tokens is not None else 4096
        create_kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
        }

        use_cache_breakpoints = False
        user_turn: str | None = None
        if prompt_cache_enabled:
            if user_message_for_cache is not None:
                use_cache_breakpoints = True
                user_turn = user_message_for_cache
            else:
                sys_pre, u = split_full_prompt_for_cache(prompt)
                if sys_pre:
                    use_cache_breakpoints = True
                    user_turn = u

        if use_cache_breakpoints and user_turn is not None:
            create_kwargs["system"] = [
                {
                    "type": "text",
                    "text": EQUITY_ANALYST_SYSTEM_PROMPT,
                    "cache_control": ephemeral_cache_control(),
                }
            ]
            create_kwargs["messages"] = [{"role": "user", "content": user_turn}]
        else:
            create_kwargs["messages"] = [{"role": "user", "content": prompt}]

        if enable_web_search:
            tool: dict[str, Any] = {"type": "web_search_20260209", "name": "web_search"}
            if use_cache_breakpoints:
                tool["cache_control"] = ephemeral_cache_control()
            create_kwargs["tools"] = cast(Any, [tool])
            if force_tool_use:
                create_kwargs["tool_choice"] = {"type": "any"}

        logger.debug(
            "Anthropic request shape model=%s web_search=%s prompt_chars=%s max_tokens=%s "
            "prompt_cache=%s cache_breakpoints=%s",
            self._model,
            enable_web_search,
            len(prompt),
            max_tokens,
            prompt_cache_enabled,
            use_cache_breakpoints,
        )
        logger.info("Calling provider %s", self.name)
        # Anthropic requires streaming for requests that may exceed ~10 minutes (e.g. web_search).
        async with self._client.messages.stream(**create_kwargs) as stream:
            await stream.until_done()
            msg = await stream.get_final_message()

        text_parts: list[str] = []
        for block in msg.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(str(getattr(block, "text", "")))
        text = "".join(text_parts).strip()

        resolved_model = str(getattr(msg, "model", self._model))

        usage = ProviderUsage(
            input_tokens=getattr(msg.usage, "input_tokens", None),
            output_tokens=getattr(msg.usage, "output_tokens", None),
            total_tokens=None,
        )
        latency_s = time.perf_counter() - start

        cache_read = int(getattr(msg.usage, "cache_read_input_tokens", 0) or 0)
        cache_creation = int(getattr(msg.usage, "cache_creation_input_tokens", 0) or 0)
        inp = int(getattr(msg.usage, "input_tokens", 0) or 0)
        out = int(getattr(msg.usage, "output_tokens", 0) or 0)
        logger.info(
            "Anthropic cache stats cache_read=%s cache_creation=%s input=%s output=%s",
            cache_read,
            cache_creation,
            inp,
            out,
        )

        logger.info(
            "Completed provider %s model=%s latency_s=%.3f",
            self.name,
            resolved_model,
            latency_s,
        )
        return ProviderResponse(
            provider_name=self.name,
            model=resolved_model,
            text=text,
            usage=usage,
            latency_s=latency_s,
            raw=msg,
        )
