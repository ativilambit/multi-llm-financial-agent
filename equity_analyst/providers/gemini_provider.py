from __future__ import annotations

import logging
import os
import re
import time
from typing import Any

from google import genai
from google.genai import types

from equity_analyst.gemini_cache import (
    GeminiCacheIndex,
    gemini_cache_tools_signature,
    prefix_sha256,
)
from equity_analyst.providers.base import LLMProvider
from equity_analyst.types import ProviderResponse, ProviderUsage

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-3.1-pro-preview"
_FLASH_MIN_CACHE_TOKENS = 1024
_PRO_MIN_CACHE_TOKENS = 4096


def gemini_explicit_cache_min_input_tokens(model: str) -> int:
    """Minimum cached input tokens per Gemini explicit caching docs (by model family)."""
    m = model.lower()
    if "flash" in m:
        return _FLASH_MIN_CACHE_TOKENS
    if "pro" in m:
        return _PRO_MIN_CACHE_TOKENS
    return _PRO_MIN_CACHE_TOKENS


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(
        self,
        *,
        model: str = DEFAULT_GEMINI_MODEL,
        client: Any | None = None,
        cache_index: GeminiCacheIndex | None = None,
        cache_ttl_s: int = 3600,
    ) -> None:
        self._model = model
        self._client = client or genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self._cache_index = cache_index
        self._cache_ttl_s = cache_ttl_s

    async def _count_cache_prefix_tokens(self, cacheable_prefix: str) -> int:
        try:
            resp = await self._client.aio.models.count_tokens(
                model=self._model,
                contents=cacheable_prefix,
            )
            return int(resp.total_tokens or 0)
        except Exception as e:
            logger.warning(
                "Gemini count_tokens failed; skipping cache feasibility check error=%s",
                type(e).__name__,
            )
            return 0

    async def generate(
        self,
        prompt: str,
        *,
        enable_web_search: bool = True,
        max_output_tokens: int | None = None,
        cacheable_prefix: str | None = None,
        user_message_for_cache: str | None = None,
        thinking_budget: int | None = None,
    ) -> ProviderResponse:
        start = time.perf_counter()
        use_cache = (
            cacheable_prefix is not None
            and self._cache_index is not None
            and cacheable_prefix != ""
        )
        user_turn: str | None = None
        if use_cache:
            if user_message_for_cache is not None:
                user_turn = user_message_for_cache
            else:
                sep = f"{cacheable_prefix}\n\n"
                user_turn = prompt[len(sep) :] if prompt.startswith(sep) else None
            if not user_turn:
                use_cache = False

        min_toks = gemini_explicit_cache_min_input_tokens(self._model)
        prefix_tokens = 0
        if use_cache:
            assert cacheable_prefix is not None
            rough_estimate = len(cacheable_prefix) // 4
            if rough_estimate < min_toks:
                logger.info(
                    "Gemini cache skipped (rough estimate below min) prefix_chars=%s rough_tokens=%s min=%s",
                    len(cacheable_prefix),
                    rough_estimate,
                    min_toks,
                )
                use_cache = False
            else:
                prefix_tokens = await self._count_cache_prefix_tokens(cacheable_prefix)
                if prefix_tokens < min_toks:
                    logger.info(
                        "Gemini cache skipped (precise count below min) prefix_tokens=%s min=%s",
                        prefix_tokens,
                        min_toks,
                    )
                    use_cache = False

        tools_sig = gemini_cache_tools_signature(enable_web_search)
        gen_cfg: dict[str, Any] = {}
        if max_output_tokens is not None:
            gen_cfg["max_output_tokens"] = max_output_tokens
        if thinking_budget is not None:
            # Gemini 3 shares max_output_tokens with internal "thinking"; set an explicit
            # budget so callers can reserve the completion cap for visible output (JSON, etc.).
            gen_cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=thinking_budget)

        contents: str
        uses_explicit_cache = False

        if use_cache and user_turn is not None and self._cache_index is not None:
            assert cacheable_prefix is not None
            hit = self._cache_index.lookup(cacheable_prefix, self._model, tools_sig)
            if hit:
                gen_cfg["cached_content"] = hit
                uses_explicit_cache = True
                logger.info(
                    "Gemini cache hit name=%s tokens_saved=%s",
                    hit,
                    prefix_tokens,
                )
                contents = user_turn
            else:
                logger.info(
                    "Gemini cache miss creating new entry tokens=%s ttl_s=%s",
                    prefix_tokens,
                    self._cache_ttl_s,
                )
                cache_create: dict[str, Any] = {
                    "system_instruction": cacheable_prefix,
                    "display_name": _cache_display_name(cacheable_prefix, self._model),
                    "ttl": f"{self._cache_ttl_s}s",
                }
                if enable_web_search:
                    cache_create["tools"] = [types.Tool(google_search=types.GoogleSearch())]
                cache = await self._client.aio.caches.create(
                    model=self._model,
                    config=types.CreateCachedContentConfig(**cache_create),
                )
                cname = str(cache.name)
                self._cache_index.store(
                    cacheable_prefix, self._model, cname, self._cache_ttl_s, tools_sig
                )
                gen_cfg["cached_content"] = cname
                uses_explicit_cache = True
                contents = user_turn
        else:
            contents = prompt

        if not uses_explicit_cache and enable_web_search:
            gen_cfg["tools"] = [types.Tool(google_search=types.GoogleSearch())]

        config: types.GenerateContentConfig | None = (
            types.GenerateContentConfig(**gen_cfg) if gen_cfg else None
        )
        logger.debug(
            "Gemini request shape model=%s web_search=%s cached_content=%s prompt_chars=%s contents_chars=%s",
            self._model,
            enable_web_search,
            getattr(config, "cached_content", None) if config is not None else None,
            len(prompt),
            len(contents),
        )
        logger.info("Calling provider %s", self.name)
        msg = await self._client.aio.models.generate_content(
            model=self._model,
            contents=contents,
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


def _cache_display_name(cacheable_prefix: str, model: str) -> str:
    h = prefix_sha256(cacheable_prefix)[:12]
    safe_model = re.sub(r"[^a-zA-Z0-9._-]+", "-", model)[:48]
    return f"equity-{safe_model}-{h}"
