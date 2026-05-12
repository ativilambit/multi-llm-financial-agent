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
from equity_analyst.prompt_export import maybe_export_prompt
from equity_analyst.prompt_parts import EQUITY_ANALYST_SYSTEM_PROMPT
from equity_analyst.providers.base import LLMProvider
from equity_analyst.types import ProviderResponse, ProviderUsage

logger = logging.getLogger(__name__)

DEFAULT_GEMINI_MODEL = "gemini-3.1-pro-preview"
_FLASH_MIN_CACHE_TOKENS = 1024
_PRO_MIN_CACHE_TOKENS = 4096

# Gemini 3+ thinking-only models reject thinking_budget=0 ("Budget 0 is invalid").
_GEMINI_MIN_THINKING_BUDGET_ENV = "GEMINI_MIN_THINKING_BUDGET"
_DEFAULT_MIN_THINKING_BUDGET_FOR_THINKING_ONLY = 1024
_THINKING_INVALID_ESCALATION = 8192
_GEMINI_FLASH_SUMMARIZER_THINKING_BUDGET_ENV = "GEMINI_FLASH_SUMMARIZER_THINKING_BUDGET"
_DEFAULT_FLASH_SUMMARIZER_THINKING_BUDGET = 8192


def gemini_model_requires_nonzero_thinking_budget(model: str) -> bool:
    """True for model ids that only run in thinking mode (cannot use thinking_budget=0)."""
    return "gemini-3" in model.lower()


def gemini_min_thinking_budget_for_thinking_only_models() -> int:
    """Minimum thinking budget when the API forbids zero (override via GEMINI_MIN_THINKING_BUDGET)."""
    raw = os.environ.get(_GEMINI_MIN_THINKING_BUDGET_ENV, str(_DEFAULT_MIN_THINKING_BUDGET_FOR_THINKING_ONLY))
    try:
        n = int(raw)
    except ValueError:
        n = _DEFAULT_MIN_THINKING_BUDGET_FOR_THINKING_ONLY
    return max(1, n)


def summarizer_thinking_budget_candidates(*, model: str) -> list[int]:
    """Ordered thinking budgets for pre-synthesis Gemini summarizer (``requested=0`` semantics).

    Gemini 3 Flash preview models default to a **large** first thinking budget so the model
    can plan a long visible summary; other Gemini 3 models keep a smaller first attempt.
    """
    m = model.lower()
    if not gemini_model_requires_nonzero_thinking_budget(model):
        seq = thinking_budget_candidates(model=model, requested=0)
        out_i: list[int] = []
        for x in seq:
            if x is None:
                continue
            if not out_i or out_i[-1] != x:
                out_i.append(int(x))
        return out_i

    d = gemini_min_thinking_budget_for_thinking_only_models()
    esc = _THINKING_INVALID_ESCALATION
    if "flash-preview" in m:
        raw = os.environ.get(
            _GEMINI_FLASH_SUMMARIZER_THINKING_BUDGET_ENV,
            str(_DEFAULT_FLASH_SUMMARIZER_THINKING_BUDGET),
        )
        try:
            first = int(str(raw).strip())
        except ValueError:
            first = _DEFAULT_FLASH_SUMMARIZER_THINKING_BUDGET
        first = max(1, first)
        seq_i = [first, d, esc]
    else:
        seq_i = [d, esc]
    out: list[int] = []
    for x in seq_i:
        if not out or out[-1] != x:
            out.append(x)
    return out


def summarizer_retry_thinking_budget_candidates(*, model: str) -> list[int]:
    """Thinking budgets for the single retention-floor retry (prefer large planning budget)."""
    if not gemini_model_requires_nonzero_thinking_budget(model):
        return summarizer_thinking_budget_candidates(model=model)
    d = gemini_min_thinking_budget_for_thinking_only_models()
    esc = _THINKING_INVALID_ESCALATION
    seq_i = [esc, d]
    out: list[int] = []
    for x in seq_i:
        if not out or out[-1] != x:
            out.append(x)
    return out


def thinking_budget_candidates(*, model: str, requested: int | None) -> list[int | None]:
    """Ordered thinking budgets to try for ``generate`` (None = omit ``thinking_config``)."""
    if requested is None:
        return [None]
    if requested != 0:
        return [requested]
    if gemini_model_requires_nonzero_thinking_budget(model):
        d = gemini_min_thinking_budget_for_thinking_only_models()
        if d >= _THINKING_INVALID_ESCALATION:
            return [d]
        return [d, _THINKING_INVALID_ESCALATION]
    seq = [0, gemini_min_thinking_budget_for_thinking_only_models(), _THINKING_INVALID_ESCALATION]
    out: list[int | None] = []
    for x in seq:
        if out and out[-1] == x:
            continue
        out.append(x)
    return out


def gemini_thinking_budget_invalid_client_error(exc: BaseException) -> bool:
    try:
        from google.genai import errors as ge

        if not isinstance(exc, ge.ClientError):
            return False
        if int(getattr(exc, "code", 0) or 0) != 400:
            return False
    except Exception:
        return False
    msg = str(exc).lower()
    return "budget 0" in msg or "thinking mode" in msg


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

        thinking_seq = thinking_budget_candidates(model=self._model, requested=thinking_budget)
        msg: Any | None = None
        for attempt_i, tb in enumerate(thinking_seq):
            trial_cfg = dict(gen_cfg)
            if tb is not None:
                trial_cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=tb)
            else:
                trial_cfg.pop("thinking_config", None)
            config: types.GenerateContentConfig | None = (
                types.GenerateContentConfig(**trial_cfg) if trial_cfg else None
            )
            logger.debug(
                "Gemini request shape model=%s web_search=%s cached_content=%s "
                "prompt_chars=%s contents_chars=%s thinking_budget=%s attempt=%s",
                self._model,
                enable_web_search,
                getattr(config, "cached_content", None) if config is not None else None,
                len(prompt),
                len(contents),
                tb,
                attempt_i,
            )
            if uses_explicit_cache and user_turn is not None and cacheable_prefix is not None:
                exp_sys = cacheable_prefix
                exp_user = user_turn
            else:
                sep = f"{EQUITY_ANALYST_SYSTEM_PROMPT}\n\n"
                if prompt.startswith(sep):
                    exp_sys = EQUITY_ANALYST_SYSTEM_PROMPT
                    exp_user = prompt[len(sep) :]
                else:
                    exp_sys = ""
                    exp_user = contents
            exp_cfg: dict[str, Any] = {
                "model": self._model,
                "max_output_tokens": max_output_tokens,
                "thinking_budget": tb,
                "web_search": enable_web_search,
                "explicit_content_cache": uses_explicit_cache,
                "thinking_attempt_index": attempt_i,
            }
            if trial_cfg.get("cached_content") is not None:
                exp_cfg["cached_content"] = str(trial_cfg.get("cached_content"))
            await maybe_export_prompt(
                provider=self.name,
                model=self._model,
                system=exp_sys,
                user=exp_user,
                config=exp_cfg,
            )
            logger.info("Calling provider %s", self.name)
            try:
                msg = await self._client.aio.models.generate_content(
                    model=self._model,
                    contents=contents,
                    config=config,
                )
            except Exception as exc:
                last = attempt_i == len(thinking_seq) - 1
                if not gemini_thinking_budget_invalid_client_error(exc) or last:
                    raise
                logger.warning(
                    "Gemini generate_content rejected thinking_budget=%s; will retry (%s/%s) detail=%s",
                    tb,
                    attempt_i + 1,
                    len(thinking_seq) - 1,
                    exc,
                )
                continue
            if attempt_i > 0:
                logger.info(
                    "Gemini generate succeeded after thinking_budget retries final_budget=%s",
                    tb,
                )
            break
        assert msg is not None

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
