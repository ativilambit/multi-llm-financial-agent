from __future__ import annotations

import logging
import os
from dataclasses import replace

from google import genai
from google.genai import types

from equity_analyst.prompt_parts import _load_prompt_file
from equity_analyst.retry import async_retry_call
from equity_analyst.types import ProviderResponse

logger = logging.getLogger(__name__)

_PROMPT_BASENAME = "provider_summarize_system.md"


def summarize_system_prompt() -> str:
    return _load_prompt_file(_PROMPT_BASENAME)


def _estimate_tokens(s: str) -> int:
    return max(1, len(s) // 4)


def _total_body_tokens(healthy: dict[str, ProviderResponse]) -> int:
    return sum(_estimate_tokens(r.text) for r in healthy.values())


def _shrink_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    mark = "\n\n...[truncated before summarizer input budget]...\n\n"
    inner = max_chars - len(mark)
    if inner < 80:
        return text[: max(40, max_chars - 40)] + "\n...(truncated)...\n"
    head = inner // 2
    tail = inner - head
    return text[:head] + mark + text[-tail:]


async def _generate_summary(
    *,
    user_message: str,
    model: str,
    max_output_tokens: int,
    client: genai.Client | None = None,
    retry_max_attempts: int = 3,
    retry_base_delay_s: float = 2.0,
) -> str:
    owned = client is None
    c = client or genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    try:
        config = types.GenerateContentConfig(
            system_instruction=summarize_system_prompt(),
            max_output_tokens=max_output_tokens,
        )

        async def _call() -> str:
            msg = await c.aio.models.generate_content(
                model=model,
                contents=user_message,
                config=config,
            )
            return (msg.text or "").strip()

        return await async_retry_call(
            _call,
            provider="gemini_summarizer",
            max_attempts=retry_max_attempts,
            base_delay_s=retry_base_delay_s,
        )
    finally:
        if owned:
            await c.aio.aclose()


async def summarize_provider_body_if_needed(
    *,
    text: str,
    provider_name: str,
    symbol: str | None,
    threshold: int,
    model: str,
    max_output_tokens: int,
    max_input_tokens: int,
    client: genai.Client | None = None,
    retry_max_attempts: int = 3,
    retry_base_delay_s: float = 2.0,
) -> str:
    est = _estimate_tokens(text)
    if est < threshold:
        return text

    sym = symbol or "unknown"
    budget_chars = max(256, max_input_tokens * 4 - 512)
    body = _shrink_text(text, budget_chars) if est > max_input_tokens else text
    user_message = (
        f"### Context\n"
        f"- Equity symbol: {sym}\n"
        f"- Source provider: {provider_name}\n\n"
        f"### Provider report body\n\n"
        f"{body}"
    )
    try:
        out = await _generate_summary(
            user_message=user_message,
            model=model,
            max_output_tokens=max_output_tokens,
            client=client,
            retry_max_attempts=retry_max_attempts,
            retry_base_delay_s=retry_base_delay_s,
        )
    except Exception as exc:
        logger.warning(
            "provider summarization failed provider=%s error=%s; using original body",
            provider_name,
            type(exc).__name__,
            exc_info=True,
        )
        return text
    if not out:
        logger.warning(
            "provider summarization returned empty provider=%s; using original body",
            provider_name,
        )
        return text
    return out


async def maybe_summarize_healthy_for_synthesis(
    *,
    healthy: dict[str, ProviderResponse],
    summarize_oversized_providers: bool,
    summarize_threshold_input_tokens: int,
    target_total_tokens: int | None,
    oversized_summarize_model: str,
    oversized_summarize_max_output_tokens: int,
    oversized_summarize_max_input_tokens: int,
    symbol: str | None,
    client: genai.Client | None = None,
    retry_max_attempts: int = 3,
    retry_base_delay_s: float = 2.0,
) -> tuple[dict[str, ProviderResponse], bool]:
    if not summarize_oversized_providers:
        return healthy, False

    total = _total_body_tokens(healthy)
    any_per_provider_large = any(
        _estimate_tokens(r.text) >= summarize_threshold_input_tokens for r in healthy.values()
    )
    aggregate_over = target_total_tokens is not None and total > target_total_tokens
    if not any_per_provider_large and not aggregate_over:
        return healthy, False

    shared_client: genai.Client | None = client
    owned_client = False
    if shared_client is None:
        shared_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        owned_client = True

    out: dict[str, ProviderResponse] = dict(healthy)
    summarized_any = False
    try:
        max_passes = max(8, len(out) * 4)
        for _ in range(max_passes):
            total = _total_body_tokens(out)
            overs = [
                n
                for n, r in out.items()
                if _estimate_tokens(r.text) >= summarize_threshold_input_tokens
            ]
            under_target = target_total_tokens is None or total <= target_total_tokens
            if under_target and not overs:
                break

            if target_total_tokens is not None and total > target_total_tokens:
                name = max(out.items(), key=lambda kv: _estimate_tokens(kv[1].text))[0]
            elif overs:
                name = max(overs, key=lambda n: _estimate_tokens(out[n].text))
            else:
                break

            resp = out[name]
            before = _estimate_tokens(resp.text)
            need_aggregate = target_total_tokens is not None and total > target_total_tokens
            effective_threshold = (
                0 if (need_aggregate and before < summarize_threshold_input_tokens) else summarize_threshold_input_tokens
            )
            before_text = resp.text
            new_text = await summarize_provider_body_if_needed(
                text=resp.text,
                provider_name=name,
                symbol=symbol,
                threshold=effective_threshold,
                model=oversized_summarize_model,
                max_output_tokens=oversized_summarize_max_output_tokens,
                max_input_tokens=oversized_summarize_max_input_tokens,
                client=shared_client,
                retry_max_attempts=retry_max_attempts,
                retry_base_delay_s=retry_base_delay_s,
            )
            if new_text == before_text:
                break
            after = _estimate_tokens(new_text)
            logger.info(
                "synthesizer: summarized provider=%s est_tokens=%s → %s",
                name,
                before,
                after,
            )
            out[name] = replace(resp, text=new_text)
            summarized_any = True
    finally:
        if owned_client and shared_client is not None:
            await shared_client.aio.aclose()

    return out, summarized_any
