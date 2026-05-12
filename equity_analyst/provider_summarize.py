from __future__ import annotations

import logging
import os
from dataclasses import replace

from google import genai
from google.genai import types

from equity_analyst.prompt_parts import _load_prompt_file
from equity_analyst.providers.gemini_provider import (
    gemini_thinking_budget_invalid_client_error,
    thinking_budget_candidates,
)
from equity_analyst.retry import async_retry_call
from equity_analyst.types import ProviderResponse

logger = logging.getLogger(__name__)

_PROMPT_BASENAME = "provider_summarize_system.md"


def summarize_system_prompt() -> str:
    return _load_prompt_file(_PROMPT_BASENAME)


def _estimate_tokens(s: str) -> int:
    return max(1, len(s) // 4)


def _target_summary_token_estimate(input_est_tokens: int) -> int:
    """Roughly half of the input estimate (len//4), matching the system prompt."""
    return max(1, int(input_est_tokens) // 2)


_OVERSIZED_SUMMARIZE_MAX_OUTPUT_CEILING = 128_000


def _effective_summarizer_max_output_tokens(*, input_est_tokens: int, configured_max: int) -> int:
    """Ensure the API cap is not below ~55% of the input heuristic so 50% retention is reachable."""
    floor_for_retention = (int(input_est_tokens) * 55 + 99) // 100 + 512
    return min(
        _OVERSIZED_SUMMARIZE_MAX_OUTPUT_CEILING,
        max(int(configured_max), floor_for_retention),
    )


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
        budgets = thinking_budget_candidates(model=model, requested=0)

        async def _call() -> str:
            last_failure: BaseException | None = None
            for attempt_i, tb in enumerate(budgets):
                cfg = types.GenerateContentConfig(
                    system_instruction=summarize_system_prompt(),
                    max_output_tokens=max_output_tokens,
                    thinking_config=types.ThinkingConfig(thinking_budget=tb),
                )
                try:
                    msg = await c.aio.models.generate_content(
                        model=model,
                        contents=user_message,
                        config=cfg,
                    )
                    if attempt_i > 0:
                        logger.info(
                            "gemini_summarizer: succeeded after thinking_budget retries model=%s final=%s",
                            model,
                            tb,
                        )
                    return (msg.text or "").strip()
                except Exception as exc:
                    last_failure = exc
                    if (
                        not gemini_thinking_budget_invalid_client_error(exc)
                        or attempt_i == len(budgets) - 1
                    ):
                        raise
                    logger.warning(
                        "gemini_summarizer: thinking_budget=%s rejected; retrying model=%s detail=%s",
                        tb,
                        model,
                        exc,
                    )
            raise last_failure if last_failure is not None else RuntimeError("thinking budget loop empty")

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
    target_est = _target_summary_token_estimate(est)
    effective_max_out = _effective_summarizer_max_output_tokens(
        input_est_tokens=est,
        configured_max=max_output_tokens,
    )
    user_message = (
        f"### Context\n"
        f"- Equity symbol: {sym}\n"
        f"- Source provider: {provider_name}\n"
        f"- Original body token estimate (len(text)//4 on the full provider body): **{est}**\n"
        f"- **Target summary token estimate: ~{target_est}** (same len(output)//4 heuristic; "
        f"stay within roughly ±20% unless the system prompt says otherwise)\n"
        f"- Max completion tokens reserved for this answer: **{effective_max_out}**\n\n"
        f"### Provider report body\n\n"
        f"{body}"
    )
    try:
        out = await _generate_summary(
            user_message=user_message,
            model=model,
            max_output_tokens=effective_max_out,
            client=client,
            retry_max_attempts=retry_max_attempts,
            retry_base_delay_s=retry_base_delay_s,
        )
    except Exception as exc:
        logger.warning(
            "pre_synthesis_summarize: summarization failed provider=%s error=%s; using original body",
            provider_name,
            type(exc).__name__,
            exc_info=True,
        )
        return text
    if not out:
        logger.warning(
            "pre_synthesis_summarize: summarization returned empty provider=%s; using original body",
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
    oversized_summarize_provider: str,
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
            target = _target_summary_token_estimate(before)
            retention_pct = 100.0 * after / before if before else 0.0
            logger.info(
                "pre_synthesis_summarize: condensed provider=%s est_tokens=%s → %s "
                "(target=~%s, retention=%.1f%%) summarizer=%s model=%s",
                name,
                before,
                after,
                target,
                retention_pct,
                oversized_summarize_provider,
                oversized_summarize_model,
            )
            out[name] = replace(resp, text=new_text)
            summarized_any = True
    finally:
        if owned_client and shared_client is not None:
            await shared_client.aio.aclose()

    return out, summarized_any
