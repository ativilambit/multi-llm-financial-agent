from __future__ import annotations

import logging
import os
from dataclasses import replace

from google import genai
from google.genai import types

from equity_analyst.prompt_export import (
    current_prompt_call_meta,
    logical_prompt_split,
    maybe_export_prompt,
    prompt_call_context,
)
from equity_analyst.prompt_parts import _load_prompt_file
from equity_analyst.providers.base import LLMProvider
from equity_analyst.providers.gemini_provider import (
    gemini_thinking_budget_invalid_client_error,
    summarizer_retry_thinking_budget_candidates,
    summarizer_thinking_budget_candidates,
)
from equity_analyst.retry import async_retry_call
from equity_analyst.types import ProviderResponse

logger = logging.getLogger(__name__)

_PROMPT_BASENAME = "provider_summarize_system.md"

# Minimum visible summary length vs ~50% target (user-turn floor language uses int(target * ratio)).
SUMMARIZER_MIN_LENGTH_FACTOR = 0.85


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


def _oversized_summarize_user_message(
    *,
    sym: str,
    provider_name: str,
    est: int,
    target_est: int,
    effective_max_out: int,
    body: str,
    floor_strict_extra: bool,
) -> str:
    min_tokens = max(1, int(target_est * SUMMARIZER_MIN_LENGTH_FACTOR))
    parts: list[str] = [
        "### Context\n",
        f"- Equity symbol: {sym}\n",
        f"- Source provider: {provider_name}\n",
        f"- Original body token estimate (len(text)//4 on the full provider body): **{est}**\n",
        "- **Minimum length**: produce **at least ~"
        f"{min_tokens} tokens** of summary by the len(text)//4 heuristic on your output. "
        "Do not stop early. If the source has more content than fits in this budget, prioritize **breadth** "
        "(sections, facts, citations) at this length rather than condensing to fewer tokens.\n",
        f"- Reference midpoint (~50% retention of the **{est}** estimated input tokens): **~{target_est} tokens**.\n",
        f"- Max completion tokens reserved for this answer: **{effective_max_out}**\n",
    ]
    if floor_strict_extra:
        parts.append(
            "\n### Compression retry (strict floor)\n\n"
            "Your previous summary was **too short** relative to the minimum length requirement. "
            "Expand aggressively: restore dropped sections, tables (per the system rules on tables), "
            "and quantitative detail until you meet the minimum. Do not stop because the draft "
            "\"feels long enough\".\n\n",
        )
    parts.append("### Provider report body\n\n")
    parts.append(body)
    return "".join(parts)


def _retention_ratio(*, output_text: str, input_est_tokens: int) -> float:
    return _estimate_tokens(output_text) / float(max(1, int(input_est_tokens)))


def _pick_longer_by_retention(*, a: str, b: str, input_est_tokens: int) -> str:
    if _retention_ratio(output_text=b, input_est_tokens=input_est_tokens) > _retention_ratio(
        output_text=a,
        input_est_tokens=input_est_tokens,
    ):
        return b
    return a


async def _generate_summary(
    *,
    user_message: str,
    model: str,
    max_output_tokens: int,
    thinking_budgets: list[int],
    client: genai.Client | None = None,
    retry_max_attempts: int = 3,
    retry_base_delay_s: float = 2.0,
) -> str:
    owned = client is None
    c = client or genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    try:
        async def _call() -> str:
            last_failure: BaseException | None = None
            for attempt_i, tb in enumerate(thinking_budgets):
                cfg = types.GenerateContentConfig(
                    system_instruction=summarize_system_prompt(),
                    max_output_tokens=max_output_tokens,
                    thinking_config=types.ThinkingConfig(thinking_budget=tb),
                )
                prev = current_prompt_call_meta()
                prev_it = prev.iteration if prev is not None else None
                with prompt_call_context(node="pre_synthesis_summarize", iteration=prev_it):
                    await maybe_export_prompt(
                        provider="gemini_sdk",
                        model=model,
                        system=summarize_system_prompt(),
                        user=user_message,
                        config={
                            "model": model,
                            "max_output_tokens": max_output_tokens,
                            "thinking_budget": tb,
                            "web_search": False,
                            "thinking_attempt_index": attempt_i,
                        },
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
                            or attempt_i == len(thinking_budgets) - 1
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


async def _generate_summary_via_llm_provider(
    *,
    provider: LLMProvider,
    user_message: str,
    max_output_tokens: int,
    retry_max_attempts: int = 3,
    retry_base_delay_s: float = 2.0,
) -> str:
    system = summarize_system_prompt()
    prompt = f"{system}\n\n---\n\n{user_message}"

    async def _call() -> str:
        prev = current_prompt_call_meta()
        prev_it = prev.iteration if prev is not None else None
        with prompt_call_context(node="pre_synthesis_summarize", iteration=prev_it), logical_prompt_split(
            system, user_message
        ):
            resp = await provider.generate(
                prompt,
                enable_web_search=False,
                max_output_tokens=max_output_tokens,
            )
            return (resp.text or "").strip()

    return await async_retry_call(
        _call,
        provider=f"oversized_summarize_fallback:{provider.name}",
        max_attempts=retry_max_attempts,
        base_delay_s=retry_base_delay_s,
    )


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
    oversized_summarize_min_retention: float = 0.40,
    oversized_summarize_provider: str = "gemini",
    oversized_summarize_fallback_provider: LLMProvider | None = None,
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
    user_message = _oversized_summarize_user_message(
        sym=sym,
        provider_name=provider_name,
        est=est,
        target_est=target_est,
        effective_max_out=effective_max_out,
        body=body,
        floor_strict_extra=False,
    )
    thinking_budgets = summarizer_thinking_budget_candidates(model=model)
    try:
        out = await _generate_summary(
            user_message=user_message,
            model=model,
            max_output_tokens=effective_max_out,
            thinking_budgets=thinking_budgets,
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

    r1 = _retention_ratio(output_text=out, input_est_tokens=est)
    out_best = out
    r_best = r1
    used_floor_retry = False
    used_fallback = False

    if r1 < oversized_summarize_min_retention:
        retry_budgets = summarizer_retry_thinking_budget_candidates(model=model)
        tb_retry0 = retry_budgets[0]
        logger.info(
            "pre_synthesis_summarize: first_pass provider=%s est_tokens=%s → %s "
            "(target=~%s, retention=%.1f%%); below floor=%.1f%%; retrying with thinking_budget=%s "
            "prompt=floor-strict",
            provider_name,
            est,
            _estimate_tokens(out),
            target_est,
            100.0 * r1,
            100.0 * oversized_summarize_min_retention,
            tb_retry0,
        )
        retry_msg = _oversized_summarize_user_message(
            sym=sym,
            provider_name=provider_name,
            est=est,
            target_est=target_est,
            effective_max_out=effective_max_out,
            body=body,
            floor_strict_extra=True,
        )
        try:
            out2 = await _generate_summary(
                user_message=retry_msg,
                model=model,
                max_output_tokens=effective_max_out,
                thinking_budgets=retry_budgets,
                client=client,
                retry_max_attempts=retry_max_attempts,
                retry_base_delay_s=retry_base_delay_s,
            )
        except Exception as exc:
            logger.warning(
                "pre_synthesis_summarize: floor-strict retry failed provider=%s error=%s; keeping first pass",
                provider_name,
                type(exc).__name__,
                exc_info=True,
            )
            out2 = ""

        if out2:
            used_floor_retry = True
            out_best = _pick_longer_by_retention(a=out, b=out2, input_est_tokens=est)
            r_best = _retention_ratio(output_text=out_best, input_est_tokens=est)
            if r_best >= oversized_summarize_min_retention:
                logger.info(
                    "pre_synthesis_summarize: retry succeeded retention=%.1f%%",
                    100.0 * r_best,
                )
            else:
                logger.warning(
                    "pre_synthesis_summarize: retry below floor; keeping result (retention=%.1f%%)",
                    100.0 * r_best,
                )

    if (
        r_best < oversized_summarize_min_retention
        and oversized_summarize_provider == "gemini"
        and oversized_summarize_fallback_provider is not None
    ):
        fbp = oversized_summarize_fallback_provider
        fb_msg = _oversized_summarize_user_message(
            sym=sym,
            provider_name=provider_name,
            est=est,
            target_est=target_est,
            effective_max_out=effective_max_out,
            body=body,
            floor_strict_extra=True,
        )
        try:
            out3 = await _generate_summary_via_llm_provider(
                provider=fbp,
                user_message=fb_msg,
                max_output_tokens=effective_max_out,
                retry_max_attempts=retry_max_attempts,
                retry_base_delay_s=retry_base_delay_s,
            )
        except Exception as exc:
            logger.warning(
                "pre_synthesis_summarize: fallback summarizer failed provider=%s fallback=%s error=%s",
                provider_name,
                fbp.name,
                type(exc).__name__,
                exc_info=True,
            )
            out3 = ""

        if out3:
            used_fallback = True
            before_pick = out_best
            out_best = _pick_longer_by_retention(a=out_best, b=out3, input_est_tokens=est)
            r_best = _retention_ratio(output_text=out_best, input_est_tokens=est)
            if out_best != before_pick:
                logger.info(
                    "pre_synthesis_summarize: fallback summarizer=%s improved retention to %.1f%%",
                    fbp.name,
                    100.0 * r_best,
                )

    suffix_parts: list[str] = []
    if used_floor_retry and r_best >= oversized_summarize_min_retention:
        suffix_parts.append("after floor-strict retry")
    elif used_floor_retry:
        suffix_parts.append("after floor-strict retry (still below floor)")
    if used_fallback:
        fb2 = oversized_summarize_fallback_provider
        if fb2 is not None:
            suffix_parts.append(f"fallback={fb2.name}")

    suffix = f", {'; '.join(suffix_parts)}" if suffix_parts else ""
    logger.info(
        "pre_synthesis_summarize: condensed provider=%s est_tokens=%s → %s "
        "(target=~%s, retention=%.1f%%)%s summarizer=%s model=%s",
        provider_name,
        est,
        _estimate_tokens(out_best),
        target_est,
        100.0 * r_best,
        suffix,
        oversized_summarize_provider,
        model,
    )

    return out_best


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
    oversized_summarize_min_retention: float = 0.40,
    oversized_summarize_fallback_provider: LLMProvider | None = None,
    symbol: str | None = None,
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
                oversized_summarize_min_retention=oversized_summarize_min_retention,
                oversized_summarize_provider=oversized_summarize_provider,
                oversized_summarize_fallback_provider=oversized_summarize_fallback_provider,
            )
            if new_text == before_text:
                break
            out[name] = replace(resp, text=new_text)
            summarized_any = True
    finally:
        if owned_client and shared_client is not None:
            await shared_client.aio.aclose()

    return out, summarized_any
