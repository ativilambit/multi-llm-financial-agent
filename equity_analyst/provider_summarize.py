from __future__ import annotations

import logging
import os
from dataclasses import replace

from google import genai
from google.genai import types

from equity_analyst.prompt_parts import _load_prompt_file
from equity_analyst.types import ProviderResponse

logger = logging.getLogger(__name__)

_PROMPT_BASENAME = "provider_summarize_system.md"


def summarize_system_prompt() -> str:
    return _load_prompt_file(_PROMPT_BASENAME)


def _estimate_tokens(s: str) -> int:
    return max(1, len(s) // 4)


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
) -> str:
    owned = client is None
    c = client or genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    try:
        config = types.GenerateContentConfig(
            system_instruction=summarize_system_prompt(),
            max_output_tokens=max_output_tokens,
        )
        msg = await c.aio.models.generate_content(
            model=model,
            contents=user_message,
            config=config,
        )
        return (msg.text or "").strip()
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
    oversized_summarize_model: str,
    oversized_summarize_max_output_tokens: int,
    oversized_summarize_max_input_tokens: int,
    symbol: str | None,
    client: genai.Client | None = None,
) -> dict[str, ProviderResponse]:
    if not summarize_oversized_providers:
        return healthy
    if not any(_estimate_tokens(r.text) >= summarize_threshold_input_tokens for r in healthy.values()):
        return healthy

    shared_client: genai.Client | None = client
    owned_client = False
    if shared_client is None:
        shared_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        owned_client = True

    out: dict[str, ProviderResponse] = {}
    try:
        for name, resp in healthy.items():
            before = _estimate_tokens(resp.text)
            if before < summarize_threshold_input_tokens:
                out[name] = resp
                continue
            new_text = await summarize_provider_body_if_needed(
                text=resp.text,
                provider_name=name,
                symbol=symbol,
                threshold=summarize_threshold_input_tokens,
                model=oversized_summarize_model,
                max_output_tokens=oversized_summarize_max_output_tokens,
                max_input_tokens=oversized_summarize_max_input_tokens,
                client=shared_client,
            )
            after = _estimate_tokens(new_text)
            logger.info(
                "synthesizer: summarized provider=%s est_tokens=%s → %s",
                name,
                before,
                after,
            )
            out[name] = replace(resp, text=new_text)
    finally:
        if owned_client and shared_client is not None:
            await shared_client.aio.aclose()

    return out
