from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from equity_analyst.prompt_parts import _load_prompt_file
from equity_analyst.provider_runtime import partition_provider_responses
from equity_analyst.provider_summarize import maybe_summarize_healthy_for_synthesis
from equity_analyst.providers.anthropic_provider import AnthropicProvider
from equity_analyst.providers.base import LLMProvider
from equity_analyst.retry import async_retry_call
from equity_analyst.types import ProviderResponse, ProviderUsage

logger = logging.getLogger(__name__)


def detect_max_tokens_truncation(raw: Any) -> tuple[bool, str | None]:
    """Best-effort detection of provider-side ``MAX_TOKENS``-style truncation.

    Returns ``(was_truncated, finish_reason_label)``. The label is the raw
    provider value (e.g. ``"MAX_TOKENS"``, ``"max_tokens"``,
    ``"max_output_tokens"``) when truncation is detected, else ``None``.

    Looks for Gemini ``candidates[0].finish_reason``, Anthropic ``stop_reason``,
    and OpenAI/Grok Responses ``incomplete_details.reason``. Returns
    ``(False, None)`` for unrecognised shapes so callers can ignore safely.
    """
    if raw is None:
        return False, None

    candidates = getattr(raw, "candidates", None)
    if candidates:
        first = candidates[0]
        fr = getattr(first, "finish_reason", None)
        label: str | None = None if fr is None else (getattr(fr, "name", None) or str(fr))
        if label and "MAX_TOKENS" in label.upper():
            return True, label

    stop_reason = getattr(raw, "stop_reason", None)
    if isinstance(stop_reason, str) and stop_reason.lower() == "max_tokens":
        return True, stop_reason

    incomplete = getattr(raw, "incomplete_details", None)
    if incomplete is not None:
        reason = getattr(incomplete, "reason", None)
        if isinstance(reason, str) and reason.lower() in {
            "max_output_tokens",
            "max_tokens",
        }:
            return True, reason

    return False, None


def __getattr__(name: str) -> str:
    if name == "SYNTHESIS_SYSTEM_PROMPT":
        return _load_prompt_file("synthesizer_system.md")
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def _estimate_tokens(s: str) -> int:
    return max(1, len(s) // 4)


def _shrink_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    mark = "\n\n...[truncated for synthesizer input budget]...\n\n"
    inner = max_chars - len(mark)
    if inner < 80:
        return text[: max(40, max_chars - 40)] + "\n...(truncated)...\n"
    head = inner // 2
    tail = inner - head
    return text[:head] + mark + text[-tail:]


def _failure_note_line(failed: dict[str, ProviderResponse]) -> str:
    parts: list[str] = []
    for name, resp in failed.items():
        reason = resp.model if resp.model.startswith("error:") else "empty_response"
        parts.append(f"{name} ({reason})")
    return (
        "Note: provider(s) "
        + ", ".join(parts)
        + " failed and are excluded from synthesis."
    )


def _trim_healthy_bodies_to_token_budget(
    *,
    healthy: dict[str, ProviderResponse],
    fixed_intro: str,
    failure_note: str,
    max_input_tokens: int,
) -> tuple[dict[str, str], int, int]:
    bodies: dict[str, str] = {k: v.text for k, v in healthy.items()}

    def assemble(trimmed: dict[str, str]) -> str:
        blocks: list[str] = []
        for name, resp in healthy.items():
            blocks.append(f"## Provider: {name}\nModel: {resp.model}\n\n{trimmed[name]}\n")
        body = "\n".join(blocks)
        prefix = f"{failure_note}\n\n" if failure_note else ""
        return prefix + fixed_intro + "\n" + body

    before = _estimate_tokens(assemble(bodies))
    if before <= max_input_tokens:
        return bodies, before, before

    scale = 1.0
    trimmed = dict(bodies)
    after = before
    for _ in range(48):
        trimmed = {
            k: _shrink_text(bodies[k], max(80, int(len(bodies[k]) * scale))) for k in bodies
        }
        after = _estimate_tokens(assemble(trimmed))
        if after <= max_input_tokens:
            logger.info("synthesizer: trimmed inputs from %s to %s tokens", before, after)
            return trimmed, before, after
        scale *= 0.82

    logger.warning(
        "synthesizer: could not fully meet input token budget (estimated %s vs max %s)",
        after,
        max_input_tokens,
    )
    return trimmed, before, after


@dataclass(frozen=True)
class SynthesisResult:
    response: ProviderResponse
    prompt: str


def format_synthesis_artifact_markdown(
    *,
    synthesis: SynthesisResult,
    responses: dict[str, ProviderResponse],
) -> str:
    model = synthesis.response.model
    if not model.startswith("error:"):
        return synthesis.response.text.rstrip() + "\n"
    if model == "error:AllProvidersFailed":
        return synthesis.response.text.rstrip() + "\n"
    parts = [
        "# Synthesis degraded",
        "",
        f"Synthesis stage failed (`{model}`).",
        "",
        synthesis.response.text.strip(),
        "",
        "## Per-provider outputs (raw)",
        "",
    ]
    for name, resp in responses.items():
        parts.append(f"### {name} (model `{resp.model}`)\n\n{resp.text}\n")
    return "\n".join(parts).rstrip() + "\n"


class Synthesizer:
    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    async def synthesize(
        self,
        *,
        original_prompt: str,
        responses: dict[str, ProviderResponse],
        enable_web_search: bool = True,
        max_output_tokens: int | None = None,
        synthesizer_max_input_tokens: int = 100_000,
        retry_max_attempts: int = 3,
        retry_base_delay_s: float = 2.0,
        anthropic_force_tool_use: bool = True,
        symbol: str | None = None,
        summarize_oversized_providers: bool = True,
        summarize_threshold_input_tokens: int = 8000,
        oversized_summarize_provider: str = "gemini",
        oversized_summarize_model: str = "gemini-3-flash-preview",
        oversized_summarize_max_output_tokens: int = 8192,
        oversized_summarize_max_input_tokens: int = 100_000,
        refinement_markdown: str | None = None,
    ) -> SynthesisResult:
        healthy, failed = partition_provider_responses(responses)

        if not healthy:
            lines = [
                "# All providers failed",
                "",
                "No LLM synthesis was performed because every provider response was empty or an error.",
                "",
                "## Provider errors",
                "",
            ]
            for name, resp in responses.items():
                lines.append(f"### {name} (model `{resp.model}`)\n")
                lines.append(resp.text.rstrip() + "\n")
            text = "\n".join(lines).rstrip() + "\n"
            return SynthesisResult(
                response=ProviderResponse(
                    provider_name=self._provider.name,
                    model="error:AllProvidersFailed",
                    text=text,
                    usage=ProviderUsage(),
                    latency_s=None,
                    raw=None,
                ),
                prompt="(skipped: zero healthy provider responses)",
            )

        failure_note = _failure_note_line(failed) if failed else ""
        target_body_tokens = max(8_000, synthesizer_max_input_tokens - 3_000)
        healthy, summarized_any = await maybe_summarize_healthy_for_synthesis(
            healthy=healthy,
            summarize_oversized_providers=summarize_oversized_providers,
            summarize_threshold_input_tokens=summarize_threshold_input_tokens,
            target_total_tokens=target_body_tokens,
            oversized_summarize_provider=oversized_summarize_provider,
            oversized_summarize_model=oversized_summarize_model,
            oversized_summarize_max_output_tokens=oversized_summarize_max_output_tokens,
            oversized_summarize_max_input_tokens=oversized_summarize_max_input_tokens,
            symbol=symbol,
            retry_max_attempts=retry_max_attempts,
            retry_base_delay_s=retry_base_delay_s,
        )
        if summarized_any:
            total_after = sum(_estimate_tokens(r.text) for r in healthy.values())
            logger.info(
                "synthesizer: total tokens after summarization=%s target=%s",
                total_after,
                target_body_tokens,
            )
        refine = (refinement_markdown or "").strip()
        refine_block = f"\n\n### Iterative refinement task\n{refine}\n" if refine else ""
        fixed_intro = (
            f"{_load_prompt_file('synthesizer_system.md')}\n\n"
            f"### Original user prompt\n{original_prompt}"
            f"{refine_block}\n\n"
            f"### Provider responses\n\n"
        )
        trimmed_bodies, _est_before, _est_after = _trim_healthy_bodies_to_token_budget(
            healthy=healthy,
            fixed_intro=fixed_intro,
            failure_note=failure_note,
            max_input_tokens=synthesizer_max_input_tokens,
        )
        blocks: list[str] = []
        for name, resp in healthy.items():
            blocks.append(f"## Provider: {name}\nModel: {resp.model}\n\n{trimmed_bodies[name]}\n")
        blocks_joined = "\n".join(blocks)
        prefix = f"{failure_note}\n\n" if failure_note else ""
        synthesis_prompt = prefix + fixed_intro + blocks_joined

        logger.info(
            "Synthesis start provider=%s response_count=%s healthy_count=%s prompt_chars=%s web_search=%s",
            self._provider.name,
            len(responses),
            len(healthy),
            len(synthesis_prompt),
            enable_web_search,
        )

        async def _call() -> ProviderResponse:
            if isinstance(self._provider, AnthropicProvider):
                return await self._provider.generate(
                    synthesis_prompt,
                    enable_web_search=enable_web_search,
                    max_output_tokens=max_output_tokens,
                    force_tool_use=anthropic_force_tool_use,
                )
            return await self._provider.generate(
                synthesis_prompt,
                enable_web_search=enable_web_search,
                max_output_tokens=max_output_tokens,
            )

        resp = await async_retry_call(
            _call,
            provider=self._provider.name,
            max_attempts=retry_max_attempts,
            base_delay_s=retry_base_delay_s,
        )
        logger.info(
            "Synthesis end model=%s latency_s=%s",
            resp.model,
            f"{resp.latency_s:.3f}" if resp.latency_s is not None else "n/a",
        )
        truncated, finish_label = detect_max_tokens_truncation(resp.raw)
        if truncated:
            logger.warning(
                "Synthesizer output truncated by provider cap: provider=%s model=%s "
                "finish_reason=%s output_tokens=%s max_output_tokens=%s — raise "
                "synthesizer_max_output_tokens to recover the full synthesis.",
                self._provider.name,
                resp.model,
                finish_label,
                resp.usage.output_tokens,
                max_output_tokens,
            )
        return SynthesisResult(response=resp, prompt=synthesis_prompt)
