from __future__ import annotations

import logging
from dataclasses import dataclass

from equity_analyst.provider_runtime import partition_provider_responses
from equity_analyst.providers.base import LLMProvider
from equity_analyst.retry import async_retry_call
from equity_analyst.types import ProviderResponse, ProviderUsage

logger = logging.getLogger(__name__)

SYNTHESIS_SYSTEM_PROMPT = """You are a synthesis agent. You will be given multiple LLM providers' raw answers to the same 13-section equity/options prompt.

Your job:
- Compare the answers and flag key disagreements.
- Identify likely hallucinations or claims that are unverifiable/unsupported; explicitly label them.
- Produce a balanced consensus answer that keeps the original structure: ALL 13 numbered sections must be present and numbered 1..13.
- Provide explicit confidence levels (e.g., High/Medium/Low) for each numbered section, and an overall confidence.
- After the full synthesis, on its own line, print exactly: OVERALL_CONFIDENCE: <a number from 0.0 to 1.0>
- Prefer grounded claims with sources/citations; if sources are missing or conflicting, say so.
- Do not drop sections even if data is unavailable; state what you can/cannot verify.
"""


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
        synthesizer_max_input_tokens: int = 20_000,
        retry_max_attempts: int = 3,
        retry_base_delay_s: float = 2.0,
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
        fixed_intro = (
            f"{SYNTHESIS_SYSTEM_PROMPT}\n\n"
            f"### Original user prompt\n{original_prompt}\n\n"
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
        return SynthesisResult(response=resp, prompt=synthesis_prompt)
