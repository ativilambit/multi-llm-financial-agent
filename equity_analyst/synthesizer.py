from __future__ import annotations

from dataclasses import dataclass

from equity_analyst.providers.base import LLMProvider
from equity_analyst.types import ProviderResponse

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


@dataclass(frozen=True)
class SynthesisResult:
    response: ProviderResponse
    prompt: str


class Synthesizer:
    def __init__(self, provider: LLMProvider) -> None:
        self._provider = provider

    async def synthesize(
        self,
        *,
        original_prompt: str,
        responses: dict[str, ProviderResponse],
        enable_web_search: bool = True,
    ) -> SynthesisResult:
        blocks: list[str] = []
        for name, resp in responses.items():
            blocks.append(f"## Provider: {name}\nModel: {resp.model}\n\n{resp.text}\n")

        synthesis_prompt = (
            f"{SYNTHESIS_SYSTEM_PROMPT}\n\n"
            f"### Original user prompt\n{original_prompt}\n\n"
            f"### Provider responses\n\n" + "\n".join(blocks)
        )
        resp = await self._provider.generate(synthesis_prompt, enable_web_search=enable_web_search)
        return SynthesisResult(response=resp, prompt=synthesis_prompt)

