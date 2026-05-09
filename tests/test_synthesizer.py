from __future__ import annotations

import pytest

from equity_analyst.providers.base import LLMProvider
from equity_analyst.synthesizer import SYNTHESIS_SYSTEM_PROMPT, Synthesizer
from equity_analyst.types import ProviderResponse, ProviderUsage


class _RecordingProvider(LLMProvider):
    name = "recording"

    def __init__(self) -> None:
        self.last_prompt: str | None = None

    async def generate(self, prompt: str, *, enable_web_search: bool = True) -> ProviderResponse:
        self.last_prompt = prompt
        return ProviderResponse(
            provider_name="recording",
            model="fake-model",
            text="synth",
            usage=ProviderUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            raw=None,
        )


@pytest.mark.asyncio
async def test_synthesizer_includes_all_provider_outputs_and_instructions() -> None:
    p = _RecordingProvider()
    s = Synthesizer(p)

    responses = {
        "anthropic": ProviderResponse(
            provider_name="anthropic",
            model="claude",
            text="A",
            usage=ProviderUsage(),
            raw=None,
        ),
        "openai": ProviderResponse(
            provider_name="openai",
            model="gpt",
            text="B",
            usage=ProviderUsage(),
            raw=None,
        ),
    }

    await s.synthesize(original_prompt="ORIG", responses=responses, enable_web_search=False)
    assert p.last_prompt is not None
    assert SYNTHESIS_SYSTEM_PROMPT.strip() in p.last_prompt
    assert "Provider: anthropic" in p.last_prompt
    assert "Provider: openai" in p.last_prompt
    assert "disagreements" in p.last_prompt.lower()
    assert "confidence" in p.last_prompt.lower()

