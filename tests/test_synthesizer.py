from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from equity_analyst.config import RunConfig
from equity_analyst.providers.base import LLMProvider
from equity_analyst.synthesizer import (
    SYNTHESIS_SYSTEM_PROMPT,
    Synthesizer,
    detect_max_tokens_truncation,
    provider_finish_reason_label,
)
from equity_analyst.types import ProviderResponse, ProviderUsage


class _DummyAio:
    async def aclose(self) -> None:
        return None


class _DummyGeminiClient:
    aio = _DummyAio()


class _RecordingProvider(LLMProvider):
    name = "recording"

    def __init__(self) -> None:
        self.last_prompt: str | None = None

    async def generate(
        self, prompt: str, *, enable_web_search: bool = True, max_output_tokens: int | None = None
    ) -> ProviderResponse:
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


@pytest.mark.asyncio
async def test_synthesizer_passes_summarize_flag_to_maybe_summarize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, bool] = {}

    async def fake_maybe(*, healthy: object, summarize_oversized_providers: bool, **_: object) -> object:
        captured["summarize_oversized_providers"] = summarize_oversized_providers
        return healthy, False

    monkeypatch.setattr(
        "equity_analyst.synthesizer.maybe_summarize_healthy_for_synthesis",
        fake_maybe,
    )

    p = _RecordingProvider()
    s = Synthesizer(p)
    responses = {
        "openai": ProviderResponse(
            provider_name="openai",
            model="gpt",
            text="ok",
            usage=ProviderUsage(),
            raw=None,
        ),
    }
    await s.synthesize(
        original_prompt="ORIG",
        responses=responses,
        enable_web_search=False,
        summarize_oversized_providers=False,
    )
    assert captured["summarize_oversized_providers"] is False


def test_default_synthesizer_max_input_tokens_is_100k() -> None:
    cfg = RunConfig.model_validate(
        {
            "symbol": "X",
            "today_date": "d",
            "today_session": "s",
            "earnings_date": "e",
            "earnings_timing": "t",
            "target_dates": [],
            "next_trading_day": "n",
            "followup_open_date": "f",
            "providers": ["openai"],
        }
    )
    assert cfg.synthesizer_max_input_tokens == 100_000


@pytest.mark.asyncio
async def test_summarizer_runs_when_aggregate_exceeds_target_even_if_individuals_below_threshold(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from equity_analyst import provider_summarize as ps

    summarized_providers: list[str] = []

    async def fake_gen(*, user_message: str, **_kwargs: object) -> str:
        for key in ("anthropic", "openai", "grok", "gemini"):
            if f"Source provider: {key}" in user_message:
                summarized_providers.append(key)
                break
        return "y" * 12_000

    monkeypatch.setattr(ps, "_generate_summary", fake_gen)
    monkeypatch.setattr(ps.genai, "Client", lambda **_kw: _DummyGeminiClient())
    monkeypatch.setattr(
        "equity_analyst.synthesizer._load_prompt_file",
        lambda name: "brief sys" if name == "synthesizer_system.md" else "",
    )

    p = _RecordingProvider()
    s = Synthesizer(p)
    body = "x" * (6000 * 4)
    responses = {
        "anthropic": ProviderResponse(
            provider_name="anthropic",
            model="claude",
            text=body,
            usage=ProviderUsage(),
            raw=None,
        ),
        "openai": ProviderResponse(
            provider_name="openai",
            model="gpt",
            text=body,
            usage=ProviderUsage(),
            raw=None,
        ),
        "grok": ProviderResponse(
            provider_name="grok",
            model="grok",
            text=body,
            usage=ProviderUsage(),
            raw=None,
        ),
        "gemini": ProviderResponse(
            provider_name="gemini",
            model="gemini",
            text=body,
            usage=ProviderUsage(),
            raw=None,
        ),
    }

    with caplog.at_level(logging.INFO, logger="equity_analyst.synthesizer"):
        await s.synthesize(
            original_prompt="ORIG",
            responses=responses,
            enable_web_search=False,
            synthesizer_max_input_tokens=20_000,
        )

    assert summarized_providers, "Flash summarization should run for aggregate oversize"
    assert summarized_providers[0] == "anthropic"
    assert any("total tokens after summarization=" in r.message for r in caplog.records)
    assert not any("trimmed inputs" in r.message for r in caplog.records)


def test_detect_max_tokens_truncation_gemini_shape() -> None:
    raw = SimpleNamespace(
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="MAX_TOKENS"))],
    )
    truncated, label = detect_max_tokens_truncation(raw)
    assert truncated is True
    assert label == "MAX_TOKENS"


def test_detect_max_tokens_truncation_anthropic_shape() -> None:
    raw = SimpleNamespace(stop_reason="max_tokens")
    truncated, label = detect_max_tokens_truncation(raw)
    assert truncated is True
    assert label == "max_tokens"


def test_detect_max_tokens_truncation_openai_responses_shape() -> None:
    raw = SimpleNamespace(incomplete_details=SimpleNamespace(reason="max_output_tokens"))
    truncated, label = detect_max_tokens_truncation(raw)
    assert truncated is True
    assert label == "max_output_tokens"


def test_detect_max_tokens_truncation_clean_stop() -> None:
    raw = SimpleNamespace(
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"))],
        stop_reason="end_turn",
        incomplete_details=None,
    )
    truncated, label = detect_max_tokens_truncation(raw)
    assert truncated is False
    assert label is None


def test_detect_max_tokens_truncation_handles_none_raw() -> None:
    truncated, label = detect_max_tokens_truncation(None)
    assert truncated is False
    assert label is None


def test_provider_finish_reason_label_gemini_candidate() -> None:
    fr = SimpleNamespace(name="STOP")
    cand = SimpleNamespace(finish_reason=fr)
    raw = SimpleNamespace(candidates=[cand])
    assert provider_finish_reason_label(raw) == "STOP"


def test_provider_finish_reason_label_none_raw() -> None:
    assert provider_finish_reason_label(None) is None


def test_provider_finish_reason_label_anthropic_stop_reason() -> None:
    raw = SimpleNamespace(candidates=None, stop_reason="end_turn")
    assert provider_finish_reason_label(raw) == "end_turn"


class _MaxTokensProvider(LLMProvider):
    name = "gemini"

    def __init__(self) -> None:
        self.last_max_output_tokens: int | None = None

    async def generate(
        self, prompt: str, *, enable_web_search: bool = True, max_output_tokens: int | None = None
    ) -> ProviderResponse:
        self.last_max_output_tokens = max_output_tokens
        raw = SimpleNamespace(
            candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="MAX_TOKENS"))],
        )
        return ProviderResponse(
            provider_name="gemini",
            model="gemini-3.1-pro-preview",
            text="truncated body",
            usage=ProviderUsage(input_tokens=1000, output_tokens=24_000, total_tokens=25_000),
            raw=raw,
        )


@pytest.mark.asyncio
async def test_synthesizer_warns_when_provider_reports_max_tokens(
    caplog: pytest.LogCaptureFixture,
) -> None:
    p = _MaxTokensProvider()
    s = Synthesizer(p)
    responses = {
        "openai": ProviderResponse(
            provider_name="openai",
            model="gpt",
            text="ok",
            usage=ProviderUsage(),
            raw=None,
        ),
    }

    with caplog.at_level(logging.WARNING, logger="equity_analyst.synthesizer"):
        await s.synthesize(
            original_prompt="ORIG",
            responses=responses,
            enable_web_search=False,
            max_output_tokens=24_000,
        )

    truncation_warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING and "Synthesizer output truncated" in r.message
    ]
    assert len(truncation_warnings) == 1
    rendered = truncation_warnings[0].getMessage()
    assert "MAX_TOKENS" in rendered
    assert "24000" in rendered.replace(",", "")
    assert "output_tokens=24000" in rendered.replace(",", "")


@pytest.mark.asyncio
async def test_synthesizer_no_warning_on_clean_stop(
    caplog: pytest.LogCaptureFixture,
) -> None:
    p = _RecordingProvider()
    s = Synthesizer(p)
    responses = {
        "openai": ProviderResponse(
            provider_name="openai",
            model="gpt",
            text="ok",
            usage=ProviderUsage(),
            raw=None,
        ),
    }
    with caplog.at_level(logging.WARNING, logger="equity_analyst.synthesizer"):
        await s.synthesize(
            original_prompt="ORIG",
            responses=responses,
            enable_web_search=False,
            max_output_tokens=24_000,
        )
    assert not any(
        "Synthesizer output truncated" in r.getMessage() for r in caplog.records
    )

