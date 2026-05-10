from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from equity_analyst.provider_summarize import (
    maybe_summarize_healthy_for_synthesis,
)
from equity_analyst.types import ProviderResponse, ProviderUsage


class _DummyAio:
    async def aclose(self) -> None:
        return None


class _DummyGeminiClient:
    """Avoid constructing google.genai.Client in unit tests (API key not required)."""

    aio = _DummyAio()


def _resp(*, name: str, text: str) -> ProviderResponse:
    return ProviderResponse(
        provider_name=name,
        model="m",
        text=text,
        usage=ProviderUsage(),
        raw=None,
    )


@pytest.mark.asyncio
async def test_oversized_body_calls_summarizer_once(monkeypatch: pytest.MonkeyPatch) -> None:
    from equity_analyst import provider_summarize as ps

    calls: list[object] = []

    async def fake_gen(**_kwargs: object) -> str:
        calls.append(True)
        return "[compressed]\n\nshort"

    monkeypatch.setattr(ps, "_generate_summary", fake_gen)

    big = "word " * 9000  # len//4 >> 8000
    healthy = {"openai": _resp(name="openai", text=big)}
    out, did = await maybe_summarize_healthy_for_synthesis(
        healthy=healthy,
        summarize_oversized_providers=True,
        summarize_threshold_input_tokens=8000,
        target_total_tokens=None,
        oversized_summarize_model="gemini-3-flash-preview",
        oversized_summarize_max_output_tokens=8192,
        oversized_summarize_max_input_tokens=100_000,
        symbol="MNDY",
        client=_DummyGeminiClient(),
    )
    assert did is True
    assert len(calls) == 1
    assert out["openai"].text == "[compressed]\n\nshort"
    assert out["openai"].model == "m"


@pytest.mark.asyncio
async def test_under_threshold_skips_api(monkeypatch: pytest.MonkeyPatch) -> None:
    from equity_analyst import provider_summarize as ps

    mock = AsyncMock()
    monkeypatch.setattr(ps, "_generate_summary", mock)

    healthy = {"openai": _resp(name="openai", text="small")}
    out, did = await maybe_summarize_healthy_for_synthesis(
        healthy=healthy,
        summarize_oversized_providers=True,
        summarize_threshold_input_tokens=8000,
        target_total_tokens=None,
        oversized_summarize_model="gemini-3-flash-preview",
        oversized_summarize_max_output_tokens=8192,
        oversized_summarize_max_input_tokens=100_000,
        symbol="MNDY",
        client=None,
    )
    assert did is False
    mock.assert_not_called()
    assert out["openai"].text == "small"


@pytest.mark.asyncio
async def test_summarize_disabled_skips_api(monkeypatch: pytest.MonkeyPatch) -> None:
    from equity_analyst import provider_summarize as ps

    mock = AsyncMock()
    monkeypatch.setattr(ps, "_generate_summary", mock)

    big = "word " * 9000
    healthy = {"openai": _resp(name="openai", text=big)}
    out, did = await maybe_summarize_healthy_for_synthesis(
        healthy=healthy,
        summarize_oversized_providers=False,
        summarize_threshold_input_tokens=8000,
        target_total_tokens=None,
        oversized_summarize_model="gemini-3-flash-preview",
        oversized_summarize_max_output_tokens=8192,
        oversized_summarize_max_input_tokens=100_000,
        symbol="MNDY",
        client=None,
    )
    assert did is False
    mock.assert_not_called()
    assert out is healthy


@pytest.mark.asyncio
async def test_summarize_exception_preserves_original(monkeypatch: pytest.MonkeyPatch) -> None:
    from equity_analyst import provider_summarize as ps

    async def boom(**_kwargs: object) -> str:
        raise RuntimeError("api down")

    monkeypatch.setattr(ps, "_generate_summary", boom)

    big = "RETAIN_ME " * 9000
    healthy = {"openai": _resp(name="openai", text=big)}
    out, _did = await maybe_summarize_healthy_for_synthesis(
        healthy=healthy,
        summarize_oversized_providers=True,
        summarize_threshold_input_tokens=8000,
        target_total_tokens=None,
        oversized_summarize_model="gemini-3-flash-preview",
        oversized_summarize_max_output_tokens=8192,
        oversized_summarize_max_input_tokens=100_000,
        symbol="MNDY",
        client=_DummyGeminiClient(),
    )
    assert "RETAIN_ME" in out["openai"].text
    assert out["openai"].text == big
