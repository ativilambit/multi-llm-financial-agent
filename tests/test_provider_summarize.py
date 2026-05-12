from __future__ import annotations

import logging
from unittest.mock import AsyncMock

import httpx
import pytest

from equity_analyst.provider_summarize import (
    maybe_summarize_healthy_for_synthesis,
    summarize_provider_body_if_needed,
)
from equity_analyst.types import ProviderResponse, ProviderUsage


class _DummyAio:
    async def aclose(self) -> None:
        return None


class _DummyGeminiClient:
    """Avoid constructing google.genai.Client in unit tests (API key not required)."""

    aio = _DummyAio()


class _SummarizeGenContentModels:
    def __init__(self) -> None:
        self.calls = 0

    async def generate_content(self, **_kwargs: object) -> object:
        from google.genai import errors as ge

        self.calls += 1
        if self.calls <= 2:
            req = httpx.Request("POST", "https://generativelanguage.googleapis.com/x")
            resp = httpx.Response(429, request=req)
            raise ge.ClientError(429, {"error": {}}, resp)
        return type("_M", (), {"text": "third_try_ok"})()


class _SummarizeGenContentAio:
    def __init__(self) -> None:
        self.models = _SummarizeGenContentModels()

    async def aclose(self) -> None:
        return None


class _SummarizeGeminiClient:
    def __init__(self) -> None:
        self.aio = _SummarizeGenContentAio()


def _resp(*, name: str, text: str) -> ProviderResponse:
    return ProviderResponse(
        provider_name=name,
        model="m",
        text=text,
        usage=ProviderUsage(),
        raw=None,
    )


@pytest.mark.asyncio
async def test_oversized_body_calls_summarizer_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from equity_analyst import provider_summarize as ps

    calls: list[object] = []

    async def fake_gen(**_kwargs: object) -> str:
        calls.append(True)
        return "[compressed]\n\nshort"

    monkeypatch.setattr(ps, "_generate_summary", fake_gen)

    big = "word " * 9000  # len//4 >> 8000
    healthy = {"openai": _resp(name="openai", text=big)}
    with caplog.at_level(logging.INFO, logger="equity_analyst.provider_summarize"):
        out, did = await maybe_summarize_healthy_for_synthesis(
            healthy=healthy,
            summarize_oversized_providers=True,
            summarize_threshold_input_tokens=8000,
            target_total_tokens=None,
            oversized_summarize_provider="gemini",
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
    assert "pre_synthesis_summarize: condensed provider=openai" in caplog.text
    assert "summarizer=gemini model=gemini-3-flash-preview" in caplog.text


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
        oversized_summarize_provider="gemini",
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
        oversized_summarize_provider="gemini",
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
async def test_summarize_retries_genai_429_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_sleep(_s: float) -> None:
        return None

    monkeypatch.setattr("equity_analyst.retry.asyncio.sleep", fake_sleep)

    gc = _SummarizeGeminiClient()
    big = "word " * 9000
    out = await summarize_provider_body_if_needed(
        text=big,
        provider_name="openai",
        symbol="MNDY",
        threshold=0,
        model="gemini-test",
        max_output_tokens=1024,
        max_input_tokens=100_000,
        client=gc,
        retry_max_attempts=4,
        retry_base_delay_s=0.01,
    )
    assert out == "third_try_ok"
    assert gc.aio.models.calls == 3


class _Always429Models:
    async def generate_content(self, **_kwargs: object) -> None:
        from google.genai import errors as ge

        req = httpx.Request("POST", "https://generativelanguage.googleapis.com/x")
        raise ge.ClientError(429, {"error": {}}, httpx.Response(429, request=req))


class _Always429Aio:
    def __init__(self) -> None:
        self.models = _Always429Models()

    async def aclose(self) -> None:
        return None


class _Always429Client:
    def __init__(self) -> None:
        self.aio = _Always429Aio()


@pytest.mark.asyncio
async def test_summarize_all_retries_fail_falls_back_to_raw(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_sleep(_s: float) -> None:
        return None

    monkeypatch.setattr("equity_analyst.retry.asyncio.sleep", fake_sleep)

    big = "KEEP_ME " * 9000
    out = await summarize_provider_body_if_needed(
        text=big,
        provider_name="openai",
        symbol="MNDY",
        threshold=0,
        model="gemini-test",
        max_output_tokens=1024,
        max_input_tokens=100_000,
        client=_Always429Client(),
        retry_max_attempts=2,
        retry_base_delay_s=0.01,
    )
    assert out == big


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
        oversized_summarize_provider="gemini",
        oversized_summarize_model="gemini-3-flash-preview",
        oversized_summarize_max_output_tokens=8192,
        oversized_summarize_max_input_tokens=100_000,
        symbol="MNDY",
        client=_DummyGeminiClient(),
    )
    assert "RETAIN_ME" in out["openai"].text
    assert out["openai"].text == big
