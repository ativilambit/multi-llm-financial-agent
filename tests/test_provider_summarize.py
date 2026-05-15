from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from equity_analyst.provider_summarize import (
    maybe_summarize_healthy_for_synthesis,
    summarize_provider_body_if_needed,
)
from equity_analyst.providers.gemini_provider import summarizer_thinking_budget_candidates
from equity_analyst.types import ProviderResponse, ProviderUsage

REPO_ROOT = Path(__file__).resolve().parent.parent
_PROVIDER_SUMMARIZE_PROMPT = REPO_ROOT / "prompts" / "provider_summarize_system.md"


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


def test_provider_summarize_system_prompt_retention_guidance() -> None:
    raw = _PROVIDER_SUMMARIZE_PROMPT.read_text(encoding="utf-8")
    lower = raw.lower()
    assert "50%" in raw or "half" in lower
    assert "minimum" in lower
    assert "table" in lower
    assert "probabilit" in lower


@pytest.mark.asyncio
async def test_oversized_body_calls_summarizer_once(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from equity_analyst import provider_summarize as ps

    calls: list[object] = []

    async def fake_gen(**_kwargs: object) -> str:
        calls.append(True)
        return "[compressed]\n\n" + "w" * (5000 * 4)

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
    assert out["openai"].text == "[compressed]\n\n" + "w" * (5000 * 4)
    assert out["openai"].model == "m"
    assert "pre_synthesis_summarize: condensed body_from=openai" in caplog.text
    assert "target=~" in caplog.text
    assert "retention=" in caplog.text
    assert "summarizer_api=gemini model=gemini-3-flash-preview" in caplog.text


@pytest.mark.asyncio
async def test_summarizer_raises_output_cap_and_injects_target_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Oversized summarizer must pass a max_output high enough for ~50% retention and echo a numeric target."""
    from equity_analyst import provider_summarize as ps

    captured: dict[str, object] = {}

    async def fake_gen(
        *,
        user_message: str,
        model: str,
        max_output_tokens: int,
        thinking_budgets: list[int],
        client: object | None = None,
        retry_max_attempts: int = 3,
        retry_base_delay_s: float = 2.0,
    ) -> str:
        captured["max_output_tokens"] = max_output_tokens
        captured["user_message"] = user_message
        captured["thinking_budgets"] = thinking_budgets
        # ~50% of input token estimate: 4000 est-tokens → 16_000 chars
        out_chars = (8000 // 2) * 4
        return "S" * out_chars

    monkeypatch.setattr(ps, "_generate_summary", fake_gen)

    text = "W" * (8000 * 4)
    out = await summarize_provider_body_if_needed(
        text=text,
        provider_name="openai",
        symbol="SE",
        threshold=0,
        model="gemini-3-flash-preview",
        max_output_tokens=8192,
        max_input_tokens=100_000,
        client=_DummyGeminiClient(),
    )
    before_est = 8000
    assert int(captured["max_output_tokens"]) >= int(before_est * 0.45)
    um = str(captured["user_message"])
    assert "Minimum length" in um
    assert "at least ~3400" in um
    assert "Max completion tokens reserved" in um
    after_est = max(1, len(out) // 4)
    retention = after_est / before_est
    assert 0.40 <= retention <= 0.60


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
        oversized_summarize_min_retention=0.0,
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


def test_summarizer_thinking_budget_gemini_3_flash_preview_starts_at_8192() -> None:
    assert summarizer_thinking_budget_candidates(model="gemini-3-flash-preview")[0] == 8192


def test_summarizer_thinking_budget_gemini_3_pro_starts_at_min_not_flash_default() -> None:
    seq = summarizer_thinking_budget_candidates(model="gemini-3.1-pro-preview")
    assert seq[0] == 1024
    assert 8192 in seq


@pytest.mark.asyncio
async def test_gemini_flash_summarizer_first_generate_uses_thinking_budget_8192() -> None:
    captured: list[int] = []

    class Models:
        async def generate_content(self, **_kw: object) -> object:
            cfg = _kw["config"]
            captured.append(int(cfg.thinking_config.thinking_budget))
            return SimpleNamespace(text="z" * (6000 * 4))

    class Aio:
        def __init__(self) -> None:
            self.models = Models()

        async def aclose(self) -> None:
            return None

    class Client:
        def __init__(self) -> None:
            self.aio = Aio()

    gc = Client()
    big = "a" * (9000 * 4)
    out = await summarize_provider_body_if_needed(
        text=big,
        provider_name="anthropic",
        symbol="X",
        threshold=0,
        model="gemini-3-flash-preview",
        max_output_tokens=8192,
        max_input_tokens=100_000,
        client=gc,
        oversized_summarize_min_retention=0.35,
    )
    assert captured[0] == 8192
    assert len(out) > 10_000


@pytest.mark.asyncio
async def test_retention_below_floor_triggers_floor_strict_retry(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def fake_sleep(_s: float) -> None:
        return None

    monkeypatch.setattr("equity_analyst.retry.asyncio.sleep", fake_sleep)

    calls = [0]

    class Models:
        async def generate_content(self, **_kw: object) -> object:
            calls[0] += 1
            if calls[0] == 1:
                return SimpleNamespace(text="brief")
            return SimpleNamespace(text="W" * (5000 * 4))

    class Aio:
        def __init__(self) -> None:
            self.models = Models()

        async def aclose(self) -> None:
            return None

    class Client:
        def __init__(self) -> None:
            self.aio = Aio()

    gc = Client()
    big = "b" * (9000 * 4)
    with caplog.at_level(logging.INFO, logger="equity_analyst.provider_summarize"):
        out = await summarize_provider_body_if_needed(
            text=big,
            provider_name="anthropic",
            symbol="X",
            threshold=0,
            model="gemini-3-flash-preview",
            max_output_tokens=8192,
            max_input_tokens=100_000,
            client=gc,
        )
    assert calls[0] >= 2
    assert len(out) > 1000
    assert "first_pass" in caplog.text
    assert "floor-strict" in caplog.text
    assert "retry succeeded" in caplog.text
