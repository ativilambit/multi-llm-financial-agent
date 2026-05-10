from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import anthropic
import httpx
import pytest

from equity_analyst.config import RunConfig
from equity_analyst.orchestrator import Orchestrator
from equity_analyst.prompt_parts import EQUITY_ANALYST_SYSTEM_PROMPT
from equity_analyst.providers.base import LLMProvider
from equity_analyst.providers.gemini_provider import GeminiProvider
from equity_analyst.providers.registry import ProviderRegistry
from equity_analyst.types import ProviderResponse, ProviderUsage


def _write_minimal_sa_key_json(path: Path) -> None:
    from test_drive_uploader import _minimal_valid_rsa_pem

    path.write_text(
        json.dumps(
            {
                "type": "service_account",
                "project_id": "test",
                "private_key_id": "x",
                "private_key": _minimal_valid_rsa_pem(),
                "client_email": "sa@test.iam.gserviceaccount.com",
                "client_id": "1",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        ),
        encoding="utf-8",
    )


class _SleepyProvider(LLMProvider):
    def __init__(self, *, name: str, delay_s: float, text: str):
        self.name = name
        self._delay_s = delay_s
        self._text = text

    async def generate(
        self, prompt: str, *, enable_web_search: bool = True, max_output_tokens: int | None = None
    ) -> ProviderResponse:
        await asyncio.sleep(self._delay_s)
        return ProviderResponse(
            provider_name=self.name,
            model="fake",
            text=self._text,
            usage=ProviderUsage(input_tokens=1, output_tokens=2, total_tokens=3),
            raw=None,
        )


@pytest.mark.asyncio
async def test_orchestrator_parallel_and_writes_outputs(
    tmp_path: Path, monkeypatch: Any, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.chdir(tmp_path)

    repo_root = Path(__file__).resolve().parents[1]
    prompt_path = repo_root / "prompts" / "equity_analyst.j2"

    cfg = RunConfig.model_validate(
        {
            "symbol": "MNDY",
            "company_name": None,
            "today_low": 68,
            "today_high": 74,
            "current_price": 73.24,
            "today_date": "Fri May 8, 2026",
            "today_session": "after the market trading window",
            "earnings_date": "Mon May 11 2026",
            "earnings_timing": "early morning et, before the market open",
            "target_dates": ["Mon May 11", "Fri May 15", "Fri May 22", "Fri May 29", "Fri Jun 5"],
            "next_trading_day": "Tues May 12",
            "followup_open_date": "Mon May 18",
            "historical_quarters": 11,
            "short_interest_lookbacks": ["last month"],
            "providers": ["anthropic", "openai", "gemini", "grok"],
            "synthesizer": "gemini",
        }
    )

    def _fake_registry() -> ProviderRegistry:
        reg = ProviderRegistry()
        reg.register("anthropic", lambda **_: _SleepyProvider(name="anthropic", delay_s=0.25, text="A"))
        reg.register("openai", lambda **_: _SleepyProvider(name="openai", delay_s=0.25, text="B"))
        reg.register("gemini", lambda **_: _SleepyProvider(name="gemini", delay_s=0.25, text="G"))
        reg.register("grok", lambda **_: _SleepyProvider(name="grok", delay_s=0.25, text="K"))
        reg.register("gemini", lambda **_: _SleepyProvider(name="gemini", delay_s=0.0, text="SYNTH"))
        return reg

    import equity_analyst.orchestrator as orch_mod

    monkeypatch.setattr(
        orch_mod.ProviderRegistry,
        "default",
        classmethod(lambda cls: _fake_registry()),
    )

    pdf_md_paths: list[Path] = []

    def _capture_pdf(**kwargs: Any) -> None:
        pdf_md_paths.append(kwargs["md_path"])

    monkeypatch.setattr(orch_mod, "maybe_write_pdf_sibling", _capture_pdf)

    orch = Orchestrator(config=cfg, prompt_path=prompt_path)

    started = asyncio.get_event_loop().time()
    with caplog.at_level(logging.INFO, logger="equity_analyst.orchestrator"):
        synthesis, artifacts = await orch.run_async(dry_run=False, enable_web_search=False)
    out_dir = artifacts.output_dir
    elapsed = asyncio.get_event_loop().time() - started

    assert elapsed < 0.50
    assert "SYNTH" in synthesis

    assert out_dir.exists()
    assert (out_dir / "claude.md").exists()
    assert (out_dir / "openai.md").exists()
    assert (out_dir / "gemini.md").exists()
    assert (out_dir / "grok.md").exists()
    assert (out_dir / "synthesis.md").exists()
    assert artifacts.synthesis_file in pdf_md_paths
    assert (out_dir / "run.json").exists()
    assert (out_dir / "agent.log").exists()
    assert any("Run start" in r.message for r in caplog.records)
    assert "Run start" in (out_dir / "agent.log").read_text(encoding="utf-8")


class _RecordingSynthProvider(LLMProvider):
    name = "gemini"

    def __init__(self) -> None:
        self.last_prompt: str | None = None

    async def generate(
        self, prompt: str, *, enable_web_search: bool = True, max_output_tokens: int | None = None
    ) -> ProviderResponse:
        self.last_prompt = prompt
        return ProviderResponse(
            provider_name="gemini",
            model="fake",
            text="SYNTH",
            usage=ProviderUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            raw=None,
        )


class _BadSynthesizer:
    def __init__(self, _provider: LLMProvider) -> None:
        pass

    async def synthesize(self, **_kwargs: Any) -> Any:
        req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        resp = httpx.Response(429, request=req)
        raise anthropic.RateLimitError("rate limited", response=resp, body=None)


class _FlakyAnthropic(LLMProvider):
    name = "anthropic"

    def __init__(self) -> None:
        self._n = 0

    async def generate(
        self, prompt: str, *, enable_web_search: bool = True, max_output_tokens: int | None = None
    ) -> ProviderResponse:
        self._n += 1
        if self._n == 1:
            req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
            resp = httpx.Response(429, request=req)
            raise anthropic.RateLimitError("rl", response=resp, body=None)
        return ProviderResponse(
            provider_name="anthropic",
            model="claude-ok",
            text="A",
            usage=ProviderUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            raw=None,
        )


@pytest.mark.asyncio
async def test_synthesizer_failure_does_not_crash_run(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]
    prompt_path = repo_root / "prompts" / "equity_analyst.j2"

    cfg = RunConfig.model_validate(
        {
            "symbol": "MNDY",
            "company_name": None,
            "today_low": 68,
            "today_high": 74,
            "current_price": 73.24,
            "today_date": "Fri May 8, 2026",
            "today_session": "after the market trading window",
            "earnings_date": "Mon May 11 2026",
            "earnings_timing": "early morning et, before the market open",
            "target_dates": ["Mon May 11"],
            "next_trading_day": "Tues May 12",
            "followup_open_date": "Mon May 18",
            "historical_quarters": 11,
            "short_interest_lookbacks": ["last month"],
            "providers": ["anthropic", "openai"],
            "synthesizer": "gemini",
        }
    )

    def _fake_registry() -> ProviderRegistry:
        reg = ProviderRegistry()
        reg.register("anthropic", lambda **_: _SleepyProvider(name="anthropic", delay_s=0.0, text="A"))
        reg.register("openai", lambda **_: _SleepyProvider(name="openai", delay_s=0.0, text="B"))
        reg.register("gemini", lambda **_: _RecordingSynthProvider())
        return reg

    import equity_analyst.orchestrator as orch_mod

    monkeypatch.setattr(orch_mod.ProviderRegistry, "default", classmethod(lambda cls: _fake_registry()))
    monkeypatch.setattr(orch_mod, "Synthesizer", _BadSynthesizer)

    orch = Orchestrator(config=cfg, prompt_path=prompt_path)
    synthesis, artifacts = await orch.run_async(dry_run=False, enable_web_search=False)
    out_dir = artifacts.output_dir

    assert "# Synthesis degraded" in (out_dir / "synthesis.md").read_text(encoding="utf-8")
    assert "RateLimitError" in synthesis or "failed" in synthesis.lower()
    meta = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
    assert any(e.get("stage") == "synthesis" for e in meta.get("errors", []))


@pytest.mark.asyncio
async def test_error_responses_excluded_from_synthesis(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]
    prompt_path = repo_root / "prompts" / "equity_analyst.j2"

    cfg = RunConfig.model_validate(
        {
            "symbol": "MNDY",
            "company_name": None,
            "today_low": 68,
            "today_high": 74,
            "current_price": 73.24,
            "today_date": "Fri May 8, 2026",
            "today_session": "after the market trading window",
            "earnings_date": "Mon May 11 2026",
            "earnings_timing": "early morning et, before the market open",
            "target_dates": ["Mon May 11"],
            "next_trading_day": "Tues May 12",
            "followup_open_date": "Mon May 18",
            "historical_quarters": 11,
            "short_interest_lookbacks": ["last month"],
            "providers": ["anthropic", "openai"],
            "synthesizer": "gemini",
        }
    )

    class _OpenAIErr(LLMProvider):
        name = "openai"

        async def generate(
            self, prompt: str, *, enable_web_search: bool = True, max_output_tokens: int | None = None
        ) -> ProviderResponse:
            return ProviderResponse(
                provider_name="openai",
                model="error:timeout",
                text="BADBODY_UNIQUE_XYZ",
                usage=ProviderUsage(),
                raw=None,
            )

    rec = _RecordingSynthProvider()

    def _fake_registry() -> ProviderRegistry:
        reg = ProviderRegistry()
        reg.register("anthropic", lambda **_: _SleepyProvider(name="anthropic", delay_s=0.0, text="GOODBODY"))
        reg.register("openai", lambda **_: _OpenAIErr())
        reg.register("gemini", lambda **_: rec)
        return reg

    import equity_analyst.orchestrator as orch_mod

    monkeypatch.setattr(orch_mod.ProviderRegistry, "default", classmethod(lambda cls: _fake_registry()))

    orch = Orchestrator(config=cfg, prompt_path=prompt_path)
    await orch.run_async(dry_run=False, enable_web_search=False)
    assert rec.last_prompt is not None
    assert "BADBODY_UNIQUE_XYZ" not in rec.last_prompt
    assert "GOODBODY" in rec.last_prompt
    assert "excluded from synthesis" in rec.last_prompt


@pytest.mark.asyncio
async def test_all_providers_failed_skips_synthesizer(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]
    prompt_path = repo_root / "prompts" / "equity_analyst.j2"

    class _CountSynth(LLMProvider):
        name = "gemini"
        calls = 0

        async def generate(
            self, prompt: str, *, enable_web_search: bool = True, max_output_tokens: int | None = None
        ) -> ProviderResponse:
            self.calls += 1
            return ProviderResponse(
                provider_name="gemini",
                model="m",
                text="SYN",
                usage=ProviderUsage(),
                raw=None,
            )

    class _AlwaysErr(LLMProvider):
        def __init__(self, name: str) -> None:
            self.name = name

        async def generate(
            self, prompt: str, *, enable_web_search: bool = True, max_output_tokens: int | None = None
        ) -> ProviderResponse:
            return ProviderResponse(
                provider_name=self.name,
                model="error:timeout",
                text="e",
                usage=ProviderUsage(),
                raw=None,
            )

    counter = _CountSynth()

    cfg = RunConfig.model_validate(
        {
            "symbol": "MNDY",
            "company_name": None,
            "today_low": 68,
            "today_high": 74,
            "current_price": 73.24,
            "today_date": "Fri May 8, 2026",
            "today_session": "after the market trading window",
            "earnings_date": "Mon May 11 2026",
            "earnings_timing": "early morning et, before the market open",
            "target_dates": ["Mon May 11"],
            "next_trading_day": "Tues May 12",
            "followup_open_date": "Mon May 18",
            "historical_quarters": 11,
            "short_interest_lookbacks": ["last month"],
            "providers": ["anthropic", "openai"],
            "synthesizer": "gemini",
        }
    )

    def _fake_registry() -> ProviderRegistry:
        reg = ProviderRegistry()
        reg.register("anthropic", lambda **_: _AlwaysErr("anthropic"))
        reg.register("openai", lambda **_: _AlwaysErr("openai"))
        reg.register("gemini", lambda **_: counter)
        return reg

    import equity_analyst.orchestrator as orch_mod

    monkeypatch.setattr(orch_mod.ProviderRegistry, "default", classmethod(lambda cls: _fake_registry()))

    orch = Orchestrator(config=cfg, prompt_path=prompt_path)
    await orch.run_async(dry_run=False, enable_web_search=False)
    assert counter.calls == 0


@pytest.mark.asyncio
async def test_retry_on_rate_limit_then_success(
    tmp_path: Path, monkeypatch: Any, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.chdir(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]
    prompt_path = repo_root / "prompts" / "equity_analyst.j2"

    cfg = RunConfig.model_validate(
        {
            "symbol": "MNDY",
            "company_name": None,
            "today_low": 68,
            "today_high": 74,
            "current_price": 73.24,
            "today_date": "Fri May 8, 2026",
            "today_session": "after the market trading window",
            "earnings_date": "Mon May 11 2026",
            "earnings_timing": "early morning et, before the market open",
            "target_dates": ["Mon May 11"],
            "next_trading_day": "Tues May 12",
            "followup_open_date": "Mon May 18",
            "historical_quarters": 11,
            "short_interest_lookbacks": ["last month"],
            "providers": ["anthropic", "openai"],
            "synthesizer": "gemini",
            "retry_max_attempts": 3,
        }
    )

    async def instant_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("equity_analyst.retry.asyncio.sleep", instant_sleep)

    def _fake_registry() -> ProviderRegistry:
        reg = ProviderRegistry()
        reg.register("anthropic", lambda **_: _FlakyAnthropic())
        reg.register("openai", lambda **_: _SleepyProvider(name="openai", delay_s=0.0, text="B"))
        reg.register("gemini", lambda **_: _SleepyProvider(name="gemini", delay_s=0.0, text="SYNTH"))
        return reg

    import equity_analyst.orchestrator as orch_mod

    monkeypatch.setattr(orch_mod.ProviderRegistry, "default", classmethod(lambda cls: _fake_registry()))

    orch = Orchestrator(config=cfg, prompt_path=prompt_path)
    with caplog.at_level(logging.INFO, logger="equity_analyst.retry"):
        text, _artifacts = await orch.run_async(dry_run=False, enable_web_search=False)
    assert "SYNTH" in text
    assert any("retrying provider=anthropic" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_yaml_model_override_passed_to_provider_registry(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.chdir(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]
    prompt_path = repo_root / "prompts" / "equity_analyst.j2"

    captured: dict[str, Any] = {}

    def _anth_factory(**kwargs: Any) -> LLMProvider:
        captured.update(kwargs)
        return _SleepyProvider(name="anthropic", delay_s=0.0, text="A")

    cfg = RunConfig.model_validate(
        {
            "symbol": "MNDY",
            "company_name": None,
            "today_low": 68,
            "today_high": 74,
            "current_price": 73.24,
            "today_date": "Fri May 8, 2026",
            "today_session": "after the market trading window",
            "earnings_date": "Mon May 11 2026",
            "earnings_timing": "early morning et, before the market open",
            "target_dates": ["Mon May 11"],
            "next_trading_day": "Tues May 12",
            "followup_open_date": "Mon May 18",
            "historical_quarters": 11,
            "short_interest_lookbacks": ["last month"],
            "providers": [{"name": "anthropic", "model": "custom-opus-from-yaml"}, {"name": "openai"}],
            "synthesizer": "gemini",
        }
    )

    def _fake_registry() -> ProviderRegistry:
        reg = ProviderRegistry()
        reg.register("anthropic", _anth_factory)
        reg.register("openai", lambda **_: _SleepyProvider(name="openai", delay_s=0.0, text="B"))
        reg.register("gemini", lambda **_: _SleepyProvider(name="gemini", delay_s=0.0, text="SYNTH"))
        return reg

    import equity_analyst.orchestrator as orch_mod

    monkeypatch.setattr(orch_mod.ProviderRegistry, "default", classmethod(lambda cls: _fake_registry()))

    orch = Orchestrator(config=cfg, prompt_path=prompt_path)
    await orch.run_async(dry_run=False, enable_web_search=False)
    assert captured.get("model") == "custom-opus-from-yaml"


class _RecordingMaxOut(LLMProvider):
    def __init__(self, name: str) -> None:
        self.name = name
        self.last_max_output_tokens: int | None = None

    async def generate(
        self, prompt: str, *, enable_web_search: bool = True, max_output_tokens: int | None = None
    ) -> ProviderResponse:
        self.last_max_output_tokens = max_output_tokens
        return ProviderResponse(
            provider_name=self.name,
            model="fake",
            text="x",
            usage=ProviderUsage(),
            raw=None,
        )


@pytest.mark.asyncio
async def test_synthesizer_gets_separate_max_output_tokens_from_fan_out(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.chdir(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]
    prompt_path = repo_root / "prompts" / "equity_analyst.j2"

    fan_a = _RecordingMaxOut("anthropic")
    fan_b = _RecordingMaxOut("openai")
    synth = _RecordingMaxOut("gemini")

    cfg = RunConfig.model_validate(
        {
            "symbol": "MNDY",
            "company_name": None,
            "today_low": 68,
            "today_high": 74,
            "current_price": 73.24,
            "today_date": "Fri May 8, 2026",
            "today_session": "after the market trading window",
            "earnings_date": "Mon May 11 2026",
            "earnings_timing": "early morning et, before the market open",
            "target_dates": ["Mon May 11"],
            "next_trading_day": "Tues May 12",
            "followup_open_date": "Mon May 18",
            "historical_quarters": 11,
            "short_interest_lookbacks": ["last month"],
            "providers": ["anthropic", "openai"],
            "synthesizer": "gemini",
            "max_output_tokens": 4096,
            "synthesizer_max_output_tokens": 24_000,
        }
    )

    def _fake_registry() -> ProviderRegistry:
        reg = ProviderRegistry()
        reg.register("anthropic", lambda **_: fan_a)
        reg.register("openai", lambda **_: fan_b)
        reg.register("gemini", lambda **_: synth)
        return reg

    import equity_analyst.orchestrator as orch_mod

    monkeypatch.setattr(orch_mod.ProviderRegistry, "default", classmethod(lambda cls: _fake_registry()))

    orch = Orchestrator(config=cfg, prompt_path=prompt_path)
    await orch.run_async(dry_run=False, enable_web_search=False)
    assert fan_a.last_max_output_tokens == 4096
    assert fan_b.last_max_output_tokens == 4096
    assert synth.last_max_output_tokens == 24_000


@pytest.mark.asyncio
async def test_synthesizer_max_output_tokens_cli_override_applied(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.chdir(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]
    prompt_path = repo_root / "prompts" / "equity_analyst.j2"

    synth = _RecordingMaxOut("gemini")

    cfg = RunConfig.model_validate(
        {
            "symbol": "MNDY",
            "company_name": None,
            "today_low": 68,
            "today_high": 74,
            "current_price": 73.24,
            "today_date": "Fri May 8, 2026",
            "today_session": "after the market trading window",
            "earnings_date": "Mon May 11 2026",
            "earnings_timing": "early morning et, before the market open",
            "target_dates": ["Mon May 11"],
            "next_trading_day": "Tues May 12",
            "followup_open_date": "Mon May 18",
            "historical_quarters": 11,
            "short_interest_lookbacks": ["last month"],
            "providers": ["anthropic", "openai"],
            "synthesizer": "gemini",
        }
    ).model_copy(update={"synthesizer_max_output_tokens": 50_000})

    def _fake_registry() -> ProviderRegistry:
        reg = ProviderRegistry()
        reg.register("anthropic", lambda **_: _SleepyProvider(name="anthropic", delay_s=0.0, text="A"))
        reg.register("openai", lambda **_: _SleepyProvider(name="openai", delay_s=0.0, text="B"))
        reg.register("gemini", lambda **_: synth)
        return reg

    import equity_analyst.orchestrator as orch_mod

    monkeypatch.setattr(orch_mod.ProviderRegistry, "default", classmethod(lambda cls: _fake_registry()))

    orch = Orchestrator(config=cfg, prompt_path=prompt_path)
    await orch.run_async(dry_run=False, enable_web_search=False)
    assert synth.last_max_output_tokens == 50_000


@pytest.mark.asyncio
async def test_fan_out_per_provider_max_output_tokens_override(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.chdir(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]
    prompt_path = repo_root / "prompts" / "equity_analyst.j2"

    rec_a = _RecordingMaxOut("anthropic")
    rec_o = _RecordingMaxOut("openai")
    rec_g = _RecordingMaxOut("grok")
    synth = _RecordingMaxOut("gemini")

    cfg = RunConfig.model_validate(
        {
            "symbol": "MNDY",
            "company_name": None,
            "today_low": 68,
            "today_high": 74,
            "current_price": 73.24,
            "today_date": "Fri May 8, 2026",
            "today_session": "after the market trading window",
            "earnings_date": "Mon May 11 2026",
            "earnings_timing": "early morning et, before the market open",
            "target_dates": ["Mon May 11"],
            "next_trading_day": "Tues May 12",
            "followup_open_date": "Mon May 18",
            "historical_quarters": 11,
            "short_interest_lookbacks": ["last month"],
            "providers": [
                {"name": "anthropic", "max_output_tokens": 24_000},
                {"name": "openai"},
                {"name": "grok", "max_output_tokens": 12_000},
            ],
            "synthesizer": "gemini",
            "max_output_tokens": 8888,
        }
    )

    def _fake_registry() -> ProviderRegistry:
        reg = ProviderRegistry()
        reg.register("anthropic", lambda **_: rec_a)
        reg.register("openai", lambda **_: rec_o)
        reg.register("grok", lambda **_: rec_g)
        reg.register("gemini", lambda **_: synth)
        return reg

    import equity_analyst.orchestrator as orch_mod

    monkeypatch.setattr(orch_mod.ProviderRegistry, "default", classmethod(lambda cls: _fake_registry()))

    orch = Orchestrator(config=cfg, prompt_path=prompt_path)
    await orch.run_async(dry_run=False, enable_web_search=False)
    assert rec_a.last_max_output_tokens == 24_000
    assert rec_o.last_max_output_tokens == 8888
    assert rec_g.last_max_output_tokens == 12_000


class _RecordingGeminiFanOut(GeminiProvider):
    """Real GeminiProvider subclass so orchestrator's isinstance check fires; records generate kwargs."""

    name = "gemini"

    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] | None = None

    async def generate(  # type: ignore[override]
        self,
        prompt: str,
        *,
        enable_web_search: bool = True,
        max_output_tokens: int | None = None,
        cacheable_prefix: str | None = None,
        user_message_for_cache: str | None = None,
    ) -> ProviderResponse:
        self.last_kwargs = {
            "prompt": prompt,
            "enable_web_search": enable_web_search,
            "max_output_tokens": max_output_tokens,
            "cacheable_prefix": cacheable_prefix,
            "user_message_for_cache": user_message_for_cache,
        }
        return ProviderResponse(
            provider_name="gemini",
            model="gemini-3-flash-preview",
            text="G",
            usage=ProviderUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            raw=None,
        )


@pytest.mark.asyncio
async def test_fan_out_gemini_receives_cacheable_prefix_when_caching_enabled(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.chdir(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]
    prompt_path = repo_root / "prompts" / "equity_analyst.j2"

    fan_gem = _RecordingGeminiFanOut()
    synth_gem = _RecordingGeminiFanOut()

    cfg = RunConfig.model_validate(
        {
            "symbol": "MNDY",
            "company_name": None,
            "today_low": 68,
            "today_high": 74,
            "current_price": 73.24,
            "today_date": "Fri May 8, 2026",
            "today_session": "after the market trading window",
            "earnings_date": "Mon May 11 2026",
            "earnings_timing": "early morning et, before the market open",
            "target_dates": ["Mon May 11"],
            "next_trading_day": "Tues May 12",
            "followup_open_date": "Mon May 18",
            "historical_quarters": 11,
            "short_interest_lookbacks": ["last month"],
            "providers": [
                {"name": "anthropic"},
                {"name": "openai"},
                {"name": "grok"},
                {"name": "gemini", "model": "gemini-3-flash-preview"},
            ],
            "synthesizer": "gemini",
            "prompt_cache_enabled": True,
        }
    )

    gemini_calls = iter([fan_gem, synth_gem])

    def _fake_registry() -> ProviderRegistry:
        reg = ProviderRegistry()
        reg.register("anthropic", lambda **_: _SleepyProvider(name="anthropic", delay_s=0.0, text="A"))
        reg.register("openai", lambda **_: _SleepyProvider(name="openai", delay_s=0.0, text="B"))
        reg.register("grok", lambda **_: _SleepyProvider(name="grok", delay_s=0.0, text="K"))
        reg.register("gemini", lambda **_: next(gemini_calls))
        return reg

    import equity_analyst.orchestrator as orch_mod

    monkeypatch.setattr(orch_mod.ProviderRegistry, "default", classmethod(lambda cls: _fake_registry()))

    orch = Orchestrator(config=cfg, prompt_path=prompt_path)
    await orch.run_async(dry_run=False, enable_web_search=False)

    assert fan_gem.last_kwargs is not None
    assert fan_gem.last_kwargs["cacheable_prefix"] == EQUITY_ANALYST_SYSTEM_PROMPT
    assert fan_gem.last_kwargs["user_message_for_cache"] is not None
    assert EQUITY_ANALYST_SYSTEM_PROMPT not in fan_gem.last_kwargs["user_message_for_cache"]


@pytest.mark.asyncio
async def test_fan_out_gemini_skips_cacheable_prefix_when_caching_disabled(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.chdir(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]
    prompt_path = repo_root / "prompts" / "equity_analyst.j2"

    fan_gem = _RecordingGeminiFanOut()
    synth_gem = _RecordingGeminiFanOut()

    cfg = RunConfig.model_validate(
        {
            "symbol": "MNDY",
            "company_name": None,
            "today_low": 68,
            "today_high": 74,
            "current_price": 73.24,
            "today_date": "Fri May 8, 2026",
            "today_session": "after the market trading window",
            "earnings_date": "Mon May 11 2026",
            "earnings_timing": "early morning et, before the market open",
            "target_dates": ["Mon May 11"],
            "next_trading_day": "Tues May 12",
            "followup_open_date": "Mon May 18",
            "historical_quarters": 11,
            "short_interest_lookbacks": ["last month"],
            "providers": [
                {"name": "openai"},
                {"name": "gemini", "model": "gemini-3-flash-preview"},
            ],
            "synthesizer": "gemini",
            "prompt_cache_enabled": False,
        }
    )

    gemini_calls = iter([fan_gem, synth_gem])

    def _fake_registry() -> ProviderRegistry:
        reg = ProviderRegistry()
        reg.register("openai", lambda **_: _SleepyProvider(name="openai", delay_s=0.0, text="B"))
        reg.register("gemini", lambda **_: next(gemini_calls))
        return reg

    import equity_analyst.orchestrator as orch_mod

    monkeypatch.setattr(orch_mod.ProviderRegistry, "default", classmethod(lambda cls: _fake_registry()))

    orch = Orchestrator(config=cfg, prompt_path=prompt_path)
    await orch.run_async(dry_run=False, enable_web_search=False)

    assert fan_gem.last_kwargs is not None
    assert fan_gem.last_kwargs["cacheable_prefix"] is None


@pytest.mark.asyncio
async def test_per_provider_request_timeout_override_honored(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.chdir(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]
    prompt_path = repo_root / "prompts" / "equity_analyst.j2"

    wait_for_timeouts: list[float] = []
    real_wait_for = asyncio.wait_for

    async def capturing_wait_for(fut: Any, timeout: float | None = None) -> Any:
        if timeout is not None:
            wait_for_timeouts.append(float(timeout))
        return await real_wait_for(fut, timeout)

    cfg = RunConfig.model_validate(
        {
            "symbol": "MNDY",
            "company_name": None,
            "today_low": 68,
            "today_high": 74,
            "current_price": 73.24,
            "today_date": "Fri May 8, 2026",
            "today_session": "after the market trading window",
            "earnings_date": "Mon May 11 2026",
            "earnings_timing": "early morning et, before the market open",
            "target_dates": ["Mon May 11"],
            "next_trading_day": "Tues May 12",
            "followup_open_date": "Mon May 18",
            "historical_quarters": 11,
            "short_interest_lookbacks": ["last month"],
            "request_timeout_s": 180,
            "providers": [
                {"name": "anthropic", "model": "claude-opus-4-7"},
                {"name": "openai", "request_timeout_s": 600},
            ],
            "synthesizer": "gemini",
        }
    )

    def _fake_registry() -> ProviderRegistry:
        reg = ProviderRegistry()
        reg.register("anthropic", lambda **_: _SleepyProvider(name="anthropic", delay_s=0.0, text="A"))
        reg.register("openai", lambda **_: _SleepyProvider(name="openai", delay_s=0.0, text="B"))
        reg.register("gemini", lambda **_: _SleepyProvider(name="gemini", delay_s=0.0, text="SYNTH"))
        return reg

    import equity_analyst.orchestrator as orch_mod

    monkeypatch.setattr(orch_mod.asyncio, "wait_for", capturing_wait_for)
    monkeypatch.setattr(orch_mod.ProviderRegistry, "default", classmethod(lambda cls: _fake_registry()))

    orch = Orchestrator(config=cfg, prompt_path=prompt_path)
    await orch.run_async(dry_run=False, enable_web_search=False)

    assert wait_for_timeouts.count(600.0) == 1
    assert 180.0 in wait_for_timeouts


@pytest.mark.asyncio
async def test_drive_upload_invoked_and_run_json_has_url(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DRIVE_AUTH_MODE", raising=False)
    repo_root = Path(__file__).resolve().parents[1]
    prompt_path = repo_root / "prompts" / "equity_analyst.j2"

    cfg = RunConfig.model_validate(
        {
            "symbol": "MNDY",
            "company_name": None,
            "today_low": 68,
            "today_high": 74,
            "current_price": 73.24,
            "today_date": "Fri May 8, 2026",
            "today_session": "after the market trading window",
            "earnings_date": "Mon May 11 2026",
            "earnings_timing": "early morning et, before the market open",
            "target_dates": ["Mon May 11"],
            "next_trading_day": "Tues May 12",
            "followup_open_date": "Mon May 18",
            "historical_quarters": 11,
            "short_interest_lookbacks": ["last month"],
            "providers": ["anthropic", "openai"],
            "synthesizer": "gemini",
            "drive_upload_enabled": True,
            "drive_auth_mode": "service_account",
            "drive_credentials_path": str(tmp_path / "sa.json"),
            "drive_root_folder_id": "ROOT",
        }
    )

    def _fake_registry() -> ProviderRegistry:
        reg = ProviderRegistry()
        reg.register("anthropic", lambda **_: _SleepyProvider(name="anthropic", delay_s=0.0, text="A"))
        reg.register("openai", lambda **_: _SleepyProvider(name="openai", delay_s=0.0, text="B"))
        reg.register("gemini", lambda **_: _SleepyProvider(name="gemini", delay_s=0.0, text="SYNTH"))
        return reg

    import equity_analyst.orchestrator as orch_mod

    monkeypatch.setattr(orch_mod.ProviderRegistry, "default", classmethod(lambda cls: _fake_registry()))

    _write_minimal_sa_key_json(tmp_path / "sa.json")
    monkeypatch.setattr(
        "equity_analyst.drive_uploader._drive_root_preflight_probe",
        lambda *_a, **_k: {
            "id": "ROOT",
            "name": "root",
            "mimeType": "application/vnd.google-apps.folder",
            "driveId": "D1",
            "capabilities": {"canAddChildren": True},
        },
    )

    uploads: list[Path] = []

    async def fake_drive(c: RunConfig, out: Path, **kwargs: Any) -> str | None:
        uploads.append(out)
        p = out / "run.json"
        meta = json.loads(p.read_text(encoding="utf-8"))
        meta["drive_folder_url"] = "https://drive.google.com/drive/folders/xyz"
        p.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return meta["drive_folder_url"]

    monkeypatch.setattr(orch_mod, "maybe_upload_run_to_drive", fake_drive)

    orch = Orchestrator(config=cfg, prompt_path=prompt_path)
    await orch.run_async(dry_run=False, enable_web_search=False)
    assert len(uploads) == 1
    out_dir = uploads[0]
    meta = json.loads((out_dir / "run.json").read_text(encoding="utf-8"))
    assert meta.get("drive_folder_url") == "https://drive.google.com/drive/folders/xyz"


@pytest.mark.asyncio
async def test_drive_upload_disabled_skips_hook(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]
    prompt_path = repo_root / "prompts" / "equity_analyst.j2"

    cfg = RunConfig.model_validate(
        {
            "symbol": "MNDY",
            "company_name": None,
            "today_low": 68,
            "today_high": 74,
            "current_price": 73.24,
            "today_date": "Fri May 8, 2026",
            "today_session": "after the market trading window",
            "earnings_date": "Mon May 11 2026",
            "earnings_timing": "early morning et, before the market open",
            "target_dates": ["Mon May 11"],
            "next_trading_day": "Tues May 12",
            "followup_open_date": "Mon May 18",
            "historical_quarters": 11,
            "short_interest_lookbacks": ["last month"],
            "providers": ["anthropic", "openai"],
            "synthesizer": "gemini",
            "drive_upload_enabled": False,
        }
    )

    def _fake_registry() -> ProviderRegistry:
        reg = ProviderRegistry()
        reg.register("anthropic", lambda **_: _SleepyProvider(name="anthropic", delay_s=0.0, text="A"))
        reg.register("openai", lambda **_: _SleepyProvider(name="openai", delay_s=0.0, text="B"))
        reg.register("gemini", lambda **_: _SleepyProvider(name="gemini", delay_s=0.0, text="SYNTH"))
        return reg

    import equity_analyst.orchestrator as orch_mod

    monkeypatch.setattr(orch_mod.ProviderRegistry, "default", classmethod(lambda cls: _fake_registry()))

    called = False

    async def fake_drive(*_a: Any, **_k: Any) -> str | None:
        nonlocal called
        called = True
        return None

    monkeypatch.setattr(orch_mod, "maybe_upload_run_to_drive", fake_drive)

    orch = Orchestrator(config=cfg, prompt_path=prompt_path)
    await orch.run_async(dry_run=False, enable_web_search=False)
    assert called is False


@pytest.mark.asyncio
async def test_drive_upload_failure_still_completes_run(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DRIVE_AUTH_MODE", raising=False)
    repo_root = Path(__file__).resolve().parents[1]
    prompt_path = repo_root / "prompts" / "equity_analyst.j2"

    cfg = RunConfig.model_validate(
        {
            "symbol": "MNDY",
            "company_name": None,
            "today_low": 68,
            "today_high": 74,
            "current_price": 73.24,
            "today_date": "Fri May 8, 2026",
            "today_session": "after the market trading window",
            "earnings_date": "Mon May 11 2026",
            "earnings_timing": "early morning et, before the market open",
            "target_dates": ["Mon May 11"],
            "next_trading_day": "Tues May 12",
            "followup_open_date": "Mon May 18",
            "historical_quarters": 11,
            "short_interest_lookbacks": ["last month"],
            "providers": ["anthropic", "openai"],
            "synthesizer": "gemini",
            "drive_upload_enabled": True,
            "drive_auth_mode": "service_account",
            "drive_credentials_path": str(tmp_path / "sa.json"),
            "drive_root_folder_id": "ROOT",
        }
    )

    def _fake_registry() -> ProviderRegistry:
        reg = ProviderRegistry()
        reg.register("anthropic", lambda **_: _SleepyProvider(name="anthropic", delay_s=0.0, text="A"))
        reg.register("openai", lambda **_: _SleepyProvider(name="openai", delay_s=0.0, text="B"))
        reg.register("gemini", lambda **_: _SleepyProvider(name="gemini", delay_s=0.0, text="SYNTH"))
        return reg

    import equity_analyst.orchestrator as orch_mod

    monkeypatch.setattr(orch_mod.ProviderRegistry, "default", classmethod(lambda cls: _fake_registry()))

    _write_minimal_sa_key_json(tmp_path / "sa.json")
    monkeypatch.setattr(
        "equity_analyst.drive_uploader._drive_root_preflight_probe",
        lambda *_a, **_k: {
            "id": "ROOT",
            "name": "root",
            "mimeType": "application/vnd.google-apps.folder",
            "driveId": "D1",
            "capabilities": {"canAddChildren": True},
        },
    )

    async def boom(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("upload exploded")

    monkeypatch.setattr("equity_analyst.drive_uploader.asyncio.to_thread", boom)

    orch = Orchestrator(config=cfg, prompt_path=prompt_path)
    text, artifacts = await orch.run_async(dry_run=False, enable_web_search=False)
    assert "SYNTH" in text
    assert (artifacts.output_dir / "run.json").is_file()
    meta = json.loads((artifacts.output_dir / "run.json").read_text(encoding="utf-8"))
    assert "drive_folder_url" not in meta

