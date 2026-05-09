from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from equity_analyst.config import RunConfig, load_config
from equity_analyst.orchestrator import Orchestrator
from equity_analyst.providers.base import LLMProvider
from equity_analyst.providers.registry import ProviderRegistry
from equity_analyst.types import ProviderResponse, ProviderUsage


class _Slow(LLMProvider):
    def __init__(self, *, name: str, delay_s: float, text: str) -> None:
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
            usage=ProviderUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            raw=None,
        )


class _Boom(LLMProvider):
    def __init__(self, *, name: str) -> None:
        self.name = name

    async def generate(
        self, prompt: str, *, enable_web_search: bool = True, max_output_tokens: int | None = None
    ) -> ProviderResponse:
        msg = f"{self.name}-boom"
        raise RuntimeError(msg)


@pytest.mark.asyncio
async def test_orchestrator_timeout_does_not_block_other_providers(
    tmp_path: Path, monkeypatch: Any
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
            "target_dates": ["Mon May 11", "Fri May 15"],
            "next_trading_day": "Tues May 12",
            "followup_open_date": "Mon May 18",
            "historical_quarters": 11,
            "short_interest_lookbacks": ["last month"],
            "providers": [
                {"name": "slow", "request_timeout_s": 0.15},
                {"name": "fast"},
            ],
            "synthesizer": "synth",
            "request_timeout_s": 5.0,
        }
    )

    def _fake_registry() -> ProviderRegistry:
        reg = ProviderRegistry()
        reg.register("slow", lambda: _Slow(name="slow", delay_s=2.0, text="S"))
        reg.register("fast", lambda: _Slow(name="fast", delay_s=0.05, text="F"))
        reg.register("synth", lambda: _Slow(name="synth", delay_s=0.0, text="SYN"))
        return reg

    import equity_analyst.orchestrator as orch_mod

    monkeypatch.setattr(
        orch_mod.ProviderRegistry,
        "default",
        classmethod(lambda cls: _fake_registry()),
    )

    orch = Orchestrator(config=cfg, prompt_path=prompt_path)
    synthesis, artifacts = await orch.run_async(dry_run=False, enable_web_search=False)
    run_meta = json.loads((artifacts.output_dir / "run.json").read_text(encoding="utf-8"))

    assert "SYN" in synthesis
    slow = run_meta["providers"]["slow"]
    assert slow["model"] == "error:timeout"
    assert run_meta["providers"]["fast"]["model"] == "fake"
    assert "timing" in run_meta
    assert "parallel_provider_batch_wall_s" in run_meta["timing"]


@pytest.mark.asyncio
async def test_orchestrator_one_provider_failure_others_continue(
    tmp_path: Path, monkeypatch: Any
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
            "target_dates": ["Mon May 11", "Fri May 15"],
            "next_trading_day": "Tues May 12",
            "followup_open_date": "Mon May 18",
            "historical_quarters": 11,
            "short_interest_lookbacks": ["last month"],
            "providers": ["boom", "fast"],
            "synthesizer": "synth",
            "request_timeout_s": 5.0,
        }
    )

    def _fake_registry() -> ProviderRegistry:
        reg = ProviderRegistry()
        reg.register("boom", lambda: _Boom(name="boom"))
        reg.register("fast", lambda: _Slow(name="fast", delay_s=0.05, text="F"))
        reg.register("synth", lambda: _Slow(name="synth", delay_s=0.0, text="SYN"))
        return reg

    import equity_analyst.orchestrator as orch_mod

    monkeypatch.setattr(
        orch_mod.ProviderRegistry,
        "default",
        classmethod(lambda cls: _fake_registry()),
    )

    orch = Orchestrator(config=cfg, prompt_path=prompt_path)
    synthesis, artifacts = await orch.run_async(dry_run=False, enable_web_search=False)
    run_meta = json.loads((artifacts.output_dir / "run.json").read_text(encoding="utf-8"))

    assert "SYN" in synthesis
    assert run_meta["providers"]["boom"]["model"] == "error:RuntimeError"
    assert run_meta["providers"]["fast"]["model"] == "fake"


def test_yaml_per_provider_web_search_override(tmp_path: Path) -> None:
    yml = tmp_path / "c.yaml"
    yml.write_text(
        yaml.safe_dump(
            {
                "symbol": "X",
                "today_low": 1,
                "today_high": 2,
                "current_price": 1.5,
                "today_date": "d",
                "today_session": "s",
                "earnings_date": "e",
                "earnings_timing": "t",
                "target_dates": [],
                "next_trading_day": "n",
                "followup_open_date": "f",
                "historical_quarters": 1,
                "short_interest_lookbacks": [],
                "providers": [
                    {"name": "anthropic", "web_search": True},
                    {"name": "openai", "web_search": False},
                ],
                "synthesizer": "anthropic",
            }
        ),
        encoding="utf-8",
    )
    cfg = load_config(str(yml))
    from equity_analyst.provider_runtime import effective_web_search

    a, o = cfg.providers
    assert effective_web_search(run_default=False, pc=a) is True
    assert effective_web_search(run_default=True, pc=o) is False


@pytest.mark.asyncio
async def test_run_json_timing_present_after_live_run(tmp_path: Path, monkeypatch: Any) -> None:
    """Reuses orchestrator happy path from test_orchestrator to assert timing keys."""
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
            "target_dates": ["Mon May 11", "Fri May 15"],
            "next_trading_day": "Tues May 12",
            "followup_open_date": "Mon May 18",
            "historical_quarters": 11,
            "short_interest_lookbacks": ["last month"],
            "providers": ["anthropic", "openai"],
            "synthesizer": "synth",
        }
    )

    class _Fast(LLMProvider):
        def __init__(self, *, name: str, text: str) -> None:
            self.name = name
            self._text = text

        async def generate(
            self,
            prompt: str,
            *,
            enable_web_search: bool = True,
            max_output_tokens: int | None = None,
        ) -> ProviderResponse:
            return ProviderResponse(
                provider_name=self.name,
                model="fake",
                text=self._text,
                usage=ProviderUsage(input_tokens=1, output_tokens=2, total_tokens=3),
                raw=None,
            )

    def _fake_registry() -> ProviderRegistry:
        reg = ProviderRegistry()
        reg.register("anthropic", lambda: _Fast(name="anthropic", text="A"))
        reg.register("openai", lambda: _Fast(name="openai", text="B"))
        reg.register("synth", lambda: _Fast(name="synth", text="SYNTH"))
        return reg

    import equity_analyst.orchestrator as orch_mod

    monkeypatch.setattr(
        orch_mod.ProviderRegistry,
        "default",
        classmethod(lambda cls: _fake_registry()),
    )

    orch = Orchestrator(config=cfg, prompt_path=prompt_path)
    _, artifacts = await orch.run_async(dry_run=False, enable_web_search=False)
    run_meta = json.loads((artifacts.output_dir / "run.json").read_text(encoding="utf-8"))
    t = run_meta["timing"]
    assert set(t.keys()) >= {
        "parallel_provider_batch_wall_s",
        "synthesis_wall_s",
        "total_wall_s",
        "per_provider",
    }
    assert "anthropic" in t["per_provider"] and "openai" in t["per_provider"]
