from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import pytest

from equity_analyst.config import RunConfig
from equity_analyst.orchestrator import Orchestrator
from equity_analyst.providers.base import LLMProvider
from equity_analyst.providers.registry import ProviderRegistry
from equity_analyst.types import ProviderResponse, ProviderUsage


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
            "synthesizer": "synth",
        }
    )

    def _fake_registry() -> ProviderRegistry:
        reg = ProviderRegistry()
        reg.register("anthropic", lambda: _SleepyProvider(name="anthropic", delay_s=0.25, text="A"))
        reg.register("openai", lambda: _SleepyProvider(name="openai", delay_s=0.25, text="B"))
        reg.register("gemini", lambda: _SleepyProvider(name="gemini", delay_s=0.25, text="G"))
        reg.register("grok", lambda: _SleepyProvider(name="grok", delay_s=0.25, text="K"))
        reg.register("synth", lambda: _SleepyProvider(name="synth", delay_s=0.0, text="SYNTH"))
        return reg

    import equity_analyst.orchestrator as orch_mod

    monkeypatch.setattr(
        orch_mod.ProviderRegistry,
        "default",
        classmethod(lambda cls: _fake_registry()),
    )

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
    assert (out_dir / "run.json").exists()
    assert (out_dir / "agent.log").exists()
    assert any("Run start" in r.message for r in caplog.records)
    assert "Run start" in (out_dir / "agent.log").read_text(encoding="utf-8")
