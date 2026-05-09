from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver

from equity_analyst.config import RunConfig
from equity_analyst.iterative import (
    build_initial_refinement_state,
    compile_refinement_workflow,
    dry_run_compile_only,
    parse_overall_confidence,
)
from equity_analyst.providers.base import LLMProvider
from equity_analyst.providers.registry import ProviderRegistry
from equity_analyst.types import ProviderResponse, ProviderUsage


def _base_cfg(**kwargs: Any) -> RunConfig:
    d: dict[str, Any] = {
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
        "providers": ["openai"],
        "synthesizer": "synth",
    }
    d.update(kwargs)
    return RunConfig.model_validate(d)


class _Txt(LLMProvider):
    def __init__(self, name: str, text: str) -> None:
        self.name = name
        self._text = text

    async def generate(
        self, prompt: str, *, enable_web_search: bool = True, max_output_tokens: int | None = None
    ) -> ProviderResponse:
        return ProviderResponse(
            provider_name=self.name,
            model="m",
            text=self._text,
            usage=ProviderUsage(),
            raw=None,
        )


class _SynthCalls(LLMProvider):
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0

    async def generate(
        self, prompt: str, *, enable_web_search: bool = True, max_output_tokens: int | None = None
    ) -> ProviderResponse:
        self.calls += 1
        if self.calls <= 2:
            body = "low\nOVERALL_CONFIDENCE: 0.2\n"
        else:
            body = "high\nOVERALL_CONFIDENCE: 0.95\n"
        return ProviderResponse(
            provider_name=self.name,
            model="m",
            text=body,
            usage=ProviderUsage(),
            raw=None,
        )


class _VerCalls(LLMProvider):
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0

    async def generate(
        self, prompt: str, *, enable_web_search: bool = True, max_output_tokens: int | None = None
    ) -> ProviderResponse:
        self.calls += 1
        if self.calls == 1:
            payload = '{"verified":[],"contradicted":["numbers look off"],"unverifiable":[]}'
        else:
            payload = '{"verified":[],"contradicted":[],"unverifiable":[]}'
        return ProviderResponse(
            provider_name=self.name,
            model="m",
            text=payload,
            usage=ProviderUsage(),
            raw=None,
        )


class _SynthLow(LLMProvider):
    def __init__(self, name: str) -> None:
        self.name = name

    async def generate(
        self, prompt: str, *, enable_web_search: bool = True, max_output_tokens: int | None = None
    ) -> ProviderResponse:
        return ProviderResponse(
            provider_name=self.name,
            model="m",
            text="x\nOVERALL_CONFIDENCE: 0.1\n",
            usage=ProviderUsage(),
            raw=None,
        )


class _VerBad(LLMProvider):
    def __init__(self, name: str) -> None:
        self.name = name

    async def generate(
        self, prompt: str, *, enable_web_search: bool = True, max_output_tokens: int | None = None
    ) -> ProviderResponse:
        return ProviderResponse(
            provider_name=self.name,
            model="m",
            text='{"verified":[],"contradicted":["always"],"unverifiable":[]}',
            usage=ProviderUsage(),
            raw=None,
        )


class _SynthCheckpoint(LLMProvider):
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0

    async def generate(
        self, prompt: str, *, enable_web_search: bool = True, max_output_tokens: int | None = None
    ) -> ProviderResponse:
        self.calls += 1
        body = "a\nOVERALL_CONFIDENCE: 0.1\n" if self.calls < 2 else "b\nOVERALL_CONFIDENCE: 0.99\n"
        return ProviderResponse(
            provider_name=self.name,
            model="m",
            text=body,
            usage=ProviderUsage(),
            raw=None,
        )


def _initial_state(cfg: RunConfig, out: Path) -> dict[str, Any]:
    st = build_initial_refinement_state(cfg=cfg, rendered_text="PROMPT", output_dir=out)
    st["max_iterations"] = 5
    st["confidence_threshold"] = 0.85
    st["enable_web_search"] = False
    return st


@pytest.mark.asyncio
async def test_loop_converges_when_synthesis_high_confidence(tmp_path: Path) -> None:
    reg = ProviderRegistry()
    reg.register("openai", lambda: _Txt("openai", "fan"))
    reg.register("synth", lambda: _Txt("synth", "ok\nOVERALL_CONFIDENCE: 0.95\n"))
    reg.register(
        "anthropic",
        lambda: _Txt("anthropic", '{"verified":[],"contradicted":[],"unverifiable":[]}'),
    )
    cfg = _base_cfg()
    out = tmp_path / "o"
    out.mkdir()
    app = compile_refinement_workflow(registry=reg, checkpointer=MemorySaver())
    final = await app.ainvoke(
        _initial_state(cfg, out), config={"configurable": {"thread_id": "t1"}}
    )
    assert len(final["provider_responses"]) == 1
    assert (out / "synthesis.md").is_file()
    run_json = out / "run.json"
    assert run_json.is_file()
    meta = json.loads(run_json.read_text(encoding="utf-8"))
    assert "timing" in meta
    assert "iterations" in meta["timing"]


@pytest.mark.asyncio
async def test_loop_continues_on_low_confidence(tmp_path: Path) -> None:
    reg = ProviderRegistry()
    reg.register("openai", lambda: _Txt("openai", "fan"))
    synth = _SynthCalls("synth")
    reg.register("synth", lambda: synth)
    reg.register(
        "anthropic",
        lambda: _Txt("anthropic", '{"verified":[],"contradicted":[],"unverifiable":[]}'),
    )
    cfg = _base_cfg()
    out = tmp_path / "o"
    out.mkdir()
    app = compile_refinement_workflow(registry=reg, checkpointer=MemorySaver())
    final = await app.ainvoke(
        _initial_state(cfg, out), config={"configurable": {"thread_id": "t2"}}
    )
    assert len(final["provider_responses"]) == 3


@pytest.mark.asyncio
async def test_loop_continues_on_contradictions(tmp_path: Path) -> None:
    reg = ProviderRegistry()
    reg.register("openai", lambda: _Txt("openai", "fan"))
    reg.register("synth", lambda: _Txt("synth", "syn\nOVERALL_CONFIDENCE: 0.95\n"))
    ver = _VerCalls("anthropic")
    reg.register("anthropic", lambda: ver)
    cfg = _base_cfg()
    out = tmp_path / "o"
    out.mkdir()
    st = _initial_state(cfg, out)
    app = compile_refinement_workflow(registry=reg, checkpointer=MemorySaver())
    final = await app.ainvoke(st, config={"configurable": {"thread_id": "t3"}})
    assert len(final["provider_responses"]) >= 2
    assert final["followup_questions"]


@pytest.mark.asyncio
async def test_max_iterations_cutoff(tmp_path: Path) -> None:
    reg = ProviderRegistry()
    reg.register("openai", lambda: _Txt("openai", "fan"))
    reg.register("synth", lambda: _SynthLow("synth"))
    reg.register("anthropic", lambda: _VerBad("anthropic"))
    cfg = _base_cfg()
    out = tmp_path / "o"
    out.mkdir()
    st = _initial_state(cfg, out)
    st["max_iterations"] = 3
    app = compile_refinement_workflow(registry=reg, checkpointer=MemorySaver())
    final = await app.ainvoke(st, config={"configurable": {"thread_id": "t4"}})
    assert len(final["provider_responses"]) == 3
    assert (out / "synthesis.md").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_checkpoint_resume(tmp_path: Path) -> None:
    reg = ProviderRegistry()
    reg.register("openai", lambda: _Txt("openai", "fan"))
    synth_ck = _SynthCheckpoint("synth")
    reg.register("synth", lambda: synth_ck)
    reg.register(
        "anthropic",
        lambda: _Txt("anthropic", '{"verified":[],"contradicted":[],"unverifiable":[]}'),
    )
    cfg = _base_cfg()
    out = tmp_path / "o"
    out.mkdir()
    st = _initial_state(cfg, out)
    saver = MemorySaver()
    app = compile_refinement_workflow(
        registry=reg, checkpointer=saver, interrupt_before=["finalize"]
    )
    cfg_g: dict = {"configurable": {"thread_id": "t5"}}
    await app.ainvoke(st, config=cfg_g)
    mid = app.get_state(cfg_g)
    assert mid.values is not None
    vals = mid.values
    assert len(vals.get("provider_responses", [])) == 2
    assert not (out / "synthesis.md").exists()
    await app.ainvoke(None, config=cfg_g)
    assert (out / "synthesis.md").exists()


def test_parse_overall_confidence() -> None:
    assert parse_overall_confidence("foo\nOVERALL_CONFIDENCE: 0.75\n") == 0.75
    assert parse_overall_confidence("no marker") is None


def test_dry_run_compile_nodes() -> None:
    nodes = dry_run_compile_only(registry=ProviderRegistry.default())
    assert "fan_out" in nodes
    assert "finalize" in nodes
