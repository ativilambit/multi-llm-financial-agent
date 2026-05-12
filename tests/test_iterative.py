from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, ClassVar

import pytest
from langgraph.checkpoint.memory import MemorySaver

from equity_analyst.config import RunConfig
from equity_analyst.iterative import (
    CHANGELOG_ROUND_SUMMARY_MAX_CHARS,
    VERIFIER_INSTRUCTION_PREFIX,
    augment_verifier_result_with_sigma_structural_checks,
    build_initial_refinement_state,
    compile_refinement_workflow,
    compute_refinement_route_command,
    dry_run_compile_only,
    parse_overall_confidence,
    parse_verifier_json,
    round_summary_for_changelog,
    sigma_band_sqrt_ratio_followups,
)
from equity_analyst.prompt_parts import EQUITY_ANALYST_SYSTEM_PROMPT
from equity_analyst.prompting import RenderedPrompt
from equity_analyst.providers.base import LLMProvider
from equity_analyst.providers.openai_provider import OpenAIProvider
from equity_analyst.providers.registry import ProviderRegistry
from equity_analyst.types import ProviderResponse, ProviderUsage
from tests.test_providers import _FakeOpenAIClient

_FANOUT_CALL_COUNTS: defaultdict[str, int] = defaultdict(int)


def test_verifier_instruction_accepts_dual_sd_anchors() -> None:
    assert "same-day intraday" in VERIFIER_INSTRUCTION_PREFIX
    assert "prior-close" in VERIFIER_INSTRUCTION_PREFIX


def test_verifier_instruction_sigma_structural_checks() -> None:
    assert "band structural checks" in VERIFIER_INSTRUCTION_PREFIX
    assert "HV30 sqrt(t) scaling" in VERIFIER_INSTRUCTION_PREFIX


def test_verifier_instruction_section8_citation_heuristic() -> None:
    assert "section 8" in VERIFIER_INSTRUCTION_PREFIX
    assert "800" in VERIFIER_INSTRUCTION_PREFIX
    assert "Source:" in VERIFIER_INSTRUCTION_PREFIX


def test_sigma_band_sqrt_ratio_followups_flags_may13_to_may19_example() -> None:
    """60% vs 75% over 4 trading sessions implies ratio 1.25 vs √4=2.0 (>25% off)."""
    qs = sigma_band_sqrt_ratio_followups(
        width_early=60.0,
        width_late=75.0,
        trading_day_span=4,
        session_early="Wednesday, May 13",
        session_late="Tuesday, May 19",
        tolerance=0.25,
    )
    assert len(qs) == 1
    assert "sqrt-t" in qs[0]
    assert "1.25" in qs[0]
    assert "2.00" in qs[0]


def test_sigma_band_sqrt_ratio_followups_within_tolerance_empty() -> None:
    assert sigma_band_sqrt_ratio_followups(
        width_early=60.0,
        width_late=120.0,
        trading_day_span=4,
        tolerance=0.25,
    ) == []


def test_augment_verifier_sigma_structural_checks_from_synthesis_text() -> None:
    sg = chr(0x03C3)
    pm = chr(0x00B1)
    syn = f"""\
Wednesday, May 13 (First Post-Earnings Session Open/Close):
  - 3{sg}: $75.76 - $176.76 ({pm}60.0%)

Tuesday, May 19 (~1 Week Post-Earnings Open/Close):
  - 3{sg}: $31.57 - $220.95 ({pm}75.0%)
"""
    base = parse_verifier_json(
        '{"verified":[],"contradicted":[],"unverifiable":[],'
        '"refresh_facts":false,"refan_out_providers":[],"refan_out_all":false}'
    )
    out = augment_verifier_result_with_sigma_structural_checks(syn, base)
    unver = out["unverifiable"]
    assert any("scaling check" in u for u in unver)
    assert any("sqrt-t" in u for u in unver)


def test_augment_verifier_flags_expiry_not_in_verified_chain() -> None:
    sg = chr(0x03C3)
    syn = f"3{sg}: 1% derived from 2026-05-15 weekly expiry\n"
    base = parse_verifier_json(
        '{"verified":[],"contradicted":[],"unverifiable":[],'
        '"sigma_band_sessions":[{"session":"t","sigma_baseline":"2026-05-99 weekly",'
        '"sigma_scaling_check_passed":true}],'
        '"refresh_facts":false,"refan_out_providers":[],"refan_out_all":false}'
    )
    oc = {"options_chain_available": True, "available_expiries": ["2026-05-15", "2026-05-22"]}
    out = augment_verifier_result_with_sigma_structural_checks(syn, base, options_chain_data=oc, symbol="DT")
    joined = " ".join(out["unverifiable"])
    assert "2026-05-99" in joined
    assert "verified chain" in joined.lower()
    raw = (
        '{"verified":[],"contradicted":[],"unverifiable":[],'
        '"sigma_band_sessions":[{"session":"May 13","sigma_baseline":"2026-05-16",'
        '"sigma_scaling_check_passed":false}],"sigma_scaling_aggregate_passed":false}'
    )
    out = parse_verifier_json(raw)
    assert out["sigma_band_sessions"][0]["session"] == "May 13"
    assert out["sigma_band_sessions"][0]["sigma_scaling_check_passed"] is False
    assert out["sigma_scaling_aggregate_passed"] is False


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
        "synthesizer": "gemini",
        "facts_packet_enabled": False,
        "conditional_fanout_enabled": False,
    }
    d.update(kwargs)
    return RunConfig.model_validate(d)


class _RecordingOpenAI(OpenAIProvider):
    """Captures ``cacheable_prefix`` for each fan-out OpenAI call (real request shape via super)."""

    captured: ClassVar[list[str | None]] = []

    def __init__(self) -> None:
        super().__init__(model="gpt-5.5", client=_FakeOpenAIClient())  # type: ignore[arg-type]

    async def generate(
        self,
        prompt: str,
        *,
        enable_web_search: bool = True,
        max_output_tokens: int | None = None,
        cacheable_prefix: str | None = None,
        user_message_for_cache: str | None = None,
    ) -> ProviderResponse:
        _RecordingOpenAI.captured.append(cacheable_prefix)
        return await super().generate(
            prompt,
            enable_web_search=enable_web_search,
            max_output_tokens=max_output_tokens,
            cacheable_prefix=cacheable_prefix,
            user_message_for_cache=user_message_for_cache,
        )


def _refinement_state_with_equity_system(cfg: RunConfig, out: Path) -> dict[str, Any]:
    user_body = "Test user message for iterative cache stability.\n"
    full = f"{EQUITY_ANALYST_SYSTEM_PROMPT}\n\n{user_body}"
    rendered = RenderedPrompt(
        template_path="t",
        text=full,
        context={},
        user_message_text=user_body,
    )
    st = build_initial_refinement_state(cfg=cfg, rendered=rendered, output_dir=out)
    st["max_iterations"] = 5
    st["confidence_threshold"] = 0.85
    st["enable_web_search"] = False
    return st


_VERIFIER_MARK = "You are a financial fact-checker"


class _GeminiSplit(LLMProvider):
    """One registry slot for `gemini`: synthesis vs verifier prompts."""

    def __init__(self, synth: LLMProvider, ver: LLMProvider) -> None:
        self.name = "gemini"
        self._synth = synth
        self._ver = ver

    async def generate(
        self,
        prompt: str,
        *,
        enable_web_search: bool = True,
        max_output_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        if _VERIFIER_MARK in prompt:
            return await self._ver.generate(
                prompt,
                enable_web_search=enable_web_search,
                max_output_tokens=max_output_tokens,
                **kwargs,
            )
        return await self._synth.generate(
            prompt,
            enable_web_search=enable_web_search,
            max_output_tokens=max_output_tokens,
            **kwargs,
        )


class _CaptureCreateRegistry(ProviderRegistry):
    def __init__(self) -> None:
        super().__init__()
        self.create_calls: list[tuple[str, dict[str, Any]]] = []

    def create(  # type: ignore[override]
        self,
        name: str,
        *,
        model: str | None = None,
        client: Any | None = None,
        gemini_cache_index: Any | None = None,
        gemini_cache_ttl_s: int | None = None,
    ) -> LLMProvider:
        self.create_calls.append(
            (
                name,
                {
                    "model": model,
                    "gemini_cache_index": gemini_cache_index,
                    "gemini_cache_ttl_s": gemini_cache_ttl_s,
                },
            )
        )
        return super().create(
            name,
            model=model,
            client=client,
            gemini_cache_index=gemini_cache_index,
            gemini_cache_ttl_s=gemini_cache_ttl_s,
        )


class _Txt(LLMProvider):
    def __init__(self, name: str, text: str) -> None:
        self.name = name
        self._text = text

    async def generate(
        self,
        prompt: str,
        *,
        enable_web_search: bool = True,
        max_output_tokens: int | None = None,
        **kwargs: Any,
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
        self,
        prompt: str,
        *,
        enable_web_search: bool = True,
        max_output_tokens: int | None = None,
        **kwargs: Any,
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
        self,
        prompt: str,
        *,
        enable_web_search: bool = True,
        max_output_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        self.calls += 1
        if self.calls == 1:
            payload = (
                '{"verified":[],"contradicted":["numbers look off"],"unverifiable":[],'
                '"refresh_facts":false,"refan_out_providers":[],"refan_out_all":false}'
            )
        else:
            payload = (
                '{"verified":[],"contradicted":[],"unverifiable":[],'
                '"refresh_facts":false,"refan_out_providers":[],"refan_out_all":false}'
            )
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
        self,
        prompt: str,
        *,
        enable_web_search: bool = True,
        max_output_tokens: int | None = None,
        **kwargs: Any,
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
        self,
        prompt: str,
        *,
        enable_web_search: bool = True,
        max_output_tokens: int | None = None,
        **kwargs: Any,
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
        self,
        prompt: str,
        *,
        enable_web_search: bool = True,
        max_output_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        self.calls += 1
        body = (
            "a\nOVERALL_CONFIDENCE: 0.1\n"
            if self.calls < 2
            else "b\nOVERALL_CONFIDENCE: 0.99\n"
        )
        return ProviderResponse(
            provider_name=self.name,
            model="m",
            text=body,
            usage=ProviderUsage(),
            raw=None,
        )


def _initial_state(cfg: RunConfig, out: Path) -> dict[str, Any]:
    rendered = RenderedPrompt(
        template_path="t",
        text="PROMPT",
        context={},
        user_message_text="U",
    )
    st = build_initial_refinement_state(cfg=cfg, rendered=rendered, output_dir=out)
    st["max_iterations"] = 5
    st["confidence_threshold"] = 0.85
    st["enable_web_search"] = False
    return st


@pytest.mark.asyncio
async def test_iterative_maybe_write_pdf_sibling_targets_expected_md_paths(
    tmp_path: Path, monkeypatch: Any
) -> None:
    pdf_md_paths: list[Path] = []

    def _capture_pdf(**kwargs: Any) -> None:
        pdf_md_paths.append(kwargs["md_path"])

    monkeypatch.setattr("equity_analyst.iterative.maybe_write_pdf_sibling", _capture_pdf)

    reg = ProviderRegistry()
    reg.register("openai", lambda **_: _Txt("openai", "fan"))
    reg.register(
        "gemini",
        lambda **_: _GeminiSplit(
            _Txt("gemini", "ok\nOVERALL_CONFIDENCE: 0.95\n"),
            _Txt(
                "gemini",
                '{"verified":[],"contradicted":[],"unverifiable":[],'
                '"refresh_facts":false,"refan_out_providers":[],"refan_out_all":false}',
            ),
        ),
    )
    cfg = _base_cfg()
    out = tmp_path / "o"
    out.mkdir()
    app = compile_refinement_workflow(registry=reg, checkpointer=MemorySaver())
    await app.ainvoke(_initial_state(cfg, out), config={"configurable": {"thread_id": "t-pdf"}})
    names = {p.name for p in pdf_md_paths}
    assert "iteration_1_synthesis.md" in names
    assert "iteration_1_verify.md" in names
    assert "synthesis.md" in names
    assert "iteration_1.md" in names


@pytest.mark.asyncio
async def test_loop_converges_when_synthesis_high_confidence(tmp_path: Path) -> None:
    reg = ProviderRegistry()
    reg.register("openai", lambda **_: _Txt("openai", "fan"))
    reg.register(
        "gemini",
        lambda **_: _GeminiSplit(
            _Txt("gemini", "ok\nOVERALL_CONFIDENCE: 0.95\n"),
            _Txt(
                "gemini",
                '{"verified":[],"contradicted":[],"unverifiable":[],'
                '"refresh_facts":false,"refan_out_providers":[],"refan_out_all":false}',
            ),
        ),
    )
    cfg = _base_cfg()
    out = tmp_path / "o"
    out.mkdir()
    app = compile_refinement_workflow(registry=reg, checkpointer=MemorySaver())
    final = await app.ainvoke(_initial_state(cfg, out), config={"configurable": {"thread_id": "t1"}})
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
    reg.register("openai", lambda **_: _Txt("openai", "fan"))
    synth = _SynthCalls("gemini")
    reg.register(
        "gemini",
        lambda **_: _GeminiSplit(
            synth,
            _Txt(
                "gemini",
                '{"verified":[],"contradicted":[],"unverifiable":[],'
                '"refresh_facts":false,"refan_out_providers":[],"refan_out_all":false}',
            ),
        ),
    )
    cfg = _base_cfg()
    out = tmp_path / "o"
    out.mkdir()
    app = compile_refinement_workflow(registry=reg, checkpointer=MemorySaver())
    final = await app.ainvoke(_initial_state(cfg, out), config={"configurable": {"thread_id": "t2"}})
    assert len(final["provider_responses"]) == 3


@pytest.mark.asyncio
async def test_iterative_openai_cacheable_prefix_identical_across_rounds(tmp_path: Path) -> None:
    _RecordingOpenAI.captured.clear()
    reg = ProviderRegistry()
    reg.register("openai", lambda **_: _RecordingOpenAI())
    ver = _VerCalls("gemini")
    reg.register(
        "gemini",
        lambda **_: _GeminiSplit(
            _Txt("gemini", "syn\nOVERALL_CONFIDENCE: 0.95\n"),
            ver,
        ),
    )
    cfg = _base_cfg()
    out = tmp_path / "o"
    out.mkdir()
    app = compile_refinement_workflow(registry=reg, checkpointer=MemorySaver())
    await app.ainvoke(
        _refinement_state_with_equity_system(cfg, out),
        config={"configurable": {"thread_id": "openai-cache-prefix"}},
    )
    assert len(_RecordingOpenAI.captured) >= 2
    assert all(p == EQUITY_ANALYST_SYSTEM_PROMPT for p in _RecordingOpenAI.captured)
    assert all(p for p in _RecordingOpenAI.captured)


@pytest.mark.asyncio
async def test_loop_continues_on_contradictions(tmp_path: Path) -> None:
    reg = ProviderRegistry()
    reg.register("openai", lambda **_: _Txt("openai", "fan"))
    ver = _VerCalls("gemini")
    reg.register(
        "gemini",
        lambda **_: _GeminiSplit(
            _Txt("gemini", "syn\nOVERALL_CONFIDENCE: 0.95\n"),
            ver,
        ),
    )
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
    reg.register("openai", lambda **_: _Txt("openai", "fan"))
    reg.register(
        "gemini",
        lambda **_: _GeminiSplit(
            _SynthLow("gemini"),
            _VerBad("gemini"),
        ),
    )
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
    reg.register("openai", lambda **_: _Txt("openai", "fan"))
    synth_ck = _SynthCheckpoint("gemini")
    reg.register(
        "gemini",
        lambda **_: _GeminiSplit(
            synth_ck,
            _Txt(
                "gemini",
                '{"verified":[],"contradicted":[],"unverifiable":[],'
                '"refresh_facts":false,"refan_out_providers":[],"refan_out_all":false}',
            ),
        ),
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


def test_parse_verifier_json_handles_prose_wrapped_json() -> None:
    text = (
        'Here is the verification:\n{"verified": ["claim 1"], "contradicted": [], "unverifiable": []}\n'
    )
    out = parse_verifier_json(text)
    assert out["verified"] == ["claim 1"]
    assert out["contradicted"] == []
    assert out["unverifiable"] == []
    assert out.get("refresh_facts") is False
    assert out.get("refan_out_all") is False
    assert out.get("refan_out_providers") == []


def test_parse_verifier_json_handles_markdown_fences() -> None:
    text = """```json
{"verified": ["a"], "contradicted": ["b"], "unverifiable": []}
```"""
    out = parse_verifier_json(text)
    assert out["verified"] == ["a"]
    assert out["contradicted"] == ["b"]
    assert out["unverifiable"] == []


def test_parse_verifier_json_handles_alternate_key_names() -> None:
    payload = (
        '{"verified_claims": ["ok"], "contradictions": ["bad"], "unverifiable_claims": ["maybe"]}'
    )
    out = parse_verifier_json(payload)
    assert out["verified"] == ["ok"]
    assert out["contradicted"] == ["bad"]
    assert out["unverifiable"] == ["maybe"]


_ITERATION_1_TRUNCATED_VERIFIER_RAW = r"""{
  "verified": [
    "Piper Sandler downgraded from Overweight to Neutral, cutting the PT from $115/$100 to $85.",
    "Loop Capital downgraded from Strong Buy to Hold, cutting the PT to $80.",
    "KeyBanc / Canaccord cut targets significantly (to $140).",
    "Average consensus targets are still technically higher than the $73 spot price (averaging ~$122).",
    "Q4 2025 (Feb 2026): Beat EPS, T+1 move approx. -20.8% to -21.6%.",
    "Q3 2025 (Nov 2025): Beat EPS, T+1 move approx. -12.3% to -17.4%.",
    "Q2 2025 (Aug 2025): Beat EPS, T+1 move approx. -29.8%.",
    "Put/Call ratio is approximately 0.625 (or 1.6 calls traded for every 1 put).",
    "YTD Performance: Down ~48% to ~50% from recent highs.",
    "Volatility: IV is in the 100th percentile (annualized IV >170% on the front month).",
    "Gemini claims a precise 16.3% of float sold short.",
    "Momentum: RSI is sitting between 38 and 42"
  ],
  "contradicted": [
    "Q1 2025 (May 2025): Beat EPS, flat to slightly positive (+0.3% to +3%).",
    "Moving Averages: Trading significantly below the 20-day, 50-day, and 200-day SMAs (20 SMA cited near $73.90)."
  ],
  "unverifiable":"""


def test_parse_verifier_json_repairs_truncated_array() -> None:
    out = parse_verifier_json(
        _ITERATION_1_TRUNCATED_VERIFIER_RAW,
        provider_finish_reason="MAX_TOKENS",
    )
    assert out.get("_truncated") is True
    assert len(out["verified"]) >= 8
    assert len(out["contradicted"]) == 2
    assert out["unverifiable"] == []


def test_parse_verifier_json_repairs_minimal_truncation() -> None:
    out = parse_verifier_json(
        '{"verified": ["a", "b",',
        provider_finish_reason="MAX_TOKENS",
    )
    assert out.get("_truncated") is True
    assert out["verified"] == ["a", "b"]
    assert out["contradicted"] == []
    assert out["unverifiable"] == []


def test_parse_verifier_json_truncation_warning_includes_finish_reason_and_raw_bytes(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="equity_analyst.iterative")
    truncated_in = '{"verified": ["a"],'
    parse_verifier_json(truncated_in, provider_finish_reason="MAX_TOKENS")
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "finish_reason=MAX_TOKENS" in joined
    assert "raw_bytes=" in joined
    assert str(len(truncated_in.encode("utf-8"))) in joined


def test_parse_verifier_json_finish_stop_valid_json_no_truncation_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="equity_analyst.iterative")
    payload = '{"verified": ["x"], "contradicted": [], "unverifiable": []}'
    parse_verifier_json(payload, provider_finish_reason="STOP")
    assert not any("response was truncated" in r.getMessage() for r in caplog.records)


def test_parse_verifier_json_repair_finish_stop_no_truncated_marker() -> None:
    out = parse_verifier_json(
        '{"verified": ["a", "b",',
        provider_finish_reason="STOP",
    )
    assert out.get("_truncated") is not True
    assert out["verified"] == ["a", "b"]


def _route_state(
    *,
    synthesis_text: str,
    verification: dict[str, Any],
    synthesis_passes: int = 1,
    provider_rounds: int = 1,
    max_iterations: int = 5,
    confidence_threshold: float = 0.85,
    snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "synthesis_history": [synthesis_text] * synthesis_passes,
        "verification_history": [verification],
        "provider_responses": [{"responses": {}}] * provider_rounds,
        "max_iterations": max_iterations,
        "confidence_threshold": confidence_threshold,
        "iterative_config_snapshot": snapshot or {},
    }


def test_compute_refinement_route_verify_only_unverifiable_only() -> None:
    unver = [f"item {i}" for i in range(4)]
    st = _route_state(
        synthesis_text="Report body\nOVERALL_CONFIDENCE: 0.81\n",
        verification={"contradicted": [], "unverifiable": unver},
    )
    cmd = compute_refinement_route_command(st)  # type: ignore[arg-type]
    assert cmd.goto == "synthesize"
    upd = cmd.update or {}
    assert upd.get("last_route_followup_questions") == [f"Cite or verify: {u}" for u in unver]


def test_compute_refinement_route_fan_out_on_contradictions() -> None:
    st = _route_state(
        synthesis_text="Report body\nOVERALL_CONFIDENCE: 0.50\n",
        verification={"contradicted": ["c1"], "unverifiable": ["u1"]},
    )
    cmd = compute_refinement_route_command(st)  # type: ignore[arg-type]
    assert cmd.goto == "fan_out"


def test_compute_refinement_route_fan_out_high_unverifiable_low_confidence() -> None:
    unver = [f"item {i}" for i in range(7)]
    st = _route_state(
        synthesis_text="Report body\nOVERALL_CONFIDENCE: 0.75\n",
        verification={"contradicted": [], "unverifiable": unver},
        snapshot={
            "unverifiable_count_threshold_for_fanout": 3,
            "unverifiable_fanout_confidence_below": 0.8,
        },
    )
    cmd = compute_refinement_route_command(st)  # type: ignore[arg-type]
    assert cmd.goto == "fan_out"


def test_compute_refinement_route_stop_clean_verification() -> None:
    st = _route_state(
        synthesis_text="Report body\nOVERALL_CONFIDENCE: 0.92\n",
        verification={"contradicted": [], "unverifiable": []},
    )
    cmd = compute_refinement_route_command(st)  # type: ignore[arg-type]
    assert cmd.goto == "finalize"


class _CountingFan(LLMProvider):
    def __init__(self, name: str) -> None:
        self.name = name

    async def generate(
        self,
        prompt: str,
        *,
        enable_web_search: bool = True,
        max_output_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        _FANOUT_CALL_COUNTS[self.name] += 1
        return ProviderResponse(
            provider_name=self.name,
            model="m",
            text=f"fan-{self.name}\n",
            usage=ProviderUsage(),
            raw=None,
        )


class _CountingFanDistinct(LLMProvider):
    """Like ``_CountingFan`` but body changes each ``generate`` so iteration artifacts differ."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0

    async def generate(
        self,
        prompt: str,
        *,
        enable_web_search: bool = True,
        max_output_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        self.calls += 1
        _FANOUT_CALL_COUNTS[self.name] += 1
        return ProviderResponse(
            provider_name=self.name,
            model="m",
            text=f"fan-{self.name}-call-{self.calls}\n",
            usage=ProviderUsage(),
            raw=None,
        )


class _VerSkipFan(LLMProvider):
    """First round contradicts; second clears (no re-fan-out requested)."""

    def __init__(self) -> None:
        self.name = "gemini"
        self.k = 0

    async def generate(
        self,
        prompt: str,
        *,
        enable_web_search: bool = True,
        max_output_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        self.k += 1
        if self.k == 1:
            body = (
                '{"verified":[],"contradicted":["issue"],"unverifiable":[],'
                '"refresh_facts":false,"refan_out_providers":[],"refan_out_all":false}'
            )
        else:
            body = (
                '{"verified":[],"contradicted":[],"unverifiable":[],'
                '"refresh_facts":false,"refan_out_providers":[],"refan_out_all":false}'
            )
        return ProviderResponse(
            provider_name=self.name,
            model="m",
            text=body,
            usage=ProviderUsage(),
            raw=None,
        )


class _VerTwoContradictionsThenClear(LLMProvider):
    """First verification emits two contradictions (two router follow-ups); second clears."""

    def __init__(self) -> None:
        self.name = "gemini"
        self.k = 0

    async def generate(
        self,
        prompt: str,
        *,
        enable_web_search: bool = True,
        max_output_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        self.k += 1
        if self.k == 1:
            body = (
                '{"verified":[],"contradicted":["numbers look off","ratios disagree"],'
                '"unverifiable":[],'
                '"refresh_facts":false,"refan_out_providers":[],"refan_out_all":false}'
            )
        else:
            body = (
                '{"verified":[],"contradicted":[],"unverifiable":[],'
                '"refresh_facts":false,"refan_out_providers":[],"refan_out_all":false}'
            )
        return ProviderResponse(
            provider_name=self.name,
            model="m",
            text=body,
            usage=ProviderUsage(),
            raw=None,
        )


class _VerRefanOpenAi(LLMProvider):
    """Request only ``openai`` on the second fan-out round."""

    def __init__(self) -> None:
        self.name = "gemini"
        self.k = 0

    async def generate(
        self,
        prompt: str,
        *,
        enable_web_search: bool = True,
        max_output_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        self.k += 1
        if self.k == 1:
            body = (
                '{"verified":[],"contradicted":["issue"],"unverifiable":[],'
                '"refresh_facts":false,"refan_out_providers":["openai"],"refan_out_all":false}'
            )
        else:
            body = (
                '{"verified":[],"contradicted":[],"unverifiable":[],'
                '"refresh_facts":false,"refan_out_providers":[],"refan_out_all":false}'
            )
        return ProviderResponse(
            provider_name=self.name,
            model="m",
            text=body,
            usage=ProviderUsage(),
            raw=None,
        )


class _VerRefreshFacts(LLMProvider):
    def __init__(self) -> None:
        self.name = "gemini"
        self.k = 0

    async def generate(
        self,
        prompt: str,
        *,
        enable_web_search: bool = True,
        max_output_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        self.k += 1
        if self.k == 1:
            body = (
                '{"verified":[],"contradicted":["issue"],"unverifiable":[],'
                '"refresh_facts":true,"refan_out_providers":[],"refan_out_all":false}'
            )
        else:
            body = (
                '{"verified":[],"contradicted":[],"unverifiable":[],'
                '"refresh_facts":false,"refan_out_providers":[],"refan_out_all":false}'
            )
        return ProviderResponse(
            provider_name=self.name,
            model="m",
            text=body,
            usage=ProviderUsage(),
            raw=None,
        )


class _VerRaw(LLMProvider):
    """Verifier stub returning a fixed body (used inside `_GeminiSplit`)."""

    def __init__(self, raw: str) -> None:
        self.name = "gemini"
        self._raw = raw

    async def generate(
        self,
        prompt: str,
        *,
        enable_web_search: bool = True,
        max_output_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        return ProviderResponse(
            provider_name=self.name,
            model="m",
            text=self._raw,
            usage=ProviderUsage(),
            raw=None,
        )


@pytest.mark.asyncio
async def test_verify_node_persists_raw_response(tmp_path: Path) -> None:
    raw = "RAW_VERIFIER_BODY_FOR_DISK\n"
    reg = ProviderRegistry()
    reg.register("openai", lambda **_: _Txt("openai", "fan"))
    reg.register(
        "gemini",
        lambda **_: _GeminiSplit(
            _Txt("gemini", "ok\nOVERALL_CONFIDENCE: 0.95\n"),
            _VerRaw(raw),
        ),
    )
    cfg = _base_cfg()
    out = tmp_path / "o"
    out.mkdir()
    app = compile_refinement_workflow(registry=reg, checkpointer=MemorySaver())
    await app.ainvoke(_initial_state(cfg, out), config={"configurable": {"thread_id": "raw-ver"}})
    raw_path = out / "iterations" / "iteration_1_verify_raw.md"
    assert raw_path.is_file()
    assert raw_path.read_text(encoding="utf-8") == raw


@pytest.mark.asyncio
async def test_verify_registry_create_passes_verifier_gemini_model(tmp_path: Path) -> None:
    reg = _CaptureCreateRegistry()
    reg.register("openai", lambda **_: _Txt("openai", "fan"))
    reg.register(
        "gemini",
        lambda **_: _GeminiSplit(
            _Txt("gemini", "ok\nOVERALL_CONFIDENCE: 0.95\n"),
            _Txt(
                "gemini",
                '{"verified":[],"contradicted":[],"unverifiable":[],'
                '"refresh_facts":false,"refan_out_providers":[],"refan_out_all":false}',
            ),
        ),
    )
    cfg = _base_cfg(verifier_provider="gemini", verifier_model="gemini-custom-for-test")
    out = tmp_path / "o"
    out.mkdir()
    app = compile_refinement_workflow(registry=reg, checkpointer=MemorySaver())
    await app.ainvoke(_initial_state(cfg, out), config={"configurable": {"thread_id": "t-verify-model"}})
    gemini_kw = [kw for name, kw in reg.create_calls if name == "gemini"]
    assert any(kw.get("model") == "gemini-custom-for-test" for kw in gemini_kw)
    verify_row = next(kw for kw in gemini_kw if kw.get("model") == "gemini-custom-for-test")
    assert verify_row["gemini_cache_index"] is None
    assert verify_row["gemini_cache_ttl_s"] == 3600


@pytest.mark.asyncio
async def test_iteration_two_conditional_fanout_skips_fan_providers(tmp_path: Path) -> None:
    _FANOUT_CALL_COUNTS.clear()
    reg = ProviderRegistry()

    def _fan_factory(name: str):
        return lambda **__: _CountingFan(name)

    for n in ("anthropic", "openai", "grok"):
        reg.register(n, _fan_factory(n))
    synth = _SynthCalls("gemini")
    ver = _VerSkipFan()
    reg.register("gemini", lambda **_: _GeminiSplit(synth, ver))
    cfg = _base_cfg(
        providers=["anthropic", "openai", "grok"],
        synthesizer="gemini",
        facts_packet_enabled=False,
        conditional_fanout_enabled=True,
        fan_out_on_continue=False,
    )
    out = tmp_path / "o"
    out.mkdir()
    app = compile_refinement_workflow(registry=reg, checkpointer=MemorySaver())
    await app.ainvoke(_initial_state(cfg, out), config={"configurable": {"thread_id": "cond-skip"}})
    assert _FANOUT_CALL_COUNTS["anthropic"] == 1
    assert _FANOUT_CALL_COUNTS["openai"] == 1
    assert _FANOUT_CALL_COUNTS["grok"] == 1


@pytest.mark.asyncio
async def test_iteration_two_router_followups_rerun_fan_out_under_conditional(
    tmp_path: Path,
) -> None:
    """Router continue(fan_out) with follow-ups must invoke providers when conditional fan-out is on."""
    _FANOUT_CALL_COUNTS.clear()
    reg = ProviderRegistry()
    _openai_fan = _CountingFanDistinct("openai")
    reg.register("openai", lambda **_: _openai_fan)
    synth = _SynthCalls("gemini")
    ver = _VerTwoContradictionsThenClear()
    reg.register("gemini", lambda **_: _GeminiSplit(synth, ver))
    cfg = _base_cfg(
        providers=["openai"],
        synthesizer="gemini",
        facts_packet_enabled=False,
        conditional_fanout_enabled=True,
    )
    assert cfg.fan_out_on_continue is True
    out = tmp_path / "o-router-fanout"
    out.mkdir()
    app = compile_refinement_workflow(registry=reg, checkpointer=MemorySaver())
    await app.ainvoke(_initial_state(cfg, out), config={"configurable": {"thread_id": "router-fanout"}})
    assert _FANOUT_CALL_COUNTS["openai"] >= 2
    p1 = (out / "iterations" / "iteration_1_providers.md").read_text(encoding="utf-8")
    p2 = (out / "iterations" / "iteration_2_providers.md").read_text(encoding="utf-8")
    assert p1 != p2
    assert "fan-out skipped" not in p2


@pytest.mark.asyncio
async def test_iteration_two_refan_out_providers_only_openai(tmp_path: Path) -> None:
    _FANOUT_CALL_COUNTS.clear()
    reg = ProviderRegistry()
    reg.register("openai", lambda **_: _CountingFan("openai"))
    reg.register("grok", lambda **_: _CountingFan("grok"))
    synth = _SynthCalls("gemini")
    ver = _VerRefanOpenAi()
    reg.register("gemini", lambda **_: _GeminiSplit(synth, ver))
    cfg = _base_cfg(
        providers=["openai", "grok"],
        synthesizer="gemini",
        facts_packet_enabled=False,
        conditional_fanout_enabled=True,
    )
    out = tmp_path / "o2"
    out.mkdir()
    app = compile_refinement_workflow(registry=reg, checkpointer=MemorySaver())
    await app.ainvoke(_initial_state(cfg, out), config={"configurable": {"thread_id": "refan-partial"}})
    assert _FANOUT_CALL_COUNTS["openai"] == 2
    assert _FANOUT_CALL_COUNTS["grok"] == 1


@pytest.mark.asyncio
async def test_refresh_facts_reruns_extractor(monkeypatch: Any, tmp_path: Path) -> None:
    calls = {"n": 0}

    async def _track_extract(**kwargs: Any) -> str:
        calls["n"] += 1
        return "# Market facts (frozen from iteration 1)\n\n- extracted\n"

    monkeypatch.setattr("equity_analyst.iterative.extract_facts_packet", _track_extract)

    reg = ProviderRegistry()
    reg.register("openai", lambda **_: _Txt("openai", "fan"))
    synth = _SynthCalls("gemini")
    ver = _VerRefreshFacts()
    reg.register("gemini", lambda **_: _GeminiSplit(synth, ver))
    cfg = _base_cfg(
        facts_packet_enabled=True,
        conditional_fanout_enabled=False,
    )
    out = tmp_path / "o3"
    out.mkdir()
    app = compile_refinement_workflow(registry=reg, checkpointer=MemorySaver())
    await app.ainvoke(_initial_state(cfg, out), config={"configurable": {"thread_id": "refresh"}})
    assert calls["n"] >= 2


def test_parse_verifier_json_sections_to_revise() -> None:
    raw = '{"verified":[],"contradicted":[],"unverifiable":[],"sections_to_revise":[9, 11, 3, 13, "2"]}'
    out = parse_verifier_json(raw)
    assert out["sections_to_revise"] == [9, 11, 3, 2]


class _CaptureOpenAiFan(LLMProvider):
    prompts: ClassVar[list[str]] = []

    def __init__(self) -> None:
        self.name = "openai"

    async def generate(
        self,
        prompt: str,
        *,
        enable_web_search: bool = True,
        max_output_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        _CaptureOpenAiFan.prompts.append(prompt)
        return ProviderResponse(
            provider_name=self.name,
            model="m",
            text="fan\n",
            usage=ProviderUsage(),
            raw=None,
        )


@pytest.mark.asyncio
async def test_iteration_two_fan_out_prompt_includes_refinement_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _fake_facts(**kwargs: Any) -> str:
        return "# Market facts (frozen)\n\n- PCR: 0.5\n"

    monkeypatch.setattr("equity_analyst.iterative.extract_facts_packet", _fake_facts)

    _CaptureOpenAiFan.prompts.clear()
    reg = ProviderRegistry()
    reg.register("openai", lambda **_: _CaptureOpenAiFan())
    synth = _SynthCalls("gemini")
    ver = _VerCalls("gemini")
    reg.register("gemini", lambda **_: _GeminiSplit(synth, ver))
    cfg = _base_cfg(
        facts_packet_enabled=True,
        conditional_fanout_enabled=False,
    )
    out = tmp_path / "refine-prompt"
    out.mkdir()
    app = compile_refinement_workflow(registry=reg, checkpointer=MemorySaver())
    await app.ainvoke(_initial_state(cfg, out), config={"configurable": {"thread_id": "refine-prompt"}})
    assert len(_CaptureOpenAiFan.prompts) >= 2
    second = _CaptureOpenAiFan.prompts[1]
    assert "# REFINEMENT MODE" in second
    assert "DO NOT re-derive" in second
    assert "# Prior synthesis (round 1)" in second
    assert "FACTS (frozen from iteration 1" in second


class _CaptureOpenAiFan2(LLMProvider):
    prompts: ClassVar[list[str]] = []

    def __init__(self) -> None:
        self.name = "openai"

    async def generate(
        self,
        prompt: str,
        *,
        enable_web_search: bool = True,
        max_output_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        _CaptureOpenAiFan2.prompts.append(prompt)
        return ProviderResponse(
            provider_name=self.name,
            model="m",
            text="fan\n",
            usage=ProviderUsage(),
            raw=None,
        )


@pytest.mark.asyncio
async def test_refinement_mode_prompt_respects_config_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _fake_facts(**kwargs: Any) -> str:
        return "# Market facts (frozen)\n\n- PCR: 0.5\n"

    monkeypatch.setattr("equity_analyst.iterative.extract_facts_packet", _fake_facts)

    _CaptureOpenAiFan2.prompts.clear()
    reg = ProviderRegistry()
    reg.register("openai", lambda **_: _CaptureOpenAiFan2())
    synth = _SynthCalls("gemini")
    ver = _VerCalls("gemini")
    reg.register("gemini", lambda **_: _GeminiSplit(synth, ver))
    cfg = _base_cfg(
        facts_packet_enabled=True,
        conditional_fanout_enabled=False,
        refinement_mode_prompt_enabled=False,
    )
    out2 = tmp_path / "refine-off"
    out2.mkdir()
    app = compile_refinement_workflow(registry=reg, checkpointer=MemorySaver())
    await app.ainvoke(_initial_state(cfg, out2), config={"configurable": {"thread_id": "refine-off"}})
    assert len(_CaptureOpenAiFan2.prompts) >= 2
    second = _CaptureOpenAiFan2.prompts[1]
    assert "# REFINEMENT MODE" not in second


def test_round_summary_short_text_passes_through() -> None:
    text = "Synthesis body shorter than max_chars."
    out = round_summary_for_changelog(text, iteration_index=1)
    assert out == text
    assert "abridged" not in out


def test_round_summary_truncates_at_paragraph_boundary_with_pointer() -> None:
    paragraph_a = "A" * 800
    paragraph_b = "B" * 800
    paragraph_c = "C" * 800
    text = f"{paragraph_a}\n\n{paragraph_b}\n\n{paragraph_c}"
    assert len(text) > CHANGELOG_ROUND_SUMMARY_MAX_CHARS

    out = round_summary_for_changelog(text, iteration_index=2)

    pointer = "iterations/iteration_2_synthesis.md"
    assert pointer in out
    body = out.split("\n\n…")[0]
    assert body.endswith(("A", "B"))
    assert not body.endswith("…")
    assert "C" * 50 not in body, "Third paragraph must be excluded"


def test_round_summary_never_cuts_mid_sentence() -> None:
    # Mimic the real-world regression: full synthesis is a continuous block
    # with a sentence trailing into the truncation window — the preview must
    # NOT end on a partial word like "The pr..." (this was the user's bug).
    sentence_one = (
        "Key disagreements: "
        + "Anthropic and OpenAI converged on a Friday close of $6.48. "
    ) * 30
    sentence_two = (
        "Temporal/Methodological disagreement: The prompt asks for Monday "
        "BMO analysis, but Anthropic and OpenAI correctly identify AMC."
    )
    text = sentence_one + sentence_two

    out = round_summary_for_changelog(text, iteration_index=1)

    assert "abridged" in out
    body = out.split("\n\n…")[0]
    # Should end on a sentence terminator (period/!/?), not mid-word.
    last_char = body.rstrip()[-1]
    assert last_char in {".", "!", "?"}, (
        f"Preview ended on {last_char!r} — should end at a sentence boundary"
    )
    # The specific mid-word fragment "The pr" (the bug shape) must not be the tail.
    assert not body.rstrip().endswith("The pr")


class _LongSynth(LLMProvider):
    """Synthesis body long enough to trigger legacy changelog abridgement."""

    def __init__(self, name: str, *, pad_len: int = 2500, tail: str = "UNIQUE_INLINE_TAIL_ZZ") -> None:
        self.name = name
        self._text = ("x" * pad_len) + "\n\n" + tail + "\n\nOVERALL_CONFIDENCE: 0.95\n"

    async def generate(
        self,
        prompt: str,
        *,
        enable_web_search: bool = True,
        max_output_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        return ProviderResponse(
            provider_name=self.name,
            model="m",
            text=self._text,
            usage=ProviderUsage(),
            raw=None,
        )


@pytest.mark.asyncio
async def test_finalize_synthesis_md_changelog_full_text_no_abridged_placeholder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_sources: list[tuple[str, str]] = []

    def _capture_pdf(**kwargs: Any) -> None:
        p = kwargs["md_path"]
        pdf_sources.append((p.name, kwargs["markdown_text"]))

    monkeypatch.setattr("equity_analyst.iterative.maybe_write_pdf_sibling", _capture_pdf)

    reg = ProviderRegistry()
    reg.register("openai", lambda **_: _Txt("openai", "fan"))
    reg.register(
        "gemini",
        lambda **_: _GeminiSplit(
            _LongSynth("gemini"),
            _Txt(
                "gemini",
                '{"verified":[],"contradicted":[],"unverifiable":[],'
                '"refresh_facts":false,"refan_out_providers":[],"refan_out_all":false}',
            ),
        ),
    )
    cfg = _base_cfg()
    out = tmp_path / "full-changelog"
    out.mkdir()
    app = compile_refinement_workflow(registry=reg, checkpointer=MemorySaver())
    await app.ainvoke(_initial_state(cfg, out), config={"configurable": {"thread_id": "full-changelog"}})

    md = (out / "synthesis.md").read_text(encoding="utf-8")
    pre_final, _, _post = md.partition("## Final synthesis (last round)")
    assert "abridged" not in pre_final
    assert "UNIQUE_INLINE_TAIL_ZZ" in pre_final
    assert "### Round 1 synthesis (summary)" not in pre_final
    assert "### Round 1 synthesis\n" in pre_final

    syn_pdf = next((t for n, t in pdf_sources if n == "synthesis.md"), "")
    assert syn_pdf
    assert "abridged" not in syn_pdf
    assert "UNIQUE_INLINE_TAIL_ZZ" in syn_pdf


@pytest.mark.asyncio
async def test_finalize_synthesis_md_changelog_abridged_when_final_report_full_synthesis_off(
    tmp_path: Path,
) -> None:
    reg = ProviderRegistry()
    reg.register("openai", lambda **_: _Txt("openai", "fan"))
    reg.register(
        "gemini",
        lambda **_: _GeminiSplit(
            _LongSynth("gemini"),
            _Txt(
                "gemini",
                '{"verified":[],"contradicted":[],"unverifiable":[],'
                '"refresh_facts":false,"refan_out_providers":[],"refan_out_all":false}',
            ),
        ),
    )
    cfg = _base_cfg(final_report_full_synthesis=False)
    out = tmp_path / "abbr-changelog"
    out.mkdir()
    st = _initial_state(cfg, out)
    app = compile_refinement_workflow(registry=reg, checkpointer=MemorySaver())
    await app.ainvoke(st, config={"configurable": {"thread_id": "abbr-changelog"}})

    md = (out / "synthesis.md").read_text(encoding="utf-8")
    pre_final, _, post = md.partition("## Final synthesis (last round)")
    assert "abridged" in pre_final
    assert "UNIQUE_INLINE_TAIL_ZZ" not in pre_final
    assert "### Round 1 synthesis (summary)" in pre_final
    assert "UNIQUE_INLINE_TAIL_ZZ" in post
