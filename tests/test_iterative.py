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
    parse_verifier_json,
)
from equity_analyst.prompting import RenderedPrompt
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
        "synthesizer": "gemini",
    }
    d.update(kwargs)
    return RunConfig.model_validate(d)


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
            _Txt("gemini", '{"verified":[],"contradicted":[],"unverifiable":[]}'),
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
            _Txt("gemini", '{"verified":[],"contradicted":[],"unverifiable":[]}'),
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
            _Txt("gemini", '{"verified":[],"contradicted":[],"unverifiable":[]}'),
        ),
    )
    cfg = _base_cfg()
    out = tmp_path / "o"
    out.mkdir()
    app = compile_refinement_workflow(registry=reg, checkpointer=MemorySaver())
    final = await app.ainvoke(_initial_state(cfg, out), config={"configurable": {"thread_id": "t2"}})
    assert len(final["provider_responses"]) == 3


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
            _Txt("gemini", '{"verified":[],"contradicted":[],"unverifiable":[]}'),
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
    out = parse_verifier_json(_ITERATION_1_TRUNCATED_VERIFIER_RAW)
    assert out.get("_truncated") is True
    assert len(out["verified"]) >= 8
    assert len(out["contradicted"]) == 2
    assert out["unverifiable"] == []


def test_parse_verifier_json_repairs_minimal_truncation() -> None:
    out = parse_verifier_json('{"verified": ["a", "b",')
    assert out.get("_truncated") is True
    assert out["verified"] == ["a", "b"]
    assert out["contradicted"] == []
    assert out["unverifiable"] == []


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
            _Txt("gemini", '{"verified":[],"contradicted":[],"unverifiable":[]}'),
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
