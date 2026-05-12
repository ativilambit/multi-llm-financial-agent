from __future__ import annotations

import json
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import MemorySaver

from equity_analyst.iterative import build_initial_refinement_state, compile_refinement_workflow
from equity_analyst.prompt_export import (
    PromptExporter,
    export_prompt,
    logical_prompt_split,
    maybe_export_prompt,
    prompt_call_context,
    prompts_export_enabled,
    use_prompt_exporter,
)
from equity_analyst.prompting import RenderedPrompt
from equity_analyst.providers.openai_provider import OpenAIProvider
from equity_analyst.providers.registry import ProviderRegistry
from tests.test_iterative import _base_cfg, _GeminiSplit, _initial_state, _Txt
from tests.test_providers import _FakeOpenAIClient


@pytest.mark.asyncio
async def test_prompt_exporter_sequence_and_index(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    exp = PromptExporter(run_dir)
    await exp.record(
        provider="openai",
        model="gpt-4o",
        system="SYS",
        user="USR",
        config={"model": "gpt-4o", "web_search": False},
    )
    await exp.record(
        provider="gemini",
        model="gemini-3-pro",
        system="S2",
        user="U2",
        config={"model": "gemini-3-pro", "web_search": True},
    )
    idx = exp.finalize_index()
    assert idx is not None
    text = idx.read_text(encoding="utf-8")
    assert "0001" in text and "0002" in text
    assert "| unknown |" in text or "unknown" in text
    p1 = run_dir / "prompts" / "0001_unknown_openai_gpt-4o.md"
    assert p1.is_file()
    body = p1.read_text(encoding="utf-8")
    assert "SYS" in body and "USR" in body


def test_export_prompt_writes_sidecar(tmp_path: Path) -> None:
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    ctx = {"symbol": "ZZ", "n": 1}
    path = export_prompt(
        run_dir,
        node="fan_out",
        iteration=1,
        provider="openai",
        model="gpt-4o",
        system="A",
        user="B",
        config={"model": "gpt-4o"},
        sequence=7,
        context_sidecar=ctx,
    )
    assert path.name == "0007_fan_out_openai_gpt-4o.md"
    side = path.with_suffix(".context.json")
    assert side.is_file()
    assert json.loads(side.read_text(encoding="utf-8")) == ctx


@pytest.mark.asyncio
async def test_maybe_export_respects_logical_split(tmp_path: Path) -> None:
    run_dir = tmp_path / "r2"
    run_dir.mkdir()
    with (
        use_prompt_exporter(run_dir),
        prompt_call_context(node="synthesize", iteration=2),
        logical_prompt_split("LOGSYS", "LOGUSER"),
    ):
        await maybe_export_prompt(
            provider="openai",
            model="m",
            system="ignored",
            user="ignored",
            config={"k": 1},
        )
    md = next(p for p in (run_dir / "prompts").glob("*.md") if p.name != "prompts_index.md")
    t = md.read_text(encoding="utf-8")
    assert "LOGSYS" in t and "LOGUSER" in t


def test_prompts_export_enabled_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXPORT_PROMPTS", "0")
    assert prompts_export_enabled() is False
    monkeypatch.setenv("EXPORT_PROMPTS", "1")
    assert prompts_export_enabled() is True


@pytest.mark.asyncio
async def test_iterative_run_writes_prompts_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EXPORT_PROMPTS", raising=False)
    reg = ProviderRegistry()
    reg.register("openai", lambda **_: OpenAIProvider(model="gpt-5.5", client=_FakeOpenAIClient()))  # type: ignore[arg-type]
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
    rendered = RenderedPrompt(
        template_path="t",
        text="PROMPT",
        context={"symbol": "MNDY", "x": 1},
        user_message_text="U",
    )
    st = build_initial_refinement_state(cfg=cfg, rendered=rendered, output_dir=out)
    st["max_iterations"] = 5
    st["confidence_threshold"] = 0.85
    st["enable_web_search"] = False
    app = compile_refinement_workflow(registry=reg, checkpointer=MemorySaver())
    with use_prompt_exporter(out):
        await app.ainvoke(st, config={"configurable": {"thread_id": "tprompt"}})

    prompts = out / "prompts"
    assert prompts.is_dir()
    assert (prompts / "prompts_index.md").is_file()
    ctx_files = list(prompts.glob("*.context.json"))
    assert any("fan_out" in p.name for p in ctx_files)


@pytest.mark.asyncio
async def test_iterative_respects_export_prompts_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXPORT_PROMPTS", "0")
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
    out = tmp_path / "o2"
    out.mkdir()
    app = compile_refinement_workflow(registry=reg, checkpointer=MemorySaver())
    with use_prompt_exporter(out):
        await app.ainvoke(_initial_state(cfg, out), config={"configurable": {"thread_id": "t-off"}})
    assert not (out / "prompts").exists()
