from __future__ import annotations

import asyncio
import json
import logging
import operator
import re
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from equity_analyst.config import RunConfig
from equity_analyst.providers.registry import ProviderRegistry
from equity_analyst.synthesizer import Synthesizer
from equity_analyst.types import ProviderResponse, ProviderUsage

logger = logging.getLogger(__name__)

VERIFIER_INSTRUCTION = """You are a financial fact-checker. You receive a synthesis about an equity/options thesis.

Use web search where helpful. Focus on numerical claims (PCR, analyst targets, historical post-earnings moves, short interest, price levels).

Reply with ONLY valid JSON (no markdown fences) in this exact shape:
{"verified": ["string claims that check out"], "contradicted": ["string claims that conflict with sources"], "unverifiable": ["string claims that cannot be verified from available data"]}

Use empty arrays when appropriate. Be concise; each array entry one short sentence."""

_OVERALL_CONFIDENCE_RE = re.compile(
    r"^OVERALL_CONFIDENCE:\s*([0-9]+(?:\.[0-9]+)?)\s*$",
    re.MULTILINE | re.IGNORECASE,
)


def parse_overall_confidence(text: str) -> float | None:
    m = _OVERALL_CONFIDENCE_RE.search(text)
    if not m:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    if v < 0.0 or v > 1.0:
        return None
    return v


def _response_to_dict(r: ProviderResponse) -> dict[str, Any]:
    return {
        "provider_name": r.provider_name,
        "model": r.model,
        "text": r.text,
        "usage": asdict(r.usage),
        "latency_s": r.latency_s,
    }


def _dict_to_response(d: dict[str, Any]) -> ProviderResponse:
    u = d.get("usage") or {}
    return ProviderResponse(
        provider_name=str(d["provider_name"]),
        model=str(d["model"]),
        text=str(d["text"]),
        usage=ProviderUsage(
            input_tokens=u.get("input_tokens"),
            output_tokens=u.get("output_tokens"),
            total_tokens=u.get("total_tokens"),
        ),
        latency_s=d.get("latency_s"),
        raw=None,
    )


def _parse_verifier_json(text: str) -> dict[str, list[str]]:
    t = text.strip()
    if t.startswith("```"):
        lines = t.split("\n")
        if lines:
            lines = lines[1:]
        t = "\n".join(lines)
        if "```" in t:
            t = t.rsplit("```", 1)[0].strip()
    try:
        data = json.loads(t)
    except json.JSONDecodeError:
        return {"verified": [], "contradicted": [], "unverifiable": []}
    out: dict[str, list[str]] = {"verified": [], "contradicted": [], "unverifiable": []}
    for k in out:
        v = data.get(k)
        if isinstance(v, list):
            out[k] = [str(x) for x in v]
    return out


class RefinementState(TypedDict, total=False):
    symbol: str
    original_prompt: str
    max_iterations: int
    confidence_threshold: float
    enable_web_search: bool
    providers: list[str]
    synthesizer_name: str
    verifier_name: str
    output_dir: str
    provider_responses: Annotated[list[dict[str, Any]], operator.add]
    synthesis_history: Annotated[list[str], operator.add]
    verification_history: Annotated[list[dict[str, list[str]]], operator.add]
    followup_questions: Annotated[list[str], operator.add]
    final_report: str


def _make_refinement_nodes(registry: ProviderRegistry) -> dict[str, Any]:
    async def fan_out(state: RefinementState) -> dict[str, Any]:
        out = Path(state["output_dir"])
        extra = "\n\n".join(state.get("followup_questions", []))
        body = state["original_prompt"]
        if extra:
            body = f"{body}\n\n### Follow-up verification targets\n{extra}"
        round_idx = len(state.get("provider_responses", []))
        max_it = state["max_iterations"]
        logger.info(
            "Node fan_out iteration=%s max_iterations=%s providers=%s output_dir=%s",
            round_idx + 1,
            max_it,
            list(state["providers"]),
            str(out.resolve()),
        )

        async def _one(name: str) -> ProviderResponse:
            p = registry.create(name)
            return await p.generate(body, enable_web_search=state["enable_web_search"])

        names = state["providers"]
        res_list = await asyncio.gather(*[_one(n) for n in names])
        responses: dict[str, ProviderResponse] = {r.provider_name: r for r in res_list}

        iter_dir = out / "iterations"
        iter_dir.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        for name, resp in responses.items():
            lines.append(f"## {name}\n\n{resp.text}\n")
        (iter_dir / f"iteration_{round_idx + 1}_providers.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )

        ser = {"responses": {k: _response_to_dict(v) for k, v in responses.items()}}
        return {"provider_responses": [ser]}

    async def synthesize(state: RefinementState) -> dict[str, Any]:
        out = Path(state["output_dir"])
        round_idx = len(state.get("synthesis_history", []))
        logger.info(
            "Node synthesize iteration=%s max_iterations=%s synthesizer=%s",
            round_idx + 1,
            state["max_iterations"],
            state["synthesizer_name"],
        )
        last = state["provider_responses"][-1]
        raw = last["responses"]
        resp_map: dict[str, ProviderResponse] = {k: _dict_to_response(v) for k, v in raw.items()}
        synth_backend = registry.create(state["synthesizer_name"])
        syn = Synthesizer(synth_backend)
        result = await syn.synthesize(
            original_prompt=state["original_prompt"],
            responses=resp_map,
            enable_web_search=state["enable_web_search"],
        )
        text = result.response.text
        round_idx = len(state.get("synthesis_history", []))
        iter_dir = out / "iterations"
        (iter_dir / f"iteration_{round_idx + 1}_synthesis.md").write_text(text + "\n", encoding="utf-8")
        return {"synthesis_history": [text]}

    async def verify(state: RefinementState) -> dict[str, Any]:
        out = Path(state["output_dir"])
        round_idx = len(state.get("verification_history", []))
        logger.info(
            "Node verify iteration=%s max_iterations=%s verifier=%s",
            round_idx + 1,
            state["max_iterations"],
            state["verifier_name"],
        )
        syn = state["synthesis_history"][-1]
        prompt = f"{VERIFIER_INSTRUCTION}\n\n### Synthesis\n{syn}"
        verifier = registry.create(state["verifier_name"])
        resp = await verifier.generate(prompt, enable_web_search=state["enable_web_search"])
        data = _parse_verifier_json(resp.text)
        round_idx = len(state.get("verification_history", []))
        iter_dir = out / "iterations"
        (iter_dir / f"iteration_{round_idx + 1}_verify.md").write_text(
            json.dumps(data, indent=2) + "\n", encoding="utf-8"
        )
        return {"verification_history": [data]}

    def route(state: RefinementState) -> Command[Any]:
        syn = state["synthesis_history"][-1]
        ver = state["verification_history"][-1]
        conf = parse_overall_confidence(syn)
        n_rounds = len(state["provider_responses"])
        contrad = ver.get("contradicted") or []
        max_it = state["max_iterations"]
        logger.info(
            "Node route rounds_completed=%s max_iterations=%s overall_confidence=%s contradicted=%s",
            n_rounds,
            max_it,
            f"{conf:.4f}" if conf is not None else "none",
            len(contrad),
        )
        if n_rounds >= state["max_iterations"]:
            logger.info("Route decision: finalize (max_iterations reached)")
            return Command(goto="finalize")
        if conf is not None and conf >= state["confidence_threshold"] and not contrad:
            logger.info(
                "Route decision: finalize (confidence >= threshold and no contradictions) threshold=%s",
                state["confidence_threshold"],
            )
            return Command(goto="finalize")
        qs: list[str] = []
        for c in contrad:
            qs.append(f"Resolve with primary sources: {c}")
        for u in ver.get("unverifiable") or []:
            qs.append(f"Cite or verify: {u}")
        logger.info("Route decision: continue (fan_out) followups=%s", len(qs))
        return Command(goto="fan_out", update={"followup_questions": qs})

    async def finalize(state: RefinementState) -> dict[str, Any]:
        out = Path(state["output_dir"])
        logger.info("Node finalize output_dir=%s rounds=%s", str(out.resolve()), len(state["provider_responses"]))
        parts: list[str] = [
            f"# Refined equity report: {state['symbol']}\n",
            "## Iteration changelog\n",
        ]
        for i, syn in enumerate(state["synthesis_history"], start=1):
            parts.append(f"### Round {i} synthesis (summary)\n\n{syn[:1500]}...\n\n")
        parts.append("## Verification summary\n\n")
        for i, ver in enumerate(state["verification_history"], start=1):
            parts.append(f"### Round {i}\n```json\n{json.dumps(ver, indent=2)}\n```\n\n")
        parts.append("## Final synthesis (last round)\n\n")
        parts.append(state["synthesis_history"][-1])
        report = "\n".join(parts)
        out.mkdir(parents=True, exist_ok=True)
        (out / "synthesis.md").write_text(report + "\n", encoding="utf-8")
        iter_dir = out / "iterations"
        for i, syn in enumerate(state["synthesis_history"], start=1):
            ver = state["verification_history"][i - 1] if i <= len(state["verification_history"]) else {}
            block = f"# Iteration {i}\n\n## Synthesis\n\n{syn}\n\n## Verification\n\n{json.dumps(ver, indent=2)}\n"
            (iter_dir / f"iteration_{i}.md").write_text(block, encoding="utf-8")
        return {"final_report": report}

    return {
        "fan_out": fan_out,
        "synthesize": synthesize,
        "verify": verify,
        "route": route,
        "finalize": finalize,
    }


def compile_refinement_workflow(
    *,
    registry: ProviderRegistry,
    checkpointer: BaseCheckpointSaver[Any],
    interrupt_before: Sequence[str] | None = None,
) -> Any:
    nodes = _make_refinement_nodes(registry)
    g: StateGraph[RefinementState] = StateGraph(RefinementState)
    for name, fn in nodes.items():
        g.add_node(name, fn)
    g.add_edge(START, "fan_out")
    g.add_edge("fan_out", "synthesize")
    g.add_edge("synthesize", "verify")
    g.add_edge("verify", "route")
    g.add_edge("finalize", END)
    ib: list[str] | None = list(interrupt_before) if interrupt_before else None
    compiled = g.compile(
        checkpointer=checkpointer,
        interrupt_before=ib,
    )
    node_names = sorted(n for n in compiled.get_graph().nodes if not str(n).startswith("__"))
    logger.debug("Compiled refinement workflow nodes=%s", node_names)
    return compiled


def build_initial_refinement_state(
    *,
    cfg: RunConfig,
    rendered_text: str,
    output_dir: Path,
) -> RefinementState:
    return {
        "symbol": cfg.symbol,
        "original_prompt": rendered_text,
        "max_iterations": 3,
        "confidence_threshold": 0.85,
        "enable_web_search": True,
        "providers": list(cfg.providers),
        "synthesizer_name": cfg.synthesizer,
        "verifier_name": "anthropic",
        "output_dir": str(output_dir.resolve()),
    }


def dry_run_compile_only(*, registry: ProviderRegistry) -> list[str]:
    from langgraph.checkpoint.memory import MemorySaver

    app = compile_refinement_workflow(registry=registry, checkpointer=MemorySaver())
    nodes = app.get_graph().nodes
    out = sorted(n for n in nodes if not str(n).startswith("__"))
    logger.debug("Dry-run graph inspection nodes=%s", out)
    return out
