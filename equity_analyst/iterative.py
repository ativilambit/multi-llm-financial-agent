from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import operator
import re
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from equity_analyst.config import ProviderConfig, RunConfig
from equity_analyst.provider_runtime import (
    effective_web_search,
    failure_response,
    failure_response_from_completed,
    partition_provider_responses,
    run_error_record,
)
from equity_analyst.providers.registry import ProviderRegistry
from equity_analyst.retry import async_retry_call
from equity_analyst.synthesizer import (
    SynthesisResult,
    Synthesizer,
    format_synthesis_artifact_markdown,
)
from equity_analyst.types import ProviderResponse, ProviderUsage

logger = logging.getLogger(__name__)

VERIFIER_INSTRUCTION = """You are a financial fact-checker. You receive an excerpt of a synthesis focused on
numerical and factual claims about an equity/options thesis (and lines mentioning low confidence).

Use web search only when needed to check those claims. Do not spend effort re-verifying narrative sections
that are not represented in the excerpt.

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


def _excerpt_for_verifier(synthesis: str, *, max_chars: int = 12000) -> str:
    lines = synthesis.splitlines()
    picked: list[str] = []
    for line in lines:
        low = line.lower()
        if "confidence" in low and "low" in low:
            picked.append(line)
            continue
        if any(ch.isdigit() for ch in line) or "$" in line or "%" in line or "pcr" in low:
            picked.append(line)
    body = "\n".join(picked) if picked else synthesis
    if len(body) > max_chars:
        body = body[:max_chars] + "\n...(truncated for verification scope)..."
    return body


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


def merge_timing_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    acc: dict[int, dict[str, float]] = defaultdict(dict)
    for ev in events:
        if not isinstance(ev, dict):
            continue
        it = int(ev.get("iteration", 0))
        for k in ("providers_parallel_wall_s", "synthesis_wall_s", "verify_wall_s"):
            if k in ev and isinstance(ev[k], (int, float)):
                acc[it][k] = float(ev[k])
    rounds_out: dict[str, Any] = {}
    total_seq = 0.0
    for it in sorted(acc):
        d = acc[it]
        pw = d.get("providers_parallel_wall_s", 0.0)
        sw = d.get("synthesis_wall_s", 0.0)
        vw = d.get("verify_wall_s", 0.0)
        seq = pw + sw + vw
        total_seq += seq
        rounds_out[str(it)] = {
            "providers_parallel_wall_s": round(pw, 3),
            "synthesis_wall_s": round(sw, 3),
            "verify_wall_s": round(vw, 3),
            "sequential_round_wall_s": round(seq, 3),
        }
    return {"iterations": rounds_out, "total_sequential_wall_s": round(total_seq, 3)}


class RefinementState(TypedDict, total=False):
    symbol: str
    original_prompt: str
    max_iterations: int
    confidence_threshold: float
    enable_web_search: bool
    providers: list[str]
    provider_configs: list[dict[str, Any]]
    max_output_tokens: int
    verifier_max_output_tokens: int
    request_timeout_s: float
    retry_max_attempts: int
    retry_base_delay_s: float
    synthesizer_max_input_tokens: int
    synthesizer_name: str
    verifier_name: str
    output_dir: str
    provider_responses: Annotated[list[dict[str, Any]], operator.add]
    synthesis_history: Annotated[list[str], operator.add]
    verification_history: Annotated[list[dict[str, list[str]]], operator.add]
    followup_questions: Annotated[list[str], operator.add]
    timing_events: Annotated[list[dict[str, Any]], operator.add]
    error_events: Annotated[list[dict[str, Any]], operator.add]
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

        pcs_raw = state.get("provider_configs")
        if not pcs_raw:
            pcs_raw = [{"name": n, "web_search": None, "request_timeout_s": None} for n in state["providers"]]
        pcs = [ProviderConfig.model_validate(d) for d in pcs_raw]
        cfg_req_timeout = float(state.get("request_timeout_s", 180.0))
        mot = int(state.get("max_output_tokens", 4096))
        names = [pc.name for pc in pcs]
        retry_max = int(state.get("retry_max_attempts", 3))
        retry_base = float(state.get("retry_base_delay_s", 2.0))

        async def _heartbeat(stop: asyncio.Event, provider_names: list[str]) -> None:
            start = time.perf_counter()
            while True:
                try:
                    await asyncio.wait_for(stop.wait(), timeout=30.0)
                    return
                except TimeoutError:
                    logger.info(
                        "Still waiting on providers=%s (%ss elapsed)",
                        provider_names,
                        int(time.perf_counter() - start),
                    )

        async def _run_one(pc: ProviderConfig) -> ProviderResponse:
            t0 = time.perf_counter()
            p = registry.create(pc.name)
            ws = effective_web_search(run_default=state["enable_web_search"], pc=pc)
            to = float(pc.request_timeout_s) if pc.request_timeout_s is not None else cfg_req_timeout

            async def _attempt() -> ProviderResponse:
                return await p.generate(body, enable_web_search=ws, max_output_tokens=mot)

            try:
                return await asyncio.wait_for(
                    async_retry_call(
                        _attempt,
                        provider=pc.name,
                        max_attempts=retry_max,
                        base_delay_s=retry_base,
                    ),
                    timeout=to,
                )
            except asyncio.CancelledError:
                raise
            except TimeoutError as exc:
                return failure_response_from_completed(pc.name, exc, started_perf=t0)

        stop_hb = asyncio.Event()
        hb = asyncio.create_task(_heartbeat(stop_hb, names))
        batch_t0 = time.perf_counter()
        try:
            res_list = await asyncio.gather(*[_run_one(pc) for pc in pcs], return_exceptions=True)
        finally:
            stop_hb.set()
            hb.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await hb

        responses: dict[str, ProviderResponse] = {}
        for pc, item in zip(pcs, res_list, strict=True):
            if isinstance(item, ProviderResponse):
                responses[pc.name] = item
            elif isinstance(item, Exception):
                responses[pc.name] = failure_response(pc.name, item, latency_s=None)
            else:
                raise item

        parallel_wall = time.perf_counter() - batch_t0

        iter_dir = out / "iterations"
        iter_dir.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        for name, resp in responses.items():
            lines.append(f"## {name}\n\n{resp.text}\n")
        (iter_dir / f"iteration_{round_idx + 1}_providers.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )

        ser = {"responses": {k: _response_to_dict(v) for k, v in responses.items()}}
        it_no = round_idx + 1
        return {
            "provider_responses": [ser],
            "timing_events": [
                {"iteration": it_no, "providers_parallel_wall_s": parallel_wall},
            ],
        }

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
        mot = int(state.get("max_output_tokens", 4096))
        timeout_syn = float(state.get("request_timeout_s", 180.0))
        syn_max_in = int(state.get("synthesizer_max_input_tokens", 20_000))
        retry_max = int(state.get("retry_max_attempts", 3))
        retry_base = float(state.get("retry_base_delay_s", 2.0))
        s0 = time.perf_counter()
        it_no = round_idx + 1
        err_ev: list[dict[str, Any]] = []
        try:
            result = await asyncio.wait_for(
                syn.synthesize(
                    original_prompt=state["original_prompt"],
                    responses=resp_map,
                    enable_web_search=state["enable_web_search"],
                    max_output_tokens=mot,
                    synthesizer_max_input_tokens=syn_max_in,
                    retry_max_attempts=retry_max,
                    retry_base_delay_s=retry_base,
                ),
                timeout=timeout_syn,
            )
        except asyncio.CancelledError:
            raise
        except TimeoutError as exc:
            logger.error(
                "Synthesis failed: provider=%s error_type=%s detail=%r",
                state["synthesizer_name"],
                type(exc).__name__,
                exc,
            )
            err_ev.append(run_error_record(stage="synthesis", provider=state["synthesizer_name"], exc=exc))
            err_resp = failure_response_from_completed(state["synthesizer_name"], exc, started_perf=s0)
            result = SynthesisResult(response=err_resp, prompt="(synthesis stage timed out)")
        except Exception as exc:
            logger.error(
                "Synthesis failed: provider=%s error_type=%s detail=%r",
                state["synthesizer_name"],
                type(exc).__name__,
                exc,
            )
            err_ev.append(run_error_record(stage="synthesis", provider=state["synthesizer_name"], exc=exc))
            err_resp = failure_response_from_completed(state["synthesizer_name"], exc, started_perf=s0)
            result = SynthesisResult(
                response=err_resp,
                prompt=f"(synthesis exception: {type(exc).__name__})",
            )
        syn_wall = time.perf_counter() - s0
        _, failed_only = partition_provider_responses(resp_map)
        if result.response.model == "error:AllProvidersFailed":
            err_ev.append(
                {
                    "stage": "synthesis",
                    "provider": state["synthesizer_name"],
                    "error_type": "AllProvidersFailed",
                    "detail": f"excluded_failed_providers={sorted(failed_only)}",
                }
            )
        text = format_synthesis_artifact_markdown(synthesis=result, responses=resp_map)
        iter_dir = out / "iterations"
        (iter_dir / f"iteration_{round_idx + 1}_synthesis.md").write_text(text + "\n", encoding="utf-8")
        out_update: dict[str, Any] = {
            "synthesis_history": [result.response.text],
            "timing_events": [{"iteration": it_no, "synthesis_wall_s": syn_wall}],
        }
        if err_ev:
            out_update["error_events"] = err_ev
        return out_update

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
        focus = _excerpt_for_verifier(syn)
        prompt = f"{VERIFIER_INSTRUCTION}\n\n### Synthesis excerpt\n{focus}\n"
        verifier = registry.create(state["verifier_name"])
        vmt = int(state.get("verifier_max_output_tokens", 1536))
        timeout_v = float(state.get("request_timeout_s", 180.0))
        retry_max = int(state.get("retry_max_attempts", 3))
        retry_base = float(state.get("retry_base_delay_s", 2.0))
        v0 = time.perf_counter()
        it_no = round_idx + 1
        err_ev: list[dict[str, Any]] = []

        async def _v_attempt() -> ProviderResponse:
            return await verifier.generate(
                prompt,
                enable_web_search=state["enable_web_search"],
                max_output_tokens=vmt,
            )

        try:
            resp = await asyncio.wait_for(
                async_retry_call(
                    _v_attempt,
                    provider=state["verifier_name"],
                    max_attempts=retry_max,
                    base_delay_s=retry_base,
                ),
                timeout=timeout_v,
            )
        except asyncio.CancelledError:
            raise
        except TimeoutError as exc:
            logger.error(
                "Verification failed: provider=%s error_type=%s detail=%r",
                state["verifier_name"],
                type(exc).__name__,
                exc,
            )
            err_ev.append(run_error_record(stage="verify", provider=state["verifier_name"], exc=exc))
            resp = failure_response_from_completed(state["verifier_name"], exc, started_perf=v0)
        except Exception as exc:
            logger.error(
                "Verification failed: provider=%s error_type=%s detail=%r",
                state["verifier_name"],
                type(exc).__name__,
                exc,
            )
            err_ev.append(run_error_record(stage="verify", provider=state["verifier_name"], exc=exc))
            resp = failure_response_from_completed(state["verifier_name"], exc, started_perf=v0)
        ver_wall = time.perf_counter() - v0
        data = _parse_verifier_json(resp.text)
        iter_dir = out / "iterations"
        (iter_dir / f"iteration_{round_idx + 1}_verify.md").write_text(
            json.dumps(data, indent=2) + "\n", encoding="utf-8"
        )
        out_v: dict[str, Any] = {
            "verification_history": [data],
            "timing_events": [{"iteration": it_no, "verify_wall_s": ver_wall}],
        }
        if err_ev:
            out_v["error_events"] = err_ev
        return out_v

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

        run_json = out / "run.json"
        timing_summary = merge_timing_events(state.get("timing_events", []))
        meta = (
            json.loads(run_json.read_text(encoding="utf-8"))
            if run_json.is_file()
            else {}
        )
        meta["timing"] = timing_summary
        prior_errs = meta.get("errors")
        if not isinstance(prior_errs, list):
            prior_errs = []
        merged_errs: list[Any] = list(prior_errs) + list(state.get("error_events", []))
        meta["errors"] = merged_errs
        run_json.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        logger.info("Iterative wall-clock timing summary: %s", timing_summary)

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
        "providers": cfg.provider_names(),
        "provider_configs": [pc.model_dump() for pc in cfg.providers],
        "max_output_tokens": cfg.max_output_tokens,
        "verifier_max_output_tokens": cfg.verifier_max_output_tokens,
        "request_timeout_s": float(cfg.request_timeout_s),
        "timing_events": [],
        "error_events": [],
        "retry_max_attempts": cfg.retry_max_attempts,
        "retry_base_delay_s": float(cfg.retry_base_delay_s),
        "synthesizer_max_input_tokens": cfg.synthesizer_max_input_tokens,
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
