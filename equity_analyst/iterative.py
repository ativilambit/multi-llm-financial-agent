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

from equity_analyst.config import ProviderConfig, RunConfig, SynthesizerConfig
from equity_analyst.drive_uploader import maybe_upload_run_to_drive_raw
from equity_analyst.gemini_cache import GeminiCacheIndex
from equity_analyst.prompt_parts import EQUITY_ANALYST_SYSTEM_PROMPT
from equity_analyst.prompting import RenderedPrompt
from equity_analyst.provider_runtime import (
    effective_synthesizer_web_search,
    effective_web_search,
    failure_response,
    failure_response_from_completed,
    fan_out_max_output_tokens,
    partition_provider_responses,
    run_error_record,
)
from equity_analyst.providers.anthropic_provider import AnthropicProvider
from equity_analyst.providers.gemini_provider import GeminiProvider
from equity_analyst.providers.registry import ProviderRegistry
from equity_analyst.retry import async_retry_call
from equity_analyst.synthesizer import (
    SynthesisResult,
    Synthesizer,
    format_synthesis_artifact_markdown,
)
from equity_analyst.types import ProviderResponse, ProviderUsage

logger = logging.getLogger(__name__)

VERIFIER_INSTRUCTION_PREFIX = """You are a financial fact-checker. You receive an excerpt of a synthesis focused on
numerical and factual claims about an equity/options thesis (and lines mentioning low confidence).

Use web search only when needed to check those claims. Do not spend effort re-verifying narrative sections
that are not represented in the excerpt."""

VERIFIER_JSON_TAIL = """If you cannot perform verification (refusal, missing tools, or no relevant claims in the excerpt), you must
still respond with valid JSON only: use empty arrays for the three lists and set "notes" to a short reason.

Example (format only; replace with real claims from the excerpt):
{"verified": ["Revenue grew 12% YoY per company filing"], "contradicted": [], "unverifiable": ["Third-party estimate of Q2 margin without a cited source"], "notes": ""}

CRITICAL — your entire reply must be parseable as a single JSON object. No markdown code fences, no prose
before or after the object, no commentary outside the JSON. One line or pretty-printed is fine.

Required keys (arrays of short strings): "verified", "contradicted", "unverifiable". Optional: "notes" (string).
"""

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


_CLAIM_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "verified": ("verified", "verified_claims", "verified_items"),
    "contradicted": ("contradicted", "contradictions", "contradicted_claims"),
    "unverifiable": ("unverifiable", "unverifiable_claims"),
}


def _strip_markdown_fences(t: str) -> str:
    t = t.strip()
    if not t.startswith("```"):
        return t
    lines = t.split("\n")
    body = "\n".join(lines[1:]) if lines else ""
    if "```" in body:
        body = body.rsplit("```", 1)[0]
    return body.strip()


def _balanced_brace_objects(s: str) -> list[str]:
    n = len(s)
    out: list[str] = []
    i = 0
    while i < n:
        if s[i] != "{":
            i += 1
            continue
        depth = 0
        in_str = False
        esc = False
        start = i
        j = i
        found = False
        while j < n:
            c = s[j]
            if esc:
                esc = False
                j += 1
                continue
            if in_str:
                if c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                j += 1
                continue
            if c == '"':
                in_str = True
                j += 1
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    out.append(s[start : j + 1])
                    found = True
                    break
            j += 1
        i = j + 1 if found else i + 1
    return out


def _coerce_claim_list(val: Any) -> list[str]:
    if val is None:
        return []
    if isinstance(val, str):
        s = val.strip()
        return [s] if s else []
    if isinstance(val, (int, float, bool)):
        return [str(val)]
    if isinstance(val, list):
        acc: list[str] = []
        for x in val:
            acc.extend(_coerce_claim_list(x))
        return acc
    if isinstance(val, dict):
        acc2: list[str] = []
        for v in val.values():
            acc2.extend(_coerce_claim_list(v))
        return acc2
    return [str(val)]


def _values_for_canonical_key(data: dict[str, Any], canonical: str) -> list[str]:
    for alias in _CLAIM_KEY_ALIASES[canonical]:
        if alias not in data:
            continue
        return _coerce_claim_list(data[alias])
    return []


def _score_verification_dict(data: dict[str, Any]) -> int:
    return sum(len(_values_for_canonical_key(data, k)) for k in _CLAIM_KEY_ALIASES)


def _candidate_dicts_from_text(text: str) -> list[tuple[dict[str, Any], int]]:
    seen: set[str] = set()
    out: list[tuple[dict[str, Any], int]] = []

    def _push(raw_slice: str, d: dict[str, Any]) -> None:
        key = json.dumps(d, sort_keys=True)
        if key in seen:
            return
        seen.add(key)
        out.append((d, len(raw_slice)))

    stripped = _strip_markdown_fences(text.strip())
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            _push(stripped, obj)
    except json.JSONDecodeError:
        pass
    for sub in _balanced_brace_objects(text):
        try:
            obj = json.loads(sub)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            _push(sub, obj)
    return out


def parse_verifier_json(text: str) -> dict[str, Any]:
    """Parse verifier model output into verified / contradicted / unverifiable lists (and optional notes)."""
    raw = text
    candidates = _candidate_dicts_from_text(text)
    if not candidates:
        logger.warning(
            "verifier JSON parse failed (no object decoded); verifier_raw=%s",
            raw[:1000] if raw else "",
        )
        return {"verified": [], "contradicted": [], "unverifiable": []}

    best_data: dict[str, Any] | None = None
    best_rank: tuple[int, int] = (-1, -1)
    for data, raw_len in candidates:
        score = _score_verification_dict(data)
        rank = (score, raw_len)
        if rank > best_rank:
            best_rank = rank
            best_data = data

    assert best_data is not None
    result: dict[str, Any] = {
        "verified": _values_for_canonical_key(best_data, "verified"),
        "contradicted": _values_for_canonical_key(best_data, "contradicted"),
        "unverifiable": _values_for_canonical_key(best_data, "unverifiable"),
    }
    notes_val = best_data.get("notes")
    if isinstance(notes_val, str) and notes_val.strip():
        result["notes"] = notes_val.strip()
    return result


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
    prompt_cache_enabled: bool
    anthropic_force_tool_use: bool
    providers: list[str]
    provider_configs: list[dict[str, Any]]
    max_output_tokens: int
    verifier_max_output_tokens: int
    synthesizer_max_output_tokens: int
    request_timeout_s: float
    retry_max_attempts: int
    retry_base_delay_s: float
    synthesizer_max_input_tokens: int
    gemini_cache_ttl_s: int
    synthesizer_cfg: dict[str, Any]
    verifier_name: str
    verifier_model: str | None
    output_dir: str
    provider_responses: Annotated[list[dict[str, Any]], operator.add]
    synthesis_history: Annotated[list[str], operator.add]
    verification_history: Annotated[list[dict[str, Any]], operator.add]
    followup_questions: Annotated[list[str], operator.add]
    timing_events: Annotated[list[dict[str, Any]], operator.add]
    error_events: Annotated[list[dict[str, Any]], operator.add]
    final_report: str
    drive_upload_enabled: bool
    drive_credentials_path: str | None
    drive_root_folder_id: str | None


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
        cfg_mot = int(state.get("max_output_tokens", 16_000))
        names = [pc.name for pc in pcs]
        retry_max = int(state.get("retry_max_attempts", 3))
        retry_base = float(state.get("retry_base_delay_s", 2.0))
        gemini_cache_index: GeminiCacheIndex | None = (
            GeminiCacheIndex() if state.get("prompt_cache_enabled", True) else None
        )
        gemini_ttl = int(state.get("gemini_cache_ttl_s", 3600))

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
            p = registry.create(
                pc.name,
                model=pc.model,
                gemini_cache_index=gemini_cache_index,
                gemini_cache_ttl_s=gemini_ttl,
            )
            ws = effective_web_search(run_default=state["enable_web_search"], pc=pc)
            to = float(pc.request_timeout_s) if pc.request_timeout_s is not None else cfg_req_timeout

            async def _attempt() -> ProviderResponse:
                mot = fan_out_max_output_tokens(pc, cfg_mot)
                pce = bool(state.get("prompt_cache_enabled", True))
                if isinstance(p, AnthropicProvider):
                    static = EQUITY_ANALYST_SYSTEM_PROMPT
                    sep = f"{static}\n\n"
                    user_only = body[len(sep) :] if body.startswith(sep) else body
                    return await p.generate(
                        body,
                        enable_web_search=ws,
                        max_output_tokens=mot,
                        prompt_cache_enabled=pce,
                        user_message_for_cache=user_only,
                        force_tool_use=bool(state.get("anthropic_force_tool_use", True)),
                    )
                if isinstance(p, GeminiProvider) and pce and gemini_cache_index is not None:
                    static = EQUITY_ANALYST_SYSTEM_PROMPT
                    sep = f"{static}\n\n"
                    if body.startswith(sep):
                        return await p.generate(
                            body,
                            enable_web_search=ws,
                            max_output_tokens=mot,
                            cacheable_prefix=static,
                            user_message_for_cache=body[len(sep) :],
                        )
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
        syn_cfg = SynthesizerConfig.model_validate(state["synthesizer_cfg"])
        logger.info(
            "Node synthesize iteration=%s max_iterations=%s synthesizer=%s",
            round_idx + 1,
            state["max_iterations"],
            syn_cfg.name,
        )
        last = state["provider_responses"][-1]
        raw = last["responses"]
        resp_map: dict[str, ProviderResponse] = {k: _dict_to_response(v) for k, v in raw.items()}
        synth_backend = registry.create(syn_cfg.name, model=syn_cfg.model)
        syn = Synthesizer(synth_backend)
        synth_mot = int(state.get("synthesizer_max_output_tokens", 24_000))
        timeout_syn = (
            float(syn_cfg.request_timeout_s)
            if syn_cfg.request_timeout_s is not None
            else float(state.get("request_timeout_s", 180.0))
        )
        syn_max_in = int(state.get("synthesizer_max_input_tokens", 20_000))
        retry_max = int(state.get("retry_max_attempts", 3))
        retry_base = float(state.get("retry_base_delay_s", 2.0))
        syn_ws = effective_synthesizer_web_search(run_default=state["enable_web_search"], syn=syn_cfg)
        s0 = time.perf_counter()
        it_no = round_idx + 1
        err_ev: list[dict[str, Any]] = []
        try:
            result = await asyncio.wait_for(
                syn.synthesize(
                    original_prompt=state["original_prompt"],
                    responses=resp_map,
                    enable_web_search=syn_ws,
                    max_output_tokens=synth_mot,
                    synthesizer_max_input_tokens=syn_max_in,
                    retry_max_attempts=retry_max,
                    retry_base_delay_s=retry_base,
                    anthropic_force_tool_use=bool(state.get("anthropic_force_tool_use", True)),
                ),
                timeout=timeout_syn,
            )
        except asyncio.CancelledError:
            raise
        except TimeoutError as exc:
            logger.error(
                "Synthesis failed: provider=%s error_type=%s detail=%r",
                syn_cfg.name,
                type(exc).__name__,
                exc,
            )
            err_ev.append(run_error_record(stage="synthesis", provider=syn_cfg.name, exc=exc))
            err_resp = failure_response_from_completed(syn_cfg.name, exc, started_perf=s0)
            result = SynthesisResult(response=err_resp, prompt="(synthesis stage timed out)")
        except Exception as exc:
            logger.error(
                "Synthesis failed: provider=%s error_type=%s detail=%r",
                syn_cfg.name,
                type(exc).__name__,
                exc,
            )
            err_ev.append(run_error_record(stage="synthesis", provider=syn_cfg.name, exc=exc))
            err_resp = failure_response_from_completed(syn_cfg.name, exc, started_perf=s0)
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
                    "provider": syn_cfg.name,
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
        prompt = (
            f"{VERIFIER_INSTRUCTION_PREFIX}\n\n"
            f"### Synthesis excerpt\n{focus}\n\n"
            f"{VERIFIER_JSON_TAIL}\n"
        )
        v_model = state.get("verifier_model")
        gemini_ttl = int(state.get("gemini_cache_ttl_s", 3600))
        verifier = registry.create(
            state["verifier_name"],
            model=v_model,
            gemini_cache_index=None,
            gemini_cache_ttl_s=gemini_ttl,
        )
        vmt = int(state.get("verifier_max_output_tokens", 1536))
        timeout_v = float(state.get("request_timeout_s", 180.0))
        retry_max = int(state.get("retry_max_attempts", 3))
        retry_base = float(state.get("retry_base_delay_s", 2.0))
        v0 = time.perf_counter()
        it_no = round_idx + 1
        err_ev: list[dict[str, Any]] = []

        async def _v_attempt() -> ProviderResponse:
            if isinstance(verifier, AnthropicProvider):
                return await verifier.generate(
                    prompt,
                    enable_web_search=state["enable_web_search"],
                    max_output_tokens=vmt,
                    prompt_cache_enabled=False,
                    force_tool_use=False,
                )
            if isinstance(verifier, GeminiProvider):
                return await verifier.generate(
                    prompt,
                    enable_web_search=state["enable_web_search"],
                    max_output_tokens=vmt,
                    cacheable_prefix=None,
                )
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
        if resp.model.startswith("error:"):
            logger.error(
                "Verifier call failed (provider=%s, error=%s); verification arrays will be empty for this round.",
                state["verifier_name"],
                resp.model.removeprefix("error:"),
            )
        iter_dir = out / "iterations"
        iter_dir.mkdir(parents=True, exist_ok=True)
        (iter_dir / f"iteration_{round_idx + 1}_verify_raw.md").write_text(resp.text, encoding="utf-8")
        data = parse_verifier_json(resp.text)
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

        if bool(state.get("drive_upload_enabled", False)):
            cred_raw = state.get("drive_credentials_path")
            root_raw = state.get("drive_root_folder_id")
            await maybe_upload_run_to_drive_raw(
                drive_upload_enabled=True,
                drive_credentials_path=cred_raw if isinstance(cred_raw, str) else None,
                drive_root_folder_id=root_raw if isinstance(root_raw, str) else None,
                out_dir=out,
                run_id=out.name,
                append_synthesis_footer=True,
            )

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
    rendered: RenderedPrompt,
    output_dir: Path,
) -> RefinementState:
    return {
        "symbol": cfg.symbol,
        "original_prompt": rendered.text,
        "max_iterations": 3,
        "confidence_threshold": 0.85,
        "enable_web_search": True,
        "prompt_cache_enabled": cfg.prompt_cache_enabled,
        "anthropic_force_tool_use": cfg.anthropic_force_tool_use,
        "providers": cfg.provider_names(),
        "provider_configs": [pc.model_dump() for pc in cfg.providers],
        "max_output_tokens": cfg.max_output_tokens,
        "verifier_max_output_tokens": cfg.verifier_max_output_tokens,
        "synthesizer_max_output_tokens": cfg.synthesizer_max_output_tokens,
        "request_timeout_s": float(cfg.request_timeout_s),
        "timing_events": [],
        "error_events": [],
        "retry_max_attempts": cfg.retry_max_attempts,
        "retry_base_delay_s": float(cfg.retry_base_delay_s),
        "synthesizer_max_input_tokens": cfg.synthesizer_max_input_tokens,
        "gemini_cache_ttl_s": cfg.gemini_cache_ttl_s,
        "synthesizer_cfg": cfg.synthesizer.model_dump(mode="json"),
        "verifier_name": cfg.verifier_provider,
        "verifier_model": cfg.verifier_model,
        "output_dir": str(output_dir.resolve()),
        "drive_upload_enabled": cfg.drive_upload_enabled,
        "drive_credentials_path": cfg.drive_credentials_path,
        "drive_root_folder_id": cfg.drive_root_folder_id,
    }


def dry_run_compile_only(*, registry: ProviderRegistry) -> list[str]:
    from langgraph.checkpoint.memory import MemorySaver

    app = compile_refinement_workflow(registry=registry, checkpointer=MemorySaver())
    nodes = app.get_graph().nodes
    out = sorted(n for n in nodes if not str(n).startswith("__"))
    logger.debug("Dry-run graph inspection nodes=%s", out)
    return out
