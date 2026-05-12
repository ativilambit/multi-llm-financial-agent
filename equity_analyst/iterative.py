from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import logging
import math
import operator
import re
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, NotRequired, TypedDict, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from equity_analyst.config import ProviderConfig, RunConfig, RunEnvironment, SynthesizerConfig
from equity_analyst.drive_uploader import DriveAuthMode, maybe_upload_run_to_drive_raw
from equity_analyst.facts_packet import (
    extract_facts_packet,
    facts_frozen_user_prefix,
    write_facts_packet,
)
from equity_analyst.gemini_cache import GeminiCacheIndex
from equity_analyst.options_chain import options_chain_expiry_audit_messages
from equity_analyst.pdf_writer import maybe_write_pdf_sibling
from equity_analyst.prediction_extract import run_prediction_extract_for_run_dir
from equity_analyst.prompt_export import prompt_call_context
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
from equity_analyst.providers.gemini_provider import (
    DEFAULT_GEMINI_MODEL,
    GeminiProvider,
    gemini_model_requires_nonzero_thinking_budget,
)
from equity_analyst.providers.openai_provider import OpenAIProvider
from equity_analyst.providers.registry import ProviderRegistry
from equity_analyst.retry import async_retry_call
from equity_analyst.synthesizer import (
    SynthesisResult,
    Synthesizer,
    detect_max_tokens_truncation,
    format_synthesis_artifact_markdown,
    provider_finish_reason_label,
)
from equity_analyst.types import ProviderResponse, ProviderUsage

logger = logging.getLogger(__name__)

_CHECKPOINT_BASENAMES: tuple[str, ...] = (
    "checkpoint.sqlite",
    "checkpoint.sqlite-wal",
    "checkpoint.sqlite-shm",
    "checkpoint.sqlite-journal",
)


def _delete_checkpoint_files(run_output_dir: Path, log: logging.Logger) -> None:
    """Best-effort removal of LangGraph SQLite checkpoint files under ``run_output_dir``."""
    for name in _CHECKPOINT_BASENAMES:
        path = run_output_dir / name
        try:
            if path.is_file():
                path.unlink()
        except OSError as exc:
            log.warning(
                "Failed to remove checkpoint artifact %s (%s: %s)",
                path,
                type(exc).__name__,
                exc,
                exc_info=False,
            )


def maybe_delete_iterative_checkpoint(
    run_output_dir: Path,
    *,
    delete_checkpoint_after_success: bool,
    log: logging.Logger | None = None,
) -> None:
    if not delete_checkpoint_after_success:
        return
    _delete_checkpoint_files(run_output_dir, log or logger)


# Greek sigma in prompts / checks (avoid literal U+03C3 in source for RUF001).
_GS = chr(0x03C3)

VERIFIER_INSTRUCTION_PREFIX = (
    """You are a financial fact-checker. You receive an excerpt of a synthesis focused on
numerical and factual claims about an equity/options thesis (and lines mentioning low confidence).

The underlying equity prompt is structured in 12 numbered sections (including a mandatory bottom-up qualitative overlay in section 8 before prediction sections); the excerpt may omit section headers—still treat cited numbers and ratios as the verification target.

Use web search only when needed to check those claims. Do not spend effort re-verifying narrative sections
that are not represented in the excerpt.

If the excerpt includes **section 8** bottom-up qualitative material, add an **unverifiable** item when the first 800 characters of that section-8 passage contain **no** `http://` or `https://` URL and **no** line starting with `Source:` (heuristic for missing citations—tune to reduce false positives).

When excerpted claims concern 1-sigma / 2-sigma / 3-sigma **dollar** bands, treat **prior-close anchoring** and **labeled same-day intraday `[low-1.00, high+1.00]` (USD)** as both valid when the synthesis states which anchor it used; do not flag a contradiction solely because two runs used different branches of the equity prompt.

"""
    + f"""**{_GS} band structural checks (mandatory pass/fail in your JSON, plus cite synthesis gaps):**
- For every session or horizon in the excerpt that reports **1{_GS} / 2{_GS} / 3{_GS}** bands tied to options-implied width, require an explicit **vol baseline**: a **real listed options expiry (YYYY-MM-DD)** used for implied move, **or** the literal label **HV30 sqrt(t) scaling** (or text clearly equivalent).
- **No fake 0-DTE implied move:** if the excerpt reports same-day implied-move {_GS} for a session **without** naming a chain expiry that could support that session, add a concise **unverifiable** item naming the session and asking for the nearest weekly expiry + sqrt(target_DTE/chosen_DTE) scaling or HV30 fallback.
- **Variance-additive event+diffusion (canonical when the horizon crosses the earnings print and the target is post-event):** if the excerpt states **event_jump=** ... **%** and **daily_vol=** ... **%** (post-event decomposition), recompute **{_GS}^2(T+N) - {_GS}^2(T+1)** from the stated **1{_GS}** (or **3{_GS}/3**) half-width % for each later post-event horizon vs the first post-event row and verify it equals **(N-1) * daily_vol^2** within **+/-25%** (same N as the model's post-event day count). If it fails, add **unverifiable** items with the concrete observed vs expected numbers and ask for a corrected **daily_vol** or explicit fallback labeling.
- **sqrt(t) coherence (fallback, single IV baseline only):** when the excerpt does **not** use the variance-additive literals above and two or more horizons share one baseline regime, the ratio of **3{_GS} half-width %** values should track **sqrt(Delta trading_sessions)** within **+/-25%** unless the excerpt explicitly flags a **vol regime change**. If incompatible, add **unverifiable** follow-ups naming dates, observed ratio, expected sqrt(N), and ask to re-derive or label distinct regimes.
- If the excerpt is missing the mandatory sanity line while it contains multi-horizon **3{_GS}** % bands, add an **unverifiable** item: require **`{_GS}-scaling check (variance):`** when **event_jump=** / **daily_vol=** are present; otherwise require **`{_GS}-scaling check:`** (sqrt-t ratio form).

"""
    + """Limit each list to at most 10 items; each item must be 25 words or fewer (short sentences).
Prioritize the most material claims (numbers, ratings, P/C ratios) over narrative."""
)

VERIFIER_JSON_TAIL = """If you cannot perform verification (refusal, missing tools, or no relevant claims in the excerpt), you must
still respond with valid JSON only: use empty arrays for the three lists and set "notes" to a short reason.

Keep each of "verified", "contradicted", and "unverifiable" to at most 10 items; each string ≤ 25 words.
Prioritize material claims (figures, ratings, ratios) over narrative.

CRITICAL — your entire reply must be parseable as a single JSON object. No markdown code fences, no prose
before or after the object, no commentary outside the JSON. One line or pretty-printed is fine.

Required keys (arrays of short strings): "verified", "contradicted", "unverifiable". Optional: "notes" (string).

Optional cost-control directives (defaults conservative — leave false/empty unless truly needed):
- "refresh_facts" (boolean): true only if the frozen round-1 facts packet is materially stale or internally inconsistent with the synthesis excerpt.
- "refan_out_providers" (array of strings): provider names to re-run in parallel fan-out next iteration (e.g. ["anthropic", "openai"]). Empty means do not request extra fan-out.
- "refan_out_all" (boolean): true only if multiple provider lenses are required again; next iteration re-runs every configured fan-out provider.
- "sections_to_revise" (array of integers 1-12): optional hint for which numbered equity-report sections the next provider pass should focus on (e.g. [9, 11]). Omit or use [] if unclear.

Optional **sigma_band audit** (include when the excerpt has multi-horizon sigma bands; omit entire block if not applicable):
- "sigma_band_sessions" (array of objects): one entry per session that reports sigma bands in the excerpt. Each object: "session" (short label), "sigma_baseline" (chain **YYYY-MM-DD** expiry **or** "HV30 sqrt(t) scaling"), "sigma_scaling_check_passed" (boolean, false when baseline missing or sqrt(t) ratio off >25% vs your stated N).
- "sigma_scaling_aggregate_passed" (boolean): true only if every session entry has a baseline and cross-horizon 3-sigma ratios match sqrt(Delta trading days) within tolerance **or** the excerpt explicitly documents distinct vol regimes.

Example (format only; replace with real claims from the excerpt):
{"verified": [], "contradicted": [], "unverifiable": ["sigma baseline missing for May 12 session"], "notes": "", "refresh_facts": false, "refan_out_providers": [], "refan_out_all": false, "sections_to_revise": [], "sigma_band_sessions": [{"session": "May 13 post-earnings", "sigma_baseline": "2026-05-16 weekly expiry", "sigma_scaling_check_passed": true}], "sigma_scaling_aggregate_passed": true}

When in doubt, use "refresh_facts": false, "refan_out_providers": [], "refan_out_all": false, "sections_to_revise": [].
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


def trading_sessions_after_exclusive(start: date, end: date) -> int:
    """Count Mon-Fri sessions with ``start < session_date <= end`` (NYSE-style weekdays)."""
    if end <= start:
        return 0
    n = 0
    d = start
    while d < end:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n += 1
    return n


def trading_sessions_inclusive(start: date, end: date) -> int:
    """Count Mon-Fri sessions with ``start <= session_date <= end`` (NYSE-style weekdays)."""
    if end < start:
        return 0
    n = 0
    d = start
    while d <= end:
        if d.weekday() < 5:
            n += 1
        d += timedelta(days=1)
    return n


_EVENT_JUMP_PCT_RE = re.compile(r"event_jump\s*=\s*([\d.]+)\s*%", re.IGNORECASE)
_DAILY_VOL_PCT_RE = re.compile(r"daily_vol\s*=\s*([\d.]+)\s*%", re.IGNORECASE)


_SESSION_HEAD_RE = re.compile(
    r"\b((?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|"
    r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2}))\b",
    re.IGNORECASE,
)

_THREE_SIG_PCT_RE = re.compile(
    rf"3\s*{_GS}\s*:\s*[^\n(]*\(\s*±\s*(?P<pct>[\d.]+)\s*%\s*\)",
    re.IGNORECASE,
)


def _parse_month_day_label(label: str, *, year: int) -> date | None:
    m = re.search(
        r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|"
        r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2})\b",
        label,
        re.IGNORECASE,
    )
    if not m:
        return None
    token = m.group(1)
    day = int(m.group(2))
    for fmt in ("%b %d %Y", "%B %d %Y"):
        try:
            return datetime.strptime(f"{token} {day} {year}", fmt).date()
        except ValueError:
            continue
    return None


def extract_dated_three_sigma_half_widths(synthesis: str, *, year: int = 2026) -> list[tuple[date, float, str]]:
    """Best-effort: pair 3-sigma (+-pct%) lines with the most recent weekday+month+day header."""
    lines = synthesis.splitlines()
    current_label: str | None = None
    out: list[tuple[date, float, str]] = []
    for line in lines:
        mh = _SESSION_HEAD_RE.search(line)
        if mh:
            current_label = mh.group(1)
        m3 = _THREE_SIG_PCT_RE.search(line)
        if not m3:
            continue
        pct = float(m3.group("pct"))
        anchor = _parse_month_day_label(current_label or "", year=year)
        if anchor is None:
            anchor = _parse_month_day_label(line, year=year)
        lbl = current_label or line.strip()[:120]
        if anchor is not None:
            out.append((anchor, pct, lbl))
    return out


def sigma_band_sqrt_ratio_followups(
    *,
    width_early: float,
    width_late: float,
    trading_day_span: int,
    session_early: str = "T+1",
    session_late: str = "T+N",
    tolerance: float = 0.25,
) -> list[str]:
    """Emit follow-up questions when 3-sigma % half-widths violate sqrt(t) vs a 1-session early anchor.

    ``trading_day_span`` is the number of Mon-Fri sessions strictly after the early session date
    through the late session date (inclusive of the late session), used as N in sqrt(N) scaling.
    """
    if width_early <= 0.0 or trading_day_span <= 0:
        return []
    expected = math.sqrt(float(trading_day_span))
    if expected <= 0.0:
        return []
    observed = width_late / width_early
    rel_err = abs(observed - expected) / expected
    if rel_err <= tolerance:
        return []
    tag = f"{session_late[:14]}/{session_early[:14]}"
    return [
        f"{_GS} sqrt-t {tag}: ratio {observed:.2f} vs sqrt(N)~{expected:.2f} (N={trading_day_span}); cite expiry or regimes.",
    ]


def verify_variance_additive_sigma_band_sessions(
    sessions: list[Any],
    daily_vol_pct: float,
    event_jump_pct: float,
    tolerance: float = 0.25,
) -> list[str]:
    """Return follow-up strings when 1-sigma % half-widths violate variance-additive post-event math.

    Each session entry should be a mapping with ``session`` (str), ``N`` (int trading days from T+1
    inclusive), and ``sigma_pct`` (float, **1-sigma** +-percent half-width). ``N`` must be >= 1.

    When an ``N == 1`` row exists, later horizons are checked via
    ``sigma^2(T+N) - sigma^2(T+1)`` vs ``(N-1) * daily_vol^2``. Otherwise each row is checked against
    ``event_jump^2 + N * daily_vol^2`` on ``sigma^2``.
    """
    rows: list[tuple[str, int, float]] = []
    for raw in sessions:
        if not isinstance(raw, dict):
            continue
        sess = str(raw.get("session", "")).strip()
        if sess == "":
            continue
        n_val = raw.get("N")
        sig_val = raw.get("sigma_pct")
        try:
            n_int = int(n_val)  # type: ignore[arg-type]
            sig_f = float(sig_val)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if n_int < 1 or sig_f <= 0.0:
            continue
        rows.append((sess, n_int, sig_f))

    if not rows:
        return []

    ref_sq: float | None = None
    for _s, n_i, sig in rows:
        if n_i == 1:
            ref_sq = sig * sig
            break

    followups: list[str] = []
    ej2 = float(event_jump_pct) ** 2
    dv2 = float(daily_vol_pct) ** 2

    for sess, n_i, sig in rows:
        obs_sq = sig * sig
        if ref_sq is not None:
            if n_i == 1:
                expected_sq = ej2 + dv2
                if expected_sq <= 0.0:
                    continue
                rel = abs(obs_sq - expected_sq) / expected_sq
                if rel > tolerance:
                    followups.append(
                        f"{_GS}^2 variance {sess[:18]}: 1{_GS}^2 obs={obs_sq:.2f} vs ej^2+dv^2={expected_sq:.2f} ({rel:.0%} err); fix daily_vol or inputs.",
                    )
            else:
                expected_delta = float(n_i - 1) * dv2
                if expected_delta <= 0.0:
                    continue
                obs_delta = obs_sq - ref_sq
                rel = abs(obs_delta - expected_delta) / expected_delta
                if rel > tolerance:
                    followups.append(
                        f"{_GS}^2 variance {sess[:18]}: delta_obs={obs_delta:.2f} vs (N-1)dv^2={expected_delta:.2f} (N={n_i}, {rel:.0%} err); fix daily_vol.",
                    )
        else:
            expected_sq = ej2 + float(n_i) * dv2
            if expected_sq <= 0.0:
                continue
            rel = abs(obs_sq - expected_sq) / expected_sq
            if rel > tolerance:
                followups.append(
                    f"{_GS}^2 variance {sess[:18]}: 1{_GS}^2 obs={obs_sq:.2f} vs ej^2+N*dv^2={expected_sq:.2f} (N={n_i}, {rel:.0%} err); fix inputs.",
                )
    return followups


def _parse_variance_additive_literals(synthesis: str) -> tuple[float | None, float | None]:
    ej_m = _EVENT_JUMP_PCT_RE.search(synthesis)
    dv_m = _DAILY_VOL_PCT_RE.search(synthesis)
    if not ej_m or not dv_m:
        return None, None
    try:
        return float(ej_m.group(1)), float(dv_m.group(1))
    except ValueError:
        return None, None


def _coerce_sigma_band_sessions(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        sess = str(item.get("session", "")).strip()
        baseline = str(item.get("sigma_baseline", "")).strip()
        rec: dict[str, Any] = {"session": sess, "sigma_baseline": baseline}
        passed = item.get("sigma_scaling_check_passed")
        if isinstance(passed, bool):
            rec["sigma_scaling_check_passed"] = passed
        out.append(rec)
    return out


def augment_verifier_result_with_sigma_structural_checks(
    synthesis_text: str,
    result: dict[str, Any],
    *,
    anchor_year: int = 2026,
    sqrt_tolerance: float = 0.25,
    options_chain_data: dict[str, Any] | None = None,
    symbol: str = "",
) -> dict[str, Any]:
    """Append deterministic sigma-band structural items to ``unverifiable`` (router follow-ups)."""
    out = dict(result)
    prior = [str(x).strip() for x in (out.get("unverifiable") or []) if str(x).strip()]
    extras: list[str] = []

    sigma_lines = [ln for ln in synthesis_text.splitlines() if f"3{_GS}" in ln and "±" in ln and "%" in ln]
    ej_lit, dv_lit = _parse_variance_additive_literals(synthesis_text)
    variance_mode = ej_lit is not None and dv_lit is not None

    if sigma_lines:
        if variance_mode:
            if f"{_GS}-scaling check (variance):" not in synthesis_text:
                extras.append(
                    f"{_GS} bands: add `{_GS}-scaling check (variance):` line (delta {_GS}^2 vs (N-1)*daily_vol^2).",
                )
        elif f"{_GS}-scaling check" not in synthesis_text:
            extras.append(
                f"{_GS} bands: add mandatory `{_GS}-scaling check:` line vs sqrt(N) or annotate regimes.",
            )

    dated = extract_dated_three_sigma_half_widths(synthesis_text, year=anchor_year)
    by_date: dict[date, tuple[float, str]] = {}
    for d, pct, lbl in dated:
        by_date[d] = (pct, lbl)

    if variance_mode and len(by_date) >= 1:
        assert ej_lit is not None and dv_lit is not None
        keys_sorted = sorted(by_date)
        early_d = keys_sorted[0]
        session_payload: list[dict[str, Any]] = []
        for d in keys_sorted:
            w3, lbl = by_date[d]
            n_inc = trading_sessions_inclusive(early_d, d)
            session_payload.append({"session": lbl, "N": n_inc, "sigma_pct": w3 / 3.0})
        extras.extend(
            verify_variance_additive_sigma_band_sessions(
                session_payload,
                daily_vol_pct=dv_lit,
                event_jump_pct=ej_lit,
                tolerance=sqrt_tolerance,
            ),
        )
    elif len(by_date) >= 2:
        keys = sorted(by_date)
        early_d, late_d = keys[0], keys[-1]
        span = trading_sessions_after_exclusive(early_d, late_d)
        w_early, lbl_e = by_date[early_d]
        w_late, lbl_l = by_date[late_d]
        if span >= 2 and w_early > 0.0:
            extras.extend(
                sigma_band_sqrt_ratio_followups(
                    width_early=w_early,
                    width_late=w_late,
                    trading_day_span=span,
                    session_early=lbl_e,
                    session_late=lbl_l,
                    tolerance=sqrt_tolerance,
                ),
            )

    for row in out.get("sigma_band_sessions") or []:
        if not isinstance(row, dict):
            continue
        sess = str(row.get("session", "")).strip()
        base = str(row.get("sigma_baseline", "")).strip()
        if sess and not base:
            extras.append(f"{_GS} baseline missing for {sess}; name YYYY-MM-DD expiry or HV30 sqrt(t).")
        passed = row.get("sigma_scaling_check_passed")
        if passed is False:
            extras.append(f"{_GS} scaling failed for {sess}; re-derive or document vol regime split.")

    extras.extend(
        options_chain_expiry_audit_messages(
            synthesis_text,
            out,
            options_chain_data=options_chain_data,
            symbol=symbol,
        ),
    )

    merged = extras + [u for u in prior if u not in extras]
    deduped: list[str] = []
    seen: set[str] = set()
    for u in merged:
        if u in seen:
            continue
        seen.add(u)
        deduped.append(u)
    out["unverifiable"] = deduped[:10]
    return out


CHANGELOG_ROUND_SUMMARY_MAX_CHARS = 1500


def round_summary_for_changelog(
    synthesis_text: str,
    *,
    iteration_index: int,
    max_chars: int = CHANGELOG_ROUND_SUMMARY_MAX_CHARS,
) -> str:
    """Render a per-round preview for the final report's iteration changelog.

    The full per-round synthesis is preserved verbatim in
    ``iterations/iteration_{i}_synthesis.md``. When ``RunConfig.final_report_full_synthesis``
    is True (default), ``synthesis.md`` also inlines the full text for each round in the
    iteration changelog and repeats the last round under ``Final synthesis (last round)``.
    This helper builds a short preview for the legacy abridged changelog when that flag
    is False; it cuts at a paragraph boundary (never mid-sentence) so the abridgement is visually clear and
    can't be mistaken for an LLM ``MAX_TOKENS`` truncation.
    """
    text = synthesis_text.rstrip()
    if len(text) <= max_chars:
        return text
    window = text[:max_chars]
    cut = window.rfind("\n\n")
    if cut < max_chars // 2:
        # No paragraph boundary in the first half — fall back to the last
        # sentence terminator we can find in the window.
        candidates = [window.rfind(t) for t in (". ", "! ", "? ", ".\n", "!\n", "?\n")]
        sentence_cut = max(candidates)
        cut = sentence_cut + 1 if sentence_cut >= max_chars // 2 else max_chars
    preview = text[:cut].rstrip()
    pointer = (
        f"…(abridged; full text in `iterations/iteration_{iteration_index}_synthesis.md`)"
    )
    return f"{preview}\n\n{pointer}"


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


def _verification_dict_has_claims(data: dict[str, Any]) -> bool:
    return any(bool(_values_for_canonical_key(data, k)) for k in _CLAIM_KEY_ALIASES)


def _rstrip_trailing_json_commas(s: str) -> str:
    t = s.rstrip()
    while t.endswith(","):
        t = t[:-1].rstrip()
    return t


_REPAIR_SUFFIXES: tuple[str, ...] = (
    ' [],"notes":""}',
    " []}",
    " null}",
    "]}",  # e.g. cut mid-array: ... "a", "b",
    '"]}',
    "}",
)


def _attempt_truncated_json_repair(text: str) -> dict[str, Any] | None:
    """Try to close truncated verifier JSON and return a dict with at least one non-empty claim list."""
    t = _strip_markdown_fences(text.strip())
    start = t.find("{")
    if start < 0:
        return None
    base_full = t[start:].rstrip()
    if not base_full:
        return None

    tried: set[str] = set()

    def _try_candidate(candidate: str) -> dict[str, Any] | None:
        if candidate in tried:
            return None
        tried.add(candidate)
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        if isinstance(obj, dict) and _verification_dict_has_claims(obj):
            return obj
        return None

    max_trim = min(256, max(0, len(base_full) - 1))
    for trim in range(0, max_trim + 1):
        base = base_full[:-trim].rstrip() if trim else base_full
        if not base:
            continue
        base_variants = {base, _rstrip_trailing_json_commas(base)}
        for b in base_variants:
            if not b:
                continue
            for suf in _REPAIR_SUFFIXES:
                got = _try_candidate(b + suf)
                if got is not None:
                    return got
    return None


def _coerce_sections_to_revise(raw: Any) -> list[int]:
    """Normalize verifier ``sections_to_revise`` into unique section numbers in 1..12."""
    out: list[int] = []
    if not isinstance(raw, list):
        return out
    for x in raw:
        n: int | None = None
        if isinstance(x, bool):
            continue
        if isinstance(x, int):
            n = x
        elif isinstance(x, float) and x == int(x):
            n = int(x)
        elif isinstance(x, str):
            m = re.match(r"^\s*(\d{1,2})\b", x.strip())
            if m:
                n = int(m.group(1))
        if n is not None and 1 <= n <= 12 and n not in out:
            out.append(n)
    return out


def _finish_reason_implies_provider_truncation(finish_reason: str | None) -> bool:
    """True when the provider-reported finish reason indicates output-length truncation."""
    if not finish_reason:
        return False
    u = finish_reason.upper()
    if "MAX_TOKENS" in u or "MAX_OUTPUT" in u:
        return True
    if "MAX" in u and "TOKEN" in u:
        return True
    return u in {"LENGTH", "MODEL_LENGTH"}


def _build_verification_result(best_data: dict[str, Any], *, truncated: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "verified": _values_for_canonical_key(best_data, "verified"),
        "contradicted": _values_for_canonical_key(best_data, "contradicted"),
        "unverifiable": _values_for_canonical_key(best_data, "unverifiable"),
    }
    notes_val = best_data.get("notes")
    if isinstance(notes_val, str) and notes_val.strip():
        result["notes"] = notes_val.strip()
    result["refresh_facts"] = bool(best_data.get("refresh_facts"))
    rop = best_data.get("refan_out_providers")
    refan_list: list[str] = []
    if isinstance(rop, list):
        for x in rop:
            s = str(x).strip().lower()
            if s:
                refan_list.append(s)
    result["refan_out_providers"] = refan_list
    result["refan_out_all"] = bool(best_data.get("refan_out_all"))
    result["sections_to_revise"] = _coerce_sections_to_revise(best_data.get("sections_to_revise"))
    sbs = _coerce_sigma_band_sessions(best_data.get("sigma_band_sessions"))
    if sbs:
        result["sigma_band_sessions"] = sbs
    if "sigma_scaling_aggregate_passed" in best_data:
        result["sigma_scaling_aggregate_passed"] = bool(best_data.get("sigma_scaling_aggregate_passed"))
    if truncated:
        result["_truncated"] = True
    return result


def parse_verifier_json(
    text: str,
    *,
    provider_finish_reason: str | None = None,
    provider_raw: Any = None,
) -> dict[str, Any]:
    """Parse verifier model output into verified / contradicted / unverifiable lists (and optional notes).

    ``provider_finish_reason`` / ``provider_raw`` are used to decide whether salvaged JSON should be treated
    as provider-truncated (``_truncated`` + WARNING) vs a benign salvage path on a normal STOP completion.
    """
    raw = text
    candidates = _candidate_dicts_from_text(text)
    repair_used = False
    truncated = False
    best_data: dict[str, Any] | None = None

    if not candidates:
        repaired = _attempt_truncated_json_repair(text)
        if repaired is not None:
            best_data = repaired
            repair_used = True
        else:
            logger.warning(
                "verifier JSON parse failed (no object decoded); verifier_raw=%s",
                raw[:1000] if raw else "",
            )
            return {
                "verified": [],
                "contradicted": [],
                "unverifiable": [],
                "refresh_facts": False,
                "refan_out_providers": [],
                "refan_out_all": False,
                "sections_to_revise": [],
            }
    else:
        best_rank: tuple[int, int] = (-1, -1)
        for data, raw_len in candidates:
            score = _score_verification_dict(data)
            rank = (score, raw_len)
            if rank > best_rank:
                best_rank = rank
                best_data = data

    assert best_data is not None
    max_tokens_hit, _ = detect_max_tokens_truncation(provider_raw)
    if repair_used:
        truncated = bool(
            max_tokens_hit
            or _finish_reason_implies_provider_truncation(provider_finish_reason),
        )
    result = _build_verification_result(best_data, truncated=truncated)
    if truncated:
        raw_bytes = len(text.encode("utf-8"))
        fr = provider_finish_reason if provider_finish_reason is not None else "n/a"
        logger.warning(
            "verifier: response was truncated; finish_reason=%s raw_bytes=%s salvaged "
            "%s verified, %s contradicted, %s unverifiable items",
            fr,
            raw_bytes,
            len(result["verified"]),
            len(result["contradicted"]),
            len(result["unverifiable"]),
        )
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
    retry_max_attempts_fan_out: NotRequired[int]
    retry_base_delay_s: float
    synthesizer_max_input_tokens: int
    summarize_oversized_providers: bool
    summarize_threshold_input_tokens: int
    oversized_summarize_provider: str
    oversized_summarize_model: str
    oversized_summarize_max_output_tokens: int
    oversized_summarize_max_input_tokens: int
    oversized_summarize_min_retention: float
    oversized_summarize_fallback_provider: NotRequired[str | None]
    gemini_cache_ttl_s: int
    synthesizer_cfg: dict[str, Any]
    verifier_name: str
    verifier_model: str | None
    output_dir: str
    provider_responses: Annotated[list[dict[str, Any]], operator.add]
    synthesis_history: Annotated[list[str], operator.add]
    synthesis_meta: Annotated[list[dict[str, Any]], operator.add]
    verification_history: Annotated[list[dict[str, Any]], operator.add]
    followup_questions: Annotated[list[str], operator.add]
    timing_events: Annotated[list[dict[str, Any]], operator.add]
    error_events: Annotated[list[dict[str, Any]], operator.add]
    final_report: str
    drive_upload_enabled: bool
    drive_credentials_path: str | None
    drive_root_folder_id: str | None
    run_environment: NotRequired[str]
    drive_auth_mode: NotRequired[str]
    drive_oauth_token_path: NotRequired[str | None]
    pdf_output_enabled: NotRequired[bool]
    delete_checkpoint_after_success: NotRequired[bool]
    iterative_config_snapshot: NotRequired[dict[str, Any]]
    facts_packet_enabled: NotRequired[bool]
    conditional_fanout_enabled: NotRequired[bool]
    fan_out_on_continue: NotRequired[bool]
    refinement_mode_prompt_enabled: NotRequired[bool]
    facts_packet_md: NotRequired[str]
    last_route_followup_questions: NotRequired[list[str]]
    options_chain_data: NotRequired[dict[str, Any]]
    final_report_full_synthesis: NotRequired[bool]
    equity_prompt_render_context: NotRequired[dict[str, Any]]


def compute_refinement_route_command(state: RefinementState) -> Command[Any]:
    """Decide next graph node after verification (stop / verify-only / provider fan-out)."""
    syn = state["synthesis_history"][-1]
    ver = state["verification_history"][-1]
    conf = parse_overall_confidence(syn)
    syn_passes = len(state.get("synthesis_history") or [])
    n_provider_rounds = len(state.get("provider_responses") or [])
    contrad = ver.get("contradicted") or []
    unver = ver.get("unverifiable") or []
    n_contrad = len(contrad)
    n_unver = len(unver)
    max_it = int(state["max_iterations"])
    threshold = float(state["confidence_threshold"])
    snap: dict[str, Any] = state.get("iterative_config_snapshot") or {}
    skip_unver = bool(snap.get("unverifiable_only_skip_fan_out", True))
    unver_thr = int(snap.get("unverifiable_count_threshold_for_fanout", 3))
    conf_cut = float(snap.get("unverifiable_fanout_confidence_below", 0.8))
    force_fan = bool(snap.get("force_fan_out_on_continue", False))

    logger.info(
        "Node route rounds_completed=%s max_iterations=%s overall_confidence=%s contradicted=%s "
        "unverifiable=%s synthesis_passes=%s",
        n_provider_rounds,
        max_it,
        f"{conf:.4f}" if conf is not None else "none",
        n_contrad,
        n_unver,
        syn_passes,
    )

    if syn_passes >= max_it:
        logger.info("Route decision: finalize (max_iterations reached)")
        return Command(goto="finalize")

    if conf is not None and conf >= threshold and n_contrad == 0 and n_unver == 0:
        logger.info(
            "Route decision: stop confidence=%s contradicted=0 unverifiable=0",
            f"{conf:.2f}",
        )
        return Command(goto="finalize")

    qs: list[str] = []
    for c in contrad:
        qs.append(f"Resolve with primary sources: {c}")
    for u in unver:
        qs.append(f"Cite or verify: {u}")

    high_unver_fanout = (
        n_contrad == 0
        and n_unver >= unver_thr
        and conf is not None
        and conf < conf_cut
    )
    cite_only_mode = (
        n_contrad == 0
        and n_unver > 0
        and skip_unver
        and not force_fan
        and not high_unver_fanout
    )

    if force_fan:
        rreason = "force_fan_out_on_continue"
        if n_contrad > 0:
            logger.info(
                "Route decision: continue (fan_out) followups=%s contradicted=%s reason=%s",
                len(qs),
                n_contrad,
                rreason,
            )
        elif n_unver > 0:
            logger.info(
                "Route decision: continue (fan_out) followups=%s contradicted=0 unverifiable=%s reason=%s",
                len(qs),
                n_unver,
                rreason,
            )
        else:
            logger.info(
                "Route decision: continue (fan_out) followups=%s contradicted=%s reason=%s",
                len(qs),
                n_contrad,
                rreason,
            )
        return Command(
            goto="fan_out",
            update={
                "followup_questions": qs,
                "last_route_followup_questions": qs,
            },
        )

    if cite_only_mode:
        cite_qs = [f"Cite or verify: {u}" for u in unver]
        logger.info(
            "Route decision: continue (verify_only) cite_unverifiable=%s contradicted=0",
            n_unver,
        )
        return Command(
            goto="synthesize",
            update={
                "followup_questions": cite_qs,
                "last_route_followup_questions": cite_qs,
            },
        )

    if n_contrad > 0 and qs:
        logger.info(
            "Route decision: continue (fan_out) followups=%s contradicted=%s reason=contradictions",
            len(qs),
            n_contrad,
        )
        return Command(
            goto="fan_out",
            update={
                "followup_questions": qs,
                "last_route_followup_questions": qs,
            },
        )

    if high_unver_fanout and qs:
        logger.info(
            "Route decision: continue (fan_out) followups=%s contradicted=0 unverifiable=%s reason=high_unverifiable_count",
            len(qs),
            n_unver,
        )
        return Command(
            goto="fan_out",
            update={
                "followup_questions": qs,
                "last_route_followup_questions": qs,
            },
        )

    if qs:
        if n_unver > 0 and n_contrad == 0:
            logger.info(
                "Route decision: continue (fan_out) followups=%s contradicted=0 unverifiable=%s reason=mixed_or_low_confidence",
                len(qs),
                n_unver,
            )
        else:
            logger.info(
                "Route decision: continue (fan_out) followups=%s contradicted=%s",
                len(qs),
                n_contrad,
            )
        return Command(
            goto="fan_out",
            update={
                "followup_questions": qs,
                "last_route_followup_questions": qs,
            },
        )

    logger.info("Route decision: continue (fan_out) followups=0")
    return Command(
        goto="fan_out",
        update={
            "followup_questions": [],
            "last_route_followup_questions": [],
        },
    )


def _estimate_prompt_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _build_synthesis_refinement_markdown(state: RefinementState) -> str | None:
    """Extra markdown for the synthesizer on iteration >= 2."""
    sh = state.get("synthesis_history") or []
    if not sh:
        return None
    parts: list[str] = []
    facts = (state.get("facts_packet_md") or "").strip()
    if facts and bool(state.get("facts_packet_enabled", True)):
        parts.append("## Frozen market facts\n\n" + facts)
    parts.append("## Prior synthesis to revise\n\n" + sh[-1].strip())
    vh = state.get("verification_history") or []
    if vh:
        parts.append("## Latest verification JSON\n\n```json\n" + json.dumps(vh[-1], indent=2) + "\n```")
    fu = state.get("followup_questions") or []
    if fu:
        parts.append("## Router follow-up targets\n\n" + "\n".join(f"- {x}" for x in fu))
    parts.append(
        "Revise the full 12-section synthesis to resolve verification issues. "
        "When frozen market facts are present, prefer them over stale narrative numbers. "
        "End with the required OVERALL_CONFIDENCE line."
    )
    return "\n\n".join(parts)


def _fan_out_task_body(state: RefinementState) -> str:
    extra = "\n\n".join(state.get("followup_questions", []))
    base = state["original_prompt"]
    if extra:
        return f"{base}\n\n### Follow-up verification targets\n{extra}"
    return base


_REFINEMENT_PRIOR_SYNTHESIS_MAX_CHARS = 120_000


def _prior_synthesis_for_provider_refine(text: str, *, prior_round: int) -> str:
    """Full prior synthesis for provider refinement, abridged if it would dominate the context window."""
    t = text.rstrip()
    if len(t) <= _REFINEMENT_PRIOR_SYNTHESIS_MAX_CHARS:
        return t
    return round_summary_for_changelog(
        t,
        iteration_index=prior_round,
        max_chars=50_000,
    )


def _sections_to_revise_markdown(sections: list[int]) -> str:
    if not sections:
        return (
            "### Sections to revise\n\n"
            "No explicit section list from the verifier — revise based on the follow-up targets below "
            "and your judgment. Prefer minimal edits outside areas clearly implicated by those targets."
        )
    joined = ", ".join(str(s) for s in sections)
    return (
        f"### Sections to revise: [{joined}]\n\n"
        "Focus substantive changes on these sections. For other sections (1-12 not listed), you may leave "
        "them largely aligned with the prior synthesis — briefly confirm them or note only if you would "
        "materially change them."
    )


def _refinement_mode_block(*, iteration: int, max_iterations: int) -> str:
    """Markdown prefix for iteration 2+ when fan-out providers are actually invoked."""
    return (
        f"# REFINEMENT MODE (iteration {iteration} of {max_iterations})\n\n"
        "You already have the frozen FACTS packet above and a prior synthesis. Your job is to **refine, not re-derive**.\n\n"
        "Rules:\n"
        "- DO NOT re-fetch market data — the FACTS packet above is authoritative for the static numbers "
        "(last close, IV, PCR, short interest, 1-sigma / 2-sigma / 3-sigma implied moves, analyst targets, historical reactions).\n"
        "- DO NOT re-derive market primitives (1-sigma / 2-sigma / 3-sigma ranges, IV, PCR, baseline anchors) that are already "
        "provided in FACTS. Quote them directly.\n"
        "- DO quote the relevant numbers verbatim from FACTS where they appear in your sections.\n"
        "- DO focus your reasoning on the verifier's specific concerns, on the **follow-up verification targets** "
        "below, and on revising or strengthening the sections called out under **Sections to revise**. Adjust "
        "probabilities, ranges, qualitative emphasis, and conclusions accordingly.\n"
        "- DO NOT re-write sections that did not get verifier flags or new disagreements. You may briefly confirm them.\n"
    )


def _compose_fan_out_user_body(
    state: RefinementState,
    *,
    it_no: int,
    skip_fanout: bool,
) -> str:
    """User message body for provider fan-out (before optional facts prefix)."""
    core = state["original_prompt"]
    followups = state.get("followup_questions") or []
    extra = "\n\n".join(followups)
    follow_block = (
        f"\n\n### Follow-up verification targets\n{extra}"
        if extra
        else (
            "\n\n### Follow-up verification targets\n\n"
            "(No explicit bullet list in router output — still apply REFINEMENT MODE using the latest "
            "verification concerns from the prior round.)\n"
        )
    )

    refinement_on = (
        bool(state.get("refinement_mode_prompt_enabled", True))
        and it_no >= 2
        and not skip_fanout
        and bool(state.get("synthesis_history"))
    )
    if not refinement_on:
        return f"{core}{follow_block}" if extra else core

    static = EQUITY_ANALYST_SYSTEM_PROMPT
    sep = f"{static}\n\n"
    ver: dict[str, Any] = {}
    if state.get("verification_history"):
        ver = state["verification_history"][-1]
    sec = _coerce_sections_to_revise(ver.get("sections_to_revise"))
    sections_md = _sections_to_revise_markdown(sec)
    prior_round = it_no - 1
    prior_raw = state["synthesis_history"][-1]
    prior_excerpt = _prior_synthesis_for_provider_refine(prior_raw, prior_round=prior_round)
    refine = _refinement_mode_block(iteration=it_no, max_iterations=int(state["max_iterations"]))
    middle = (
        f"{refine}\n{sections_md}\n\n"
        f"# Prior synthesis (round {prior_round})\n\n{prior_excerpt.strip()}"
        f"{follow_block}\n\n"
    )
    if core.startswith(sep):
        user_only = core[len(sep) :]
        return f"{sep}{middle}{user_only}"
    return f"{middle}{core}"


def _make_refinement_nodes(registry: ProviderRegistry) -> dict[str, Any]:
    async def fan_out(state: RefinementState) -> dict[str, Any]:
        out = Path(state["output_dir"])
        round_idx = len(state.get("provider_responses", []))
        it_no = round_idx + 1
        max_it = state["max_iterations"]
        facts_enabled = bool(state.get("facts_packet_enabled", True))
        conditional = bool(state.get("conditional_fanout_enabled", True))
        allowed_names = frozenset(str(n) for n in state["providers"])

        ver_for_directives: dict[str, Any] = {}
        if it_no >= 2 and state.get("verification_history"):
            ver_for_directives = state["verification_history"][-1]

        facts_label = "off"
        facts_md = (state.get("facts_packet_md") or "").strip()
        state_update: dict[str, Any] = {}

        if facts_enabled and it_no >= 2 and ver_for_directives.get("refresh_facts") and state.get(
            "synthesis_history"
        ):
            snap = state.get("iterative_config_snapshot") or {}
            try:
                cfg_fp = RunConfig.model_validate(snap)
                facts_md = (
                    await extract_facts_packet(
                        synthesis_text=state["synthesis_history"][-1],
                        symbol=str(state.get("symbol", "")),
                        config=cfg_fp,
                    )
                ).strip()
                write_facts_packet(out, facts_md + ("\n" if not facts_md.endswith("\n") else ""))
                state_update["facts_packet_md"] = facts_md
                facts_label = "refreshed"
            except Exception as exc:
                logger.warning("facts_packet: refresh failed iteration=%s err=%r", it_no, exc)
                facts_label = "frozen" if facts_md else "off"
        elif facts_enabled and it_no >= 2 and facts_md:
            facts_label = "frozen"
        elif facts_enabled and it_no == 1:
            facts_label = "pending"

        refan_all = bool(ver_for_directives.get("refan_out_all"))
        raw_refan = ver_for_directives.get("refan_out_providers") or []
        refan_set = {str(x).strip().lower() for x in raw_refan if str(x).strip()} & allowed_names

        fan_out_on_continue = bool(state.get("fan_out_on_continue", True))
        last_route_followups = [str(x).strip() for x in (state.get("last_route_followup_questions") or []) if str(x).strip()]
        router_requests_fanout = fan_out_on_continue and bool(last_route_followups)

        skip_fanout = (
            it_no >= 2
            and conditional
            and not refan_all
            and not refan_set
            and not router_requests_fanout
        )
        partial_fanout = it_no >= 2 and conditional and bool(refan_set) and not refan_all

        task_body = _compose_fan_out_user_body(state, it_no=it_no, skip_fanout=skip_fanout)
        body = task_body
        if facts_enabled and it_no >= 2 and facts_md:
            body = facts_frozen_user_prefix(facts_markdown=facts_md) + task_body

        if skip_fanout:
            n_drop = len(last_route_followups)
            if not fan_out_on_continue and last_route_followups:
                skip_reason = "fan_out_on_continue_disabled"
            else:
                skip_reason = "conditional_fanout_without_verifier_refan_and_no_router_followups"
            logger.info(
                "Iteration %s: fan_out=skipped (reason=%s) followups_dropped=%s facts_packet=%s",
                it_no,
                skip_reason,
                n_drop,
                facts_label,
            )
        else:
            if it_no < 2:
                run_reason = "initial_round"
            elif refan_all:
                run_reason = "verifier_refan_out_all"
            elif refan_set:
                run_reason = "verifier_refan_out_providers"
            elif router_requests_fanout:
                run_reason = "router_continue_fan_out"
            elif not conditional:
                run_reason = "conditional_fanout_disabled"
            else:
                run_reason = "provider_fanout"
            n_fu = len(last_route_followups)
            run_names = sorted(refan_set) if partial_fanout else list(state["providers"])
            logger.info(
                "Iteration %s: fan_out=running providers=%s followups=%s reason=%s facts_packet=%s",
                it_no,
                run_names,
                n_fu,
                run_reason,
                facts_label,
            )
        est_full = _estimate_prompt_tokens(_fan_out_task_body(state))
        n_providers = len(state["providers"])
        saved_tokens = 0
        if skip_fanout:
            saved_tokens = est_full * n_providers
        elif partial_fanout:
            saved_tokens = est_full * max(0, n_providers - len(refan_set))

        if saved_tokens > 0:
            logger.info(
                "Iteration %s saved approx %s tokens vs full re-run",
                it_no,
                saved_tokens,
            )

        logger.info(
            "Node fan_out iteration=%s max_iterations=%s providers=%s output_dir=%s",
            it_no,
            max_it,
            list(state["providers"]),
            str(out.resolve()),
        )

        fan_ctx_raw = state.get("equity_prompt_render_context")
        fan_ctx = fan_ctx_raw if isinstance(fan_ctx_raw, dict) else None
        with prompt_call_context(node="fan_out", iteration=it_no, analyst_render_context=fan_ctx):
            pcs_raw = state.get("provider_configs")
            if not pcs_raw:
                pcs_raw = [{"name": n, "web_search": None, "request_timeout_s": None} for n in state["providers"]]
            pcs = [ProviderConfig.model_validate(d) for d in pcs_raw]
            cfg_req_timeout = float(state.get("request_timeout_s", 180.0))
            cfg_mot = int(state.get("max_output_tokens", 16_000))
            retry_default = int(state.get("retry_max_attempts", 3))
            retry_max = int(state.get("retry_max_attempts_fan_out", retry_default))
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

            async def _run_one(pc: ProviderConfig, prompt_body: str) -> ProviderResponse:
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
                        user_only = prompt_body[len(sep) :] if prompt_body.startswith(sep) else prompt_body
                        return await p.generate(
                            prompt_body,
                            enable_web_search=ws,
                            max_output_tokens=mot,
                            prompt_cache_enabled=pce,
                            user_message_for_cache=user_only,
                            force_tool_use=bool(state.get("anthropic_force_tool_use", True)),
                        )
                    if isinstance(p, GeminiProvider) and pce and gemini_cache_index is not None:
                        static = EQUITY_ANALYST_SYSTEM_PROMPT
                        sep = f"{static}\n\n"
                        if prompt_body.startswith(sep):
                            return await p.generate(
                                prompt_body,
                                enable_web_search=ws,
                                max_output_tokens=mot,
                                cacheable_prefix=static,
                                user_message_for_cache=prompt_body[len(sep) :],
                            )
                    if isinstance(p, OpenAIProvider):
                        static = EQUITY_ANALYST_SYSTEM_PROMPT
                        sep = f"{static}\n\n"
                        if prompt_body.startswith(sep):
                            return await p.generate(
                                prompt_body,
                                enable_web_search=ws,
                                max_output_tokens=mot,
                                cacheable_prefix=static,
                                user_message_for_cache=prompt_body[len(sep) :],
                            )
                    return await p.generate(prompt_body, enable_web_search=ws, max_output_tokens=mot)

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

            parallel_wall = 0.0
            responses: dict[str, ProviderResponse] = {}

            if skip_fanout:
                prev = state["provider_responses"][-1]["responses"]
                responses = {k: _dict_to_response(v) for k, v in prev.items()}
            else:
                to_run = pcs if not partial_fanout else [pc for pc in pcs if pc.name in refan_set]
                if partial_fanout and not to_run:
                    logger.warning(
                        "refan_out_providers produced an empty runnable set; falling back to full fan-out",
                    )
                    to_run = pcs
                    partial_fanout = False
                stop_hb = asyncio.Event()
                hb = asyncio.create_task(_heartbeat(stop_hb, [pc.name for pc in to_run]))
                batch_t0 = time.perf_counter()
                try:
                    res_list = await asyncio.gather(
                        *[_run_one(pc, body) for pc in to_run],
                        return_exceptions=True,
                    )
                finally:
                    stop_hb.set()
                    hb.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await hb
                parallel_wall = time.perf_counter() - batch_t0
                ran: dict[str, ProviderResponse] = {}
                for pc, item in zip(to_run, res_list, strict=True):
                    if isinstance(item, ProviderResponse):
                        ran[pc.name] = item
                    elif isinstance(item, Exception):
                        ran[pc.name] = failure_response(pc.name, item, latency_s=None)
                    else:
                        raise item
                if partial_fanout:
                    prev = state["provider_responses"][-1]["responses"]
                    for pc in pcs:
                        if pc.name in refan_set:
                            responses[pc.name] = ran[pc.name]
                        else:
                            responses[pc.name] = _dict_to_response(prev[pc.name])
                else:
                    responses = ran

        iter_dir = out / "iterations"
        iter_dir.mkdir(parents=True, exist_ok=True)
        lines_out: list[str] = []
        for name, resp in responses.items():
            if skip_fanout:
                lines_out.append(
                    f"## {name}\n\n(fan-out skipped — reused iteration {round_idx} body)\n\n{resp.text}\n",
                )
            else:
                lines_out.append(f"## {name}\n\n{resp.text}\n")
        (iter_dir / f"iteration_{round_idx + 1}_providers.md").write_text("\n".join(lines_out), encoding="utf-8")

        ser = {"responses": {k: _response_to_dict(v) for k, v in responses.items()}}
        out_fan: dict[str, Any] = {
            "provider_responses": [ser],
            "timing_events": [
                {"iteration": it_no, "providers_parallel_wall_s": parallel_wall},
            ],
        }
        out_fan.update(state_update)
        return out_fan

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
        syn_max_in = int(state.get("synthesizer_max_input_tokens", 100_000))
        retry_max = int(state.get("retry_max_attempts", 3))
        retry_base = float(state.get("retry_base_delay_s", 2.0))
        syn_ws = effective_synthesizer_web_search(run_default=state["enable_web_search"], syn=syn_cfg)
        summarize_fallback_llm = None
        fb_name = state.get("oversized_summarize_fallback_provider")
        if not fb_name and state.get("iterative_config_snapshot"):
            fb_name = (state["iterative_config_snapshot"] or {}).get("oversized_summarize_fallback_provider")
        if fb_name:
            pcs_fb = [ProviderConfig.model_validate(d) for d in state["provider_configs"]]
            for pc in pcs_fb:
                if pc.name == fb_name:
                    summarize_fallback_llm = registry.create(pc.name, model=pc.model)
                    break
        s0 = time.perf_counter()
        it_no = round_idx + 1
        err_ev: list[dict[str, Any]] = []
        refinement = _build_synthesis_refinement_markdown(state) if round_idx >= 1 else None
        try:
            with prompt_call_context(node="synthesize", iteration=it_no):
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
                        symbol=state.get("symbol"),
                        summarize_oversized_providers=bool(state.get("summarize_oversized_providers", True)),
                        summarize_threshold_input_tokens=int(
                            state.get("summarize_threshold_input_tokens", 8000),
                        ),
                        oversized_summarize_provider=str(state.get("oversized_summarize_provider", "gemini")),
                        oversized_summarize_model=str(
                            state.get("oversized_summarize_model", "gemini-3-flash-preview"),
                        ),
                        oversized_summarize_max_output_tokens=int(
                            state.get("oversized_summarize_max_output_tokens", 8192),
                        ),
                        oversized_summarize_max_input_tokens=int(
                            state.get("oversized_summarize_max_input_tokens", 100_000),
                        ),
                        oversized_summarize_min_retention=float(
                            state.get("oversized_summarize_min_retention", 0.40),
                        ),
                        oversized_summarize_fallback_provider=summarize_fallback_llm,
                        refinement_markdown=refinement,
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
        syn_path = iter_dir / f"iteration_{round_idx + 1}_synthesis.md"
        syn_body = text + "\n"
        syn_path.write_text(syn_body, encoding="utf-8")
        maybe_write_pdf_sibling(
            pdf_output_enabled=bool(state.get("pdf_output_enabled", True)),
            md_path=syn_path,
            markdown_text=syn_body,
        )
        facts_state: dict[str, Any] = {}
        if bool(state.get("facts_packet_enabled", True)) and round_idx == 0 and not result.response.model.startswith(
            "error:",
        ):
            snap = state.get("iterative_config_snapshot") or {}
            try:
                cfg0 = RunConfig.model_validate(snap)
                md = await extract_facts_packet(
                    synthesis_text=result.response.text,
                    symbol=str(state.get("symbol", "")),
                    config=cfg0,
                )
                write_facts_packet(out, md)
                facts_state["facts_packet_md"] = md.rstrip() + "\n"
            except Exception as exc:
                logger.warning("facts_packet: initial extraction failed err=%r", exc)

        out_update: dict[str, Any] = {
            "synthesis_history": [result.response.text],
            "synthesis_meta": [
                {
                    "provider": result.response.provider_name,
                    "model": result.response.model,
                    "usage": asdict(result.response.usage),
                    "latency_s": result.response.latency_s,
                }
            ],
            "timing_events": [{"iteration": it_no, "synthesis_wall_s": syn_wall}],
        }
        out_update.update(facts_state)
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
        vmt = int(state.get("verifier_max_output_tokens", 16_384))
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
                # Gemini 3 shares max_output_tokens with internal "thinking"; callers pass
                # thinking_budget=0 to reserve visible completion budget. Gemini 3 rejects
                # thinking_budget=0 — GeminiProvider maps it to a positive budget; raise the
                # completion cap so JSON still fits.
                vm = v_model if isinstance(v_model, str) and v_model.strip() else None
                resolved_verifier_model = vm or DEFAULT_GEMINI_MODEL
                eff_vmt = (
                    max(vmt, 32_768)
                    if gemini_model_requires_nonzero_thinking_budget(resolved_verifier_model)
                    else vmt
                )
                return await verifier.generate(
                    prompt,
                    enable_web_search=state["enable_web_search"],
                    max_output_tokens=eff_vmt,
                    cacheable_prefix=None,
                    thinking_budget=0,
                )
            return await verifier.generate(
                prompt,
                enable_web_search=state["enable_web_search"],
                max_output_tokens=vmt,
            )

        try:
            with prompt_call_context(node="verify", iteration=it_no):
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
        data = parse_verifier_json(
            resp.text,
            provider_finish_reason=provider_finish_reason_label(resp.raw),
            provider_raw=resp.raw,
        )
        data = augment_verifier_result_with_sigma_structural_checks(
            syn,
            data,
            options_chain_data=state.get("options_chain_data"),
            symbol=str(state.get("symbol", "")),
        )
        verify_body = json.dumps(data, indent=2) + "\n"
        verify_path = iter_dir / f"iteration_{round_idx + 1}_verify.md"
        verify_path.write_text(verify_body, encoding="utf-8")
        maybe_write_pdf_sibling(
            pdf_output_enabled=bool(state.get("pdf_output_enabled", True)),
            md_path=verify_path,
            markdown_text=verify_body,
        )
        out_v: dict[str, Any] = {
            "verification_history": [data],
            "timing_events": [{"iteration": it_no, "verify_wall_s": ver_wall}],
        }
        if err_ev:
            out_v["error_events"] = err_ev
        return out_v

    def route(state: RefinementState) -> Command[Any]:
        return compute_refinement_route_command(state)

    async def finalize(state: RefinementState) -> dict[str, Any]:
        out = Path(state["output_dir"])
        logger.info("Node finalize output_dir=%s rounds=%s", str(out.resolve()), len(state["provider_responses"]))
        parts: list[str] = [
            f"# Refined equity report: {state['symbol']}\n",
            "## Iteration changelog\n",
        ]
        full_changelog = bool(state.get("final_report_full_synthesis", True))
        for i, syn in enumerate(state["synthesis_history"], start=1):
            body = syn.rstrip() if full_changelog else round_summary_for_changelog(syn, iteration_index=i)
            round_heading = (
                f"### Round {i} synthesis\n\n" if full_changelog else f"### Round {i} synthesis (summary)\n\n"
            )
            parts.append(f"{round_heading}{body}\n\n")
        parts.append("## Verification summary\n\n")
        trunc_notes = [
            f"(round {i} verifier output was truncated; partial recovery)"
            for i, ver in enumerate(state["verification_history"], start=1)
            if ver.get("_truncated")
        ]
        if trunc_notes:
            parts.append(" ".join(trunc_notes) + "\n\n")
        for i, ver in enumerate(state["verification_history"], start=1):
            parts.append(f"### Round {i}\n```json\n{json.dumps(ver, indent=2)}\n```\n\n")
        parts.append("## Final synthesis (last round)\n\n")
        parts.append(state["synthesis_history"][-1])
        report = "\n".join(parts)
        out.mkdir(parents=True, exist_ok=True)
        pdf_on = bool(state.get("pdf_output_enabled", True))
        final_syn_path = out / "synthesis.md"
        final_syn_body = report + "\n"
        final_syn_path.write_text(final_syn_body, encoding="utf-8")
        maybe_write_pdf_sibling(
            pdf_output_enabled=pdf_on,
            md_path=final_syn_path,
            markdown_text=final_syn_body,
        )
        iter_dir = out / "iterations"
        for i, syn in enumerate(state["synthesis_history"], start=1):
            ver = state["verification_history"][i - 1] if i <= len(state["verification_history"]) else {}
            block = f"# Iteration {i}\n\n## Synthesis\n\n{syn}\n\n## Verification\n\n{json.dumps(ver, indent=2)}\n"
            iter_md = iter_dir / f"iteration_{i}.md"
            iter_md.write_text(block, encoding="utf-8")
            maybe_write_pdf_sibling(
                pdf_output_enabled=pdf_on,
                md_path=iter_md,
                markdown_text=block,
            )

        run_json = out / "run.json"
        timing_summary = merge_timing_events(state.get("timing_events", []))
        meta = (
            json.loads(run_json.read_text(encoding="utf-8"))
            if run_json.is_file()
            else {}
        )
        meta["timing"] = timing_summary
        meta["iterations_completed"] = len(state.get("provider_responses", []))
        meta["verification_history"] = state.get("verification_history", [])
        syn_meta = state.get("synthesis_meta") or []
        if syn_meta and isinstance(syn_meta[-1], dict):
            meta["synthesis"] = syn_meta[-1]
        prior_errs = meta.get("errors")
        if not isinstance(prior_errs, list):
            prior_errs = []
        merged_errs: list[Any] = list(prior_errs) + list(state.get("error_events", []))
        meta["errors"] = merged_errs
        run_json.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        logger.info("Iterative wall-clock timing summary: %s", timing_summary)

        cfg_for_extract: RunConfig | None = None
        try:
            cfg_raw = meta.get("config")
            if isinstance(cfg_raw, dict):
                cfg_for_extract = RunConfig.model_validate(cfg_raw)
        except Exception:
            logger.warning("finalize: invalid config snapshot; skipping prediction_extract")
            cfg_for_extract = None
        if cfg_for_extract is not None and cfg_for_extract.prediction_extract_enabled:
            await run_prediction_extract_for_run_dir(run_dir=out, cfg=cfg_for_extract)

        if bool(state.get("drive_upload_enabled", False)):
            cred_raw = state.get("drive_credentials_path")
            root_raw = state.get("drive_root_folder_id")
            auth_raw = state.get("drive_auth_mode")
            oauth_tok_raw = state.get("drive_oauth_token_path")
            env_raw = state.get("run_environment")
            run_env: RunEnvironment = (
                cast(RunEnvironment, env_raw) if env_raw in ("production", "test") else "production"
            )
            resolved_mode: DriveAuthMode = (
                cast(DriveAuthMode, auth_raw)
                if auth_raw in ("service_account", "oauth_user")
                else "service_account"
            )
            await maybe_upload_run_to_drive_raw(
                drive_upload_enabled=True,
                drive_credentials_path=cred_raw if isinstance(cred_raw, str) else None,
                drive_root_folder_id=root_raw if isinstance(root_raw, str) else None,
                out_dir=out,
                run_id=out.name,
                append_synthesis_footer=True,
                drive_auth_mode=resolved_mode,
                drive_oauth_token_path=oauth_tok_raw if isinstance(oauth_tok_raw, str) else None,
                run_environment=run_env,
            )

        maybe_delete_iterative_checkpoint(
            out,
            delete_checkpoint_after_success=bool(state.get("delete_checkpoint_after_success", True)),
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
        "retry_max_attempts_fan_out": cfg.retry_max_attempts_fan_out,
        "retry_base_delay_s": float(cfg.retry_base_delay_s),
        "synthesizer_max_input_tokens": cfg.synthesizer_max_input_tokens,
        "summarize_oversized_providers": cfg.summarize_oversized_providers,
        "summarize_threshold_input_tokens": cfg.summarize_threshold_input_tokens,
        "oversized_summarize_provider": cfg.oversized_summarize_provider,
        "oversized_summarize_model": cfg.oversized_summarize_model,
        "oversized_summarize_max_output_tokens": cfg.oversized_summarize_max_output_tokens,
        "oversized_summarize_max_input_tokens": cfg.oversized_summarize_max_input_tokens,
        "oversized_summarize_min_retention": cfg.oversized_summarize_min_retention,
        "oversized_summarize_fallback_provider": cfg.oversized_summarize_fallback_provider,
        "gemini_cache_ttl_s": cfg.gemini_cache_ttl_s,
        "synthesizer_cfg": cfg.synthesizer.model_dump(mode="json"),
        "verifier_name": cfg.verifier_provider,
        "verifier_model": cfg.verifier_model,
        "output_dir": str(output_dir.resolve()),
        "drive_upload_enabled": cfg.drive_upload_enabled,
        "drive_credentials_path": cfg.drive_credentials_path,
        "drive_root_folder_id": cfg.drive_root_folder_id,
        "run_environment": cfg.run_environment,
        "drive_auth_mode": cfg.drive_auth_mode,
        "drive_oauth_token_path": cfg.drive_oauth_token_path,
        "pdf_output_enabled": cfg.pdf_output_enabled,
        "delete_checkpoint_after_success": cfg.delete_checkpoint_after_success,
        "iterative_config_snapshot": cfg.model_dump(mode="json"),
        "facts_packet_enabled": cfg.facts_packet_enabled,
        "conditional_fanout_enabled": cfg.conditional_fanout_enabled,
        "fan_out_on_continue": cfg.fan_out_on_continue,
        "refinement_mode_prompt_enabled": cfg.refinement_mode_prompt_enabled,
        "options_chain_data": rendered.context.get("options_chain_data") or {},
        "final_report_full_synthesis": cfg.final_report_full_synthesis,
        "equity_prompt_render_context": copy.deepcopy(rendered.context),
    }


def dry_run_compile_only(*, registry: ProviderRegistry) -> list[str]:
    from langgraph.checkpoint.memory import MemorySaver

    app = compile_refinement_workflow(registry=registry, checkpointer=MemorySaver())
    nodes = app.get_graph().nodes
    out = sorted(n for n in nodes if not str(n).startswith("__"))
    logger.debug("Dry-run graph inspection nodes=%s", out)
    return out
