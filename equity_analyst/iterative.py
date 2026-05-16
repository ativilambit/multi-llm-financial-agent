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
from equity_analyst.drift_bounds import (
    PROB_UP_MISMATCH_TOLERANCE_PP,
    bound_daily_drift,
    computed_prob_up_pct,
)
from equity_analyst.drive_uploader import DriveAuthMode, maybe_upload_run_to_drive_raw
from equity_analyst.facts_packet import (
    extract_facts_packet,
    facts_frozen_user_prefix,
    write_facts_packet,
)
from equity_analyst.gemini_cache import GeminiCacheIndex
from equity_analyst.options_chain import (
    _parse_earnings_calendar_date,
    event_jump_implied_move_pct_from_prompt_dict,
    options_chain_expiry_audit_messages,
)
from equity_analyst.pdf_writer import maybe_write_pdf_sibling
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
from equity_analyst.run_json_serde import canonical_run_document_dict, format_run_json_for_disk
from equity_analyst.sigma_summary import (
    SigmaSummaryFileModel,
    parse_sigma_summary_json,
    sigma_summary_json_present_but_invalid,
)
from equity_analyst.synthesizer import (
    SynthesisResult,
    Synthesizer,
    detect_max_tokens_truncation,
    format_synthesis_artifact_markdown,
    provider_finish_reason_label,
)
from equity_analyst.synthesizer_blend import (
    ADVISORY_BLEND_PROVIDER_SPREAD_THRESHOLD_POINTS,
    T0BlendPreset,
    horizon_blend_ratio_followups,
    normalize_t0_blend_preset,
    suggested_blend_consistency_followups,
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

When the excerpt states **horizon blend** defaults, the equity/synthesizer prompts fix literals as **qual : quant** (qualitative first). For **T-0 / T+1..T+5** rows the digits must match the fenced table (**49 : 51**). For **T-3..T-1** use the fenced table (**55 : 45**). Flag **unverifiable** if the excerpt uses digit-inverted colon pairs, emits **both** canonical and inverted pairs, swaps **Quant**/**Qual** lens names against **qual**/**quant** ordering, uses a **quant-then-qual** colon label for the blend column, uses **qualitative-colon-quantitative** as a pseudo-blend header, or uses **%-wording** that contradicts the fenced digit pairs. (Deterministic post-pass also enforces this—still flag if you see it in your excerpt.)

"""
    + f"""When multi-provider material shows **advisory** (non-canonical) **`qual : quant`** integers for the **same** horizon bucket disagreeing by more than **{ADVISORY_BLEND_PROVIDER_SPREAD_THRESHOLD_POINTS}** points on **either** int, check that the synthesis includes a clearly labeled **`### Final suggested blend (advisory — consensus)`** table (four horizon rows) whose per-row **Dissent notes** name **dissenting providers together with those providers' stated `qual : quant` pairs** when those pairs appear in provider text. If that heading and table are **missing**, add one **unverifiable** item requesting them (short).

Synthesis MUST NOT introduce numeric qualitative tilts. Flag any '+5', '+10', or 'tilt' applied to quantitative values (probabilities, sigma bands, scenario weights).

When the excerpt concerns **post-earnings sigma bands** (variance-additive horizons), require **machine-parseable** inputs before any band table: the literal tokens ``event_jump=`` and ``daily_vol=`` (exact spelling, ASCII ``=``) with **two decimals** and ``%/day`` on ``daily_vol``, inside a fenced code block—same mandatory shape as the equity prompt (no LaTeX ``\\_`` escapes, no Markdown italics, no Unicode multipliers). If either token is missing from the excerpt, add an **unverifiable** item demanding those two lines in that exact format (plus ``iv_crush_multiplier=`` / ``daily_vol_raw=`` / ``daily_vol=`` when IV crush context applies).

When excerpted claims concern 1-sigma / 2-sigma / 3-sigma **dollar** bands, treat **prior-close anchoring** and **labeled same-day intraday `[low-1.00, high+1.00]` (USD)** as both valid when the synthesis states which anchor it used; do not flag a contradiction solely because two runs used different branches of the equity prompt.

"""
    + f"""**{_GS} band structural checks (mandatory pass/fail in your JSON, plus cite synthesis gaps):**
- For every session or horizon in the excerpt that reports **1{_GS} / 2{_GS} / 3{_GS}** bands tied to options-implied width, require an explicit **vol baseline**: a **real listed options expiry (YYYY-MM-DD)** used for implied move, **or** the literal label **HV30 sqrt(t) scaling** (or text clearly equivalent).
- **No fake 0-DTE implied move:** if the excerpt reports same-day implied-move {_GS} for a session **without** naming a chain expiry that could support that session, add a concise **unverifiable** item naming the session and asking for the nearest weekly expiry + sqrt(target_DTE/chosen_DTE) scaling or HV30 fallback.
- **Variance-additive event+diffusion (canonical when the horizon crosses the earnings print and the target is post-event):** if the excerpt states **event_jump=** ... **%** and **daily_vol=** ... **%** (post-event decomposition), verify each dated **1{_GS}** (or **3{_GS}/3**) half-width satisfies **{_GS}^2 ≈ event_jump^2 + n·daily_vol^2** where **n** counts NYSE weekdays **strictly after** the earnings **calendar** date through that row's date (same **n** as the server {_GS} table: **n=0** on the earnings calendar session is the raw jump only). For two rows with indices n1<n2, **{_GS}^2(n2)-{_GS}^2(n1)** should match **(n2-n1)·daily_vol^2** within **+/-25%**. If it fails, add **unverifiable** items with concrete numbers and ask for a corrected **daily_vol** or explicit regime labeling.
- **sqrt(t) coherence (fallback, single IV baseline only):** apply **only** when the excerpt has **no** ``event_jump=`` / ``daily_vol=`` literals **and** the dated {_GS} rows are **entirely pre-earnings** or **entirely post-earnings** (no earnings session **between** the earliest and latest dated row). When those literals **are** present, use the **variance-additive** check instead — do **not** demand sqrt(t) ratios across an event jump. When sqrt(t) **does** apply (pre-event constant-IV or post-event single-baseline window), the ratio of **3{_GS} half-width %** values should track **sqrt(Delta trading_sessions)** within **+/-25%** unless the excerpt explicitly flags a **vol regime change**. If incompatible, add **unverifiable** follow-ups naming dates, observed ratio, expected sqrt(N), and ask to re-derive or label distinct regimes.
- **Unsourced options-chain numerics:** Flag any options-chain numeric claim (**PCR**, **IV**, **OI**, **volume**, **premium**, **breakeven** for a **non-current** / historical session, etc.) that lacks an inline ``http(s)://`` URL or a ``Source:`` attribution **in the same paragraph**. Add to **unverifiable** with: ``Cite or verify: <provider/synthesis> claims <metric>=<value> for <date> without a primary source.``
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


def _dated_rows_for_variance_additive_check(
    by_date: dict[date, tuple[float, str]],
    earnings_calendar: date | None,
) -> dict[date, tuple[float, str]]:
    """Keep only rows on/after ``earnings_calendar`` when that calendar date is known."""
    if earnings_calendar is None or not by_date:
        return dict(by_date)
    post = {d: v for d, v in by_date.items() if d >= earnings_calendar}
    return post if post else {}


def _merge_dated_sigma_rows_first_wins(
    dated: list[tuple[date, float, float, str]],
) -> tuple[dict[date, tuple[float, str]], dict[date, tuple[float, str]]]:
    """First row per calendar date wins; returns ``(by_three_sigma_half, by_one_sigma_half)``."""
    by_w3: dict[date, tuple[float, str]] = {}
    by_s1: dict[date, tuple[float, str]] = {}
    for d, w3_half, s1_half, lbl in dated:
        if d not in by_w3:
            by_w3[d] = (w3_half, lbl)
            by_s1[d] = (s1_half, lbl)
    return by_w3, by_s1


# Loosely match styled literals: optional LaTeX ``\\`` before ``_``, ``:`` or ``=``, optional ``\\`` before ``%``,
# optional ``/day`` suffix on daily_vol.
_EVENT_JUMP_PCT_RE = re.compile(
    r"event\\?_jump\s*[:=]\s*(\d+(?:\.\d+)?|\.\d+)\s*\\?%",
    re.IGNORECASE,
)
_DAILY_VOL_PCT_RE = re.compile(
    r"daily\\?_vol\s*[:=]\s*(\d+(?:\.\d+)?|\.\d+)\s*\\?%(?:\s*/\s*day)?",
    re.IGNORECASE,
)


_SESSION_HEAD_RE = re.compile(
    r"\b((?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|"
    r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2}))\b",
    re.IGNORECASE,
)

# Month + day with optional weekday (incl. abbreviations) and optional 4-digit year.
_WDAY_HEAD = r"(?:Mon(?:day)?|Tue(?:sday)?|Wed(?:nesday)?|Thu(?:rsday)?|Fri(?:day)?|Sat(?:urday)?|Sun(?:day)?)"
_SESSION_HEAD_RELAXED_RE = re.compile(
    rf"\b((?:(?:{_WDAY_HEAD})\s*,?\s+)?"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|"
    r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+"
    r"(\d{1,2})(?:\s*,\s*((?:19|20)\d{2}))?)\b",
    re.IGNORECASE,
)

# Month-first dated headings (Grok bold lines like ``**May 13 2026 (earnings day, BMO …)**``).
_SESSION_HEAD_MONTH_FIRST_RE = re.compile(
    r"(?i)^(?:[\-\*]+\s*)?(?:#{1,6}\s+)?(?:\*\*)?\s*"
    r"(January|February|March|April|May|June|July|August|September|October|November|December|"
    r"Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+"
    r"(\d{1,2})(?:\s*,?\s*((?:19|20)\d{2}))?",
)

_THREE_SIG_PCT_RE = re.compile(
    rf"3\s*{_GS}\s*:\s*[^\n(]*\(\s*±\s*(?P<pct>[\d.]+)\s*%\s*\)",
    re.IGNORECASE,
)

_ONE_SIG_PCT_RE = re.compile(
    rf"1\s*{_GS}\s*:\s*[^\n(]*\(\s*±\s*(?P<pct>[\d.]+)\s*%\s*\)",
    re.IGNORECASE,
)


def _parse_month_day_label(label: str, *, year: int) -> date | None:
    m = re.search(
        r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|"
        r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+(\d{1,2})"
        r"(?:\s*,?\s*((?:19|20)\d{2}))?\b",
        label,
        re.IGNORECASE,
    )
    if not m:
        return None
    token = m.group(1)
    day = int(m.group(2))
    eff_year = int(m.group(3)) if m.group(3) else year
    for fmt in ("%b %d %Y", "%B %d %Y"):
        try:
            return datetime.strptime(f"{token} {day} {eff_year}", fmt).date()
        except ValueError:
            continue
    return None


def _normalize_sigma_session_tag(tag: str) -> str:
    t = re.sub(r"\s+", " ", tag.strip().lower())
    return t[:220]


def _dedupe_duplicate_date_session_blocks(
    rows: list[tuple[date, float, float, str]],
) -> list[tuple[date, float, float, str]]:
    """Drop only duplicate ``(calendar date, session header)`` blocks; keep distinct sessions."""
    seen: set[tuple[date, str]] = set()
    out: list[tuple[date, float, float, str]] = []
    for d, w3, s1, lbl in rows:
        key = (d, _normalize_sigma_session_tag(lbl))
        if key in seen:
            continue
        seen.add(key)
        out.append((d, w3, s1, lbl))
    return out


def _line_opens_sigma_horizon_block(line: str, *, year: int) -> bool:
    """True when ``line`` starts a dated sigma-horizon block (distinct from incidental calendar mentions)."""
    s = line.strip()
    if not s or len(s) > 360:
        return False
    if s.startswith("|"):
        return False
    if _parse_month_day_label(s, year=year) is None:
        return False
    if re.match(r"^#{1,6}\s+", s):
        return True
    head_window = s[:180]
    if _SESSION_HEAD_RE.search(head_window) is not None:
        return True
    if _SESSION_HEAD_RELAXED_RE.search(head_window) is not None:
        return True
    if (
        re.search(r"(?i)\bT\s*\+\s*\d+\b|\bT0\b", head_window)
        and _SESSION_HEAD_RELAXED_RE.search(s) is not None
    ):
        return True
    if s.startswith("**") and len(s) < 260:
        if _SESSION_HEAD_MONTH_FIRST_RE.match(s) is None:
            return False
        low = s.lower()
        if any(
            k in low
            for k in (
                "earnings",
                "eod",
                "bmo",
                "amc",
                "post-earnings",
                "post print",
                "open",
                "close",
                "horizon",
                "t+",
                "t0",
                "trading day",
                "next trading",
                "one trading week",
            )
        ):
            return True
        if f"3{_GS}" in s or f"1{_GS}" in s:
            return True
    return False


def _bind_sigma_widths_in_block(block_text: str) -> tuple[float, float] | None:
    """Pick paired 3-sigma / 1-sigma half-width percents inside one horizon block.

    Returns ``(three_sigma_half_width_pct, one_sigma_half_width_pct)``, preferring the **last**
    3-sigma row in the block paired with the closest 1-sigma line (handles out-of-order lines).
    """
    lines = block_text.splitlines()
    ones: list[tuple[int, float]] = []
    threes: list[tuple[int, float]] = []
    for i, line in enumerate(lines):
        m1 = _ONE_SIG_PCT_RE.search(line)
        if m1:
            with contextlib.suppress(ValueError):
                ones.append((i, float(m1.group("pct"))))
        m3 = _THREE_SIG_PCT_RE.search(line)
        if m3:
            with contextlib.suppress(ValueError):
                threes.append((i, float(m3.group("pct"))))
    if not threes and not ones:
        return None
    if threes and not ones:
        w3 = threes[-1][1]
        return (w3, w3 / 3.0)
    if ones and not threes:
        sig = ones[-1][1]
        return (3.0 * sig, sig)
    ti, tw = threes[-1]
    _, ow = min(ones, key=lambda x: abs(x[0] - ti))
    return (tw, ow)


def _iter_sigma_horizon_blocks(synthesis: str, *, year: int) -> list[tuple[str, str]]:
    """Split ``synthesis`` into ``(header_line, body_text)`` horizons in document order."""
    lines = synthesis.splitlines()
    out: list[tuple[str, str]] = []
    header_for_next: str | None = None
    buf: list[str] = []
    for line in lines:
        if _line_opens_sigma_horizon_block(line, year=year):
            if header_for_next is None and buf:
                out.append(("", "\n".join(buf)))
                buf = []
            elif header_for_next is not None:
                out.append((header_for_next, "\n".join(buf)))
                buf = []
            header_for_next = line.strip()
        else:
            buf.append(line)
    if header_for_next is not None:
        out.append((header_for_next, "\n".join(buf)))
    elif buf:
        out.append(("", "\n".join(buf)))
    return out


def extract_dated_three_sigma_half_widths(
    synthesis: str,
    *,
    year: int = 2026,
) -> list[tuple[date, float, float, str]]:
    """Pair dated 3-sigma (plus-minus pct) lines with session headers.

    Splits the text into **horizon blocks** (nearest preceding dated header such as
    ``### T0 — Wed May 13, 2026 (BMO)`` or ``**May 13 2026 (earnings day, BMO — open & close)**``),
    then binds 1-sigma / 3-sigma lines **within** each block so rows from different sessions on the
    same calendar day do not cross-contaminate.

    Returns ``(session_date, three_sigma_half_width_pct, one_sigma_half_width_pct, label)`` in block
    order. Adjacent duplicate ``(date, header)`` blocks are dropped. Downstream
    :func:`_merge_dated_sigma_rows_first_wins` keeps the first block per **calendar date** for the
    variance-additive identity (canonical row per date).
    """
    rows: list[tuple[date, float, float, str]] = []
    for header, body in _iter_sigma_horizon_blocks(synthesis, year=year):
        block_text = f"{header}\n{body}" if body else header
        bind = _bind_sigma_widths_in_block(block_text)
        if bind is None:
            continue
        w3_half, s1_half = bind
        block_date = _parse_month_day_label(header, year=year)
        if block_date is None:
            block_date = _parse_month_day_label(block_text[:1200], year=year)
        if block_date is None:
            continue
        lbl = header.strip() if header else "(preamble)"
        if not lbl:
            lbl = "(preamble)"
        rows.append((block_date, w3_half, s1_half, lbl))
    return _dedupe_duplicate_date_session_blocks(rows)


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
    d_late = session_late.strip()[:24] if session_late else "T+N"
    d_early = session_early.strip()[:24] if session_early else "T+1"
    pct_off = rel_err * 100.0
    return [
        f"{_GS} pre-event sqrt-t: 3{_GS}({d_late})/3{_GS}({d_early})={observed:.2f} "
        f"expected ~sqrt(N)={expected:.2f} (N={trading_day_span}; pre-event constant-IV scaling), "
        f"off by {pct_off:.0f}%.",
    ]


def verify_variance_additive_sigma_band_sessions(
    sessions: list[Any],
    daily_vol_pct: float,
    event_jump_pct: float,
    tolerance: float = 0.25,
) -> list[str]:
    """Return follow-up strings when 1-sigma percent half-widths violate variance-additive post-event math.

    Each session entry uses ``session`` (label), ``N`` (weekday **diffusion index**: count of
    regular sessions **strictly after** the earnings **calendar** date through the row's session
    date, inclusive of the row — same rule as :func:`trading_sessions_after_exclusive`), and
    ``sigma_pct`` (1-sigma plus-minus half-width as a percent).

    ``event_jump`` follows the equity / synthesizer prompts: **ATM straddle implied move (%)** in
    the **same half-width percent units** as ``sigma_pct`` (if a model quotes a full width, it must
    halve before emitting the literal).

    Identity checked: ``sigma_pct**2 ≈ event_jump**2 + N * daily_vol**2`` (relative tolerance per row).
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
        if n_int < 0 or sig_f <= 0.0:
            continue
        rows.append((sess, n_int, sig_f))

    if not rows:
        return []

    ej2 = float(event_jump_pct) ** 2
    dv2 = float(daily_vol_pct) ** 2

    followups: list[str] = []
    for sess, n_i, sig in rows:
        expected_sq = ej2 + float(n_i) * dv2
        if expected_sq <= 0.0:
            continue
        obs_sq = sig * sig
        rel = abs(obs_sq - expected_sq) / expected_sq
        if rel > tolerance:
            followups.append(
                f"{_GS}^2 variance {sess[:18]}: {_GS}^2(n={n_i})={obs_sq:.2f} expected "
                f"ej^2+n·daily_vol^2={expected_sq:.2f}, off by {rel:.0%}; fix daily_vol or inputs.",
            )
    return followups


def verify_iv_crush_daily_vol_followups(
    synthesis_text: str,
    *,
    iv_crush_multiplier: float | None,
    hv30_annualized_pct: float | None,
    symbol: str = "",
    tolerance: float = 0.10,
) -> list[str]:
    """When chain + history supply an IV crush multiplier, flag ``daily_vol=`` that ignores it."""
    if iv_crush_multiplier is None or hv30_annualized_pct is None:
        return []
    dv_m = _DAILY_VOL_PCT_RE.search(synthesis_text)
    if dv_m is None:
        return []
    try:
        claimed = float(dv_m.group(1))
    except ValueError:
        return []
    expected = (float(hv30_annualized_pct) / math.sqrt(252.0)) * float(iv_crush_multiplier)
    if expected <= 0.0:
        return []
    rel = abs(claimed - expected) / expected
    if rel <= tolerance:
        return []
    sym = symbol.strip().upper() or "TICKER"
    return [
        f"Cite or verify: {sym} daily_vol claimed={claimed:.2f}%/day; expected={expected:.2f}%/day "
        f"after IV crush (multiplier={iv_crush_multiplier:.2f} from chain).",
    ]


def _parse_variance_additive_literals(synthesis: str) -> tuple[float | None, float | None]:
    ej_m = _EVENT_JUMP_PCT_RE.search(synthesis)
    dv_m = _DAILY_VOL_PCT_RE.search(synthesis)
    if not ej_m or not dv_m:
        return None, None
    try:
        return float(ej_m.group(1)), float(dv_m.group(1))
    except ValueError:
        return None, None


def should_apply_sqrt_t_three_sigma_ratio(
    *,
    variance_literals_present: bool,
    early_d: date,
    late_d: date,
    earnings_date: date | None,
) -> bool:
    """True only when constant-IV sqrt(t) scaling is appropriate for earliest vs latest 3-sigma % rows."""
    if variance_literals_present:
        return False
    if late_d <= early_d:
        return False
    if earnings_date is None:
        return True
    if early_d < earnings_date <= late_d:
        return False
    entirely_pre = late_d < earnings_date
    entirely_post = early_d >= earnings_date
    return entirely_pre or entirely_post


_PCR_NUMERIC_IN_PARAGRAPH_RE = re.compile(
    r"(?is)\b(?:PCR|put\s*/\s*call\s+ratio|P/?C\s+ratio)\b.{0,240}?\b(\d+\.\d+)\b",
)


def followups_from_unsourced_options_chain_numeric_claims(
    synthesis_text: str,
    *,
    context_label: str = "synthesis",
) -> list[str]:
    """Heuristic: PCR / put-call ratio decimals without URL or ``Source:`` in the same paragraph."""
    followups: list[str] = []
    for block in re.split(r"\n\s*\n+", synthesis_text.strip()):
        b = block.strip()
        if not b or len(b) > 800:
            continue
        m = _PCR_NUMERIC_IN_PARAGRAPH_RE.search(b)
        if not m:
            continue
        low = b.lower()
        if "http://" in low or "https://" in low:
            continue
        if "source:" in low:
            continue
        val = m.group(1)
        followups.append(
            f"Cite or verify: {context_label} claims PCR={val} without a primary source.",
        )
    out: list[str] = []
    seen: set[str] = set()
    for f in followups:
        if f in seen:
            continue
        seen.add(f)
        out.append(f)
    return out


def sigma_sessions_payload_from_sigma_summary_model(
    parsed: SigmaSummaryFileModel,
    *,
    earn_cal: date | None,
    earnings_timing: str | None,
) -> tuple[list[dict[str, Any]], int, str | None]:
    """Build ``verify_variance_additive_sigma_band_sessions`` rows from structured ``sigma_summary`` JSON."""
    by_date: dict[date, tuple[float, str]] = {}
    for row in parsed.sigma_summary.sessions:
        d = row.date
        if d in by_date:
            continue
        by_date[d] = (float(row.one_sigma_half_width_pct), row.label.strip())

    if not by_date:
        return [], 0, "missing_sigma_summary_sessions"

    by_rows = _dated_rows_for_variance_additive_check(
        by_date, earn_cal if earn_cal is not None else None
    )

    if earn_cal is not None and not by_rows and by_date:
        return [], 0, "missing_on_or_after_earnings_date_rows"

    if not by_rows:
        return [], 0, "missing_dated_rows"

    keys_sorted = sorted(by_rows)
    baseline = earn_cal if earn_cal is not None else keys_sorted[0]
    sessions_payload: list[dict[str, Any]] = []
    for d in keys_sorted:
        sig_half, lbl = by_rows[d]
        n_inc = trading_sessions_after_exclusive(baseline, d)
        sessions_payload.append(
            {
                "session": lbl,
                "session_date": d.isoformat(),
                "N": n_inc,
                "sigma_pct": sig_half,
            },
        )

    return sessions_payload, len(sessions_payload), None


def prob_up_followups_for_parsed_sigma_summary(
    parsed: SigmaSummaryFileModel,
    *,
    earn_cal: date | None,
    earnings_timing: str | None,
    context_label: str,
) -> tuple[list[str], str | None]:
    """Compare emitted ``prob_up_pct`` vs bounded-drift Phi(mu N / sigma); returns follow-ups and optional clamp note."""
    pl = parsed.sigma_summary
    if not any(s.prob_up_pct is not None for s in pl.sessions):
        return [], None
    if pl.drift_source is None or pl.daily_drift_pct is None:
        return [], None
    mu_raw = float(pl.daily_drift_pct)
    mu_bounded, drift_warn = bound_daily_drift(mu_raw, pl.drift_source)
    sessions_payload, _, err = sigma_sessions_payload_from_sigma_summary_model(
        parsed,
        earn_cal=earn_cal,
        earnings_timing=earnings_timing,
    )
    if err is not None:
        return (
            [
                "Cite or verify: "
                f"{context_label} sigma_summary includes prob_up_pct but session alignment failed ({err}).",
            ],
            drift_warn,
        )
    emitted_by_date: dict[date, float] = {}
    for s in pl.sessions:
        if s.prob_up_pct is None:
            continue
        if s.date not in emitted_by_date:
            emitted_by_date[s.date] = float(s.prob_up_pct)

    followups: list[str] = []
    for row in sessions_payload:
        raw_sd = row.get("session_date")
        if not isinstance(raw_sd, str):
            continue
        try:
            d_key = date.fromisoformat(raw_sd)
        except ValueError:
            continue
        emitted = emitted_by_date.get(d_key)
        if emitted is None:
            continue
        n_inc = int(row["N"])
        sig_half = float(row["sigma_pct"])
        expected = computed_prob_up_pct(mu_bounded, sig_half, n_inc)
        if abs(emitted - expected) > PROB_UP_MISMATCH_TOLERANCE_PP:
            followups.append(
                "Cite or verify: "
                f"{context_label} session {raw_sd} prob_up={emitted:.1f}% but computed={expected:.1f}% "
                f"from drift={mu_bounded:+.4f}%/day sigma={sig_half:.2f}% N={n_inc}",
            )
    return followups, drift_warn


def _legacy_sigma_variance_sessions(
    provider_text: str,
    *,
    anchor_year: int,
    earn_cal: date | None,
    earnings_timing: str | None,
) -> tuple[list[dict[str, Any]], int, str | None]:
    """Legacy markdown extraction for per-provider sigma variance rows (regex on 1-sigma/3-sigma lines)."""
    dated = extract_dated_three_sigma_half_widths(provider_text, year=anchor_year)
    if logger.isEnabledFor(logging.DEBUG) and dated:
        preview = ", ".join(
            f"({d.isoformat()}, {lbl[:44]!r}, 1{_GS}={s1:.2f}%, 3{_GS}={w3:.2f}%)"
            for d, w3, s1, lbl in dated[:24]
        )
        tail = " ..." if len(dated) > 24 else ""
        logger.debug("parsed %d sigma blocks: [%s]%s", len(dated), preview, tail)
    _by_w3, by_s1 = _merge_dated_sigma_rows_first_wins(dated)

    if not by_s1:
        return [], 0, "missing_dated_rows"

    by_rows = _dated_rows_for_variance_additive_check(
        by_s1, earn_cal if earn_cal is not None else None
    )
    if earn_cal is not None and not by_rows and by_s1:
        return [], 0, "missing_on_or_after_earnings_date_rows"

    if not by_rows:
        return [], 0, "missing_dated_rows"

    keys_sorted = sorted(by_rows)
    baseline = earn_cal if earn_cal is not None else keys_sorted[0]
    sessions_payload: list[dict[str, Any]] = []
    for d in keys_sorted:
        sig_half, lbl = by_rows[d]
        n_inc = trading_sessions_after_exclusive(baseline, d)
        sessions_payload.append({"session": lbl, "N": n_inc, "sigma_pct": sig_half})

    return sessions_payload, len(sessions_payload), None


def per_provider_sigma_variance_check(
    provider_text: str,
    tolerance: float = 0.25,
    *,
    anchor_year: int = 2026,
    earnings_date: str | None = None,
    earnings_timing: str | None = None,
    enabled: bool = True,
    provider_label: str | None = None,
    reference_event_jump_pct: float | None = None,
) -> dict[str, Any]:
    """Parse one provider's text for variance-additive sigma literals and verify the identity.

    Returns a dict with the structured result fields used by the synthesizer wiring:

    - ``passed`` (``True`` / ``False`` / ``None``): ``True`` when the variance identity holds within
      ``tolerance``. ``False`` on a concrete math mismatch. ``None`` when literals exist but dated
      rows cannot be verified (missing rows / wrong session window), see ``reason``.
    - ``followups`` (list[str]): per-row follow-up strings — empty when ``passed`` is not ``False``.
    - ``event_jump`` / ``daily_vol``: parsed literals when present.
    - ``sessions`` (int): dated rows used in the check.
    - ``applicable`` (bool): ``False`` only when literals are missing (or checks disabled).
    - ``reason`` (str): machine tag or first follow-up line.
    - ``sigma_check_source`` (``\"sigma_summary_json\"`` / ``\"legacy_regex\"`` / ``None``): which
      input path supplied session half-widths for the variance identity.
    """
    if not enabled:
        logger.info("sigma_variance_check disabled per config")
        return {
            "passed": None,
            "followups": [],
            "event_jump": None,
            "daily_vol": None,
            "sessions": 0,
            "applicable": False,
            "reason": "disabled_per_config",
            "sigma_check_source": None,
        }

    ej_lit, dv_lit = _parse_variance_additive_literals(provider_text)
    if ej_lit is None or dv_lit is None:
        return {
            "passed": None,
            "followups": [],
            "event_jump": ej_lit,
            "daily_vol": dv_lit,
            "sessions": 0,
            "applicable": False,
            "reason": "missing_literals",
            "sigma_check_source": None,
        }

    earn_cal: date | None = None
    ed_raw = (earnings_date or "").strip()
    if ed_raw:
        earn_cal = _parse_earnings_calendar_date(ed_raw)

    parsed = parse_sigma_summary_json(provider_text)
    if parsed is not None:
        sessions_payload, n_sessions, err = sigma_sessions_payload_from_sigma_summary_model(
            parsed,
            earn_cal=earn_cal,
            earnings_timing=earnings_timing,
        )
        if err is not None:
            return {
                "passed": None,
                "followups": [],
                "event_jump": ej_lit,
                "daily_vol": dv_lit,
                "sessions": 0,
                "applicable": True,
                "reason": err,
                "sigma_check_source": "sigma_summary_json",
            }
        source = "sigma_summary_json"
    else:
        sessions_payload, n_sessions, err = _legacy_sigma_variance_sessions(
            provider_text,
            anchor_year=anchor_year,
            earn_cal=earn_cal,
            earnings_timing=earnings_timing,
        )
        if err is not None:
            reason_out = err
            if err == "missing_dated_rows" and sigma_summary_json_present_but_invalid(
                provider_text
            ):
                reason_out = "missing_sigma_summary_json"
            return {
                "passed": None,
                "followups": [],
                "event_jump": ej_lit,
                "daily_vol": dv_lit,
                "sessions": 0,
                "applicable": True,
                "reason": reason_out,
                "sigma_check_source": "legacy_regex",
            }
        source = "legacy_regex"

    followups = list(
        verify_variance_additive_sigma_band_sessions(
            sessions_payload,
            daily_vol_pct=dv_lit,
            event_jump_pct=ej_lit,
            tolerance=tolerance,
        ),
    )
    ref_ej = reference_event_jump_pct
    if ref_ej is not None and ref_ej > 5.0 and ej_lit is not None and ej_lit < 1.0:
        followups.insert(
            0,
            "Cite or verify: parsed event_jump="
            f"{ej_lit:.2f}% appears to be decimal form; expected percent form ~{ref_ej:.2f}% "
            "(from front weekly straddle).",
        )
    ctx = (provider_label or "Provider").strip() or "Provider"
    drift_warn: str | None = None
    if parsed is not None:
        prob_fus, drift_warn = prob_up_followups_for_parsed_sigma_summary(
            parsed,
            earn_cal=earn_cal,
            earnings_timing=earnings_timing,
            context_label=ctx,
        )
        followups = list(followups) + list(prob_fus)

    passed = not followups
    reason = "" if passed else (followups[0] if followups else "variance identity drift")
    out: dict[str, Any] = {
        "passed": passed,
        "followups": followups,
        "event_jump": ej_lit,
        "daily_vol": dv_lit,
        "sessions": n_sessions,
        "applicable": True,
        "reason": reason,
        "sigma_check_source": source,
    }
    if drift_warn:
        out["drift_clamp_warning"] = drift_warn
    return out


def compute_severity_for_sigma_variance_results(
    results: list[dict[str, Any]], *, quorum_for_error: int
) -> list[dict[str, Any]]:
    """Assign per-row ``severity`` after a full fan-out round (mutates ``results`` in place).

    - ``info`` — applicable and ``passed`` is ``True``
    - ``warning`` — applicable ``passed=False`` but fewer than ``quorum_for_error`` providers failed
      the variance identity in this round; **or** ``missing_literals`` with fewer than quorum omitters
    - ``error`` — applicable ``passed=False`` and the count of such failures in this round is
      **≥ quorum_for_error**; **or** ``missing_literals`` with **≥ quorum_for_error** omitters
    - ``na`` — ``passed`` is ``None``, disabled / not applicable (except missing-literals cohort above),
      or other non-matching rows
    """
    q = max(1, min(10, int(quorum_for_error)))
    for r in results:
        r.pop("isolated", None)
        r.pop("peers_failed", None)
        r.pop("peers_literals_missing", None)

    n_fail = sum(1 for r in results if r.get("applicable") is True and r.get("passed") is False)
    n_miss = sum(
        1
        for r in results
        if r.get("applicable") is not True
        and str(r.get("reason", "")).strip() == "missing_literals"
    )

    for r in results:
        if r.get("applicable") is not True:
            reason = str(r.get("reason", "")).strip()
            if reason == "missing_literals":
                if n_miss >= q:
                    r["severity"] = "error"
                    r["peers_literals_missing"] = n_miss
                elif n_miss > 0:
                    r["severity"] = "warning"
                    if n_miss == 1:
                        r["isolated"] = True
                else:
                    r["severity"] = "na"
            else:
                r["severity"] = "na"
            continue

        passed = r.get("passed")
        if passed is True:
            r["severity"] = "info"
        elif passed is None:
            r["severity"] = "na"
        else:
            if n_fail >= q:
                r["severity"] = "error"
                r["peers_failed"] = n_fail
            else:
                r["severity"] = "warning"
                if n_fail == 1:
                    r["isolated"] = True
    return results


def _per_provider_sigma_checks_with_severity(
    checks: Sequence[dict[str, Any]], *, quorum_for_error: int
) -> list[dict[str, Any]]:
    """Return a list copy with ``severity`` populated (recomputes when missing — e.g. legacy state)."""
    lst = [dict(c) for c in checks]
    if lst and not any("severity" in x for x in lst):
        compute_severity_for_sigma_variance_results(lst, quorum_for_error=quorum_for_error)
    return lst


def sigma_missing_literal_router_followups(
    checks: Sequence[dict[str, Any]], *, quorum_for_error: int = 2
) -> list[str]:
    """When **≥ quorum** providers omit parseable ``event_jump=`` / ``daily_vol=``, nudge the next fan-out."""
    lst = _per_provider_sigma_checks_with_severity(checks, quorum_for_error=quorum_for_error)
    missing: list[str] = []
    for c in lst:
        if str(c.get("reason", "")).strip() != "missing_literals":
            continue
        if c.get("severity") != "error":
            continue
        prov = str(c.get("provider", "")).strip() or "unknown"
        missing.append(
            "Cite or verify: Provider "
            f"{prov} did not include the mandatory event_jump= / daily_vol= literals; "
            "please include them in the exact format specified.",
        )
    return missing


def sigma_variance_mismatch_router_followups(
    checks: Sequence[dict[str, Any]], *, quorum_for_error: int = 2
) -> list[str]:
    """Fan-out router prompts when **≥ quorum** providers fail the applicable variance identity."""
    lst = _per_provider_sigma_checks_with_severity(checks, quorum_for_error=quorum_for_error)
    out: list[str] = []
    for c in lst:
        if c.get("severity") != "error":
            continue
        if c.get("applicable") is not True or c.get("passed") is not False:
            continue
        prov = str(c.get("provider", "")).strip() or "unknown"
        fus = [str(x).strip() for x in (c.get("followups") or []) if str(x).strip()]
        if fus:
            out.append(f"Cite or verify: Provider {prov} sigma variance check: {fus[0]}")
        else:
            out.append(
                f"Cite or verify: Provider {prov} failed the sigma variance-additive identity; "
                "reconcile dated 1-sigma rows vs event_jump= / daily_vol= or label an explicit regime change.",
            )
    return out


def render_per_provider_sigma_checks_markdown(checks: list[dict[str, Any]]) -> str:
    """Render the per-provider sigma-variance summary as a short markdown table for the synthesizer.

    ``checks`` items are the records stashed in ``state["per_provider_sigma_checks"]`` (provider,
    model, plus the fields from :func:`per_provider_sigma_variance_check`). When the input list
    is empty, returns an empty string so the synthesizer prompt can omit the section entirely.
    """
    if not checks:
        return ""
    lines = [
        "| Provider | Model | event_jump | daily_vol | sessions | passed | severity | reason |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in checks:
        prov = str(c.get("provider", "")).strip() or "?"
        model = str(c.get("model", "")).strip() or "?"
        ej_v = c.get("event_jump")
        dv_v = c.get("daily_vol")
        ej_s = f"{float(ej_v):.2f}%" if isinstance(ej_v, (int, float)) else "n/a"
        dv_s = f"{float(dv_v):.2f}%" if isinstance(dv_v, (int, float)) else "n/a"
        sess = int(c.get("sessions") or 0)
        if not c.get("applicable", False) or c.get("passed") is None:
            passed_s = "n/a"
        elif c.get("passed") is True:
            passed_s = "True"
        else:
            passed_s = "False"
        sev = str(c.get("severity") or "").strip() or "n/a"
        reason = str(c.get("reason") or "").strip().replace("|", "/")
        if len(reason) > 80:
            reason = reason[:77] + "..."
        lines.append(
            f"| {prov} | {model} | {ej_s} | {dv_s} | {sess} | {passed_s} | {sev} | {reason} |",
        )
    return "\n".join(lines)


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


def _latest_provider_iteration_bundle(state: Any) -> str:
    """Concatenate the most recent fan-out round's provider bodies for deterministic advisory checks."""
    pr = state.get("provider_responses") or []
    if not pr:
        return ""
    raw = pr[-1].get("responses") or {}
    if not isinstance(raw, dict):
        return ""
    parts: list[str] = []
    for name, v in raw.items():
        if not isinstance(v, dict):
            continue
        txt = str(v.get("text") or "").strip()
        if txt:
            parts.append(f"## {name}\n{txt}")
    return "\n\n".join(parts)


def augment_verifier_result_with_sigma_structural_checks(
    synthesis_text: str,
    result: dict[str, Any],
    *,
    anchor_year: int = 2026,
    sqrt_tolerance: float = 0.25,
    options_chain_data: dict[str, Any] | None = None,
    symbol: str = "",
    iv_crush_multiplier: float | None = None,
    hv30_annualized_pct: float | None = None,
    earnings_date: date | None = None,
    earnings_timing: str | None = None,
    computed_sigma_bands_table: dict[str, Any] | None = None,
    t0_blend_preset: T0BlendPreset = "default",
    provider_iteration_bundle: str | None = None,
) -> dict[str, Any]:
    """Append deterministic sigma-band structural items to ``unverifiable`` (router follow-ups)."""
    _ = earnings_timing  # reserved; structural sigma checks use earnings calendar date only.
    out = dict(result)
    prior = [str(x).strip() for x in (out.get("unverifiable") or []) if str(x).strip()]
    extras: list[str] = []
    tbl_sigma = computed_sigma_bands_table or {}

    sigma_lines = [
        ln for ln in synthesis_text.splitlines() if f"3{_GS}" in ln and "±" in ln and "%" in ln
    ]
    ej_lit, dv_lit = _parse_variance_additive_literals(synthesis_text)
    variance_mode = ej_lit is not None and dv_lit is not None

    if sigma_lines:
        if variance_mode:
            if f"{_GS}-scaling check (variance):" not in synthesis_text:
                extras.append(
                    f"{_GS} bands: add `{_GS}-scaling check (variance):` line "
                    f"(each row: {_GS}^2 ≈ ej^2 + n·daily_vol^2; deltas vs (n2-n1)·daily_vol^2).",
                )
        elif f"{_GS}-scaling check" not in synthesis_text:
            extras.append(
                f"{_GS} bands: add mandatory `{_GS}-scaling check:` line vs sqrt(N) or annotate regimes.",
            )

    dated = extract_dated_three_sigma_half_widths(synthesis_text, year=anchor_year)
    by_w3, by_s1 = _merge_dated_sigma_rows_first_wins(dated)

    by_date_var = _dated_rows_for_variance_additive_check(
        by_s1,
        earnings_date if earnings_date is not None else None,
    )

    strict_tbl = (
        isinstance(computed_sigma_bands_table, dict)
        and isinstance(computed_sigma_bands_table.get("sessions"), list)
        and len(computed_sigma_bands_table["sessions"]) > 0
    )
    if strict_tbl:
        from equity_analyst.sigma_compute import verify_emitted_sigma_bands_match_computed

        extras.extend(
            verify_emitted_sigma_bands_match_computed(
                synthesis_text,
                computed_sigma_bands_table,
                tolerance_pp=1.0,
            ),
        )
    elif variance_mode and len(by_date_var) >= 1:
        assert ej_lit is not None and dv_lit is not None
        keys_sorted = sorted(by_date_var)
        baseline = earnings_date if earnings_date is not None else keys_sorted[0]
        session_payload: list[dict[str, Any]] = []
        for d in keys_sorted:
            sig_half, lbl = by_date_var[d]
            n_inc = trading_sessions_after_exclusive(baseline, d)
            session_payload.append({"session": lbl, "N": n_inc, "sigma_pct": sig_half})
        extras.extend(
            verify_variance_additive_sigma_band_sessions(
                session_payload,
                daily_vol_pct=dv_lit,
                event_jump_pct=ej_lit,
                tolerance=sqrt_tolerance,
            ),
        )
    elif len(by_w3) >= 2:
        keys = sorted(by_w3)
        early_d, late_d = keys[0], keys[-1]
        if should_apply_sqrt_t_three_sigma_ratio(
            variance_literals_present=variance_mode,
            early_d=early_d,
            late_d=late_d,
            earnings_date=earnings_date,
        ):
            span = trading_sessions_after_exclusive(early_d, late_d)
            w_early, lbl_e = by_w3[early_d]
            w_late, lbl_l = by_w3[late_d]
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
            extras.append(
                f"{_GS} baseline missing for {sess}; name YYYY-MM-DD expiry or HV30 sqrt(t)."
            )
        passed = row.get("sigma_scaling_check_passed")
        if passed is False:
            extras.append(
                f"{_GS} scaling failed for {sess}; re-derive or document vol regime split."
            )

    extras.extend(
        options_chain_expiry_audit_messages(
            synthesis_text,
            out,
            options_chain_data=options_chain_data,
            symbol=symbol,
        ),
    )

    extras.extend(
        verify_iv_crush_daily_vol_followups(
            synthesis_text,
            iv_crush_multiplier=iv_crush_multiplier,
            hv30_annualized_pct=hv30_annualized_pct,
            symbol=symbol,
        ),
    )

    if (
        str(tbl_sigma.get("expiry_class") or "") == "monthly"
        and "monthly-expiry sourced" not in synthesis_text.lower()
    ):
        extras.append(
            "MUST (warning): when server `expiry_class` is `monthly`, synthesis must include the verbatim "
            'label **"Monthly-expiry sourced"** (case-insensitive match) describing the sigma ladder sourcing.',
        )

    blend_followups = horizon_blend_ratio_followups(
        synthesis_text,
        t0_blend_preset=t0_blend_preset,
    )
    blend_soft = suggested_blend_consistency_followups(
        synthesis_text,
        provider_iteration_bundle=provider_iteration_bundle or "",
    )

    syn_parsed = parse_sigma_summary_json(synthesis_text)
    if syn_parsed is not None:
        prob_extras, drift_note = prob_up_followups_for_parsed_sigma_summary(
            syn_parsed,
            earn_cal=earnings_date,
            earnings_timing=earnings_timing,
            context_label="synthesis",
        )
        extras.extend(prob_extras)
        if drift_note:
            extras.append(f"Note: {drift_note}")

    extras.extend(followups_from_unsourced_options_chain_numeric_claims(synthesis_text))
    merged = (
        blend_followups
        + extras
        + [
            u
            for u in prior
            if u not in blend_followups and u not in extras
        ]
        + blend_soft
    )
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
    pointer = f"…(abridged; full text in `iterations/iteration_{iteration_index}_synthesis.md`)"
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
        result["sigma_scaling_aggregate_passed"] = bool(
            best_data.get("sigma_scaling_aggregate_passed")
        )
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
            max_tokens_hit or _finish_reason_implies_provider_truncation(provider_finish_reason),
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
    per_provider_sigma_checks: NotRequired[list[dict[str, Any]]]
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
    earnings_date: NotRequired[str]
    earnings_timing: NotRequired[str | None]
    computed_sigma_bands_table: NotRequired[dict[str, Any] | None]
    t0_blend_preset: NotRequired[str]
    persist_run_json_to_disk: NotRequired[bool]
    run_meta_seed: NotRequired[dict[str, Any]]
    final_run_meta: NotRequired[dict[str, Any]]


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
    sigma_quorum = int(snap.get("sigma_variance_check_quorum_for_error", 2))
    checks_raw = state.get("per_provider_sigma_checks") or []
    sigma_router_qs = [
        *sigma_missing_literal_router_followups(checks_raw, quorum_for_error=sigma_quorum),
        *sigma_variance_mismatch_router_followups(checks_raw, quorum_for_error=sigma_quorum),
    ]

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

    if (
        conf is not None
        and conf >= threshold
        and n_contrad == 0
        and n_unver == 0
        and not sigma_router_qs
    ):
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
    for sq in sigma_router_qs:
        if sq not in qs:
            qs.append(sq)

    missing_synth_sigma_json = any(
        "missing valid sigma_summary json" in str(u).lower() for u in (unver or [])
    )
    if n_contrad == 0 and missing_synth_sigma_json:
        _gs = chr(0x03C3)
        pr = (
            f"PRIORITY — Before any {_gs} bands in sections 1/9/11, emit the mandatory fenced strict-JSON block "
            "(language tag json) whose root contains sigma_summary; copy % half-widths from "
            f"Server-computed {_gs} bands verbatim when that section is present."
        )
        qs_syn = [pr]
        for x in qs:
            if x not in qs_syn:
                qs_syn.append(x)
        logger.info(
            "Route decision: continue (synthesize_only) reason=missing_synthesis_sigma_summary_json",
        )
        return Command(
            goto="synthesize",
            update={
                "followup_questions": qs_syn,
                "last_route_followup_questions": qs_syn,
            },
        )

    high_unver_fanout = (
        n_contrad == 0 and n_unver >= unver_thr and conf is not None and conf < conf_cut
    )
    cite_only_mode = (
        n_contrad == 0 and n_unver > 0 and skip_unver and not force_fan and not high_unver_fanout
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
        parts.append(
            "## Latest verification JSON\n\n```json\n" + json.dumps(vh[-1], indent=2) + "\n```"
        )
    fu = state.get("followup_questions") or []
    if fu:
        parts.append("## Router follow-up targets\n\n" + "\n".join(f"- {x}" for x in fu))
    parts.append(
        "Revise the full 12-section synthesis to resolve verification issues. "
        "When frozen market facts are present, prefer them over stale narrative numbers. "
        "End with the required OVERALL_CONFIDENCE line."
    )
    return "\n\n".join(parts)


def build_per_provider_sigma_checks_markdown(state: RefinementState) -> str:
    """Render the synthesizer-facing ``per_provider_sigma_checks_markdown`` block.

    Returns an empty string when the fan-out node has not yet stashed any checks (e.g. dry
    runs or older checkpoints replayed without the field).
    """
    checks = state.get("per_provider_sigma_checks") or []
    return render_per_provider_sigma_checks_markdown(list(checks))


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
        "- MUST NOT alter **horizon blend** digit pairs, fenced blend-table rows, or **qual-then-quant** lens "
        "wording when editing for clarity — copy them verbatim from the prior synthesis or the equity prompt "
        "fence; never digit-invert, never swap lens labels, never substitute **quant-then-qual** phrasing.\n"
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
    from equity_analyst.sigma_compute import format_computed_probabilities_reference_markdown

    ed_ref = (state.get("earnings_date") or "").strip()
    if not ed_ref:
        eq_ctx_ref = state.get("equity_prompt_render_context")
        if isinstance(eq_ctx_ref, dict):
            ed_raw_ref = eq_ctx_ref.get("earnings_date")
            if isinstance(ed_raw_ref, str) and ed_raw_ref.strip():
                ed_ref = ed_raw_ref.strip()
    et_ref = (state.get("earnings_timing") or "").strip() or None
    if ed_ref:
        prob_ref = format_computed_probabilities_reference_markdown(
            prior_raw,
            earnings_date=ed_ref,
            earnings_timing=et_ref,
        ).strip()
        if prob_ref:
            middle = prob_ref + "\n\n" + middle
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

        if (
            facts_enabled
            and it_no >= 2
            and ver_for_directives.get("refresh_facts")
            and state.get("synthesis_history")
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
        last_route_followups = [
            str(x).strip()
            for x in (state.get("last_route_followup_questions") or [])
            if str(x).strip()
        ]
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
                pcs_raw = [
                    {"name": n, "web_search": None, "request_timeout_s": None}
                    for n in state["providers"]
                ]
            pcs = [ProviderConfig.model_validate(d) for d in pcs_raw]
            cfg_req_timeout = float(state.get("request_timeout_s", 180.0))
            cfg_mot = int(state.get("max_output_tokens", 32_000))
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
                to = (
                    float(pc.request_timeout_s)
                    if pc.request_timeout_s is not None
                    else cfg_req_timeout
                )

                async def _attempt() -> ProviderResponse:
                    mot = fan_out_max_output_tokens(pc, cfg_mot)
                    pce = bool(state.get("prompt_cache_enabled", True))
                    if isinstance(p, AnthropicProvider):
                        static = EQUITY_ANALYST_SYSTEM_PROMPT
                        sep = f"{static}\n\n"
                        user_only = (
                            prompt_body[len(sep) :] if prompt_body.startswith(sep) else prompt_body
                        )
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
                    return await p.generate(
                        prompt_body, enable_web_search=ws, max_output_tokens=mot
                    )

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
        (iter_dir / f"iteration_{round_idx + 1}_providers.md").write_text(
            "\n".join(lines_out), encoding="utf-8"
        )

        ser = {"responses": {k: _response_to_dict(v) for k, v in responses.items()}}

        ed_ctx = state.get("earnings_date")
        earn_date_s: str | None = None
        if isinstance(ed_ctx, str) and ed_ctx.strip():
            earn_date_s = ed_ctx.strip()
        else:
            eq_ctx_fan = state.get("equity_prompt_render_context")
            if isinstance(eq_ctx_fan, dict):
                raw_ed_fan = eq_ctx_fan.get("earnings_date")
                if isinstance(raw_ed_fan, str) and raw_ed_fan.strip():
                    earn_date_s = raw_ed_fan.strip()
        et_ctx = state.get("earnings_timing")
        earn_timing_fan: str | None = None
        if isinstance(et_ctx, str) and et_ctx.strip():
            earn_timing_fan = et_ctx.strip()

        snap_fan = state.get("iterative_config_snapshot") or {}
        sigma_check_enabled = bool(snap_fan.get("per_provider_sigma_variance_check", True))

        ref_ej: float | None = None
        tbl_fan = state.get("computed_sigma_bands_table")
        if isinstance(tbl_fan, dict):
            raw_ej = tbl_fan.get("sigma_event_jump_for_ladder_pct")
            if raw_ej is None:
                raw_ej = tbl_fan.get("event_jump_pct")
            if isinstance(raw_ej, (int, float)) and float(raw_ej) > 0:
                ref_ej = float(raw_ej)
        if ref_ej is None:
            oc_fan = state.get("options_chain_data")
            if (
                isinstance(oc_fan, dict)
                and bool(oc_fan.get("options_chain_available"))
                and earn_date_s
            ):
                mwl = int(oc_fan.get("max_weekly_lookahead_days") or 14)
                ref_ej = event_jump_implied_move_pct_from_prompt_dict(
                    oc_fan,
                    earnings_date=earn_date_s,
                    max_weekly_lookahead_days=mwl,
                )

        per_provider_checks: list[dict[str, Any]] = []
        for name, resp in responses.items():
            check = per_provider_sigma_variance_check(
                resp.text,
                earnings_date=earn_date_s,
                earnings_timing=earn_timing_fan,
                enabled=sigma_check_enabled,
                provider_label=name,
                reference_event_jump_pct=ref_ej,
            )
            rec: dict[str, Any] = {"provider": name, "model": resp.model}
            rec.update(check)
            per_provider_checks.append(rec)

        sigma_quorum = int(snap_fan.get("sigma_variance_check_quorum_for_error", 2))
        compute_severity_for_sigma_variance_results(
            per_provider_checks, quorum_for_error=sigma_quorum
        )

        for rec in per_provider_checks:
            name = str(rec.get("provider", "")).strip()
            applicable = bool(rec.get("applicable"))
            passed = rec.get("passed")
            ej = rec.get("event_jump")
            dv = rec.get("daily_vol")
            sess = int(rec.get("sessions") or 0)
            ej_s = f"{float(ej):.2f}%" if isinstance(ej, (int, float)) else "n/a"
            dv_s = f"{float(dv):.2f}%" if isinstance(dv, (int, float)) else "n/a"
            sev = str(rec.get("severity") or "n/a")
            if not applicable:
                logger.info(
                    "sigma_variance_check provider=%s passed=n/a severity=%s event_jump=%s daily_vol=%s sessions=%s reason=%s",
                    name,
                    sev,
                    ej_s,
                    dv_s,
                    sess,
                    str(rec.get("reason") or "missing_literals"),
                )
            elif passed is True:
                logger.info(
                    "sigma_variance_check provider=%s passed=True severity=info event_jump=%s daily_vol=%s sessions=%s",
                    name,
                    ej_s,
                    dv_s,
                    sess,
                )
            elif passed is None:
                logger.info(
                    "sigma_variance_check provider=%s passed=n/a severity=%s event_jump=%s daily_vol=%s sessions=%s reason=%s",
                    name,
                    sev,
                    ej_s,
                    dv_s,
                    sess,
                    str(rec.get("reason") or "unverifiable"),
                )
            elif sev == "warning":
                logger.info(
                    "sigma_variance_check provider=%s passed=False severity=warning reason=%r isolated=%s",
                    name,
                    str(rec.get("reason") or "variance identity drift"),
                    bool(rec.get("isolated")),
                )
            else:
                logger.info(
                    "sigma_variance_check provider=%s passed=False severity=error reason=%r peers_failed=%s",
                    name,
                    str(rec.get("reason") or "variance identity drift"),
                    int(rec.get("peers_failed") or 0),
                )

        out_fan: dict[str, Any] = {
            "provider_responses": [ser],
            "timing_events": [
                {"iteration": it_no, "providers_parallel_wall_s": parallel_wall},
            ],
            "per_provider_sigma_checks": per_provider_checks,
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
        syn_ws = effective_synthesizer_web_search(
            run_default=state["enable_web_search"], syn=syn_cfg
        )
        summarize_fallback_llm = None
        fb_name = state.get("oversized_summarize_fallback_provider")
        if not fb_name and state.get("iterative_config_snapshot"):
            fb_name = (state["iterative_config_snapshot"] or {}).get(
                "oversized_summarize_fallback_provider"
            )
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
        sigma_checks_md = build_per_provider_sigma_checks_markdown(state)
        eq_ctx_syn = state.get("equity_prompt_render_context")
        csb_md = ""
        if isinstance(eq_ctx_syn, dict):
            raw_csb = eq_ctx_syn.get("computed_sigma_bands_markdown")
            if isinstance(raw_csb, str) and raw_csb.strip():
                csb_md = raw_csb.strip()
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
                        summarize_oversized_providers=bool(
                            state.get("summarize_oversized_providers", True)
                        ),
                        summarize_threshold_input_tokens=int(
                            state.get("summarize_threshold_input_tokens", 8000),
                        ),
                        oversized_summarize_provider=str(
                            state.get("oversized_summarize_provider", "gemini")
                        ),
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
                        per_provider_sigma_checks_markdown=sigma_checks_md,
                        computed_sigma_bands_markdown=csb_md or None,
                        t0_blend_preset=normalize_t0_blend_preset(
                            state.get("t0_blend_preset", "default")
                        ),
                        run_id=out.name,
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
        text = format_synthesis_artifact_markdown(
            synthesis=result,
            responses=resp_map,
            computed_sigma_bands_markdown=csb_md or None,
        )
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
        if (
            bool(state.get("facts_packet_enabled", True))
            and round_idx == 0
            and not result.response.model.startswith(
                "error:",
            )
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
            err_ev.append(
                run_error_record(stage="verify", provider=state["verifier_name"], exc=exc)
            )
            resp = failure_response_from_completed(state["verifier_name"], exc, started_perf=v0)
        except Exception as exc:
            logger.error(
                "Verification failed: provider=%s error_type=%s detail=%r",
                state["verifier_name"],
                type(exc).__name__,
                exc,
            )
            err_ev.append(
                run_error_record(stage="verify", provider=state["verifier_name"], exc=exc)
            )
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
        (iter_dir / f"iteration_{round_idx + 1}_verify_raw.md").write_text(
            resp.text, encoding="utf-8"
        )
        data = parse_verifier_json(
            resp.text,
            provider_finish_reason=provider_finish_reason_label(resp.raw),
            provider_raw=resp.raw,
        )
        eq_ctx_raw = state.get("equity_prompt_render_context")
        iv_c: float | None = None
        hv_p: float | None = None
        if isinstance(eq_ctx_raw, dict):
            raw_iv = eq_ctx_raw.get("iv_crush_multiplier")
            if isinstance(raw_iv, (int, float)) and not isinstance(raw_iv, bool):
                iv_c = float(raw_iv)
            raw_hv = eq_ctx_raw.get("hv30_annualized_pct")
            if isinstance(raw_hv, (int, float)) and not isinstance(raw_hv, bool):
                hv_p = float(raw_hv)
        earn_calendar: date | None = None
        ed_raw: str | None = None
        raw_top = state.get("earnings_date")
        if isinstance(raw_top, str) and raw_top.strip():
            ed_raw = raw_top.strip()
        elif isinstance(eq_ctx_raw, dict):
            ctx_ed = eq_ctx_raw.get("earnings_date")
            if isinstance(ctx_ed, str) and ctx_ed.strip():
                ed_raw = ctx_ed.strip()
        if ed_raw:
            earn_calendar = _parse_earnings_calendar_date(ed_raw)
        et_raw = state.get("earnings_timing")
        earn_timing_s: str | None = None
        if isinstance(et_raw, str) and et_raw.strip():
            earn_timing_s = et_raw.strip()
        data = augment_verifier_result_with_sigma_structural_checks(
            syn,
            data,
            options_chain_data=state.get("options_chain_data"),
            symbol=str(state.get("symbol", "")),
            iv_crush_multiplier=iv_c,
            hv30_annualized_pct=hv_p,
            earnings_date=earn_calendar,
            earnings_timing=earn_timing_s,
            computed_sigma_bands_table=(
                state.get("computed_sigma_bands_table")
                if isinstance(state.get("computed_sigma_bands_table"), dict)
                else None
            ),
            t0_blend_preset=normalize_t0_blend_preset(state.get("t0_blend_preset", "default")),
            provider_iteration_bundle=_latest_provider_iteration_bundle(state),
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
        logger.info(
            "Node finalize output_dir=%s rounds=%s",
            str(out.resolve()),
            len(state["provider_responses"]),
        )
        parts: list[str] = [
            f"# Refined equity report: {state['symbol']}\n",
            "## Iteration changelog\n",
        ]
        full_changelog = bool(state.get("final_report_full_synthesis", True))
        for i, syn in enumerate(state["synthesis_history"], start=1):
            body = (
                syn.rstrip()
                if full_changelog
                else round_summary_for_changelog(syn, iteration_index=i)
            )
            round_heading = (
                f"### Round {i} synthesis\n\n"
                if full_changelog
                else f"### Round {i} synthesis (summary)\n\n"
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
            ver = (
                state["verification_history"][i - 1]
                if i <= len(state["verification_history"])
                else {}
            )
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
        seed = state.get("run_meta_seed")
        if run_json.is_file():
            meta = json.loads(run_json.read_text(encoding="utf-8"))
            if not isinstance(meta, dict):
                meta = {}
        elif isinstance(seed, dict):
            meta = copy.deepcopy(seed)
        else:
            meta = {}
        meta["timing"] = timing_summary
        meta["iterations_completed"] = len(state.get("provider_responses", []))
        meta["verification_history"] = state.get("verification_history", [])
        meta["options_chain_data"] = state.get("options_chain_data") or {}
        meta["computed_sigma_bands_table"] = (
            state.get("computed_sigma_bands_table")
            if isinstance(state.get("computed_sigma_bands_table"), dict)
            else None
        )
        syn_meta = state.get("synthesis_meta") or []
        if syn_meta and isinstance(syn_meta[-1], dict):
            meta["synthesis"] = syn_meta[-1]
        prior_errs = meta.get("errors")
        if not isinstance(prior_errs, list):
            prior_errs = []
        merged_errs: list[Any] = list(prior_errs) + list(state.get("error_events", []))
        meta["errors"] = merged_errs
        snap_cfg = meta.get("config")
        if isinstance(snap_cfg, dict):
            with contextlib.suppress(Exception):
                rc = RunConfig.model_validate(snap_cfg)
                meta["run_profile"] = rc.run_profile
                meta["env"] = rc.env
        snap = state.get("iterative_config_snapshot") or {}
        persist_rj = bool(
            snap.get("persist_run_json_to_disk", state.get("persist_run_json_to_disk", True))
        )
        run_doc = canonical_run_document_dict(meta)
        if persist_rj:
            run_json.write_text(format_run_json_for_disk(meta), encoding="utf-8")
        logger.info("Iterative wall-clock timing summary: %s", timing_summary)

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
            delete_checkpoint_after_success=bool(
                state.get("delete_checkpoint_after_success", True)
            ),
        )

        return {"final_report": report, "final_run_meta": run_doc}

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
        "earnings_date": cfg.earnings_date,
        "earnings_timing": cfg.earnings_timing,
        "computed_sigma_bands_table": (
            rendered.context.get("computed_sigma_bands_table")
            if isinstance(rendered.context.get("computed_sigma_bands_table"), dict)
            else None
        ),
        "t0_blend_preset": cfg.t0_blend_preset,
        "persist_run_json_to_disk": cfg.persist_run_json_to_disk,
    }


def dry_run_compile_only(*, registry: ProviderRegistry) -> list[str]:
    from langgraph.checkpoint.memory import MemorySaver

    app = compile_refinement_workflow(registry=registry, checkpointer=MemorySaver())
    nodes = app.get_graph().nodes
    out = sorted(n for n in nodes if not str(n).startswith("__"))
    logger.debug("Dry-run graph inspection nodes=%s", out)
    return out
