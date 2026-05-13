"""Server-side variance-additive σ bands and deterministic P(up) for prompt + verifier."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date
from typing import Any

from equity_analyst.drift_bounds import bound_daily_drift, computed_prob_up_pct
from equity_analyst.options_chain import (
    _parse_earnings_calendar_date,
    event_jump_implied_move_pct_from_prompt_dict,
    iv_crush_multiplier,
)
from equity_analyst.outcome_tracker import (
    compute_pead_avg_drift_pct,
    compute_realized_post_earnings_daily_vol_pct,
    compute_recent_momentum_drift_pct,
    fetch_hv30_annualized_percent,
    parse_equity_session_date_hint,
)
from equity_analyst.sigma_summary import parse_sigma_summary_json

logger = logging.getLogger(__name__)

MISSING_VALID_SIGMA_SUMMARY_JSON_MESSAGE = (
    "Cite or verify: PRIORITY — synthesis missing valid sigma_summary JSON; cannot match server σ table "
    "(emit the last fenced json code block with root key sigma_summary; next pass is synthesize-only unless "
    "contradictions require providers)."
)


@dataclass(frozen=True)
class ComputedSigmaSessionRow:
    session_date: date
    label: str
    n_trading: int
    one_sigma_half_width_pct: float
    two_sigma_half_width_pct: float
    three_sigma_half_width_pct: float
    one_sigma_low_dollar: float
    one_sigma_high_dollar: float
    two_sigma_low_dollar: float
    two_sigma_high_dollar: float
    three_sigma_low_dollar: float
    three_sigma_high_dollar: float
    prob_up_pct: float


@dataclass(frozen=True)
class ComputedSigmaBandsTable:
    anchor_price: float
    anchor_type: str
    event_jump_pct: float
    daily_vol_pct: float
    daily_vol_source: str
    daily_drift_pct: float
    drift_source_note: str
    sessions: tuple[ComputedSigmaSessionRow, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "anchor_price": self.anchor_price,
            "anchor_type": self.anchor_type,
            "event_jump_pct": self.event_jump_pct,
            "daily_vol_pct": self.daily_vol_pct,
            "daily_vol_source": self.daily_vol_source,
            "daily_drift_pct": self.daily_drift_pct,
            "drift_source_note": self.drift_source_note,
            "sessions": [
                {
                    "session_date": s.session_date.isoformat(),
                    "label": s.label,
                    "N": s.n_trading,
                    "one_sigma_half_width_pct": s.one_sigma_half_width_pct,
                    "two_sigma_half_width_pct": s.two_sigma_half_width_pct,
                    "three_sigma_half_width_pct": s.three_sigma_half_width_pct,
                    "one_sigma_low_dollar": s.one_sigma_low_dollar,
                    "one_sigma_high_dollar": s.one_sigma_high_dollar,
                    "two_sigma_low_dollar": s.two_sigma_low_dollar,
                    "two_sigma_high_dollar": s.two_sigma_high_dollar,
                    "three_sigma_low_dollar": s.three_sigma_low_dollar,
                    "three_sigma_high_dollar": s.three_sigma_high_dollar,
                    "prob_up_pct": s.prob_up_pct,
                }
                for s in self.sessions
            ],
        }


def resolve_daily_vol_pct_for_sigma(
    symbol: str,
    oc_data: dict[str, Any],
    *,
    earnings_date: str,
) -> tuple[float | None, str]:
    """Canonical daily vol %/day for post-event diffusion (HV30 path with optional IV crush)."""
    hv = fetch_hv30_annualized_percent(symbol)
    if hv is not None and hv > 0:
        base = float(hv) / math.sqrt(252.0)
        ivm: float | None = None
        if oc_data.get("options_chain_available"):
            ivm = iv_crush_multiplier(oc_data, earnings_date=earnings_date)
        if ivm is not None:
            return base * ivm, "HV30/sqrt252*iv_crush_multiplier"
        return base, "HV30/sqrt252"
    rv = compute_realized_post_earnings_daily_vol_pct(symbol)
    if rv is not None and rv > 0:
        return float(rv), "realized_post_earnings_5d_abs_mean"
    return None, "unavailable"


def _collect_unique_session_dates(
    *,
    earnings_date: str,
    next_trading_day: str,
    target_dates: list[str],
) -> list[tuple[date, str, str]]:
    """Return sorted unique ``(date, iso, label)`` rows (first label wins per calendar date)."""
    raw_pairs: list[tuple[str, str]] = [
        (earnings_date, "earnings_date"),
        (next_trading_day, "next_trading_day"),
    ]
    for i, td in enumerate(target_dates):
        raw_pairs.append((td, f"target_dates[{i}]"))
    seen: set[date] = set()
    out: list[tuple[date, str, str]] = []
    for text, origin in raw_pairs:
        d = parse_equity_session_date_hint(text)
        if d is None:
            continue
        if d in seen:
            continue
        seen.add(d)
        out.append((d, d.isoformat(), origin))
    out.sort(key=lambda t: t[0])
    return out


def compute_sigma_bands_server_side(
    *,
    anchor_price: float,
    anchor_type: str,
    earnings_date: str,
    earnings_timing: str | None,
    target_dates: list[str],
    next_trading_day: str,
    event_jump_pct: float,
    daily_vol_pct: float,
    daily_vol_source: str,
    daily_drift_pct: float,
    drift_source_note: str,
) -> ComputedSigmaBandsTable | None:
    """Compute σ bands for every collected session date using variance-additive math."""
    if anchor_price <= 0 or event_jump_pct < 0 or daily_vol_pct < 0:
        return None
    earn_cal = _parse_earnings_calendar_date(earnings_date)
    if earn_cal is None:
        return None
    from equity_analyst.iterative import (
        anchor_session_date_for_variance_check,
        trading_sessions_inclusive,
    )

    anchor_session = anchor_session_date_for_variance_check(earn_cal, earnings_timing)
    if anchor_session is None:
        return None
    rows_in = _collect_unique_session_dates(
        earnings_date=earnings_date,
        next_trading_day=next_trading_day,
        target_dates=target_dates,
    )
    if not rows_in:
        return None
    sessions: list[ComputedSigmaSessionRow] = []
    for d, iso, origin in rows_in:
        n_inc = trading_sessions_inclusive(anchor_session, d)
        sigma1 = math.sqrt(event_jump_pct**2 + float(n_inc) * daily_vol_pct**2)
        sigma2 = 2.0 * sigma1
        sigma3 = 3.0 * sigma1
        lo1 = anchor_price * (1.0 - sigma1 / 100.0)
        hi1 = anchor_price * (1.0 + sigma1 / 100.0)
        lo2 = anchor_price * (1.0 - sigma2 / 100.0)
        hi2 = anchor_price * (1.0 + sigma2 / 100.0)
        lo3 = anchor_price * (1.0 - sigma3 / 100.0)
        hi3 = anchor_price * (1.0 + sigma3 / 100.0)
        p_up = computed_prob_up_pct(daily_drift_pct, sigma1, n_inc)
        sessions.append(
            ComputedSigmaSessionRow(
                session_date=d,
                label=f"{iso} ({origin})",
                n_trading=n_inc,
                one_sigma_half_width_pct=float(sigma1),
                two_sigma_half_width_pct=float(sigma2),
                three_sigma_half_width_pct=float(sigma3),
                one_sigma_low_dollar=float(lo1),
                one_sigma_high_dollar=float(hi1),
                two_sigma_low_dollar=float(lo2),
                two_sigma_high_dollar=float(hi2),
                three_sigma_low_dollar=float(lo3),
                three_sigma_high_dollar=float(hi3),
                prob_up_pct=float(p_up),
            ),
        )
    return ComputedSigmaBandsTable(
        anchor_price=float(anchor_price),
        anchor_type=anchor_type,
        event_jump_pct=float(event_jump_pct),
        daily_vol_pct=float(daily_vol_pct),
        daily_vol_source=daily_vol_source,
        daily_drift_pct=float(daily_drift_pct),
        drift_source_note=drift_source_note,
        sessions=tuple(sessions),
    )


def format_computed_sigma_bands_markdown(table: ComputedSigmaBandsTable) -> str:
    """Human-readable table for equity / synthesizer prompts."""
    lines = [
        "**Server-computed σ bands (variance-additive)** — use **verbatim** in σ JSON and prose when this block is present.",
        "",
        f"- Anchor: **${table.anchor_price:.2f}** (`anchor_type={table.anchor_type!r}`)",
        f"- `event_jump` (front weekly ATM straddle / spot): **{table.event_jump_pct:.2f}%**",
        f"- `daily_vol`: **{table.daily_vol_pct:.2f}%/day** ({table.daily_vol_source})",
        f"- `daily_drift_pct` (bounded, for P(up)): **{table.daily_drift_pct:+.4f}%/day** ({table.drift_source_note})",
        "",
        "| Date | N | 1σ ±% | 2σ ±% | 3σ ±% | 1σ $ low | 1σ $ high | P(up)% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in table.sessions:
        lines.append(
            f"| {s.session_date.isoformat()} | {s.n_trading} | {s.one_sigma_half_width_pct:.2f} | "
            f"{s.two_sigma_half_width_pct:.2f} | {s.three_sigma_half_width_pct:.2f} | "
            f"{s.one_sigma_low_dollar:.2f} | {s.one_sigma_high_dollar:.2f} | {s.prob_up_pct:.1f} |",
        )
    lines.append("")
    lines.append(
        "Emit `sigma_summary` JSON whose `sessions[*].one_sigma_half_width_pct` / `three_sigma_half_width_pct` "
        "and dollar ranges match these rows within **±1 percentage point** on the % columns."
    )
    return "\n".join(lines)


def try_build_computed_sigma_bundle(
    *,
    symbol: str,
    anchor_price: float | None,
    same_day_intraday_available: bool,
    earnings_date: str,
    earnings_timing: str | None,
    target_dates: list[str],
    next_trading_day: str,
    oc_data: dict[str, Any],
) -> tuple[bool, str, dict[str, Any] | None, str]:
    """Return ``(available, markdown, json_dict_for_verifier, event_daily_tag)``."""
    spot = oc_data.get("spot")
    ap = float(anchor_price) if anchor_price is not None and anchor_price > 0 else None
    if ap is None and isinstance(spot, (int, float)) and float(spot) > 0:
        ap = float(spot)
    if ap is None or ap <= 0:
        return False, "", None, ""
    ej = event_jump_implied_move_pct_from_prompt_dict(oc_data, earnings_date=earnings_date)
    dv, dv_src = resolve_daily_vol_pct_for_sigma(symbol, oc_data, earnings_date=earnings_date)
    if ej is None and bool(oc_data.get("options_chain_available")):
        logger.warning(
            "try_build_computed_sigma_bundle: event_jump unavailable (symbol=%s earnings_date=%s); "
            "computed σ bundle suppressed",
            symbol,
            earnings_date,
        )
    if ej is None or dv is None:
        return False, "", None, ""

    pead = compute_pead_avg_drift_pct(symbol)
    mom = compute_recent_momentum_drift_pct(symbol, lookback_days=10)
    drift_note: str
    drift_val: float
    if pead is not None:
        from equity_analyst.drift_bounds import bound_daily_drift

        drift_val, _ = bound_daily_drift(float(pead), "PEAD_avg")
        drift_note = "PEAD_avg (bounded)"
    elif mom is not None:
        from equity_analyst.drift_bounds import bound_daily_drift

        drift_val, _ = bound_daily_drift(float(mom), "recent_momentum")
        drift_note = "recent_momentum (bounded)"
    else:
        drift_val = 0.0
        drift_note = "default_zero (no PEAD/momentum)"

    atype = "same_day_intraday" if same_day_intraday_available else "prior_close"

    table = compute_sigma_bands_server_side(
        anchor_price=ap,
        anchor_type=atype,
        earnings_date=earnings_date,
        earnings_timing=earnings_timing,
        target_dates=target_dates,
        next_trading_day=next_trading_day,
        event_jump_pct=float(ej),
        daily_vol_pct=float(dv),
        daily_vol_source=dv_src,
        daily_drift_pct=float(drift_val),
        drift_source_note=drift_note,
    )
    if table is None:
        return False, "", None, ""
    tag = f"event_jump={ej:.2f}% daily_vol={dv:.2f}%/day ({dv_src})"
    return True, format_computed_sigma_bands_markdown(table), table.to_json_dict(), tag


def format_computed_probabilities_reference_markdown(
    synthesis_text: str,
    *,
    earnings_date: str,
    earnings_timing: str | None,
) -> str:
    """Iteration ≥2 helper: table of server-recomputed P(up) from emitted drift + σ rows."""
    from equity_analyst.iterative import sigma_sessions_payload_from_sigma_summary_model

    parsed = parse_sigma_summary_json(synthesis_text)
    if parsed is None:
        return ""
    pl = parsed.sigma_summary
    if not any(s.prob_up_pct is not None for s in pl.sessions):
        return ""
    if pl.drift_source is None or pl.daily_drift_pct is None:
        return ""
    earn_cal = _parse_earnings_calendar_date(earnings_date)
    if earn_cal is None:
        return ""
    mu_b, _ = bound_daily_drift(float(pl.daily_drift_pct), pl.drift_source)
    sess_pl, _, err = sigma_sessions_payload_from_sigma_summary_model(
        parsed,
        earn_cal=earn_cal,
        earnings_timing=earnings_timing,
    )
    if err is not None:
        return ""
    lines = [
        "### Server-computed P(up) reference (bounded drift)",
        "",
        f"Using drift **{mu_b:+.4f}%/day** from your last `sigma_summary` (`drift_source={pl.drift_source!r}`), "
        "recomputed vs your `one_sigma_half_width_pct` and server `N`:",
        "",
        "| Date | N | σ(1σ) % | P(up) computed |",
        "|---|---:|---:|---:|",
    ]
    for row in sess_pl:
        raw = row.get("session_date")
        if not isinstance(raw, str):
            continue
        sig = float(row["sigma_pct"])
        n_inc = int(row["N"])
        p = computed_prob_up_pct(mu_b, sig, n_inc)
        lines.append(f"| {raw} | {n_inc} | {sig:.2f} | {p:.1f} |")
    lines.append("")
    return "\n".join(lines)


def verify_emitted_sigma_bands_match_computed(
    synthesis_text: str,
    computed_table: dict[str, Any] | None,
    *,
    tolerance_pp: float = 1.0,
) -> list[str]:
    """Compare synthesis ``sigma_summary`` JSON % half-widths to the server pre-computed table."""
    if not computed_table:
        return []
    rows = computed_table.get("sessions")
    if not isinstance(rows, list) or not rows:
        return []
    parsed = parse_sigma_summary_json(synthesis_text)
    if parsed is None:
        return [MISSING_VALID_SIGMA_SUMMARY_JSON_MESSAGE]
    by_date: dict[str, dict[str, Any]] = {}
    for r in rows:
        if not isinstance(r, dict):
            continue
        k = str(r.get("session_date") or "").strip()[:10]
        if len(k) == 10 and k[4] == "-" and k[7] == "-":
            by_date[k] = r
    out: list[str] = []
    for sess in parsed.sigma_summary.sessions:
        key = sess.date.isoformat()
        comp = by_date.get(key)
        if comp is None:
            continue
        try:
            c1 = float(comp["one_sigma_half_width_pct"])
            c3 = float(comp["three_sigma_half_width_pct"])
        except (TypeError, ValueError, KeyError):
            continue
        e1 = float(sess.one_sigma_half_width_pct)
        e3 = float(sess.three_sigma_half_width_pct)
        if abs(e1 - c1) > tolerance_pp:
            out.append(
                "Cite or verify: "
                f"synthesis session {key} 1σ half-width emitted {e1:.2f}% vs server-computed {c1:.2f}% "
                f"(tolerance {tolerance_pp:.1f}pp).",
            )
        if abs(e3 - c3) > tolerance_pp:
            out.append(
                "Cite or verify: "
                f"synthesis session {key} 3σ half-width emitted {e3:.2f}% vs server-computed {c3:.2f}% "
                f"(tolerance {tolerance_pp:.1f}pp).",
            )
    return out
