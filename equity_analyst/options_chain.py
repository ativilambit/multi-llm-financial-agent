from __future__ import annotations

import functools
import logging
import math
import re
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Literal, cast

logger = logging.getLogger(__name__)

ISO_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


class OptionsChainSoftFetchError(Exception):
    """Raised from the LRU-cached fetch path so soft failures are not cached."""

    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__(payload.get("fetch_error") or "options_chain_unavailable")
        self.payload = payload


def options_chain_expiry_audit_messages(
    synthesis_text: str,
    verifier_result: dict[str, Any],
    *,
    options_chain_data: dict[str, Any] | None,
    symbol: str,
) -> list[str]:
    """Emit short ``unverifiable`` strings when sigma-baseline text cites option expiries absent from Yahoo chain."""
    if not options_chain_data or not options_chain_data.get("options_chain_available"):
        return []
    verified_raw = options_chain_data.get("available_expiries") or []
    verified = frozenset(str(x) for x in verified_raw)
    if not verified:
        return []
    sym = symbol.strip().upper() or "TICKER"
    extras: list[str] = []
    _gs = chr(0x03C3)

    def _fmt_verified() -> str:
        ordered = sorted(verified)
        head = ", ".join(ordered[:10])
        return f"{head}, …" if len(ordered) > 10 else head

    for row in verifier_result.get("sigma_band_sessions") or []:
        if not isinstance(row, dict):
            continue
        base = str(row.get("sigma_baseline", ""))
        sess = str(row.get("session", "")).strip() or "sigma-band session"
        for m in ISO_DATE_RE.findall(base):
            if m not in verified:
                extras.append(
                    f"Cite or verify: {sym} claims expiry {m} for {sess} but verified chain shows [{_fmt_verified()}].",
                )

    cues = ("expiry", "expiration", "weekly", "chain", "derived from", "option chain")
    for line in synthesis_text.splitlines():
        low = line.lower()
        if f"3{_gs}" not in line:
            continue
        if not any(c in low for c in cues):
            continue
        for m in ISO_DATE_RE.findall(line):
            if m not in verified:
                extras.append(
                    f"Cite or verify: {sym} claims expiry {m} but verified chain shows [{_fmt_verified()}].",
                )
    return extras


def _parse_earnings_calendar_date(earnings_date: str) -> date | None:
    """Parse ``earnings_date`` using the same fuzzy parser as outcome auto-fetch."""
    try:
        from equity_analyst.outcome_tracker import _parse_earnings_date_fuzzy
    except ImportError:  # pragma: no cover
        logger.warning("options_chain: outcome_tracker import failed for earnings parse")
        return None
    dt = _parse_earnings_date_fuzzy(earnings_date)
    if dt is None:
        return None
    return dt.date()


def _parse_today_date(today_date: str) -> date | None:
    try:
        from equity_analyst.outcome_tracker import _parse_earnings_date_fuzzy
    except ImportError:  # pragma: no cover
        return None
    dt = _parse_earnings_date_fuzzy(today_date)
    if dt is None:
        return None
    return dt.date()


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


def _trading_dte(as_of: date, expiry: date) -> int:
    """Trading sessions from the session after ``as_of`` through ``expiry`` inclusive."""
    if expiry <= as_of:
        return 0
    return trading_sessions_after_exclusive(as_of, expiry)


def _week_start_monday(d: date) -> date:
    return d - timedelta(days=d.weekday())


def _is_third_friday(d: date) -> bool:
    if d.weekday() != 4:
        return False
    return 15 <= d.day <= 21


def is_standard_monthly_expiration(d: date) -> bool:
    """True for standard **monthly** equity options expiries (3rd Friday of the month).

    OCC/CBOE may shift the trading week when the 3rd Friday is an exchange holiday (contracts can
    list on the **prior Thursday** in that edge case). We treat **that Thursday** as ``monthly`` too
    when the following calendar day is the 3rd Friday — otherwise only the Friday pattern matches.
    Quarterlies/LEAPS on other calendars are **not** distinguished here and classify as non-monthly.
    """
    if _is_third_friday(d):
        return True
    if d.weekday() == 3:  # Thursday before a 3rd Friday (documented holiday nuance)
        nxt = d + timedelta(days=1)
        if _is_third_friday(nxt):
            return True
    return False


def expiry_listing_kind(d: date) -> Literal["monthly", "weekly"]:
    """Coarse listing bucket for an expiry date (``monthly`` = standard monthly rule)."""
    return "monthly" if is_standard_monthly_expiration(d) else "weekly"


def pick_front_listed_expiry_for_earnings(
    available: list[date],
    earn: date,
    *,
    max_weekly_lookahead_days: int,
) -> tuple[date | None, Literal["weekly", "monthly", "none"], str]:
    """Choose the listed expiry used for event straddle / sigma anchoring (weekly preferred)."""
    if not available:
        return None, "none", "no_listed_expiry_in_window"
    win_end = earn + timedelta(days=max(1, int(max_weekly_lookahead_days)))
    weeklies = [
        d
        for d in available
        if d >= earn and d <= win_end and not is_standard_monthly_expiration(d)
    ]
    if weeklies:
        return min(weeklies), "weekly", "soonest_non_monthly_listing_within_lookahead_window"
    monthlies = [d for d in available if d >= earn and is_standard_monthly_expiration(d)]
    if monthlies:
        return min(monthlies), "monthly", "fallback_nearest_standard_monthly_on_or_after_earnings"
    return None, "none", "no_listed_expiry_in_window"


def _coerce_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(x):
        return None
    return x


def _coerce_int(v: Any) -> int:
    if v is None:
        return 0
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 0


def _parse_expiry_list(raw: list[str] | tuple[str, ...]) -> list[date]:
    out: list[date] = []
    for s in raw:
        try:
            out.append(date.fromisoformat(str(s)[:10]))
        except ValueError:
            continue
    return sorted(set(out))


def _forward_variance_expiry_bracket(available: list[date], earnings: date) -> tuple[date | None, date | None]:
    """Return ``(T1, T2)`` for forward implied variance: closest listed expiry **before** earnings, then first on/after."""
    if not available:
        return None, None
    pre = [d for d in available if d < earnings]
    post = [d for d in available if d >= earnings]
    t1 = max(pre) if pre else None
    t2 = min(post) if post else None
    return t1, t2


def _select_relevant_expiries_with_rationale(available: list[date], earnings: date) -> list[tuple[date, str]]:
    """Pick earnings-week, pre-earnings same-week, T+1w-ish Friday, and nearest monthly (3rd Fri), with reasons."""
    if not available:
        return []
    chosen: list[tuple[date, str]] = []
    seen: set[date] = set()

    def _add(d: date, reason: str) -> None:
        if d in seen:
            return
        seen.add(d)
        chosen.append((d, reason))

    # 1) Nearest expiry on or after earnings (earnings-week contract).
    post = [d for d in available if d >= earnings]
    if post:
        _add(min(post), "nearest listed expiry on or after earnings calendar date (event-week)")

    # 2) Nearest expiry before earnings in the same ISO week as earnings (pre-earnings chain).
    w0 = _week_start_monday(earnings)
    pre_same_week = [d for d in available if earnings > d >= w0]
    if pre_same_week:
        _add(max(pre_same_week), "latest listed expiry before earnings in the same ISO week as earnings")

    # 3) Following Friday ~T+1 week (closest listed expiry to earnings + 7d, after earnings).
    target = earnings + timedelta(days=7)
    after_earn = [d for d in available if d > earnings]
    if after_earn:
        _add(
            min(after_earn, key=lambda d: (abs((d - target).days), d)),
            "listed expiry closest to earnings+7d (post-earnings weekly anchor)",
        )

    # 4) Monthly (3rd Friday) closest to earnings.
    thirds = [d for d in available if _is_third_friday(d)]
    if thirds:
        _add(
            min(thirds, key=lambda d: (abs((d - earnings).days), d)),
            "monthly (3rd Friday) expiry closest to earnings calendar date",
        )

    # If nothing matched (odd chain), fall back to closest overall.
    if not chosen:
        _add(
            min(available, key=lambda d: (abs((d - earnings).days), d)),
            "fallback: listed expiry closest to earnings calendar date",
        )
    return chosen


def _select_relevant_expiries(available: list[date], earnings: date) -> list[date]:
    """Pick earnings-week, pre-earnings same-week, T+1w-ish Friday, and nearest monthly (3rd Fri)."""
    return [d for d, _ in _select_relevant_expiries_with_rationale(available, earnings)]


def _implied_move_total_percent_from_row(row: dict[str, Any], *, spot: float | None) -> float | None:
    """Straddle-implied move in **percent** (e.g. 11.31 for ~11.31% move), from snapshot row + spot."""
    im = _coerce_float(row.get("implied_move_pct"))
    if im is not None and im > 0:
        if im < 2.0:
            return float(im * 100.0)
        return float(im)
    s = _coerce_float(spot)
    straddle = _coerce_float(row.get("atm_straddle_mid"))
    if straddle is not None and s is not None and s > 0 and straddle > 0:
        return float(straddle / s * 100.0)
    iv_atm = _atm_iv_proxy_ex(row)
    exp_s = str(row.get("expiry_date") or "")[:10]
    td = _coerce_int(row.get("dte"))
    cal_d: int | None = None
    try:
        date.fromisoformat(exp_s)
        # Without as_of, use DTE-derived calendar max(1, td) as weak proxy
        if td > 0:
            cal_d = max(td, 1)
    except ValueError:
        cal_d = max(td, 1) if td > 0 else None
    if iv_atm is not None and iv_atm > 0 and cal_d is not None and cal_d > 0:
        return float(iv_atm * math.sqrt(float(cal_d) / 365.0) * 100.0)
    return None


def apply_options_chain_event_expiry_resolution(
    oc_data: dict[str, Any],
    *,
    earnings_date: str,
    max_weekly_lookahead_days: int = 14,
) -> None:
    """Populate ``expiry_used`` / ``expiry_class`` / straddle literals for sigma + prompts (mutates ``oc_data``)."""
    if oc_data.get("_event_expiry_resolution_applied"):
        return
    oc_data["_event_expiry_resolution_applied"] = True
    oc_data["max_weekly_lookahead_days"] = int(max_weekly_lookahead_days)
    if not oc_data.get("options_chain_available"):
        oc_data.setdefault("expiry_class", "none")
        oc_data.setdefault("expiry_used", None)
        oc_data.setdefault("event_jump_source", "unavailable")
        oc_data.setdefault("lit_event_straddle_move_pct", None)
        oc_data.setdefault("event_jump_for_sigma_pct", None)
        oc_data.setdefault("event_expiry_calendar_days_after_earn", None)
        oc_data.setdefault("front_contract_selection_note", "chain_unavailable")
        return

    earn = _parse_earnings_calendar_date(earnings_date)
    avail = _available_expiry_dates_from_oc(oc_data)
    spot = _coerce_float(oc_data.get("spot"))
    if earn is None or not avail:
        oc_data["expiry_class"] = "none"
        oc_data["expiry_used"] = None
        oc_data["event_jump_source"] = "unavailable"
        oc_data["lit_event_straddle_move_pct"] = None
        oc_data["event_jump_for_sigma_pct"] = None
        oc_data["event_expiry_calendar_days_after_earn"] = None
        oc_data["front_contract_selection_note"] = "no_listed_expiry_in_window"
        return

    exp_d, eclass, note = pick_front_listed_expiry_for_earnings(
        avail,
        earn,
        max_weekly_lookahead_days=max_weekly_lookahead_days,
    )
    oc_data["front_contract_selection_note"] = note
    if exp_d is None or eclass == "none":
        oc_data["expiry_class"] = "none"
        oc_data["expiry_used"] = None
        oc_data["event_jump_source"] = "unavailable"
        oc_data["lit_event_straddle_move_pct"] = None
        oc_data["event_jump_for_sigma_pct"] = 0.0
        oc_data["event_expiry_calendar_days_after_earn"] = None
        return

    oc_data["expiry_used"] = exp_d.isoformat()
    oc_data["expiry_class"] = eclass
    oc_data["event_expiry_calendar_days_after_earn"] = int((exp_d - earn).days)
    row = _selected_row_for_calendar_expiry(oc_data, exp_d)
    if row is None:
        oc_data["event_jump_source"] = "unavailable"
        oc_data["lit_event_straddle_move_pct"] = None
        oc_data["event_jump_for_sigma_pct"] = 0.0
        oc_data["front_contract_selection_note"] = note + ";missing_selected_expiries_row"
        return

    lit = _implied_move_total_percent_from_row(row, spot=spot)
    oc_data["lit_event_straddle_move_pct"] = lit
    if lit is None or lit <= 0:
        oc_data["event_jump_source"] = "unavailable"
        oc_data["event_jump_for_sigma_pct"] = 0.0
        return

    if eclass == "weekly":
        oc_data["event_jump_source"] = "front_weekly_straddle"
        oc_data["event_jump_for_sigma_pct"] = float(lit)
        return

    # Monthly: provisional; sigma_compute may replace event_jump_for_sigma with event-only estimate.
    oc_data["event_jump_source"] = "monthly_straddle_with_residual"
    oc_data["event_jump_for_sigma_pct"] = float(lit)


def _pick_atm_strike(strikes: list[float], spot: float) -> float | None:
    if not strikes or spot <= 0:
        return None
    return min(strikes, key=lambda k: (abs(k - spot), k))


def _row_at_strike(df: Any, strike: float) -> dict[str, Any] | None:
    if df is None or getattr(df, "empty", True):
        return None
    try:
        sub = df[df["strike"] == strike]
    except Exception:
        return None
    if getattr(sub, "empty", True):
        try:
            idx = (df["strike"] - strike).abs().idxmin()
            sub = df.loc[[idx]]
        except Exception:
            return None
    if getattr(sub, "empty", True):
        return None
    try:
        return cast(dict[str, Any], sub.iloc[0].to_dict())
    except Exception:
        return None


def _col(row: dict[str, Any], *names: str) -> Any:
    for n in names:
        if n in row and row[n] is not None:
            return row[n]
    return None


def _mid_from_row(row: dict[str, Any]) -> float | None:
    bid = _coerce_float(_col(row, "bid", "Bid"))
    ask = _coerce_float(_col(row, "ask", "Ask"))
    last = _coerce_float(_col(row, "lastPrice", "last", "Last"))
    if bid is not None and ask is not None and bid > 0 and ask > 0:
        return (bid + ask) / 2.0
    return last


def _iv_from_row(row: dict[str, Any]) -> float | None:
    return _coerce_float(_col(row, "impliedVolatility"))


def _totals(df: Any) -> tuple[int, int]:
    if df is None or getattr(df, "empty", True):
        return 0, 0
    vol = 0
    oi = 0
    try:
        if "volume" in df.columns:
            vol = int(df["volume"].fillna(0).sum())
        if "openInterest" in df.columns:
            oi = int(df["openInterest"].fillna(0).sum())
    except Exception:
        return 0, 0
    return vol, oi


def _skew_proxy_iv(calls: Any, puts: Any, spot: float) -> tuple[float | None, str]:
    """Approximate 25-delta skew: IV at ~+5% call strike minus IV at ~-5% put strike."""
    if spot <= 0:
        return None, "spot unavailable"
    tgt_c = spot * 1.05
    tgt_p = spot * 0.95
    try:
        c_strikes = [float(x) for x in calls["strike"].tolist()] if calls is not None else []
        p_strikes = [float(x) for x in puts["strike"].tolist()] if puts is not None else []
    except Exception:
        return None, "strike scan failed"
    if not c_strikes or not p_strikes:
        return None, "missing strikes"
    sc = min((s for s in c_strikes if s >= tgt_c), default=min(c_strikes, key=lambda s: abs(s - tgt_c)))
    sp_ = max((s for s in p_strikes if s <= tgt_p), default=min(p_strikes, key=lambda s: abs(s - tgt_p)))
    rc = _row_at_strike(calls, sc)
    rp = _row_at_strike(puts, sp_)
    if not rc or not rp:
        return None, "rows missing"
    iv_c = _iv_from_row(rc)
    iv_p = _iv_from_row(rp)
    if iv_c is None or iv_p is None:
        return None, "IV NaN on proxy strikes"
    note = "approx +/-5% strike proxy for 25d; yfinance has no delta/Greeks"
    return iv_c - iv_p, note


@dataclass
class ExpirySnapshot:
    expiry_date: str
    dte: int
    atm_strike: float | None
    atm_call_bid: float | None
    atm_call_ask: float | None
    atm_call_mid: float | None
    atm_call_last: float | None
    atm_call_iv: float | None
    atm_put_bid: float | None
    atm_put_ask: float | None
    atm_put_mid: float | None
    atm_put_last: float | None
    atm_put_iv: float | None
    atm_straddle_mid: float | None
    implied_move_pct: float | None
    expected_move_dollar: float | None
    skew_25d_call_minus_put_iv: float | None
    skew_25d_note: str
    total_call_volume: int
    total_put_volume: int
    put_call_ratio: float | None
    total_call_oi: int
    total_put_oi: int
    put_call_ratio_oi: float | None
    selection_rationale: str = ""


@dataclass
class OptionsChainSnapshot:
    as_of: str
    symbol: str
    spot: float | None
    available_expiries: list[str]
    selected_expiries: list[ExpirySnapshot]
    fetch_error: str | None = None

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "options_chain_available": bool(self.selected_expiries or self.available_expiries),
            "as_of": self.as_of,
            "symbol": self.symbol.upper(),
            "spot": self.spot,
            "available_expiries": list(self.available_expiries),
            "selected_expiries": [asdict(x) for x in self.selected_expiries],
            "fetch_error": self.fetch_error,
        }

    def to_markdown_table(self, earnings_date: str | None = None) -> str:
        if not self.selected_expiries and not self.available_expiries:
            return "_No option expiries in snapshot._\n"
        lines: list[str] = []
        lines.append(f"**Verified options chain** - `{self.symbol.upper()}` as-of `{self.as_of}` - spot: ")
        lines.append(f"`{self.spot:.4f}`\n" if self.spot else "`n/a`\n")
        _s = chr(0x03C3)
        lines.append(
            f"\n| Expiry | DTE | ATM strike | Straddle mid | Implied move % | "
            f"EM $ (+/-1{_s}) | Call IV | Put IV | Skew (25d*) | Tot C vol | Tot P vol | PCR vol | PCR OI |\n"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for ex in self.selected_expiries:
            im = ex.implied_move_pct
            im_s = f"{im * 100:.3f}%" if im is not None else ""
            skew = ex.skew_25d_call_minus_put_iv
            skew_s = f"{skew:.4f}" if skew is not None else "n/a"
            pcr_v = ex.put_call_ratio
            pcr_v_s = f"{pcr_v:.3f}" if pcr_v is not None else "n/a"
            pcr_oi = ex.put_call_ratio_oi
            pcr_oi_s = f"{pcr_oi:.3f}" if pcr_oi is not None else "n/a"
            lines.append(
                f"| `{ex.expiry_date}` | {ex.dte} | "
                f"{ex.atm_strike if ex.atm_strike is not None else ''} | "
                f"{ex.atm_straddle_mid if ex.atm_straddle_mid is not None else ''} | "
                f"{im_s} | "
                f"{ex.expected_move_dollar if ex.expected_move_dollar is not None else ''} | "
                f"{ex.atm_call_iv if ex.atm_call_iv is not None else ''} | "
                f"{ex.atm_put_iv if ex.atm_put_iv is not None else ''} | "
                f"{skew_s} | {ex.total_call_volume} | {ex.total_put_volume} | {pcr_v_s} | {pcr_oi_s} |\n"
            )
        lines.append("\n*Skew column: " + (self.selected_expiries[0].skew_25d_note if self.selected_expiries else "n/a") + "*\n")
        avail = ", ".join(f"`{e}`" for e in self.available_expiries[:24])
        if len(self.available_expiries) > 24:
            avail += ", ..."
        lines.append(f"\n**Listed expiries (Yahoo/yfinance):** {avail or '_(none)_'}\n")
        out = "".join(lines)
        if earnings_date:
            mult = iv_crush_multiplier(self, earnings_date=earnings_date)
            if mult is not None:
                out += (
                    f"\n**IV crush ratio (post/event):** `{mult:.4f}` "
                    "(post-event weekly ATM IV ÷ event-week ATM IV; apply per Rule 2 when using HV30 "
                    "for post-print diffusion).\n"
                )
        return out


def _resolve_spot(ticker: Any) -> float | None:
    try:
        fi = getattr(ticker, "fast_info", None)
        if fi is not None:
            v = _coerce_float(
                getattr(fi, "last_price", None)
                or getattr(fi, "lastPrice", None)
                or (fi.get("last_price") if isinstance(fi, dict) else None),
            )
            if v is not None and v > 0:
                return v
    except Exception:
        pass
    try:
        hist = ticker.history(period="5d")
        if hist is not None and not getattr(hist, "empty", True):
            return _coerce_float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return None


def _build_expiry_snapshot(
    *,
    expiry_d: date,
    as_of: date,
    spot: float | None,
    calls: Any,
    puts: Any,
    selection_rationale: str = "",
) -> ExpirySnapshot | None:
    if calls is None or puts is None:
        return None
    try:
        strikes_c = [float(x) for x in calls["strike"].tolist()]
        strikes_p = [float(x) for x in puts["strike"].tolist()]
    except Exception:
        return None
    common = sorted(set(strikes_c) & set(strikes_p))
    if not common or spot is None or spot <= 0:
        return None
    atm = _pick_atm_strike(common, spot)
    if atm is None:
        return None
    c_row_d = _row_at_strike(calls, atm)
    p_row_d = _row_at_strike(puts, atm)
    if not c_row_d or not p_row_d:
        return None

    def _pack(row: dict[str, Any]) -> tuple[float | None, float | None, float | None, float | None, float | None]:
        bid = _coerce_float(_col(row, "bid", "Bid"))
        ask = _coerce_float(_col(row, "ask", "Ask"))
        mid = _mid_from_row(row)
        last = _coerce_float(_col(row, "lastPrice", "last", "Last"))
        iv = _iv_from_row(row)
        return bid, ask, mid, last, iv

    cb, ca, cm, cl, civ = _pack(c_row_d)
    pb, pa, pm, pl, piv = _pack(p_row_d)
    straddle = None
    if cm is not None and pm is not None:
        straddle = cm + pm
    elif cl is not None and pl is not None:
        straddle = cl + pl
    im_pct = (straddle / spot) if straddle is not None and spot and spot > 0 else None
    cv, coi = _totals(calls)
    pv, poi = _totals(puts)
    pcr = (pv / cv) if cv > 0 else None
    pcr_oi = (poi / coi) if coi > 0 else None
    skew, skew_note = _skew_proxy_iv(calls, puts, spot)
    dte = _trading_dte(as_of, expiry_d)
    return ExpirySnapshot(
        expiry_date=expiry_d.isoformat(),
        dte=dte,
        atm_strike=atm,
        atm_call_bid=cb,
        atm_call_ask=ca,
        atm_call_mid=cm,
        atm_call_last=cl,
        atm_call_iv=civ,
        atm_put_bid=pb,
        atm_put_ask=pa,
        atm_put_mid=pm,
        atm_put_last=pl,
        atm_put_iv=piv,
        atm_straddle_mid=straddle,
        implied_move_pct=im_pct,
        expected_move_dollar=straddle,
        skew_25d_call_minus_put_iv=skew,
        skew_25d_note=skew_note,
        total_call_volume=cv,
        total_put_volume=pv,
        put_call_ratio=pcr,
        total_call_oi=coi,
        total_put_oi=poi,
        put_call_ratio_oi=pcr_oi,
        selection_rationale=selection_rationale,
    )


def _iso_utc_z_now() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _expiry_snapshot_from_row_dict(r: dict[str, Any]) -> ExpirySnapshot:
    d = dict(r)
    d["total_call_volume"] = _coerce_int(d.get("total_call_volume"))
    d["total_put_volume"] = _coerce_int(d.get("total_put_volume"))
    d["total_call_oi"] = _coerce_int(d.get("total_call_oi"))
    d["total_put_oi"] = _coerce_int(d.get("total_put_oi"))
    d["dte"] = _coerce_int(d.get("dte"))
    if "selection_rationale" not in d or d.get("selection_rationale") is None:
        d["selection_rationale"] = ""
    return ExpirySnapshot(**d)


def _atm_iv_proxy_ex(ex: ExpirySnapshot | dict[str, Any]) -> float | None:
    """ATM implied vol: average of call and put when both present; else first available side."""
    if isinstance(ex, ExpirySnapshot):
        c, p = ex.atm_call_iv, ex.atm_put_iv
    else:
        c = _coerce_float(ex.get("atm_call_iv"))
        p = _coerce_float(ex.get("atm_put_iv"))
    if c is not None and p is not None and c > 0 and p > 0:
        return (c + p) / 2.0
    if c is not None and c > 0:
        return c
    if p is not None and p > 0:
        return p
    return None


def _selected_expiry_rows(snapshot: OptionsChainSnapshot | dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(snapshot, OptionsChainSnapshot):
        return [asdict(x) for x in snapshot.selected_expiries]
    rows = snapshot.get("selected_expiries") or []
    out: list[dict[str, Any]] = []
    for r in rows:
        if isinstance(r, dict):
            out.append(dict(r))
    return out


def iv_crush_multiplier(
    snapshot: OptionsChainSnapshot | dict[str, Any] | None,
    *,
    earnings_date: str,
) -> float | None:
    """Post/event IV ratio for HV30 diffusion adjustment (``IV_post / IV_event``).

    Uses ``selected_expiries`` only: **event-week** = first listed expiry on/after ``earnings_date``;
    **post-event** = the next listed expiry strictly after that. ATM IV is the average of ATM call
    and put IV when both exist, else the available side.

    Returns ``None`` when expiries or IVs are missing, or when the raw ratio falls outside
    ``[0.4, 1.2]`` (logged as suspicious chain data).
    """
    if snapshot is None:
        return None
    earn = _parse_earnings_calendar_date(earnings_date)
    if earn is None:
        logger.warning("iv_crush_multiplier: could not parse earnings_date=%r", earnings_date)
        return None

    rows = _selected_expiry_rows(snapshot)
    dated: list[tuple[date, dict[str, Any]]] = []
    for row in rows:
        ds = str(row.get("expiry_date") or "")[:10]
        try:
            dated.append((date.fromisoformat(ds), row))
        except ValueError:
            continue
    dated.sort(key=lambda t: t[0])
    if len(dated) < 2:
        return None

    event_i: int | None = None
    for i, (d, _) in enumerate(dated):
        if d >= earn:
            event_i = i
            break
    if event_i is None or event_i + 1 >= len(dated):
        return None

    _, row_ev = dated[event_i]
    _, row_post = dated[event_i + 1]
    iv_ev = _atm_iv_proxy_ex(row_ev)
    iv_post = _atm_iv_proxy_ex(row_post)
    if iv_ev is None or iv_post is None or iv_ev <= 0.0 or iv_post <= 0.0:
        return None

    ratio = iv_post / iv_ev
    lo, hi = 0.4, 1.2
    if ratio < lo or ratio > hi:
        logger.warning(
            "iv_crush_multiplier: ratio %.4f outside [%.1f, %.1f] (event=%s post=%s); ignoring",
            ratio,
            lo,
            hi,
            str(row_ev.get("expiry_date")),
            str(row_post.get("expiry_date")),
        )
        return None
    return float(ratio)


def event_jump_implied_move_pct_from_prompt_dict(
    oc_data: dict[str, Any],
    *,
    earnings_date: str,
    max_weekly_lookahead_days: int = 14,
) -> float | None:
    """ATM straddle / spot x 100 for the **server-resolved** front earnings contract (weekly preferred)."""
    if not oc_data.get("options_chain_available"):
        return None
    apply_options_chain_event_expiry_resolution(
        oc_data,
        earnings_date=earnings_date,
        max_weekly_lookahead_days=int(oc_data.get("max_weekly_lookahead_days") or max_weekly_lookahead_days),
    )
    lit = oc_data.get("lit_event_straddle_move_pct")
    if lit is not None and isinstance(lit, (int, float)) and float(lit) > 0:
        return float(lit)
    snap = options_chain_snapshot_from_prompt_dict(oc_data)
    if snap is None:
        return None
    earn = _parse_earnings_calendar_date(earnings_date)
    if earn is None:
        logger.warning("event_jump_implied_move_pct: could not parse earnings_date=%r", earnings_date)
        return None
    rows = _selected_expiry_rows(snap)
    dated: list[tuple[date, dict[str, Any]]] = []
    for row in rows:
        ds = str(row.get("expiry_date") or "")[:10]
        try:
            dated.append((date.fromisoformat(ds), row))
        except ValueError:
            continue
    dated.sort(key=lambda t: t[0])
    if not dated:
        return None
    event_row: dict[str, Any] | None = None
    for d, row in dated:
        if d >= earn:
            event_row = row
            break
    if event_row is None:
        return None
    im = _coerce_float(event_row.get("implied_move_pct"))
    if im is not None and im > 0:
        # Snapshot stores implied move as straddle/spot **ratio** (e.g. 0.1131 == 11.31%), not percent.
        if im < 2.0:
            return float(im * 100.0)
        return float(im)
    spot = _coerce_float(oc_data.get("spot")) or snap.spot
    straddle = _coerce_float(event_row.get("atm_straddle_mid"))
    if straddle is not None and spot is not None and spot > 0 and straddle > 0:
        return float(straddle / spot * 100.0)
    # Fallback when straddle / implied_move fields are missing: ATM IV * sqrt(calendar DTE / 365).
    iv_atm = _atm_iv_proxy_ex(event_row)
    exp_s = str(event_row.get("expiry_date") or "")[:10]
    as_of_s = str(oc_data.get("as_of") or "")[:10]
    cal_d: int | None = None
    try:
        exp_d = date.fromisoformat(exp_s)
        as_d = date.fromisoformat(as_of_s[:10]) if len(as_of_s) >= 10 else None
        if as_d is not None:
            cal_d = max((exp_d - as_d).days, 1)
    except ValueError:
        cal_d = None
    if cal_d is None:
        td = _coerce_int(event_row.get("dte"))
        cal_d = max(td, 1) if td > 0 else None
    if iv_atm is not None and iv_atm > 0 and cal_d is not None and cal_d > 0:
        move_pct = iv_atm * math.sqrt(float(cal_d) / 365.0) * 100.0
        if move_pct > 0:
            logger.info(
                "event_jump_implied_move_pct: ATM IV * sqrt(DTE/365) fallback expiry=%s cal_d=%s move=%.2f%%",
                exp_s,
                cal_d,
                move_pct,
            )
            return float(move_pct)
    return None


def _as_of_date_from_options_chain(oc_data: dict[str, Any]) -> date | None:
    as_of_s = str(oc_data.get("as_of") or "")
    if len(as_of_s) >= 10:
        try:
            return date.fromisoformat(as_of_s[:10])
        except ValueError:
            return None
    return None


def _available_expiry_dates_from_oc(oc_data: dict[str, Any]) -> list[date]:
    raw = oc_data.get("available_expiries") or []
    return _parse_expiry_list(tuple(str(x) for x in raw))


def _selected_row_for_calendar_expiry(oc_data: dict[str, Any], exp: date) -> dict[str, Any] | None:
    key = exp.isoformat()
    for row in oc_data.get("selected_expiries") or []:
        if isinstance(row, dict) and str(row.get("expiry_date") or "")[:10] == key:
            return row
    return None


def _year_fraction_act365(as_of: date, expiry: date) -> float:
    """Calendar year fraction for variance-time (ACT/365), floored for numerical stability."""
    if expiry <= as_of:
        return 1.0 / 365.0
    return max((expiry - as_of).days / 365.0, 1.0 / 365.0)


def compute_event_only_implied_move_bundle(
    oc_data: dict[str, Any],
    *,
    earnings_date: str,
    event_jump_pct: float,
    daily_vol_pct: float,
    diffusion_sessions_multiplier: float = 1.0,
    lit_straddle_move_pct: float | None = None,
    expiry_class: str | None = None,
) -> dict[str, Any]:
    """Zero-to-one **trading session** earnings implied-move estimate vs straddle-through-expiry ``event_jump``.

    Primary: practitioner forward implied variance between listed ``T1`` (latest expiry **before**
    earnings) and ``T2`` (first expiry **on or after** earnings), using ATM IVs and ACT/365 year
    fractions, then ``sigma_fwd * sqrt(1/252)`` as a one-session percent move (aligned with HV30/sqrt(252)
    ``daily_vol`` units in this repo). Fallback: ``max(0, event_jump_pct - k*daily_vol_pct*sqrt(n))``
    with ``n`` = trading sessions from ``as_of`` through ``T2`` (same ``dte`` convention as snapshots).

    When ``expiry_class`` is ``monthly`` and forward variance is unavailable, uses a variance residual
    in percent space: ``sqrt(max(0, (L/100)^2 - n*(dv/100)^2)) / sqrt(n) * 100`` with ``L`` = literal straddle
    move (``lit_straddle_move_pct``) when provided.

    Every return includes ``event_only_implied_move_method`` and ``event_only_implied_move_reason``;
    ``event_only_implied_move_pct`` may be ``null`` only with an explanatory reason.
    """
    out: dict[str, Any] = {
        "event_only_implied_move_pct": None,
        "event_only_implied_move_method": "unavailable",
        "event_only_implied_move_reason": "",
        "event_only_implied_move_T1_expiry": None,
        "event_only_implied_move_T2_expiry": None,
    }
    if not oc_data.get("options_chain_available"):
        out["event_only_implied_move_reason"] = "options_chain_available is false"
        return out
    earn = _parse_earnings_calendar_date(earnings_date)
    if earn is None:
        out["event_only_implied_move_reason"] = f"could not parse earnings_date={earnings_date!r}"
        return out
    avail = _available_expiry_dates_from_oc(oc_data)
    if not avail:
        out["event_only_implied_move_reason"] = "available_expiries is empty on chain payload"
        return out
    as_of_d = _as_of_date_from_options_chain(oc_data)
    if as_of_d is None:
        out["event_only_implied_move_reason"] = (
            "could not parse as_of calendar date from options_chain_data.as_of (need YYYY-MM-DD prefix)"
        )
        return out

    t1, t2 = _forward_variance_expiry_bracket(avail, earn)
    out["event_only_implied_move_T1_expiry"] = t1.isoformat() if t1 else None
    out["event_only_implied_move_T2_expiry"] = t2.isoformat() if t2 else None

    if t1 is not None and t2 is not None and t1 < t2:
        row1 = _selected_row_for_calendar_expiry(oc_data, t1)
        row2 = _selected_row_for_calendar_expiry(oc_data, t2)
        iv1 = _atm_iv_proxy_ex(row1) if row1 else None
        iv2 = _atm_iv_proxy_ex(row2) if row2 else None
        ty1 = _year_fraction_act365(as_of_d, t1)
        ty2 = _year_fraction_act365(as_of_d, t2)
        if (
            row1 is not None
            and row2 is not None
            and iv1 is not None
            and iv2 is not None
            and iv1 > 0
            and iv2 > 0
            and ty2 > ty1
        ):
            denom = ty2 - ty1
            var_fwd = (iv2 * iv2 * ty2 - iv1 * iv1 * ty1) / denom
            if var_fwd > 0.0 and math.isfinite(var_fwd):
                sigma_fwd = math.sqrt(var_fwd)
                move_pct = sigma_fwd * math.sqrt(1.0 / 252.0) * 100.0
                if move_pct > 0.0 and math.isfinite(move_pct):
                    out["event_only_implied_move_pct"] = float(move_pct)
                    out["event_only_implied_move_method"] = "forward_variance"
                    out["event_only_implied_move_reason"] = (
                        f"sigma_fwd from ATM IV variance swap approx: T1={t1.isoformat()} IV={iv1:.4f} "
                        f"Tyr={ty1:.5f}, T2={t2.isoformat()} IV={iv2:.4f} Tyr={ty2:.5f}; "
                        f"one-session move = sigma_fwd*sqrt(1/252) in percent"
                    )
                    return out
        out["event_only_implied_move_reason"] = (
            "forward_variance not used: missing selected_expiries rows or ATM IV for T1/T2, "
            "or non-positive / non-finite implied forward variance"
        )

    if t2 is None:
        out["event_only_implied_move_reason"] = (
            "no listed expiry on or after earnings calendar date; cannot anchor straddle window"
        )
        return out
    row_t2 = _selected_row_for_calendar_expiry(oc_data, t2)
    n_sess = _coerce_int(row_t2.get("dte")) if row_t2 else 0
    if n_sess <= 0:
        n_sess = trading_sessions_after_exclusive(as_of_d, t2)
    if daily_vol_pct < 0.0 or not math.isfinite(daily_vol_pct):
        out["event_only_implied_move_reason"] = "daily_vol_pct invalid for straddle_minus_diffusion fallback"
        return out

    lit_ej = float(lit_straddle_move_pct) if lit_straddle_move_pct is not None else float(event_jump_pct)
    if str(expiry_class or "") == "monthly" and lit_straddle_move_pct is not None and lit_ej > 0.0 and n_sess > 0:
        ve = (lit_ej / 100.0) ** 2 - float(n_sess) * (float(daily_vol_pct) / 100.0) ** 2
        if ve > 0.0 and math.isfinite(ve):
            one_sess = math.sqrt(ve) / math.sqrt(float(n_sess)) * 100.0
            if one_sess > 0.0 and math.isfinite(one_sess):
                out["event_only_implied_move_pct"] = float(one_sess)
                out["event_only_implied_move_method"] = "monthly_straddle_minus_diffusion"
                out["event_only_implied_move_reason"] = (
                    f"monthly thin chain: sqrt(max(0,(L/100)^2-n*(dv/100)^2))/sqrt(n)*100 with L={lit_ej:.4f}% "
                    f"(literal straddle), n={n_sess}, dv={daily_vol_pct:.4f}%/day"
                )
                return out

    k = float(diffusion_sessions_multiplier)
    diff_term = k * float(daily_vol_pct) * math.sqrt(float(max(n_sess, 0)))
    residual = float(event_jump_pct) - diff_term
    capped = max(0.0, residual)
    out["event_only_implied_move_pct"] = float(capped)
    out["event_only_implied_move_method"] = "straddle_minus_diffusion"
    out["event_only_implied_move_reason"] = (
        f"max(0, event_jump_pct ({event_jump_pct:.4f}%) - {k:.4f}*daily_vol_pct ({daily_vol_pct:.4f}%)"
        f"*sqrt(n)) with n={n_sess} trading sessions from as_of through T2={t2.isoformat()} (chain dte when present)"
    )
    return out


def _failed_options_prompt_dict(symbol_u: str, fetch_error: str) -> dict[str, Any]:
    return {
        "options_chain_available": False,
        "as_of": _iso_utc_z_now(),
        "symbol": symbol_u.strip().upper(),
        "spot": None,
        "available_expiries": [],
        "selected_expiries": [],
        "fetch_error": fetch_error,
    }


@functools.lru_cache(maxsize=64)
def _fetch_options_chain_snapshot_cached(
    symbol_u: str,
    earnings_key: str,
    today_key: str,
    targets_joined: str,
    lookahead_key: str,
) -> str:
    """JSON-serialized successful :meth:`OptionsChainSnapshot.to_prompt_dict` (verified chain only).

    Soft failures raise :class:`OptionsChainSoftFetchError` so they are **not** cached.
    """
    _ = targets_joined  # cache key segment for future busting
    _ = lookahead_key
    try:
        look_n = int(lookahead_key)
    except ValueError:
        look_n = 14
    data = _fetch_options_chain_snapshot_impl(symbol_u, earnings_key, today_key, max_weekly_lookahead_days=look_n)
    if not data.get("options_chain_available"):
        raise OptionsChainSoftFetchError(data)
    import json

    return json.dumps(data, sort_keys=True, default=str)


def _as_of_session_date(today_key: str) -> date:
    """Map ``today_date`` / cache key to a calendar date for DTE math (best-effort)."""
    if not today_key:
        return date.today()
    parsed = _parse_today_date(today_key)
    if parsed is not None:
        return parsed
    try:
        return date.fromisoformat(str(today_key)[:10])
    except ValueError:
        return date.today()


def _fetch_options_chain_snapshot_impl(
    symbol_u: str,
    earnings_key: str,
    today_key: str,
    *,
    max_weekly_lookahead_days: int = 14,
) -> dict[str, Any]:
    """Build a ``to_prompt_dict``-shaped mapping; ``options_chain_available`` may be false with ``fetch_error``."""
    earnings = _parse_earnings_calendar_date(earnings_key)
    if earnings is None:
        logger.warning("options_chain: could not parse earnings_date=%r", earnings_key)
        return _failed_options_prompt_dict(symbol_u, "could not parse earnings_date from config")
    as_of = _as_of_session_date(today_key)
    try:
        import yfinance as yf  # type: ignore[import-untyped]
    except ImportError as exc:
        logger.warning("options_chain: yfinance not installed: %r", exc)
        return _failed_options_prompt_dict(symbol_u, f"yfinance not installed: {exc!r}")
    try:
        ticker = yf.Ticker(symbol_u)
        opts = getattr(ticker, "options", None)
        if not opts:
            logger.warning("options_chain: no .options for symbol=%s", symbol_u)
            return _failed_options_prompt_dict(
                symbol_u,
                "yfinance returned no option expiries for this ticker",
            )
        available_dates = _parse_expiry_list(tuple(opts))
        if not available_dates:
            logger.warning("options_chain: empty parsed expiries symbol=%s", symbol_u)
            return _failed_options_prompt_dict(
                symbol_u,
                "parsed Yahoo option expiries list empty after fetch",
            )
        spot = _resolve_spot(ticker)
        date_reasons: dict[date, str] = {}
        for ed, reason in _select_relevant_expiries_with_rationale(available_dates, earnings):
            date_reasons[ed] = reason
        t1_br, t2_br = _forward_variance_expiry_bracket(available_dates, earnings)
        if t1_br is not None and t1_br not in date_reasons:
            date_reasons[t1_br] = (
                "forward-variance bracket: latest listed expiry strictly before earnings calendar date"
            )
        if t2_br is not None and t2_br not in date_reasons:
            date_reasons[t2_br] = (
                "forward-variance bracket: first listed expiry on or after earnings calendar date"
            )
        look = max(1, int(max_weekly_lookahead_days))
        fe, fe_class, _fe_note = pick_front_listed_expiry_for_earnings(
            available_dates,
            earnings,
            max_weekly_lookahead_days=look,
        )
        if fe is not None:
            date_reasons.setdefault(
                fe,
                f"front contract for event_jump / sigma ladder ({fe_class}, lookahead_days={look})",
            )
        ex_snaps: list[dict[str, Any]] = []
        for ed in sorted(date_reasons.keys()):
            reason = date_reasons[ed]
            try:
                chain = ticker.option_chain(ed.isoformat())
            except Exception as exc:
                logger.warning(
                    "options_chain: option_chain failed symbol=%s expiry=%s err=%r",
                    symbol_u,
                    ed,
                    exc,
                )
                continue
            snap = _build_expiry_snapshot(
                expiry_d=ed,
                as_of=as_of,
                spot=spot,
                calls=chain.calls,
                puts=chain.puts,
                selection_rationale=reason,
            )
            if snap is not None:
                ex_snaps.append(asdict(snap))
        out_snap = OptionsChainSnapshot(
            as_of=_iso_utc_z_now(),
            symbol=symbol_u,
            spot=spot,
            available_expiries=[d.isoformat() for d in available_dates],
            selected_expiries=[_expiry_snapshot_from_row_dict(x) for x in ex_snaps],
        )
        data = out_snap.to_prompt_dict()
        if not data.get("options_chain_available"):
            return _failed_options_prompt_dict(
                symbol_u,
                "listed expiries present but no ATM snapshot rows could be built (spot or chain data missing)",
            )
        return data
    except Exception as exc:
        logger.warning("options_chain: yfinance fetch failed symbol=%s err=%r", symbol_u, exc)
        return _failed_options_prompt_dict(symbol_u, f"yfinance error: {exc!r}")


def fetch_options_chain_prompt_dict(
    symbol: str,
    earnings_date: str,
    target_dates: list[str],
    *,
    today_date: str | None = None,
    max_weekly_lookahead_days: int = 14,
) -> dict[str, Any]:
    """Yahoo option chain as ``to_prompt_dict`` shape, including soft failures with ``fetch_error``."""
    sym = symbol.strip().upper()
    if not sym:
        return _failed_options_prompt_dict("", "empty symbol")
    td = today_date or date.today().isoformat()
    if today_date:
        parsed = _parse_today_date(today_date)
        if parsed is not None:
            td = parsed.isoformat()
    tj = ",".join(target_dates)
    lk = str(int(max_weekly_lookahead_days))
    try:
        blob = _fetch_options_chain_snapshot_cached(sym, earnings_date.strip(), td, tj, lk)
    except OptionsChainSoftFetchError as exc:
        return exc.payload
    import json

    return cast(dict[str, Any], json.loads(blob))


def fetch_options_chain_snapshot(
    symbol: str,
    earnings_date: str,
    target_dates: list[str],
    *,
    today_date: str | None = None,
) -> OptionsChainSnapshot | None:
    """Pull Yahoo option chains via yfinance; returns ``None`` on soft failure (same spirit as auto_fetch)."""
    data = fetch_options_chain_prompt_dict(symbol, earnings_date, target_dates, today_date=today_date)
    if not data.get("options_chain_available"):
        return None
    return options_chain_snapshot_from_prompt_dict(data)


def options_chain_snapshot_from_prompt_dict(data: dict[str, Any]) -> OptionsChainSnapshot | None:
    """Hydrate :class:`OptionsChainSnapshot` from :meth:`OptionsChainSnapshot.to_prompt_dict` output."""
    try:
        rows = data.get("selected_expiries") or []
        snaps = [_expiry_snapshot_from_row_dict(dict(r)) for r in rows if isinstance(r, dict)]
        return OptionsChainSnapshot(
            as_of=str(data.get("as_of") or _iso_utc_z_now()),
            symbol=str(data.get("symbol") or ""),
            spot=_coerce_float(data.get("spot")),
            available_expiries=[str(x) for x in (data.get("available_expiries") or [])],
            selected_expiries=snaps,
            fetch_error=str(data["fetch_error"]) if data.get("fetch_error") else None,
        )
    except (TypeError, ValueError, KeyError) as exc:
        logger.warning("options_chain: invalid manual/options dict err=%r", exc)
        return None


def empty_options_prompt_dict(symbol: str) -> dict[str, Any]:
    return {
        "options_chain_available": False,
        "as_of": _iso_utc_z_now(),
        "symbol": symbol.strip().upper(),
        "spot": None,
        "available_expiries": [],
        "selected_expiries": [],
        "fetch_error": None,
    }


def clear_options_chain_cache() -> None:
    """Test helper: reset in-process LRU cache."""
    _fetch_options_chain_snapshot_cached.cache_clear()
