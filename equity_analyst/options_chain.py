from __future__ import annotations

import functools
import logging
import math
import re
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, cast

logger = logging.getLogger(__name__)

ISO_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


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


def _select_relevant_expiries(available: list[date], earnings: date) -> list[date]:
    """Pick earnings-week, pre-earnings same-week, T+1w-ish Friday, and nearest monthly (3rd Fri)."""
    if not available:
        return []
    chosen: list[date] = []
    seen: set[date] = set()

    def _add(d: date) -> None:
        if d in seen:
            return
        seen.add(d)
        chosen.append(d)

    # 1) Nearest expiry on or after earnings (earnings-week contract).
    post = [d for d in available if d >= earnings]
    if post:
        _add(min(post))

    # 2) Nearest expiry before earnings in the same ISO week as earnings (pre-earnings chain).
    w0 = _week_start_monday(earnings)
    pre_same_week = [d for d in available if earnings > d >= w0]
    if pre_same_week:
        _add(max(pre_same_week))

    # 3) Following Friday ~T+1 week (closest listed expiry to earnings + 7d, after earnings).
    target = earnings + timedelta(days=7)
    after_earn = [d for d in available if d > earnings]
    if after_earn:
        _add(min(after_earn, key=lambda d: (abs((d - target).days), d)))

    # 4) Monthly (3rd Friday) closest to earnings.
    thirds = [d for d in available if _is_third_friday(d)]
    if thirds:
        _add(min(thirds, key=lambda d: (abs((d - earnings).days), d)))

    # If nothing matched (odd chain), fall back to closest overall.
    if not chosen:
        _add(min(available, key=lambda d: (abs((d - earnings).days), d)))
    return chosen


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

    def to_markdown_table(self) -> str:
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
        return "".join(lines)


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
    return ExpirySnapshot(**d)


@functools.lru_cache(maxsize=64)
def _fetch_options_chain_snapshot_cached(
    symbol_u: str,
    earnings_key: str,
    today_key: str,
    targets_joined: str,
) -> str:
    """JSON-serialized :meth:`OptionsChainSnapshot.to_prompt_dict` or empty string if fetch failed."""
    _ = targets_joined  # cache key segment for future busting
    snap = _fetch_options_chain_snapshot_impl(symbol_u, earnings_key, today_key)
    import json

    return json.dumps(snap, sort_keys=True, default=str) if snap is not None else ""


def _fetch_options_chain_snapshot_impl(symbol_u: str, earnings_key: str, today_key: str) -> dict[str, Any] | None:
    earnings = _parse_earnings_calendar_date(earnings_key)
    as_of = date.fromisoformat(today_key) if today_key else date.today()
    if earnings is None:
        logger.warning("options_chain: could not parse earnings_date=%r", earnings_key)
        return None
    try:
        import yfinance as yf  # type: ignore[import-untyped]
    except ImportError as exc:
        logger.warning("options_chain: yfinance not installed: %r", exc)
        return None
    try:
        ticker = yf.Ticker(symbol_u)
        opts = getattr(ticker, "options", None)
        if not opts:
            logger.warning("options_chain: no .options for symbol=%s", symbol_u)
            return None
        available_dates = _parse_expiry_list(tuple(opts))
        if not available_dates:
            logger.warning("options_chain: empty parsed expiries symbol=%s", symbol_u)
            return None
        spot = _resolve_spot(ticker)
        selected_d = _select_relevant_expiries(available_dates, earnings)
        ex_snaps: list[dict[str, Any]] = []
        for ed in selected_d:
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
        return out_snap.to_prompt_dict()
    except Exception as exc:
        logger.warning("options_chain: yfinance fetch failed symbol=%s err=%r", symbol_u, exc)
        return None


def fetch_options_chain_snapshot(
    symbol: str,
    earnings_date: str,
    target_dates: list[str],
    *,
    today_date: str | None = None,
) -> OptionsChainSnapshot | None:
    """Pull Yahoo option chains via yfinance; returns ``None`` on soft failure (same spirit as auto_fetch)."""
    sym = symbol.strip().upper()
    if not sym:
        return None
    td = today_date or date.today().isoformat()
    if today_date:
        parsed = _parse_today_date(today_date)
        if parsed is not None:
            td = parsed.isoformat()
    tj = ",".join(target_dates)
    blob = _fetch_options_chain_snapshot_cached(sym, earnings_date.strip(), td, tj)
    if not blob:
        return None
    import json

    data = json.loads(blob)
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
