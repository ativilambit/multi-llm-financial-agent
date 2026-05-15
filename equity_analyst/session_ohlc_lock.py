"""Post-close regular-session OHLC lock for ``runs`` rows (yfinance daily bar)."""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Any, cast
from zoneinfo import ZoneInfo

from sqlalchemy import func, select, update

from equity_analyst.db import get_async_session, is_db_available
from equity_analyst.db_models import RunRow

logger = logging.getLogger(__name__)

_NY = ZoneInfo("America/New_York")
_AFTER_CLOSE = time(16, 15)


@dataclass(frozen=True)
class SessionDailyOhlc:
    """Regular-session daily OHLC for one NYSE calendar session date."""

    open: float
    high: float
    low: float
    close: float


def _coerce_float(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        f = float(v)
        if f != f or f <= 0:
            return None
        return f
    return None


def ny_calendar_date_from_yfinance_index(ts: object) -> date:
    """Map a yfinance history index label to the **NYSE session calendar date**."""
    import pandas as pd  # type: ignore[import-untyped]

    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        return cast(date, t.tz_convert("America/New_York").date())
    return cast(date, t.date())


def pick_session_bar_from_history(
    df: Any,
    session_date: date,
) -> SessionDailyOhlc | None:
    """Pick the daily row whose NY calendar date equals ``session_date``."""
    if df is None or getattr(df, "empty", True):
        return None
    try:
        df_sorted = df.sort_index()
    except Exception as exc:
        logger.warning("session_ohlc_lock: could not sort yfinance frame error=%r", exc)
        return None
    try:
        for ts, row in df_sorted.iterrows():
            bar_d = ny_calendar_date_from_yfinance_index(ts)
            if bar_d != session_date:
                continue
            o = _coerce_float(row["Open"])
            h = _coerce_float(row["High"])
            lo = _coerce_float(row["Low"])
            c = _coerce_float(row["Close"])
            if o is None or h is None or lo is None or c is None:
                return None
            if h < lo:
                return None
            return SessionDailyOhlc(open=o, high=h, low=lo, close=c)
    except Exception as exc:
        logger.warning("session_ohlc_lock: could not scan yfinance frame error=%r", exc)
        return None
    return None


def fetch_session_daily_ohlc_yfinance(symbol: str, session_date: date) -> SessionDailyOhlc | None:
    """Fetch one regular-session daily bar for ``session_date`` (NYSE calendar) via yfinance."""
    sym = symbol.strip().upper()
    if not sym:
        return None
    start = session_date
    end = session_date + timedelta(days=5)
    try:
        import yfinance as yf  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        logger.warning("session_ohlc_lock: yfinance not installed: %r", exc)
        return None
    df = None
    try:
        ticker = yf.Ticker(sym)
        df = ticker.history(start=start.isoformat(), end=end.isoformat(), auto_adjust=False)
    except Exception as exc:
        logger.warning(
            "session_ohlc_lock: yfinance.history failed symbol=%s session_date=%s error=%r",
            sym,
            session_date,
            exc,
        )
        return None
    return pick_session_bar_from_history(df, session_date)


def default_session_date_before_today_ny(*, now_utc: datetime) -> date:
    """Previous calendar day in New York, then walk back over weekends (not holidays)."""
    ny_today = now_utc.astimezone(_NY).date()
    d = ny_today - timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def gha_auto_skip_reason(*, now_utc: datetime | None = None) -> str | None:
    """If scheduled auto-lock should no-op, return a short reason; else ``None``."""
    now = now_utc or datetime.now(tz=UTC)
    et = now.astimezone(_NY)
    if et.weekday() >= 5:
        return "weekend America/New_York"
    if et.time() < _AFTER_CLOSE:
        return f"before {_AFTER_CLOSE.isoformat(timespec='minutes')} America/New_York"
    return None


def parse_symbol_csv(raw: str | None) -> list[str]:
    if not raw or not str(raw).strip():
        return []
    return [p.strip().upper() for p in str(raw).split(",") if p.strip()]


def parse_run_id_csv(raw: str | None) -> list[str]:
    if not raw or not str(raw).strip():
        return []
    return [p.strip() for p in str(raw).split(",") if p.strip()]


def symbols_from_env() -> list[str]:
    import os

    return parse_symbol_csv(os.environ.get("EQUITY_OHLC_LOCK_SYMBOLS"))


async def resolve_run_ids_for_symbol_session_day(
    *,
    symbol: str,
    session_date: date,
    database_url: str | None,
) -> list[str]:
    """Runs for symbol with ``created_at_utc`` on ``session_date`` NY wall clock, not yet locked."""
    sym_u = symbol.strip().upper()
    ny_day = func.date(func.timezone("America/New_York", RunRow.created_at_utc))
    async with get_async_session(database_url=database_url) as session:
        res = await session.execute(
            select(RunRow.run_id).where(
                func.upper(RunRow.symbol) == sym_u,
                ny_day == session_date,
                RunRow.session_close.is_(None),
            )
        )
        return [str(r[0]) for r in res.fetchall()]


async def load_run_symbols(
    run_ids: list[str],
    *,
    database_url: str | None,
) -> dict[str, str]:
    """Map ``run_id`` -> ``symbol`` for existing rows."""
    if not run_ids:
        return {}
    async with get_async_session(database_url=database_url) as session:
        res = await session.execute(
            select(RunRow.run_id, RunRow.symbol).where(RunRow.run_id.in_(run_ids))
        )
        return {str(r[0]): str(r[1]) for r in res.fetchall()}


async def apply_session_ohlc_to_run_ids(
    *,
    run_ids: list[str],
    session_date: date,
    ohlc: SessionDailyOhlc,
    session_partial: bool,
    session_source: str,
    database_url: str | None,
    dry_run: bool,
) -> int:
    if not run_ids:
        return 0
    now = datetime.now(tz=UTC)
    values: dict[str, Any] = {
        "session_trade_date": session_date,
        "session_open": ohlc.open,
        "session_high": ohlc.high,
        "session_low": ohlc.low,
        "session_close": ohlc.close,
        "session_partial": session_partial,
        "session_snapshot_at_utc": now,
        "session_source": session_source,
        "updated_at_utc": now,
    }
    if dry_run:
        sys.stdout.write(
            "[dry-run] UPDATE runs SET "
            + ", ".join(f"{k}=%({k})s" for k in values)
            + f" WHERE run_id IN ({', '.join('%s' for _ in run_ids)});\n"
        )
        sys.stdout.write(f"[dry-run] params: {values} run_ids={run_ids!r}\n")
        return len(run_ids)
    async with get_async_session(database_url=database_url) as session:
        await session.execute(update(RunRow).where(RunRow.run_id.in_(run_ids)).values(**values))
        await session.commit()
    return len(run_ids)


async def run_lock_session_ohlc_cli(args: argparse.Namespace) -> int:
    """CLI entry for ``lock-session-ohlc``."""
    database_url = getattr(args, "database_url", None)
    if not database_url:
        import os

        database_url = os.environ.get("DATABASE_URL")

    dry_run = bool(args.dry_run)
    if not await is_db_available(database_url=database_url):
        raise SystemExit(
            "lock-session-ohlc: DATABASE_URL is unreachable. "
            "Set DATABASE_URL or pass --database-url (required even with --dry-run to resolve targets)."
        )

    gha = bool(args.gha_auto)
    run_ids = parse_run_id_csv(args.run_id)
    symbols = parse_symbol_csv(args.symbols)
    if args.symbol:
        symbols = [*symbols, args.symbol.strip().upper()]
    symbols = list(dict.fromkeys(s for s in symbols if s))

    if args.symbols_env:
        symbols = [*symbols, *symbols_from_env()]
        symbols = list(dict.fromkeys(s for s in symbols if s))

    session_date_raw = args.date

    if gha:
        skip = gha_auto_skip_reason()
        if skip is not None:
            logger.info("lock-session-ohlc: gha-auto skip (%s)", skip)
            return 0
        session_d = datetime.now(tz=UTC).astimezone(_NY).date()
        if not symbols:
            logger.warning(
                "lock-session-ohlc: gha-auto needs symbols via --symbols/--symbol or "
                "EQUITY_OHLC_LOCK_SYMBOLS; nothing to do",
            )
            return 0
    elif run_ids:
        if not session_date_raw:
            raise SystemExit("lock-session-ohlc: --date is required with --run-id")
        session_d = date.fromisoformat(str(session_date_raw))
    elif symbols:
        if session_date_raw:
            session_d = date.fromisoformat(str(session_date_raw))
        else:
            session_d = default_session_date_before_today_ny(now_utc=datetime.now(tz=UTC))
    else:
        raise SystemExit(
            "lock-session-ohlc: provide --run-id ... --date ... and/or "
            "--symbol(s) [--date], or use --gha-auto with symbols.",
        )

    session_partial = bool(args.session_partial)
    if gha:
        session_partial = False

    work_run_ids: list[str] = []
    if run_ids:
        work_run_ids = list(dict.fromkeys(run_ids))
        known = await load_run_symbols(work_run_ids, database_url=database_url)
        missing = [r for r in work_run_ids if r not in known]
        if missing:
            raise SystemExit(f"lock-session-ohlc: unknown run_id(s): {missing!r}")
    elif symbols or gha:
        for sym in symbols:
            found = await resolve_run_ids_for_symbol_session_day(
                symbol=sym, session_date=session_d, database_url=database_url
            )
            if not found:
                logger.warning(
                    "lock-session-ohlc: no unlocked runs for symbol=%s session_date=%s",
                    sym,
                    session_d.isoformat(),
                )
            work_run_ids.extend(found)
        work_run_ids = list(dict.fromkeys(work_run_ids))

    if not work_run_ids:
        logger.info("lock-session-ohlc: no run_ids matched; exiting")
        return 0

    rid_to_sym = await load_run_symbols(work_run_ids, database_url=database_url)
    by_sym: defaultdict[str, list[str]] = defaultdict(list)
    for rid in work_run_ids:
        by_sym[rid_to_sym[rid].strip().upper()].append(rid)

    for sym, rids in sorted(by_sym.items()):
        ohlc = fetch_session_daily_ohlc_yfinance(sym, session_d)
        if ohlc is None:
            logger.error(
                "lock-session-ohlc: missing yfinance bar symbol=%s session_date=%s",
                sym,
                session_d.isoformat(),
            )
            continue
        n = await apply_session_ohlc_to_run_ids(
            run_ids=rids,
            session_date=session_d,
            ohlc=ohlc,
            session_partial=session_partial,
            session_source="yfinance",
            database_url=database_url,
            dry_run=dry_run,
        )
        logger.info(
            "lock-session-ohlc: symbol=%s session_date=%s updated_runs=%d dry_run=%s",
            sym,
            session_d.isoformat(),
            n,
            dry_run,
        )
    return 0
