from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import re
import statistics
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict

from equity_analyst.config import RunProfile, run_profile_from_persisted_run_json

logger = logging.getLogger(__name__)


class RunOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_output_dir: str  # absolute path to outputs/<SYM>_<TS>Z/
    symbol: str
    recorded_at_utc: str  # ISO8601 Z
    earnings_date: str  # from run.json / config snapshot

    synthesis_path: str
    run_json_path: str

    earnings_day_open: float | None = None
    earnings_day_high: float | None = None
    earnings_day_low: float | None = None
    earnings_day_close: float | None = None
    next_trading_day_open: float | None = None
    next_trading_day_close: float | None = None
    one_week_later_close: float | None = None
    direction_vs_prior_close: Literal["up", "down", "flat"] | None = None
    notes: str | None = None
    source: Literal["manual", "yahoo_csv", "alpaca", "polygon"] = "manual"

    baseline_close_hint: float | None = None


def _iso_utc_z_now() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _infer_repo_root_from_run_dir(run_dir: Path) -> Path:
    if run_dir.parent.name != "outputs":
        raise ValueError(
            f"run_dir must be inside an outputs/ folder (got {str(run_dir)!r}, parent={run_dir.parent.name!r})"
        )
    return run_dir.parent.parent.resolve()


def _pick_synthesis_path(run_dir: Path) -> Path:
    direct = run_dir / "synthesis.md"
    if direct.is_file():
        return direct

    it_dir = run_dir / "iterations"
    if not it_dir.is_dir():
        return direct

    best: tuple[int, Path] | None = None
    for p in it_dir.glob("iteration_*_synthesis.md"):
        stem = p.name.replace("iteration_", "").replace("_synthesis.md", "")
        try:
            n = int(stem)
        except ValueError:
            continue
        if best is None or n > best[0]:
            best = (n, p)
    return best[1] if best is not None else direct


def _parse_baseline_close_hint(run_dir: Path) -> float | None:
    """
    Best-effort, non-brittle baseline hint for calibration.

    Currently reads RunConfig.current_price from run.json (YAML alias: reference_last_price).
    """
    run_json = run_dir / "run.json"
    if not run_json.is_file():
        return None
    try:
        data = json.loads(run_json.read_text(encoding="utf-8"))
    except Exception:
        return None
    cfg = data.get("config")
    if not isinstance(cfg, dict):
        return None
    v = cfg.get("current_price")
    try:
        return float(v) if v is not None else None
    except Exception:
        return None


def record_outcome(
    *,
    run_dir: Path,
    earnings_day_open: float | None = None,
    earnings_day_high: float | None = None,
    earnings_day_low: float | None = None,
    earnings_day_close: float | None = None,
    next_trading_day_open: float | None = None,
    next_trading_day_close: float | None = None,
    one_week_later_close: float | None = None,
    direction_vs_prior_close: Literal["up", "down", "flat"] | None = None,
    notes: str | None = None,
    source: Literal["manual", "yahoo_csv", "alpaca", "polygon"] = "manual",
    persist: bool = True,
) -> RunOutcome:
    run_dir = run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"--run-dir does not exist: {run_dir!s}")

    run_json = run_dir / "run.json"
    if not run_json.is_file():
        raise FileNotFoundError(f"Missing run.json at {run_json!s}")

    data = json.loads(run_json.read_text(encoding="utf-8"))
    cfg = data.get("config")
    if not isinstance(cfg, dict):
        raise ValueError("run.json missing config snapshot")

    symbol = cfg.get("symbol")
    earnings_date = cfg.get("earnings_date")
    if not symbol or not isinstance(symbol, str):
        raise ValueError("run.json config snapshot missing symbol")
    if not earnings_date or not isinstance(earnings_date, str):
        raise ValueError("run.json config snapshot missing earnings_date")

    synthesis_path = _pick_synthesis_path(run_dir)
    repo_root = _infer_repo_root_from_run_dir(run_dir)

    outcome = RunOutcome(
        run_output_dir=str(run_dir),
        symbol=symbol,
        recorded_at_utc=_iso_utc_z_now(),
        earnings_date=earnings_date,
        synthesis_path=str(synthesis_path),
        run_json_path=str(run_json),
        earnings_day_open=earnings_day_open,
        earnings_day_high=earnings_day_high,
        earnings_day_low=earnings_day_low,
        earnings_day_close=earnings_day_close,
        next_trading_day_open=next_trading_day_open,
        next_trading_day_close=next_trading_day_close,
        one_week_later_close=one_week_later_close,
        direction_vs_prior_close=direction_vs_prior_close,
        notes=notes,
        source=source,
        baseline_close_hint=_parse_baseline_close_hint(run_dir),
    )

    if persist:
        outcome_path = run_dir / "outcome.json"
        outcome_path.write_text(
            json.dumps(outcome.model_dump(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        outputs_dir = repo_root / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        registry_path = outputs_dir / "outcomes_registry.jsonl"
        with registry_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(outcome.model_dump(), sort_keys=True) + "\n")

    return outcome


@dataclass(frozen=True)
class RecordOutcomeForRunDirResult:
    """Result of :func:`record_outcome_for_run_dir` for CLI / batch reporting."""

    outcome: RunOutcome
    auto_fetch_used: bool
    yfinance_empty: bool
    auto_fetch_partial: bool


def _outcome_auto_fetch_field_coverage(outcome: RunOutcome) -> bool:
    """True when all auto-fetch OHLC window fields are non-null on the outcome."""
    vals = [
        outcome.earnings_day_open,
        outcome.earnings_day_high,
        outcome.earnings_day_low,
        outcome.earnings_day_close,
        outcome.next_trading_day_open,
        outcome.next_trading_day_close,
        outcome.one_week_later_close,
    ]
    return all(v is not None for v in vals)


def merge_auto_fetch_into_cli_fields(
    run_dir: Path,
    *,
    earnings_day_open: float | None = None,
    earnings_day_high: float | None = None,
    earnings_day_low: float | None = None,
    earnings_day_close: float | None = None,
    next_trading_day_open: float | None = None,
    next_trading_day_close: float | None = None,
    one_week_later_close: float | None = None,
    direction_vs_prior_close: Literal["up", "down", "flat"] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run yfinance merge for ``run_dir``; return ``(merged_fields, raw_fetched)``.

    Explicit arguments always win over fetched values. Used by the interactive
    ``outcome-record`` path before stdin prompts.
    """
    resolved_dir = run_dir.expanduser().resolve()
    symbol, earnings_date = read_run_metadata(resolved_dir)
    baseline_close = resolve_baseline_close_for_auto_fetch(resolved_dir)
    logger.info(
        "auto_fetch: starting symbol=%s earnings_date=%s baseline_close=%s",
        symbol,
        earnings_date,
        baseline_close,
    )
    fetched = auto_fetch_outcome(
        symbol=symbol,
        earnings_date=earnings_date,
        baseline_close=baseline_close,
    )
    merged = {
        "earnings_day_open": earnings_day_open
        if earnings_day_open is not None
        else fetched.get("earnings_day_open"),
        "earnings_day_high": earnings_day_high
        if earnings_day_high is not None
        else fetched.get("earnings_day_high"),
        "earnings_day_low": earnings_day_low
        if earnings_day_low is not None
        else fetched.get("earnings_day_low"),
        "earnings_day_close": earnings_day_close
        if earnings_day_close is not None
        else fetched.get("earnings_day_close"),
        "next_trading_day_open": next_trading_day_open
        if next_trading_day_open is not None
        else fetched.get("next_trading_day_open"),
        "next_trading_day_close": next_trading_day_close
        if next_trading_day_close is not None
        else fetched.get("next_trading_day_close"),
        "one_week_later_close": one_week_later_close
        if one_week_later_close is not None
        else fetched.get("one_week_later_close"),
        "direction_vs_prior_close": direction_vs_prior_close,
    }
    if merged["direction_vs_prior_close"] is None:
        fd = fetched.get("direction_vs_prior_close")
        if fd in ("up", "down", "flat"):
            merged["direction_vs_prior_close"] = fd
    return merged, fetched


def record_outcome_for_run_dir(
    *,
    run_dir: Path,
    auto_fetch: bool = False,
    dry_run: bool = False,
    earnings_day_open: float | None = None,
    earnings_day_high: float | None = None,
    earnings_day_low: float | None = None,
    earnings_day_close: float | None = None,
    next_trading_day_open: float | None = None,
    next_trading_day_close: float | None = None,
    one_week_later_close: float | None = None,
    direction_vs_prior_close: Literal["up", "down", "flat"] | None = None,
    notes: str | None = None,
    source: Literal["manual", "yahoo_csv", "alpaca", "polygon"] = "manual",
    db_upsert: bool = True,
) -> RecordOutcomeForRunDirResult:
    """Merge optional yfinance auto-fetch, persist outcome artifacts, best-effort DB upsert.

    This is the shared implementation used by ``outcome-record`` and ``outcome-record-batch``.
    Explicit field arguments always win over auto-fetched values. When ``dry_run`` is true,
    no files, registry lines, or DB writes are performed (values are still computed).
    """
    resolved_dir = run_dir.expanduser().resolve()
    yfinance_empty = False
    raw_fetched: dict[str, Any] = {}

    if auto_fetch:
        merged, raw_fetched = merge_auto_fetch_into_cli_fields(
            run_dir,
            earnings_day_open=earnings_day_open,
            earnings_day_high=earnings_day_high,
            earnings_day_low=earnings_day_low,
            earnings_day_close=earnings_day_close,
            next_trading_day_open=next_trading_day_open,
            next_trading_day_close=next_trading_day_close,
            one_week_later_close=one_week_later_close,
            direction_vs_prior_close=direction_vs_prior_close,
        )
        earnings_day_open = merged["earnings_day_open"]
        earnings_day_high = merged["earnings_day_high"]
        earnings_day_low = merged["earnings_day_low"]
        earnings_day_close = merged["earnings_day_close"]
        next_trading_day_open = merged["next_trading_day_open"]
        next_trading_day_close = merged["next_trading_day_close"]
        one_week_later_close = merged["one_week_later_close"]
        direction_vs_prior_close = merged["direction_vs_prior_close"]
        yfinance_empty = all(raw_fetched.get(k) is None for k in _AUTO_FETCH_FIELDS)

    persist = not dry_run
    outcome = record_outcome(
        run_dir=run_dir,
        earnings_day_open=earnings_day_open,
        earnings_day_high=earnings_day_high,
        earnings_day_low=earnings_day_low,
        earnings_day_close=earnings_day_close,
        next_trading_day_open=next_trading_day_open,
        next_trading_day_close=next_trading_day_close,
        one_week_later_close=one_week_later_close,
        direction_vs_prior_close=direction_vs_prior_close,
        notes=notes,
        source=source,
        persist=persist,
    )

    coverage = _outcome_auto_fetch_field_coverage(outcome)
    auto_fetch_partial = bool(auto_fetch and not coverage)

    if persist and db_upsert:
        persisted_profile: RunProfile = "production"
        try:
            rj = resolved_dir / "run.json"
            if rj.is_file():
                persisted_profile = run_profile_from_persisted_run_json(
                    json.loads(rj.read_text(encoding="utf-8"))
                )
        except Exception:
            persisted_profile = "production"

        if persisted_profile != "production":
            logger.info("DB write skipped: run_profile=%s (not production)", persisted_profile)
        else:
            from equity_analyst.db_ops import best_effort_upsert_outcome

            with contextlib.suppress(Exception):
                asyncio.run(
                    best_effort_upsert_outcome(
                        cfg_db_enabled=True,
                        run_id=resolved_dir.name,
                        outcome=outcome.model_dump(),
                        database_url=None,
                        run_profile=persisted_profile,
                    )
                )

    return RecordOutcomeForRunDirResult(
        outcome=outcome,
        auto_fetch_used=auto_fetch,
        yfinance_empty=yfinance_empty,
        auto_fetch_partial=auto_fetch_partial,
    )


_AUTO_FETCH_FIELDS: tuple[str, ...] = (
    "earnings_day_open",
    "earnings_day_high",
    "earnings_day_low",
    "earnings_day_close",
    "next_trading_day_open",
    "next_trading_day_close",
    "one_week_later_close",
)

_DIRECTION_FLAT_TOLERANCE = 0.001  # ±0.1% relative move counts as flat


def _coerce_float_safe(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN guard
        return None
    return f


def _parse_earnings_date_fuzzy(earnings_date: str) -> datetime | None:
    """Parse a human-style earnings date string (``Mon May 11 2026``) into a UTC datetime.

    ``dateutil.parser`` is used in fuzzy mode so leading weekday abbreviations and stray
    punctuation are tolerated. Returns ``None`` on failure rather than raising.
    """
    try:
        from dateutil.parser import parse as _parse_date  # type: ignore[import-untyped]
    except ImportError:  # pragma: no cover - dateutil is a hard dep but stay defensive
        logger.warning(
            "auto_fetch: python-dateutil not installed; cannot parse earnings_date=%r",
            earnings_date,
        )
        return None
    try:
        dt_raw: Any = _parse_date(earnings_date, fuzzy=True)
    except (ValueError, TypeError) as exc:
        logger.warning(
            "auto_fetch: could not parse earnings_date=%r error=%r",
            earnings_date,
            exc,
        )
        return None
    if not isinstance(dt_raw, datetime):
        return None
    dt: datetime = dt_raw if dt_raw.tzinfo is not None else dt_raw.replace(tzinfo=UTC)
    return dt


def auto_fetch_outcome(
    symbol: str,
    earnings_date: str,
    baseline_close: float | None = None,
) -> dict[str, Any]:
    """Fetch earnings-day OHLC + next trading day OHLC + close ~5 trading days later.

    Returns a dict with keys matching the ``outcome.json`` schema
    (``earnings_day_open``/``_high``/``_low``/``_close``,
    ``next_trading_day_open``/``_close``, ``one_week_later_close``,
    ``direction_vs_prior_close``). Any field that could not be determined is set
    to ``None`` — the function never raises on partial data so it can be wired into
    a best-effort CLI flow.
    """
    result: dict[str, Any] = {f: None for f in _AUTO_FETCH_FIELDS}
    result["direction_vs_prior_close"] = None

    parsed = _parse_earnings_date_fuzzy(earnings_date)
    if parsed is None:
        return result
    start_date = parsed.date()
    # 15 calendar days is enough to cover the earnings bar + ~10 trading days.
    end_date = start_date + timedelta(days=15)

    try:
        import yfinance as yf  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - yfinance is a hard dep
        logger.warning("auto_fetch: yfinance not installed: %r", exc)
        return result

    df = None
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            auto_adjust=False,
        )
    except Exception as exc:
        logger.warning(
            "auto_fetch: yfinance.history call failed symbol=%s start=%s error=%r",
            symbol,
            start_date,
            exc,
        )
        return result

    if df is None or getattr(df, "empty", True):
        logger.warning(
            "auto_fetch: empty bars returned for symbol=%s start=%s (ADR/recent-IPO?)",
            symbol,
            start_date,
        )
        return result

    try:
        df_sorted = df.sort_index()
        bars = list(df_sorted.itertuples())
    except Exception as exc:
        logger.warning(
            "auto_fetch: could not iterate yfinance frame symbol=%s error=%r",
            symbol,
            exc,
        )
        return result

    if not bars:
        logger.warning("auto_fetch: zero bars after sort symbol=%s", symbol)
        return result

    earnings_bar = bars[0]
    result["earnings_day_open"] = _coerce_float_safe(getattr(earnings_bar, "Open", None))
    result["earnings_day_high"] = _coerce_float_safe(getattr(earnings_bar, "High", None))
    result["earnings_day_low"] = _coerce_float_safe(getattr(earnings_bar, "Low", None))
    result["earnings_day_close"] = _coerce_float_safe(
        getattr(earnings_bar, "Close", None)
    )

    if len(bars) >= 2:
        nb = bars[1]
        result["next_trading_day_open"] = _coerce_float_safe(getattr(nb, "Open", None))
        result["next_trading_day_close"] = _coerce_float_safe(getattr(nb, "Close", None))
    else:
        logger.warning(
            "auto_fetch: missing next-trading-day bar symbol=%s (only %d bars returned)",
            symbol,
            len(bars),
        )

    # 5 trading days after the earnings bar = the 6th bar (index 5).
    if len(bars) >= 6:
        result["one_week_later_close"] = _coerce_float_safe(getattr(bars[5], "Close", None))
    elif len(bars) >= 2:
        result["one_week_later_close"] = _coerce_float_safe(getattr(bars[-1], "Close", None))
        logger.warning(
            "auto_fetch: only %d bars after earnings symbol=%s; using last bar close as one_week_later_close",
            len(bars),
            symbol,
        )
    else:
        logger.warning(
            "auto_fetch: insufficient bars for one_week_later_close symbol=%s (len=%d)",
            symbol,
            len(bars),
        )

    if baseline_close is not None and result["earnings_day_close"] is not None:
        base = float(baseline_close)
        close = float(result["earnings_day_close"])
        if base > 0:
            rel = (close - base) / base
            if rel > _DIRECTION_FLAT_TOLERANCE:
                result["direction_vs_prior_close"] = "up"
            elif rel < -_DIRECTION_FLAT_TOLERANCE:
                result["direction_vs_prior_close"] = "down"
            else:
                result["direction_vs_prior_close"] = "flat"

    fetched = {k: result[k] for k in _AUTO_FETCH_FIELDS if result[k] is not None}
    missing = sorted(k for k in _AUTO_FETCH_FIELDS if result[k] is None)
    logger.info(
        "auto_fetch: symbol=%s earnings_date=%s fetched=%s missing=%s direction=%s",
        symbol,
        start_date,
        {k: round(v, 4) for k, v in fetched.items()},
        missing,
        result["direction_vs_prior_close"],
    )
    return result


def fetch_hv30_annualized_percent(symbol: str) -> float | None:
    """Return **annualized** 30-trading-day historical volatility as a **percentage** (e.g. ``50.0`` = 50% ann).

    Uses Yahoo Finance daily closes via yfinance: sample standard deviation of the last 30
    log returns, annualized with ``* √252``. Returns ``None`` on missing data or soft errors.
    """
    sym = symbol.strip().upper()
    if not sym:
        return None
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        logger.warning("fetch_hv30: yfinance not installed: %r", exc)
        return None
    try:
        ticker = yf.Ticker(sym)
        hist = ticker.history(period="6mo", auto_adjust=False)
    except Exception as exc:
        logger.warning("fetch_hv30: yfinance.history failed symbol=%s err=%r", sym, exc)
        return None
    if hist is None or getattr(hist, "empty", True):
        logger.warning("fetch_hv30: empty history symbol=%s", sym)
        return None
    try:
        closes = [float(x) for x in hist["Close"].dropna().tolist() if float(x) > 0]
    except Exception as exc:
        logger.warning("fetch_hv30: could not read closes symbol=%s err=%r", sym, exc)
        return None
    if len(closes) < 32:
        logger.warning("fetch_hv30: insufficient closes symbol=%s n=%d", sym, len(closes))
        return None
    tail = closes[-31:]
    log_rets: list[float] = []
    for i in range(1, len(tail)):
        a, b = tail[i - 1], tail[i]
        if a > 0 and b > 0:
            log_rets.append(math.log(b / a))
    if len(log_rets) < 30:
        return None
    window = log_rets[-30:]
    try:
        sd = statistics.stdev(window)
    except statistics.StatisticsError:
        return None
    if sd <= 0 or math.isnan(sd):
        return None
    return float(sd * math.sqrt(252) * 100.0)


def fetch_earnings_day_intraday_high_low_yfinance(
    symbol: str,
    earnings_date: str,
) -> tuple[float | None, float | None]:
    """Return regular-session **Low** and **High** for the earnings-day bar from Yahoo Finance.

    Reuses :func:`auto_fetch_outcome` (first trading bar on/after the parsed earnings calendar
    date). Returns ``(None, None)`` when either value is missing or invalid. Never raises.
    """
    try:
        res = auto_fetch_outcome(symbol, earnings_date)
    except Exception as exc:  # pragma: no cover - auto_fetch is defensive
        logger.warning(
            "earnings_day_intraday: auto_fetch failed symbol=%s earnings_date=%r error=%r",
            symbol,
            earnings_date,
            exc,
        )
        return None, None
    lo = _coerce_float_safe(res.get("earnings_day_low"))
    hi = _coerce_float_safe(res.get("earnings_day_high"))
    if lo is None or hi is None or lo <= 0 or hi <= 0 or hi < lo:
        return None, None
    return lo, hi


_LAST_CLOSE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"last\s+verified\s+close[^0-9$]*\$?\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
    re.compile(r"last\s+regular[- ]session\s+close[^0-9$]*\$?\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
    re.compile(r"last\s+close[^0-9$]*\$?\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
    re.compile(r"closing\s+price[^0-9$]*\$?\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
)


def fetch_prior_session_close_yfinance(symbol: str, earnings_date: date) -> float | None:
    """Return the regular-session close of the last trading day strictly before earnings_date."""
    start = earnings_date - timedelta(days=30)
    end = earnings_date
    try:
        import pandas as pd  # type: ignore[import-untyped]
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - yfinance is a hard dep
        logger.warning(
            "prior_session_baseline: yfinance/pandas not available symbol=%s error=%r",
            symbol,
            exc,
        )
        return None

    df = None
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start.isoformat(),
            end=end.isoformat(),
            auto_adjust=False,
        )
    except Exception as exc:
        logger.warning(
            "prior_session_baseline: yfinance.history failed symbol=%s earnings_date=%s error=%r",
            symbol,
            earnings_date,
            exc,
        )
        return None

    if df is None or getattr(df, "empty", True):
        return None

    def _bar_calendar_date_us(ts: object) -> date:
        t = pd.Timestamp(ts)
        if t.tzinfo is not None:
            return cast(date, t.tz_convert("America/New_York").date())
        return cast(date, t.date())

    try:
        df_sorted = df.sort_index()
    except Exception as exc:
        logger.warning(
            "prior_session_baseline: could not sort yfinance frame symbol=%s error=%r",
            symbol,
            exc,
        )
        return None

    last_bar_date: date | None = None
    last_close: float | None = None
    try:
        for ts, row in df_sorted.iterrows():
            bar_d = _bar_calendar_date_us(ts)
            if bar_d < earnings_date:
                c = _coerce_float_safe(row["Close"])
                if c is not None and c > 0:
                    last_bar_date = bar_d
                    last_close = c
    except Exception as exc:
        logger.warning(
            "prior_session_baseline: could not scan yfinance frame symbol=%s error=%r",
            symbol,
            exc,
        )
        return None

    if last_close is None or last_bar_date is None:
        return None

    logger.info(
        "baseline_close from prior session yfinance: %.2f (date %s)",
        last_close,
        last_bar_date.isoformat(),
    )
    return last_close


def _baseline_close_from_synthesis(run_dir: Path, *, max_chars: int = 8000) -> float | None:
    synthesis = _pick_synthesis_path(run_dir)
    if not synthesis.is_file():
        return None
    try:
        text = synthesis.read_text(encoding="utf-8")[:max_chars]
    except OSError:
        return None
    for pat in _LAST_CLOSE_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        try:
            return float(m.group(1))
        except ValueError:
            continue
    return None


def resolve_baseline_close_for_auto_fetch(run_dir: Path) -> float | None:
    """Best-effort baseline close for auto-fetch direction computation.

    Order of precedence: ``RunConfig.current_price`` from ``run.json`` (the same
    hint surfaced by ``RunOutcome.baseline_close_hint``), then a regex-extracted
    figure from the first ~8000 characters of ``synthesis.md`` (e.g. "last
    verified close" / "last regular-session close" / "closing price"), then the
    regular-session close of the last Yahoo daily bar strictly before the parsed
    ``earnings_date`` (see :func:`fetch_prior_session_close_yfinance`). Returns
    ``None`` when no source yields a positive number.
    """
    primary = _parse_baseline_close_hint(run_dir)
    if primary is not None and primary > 0:
        return primary
    fallback = _baseline_close_from_synthesis(run_dir)
    if fallback is not None and fallback > 0:
        return fallback
    try:
        symbol, earnings_date_str = read_run_metadata(run_dir)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    parsed_dt = _parse_earnings_date_fuzzy(earnings_date_str)
    if parsed_dt is None:
        return None
    earnings_d = parsed_dt.date()
    yf_baseline = fetch_prior_session_close_yfinance(symbol, earnings_d)
    if yf_baseline is not None and yf_baseline > 0:
        return yf_baseline
    return None


def read_run_metadata(run_dir: Path) -> tuple[str, str]:
    """Return ``(symbol, earnings_date)`` from ``run.json``; raises on missing fields."""
    run_json = run_dir / "run.json"
    if not run_json.is_file():
        raise FileNotFoundError(f"Missing run.json at {run_json!s}")
    data = json.loads(run_json.read_text(encoding="utf-8"))
    cfg = data.get("config")
    if not isinstance(cfg, dict):
        raise ValueError("run.json missing config snapshot")
    symbol = cfg.get("symbol")
    earnings_date = cfg.get("earnings_date")
    if not isinstance(symbol, str) or not symbol:
        raise ValueError("run.json config snapshot missing symbol")
    if not isinstance(earnings_date, str) or not earnings_date:
        raise ValueError("run.json config snapshot missing earnings_date")
    return symbol, earnings_date


_BATCH_OUTPUT_DIR_RE = re.compile(r"output_dir=(\S+)")


def parse_output_dirs_from_batch_summary(batch_summary_path: Path) -> list[Path]:
    """Parse ``batch_summary.txt`` lines for ``output_dir=<path>`` (runner / batch script)."""
    if not batch_summary_path.is_file():
        raise FileNotFoundError(f"missing batch summary file: {batch_summary_path}")
    seen: set[str] = set()
    ordered: list[Path] = []
    for line in batch_summary_path.read_text(encoding="utf-8").splitlines():
        m = _BATCH_OUTPUT_DIR_RE.search(line)
        if not m:
            continue
        p = Path(m.group(1)).expanduser()
        try:
            key = str(p.resolve())
        except OSError:
            key = str(p)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(p)
    return ordered


def parse_equity_session_date_hint(text: str) -> date | None:
    """Parse a human-style session label (``Wed May 13 2026``) to a calendar date."""
    dt = _parse_earnings_date_fuzzy(text.strip())
    if dt is None:
        return None
    return dt.date()


def _yfinance_earnings_report_dates(symbol: str, limit: int = 16) -> list[date] | None:
    sym = symbol.strip().upper()
    if not sym:
        return None
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        logger.warning("earnings_dates: yfinance not installed: %r", exc)
        return None
    try:
        t = yf.Ticker(sym)
    except Exception as exc:
        logger.warning("earnings_dates: Ticker failed symbol=%s err=%r", sym, exc)
        return None
    out: list[date] = []
    cal = getattr(t, "calendar", None)
    if isinstance(cal, dict):
        raw = cal.get("Earnings Date")
        candidates: list[Any] = []
        if hasattr(raw, "tolist"):
            candidates = list(raw.tolist())  # type: ignore[union-attr]
        elif isinstance(raw, (list, tuple)):
            candidates = list(raw)
        elif raw is not None:
            candidates = [raw]
        for item in candidates:
            if hasattr(item, "date"):
                try:
                    out.append(cast(date, item.date()))
                except Exception:
                    continue
            elif isinstance(item, datetime):
                out.append(item.date())
            elif isinstance(item, date):
                out.append(item)
    if not out:
        edf = getattr(t, "earnings_dates", None)
        if edf is not None and hasattr(edf, "index"):
            try:
                for ts in list(edf.index)[:limit]:
                    if hasattr(ts, "date"):
                        out.append(cast(date, ts.date()))
                    elif isinstance(ts, date):
                        out.append(ts)
            except Exception:
                out = []
    if not out:
        return None
    uniq: list[date] = []
    seen: set[date] = set()
    for d in sorted(set(out)):
        if d in seen:
            continue
        seen.add(d)
        uniq.append(d)
    return uniq[:limit]


def _history_closes_sorted(symbol: str, *, period: str = "5y") -> list[tuple[date, float]] | None:
    sym = symbol.strip().upper()
    if not sym:
        return None
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        logger.warning("history_closes: yfinance not installed: %r", exc)
        return None
    try:
        ticker = yf.Ticker(sym)
        hist = ticker.history(period=period, auto_adjust=False)
    except Exception as exc:
        logger.warning("history_closes: yfinance.history failed symbol=%s err=%r", sym, exc)
        return None
    if hist is None or getattr(hist, "empty", True):
        return None
    rows: list[tuple[date, float]] = []
    try:
        for ts, row in hist.sort_index().iterrows():
            c = _coerce_float_safe(row.get("Close"))
            if c is None or c <= 0:
                continue
            if hasattr(ts, "date"):
                d = cast(date, ts.date())
            else:
                continue
            rows.append((d, float(c)))
    except Exception as exc:
        logger.warning("history_closes: iterate failed symbol=%s err=%r", sym, exc)
        return None
    return rows


def _five_simple_returns_after_earnings(
    closes_by_date: dict[date, float],
    earnings_day: date,
) -> list[float] | None:
    """Five close-to-close % returns on the first five **weekday** sessions strictly after ``earnings_day``."""
    ordered = sorted(closes_by_date)
    rets: list[float] = []
    prev: float | None = None
    for d in ordered:
        if d <= earnings_day:
            prev = closes_by_date[d]
            continue
        if d.weekday() >= 5:
            continue
        c = closes_by_date[d]
        if prev is None or prev <= 0:
            prev = c
            continue
        rets.append((c / prev - 1.0) * 100.0)
        prev = c
        if len(rets) >= 5:
            break
    if len(rets) < 5:
        return None
    return rets


def compute_pead_avg_drift_pct(symbol: str) -> float | None:
    """Average of mean first-5-session daily % returns over the last four past earnings windows."""
    eds = _yfinance_earnings_report_dates(symbol, limit=24)
    if not eds:
        return None
    today = date.today()
    past = sorted({d for d in eds if d < today}, reverse=True)[:4]
    if len(past) < 4:
        return None
    bars = _history_closes_sorted(symbol)
    if not bars:
        return None
    by_d = {d: c for d, c in bars}
    per_window: list[float] = []
    for ed in past:
        rets = _five_simple_returns_after_earnings(by_d, ed)
        if rets is None:
            return None
        per_window.append(sum(rets) / len(rets))
    return float(sum(per_window) / len(per_window))


def compute_realized_post_earnings_daily_vol_pct(symbol: str) -> float | None:
    """Mean of (mean |r| over first 5 post-earnings sessions) across the last four past earnings windows."""
    eds = _yfinance_earnings_report_dates(symbol, limit=24)
    if not eds:
        return None
    today = date.today()
    past = sorted({d for d in eds if d < today}, reverse=True)[:4]
    if len(past) < 4:
        return None
    bars = _history_closes_sorted(symbol)
    if not bars:
        return None
    by_d = {d: c for d, c in bars}
    per_window: list[float] = []
    for ed in past:
        rets = _five_simple_returns_after_earnings(by_d, ed)
        if rets is None:
            return None
        per_window.append(sum(abs(x) for x in rets) / len(rets))
    return float(sum(per_window) / len(per_window))


def compute_recent_momentum_drift_pct(symbol: str, lookback_days: int = 10) -> float | None:
    """Mean simple daily % return over the last ``lookback_days`` NYSE-style weekdays with closes."""
    bars = _history_closes_sorted(symbol, period="1y")
    if not bars:
        return None
    weekdays = [(d, c) for d, c in bars if d.weekday() < 5]
    if len(weekdays) < lookback_days + 1:
        return None
    tail = weekdays[-(lookback_days + 1) :]
    rets: list[float] = []
    for i in range(1, len(tail)):
        a, b = tail[i - 1][1], tail[i][1]
        if a <= 0:
            continue
        rets.append((b / a - 1.0) * 100.0)
    if len(rets) < lookback_days:
        return None
    return float(sum(rets) / len(rets))


def plan_shape_b_run_directories(
    outputs_dir: Path,
    symbols: list[str],
    since: datetime,
    *,
    newest_only: bool = True,
) -> list[tuple[str, Path | None]]:
    """Resolve ``outputs/<SYM>_<TS>/`` with ``run.json`` at or after ``since`` (UTC day boundary).

    Returns rows ``(symbol_upper, run_dir_or_none)`` in input order. ``None`` means no matching
    run directory. When ``newest_only`` is true, at most one directory per symbol; otherwise
    every matching directory is returned as separate rows.
    """
    from equity_analyst.db_backfill import iter_run_directories

    out: list[tuple[str, Path | None]] = []
    for raw in symbols:
        sym = raw.strip().upper()
        if not sym:
            continue
        dirs = iter_run_directories(
            outputs_dir,
            symbol=sym,
            since=since,
            oldest_first=False,
        )
        if not dirs:
            out.append((sym, None))
        elif newest_only:
            out.append((sym, dirs[0]))
        else:
            out.extend((sym, d) for d in dirs)
    return out

