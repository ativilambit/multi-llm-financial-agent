"""NYSE regular-session calendar helpers (weekends + standard closures).

Used to derive ``next_trading_day`` / ``followup_open_date`` labels from an earnings
calendar date when those YAML fields are omitted. This is **not** a full exchange rule
engine, but it matches the common US holiday set through ~2035 for regression tests.
"""

from __future__ import annotations

import calendar
from datetime import date, timedelta
from functools import lru_cache

# Hand-maintained batch configs use the 5th regular session after the earnings calendar day
# for ``followup_open_date`` (e.g. YETI May 14 → May 21; NVDA May 19 → May 27 with Memorial Day).
FOLLOWUP_OPEN_NTH_TRADING_DAY_AFTER_EARNINGS = 5


def _easter_sunday_gregorian(year: int) -> date:
    """Anonymous Gregorian algorithm; returns Western Easter Sunday in ``year``."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    day = ((h + ell - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _nth_weekday_of_month(year: int, month: int, weekday: int, n: int) -> date:
    """``weekday``: Monday=0 .. Sunday=6; ``n`` 1=first occurrence in month."""
    d = date(year, month, 1)
    seen = 0
    while d.month == month:
        if d.weekday() == weekday:
            seen += 1
            if seen == n:
                return d
        d += timedelta(days=1)
    raise ValueError(f"nth weekday not found year={year} month={month} weekday={weekday} n={n}")


def _last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    last = date(year, month, calendar.monthrange(year, month)[1])
    d = last
    while d.weekday() != weekday:
        d -= timedelta(days=1)
    return d


def _observed_weekend_holiday(month: int, day: int, year: int) -> date:
    """NYSE-style observed date for a fixed US calendar month/day (not Easter).

    Saturday holiday -> preceding Friday. Sunday holiday -> following Monday.
    """
    d = date(year, month, day)
    wd = d.weekday()
    if wd == 5:  # Saturday
        return d - timedelta(days=1)
    if wd == 6:  # Sunday
        return d + timedelta(days=1)
    return d


@lru_cache(maxsize=64)
def nyse_closed_dates(year: int) -> frozenset[date]:
    """Dates the NYSE is fully closed (regular session), including observed moves."""
    out: set[date] = set()
    # New Year's Day
    out.add(_observed_weekend_holiday(1, 1, year))
    # Martin Luther King Jr. Day — 3rd Monday in January
    out.add(_nth_weekday_of_month(year, 1, 0, 3))
    # Presidents' Day — 3rd Monday in February
    out.add(_nth_weekday_of_month(year, 2, 0, 3))
    # Good Friday
    easter = _easter_sunday_gregorian(year)
    out.add(easter - timedelta(days=2))
    # Memorial Day — last Monday in May
    out.add(_last_weekday_of_month(year, 5, 0))
    # Juneteenth
    out.add(_observed_weekend_holiday(6, 19, year))
    # Independence Day
    out.add(_observed_weekend_holiday(7, 4, year))
    # Labor Day — 1st Monday in September
    out.add(_nth_weekday_of_month(year, 9, 0, 1))
    # Thanksgiving — 4th Thursday in November
    out.add(_nth_weekday_of_month(year, 11, 3, 4))
    # Christmas
    out.add(_observed_weekend_holiday(12, 25, year))
    return frozenset(out)


def is_nyse_trading_day(d: date) -> bool:
    """Return True when ``d`` is a regular NYSE session (Mon-Fri, not a full closure)."""
    if d.weekday() >= 5:
        return False
    return d not in nyse_closed_dates(d.year)


def nth_nyse_trading_day_after(earnings_calendar_date: date, n: int) -> date:
    """Return the ``n``-th NYSE trading session **strictly after** ``earnings_calendar_date``.

    ``n`` uses the same counting as hand-maintained configs (e.g. ``n=1`` is the session
    after the earnings print calendar day; ``n=5`` matches ``followup_open_date`` in
    the May-2026 batch examples).
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    d = earnings_calendar_date
    seen = 0
    while seen < n:
        d += timedelta(days=1)
        if is_nyse_trading_day(d):
            seen += 1
    return d


def format_earnings_human(d: date) -> str:
    """``Tue May 19 2026`` style (matches existing YAML)."""
    return f"{d:%a %b} {d.day} {d.year}"


def format_session_label_no_year(d: date) -> str:
    """``Wed May 20`` style (matches ``next_trading_day`` / ``followup_open_date`` YAML)."""
    return f"{d:%a %b} {d.day}"
