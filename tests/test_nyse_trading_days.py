from __future__ import annotations

from datetime import date

import pytest

from equity_analyst.nyse_trading_days import (
    format_earnings_human,
    format_session_label_no_year,
    is_nyse_trading_day,
    nth_nyse_trading_day_after,
)


def test_memorial_day_2026_skipped_for_nth_session() -> None:
    assert is_nyse_trading_day(date(2026, 5, 25)) is False
    assert nth_nyse_trading_day_after(date(2026, 5, 19), 1) == date(2026, 5, 20)
    assert nth_nyse_trading_day_after(date(2026, 5, 19), 5) == date(2026, 5, 27)


def test_yeti_may_14_2026_cluster_matches_hand_configs() -> None:
    e = date(2026, 5, 14)
    assert nth_nyse_trading_day_after(e, 1) == date(2026, 5, 15)
    assert nth_nyse_trading_day_after(e, 5) == date(2026, 5, 21)
    assert format_session_label_no_year(date(2026, 5, 15)) == "Fri May 15"
    assert format_earnings_human(e) == "Thu May 14 2026"


def test_good_friday_closure() -> None:
    gf = date(2026, 4, 3)
    assert is_nyse_trading_day(gf) is False
    assert nth_nyse_trading_day_after(date(2026, 4, 2), 1) == date(2026, 4, 6)


def test_nth_requires_positive() -> None:
    with pytest.raises(ValueError):
        nth_nyse_trading_day_after(date(2026, 5, 19), 0)
