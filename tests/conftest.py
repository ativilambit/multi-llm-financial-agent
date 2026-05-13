from __future__ import annotations

import logging
import sys
from collections.abc import Generator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def reset_equity_analyst_log_handlers() -> Generator[None, None, None]:
    yield
    pkg = logging.getLogger("equity_analyst")
    for h in list(pkg.handlers):
        pkg.removeHandler(h)
        h.close()
    pkg.setLevel(logging.NOTSET)


@pytest.fixture(autouse=True)
def neutralize_prompting_yfinance_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid live Yahoo Finance calls during prompt rendering (CI / sandbox proxy safe).

    Individual tests may ``monkeypatch.setattr`` the same names for controlled values.
    """
    import equity_analyst.prompting as prompting
    import equity_analyst.sigma_compute as sigma_compute

    for mod in (prompting, sigma_compute):
        monkeypatch.setattr(mod, "fetch_hv30_annualized_percent", lambda _symbol: None)
        monkeypatch.setattr(mod, "compute_pead_avg_drift_pct", lambda _symbol: None)
        monkeypatch.setattr(mod, "compute_recent_momentum_drift_pct", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        sigma_compute,
        "compute_realized_post_earnings_daily_vol_pct",
        lambda _symbol: None,
    )
