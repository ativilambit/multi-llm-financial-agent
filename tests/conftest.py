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

