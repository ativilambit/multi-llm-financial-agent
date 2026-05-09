from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TextIO

_PACKAGE = "equity_analyst"


class _EquityAnalystStderrHandler(logging.StreamHandler[TextIO]):
    """Stream handler installed by ``configure_cli_logging`` (identified for idempotent setup)."""


class _EquityAnalystFileHandler(logging.FileHandler):
    """File handler for a single run directory (``resolved`` identifies duplicates)."""

    def __init__(self, log_path: Path, *, resolved: str, level: int) -> None:
        super().__init__(log_path, encoding="utf-8")
        self.resolved_path = resolved
        self.setLevel(level)


def _formatter() -> logging.Formatter:
    return logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


def configure_cli_logging(level: int) -> None:
    """Configure the ``equity_analyst`` package logger for stderr (idempotent)."""
    pkg = logging.getLogger(_PACKAGE)
    pkg.setLevel(level)
    for h in list(pkg.handlers):
        if isinstance(h, _EquityAnalystStderrHandler):
            pkg.removeHandler(h)
    handler = _EquityAnalystStderrHandler(sys.stderr)
    handler.setFormatter(_formatter())
    handler.setLevel(level)
    pkg.addHandler(handler)
    pkg.propagate = True


def attach_run_file_logging(log_path: Path, *, level: int | None = None) -> None:
    """Append a UTF-8 file handler for this process; skips if the same path is already attached."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pkg = logging.getLogger(_PACKAGE)
    resolved = str(log_path.resolve())
    for h in pkg.handlers:
        if isinstance(h, _EquityAnalystFileHandler) and h.resolved_path == resolved:
            return
    eff_level = level if level is not None else pkg.level
    fh = _EquityAnalystFileHandler(log_path, resolved=resolved, level=eff_level)
    fh.setFormatter(_formatter())
    pkg.addHandler(fh)
