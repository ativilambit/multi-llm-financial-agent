"""Light checks for scripts/run_all_symbols.sh (syntax only; no batch execution)."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_all_symbols.sh"


def test_run_all_symbols_bash_syntax() -> None:
    subprocess.run(
        ["bash", "-n", str(SCRIPT)],
        check=True,
        cwd=REPO_ROOT,
    )
