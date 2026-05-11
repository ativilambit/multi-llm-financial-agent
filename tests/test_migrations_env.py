"""Wiring checks for repo-root `.env` loading in Alembic's env.py."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PY = REPO_ROOT / "migrations" / "env.py"
ALEMBIC_BIN = REPO_ROOT / ".venv" / "bin" / "alembic"


def test_migrations_env_py_includes_dotenv_hook() -> None:
    text = ENV_PY.read_text(encoding="utf-8")
    assert "load_dotenv" in text
    assert "override=False" in text
    assert ".env" in text


@pytest.mark.skipif(not ALEMBIC_BIN.is_file(), reason=".venv/bin/alembic not present")
@pytest.mark.skipif(not (REPO_ROOT / ".env").is_file(), reason="repo .env not present")
def test_alembic_offline_sql_upgrade_sees_database_url_from_dotenv() -> None:
    """Without DATABASE_URL in the subprocess env, env.py must load it from .env."""
    child_env = {k: v for k, v in os.environ.items() if k != "DATABASE_URL"}
    assert "DATABASE_URL" not in child_env
    r = subprocess.run(
        [str(ALEMBIC_BIN), "upgrade", "head", "--sql"],
        cwd=str(REPO_ROOT),
        env=child_env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0, (r.stderr, r.stdout)


@pytest.mark.skipif(not ALEMBIC_BIN.is_file(), reason=".venv/bin/alembic not present")
@pytest.mark.skipif(not (REPO_ROOT / ".env").is_file(), reason="repo .env not present")
def test_alembic_history_without_shell_database_url() -> None:
    child_env = {k: v for k, v in os.environ.items() if k != "DATABASE_URL"}
    assert "DATABASE_URL" not in child_env
    r = subprocess.run(
        [str(ALEMBIC_BIN), "history"],
        cwd=str(REPO_ROOT),
        env=child_env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert r.returncode == 0, (r.stderr, r.stdout)
