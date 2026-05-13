from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


def _write_minimal_run_json(run_dir: Path, *, symbol: str = "CRCL") -> None:
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "dry_run": False,
                "timestamp_utc": "2026-05-11T12:34:56+00:00",
                "config": {
                    "symbol": symbol,
                    "today_date": "Mon May 11 2026",
                    "today_session": "regular",
                    "earnings_date": "Mon May 11 2026",
                    "next_trading_day": "Tue May 12 2026",
                    "followup_open_date": "Tue May 12 2026",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_outcome_record_upserts_db_when_available(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from equity_analyst import cli

    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "outputs" / "CRCL_20260511T123456Z"
    run_dir.mkdir(parents=True)
    _write_minimal_run_json(run_dir)

    captured: dict[str, Any] = {}

    async def _fake_upsert(*, cfg_db_enabled: bool, run_id: str, outcome: dict[str, Any], database_url: str | None, **_kw: Any):
        captured["cfg_db_enabled"] = cfg_db_enabled
        captured["run_id"] = run_id
        captured["outcome"] = outcome
        captured["database_url"] = database_url
        captured["run_profile"] = _kw.get("run_profile")

    monkeypatch.setattr("equity_analyst.db_ops.best_effort_upsert_outcome", _fake_upsert)

    code = cli.main(
        [
            "outcome-record",
            "--run-dir",
            str(run_dir),
            "--earnings-day-close",
            "12.34",
            "--direction-vs-prior-close",
            "down",
            "--notes",
            "test note",
        ]
    )
    assert code == 0

    assert (run_dir / "outcome.json").is_file()
    assert (tmp_path / "outputs" / "outcomes_registry.jsonl").is_file()

    assert captured["run_id"] == "CRCL_20260511T123456Z"
    assert captured["cfg_db_enabled"] is True
    assert captured["outcome"]["earnings_day_close"] == 12.34
    assert captured["outcome"]["direction_vs_prior_close"] == "down"
    assert captured["run_profile"] == "production"


def test_outcome_record_still_writes_files_when_db_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from equity_analyst import cli

    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "outputs" / "CRCL_20260511T123456Z"
    run_dir.mkdir(parents=True)
    _write_minimal_run_json(run_dir)

    async def _fake_upsert(*_a: Any, **_kw: Any) -> None:
        raise RuntimeError("db down")

    monkeypatch.setattr("equity_analyst.db_ops.best_effort_upsert_outcome", _fake_upsert)

    code = cli.main(["outcome-record", "--run-dir", str(run_dir), "--earnings-day-close", "1.0"])
    assert code == 0
    assert (run_dir / "outcome.json").is_file()


def test_outcome_record_skips_db_when_run_json_dev_profile(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from equity_analyst import cli

    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "outputs" / "CRCL_20260511T123456Z"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "dry_run": False,
                "run_profile": "dev",
                "timestamp_utc": "2026-05-11T12:34:56+00:00",
                "config": {
                    "symbol": "CRCL",
                    "today_date": "Mon May 11 2026",
                    "today_session": "regular",
                    "earnings_date": "Mon May 11 2026",
                    "next_trading_day": "Tue May 12 2026",
                    "followup_open_date": "Tue May 12 2026",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    called = {"n": 0}

    async def _fake_upsert(*_a: Any, **_kw: Any) -> None:
        called["n"] += 1

    monkeypatch.setattr("equity_analyst.db_ops.best_effort_upsert_outcome", _fake_upsert)

    code = cli.main(["outcome-record", "--run-dir", str(run_dir), "--earnings-day-close", "1.0"])
    assert code == 0
    assert (run_dir / "outcome.json").is_file()
    assert called["n"] == 0
