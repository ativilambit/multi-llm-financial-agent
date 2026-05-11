from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from equity_analyst.outcome_tracker import (
    RecordOutcomeForRunDirResult,
    RunOutcome,
    plan_shape_b_run_directories,
)


def _minimal_run_json(symbol: str) -> str:
    return (
        json.dumps(
            {
                "config": {
                    "symbol": symbol,
                    "earnings_date": "Mon May 11 2026",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _fake_record_result(*, symbol: str, run_dir: Path) -> RecordOutcomeForRunDirResult:
    o = RunOutcome(
        run_output_dir=str(run_dir.resolve()),
        symbol=symbol,
        recorded_at_utc="2026-05-11T12:00:00Z",
        earnings_date="Mon May 11 2026",
        synthesis_path=str(run_dir / "synthesis.md"),
        run_json_path=str(run_dir / "run.json"),
        earnings_day_close=10.0,
        direction_vs_prior_close="up",
    )
    return RecordOutcomeForRunDirResult(
        outcome=o,
        auto_fetch_used=False,
        yfinance_empty=False,
        auto_fetch_partial=False,
    )


def test_shape_a_batch_summary_invokes_recorder_per_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from equity_analyst import cli

    monkeypatch.chdir(tmp_path)
    batch = tmp_path / "outputs" / "batch_20260511T025203Z"
    batch.mkdir(parents=True)
    dirs: list[Path] = []
    for sym in ("AAA", "BBB", "CCC"):
        d = tmp_path / "outputs" / f"{sym}_20260511T010101Z"
        d.mkdir(parents=True)
        (d / "run.json").write_text(_minimal_run_json(sym), encoding="utf-8")
        (d / "synthesis.md").write_text("# x\n", encoding="utf-8")
        dirs.append(d)

    lines = [
        f"[OK]   {d.name.split('_')[0]}  duration=1s  output_dir={d.resolve()}"
        for d in dirs
    ]
    (batch / "batch_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    calls: list[Path] = []

    def _stub(**kw: object) -> RecordOutcomeForRunDirResult:
        rd = kw["run_dir"]
        assert isinstance(rd, Path)
        calls.append(rd)
        return _fake_record_result(symbol=rd.name.split("_")[0], run_dir=rd)

    monkeypatch.setattr(cli, "record_outcome_for_run_dir", _stub)

    code = cli.main(
        [
            "outcome-record-batch",
            "--batch-dir",
            str(batch),
        ]
    )
    assert code == 0
    assert len(calls) == 3
    assert {c.resolve() for c in calls} == {d.resolve() for d in dirs}

    out = capsys.readouterr().out
    assert "Batch outcome record summary" in out
    assert "Attempted:  3" in out
    assert "Recorded:   3" in out


def test_shape_b_newest_only_picks_latest_per_symbol(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    for name in ("SE_20260510T010101Z", "SE_20260512T010101Z", "ZBRA_20260511T010101Z"):
        d = outputs / name
        d.mkdir(parents=True)
        sym = name.split("_", 1)[0]
        (d / "run.json").write_text(_minimal_run_json(sym), encoding="utf-8")

    since = datetime(2026, 5, 9, tzinfo=UTC)
    plan = plan_shape_b_run_directories(
        outputs,
        ["SE", "ZBRA", "MISSING"],
        since,
        newest_only=True,
    )
    by_sym = {s: p for s, p in plan}
    assert by_sym["SE"] == outputs / "SE_20260512T010101Z"
    assert by_sym["ZBRA"] == outputs / "ZBRA_20260511T010101Z"
    assert by_sym["MISSING"] is None


def test_dry_run_writes_no_outcome_artifact(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from equity_analyst import cli

    monkeypatch.chdir(tmp_path)
    d = tmp_path / "outputs" / "ZZ_20260511T010101Z"
    d.mkdir(parents=True)
    (d / "run.json").write_text(_minimal_run_json("ZZ"), encoding="utf-8")
    (d / "synthesis.md").write_text("# s\n", encoding="utf-8")

    batch = tmp_path / "outputs" / "batch_x"
    batch.mkdir(parents=True)
    (batch / "batch_summary.txt").write_text(
        f"[OK]   ZZ  duration=1s  output_dir={d.resolve()}\n",
        encoding="utf-8",
    )

    code = cli.main(
        [
            "outcome-record-batch",
            "--batch-dir",
            str(batch),
            "--dry-run",
        ]
    )
    assert code == 0
    assert not (d / "outcome.json").exists()
    assert not (tmp_path / "outputs" / "outcomes_registry.jsonl").exists()


def test_continue_on_error_processes_remaining(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from equity_analyst import cli

    monkeypatch.chdir(tmp_path)
    batch = tmp_path / "outputs" / "batch_y"
    batch.mkdir(parents=True)
    dirs: list[Path] = []
    for sym in ("ONE", "TWO", "THREE"):
        d = tmp_path / "outputs" / f"{sym}_20260511T010101Z"
        d.mkdir(parents=True)
        (d / "run.json").write_text(_minimal_run_json(sym), encoding="utf-8")
        (d / "synthesis.md").write_text("# s\n", encoding="utf-8")
        dirs.append(d)

    lines = [f"[OK]   {d.name}  output_dir={d.resolve()}" for d in dirs]
    (batch / "batch_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    calls: list[str] = []

    def _stub(**kw: object) -> RecordOutcomeForRunDirResult:
        rd = kw["run_dir"]
        assert isinstance(rd, Path)
        calls.append(rd.name.split("_")[0])
        if rd.name.startswith("TWO_"):
            raise RuntimeError("simulated")
        return _fake_record_result(symbol=rd.name.split("_")[0], run_dir=rd)

    monkeypatch.setattr(cli, "record_outcome_for_run_dir", _stub)

    code = cli.main(
        [
            "outcome-record-batch",
            "--batch-dir",
            str(batch),
            "--continue-on-error",
        ]
    )
    assert code == 1
    assert calls == ["ONE", "TWO", "THREE"]
