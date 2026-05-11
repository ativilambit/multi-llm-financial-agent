from __future__ import annotations

import json
from pathlib import Path

from equity_analyst.outcome_tracker import record_outcome


def test_record_outcome_writes_outcome_json_and_registry(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "ACME_20260511T123456Z"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "dry_run": False,
                "timestamp_utc": "2026-05-11T12:34:56Z",
                "config": {"symbol": "ACME", "earnings_date": "2026-05-11", "current_price": 10.0},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "synthesis.md").write_text("# synthesis\n", encoding="utf-8")

    out = record_outcome(
        run_dir=run_dir,
        earnings_day_close=12.34,
        next_trading_day_close=12.0,
        direction_vs_prior_close="up",
        notes="ok",
        source="manual",
    )

    outcome_path = run_dir / "outcome.json"
    assert outcome_path.is_file()
    payload = json.loads(outcome_path.read_text(encoding="utf-8"))
    assert payload["symbol"] == "ACME"
    assert payload["earnings_date"] == "2026-05-11"
    assert payload["earnings_day_close"] == 12.34
    assert payload["next_trading_day_close"] == 12.0
    assert payload["direction_vs_prior_close"] == "up"
    assert payload["notes"] == "ok"
    assert payload["source"] == "manual"
    assert payload["run_json_path"].endswith("run.json")
    assert payload["synthesis_path"].endswith("synthesis.md")
    assert payload["baseline_close_hint"] == 10.0
    assert payload["run_output_dir"] == str(run_dir.resolve())

    registry = tmp_path / "outputs" / "outcomes_registry.jsonl"
    assert registry.is_file()
    lines = registry.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    reg_payload = json.loads(lines[0])
    assert reg_payload["run_output_dir"] == payload["run_output_dir"]

    assert out.symbol == "ACME"

