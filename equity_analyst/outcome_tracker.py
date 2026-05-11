from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict


class RunOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_output_dir: str  # absolute path to outputs/<SYM>_<TS>Z/
    symbol: str
    recorded_at_utc: str  # ISO8601 Z
    earnings_date: str  # from run.json / config snapshot

    synthesis_path: str
    run_json_path: str

    earnings_day_open: float | None = None
    earnings_day_high: float | None = None
    earnings_day_low: float | None = None
    earnings_day_close: float | None = None
    next_trading_day_open: float | None = None
    next_trading_day_close: float | None = None
    one_week_later_close: float | None = None
    direction_vs_prior_close: Literal["up", "down", "flat"] | None = None
    notes: str | None = None
    source: Literal["manual", "yahoo_csv", "alpaca", "polygon"] = "manual"

    baseline_close_hint: float | None = None


def _iso_utc_z_now() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _infer_repo_root_from_run_dir(run_dir: Path) -> Path:
    if run_dir.parent.name != "outputs":
        raise ValueError(
            f"run_dir must be inside an outputs/ folder (got {str(run_dir)!r}, parent={run_dir.parent.name!r})"
        )
    return run_dir.parent.parent.resolve()


def _pick_synthesis_path(run_dir: Path) -> Path:
    direct = run_dir / "synthesis.md"
    if direct.is_file():
        return direct

    it_dir = run_dir / "iterations"
    if not it_dir.is_dir():
        return direct

    best: tuple[int, Path] | None = None
    for p in it_dir.glob("iteration_*_synthesis.md"):
        stem = p.name.replace("iteration_", "").replace("_synthesis.md", "")
        try:
            n = int(stem)
        except ValueError:
            continue
        if best is None or n > best[0]:
            best = (n, p)
    return best[1] if best is not None else direct


def _parse_baseline_close_hint(run_dir: Path) -> float | None:
    """
    Best-effort, non-brittle baseline hint for calibration.

    Currently reads RunConfig.current_price from run.json (YAML alias: reference_last_price).
    """
    run_json = run_dir / "run.json"
    if not run_json.is_file():
        return None
    try:
        data = json.loads(run_json.read_text(encoding="utf-8"))
    except Exception:
        return None
    cfg = data.get("config")
    if not isinstance(cfg, dict):
        return None
    v = cfg.get("current_price")
    try:
        return float(v) if v is not None else None
    except Exception:
        return None


def record_outcome(
    *,
    run_dir: Path,
    earnings_day_open: float | None = None,
    earnings_day_high: float | None = None,
    earnings_day_low: float | None = None,
    earnings_day_close: float | None = None,
    next_trading_day_open: float | None = None,
    next_trading_day_close: float | None = None,
    one_week_later_close: float | None = None,
    direction_vs_prior_close: Literal["up", "down", "flat"] | None = None,
    notes: str | None = None,
    source: Literal["manual", "yahoo_csv", "alpaca", "polygon"] = "manual",
) -> RunOutcome:
    run_dir = run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"--run-dir does not exist: {run_dir!s}")

    run_json = run_dir / "run.json"
    if not run_json.is_file():
        raise FileNotFoundError(f"Missing run.json at {run_json!s}")

    data = json.loads(run_json.read_text(encoding="utf-8"))
    cfg = data.get("config")
    if not isinstance(cfg, dict):
        raise ValueError("run.json missing config snapshot")

    symbol = cfg.get("symbol")
    earnings_date = cfg.get("earnings_date")
    if not symbol or not isinstance(symbol, str):
        raise ValueError("run.json config snapshot missing symbol")
    if not earnings_date or not isinstance(earnings_date, str):
        raise ValueError("run.json config snapshot missing earnings_date")

    synthesis_path = _pick_synthesis_path(run_dir)
    repo_root = _infer_repo_root_from_run_dir(run_dir)

    outcome = RunOutcome(
        run_output_dir=str(run_dir),
        symbol=symbol,
        recorded_at_utc=_iso_utc_z_now(),
        earnings_date=earnings_date,
        synthesis_path=str(synthesis_path),
        run_json_path=str(run_json),
        earnings_day_open=earnings_day_open,
        earnings_day_high=earnings_day_high,
        earnings_day_low=earnings_day_low,
        earnings_day_close=earnings_day_close,
        next_trading_day_open=next_trading_day_open,
        next_trading_day_close=next_trading_day_close,
        one_week_later_close=one_week_later_close,
        direction_vs_prior_close=direction_vs_prior_close,
        notes=notes,
        source=source,
        baseline_close_hint=_parse_baseline_close_hint(run_dir),
    )

    outcome_path = run_dir / "outcome.json"
    outcome_path.write_text(
        json.dumps(outcome.model_dump(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    outputs_dir = repo_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    registry_path = outputs_dir / "outcomes_registry.jsonl"
    with registry_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(outcome.model_dump(), sort_keys=True) + "\n")

    return outcome

