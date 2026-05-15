from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from io import StringIO
from typing import Any
from unittest.mock import AsyncMock

import pytest

from equity_analyst.cli import _build_parser
from equity_analyst.prediction_scoreboard import (
    prediction_payload_from_row,
    scoreboard_row_to_cli_dict,
    write_scoreboard_csv,
)


def test_score_predictions_cli_parser() -> None:
    p = _build_parser()
    args = p.parse_args(
        [
            "score-predictions",
            "--since",
            "2026-01-15",
            "--symbol",
            "yetI",
            "--format",
            "table",
            "--limit",
            "42",
            "--database-url",
            "postgresql+psycopg://u:p@localhost:5432/db",
        ]
    )
    assert args.command == "score-predictions"
    assert args.since == "2026-01-15"
    assert args.symbol == "yetI"
    assert args.output_format == "table"
    assert args.limit == 42
    assert args.database_url == "postgresql+psycopg://u:p@localhost:5432/db"


def test_scoreboard_row_to_cli_dict_and_csv() -> None:
    row = {
        "run_id": "X_20260101T000000Z",
        "symbol": "X",
        "prediction_id": 7,
        "outcome_recorded_at_utc": datetime(2026, 1, 2, tzinfo=UTC),
        "earnings_day_close": Decimal("100"),
        "next_trading_day_close": Decimal("101"),
        "one_week_later_close": Decimal("102"),
        "prediction_horizon": "earnings_day_close",
        "predicted_probability_up": Decimal("0.55"),
        "predicted_range_low": Decimal("98"),
        "predicted_range_high": Decimal("102"),
        "predicted_point": Decimal("100.5"),
        "prediction_source": "llm_extracted",
    }
    d = scoreboard_row_to_cli_dict(row)
    assert d["outcome_present"] is True
    assert d["run_id"] == "X_20260101T000000Z"
    payload = prediction_payload_from_row(row)
    assert payload["horizon"] == "earnings_day_close"
    assert payload["predicted_point"] == 100.5

    buf = StringIO()
    write_scoreboard_csv([row], buf)
    lines = buf.getvalue().strip().splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("run_id,symbol,prediction_id")


@pytest.mark.asyncio
async def test_run_score_predictions_cli_uses_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    from equity_analyst import prediction_scoreboard as ps

    async def _fake_rows(
        *,
        since: datetime,
        symbol: str | None,
        limit: int,
        database_url: str | None,
    ) -> list[dict[str, Any]]:
        assert since.tzinfo is not None
        assert limit == 500
        assert symbol is None
        return [
            {
                "run_id": "AB_20260101T000000Z",
                "symbol": "AB",
                "prediction_id": 99,
                "outcome_recorded_at_utc": datetime(2026, 1, 2, tzinfo=UTC),
                "earnings_day_close": Decimal("10.0"),
                "next_trading_day_close": Decimal("10.25"),
                "one_week_later_close": Decimal("10.5"),
                "prediction_horizon": "earnings_day_close",
                "predicted_probability_up": None,
                "predicted_range_low": None,
                "predicted_range_high": None,
                "predicted_point": Decimal("10.5"),
                "prediction_source": "llm_extracted",
                "horizon_actual": Decimal("10.0"),
                "point_absolute_error": Decimal("0.5"),
            }
        ]

    monkeypatch.setattr(ps, "is_db_available", AsyncMock(return_value=True))
    monkeypatch.setattr(ps, "fetch_scoreboard_rows", _fake_rows)

    buf = StringIO()
    rc = await ps.run_score_predictions_cli(
        since=datetime(2026, 1, 1, tzinfo=UTC),
        symbol=None,
        limit=500,
        fmt="csv",
        database_url=None,
        out=buf,
    )
    assert rc == 0
    assert "AB_20260101T000000Z" in buf.getvalue()
    assert "0.500000" in buf.getvalue() or "0.5" in buf.getvalue()


def test_migration_0006_defines_scoreboard_view() -> None:
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[1]
        / "migrations"
        / "versions"
        / "0006_prediction_outcome_scoreboard.py"
    )
    text = path.read_text(encoding="utf-8")
    assert "prediction_outcome_scoreboard" in text
    assert "v_runs_outcomes_predictions" in text
    assert 'down_revision = "0005_runs_synthesis_markdown"' in text
