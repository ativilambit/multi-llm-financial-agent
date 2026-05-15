"""Query prediction vs outcome scoreboard (Postgres views + CLI helpers)."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal, TextIO

from sqlalchemy import text

from equity_analyst.db import get_async_session, is_db_available

VIEW_V_RUNS_OUTCOMES_PREDICTIONS = "v_runs_outcomes_predictions"
VIEW_PREDICTION_OUTCOME_SCOREBOARD = "prediction_outcome_scoreboard"

ScoreboardFormat = Literal["csv", "table"]


def _json_safe(v: Any) -> Any:
    if isinstance(v, datetime):
        return v.isoformat()
    if isinstance(v, Decimal):
        return float(v)
    return v


def prediction_payload_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Structured prediction fields for CSV ``prediction_json`` column."""
    return {
        "horizon": row.get("prediction_horizon"),
        "predicted_probability_up": _json_safe(row.get("predicted_probability_up")),
        "predicted_range_low": _json_safe(row.get("predicted_range_low")),
        "predicted_range_high": _json_safe(row.get("predicted_range_high")),
        "predicted_point": _json_safe(row.get("predicted_point")),
        "source": row.get("prediction_source"),
    }


def scoreboard_row_to_cli_dict(row: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten one scoreboard view row for CLI export."""
    outcome_present = row.get("outcome_recorded_at_utc") is not None
    return {
        "run_id": row.get("run_id"),
        "symbol": row.get("symbol"),
        "prediction_id": row.get("prediction_id"),
        "outcome_present": outcome_present,
        "earnings_day_close": row.get("earnings_day_close"),
        "next_trading_day_close": row.get("next_trading_day_close"),
        "one_week_later_close": row.get("one_week_later_close"),
        "prediction_json": json.dumps(prediction_payload_from_row(row), sort_keys=True),
        "horizon_actual": row.get("horizon_actual"),
        "point_absolute_error": row.get("point_absolute_error"),
    }


def _csv_cell(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, datetime):
        return v.isoformat()
    if isinstance(v, Decimal):
        return format(v, "f")
    return str(v)


def write_scoreboard_csv(rows: Sequence[Mapping[str, Any]], out: TextIO) -> None:
    headers = [
        "run_id",
        "symbol",
        "prediction_id",
        "outcome_present",
        "earnings_day_close",
        "next_trading_day_close",
        "one_week_later_close",
        "prediction_json",
        "horizon_actual",
        "point_absolute_error",
    ]
    w = csv.writer(out, lineterminator="\n")
    w.writerow(headers)
    for raw in rows:
        d = scoreboard_row_to_cli_dict(raw)
        w.writerow([_csv_cell(d[h]) for h in headers])


def write_scoreboard_table(rows: Sequence[Mapping[str, Any]], out: TextIO) -> None:
    headers = [
        "run_id",
        "symbol",
        "prediction_id",
        "outcome_present",
        "earnings_day_close",
        "next_day_close",
        "one_week_close",
        "horizon_actual",
        "point_abs_err",
        "prediction_json",
    ]
    lines: list[list[str]] = []
    for raw in rows:
        d = scoreboard_row_to_cli_dict(raw)
        pj = str(d["prediction_json"])
        if len(pj) > 64:
            pj = pj[:61] + "..."
        lines.append(
            [
                str(d["run_id"] or ""),
                str(d["symbol"] or ""),
                str(d["prediction_id"] or ""),
                str(d["outcome_present"]).lower(),
                _csv_cell(d["earnings_day_close"]),
                _csv_cell(d["next_trading_day_close"]),
                _csv_cell(d["one_week_later_close"]),
                _csv_cell(d["horizon_actual"]),
                _csv_cell(d["point_absolute_error"]),
                pj,
            ]
        )
    widths = [len(h) for h in headers]
    for line in lines:
        for i, cell in enumerate(line):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells: list[str]) -> str:
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    out.write(fmt_row(headers) + "\n")
    out.write(fmt_row(["-" * w for w in widths]) + "\n")
    for line in lines:
        out.write(fmt_row(line) + "\n")


_SCOREBOARD_SQL = f"""
SELECT *
FROM {VIEW_PREDICTION_OUTCOME_SCOREBOARD}
WHERE run_created_at_utc >= :since
  AND (:symbol IS NULL OR symbol = :symbol)
ORDER BY run_created_at_utc DESC NULLS LAST, prediction_id ASC NULLS LAST
LIMIT :limit
"""


async def fetch_scoreboard_rows(
    *,
    since: datetime,
    symbol: str | None,
    limit: int,
    database_url: str | None,
) -> list[dict[str, Any]]:
    """Return rows from ``prediction_outcome_scoreboard`` (newest runs first)."""
    sym = symbol.strip().upper() if symbol and str(symbol).strip() else None
    params: dict[str, Any] = {"since": since, "symbol": sym, "limit": limit}

    stmt = text(_SCOREBOARD_SQL)

    async with get_async_session(database_url=database_url) as owned:
        result = await owned.execute(stmt, params)
        return [dict(r) for r in result.mappings().all()]


async def run_score_predictions_cli(
    *,
    since: datetime,
    symbol: str | None,
    limit: int,
    fmt: ScoreboardFormat,
    database_url: str | None,
    out: TextIO,
) -> int:
    if not await is_db_available(database_url=database_url):
        raise SystemExit(
            "score-predictions: DATABASE_URL is unreachable. "
            "Set DATABASE_URL or pass --database-url."
        )
    rows = await fetch_scoreboard_rows(
        since=since, symbol=symbol, limit=limit, database_url=database_url
    )
    if fmt == "csv":
        write_scoreboard_csv(rows, out)
    else:
        write_scoreboard_table(rows, out)
    if fmt == "csv" and hasattr(out, "flush"):
        out.flush()
    return 0
