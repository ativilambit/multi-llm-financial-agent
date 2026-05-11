from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.sql.dml import Delete, Insert
from sqlalchemy.sql.selectable import Select

from equity_analyst.db_backfill import (
    backfill_run_directory,
    build_provider_response_rows,
    build_run_row,
    iter_run_directories,
)
from equity_analyst.db_models import ProviderResponseRow, RunRow


def _write_standard_run(
    out_dir: Path,
    *,
    run_id: str,
    symbol: str,
    earnings_date: str = "Mon May 11 2026",
    iterative: bool = False,
    extras: dict[str, Any] | None = None,
) -> Path:
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True)
    config = {
        "symbol": symbol,
        "company_name": f"{symbol} Inc.",
        "current_price": 100.0,
        "earnings_date": earnings_date,
        "earnings_timing": "early morning et, before the market open",
        "followup_open_date": "Mon May 18",
        "next_trading_day": "Tues May 12",
        "today_date": "Fri May 8, 2026",
        "today_session": "regular",
        "today_high": 105.0,
        "today_low": 95.0,
        "providers": [
            {"name": "openai", "model": "gpt-5.5", "web_search": False},
        ],
        "synthesizer": {"name": "openai", "model": "gpt-5.5"},
        "run_environment": "test",
    }
    data: dict[str, Any] = {
        "config": config,
        "started_at_utc": "2026-05-11T02:37:00+00:00",
        "finished_at_utc": "2026-05-11T02:39:59+00:00",
        "iterative": iterative,
        "providers": {
            "openai": {
                "provider_name": "openai",
                "model": "gpt-5.5",
                "latency_s": 59.7,
                "usage": {"input_tokens": 3019, "output_tokens": 4118, "total_tokens": 7137},
            }
        },
        "synthesis": {
            "provider": "openai",
            "model": "gpt-5.5",
            "latency_s": 72.0,
            "usage": {"input_tokens": 12344, "output_tokens": 4301, "total_tokens": 16645},
        },
    }
    if extras:
        data.update(extras)
    (run_dir / "run.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    (run_dir / "openai.md").write_text("# openai response\n", encoding="utf-8")
    (run_dir / "synthesis.md").write_text("# synthesis\n", encoding="utf-8")
    return run_dir


def test_build_run_row_full_metadata(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    run_dir = _write_standard_run(outputs, run_id="CRCL_20260511T023700Z", symbol="CRCL")
    data = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))

    row = build_run_row(run_id=run_dir.name, run_dir=run_dir, data=data)

    assert row["run_id"] == "CRCL_20260511T023700Z"
    assert row["symbol"] == "CRCL"
    assert row["earnings_date"] == "Mon May 11 2026"
    assert row["run_environment"] == "test"
    assert row["iterative"] is False
    assert row["iterations_completed"] is None
    assert row["synthesizer_provider"] == "openai"
    assert row["synthesizer_model"] == "gpt-5.5"
    assert row["started_at_utc"] is not None and row["started_at_utc"].year == 2026
    assert row["finished_at_utc"] is not None
    assert row["synthesis_path"] == "outputs/CRCL_20260511T023700Z/synthesis.md"
    assert row["config_snapshot"]["config"]["symbol"] == "CRCL"


def test_build_run_row_handles_legacy_string_synthesizer(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    run_dir = outputs / "MNDY_20260509T040039Z"
    run_dir.mkdir()
    # Legacy run.json: providers as list of strings, synthesizer as string, no timestamps.
    data = {
        "config": {
            "symbol": "MNDY",
            "earnings_date": "Mon May 11 2026",
            "providers": ["anthropic", "openai"],
            "synthesizer": "anthropic",
        },
        "dry_run": True,
        "providers": {
            "anthropic": {"enabled": True, "web_search": True},
            "openai": {"enabled": True, "web_search": True},
        },
        "timestamp_utc": "2026-05-09T04:00:39.563609+00:00",
    }
    (run_dir / "run.json").write_text(json.dumps(data), encoding="utf-8")

    row = build_run_row(run_id=run_dir.name, run_dir=run_dir, data=data)

    assert row["symbol"] == "MNDY"
    # No started_at_utc → falls back to timestamp_utc.
    assert row["started_at_utc"] is not None
    assert row["finished_at_utc"] is None
    assert row["synthesizer_provider"] == "anthropic"
    assert row["synthesizer_model"] is None
    assert row["iterative"] is False
    assert row["iterations_completed"] is None
    # Even with no run_environment in config_snapshot, we tolerate it.
    assert row["run_environment"] is None


def test_build_provider_response_rows_non_iterative(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    run_dir = _write_standard_run(outputs, run_id="CRCL_20260511T023700Z", symbol="CRCL")
    data = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))

    rows = build_provider_response_rows(run_id=run_dir.name, run_dir=run_dir, data=data)

    # One fan-out provider row + one synthesizer row.
    assert len(rows) == 2
    fanout = next(r for r in rows if r["iteration"] is None and r["response_path"].endswith("openai.md"))
    assert fanout["provider"] == "openai"
    assert fanout["model"] == "gpt-5.5"
    assert fanout["succeeded"] is True
    assert fanout["latency_s"] == pytest.approx(59.7)
    assert fanout["input_tokens"] == 3019
    assert fanout["output_tokens"] == 4118
    assert fanout["web_search_enabled"] is False
    assert fanout["error_kind"] is None

    syn = next(r for r in rows if r["response_path"].endswith("synthesis.md"))
    assert syn["provider"] == "openai"
    assert syn["succeeded"] is True
    assert syn["latency_s"] == pytest.approx(72.0)


def test_build_provider_response_rows_iterative(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    run_id = "ACHR_20260511T041540Z"
    run_dir = outputs / run_id
    (run_dir / "iterations").mkdir(parents=True)
    (run_dir / "iterations" / "iteration_1_providers.md").write_text("...", encoding="utf-8")
    (run_dir / "iterations" / "iteration_2_providers.md").write_text("...", encoding="utf-8")
    (run_dir / "synthesis.md").write_text("# synth\n", encoding="utf-8")
    data = {
        "config": {
            "symbol": "ACHR",
            "earnings_date": "Mon May 11 2026",
            "providers": [
                {"name": "anthropic", "model": "claude-opus-4-7"},
                {"name": "gemini", "model": "gemini-3-flash-preview"},
            ],
            "synthesizer": {"name": "gemini", "model": "gemini-3.1-pro-preview"},
        },
        "iterative": True,
        "iterations_completed": 2,
        "verification_history": [
            {"verified": ["a", "b"], "contradicted": [], "unverifiable": ["x"]},
            {"verified": ["c"], "contradicted": [], "unverifiable": []},
        ],
    }
    (run_dir / "run.json").write_text(json.dumps(data), encoding="utf-8")

    rows = build_provider_response_rows(run_id=run_id, run_dir=run_dir, data=data)

    # 2 iterations x 2 providers + 1 synthesizer row = 5 rows.
    iter_rows = [r for r in rows if r["iteration"] is not None]
    assert len(iter_rows) == 4
    assert {r["iteration"] for r in iter_rows} == {1, 2}
    assert {r["provider"] for r in iter_rows} == {"anthropic", "gemini"}
    for r in iter_rows:
        assert r["response_path"].endswith(f"iteration_{r['iteration']}_providers.md")
        # Latency / tokens are not preserved in iterative run.json — must be None.
        assert r["latency_s"] is None
        assert r["input_tokens"] is None
    syn_rows = [r for r in rows if r["iteration"] is None]
    assert len(syn_rows) == 1


def test_build_run_row_verifier_summary_sums_history(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    run_id = "ACHR_20260511T041540Z"
    run_dir = outputs / run_id
    run_dir.mkdir()
    data = {
        "config": {"symbol": "ACHR", "earnings_date": "Mon May 11 2026"},
        "iterative": True,
        "iterations_completed": 3,
        "verification_history": [
            {"verified": ["a", "b"], "contradicted": ["x"], "unverifiable": []},
            {"verified": ["c"], "contradicted": [], "unverifiable": ["y", "z"]},
            {"verified": [], "contradicted": ["w"], "unverifiable": []},
        ],
    }
    (run_dir / "run.json").write_text(json.dumps(data), encoding="utf-8")

    row = build_run_row(run_id=run_id, run_dir=run_dir, data=data)

    assert row["verifier_summary"] == {
        "verified": 3,
        "contradicted": 2,
        "unverifiable": 2,
        "rounds": 3,
    }


def test_iter_run_directories_skips_batch_dirs_and_filters(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    _write_standard_run(outputs, run_id="CRCL_20260510T120000Z", symbol="CRCL")
    _write_standard_run(outputs, run_id="MNDY_20260509T120000Z", symbol="MNDY")
    _write_standard_run(outputs, run_id="CRCL_20260511T120000Z", symbol="CRCL")
    # Batch summary dirs have no run.json.
    batch = outputs / "batch_20260511T120000Z"
    batch.mkdir()
    (batch / "batch_summary.txt").write_text("ok\n", encoding="utf-8")

    all_dirs = iter_run_directories(outputs)
    assert [p.name for p in all_dirs] == [
        "CRCL_20260510T120000Z",
        "CRCL_20260511T120000Z",
        "MNDY_20260509T120000Z",
    ]

    only_crcl = iter_run_directories(outputs, symbol="crcl")
    assert [p.name for p in only_crcl] == [
        "CRCL_20260510T120000Z",
        "CRCL_20260511T120000Z",
    ]

    newest_first = iter_run_directories(outputs, oldest_first=False)
    assert [p.name for p in newest_first] == [
        "MNDY_20260509T120000Z",
        "CRCL_20260511T120000Z",
        "CRCL_20260510T120000Z",
    ]

    from datetime import UTC, datetime
    cutoff = datetime(2026, 5, 11, tzinfo=UTC)
    since = iter_run_directories(outputs, since=cutoff)
    assert [p.name for p in since] == ["CRCL_20260511T120000Z"]


class _StubResult:
    def __init__(self, scalar: Any = None) -> None:
        self._scalar = scalar

    def scalar_one_or_none(self) -> Any:
        return self._scalar


class _StubSession:
    """Minimal async session stub that records executed statements.

    The SELECT used by ``backfill_run_directory`` to detect prior existence is
    answered from ``self.existing``; everything else is recorded for inspection.
    """

    def __init__(self, *, existing_run_ids: set[str] | None = None) -> None:
        self.existing: set[str] = set(existing_run_ids or set())
        self.executed: list[Any] = []

    async def execute(self, stmt: Any) -> _StubResult:
        self.executed.append(stmt)
        if isinstance(stmt, Select):
            # The only SELECT in backfill is "select runs.run_id where run_id = :v".
            # Pull the literal out of the compiled SQL params; if absent, return None.
            try:
                compiled = stmt.compile(compile_kwargs={"literal_binds": True})
                sql = str(compiled).lower()
            except Exception:
                sql = ""
            for rid in self.existing:
                if rid.lower() in sql:
                    return _StubResult(rid)
            return _StubResult(None)
        return _StubResult(None)

    async def commit(self) -> None:
        return None

    async def rollback(self) -> None:
        return None


@pytest.mark.asyncio
async def test_backfill_run_directory_inserts_then_idempotent(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    run_dir = _write_standard_run(outputs, run_id="CRCL_20260511T023700Z", symbol="CRCL")

    session = _StubSession()
    r1 = await backfill_run_directory(session, run_dir)  # type: ignore[arg-type]

    assert r1.inserted is True
    assert r1.providers_inserted == 2  # fan-out + synth
    assert r1.skipped_reasons == []

    # Inspect the captured statements: SELECT, INSERT runs (upsert), DELETE provider_responses, INSERT provider_responses.
    assert isinstance(session.executed[0], Select)
    assert isinstance(session.executed[1], Insert)  # upsert into runs
    assert isinstance(session.executed[2], Delete)
    assert session.executed[2].table.name == ProviderResponseRow.__tablename__
    assert isinstance(session.executed[3], Insert)
    assert session.executed[3].table.name == ProviderResponseRow.__tablename__

    # Second invocation: row now "exists" → inserted=False, providers_inserted unchanged.
    session2 = _StubSession(existing_run_ids={"CRCL_20260511T023700Z"})
    r2 = await backfill_run_directory(session2, run_dir)  # type: ignore[arg-type]
    assert r2.inserted is False
    assert r2.providers_inserted == 2
    # Same number of statements (idempotent flow).
    assert len(session2.executed) == len(session.executed)


@pytest.mark.asyncio
async def test_backfill_skips_missing_run_json(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    bare = outputs / "BAD_20260101T000000Z"
    bare.mkdir()

    session = _StubSession()
    result = await backfill_run_directory(session, bare)  # type: ignore[arg-type]

    assert result.inserted is False
    assert result.providers_inserted == 0
    assert result.skipped_reasons == ["missing run.json"]
    assert session.executed == []


@pytest.mark.asyncio
async def test_backfill_two_runs_idempotent_final_state(tmp_path: Path) -> None:
    """Spec requirement: run backfill twice over a fixture set and confirm convergence."""
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    rd1 = _write_standard_run(outputs, run_id="CRCL_20260511T023700Z", symbol="CRCL")
    rd2 = _write_standard_run(outputs, run_id="MNDY_20260509T040039Z", symbol="MNDY")

    # First pass: both new.
    session = _StubSession()
    results1 = [
        await backfill_run_directory(session, rd1),  # type: ignore[arg-type]
        await backfill_run_directory(session, rd2),  # type: ignore[arg-type]
    ]
    assert all(r.inserted for r in results1)
    pr_inserted_first = sum(r.providers_inserted for r in results1)

    # Second pass: both already present.
    session2 = _StubSession(existing_run_ids={rd1.name, rd2.name})
    results2 = [
        await backfill_run_directory(session2, rd1),  # type: ignore[arg-type]
        await backfill_run_directory(session2, rd2),  # type: ignore[arg-type]
    ]
    assert not any(r.inserted for r in results2)
    pr_inserted_second = sum(r.providers_inserted for r in results2)

    # Same provider_responses payload constructed both times.
    assert pr_inserted_first == pr_inserted_second
    # Same shape and count of executed statements per pass.
    assert len(session.executed) == len(session2.executed)


def test_run_row_table_name() -> None:
    # Sanity guard: ensure tests stay aligned with the ORM table names referenced above.
    assert RunRow.__tablename__ == "runs"
    assert ProviderResponseRow.__tablename__ == "provider_responses"
