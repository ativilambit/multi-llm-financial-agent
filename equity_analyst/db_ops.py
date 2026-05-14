from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import delete, insert, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from equity_analyst.config import RunConfig, RunEnvironment, RunProfile, env_from_persisted_run_json
from equity_analyst.db import get_async_session, is_db_available
from equity_analyst.db_models import OutcomeRow, PredictionRow, ProviderResponseRow, RunRow
from equity_analyst.provider_runtime import is_failed_provider_response
from equity_analyst.run_json_serde import canonical_run_document_dict
from equity_analyst.types import ProviderResponse

logger = logging.getLogger(__name__)


async def load_run_document_from_db(
    run_id: str, *, database_url: str | None = None
) -> dict[str, Any] | None:
    """Return ``runs.run_document`` for ``run_id``, or ``None`` if missing / unavailable."""
    if not await is_db_available(database_url=database_url):
        return None
    try:
        async with get_async_session(database_url=database_url) as session:
            doc = await session.scalar(select(RunRow.run_document).where(RunRow.run_id == run_id))
    except Exception as exc:
        logger.warning("load_run_document_from_db failed run_id=%s error=%r", run_id, exc)
        return None
    if doc is None:
        return None
    if not isinstance(doc, dict):
        return None
    return doc


def postgres_metadata_writes_enabled(*, run_profile: RunProfile, env: RunEnvironment) -> bool:
    """True when run/outcome/prediction rows may be written (subject to ``db_enabled`` and DB reachability)."""
    return run_profile == "production" or env == "test"


def _parse_dt_iso(v: Any) -> datetime | None:
    if not v or not isinstance(v, str):
        return None
    s = v.strip()
    if not s:
        return None
    # tolerate trailing 'Z' for UTC
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _relative_under_outputs(run_dir: Path, p: Path) -> str:
    # returns e.g. "<run_id>/synthesis.md" (relative to outputs/)
    try:
        return str(p.relative_to(run_dir.parent))
    except Exception:
        return str(p)


def _verifier_summary_from_history(history: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    if not history:
        return None
    verified = sum(len(h.get("verified") or []) for h in history if isinstance(h, dict))
    contradicted = sum(len(h.get("contradicted") or []) for h in history if isinstance(h, dict))
    unverifiable = sum(len(h.get("unverifiable") or []) for h in history if isinstance(h, dict))
    return {
        "verified": verified,
        "contradicted": contradicted,
        "unverifiable": unverifiable,
        "rounds": len(history),
    }


async def best_effort_upsert_run_and_responses(
    *,
    cfg: RunConfig,
    run_dir: Path,
    run_json_data: dict[str, Any],
    started_at_utc: datetime | None,
    finished_at_utc: datetime | None,
    provider_responses: Iterable[tuple[int | None, str, ProviderResponse, str, bool]],
    synthesis_path: Path,
    database_url: str | None = None,
    run_document: dict[str, Any] | None = None,
) -> None:
    if not cfg.db_enabled:
        return

    source_meta = run_document if run_document is not None else run_json_data
    if source_meta.get("dry_run") is True:
        logger.info("DB write skipped: dry_run=True")
        return
    if not postgres_metadata_writes_enabled(run_profile=cfg.run_profile, env=cfg.env):
        logger.info(
            "DB write skipped: run_profile=%s env=%s (need production profile or test tier)",
            cfg.run_profile,
            cfg.env,
        )
        return

    if not await is_db_available(database_url=database_url):
        logger.warning("DB unavailable; skipping run metadata insert run_id=%s", run_dir.name)
        return

    run_id = run_dir.name
    now = datetime.now(tz=UTC)

    doc_row = canonical_run_document_dict(source_meta)
    cfg_snap = doc_row.get("config")
    cfg_d: dict[str, Any] = cfg_snap if isinstance(cfg_snap, dict) else {}

    symbol = cfg.symbol
    sym_raw = cfg_d.get("symbol")
    if isinstance(sym_raw, str) and sym_raw.strip():
        symbol = sym_raw.strip()

    earnings_date = cfg.earnings_date
    earn_raw = cfg_d.get("earnings_date")
    if isinstance(earn_raw, str) and earn_raw.strip():
        earnings_date = earn_raw.strip()

    run_env_raw = cfg_d.get("run_environment")
    if isinstance(run_env_raw, str) and run_env_raw.strip():
        run_environment = run_env_raw.strip()
    else:
        run_environment = str(cfg.run_environment)

    env_tier = env_from_persisted_run_json(doc_row)

    started_eff = _parse_dt_iso(doc_row.get("started_at_utc")) or started_at_utc
    finished_eff = _parse_dt_iso(doc_row.get("finished_at_utc")) or finished_at_utc

    synth_raw = doc_row.get("synthesis")
    synth: dict[str, Any] = synth_raw if isinstance(synth_raw, dict) else {}
    drive_folder_url = doc_row.get("drive_folder_url")
    if drive_folder_url is not None and not isinstance(drive_folder_url, str):
        drive_folder_url = None

    iterative = bool(doc_row.get("iterative", False))
    iterations_completed = int(doc_row.get("iterations_completed", 0) or 0) or None

    verifier_summary = _verifier_summary_from_history(
        doc_row.get("verification_history")
        if isinstance(doc_row.get("verification_history"), list)
        else None
    )

    run_row: dict[str, Any] = {
        "run_id": run_id,
        "symbol": symbol,
        "earnings_date": earnings_date,
        "env": env_tier,
        "run_environment": run_environment,
        "started_at_utc": started_eff,
        "finished_at_utc": finished_eff,
        "iterative": iterative,
        "iterations_completed": iterations_completed,
        "config_snapshot": doc_row,
        "run_document": doc_row,
        "synthesis_path": _relative_under_outputs(run_dir, synthesis_path),
        "synthesizer_provider": synth.get("provider"),
        "synthesizer_model": synth.get("model"),
        "verifier_summary": verifier_summary,
        "drive_folder_url": drive_folder_url,
        "updated_at_utc": now,
    }

    try:
        async with get_async_session(database_url=database_url) as session:
            stmt = pg_insert(RunRow).values(**run_row)
            stmt = stmt.on_conflict_do_update(
                index_elements=[RunRow.run_id],
                set_={
                    k: stmt.excluded[k] for k in run_row if k not in ("run_id", "created_at_utc")
                },
            )
            await session.execute(stmt)

            pr_rows: list[dict[str, Any]] = []
            for iteration, provider_name, resp, response_path, ws_enabled in provider_responses:
                succeeded = not is_failed_provider_response(resp)
                error_kind = None
                if resp.model.startswith("error:"):
                    error_kind = resp.model.removeprefix("error:") or None
                usage = asdict(resp.usage)
                pr_rows.append(
                    {
                        "run_id": run_id,
                        "iteration": iteration,
                        "provider": provider_name,
                        "model": resp.model,
                        "latency_s": resp.latency_s,
                        "input_tokens": usage.get("input_tokens"),
                        "output_tokens": usage.get("output_tokens"),
                        "cache_read_tokens": None,
                        "web_search_enabled": ws_enabled,
                        "succeeded": succeeded,
                        "error_kind": error_kind,
                        "response_path": response_path,
                    }
                )
            if pr_rows:
                await session.execute(pg_insert(ProviderResponseRow).values(pr_rows))

            await session.commit()

        logger.info("Run record inserted run_id=%s", run_id)
    except Exception as exc:
        logger.warning("DB insert failed run_id=%s error=%r", run_id, exc)


async def best_effort_upsert_outcome(
    *,
    cfg_db_enabled: bool,
    run_id: str,
    outcome: dict[str, Any],
    database_url: str | None = None,
    run_profile: RunProfile = "production",
    env: RunEnvironment = "production",
) -> None:
    if not cfg_db_enabled:
        return
    if not postgres_metadata_writes_enabled(run_profile=run_profile, env=env):
        logger.info(
            "DB write skipped: run_profile=%s env=%s (need production profile or test tier)",
            run_profile,
            env,
        )
        return
    if not await is_db_available(database_url=database_url):
        logger.warning("DB unavailable; skipping outcome upsert run_id=%s", run_id)
        return

    row: dict[str, Any] = {
        "run_id": run_id,
        "earnings_day_open": outcome.get("earnings_day_open"),
        "earnings_day_high": outcome.get("earnings_day_high"),
        "earnings_day_low": outcome.get("earnings_day_low"),
        "earnings_day_close": outcome.get("earnings_day_close"),
        "next_trading_day_open": outcome.get("next_trading_day_open"),
        "next_trading_day_close": outcome.get("next_trading_day_close"),
        "one_week_later_close": outcome.get("one_week_later_close"),
        "direction_vs_prior_close": outcome.get("direction_vs_prior_close"),
        "source": outcome.get("source") or "manual",
        "notes": outcome.get("notes"),
    }

    try:
        async with get_async_session(database_url=database_url) as session:
            stmt = pg_insert(OutcomeRow).values(**row)
            stmt = stmt.on_conflict_do_update(
                index_elements=[OutcomeRow.run_id],
                set_={k: stmt.excluded[k] for k in row if k != "run_id"},
            )
            await session.execute(stmt)
            await session.commit()
        logger.info("Outcome upserted run_id=%s", run_id)
    except Exception as exc:
        logger.warning("DB outcome upsert failed run_id=%s error=%r", run_id, exc)


async def best_effort_replace_predictions(
    *,
    cfg_db_enabled: bool,
    run_id: str,
    rows: list[dict[str, Any]],
    database_url: str | None = None,
    run_profile: RunProfile = "production",
    env: RunEnvironment = "production",
) -> bool:
    """DELETE existing ``predictions`` for ``run_id`` then bulk INSERT ``rows``."""
    if not cfg_db_enabled:
        logger.warning(
            "prediction_extract: DB writes disabled; skipping Postgres run_id=%s", run_id
        )
        return False
    if not postgres_metadata_writes_enabled(run_profile=run_profile, env=env):
        logger.info(
            "DB write skipped: run_profile=%s env=%s (need production profile or test tier)",
            run_profile,
            env,
        )
        return False
    if not await is_db_available(database_url=database_url):
        logger.warning("prediction_extract: DB unavailable run_id=%s", run_id)
        return False
    try:
        async with get_async_session(database_url=database_url) as session:
            await session.execute(delete(PredictionRow).where(PredictionRow.run_id == run_id))
            if rows:
                await session.execute(insert(PredictionRow), rows)
            await session.commit()
        logger.info(
            "prediction_extract: Postgres updated run_id=%s rows=%s",
            run_id,
            len(rows),
        )
        return True
    except Exception as exc:
        logger.warning("prediction_extract: DB replace failed run_id=%s error=%r", run_id, exc)
        return False
