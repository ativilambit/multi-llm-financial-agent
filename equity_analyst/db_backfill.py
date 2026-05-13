"""Backfill existing ``outputs/<run_id>/run.json`` artifacts into the Postgres tables.

This is an explicit operator command (not a best-effort background hook): if the DB is
unreachable, the caller is expected to fail loud. Each run directory is treated
idempotently — ``runs`` is upserted on its primary key, and ``provider_responses``
rows for the run are deleted and re-inserted so reruns converge on the same final state.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from equity_analyst.config import env_from_persisted_run_json
from equity_analyst.db_models import ProviderResponseRow, RunRow

logger = logging.getLogger(__name__)


_PROVIDER_TO_FILENAME: dict[str, str] = {
    "anthropic": "claude.md",
    "openai": "openai.md",
    "gemini": "gemini.md",
    "grok": "grok.md",
}


def _provider_filename(provider_name: str) -> str:
    return _PROVIDER_TO_FILENAME.get(provider_name, f"{provider_name}.md")


@dataclass(frozen=True)
class BackfillResult:
    """Outcome of a single ``backfill_run_directory`` invocation."""

    run_id: str
    inserted: bool
    providers_inserted: int
    skipped_reasons: list[str] = field(default_factory=list)


def _parse_dt_iso(v: Any) -> datetime | None:
    if not isinstance(v, str):
        return None
    s = v.strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _verifier_summary_from_history(history: Any) -> dict[str, Any] | None:
    if not isinstance(history, list) or not history:
        return None
    verified = sum(
        len(h.get("verified") or []) for h in history if isinstance(h, dict)
    )
    contradicted = sum(
        len(h.get("contradicted") or []) for h in history if isinstance(h, dict)
    )
    unverifiable = sum(
        len(h.get("unverifiable") or []) for h in history if isinstance(h, dict)
    )
    return {
        "verified": verified,
        "contradicted": contradicted,
        "unverifiable": unverifiable,
        "rounds": len(history),
    }


def _normalize_providers(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Coerce a (possibly old-style string-list) providers section to dicts with ``name`` keys."""
    raw = cfg.get("providers")
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for entry in raw:
        if isinstance(entry, str):
            out.append({"name": entry})
        elif isinstance(entry, dict) and isinstance(entry.get("name"), str):
            out.append(dict(entry))
    return out


def _synthesizer_info(cfg: dict[str, Any]) -> tuple[str | None, str | None]:
    """Return (provider_name, model_id) for the synthesizer block, tolerating both shapes."""
    syn = cfg.get("synthesizer")
    if isinstance(syn, str):
        return syn, None
    if isinstance(syn, dict):
        name = syn.get("name") if isinstance(syn.get("name"), str) else None
        model = syn.get("model") if isinstance(syn.get("model"), str) else None
        return name, model
    return None, None


def _started_finished(data: dict[str, Any]) -> tuple[datetime | None, datetime | None]:
    started = _parse_dt_iso(data.get("started_at_utc"))
    finished = _parse_dt_iso(data.get("finished_at_utc"))
    if started is None:
        # Older run.json files only carry timestamp_utc (creation time).
        started = _parse_dt_iso(data.get("timestamp_utc"))
    return started, finished


def _coerce_float(v: Any) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        f = float(v)
        if f != f:  # NaN guard
            return None
        return f
    return None


def _coerce_int(v: Any) -> int | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v == int(v):
        return int(v)
    return None


def build_run_row(*, run_id: str, run_dir: Path, data: dict[str, Any]) -> dict[str, Any]:
    """Construct a ``runs`` row dict from a run directory and parsed ``run.json``.

    Older ``run.json`` files may be missing fields like ``run_environment``,
    ``iterations_completed``, or explicit start/finish timestamps; defaults are
    applied so the row is always insertable.
    """
    cfg_raw = data.get("config")
    cfg: dict[str, Any] = cfg_raw if isinstance(cfg_raw, dict) else {}
    symbol = cfg.get("symbol")
    if not isinstance(symbol, str) or not symbol:
        # Fall back to the leading segment of the run dir name (e.g. CRCL_20260511T...).
        symbol = run_id.split("_", 1)[0] or "UNKNOWN"
    earnings_date = (
        cfg.get("earnings_date") if isinstance(cfg.get("earnings_date"), str) else None
    )
    run_environment = (
        cfg.get("run_environment")
        if isinstance(cfg.get("run_environment"), str)
        else None
    )
    iterative = bool(data.get("iterative", False))
    iterations_completed = _coerce_int(data.get("iterations_completed"))

    syn_provider, syn_model = _synthesizer_info(cfg)
    syn_run = data.get("synthesis") if isinstance(data.get("synthesis"), dict) else None
    if isinstance(syn_run, dict):
        if not syn_provider and isinstance(syn_run.get("provider"), str):
            syn_provider = syn_run["provider"]
        if not syn_model and isinstance(syn_run.get("model"), str):
            syn_model = syn_run["model"]

    started_at, finished_at = _started_finished(data)
    outputs_parent = run_dir.parent.name or "outputs"
    synthesis_rel = f"{outputs_parent}/{run_id}/synthesis.md"

    verifier_summary = _verifier_summary_from_history(data.get("verification_history"))
    drive_folder_url = (
        data.get("drive_folder_url")
        if isinstance(data.get("drive_folder_url"), str)
        else None
    )

    return {
        "run_id": run_id,
        "symbol": symbol,
        "earnings_date": earnings_date,
        "env": env_from_persisted_run_json(data),
        "run_environment": run_environment,
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "iterative": iterative,
        "iterations_completed": iterations_completed,
        "config_snapshot": data,
        "synthesis_path": synthesis_rel,
        "synthesizer_provider": syn_provider,
        "synthesizer_model": syn_model,
        "verifier_summary": verifier_summary,
        "drive_folder_url": drive_folder_url,
        "updated_at_utc": datetime.now(tz=UTC),
    }


def _provider_response_path(
    *, outputs_parent: str, run_id: str, relative: str
) -> str:
    return f"{outputs_parent}/{run_id}/{relative}"


def _provider_succeeded_and_error(model: Any, *, file_exists: bool) -> tuple[bool, str | None]:
    if isinstance(model, str) and model.startswith("error:"):
        return False, model.removeprefix("error:") or None
    return bool(file_exists), None


def build_provider_response_rows(
    *,
    run_id: str,
    run_dir: Path,
    data: dict[str, Any],
) -> list[dict[str, Any]]:
    """Construct ``provider_responses`` rows for one run directory.

    Non-iterative runs use the per-provider file names emitted by ``orchestrator.py``
    (``claude.md`` / ``openai.md`` / ``gemini.md`` / ``grok.md``). Iterative runs
    do not write per-provider files inside ``iterations/``, so each iteration gets
    one row per configured provider pointing at the shared
    ``iterations/iteration_N_providers.md`` blob (latency/token fields are left
    null because that data is not preserved in ``run.json`` after the iterative
    refinement loop completes).
    """
    cfg_raw = data.get("config")
    cfg: dict[str, Any] = cfg_raw if isinstance(cfg_raw, dict) else {}
    providers_cfg = _normalize_providers(cfg)
    providers_meta_raw = data.get("providers")
    providers_meta: dict[str, Any] | None = (
        providers_meta_raw if isinstance(providers_meta_raw, dict) else None
    )
    iterative = bool(data.get("iterative", False))
    iterations_completed = _coerce_int(data.get("iterations_completed"))
    outputs_parent = run_dir.parent.name or "outputs"

    rows: list[dict[str, Any]] = []

    if not iterative and providers_cfg:
        for pc in providers_cfg:
            name = pc["name"]
            fname = _provider_filename(name)
            file_path = run_dir / fname
            meta = providers_meta.get(name) if providers_meta is not None else None
            usage = meta.get("usage") if isinstance(meta, dict) else None
            if not isinstance(usage, dict):
                usage = {}
            model: Any = None
            latency: Any = None
            if isinstance(meta, dict):
                if isinstance(meta.get("model"), str):
                    model = meta["model"]
                latency = meta.get("latency_s")
            if not isinstance(model, str):
                cfg_model = pc.get("model")
                model = cfg_model if isinstance(cfg_model, str) and cfg_model else "unknown"
            succeeded, error_kind = _provider_succeeded_and_error(
                model, file_exists=file_path.is_file()
            )
            web_search = pc.get("web_search")
            rows.append(
                {
                    "run_id": run_id,
                    "iteration": None,
                    "provider": name,
                    "model": model,
                    "latency_s": _coerce_float(latency),
                    "input_tokens": _coerce_int(usage.get("input_tokens")),
                    "output_tokens": _coerce_int(usage.get("output_tokens")),
                    "cache_read_tokens": None,
                    "web_search_enabled": (
                        bool(web_search) if isinstance(web_search, bool) else None
                    ),
                    "succeeded": succeeded,
                    "error_kind": error_kind,
                    "response_path": _provider_response_path(
                        outputs_parent=outputs_parent, run_id=run_id, relative=fname
                    ),
                }
            )
    elif iterative and providers_cfg and isinstance(iterations_completed, int) and iterations_completed > 0:
        for it in range(1, iterations_completed + 1):
            rel = f"iterations/iteration_{it}_providers.md"
            file_path = run_dir / rel
            for pc in providers_cfg:
                name = pc["name"]
                cfg_model = pc.get("model")
                model = cfg_model if isinstance(cfg_model, str) and cfg_model else "unknown"
                web_search = pc.get("web_search")
                rows.append(
                    {
                        "run_id": run_id,
                        "iteration": it,
                        "provider": name,
                        "model": model,
                        "latency_s": None,
                        "input_tokens": None,
                        "output_tokens": None,
                        "cache_read_tokens": None,
                        "web_search_enabled": (
                            bool(web_search) if isinstance(web_search, bool) else None
                        ),
                        "succeeded": file_path.is_file(),
                        "error_kind": None,
                        "response_path": _provider_response_path(
                            outputs_parent=outputs_parent, run_id=run_id, relative=rel
                        ),
                    }
                )

    # Final synthesizer row (best-effort; only emitted when run.json carries any
    # synthesizer signal or synthesis.md exists on disk).
    syn_provider, syn_model = _synthesizer_info(cfg)
    syn_run = data.get("synthesis") if isinstance(data.get("synthesis"), dict) else None
    if isinstance(syn_run, dict):
        if not syn_provider and isinstance(syn_run.get("provider"), str):
            syn_provider = syn_run["provider"]
        if not syn_model and isinstance(syn_run.get("model"), str):
            syn_model = syn_run["model"]
    synthesis_file = run_dir / "synthesis.md"
    if syn_provider and (synthesis_file.is_file() or isinstance(syn_run, dict)):
        usage = syn_run.get("usage") if isinstance(syn_run, dict) else None
        if not isinstance(usage, dict):
            usage = {}
        latency = syn_run.get("latency_s") if isinstance(syn_run, dict) else None
        model = syn_model or "unknown"
        succeeded, error_kind = _provider_succeeded_and_error(
            model, file_exists=synthesis_file.is_file()
        )
        rows.append(
            {
                "run_id": run_id,
                "iteration": None,
                "provider": syn_provider,
                "model": model,
                "latency_s": _coerce_float(latency),
                "input_tokens": _coerce_int(usage.get("input_tokens")),
                "output_tokens": _coerce_int(usage.get("output_tokens")),
                "cache_read_tokens": None,
                "web_search_enabled": None,
                "succeeded": succeeded,
                "error_kind": error_kind,
                "response_path": _provider_response_path(
                    outputs_parent=outputs_parent, run_id=run_id, relative="synthesis.md"
                ),
            }
        )

    return rows


async def backfill_run_directory(
    session: AsyncSession,
    run_dir: Path,
) -> BackfillResult:
    """Backfill a single ``outputs/<run_id>/`` directory into ``runs`` + ``provider_responses``.

    The caller owns transaction boundaries; this function does not commit. ``runs`` is
    upserted on ``run_id``; ``provider_responses`` rows for the run are removed and
    re-inserted so reruns converge on the same final state.
    """
    run_id = run_dir.name
    skipped: list[str] = []
    run_json = run_dir / "run.json"
    if not run_json.is_file():
        return BackfillResult(
            run_id=run_id,
            inserted=False,
            providers_inserted=0,
            skipped_reasons=["missing run.json"],
        )

    try:
        data = json.loads(run_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return BackfillResult(
            run_id=run_id,
            inserted=False,
            providers_inserted=0,
            skipped_reasons=[f"invalid run.json: {type(exc).__name__}: {exc}"],
        )
    if not isinstance(data, dict):
        return BackfillResult(
            run_id=run_id,
            inserted=False,
            providers_inserted=0,
            skipped_reasons=["run.json is not an object"],
        )

    existing_q = await session.execute(
        select(RunRow.run_id).where(RunRow.run_id == run_id)
    )
    already_present = existing_q.scalar_one_or_none() is not None

    run_row = build_run_row(run_id=run_id, run_dir=run_dir, data=data)
    stmt = pg_insert(RunRow).values(**run_row)
    stmt = stmt.on_conflict_do_update(
        index_elements=[RunRow.run_id],
        set_={
            k: stmt.excluded[k]
            for k in run_row
            if k not in ("run_id", "created_at_utc")
        },
    )
    await session.execute(stmt)

    await session.execute(
        delete(ProviderResponseRow).where(ProviderResponseRow.run_id == run_id)
    )
    pr_rows = build_provider_response_rows(run_id=run_id, run_dir=run_dir, data=data)
    if pr_rows:
        await session.execute(pg_insert(ProviderResponseRow).values(pr_rows))

    logger.info(
        "Backfilled run_id=%s providers=%d%s",
        run_id,
        len(pr_rows),
        " (existing row refreshed)" if already_present else "",
    )
    return BackfillResult(
        run_id=run_id,
        inserted=not already_present,
        providers_inserted=len(pr_rows),
        skipped_reasons=skipped,
    )


def _parse_run_dir_timestamp(name: str) -> datetime | None:
    """Parse the trailing ``YYYYMMDDTHHMMSSZ`` segment of a run dir name."""
    if "_" not in name:
        return None
    tail = name.rsplit("_", 1)[1]
    try:
        return datetime.strptime(tail, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    except ValueError:
        return None


def iter_run_directories(
    outputs_dir: Path,
    *,
    symbol: str | None = None,
    since: datetime | None = None,
    oldest_first: bool = True,
) -> list[Path]:
    """List ``outputs/<run_id>/`` directories containing a ``run.json`` for backfill.

    ``batch_<ts>/`` summary directories (no ``run.json``) are skipped silently.
    """
    if not outputs_dir.is_dir():
        return []
    out: list[Path] = []
    sym_prefix = f"{symbol.upper()}_" if symbol else None
    for p in outputs_dir.iterdir():
        if not p.is_dir():
            continue
        if not (p / "run.json").is_file():
            continue
        if sym_prefix is not None and not p.name.upper().startswith(sym_prefix):
            continue
        if since is not None:
            ts = _parse_run_dir_timestamp(p.name)
            if ts is not None and ts < since:
                continue
        out.append(p)
    out.sort(key=lambda d: d.name, reverse=not oldest_first)
    return out
