"""Materialize a minimal ``outputs/<run_id>/`` tree from Postgres for disk-dependent CLIs."""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from equity_analyst.db_ops import load_run_document_from_db, load_synthesis_markdown_from_db
from equity_analyst.run_json_serde import format_run_json_for_disk


@dataclass(frozen=True)
class MaterializedRunDir:
    """``run_dir`` is ``<scratch_root>/outputs/<run_id>/``; delete ``scratch_root`` when done."""

    run_dir: Path
    scratch_root: Path


async def materialize_min_outputs_run_dir(
    *,
    run_id: str,
    database_url: str | None = None,
    include_synthesis_from_db: bool = False,
) -> MaterializedRunDir:
    """Write ``run.json`` from ``runs.run_document``; optionally ``synthesis.md`` from DB column."""
    doc = await load_run_document_from_db(run_id, database_url=database_url)
    if not isinstance(doc, dict):
        raise FileNotFoundError(
            f"runs.run_document missing for run_id={run_id!r} (cannot materialize run dir)"
        )

    scratch_root = Path(tempfile.mkdtemp(prefix="eq_run_mat_"))
    run_dir = scratch_root / "outputs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(format_run_json_for_disk(doc), encoding="utf-8")

    if include_synthesis_from_db:
        syn = await load_synthesis_markdown_from_db(run_id, database_url=database_url)
        if syn:
            (run_dir / "synthesis.md").write_text(syn, encoding="utf-8")

    return MaterializedRunDir(run_dir=run_dir, scratch_root=scratch_root)


def cleanup_materialized_run_dir(mat: MaterializedRunDir | None) -> None:
    if mat is None:
        return
    shutil.rmtree(mat.scratch_root, ignore_errors=True)


async def ensure_run_dir_for_cli(
    *,
    run_id: str,
    outputs_dir: Path,
    database_url: str | None = None,
    include_synthesis_from_db: bool = False,
) -> tuple[Path, MaterializedRunDir | None]:
    """Return ``(run_dir, materialized_or_none)`` — materialize when ``outputs_dir/run_id`` is absent."""
    outputs_dir = outputs_dir.expanduser().resolve()
    candidate = outputs_dir / run_id
    if candidate.is_dir():
        return candidate, None
    mat = await materialize_min_outputs_run_dir(
        run_id=run_id,
        database_url=database_url,
        include_synthesis_from_db=include_synthesis_from_db,
    )
    return mat.run_dir, mat
