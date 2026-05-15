"""GitHub Actions helper: prediction extraction for runs without ``predictions`` rows.

Requires non-empty ``runs.synthesis_markdown`` (see migration ``0005``) and ``run_document``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


async def _run_async(args: argparse.Namespace) -> int:
    from equity_analyst.config import RunConfig, run_profile_from_persisted_run_json
    from equity_analyst.db_maintenance import select_run_ids_missing_predictions
    from equity_analyst.db_run_materialize import (
        cleanup_materialized_run_dir,
        materialize_min_outputs_run_dir,
    )
    from equity_analyst.prediction_extract import run_prediction_extract_for_run_dir

    sym_list: list[str] | None = None
    if args.symbols.strip():
        sym_list = [s.strip() for s in args.symbols.split(",") if s.strip()]

    run_ids = await select_run_ids_missing_predictions(
        lookback_days=int(args.lookback_days),
        limit=int(args.limit),
        symbols=sym_list,
        database_url=None,
    )
    if not run_ids:
        sys.stdout.write("gh_predictions_from_db: no matching run_id(s)\n")
        return 0

    if args.dry_run:
        for rid in run_ids:
            sys.stdout.write(f"DRY-RUN would predictions-extract run_id={rid}\n")
        return 0

    failed = 0
    for rid in run_ids:
        mat = await materialize_min_outputs_run_dir(
            run_id=rid,
            database_url=None,
            include_synthesis_from_db=True,
        )
        try:
            raw = json.loads((mat.run_dir / "run.json").read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("run.json is not an object")
            cfg_raw = raw.get("config")
            if not isinstance(cfg_raw, dict):
                raise ValueError("missing config in run_document")
            effective_rp = run_profile_from_persisted_run_json(raw)
            cfg = RunConfig.model_validate(cfg_raw).model_copy(update={"run_profile": effective_rp})
            await run_prediction_extract_for_run_dir(run_dir=mat.run_dir, cfg=cfg)
        except Exception as exc:
            failed += 1
            logger.error("predictions-extract failed run_id=%s: %s", rid, exc)
            if not args.continue_on_error:
                return 1
        finally:
            cleanup_materialized_run_dir(mat)

    if failed:
        sys.stdout.write(f"gh_predictions_from_db: completed with {failed} failure(s)\n")
        return 1
    sys.stdout.write(f"gh_predictions_from_db: processed {len(run_ids)} run(s)\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv(override=False)
    p = argparse.ArgumentParser(
        description="DB-only batch prediction extraction for GitHub Actions."
    )
    p.add_argument("--lookback-days", type=int, default=14)
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--symbols", default="", help="Comma-separated tickers (optional).")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    args = p.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level)), format="%(levelname)s %(message)s"
    )
    return asyncio.run(_run_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
