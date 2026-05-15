"""GitHub Actions helper: record outcomes for runs missing ``outcomes`` rows (Postgres-only).

Selects recent ``runs`` with ``run_document`` and no matching ``outcomes`` row, materializes
``outputs/<run_id>/run.json`` under a temp tree, then calls :func:`record_outcome_for_run_dir`.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


async def _run_async(args: argparse.Namespace) -> int:
    from equity_analyst.db_maintenance import select_run_ids_missing_outcomes
    from equity_analyst.db_run_materialize import (
        cleanup_materialized_run_dir,
        materialize_min_outputs_run_dir,
    )
    from equity_analyst.outcome_tracker import record_outcome_for_run_dir

    sym_list: list[str] | None = None
    if args.symbols.strip():
        sym_list = [s.strip() for s in args.symbols.split(",") if s.strip()]

    run_ids = await select_run_ids_missing_outcomes(
        lookback_days=int(args.lookback_days),
        limit=int(args.limit),
        symbols=sym_list,
        database_url=None,
    )
    if not run_ids:
        sys.stdout.write("gh_outcomes_from_db: no matching run_id(s)\n")
        return 0

    if args.dry_run:
        for rid in run_ids:
            sys.stdout.write(f"DRY-RUN would outcome-record run_id={rid}\n")
        return 0

    failed = 0
    for rid in run_ids:
        mat = await materialize_min_outputs_run_dir(
            run_id=rid,
            database_url=None,
            include_synthesis_from_db=False,
        )
        try:
            record_outcome_for_run_dir(
                run_dir=mat.run_dir,
                auto_fetch=bool(args.auto_fetch),
                dry_run=False,
                source="manual",
                db_upsert=True,
            )
        except Exception as exc:
            failed += 1
            logger.error("outcome-record failed run_id=%s: %s", rid, exc)
            if not args.continue_on_error:
                return 1
        finally:
            cleanup_materialized_run_dir(mat)

    if failed:
        sys.stdout.write(f"gh_outcomes_from_db: completed with {failed} failure(s)\n")
        return 1
    sys.stdout.write(f"gh_outcomes_from_db: processed {len(run_ids)} run(s)\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv(override=False)
    p = argparse.ArgumentParser(description="DB-only batch outcome recording for GitHub Actions.")
    p.add_argument("--lookback-days", type=int, default=14)
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--symbols", default="", help="Comma-separated tickers (optional).")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--auto-fetch", action="store_true")
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
