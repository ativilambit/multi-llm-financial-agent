from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, cast

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from equity_analyst.config import RunConfig, load_config, run_profile_from_persisted_run_json
from equity_analyst.db_ops import best_effort_upsert_run_and_responses
from equity_analyst.drive_uploader import log_drive_upload_plan_from_config
from equity_analyst.iterative import (
    build_initial_refinement_state,
    compile_refinement_workflow,
    dry_run_compile_only,
)
from equity_analyst.logging_setup import attach_run_file_logging, configure_cli_logging
from equity_analyst.orchestrator import Orchestrator
from equity_analyst.outcome_tracker import (
    merge_auto_fetch_into_cli_fields,
    parse_output_dirs_from_batch_summary,
    plan_shape_b_run_directories,
    record_outcome_for_run_dir,
)
from equity_analyst.prediction_extract import run_prediction_extract_for_run_dir
from equity_analyst.prompt_export import use_prompt_exporter
from equity_analyst.prompting import render_prompt
from equity_analyst.providers.registry import ProviderRegistry
from equity_analyst.synthesizer_blend import normalize_t0_blend_preset

logger = logging.getLogger(__name__)


def _parse_dt_iso(v: Any) -> datetime | None:
    if not v or not isinstance(v, str):
        return None
    s = v.strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _dict_to_provider_response(d: dict[str, Any]) -> Any:
    # Import locally to avoid import cycles.
    from equity_analyst.types import ProviderResponse, ProviderUsage

    u = d.get("usage") or {}
    return ProviderResponse(
        provider_name=str(d["provider_name"]),
        model=str(d["model"]),
        text=str(d["text"]),
        usage=ProviderUsage(
            input_tokens=u.get("input_tokens"),
            output_tokens=u.get("output_tokens"),
            total_tokens=u.get("total_tokens"),
        ),
        latency_s=d.get("latency_s"),
        raw=None,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="equity_analyst")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run multi-provider analysis and synthesis")
    run.add_argument("--symbol", required=False, help="Override symbol from config")
    run.add_argument(
        "--config",
        required=False,
        help="YAML config path or '-' for stdin (optional when --resume supplies run.json)",
    )
    run.add_argument(
        "--prompt-file",
        required=False,
        help="Optional Jinja2 template override path (defaults to prompts/equity_analyst.j2)",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="Render prompt and show requests without calling any provider APIs",
    )
    run.add_argument(
        "--web-search",
        "--enable-web-search",
        dest="enable_web_search",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable web search tools where supported; use --no-web-search for faster runs. "
        "Legacy: --enable-web-search / --no-enable-web-search.",
    )
    run.add_argument(
        "--iterative",
        action="store_true",
        help="Run LangGraph refinement loop (fan-out, synthesize, verify, route, finalize)",
    )
    run.add_argument(
        "--facts-packet",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Iterative: enable frozen facts packet after round 1 (default from RunConfig / FACTS_PACKET_ENABLED).",
    )
    run.add_argument(
        "--conditional-fanout",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Iterative: after round 1, skip fan-out unless verifier requests re-fan-out "
        "(default from RunConfig / CONDITIONAL_FANOUT_ENABLED).",
    )
    run.add_argument(
        "--no-db",
        action="store_true",
        help="Disable best-effort Postgres writes for this run (file artifacts are still written).",
    )
    run.add_argument(
        "--profile",
        dest="run_profile",
        choices=["production", "dev"],
        default=None,
        help="Run profile: only production persists runs/responses/outcomes/predictions to Postgres. "
        "Default from RunConfig / EQUITY_RUN_PROFILE / RUN_PROFILE (default dev).",
    )
    run.add_argument(
        "--t0-blend",
        dest="t0_blend_preset",
        choices=["default", "quant_lean", "quant_dominant", "qual_dominant"],
        default=None,
        help="Override RunConfig.t0_blend_preset for T-0 horizon qual:quant digits (T-3..T-1 and T+1..T+5 "
        "unchanged). Default from YAML / EQUITY_T0_BLEND_PRESET.",
    )
    run.add_argument("--max-iterations", type=int, default=3)
    run.add_argument("--confidence-threshold", type=float, default=0.85)
    run.add_argument(
        "--resume",
        default=None,
        help="Output folder name under outputs/ (checkpoint at outputs/<id>/checkpoint.sqlite)",
    )
    run.add_argument(
        "--keep-checkpoint",
        action="store_true",
        help="Iterative: keep checkpoint.sqlite (and wal/shm/journal siblings) after a successful run; "
        "overrides delete_checkpoint_after_success / DELETE_CHECKPOINT_AFTER_SUCCESS.",
    )
    run.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level for the equity_analyst logger (stderr and optional per-run agent.log)",
    )
    run.add_argument(
        "--retry-max-attempts",
        type=int,
        default=None,
        help="Override RunConfig.retry_max_attempts (default from YAML or 3)",
    )
    run.add_argument(
        "--retry-base-delay-s",
        type=float,
        default=None,
        help="Override RunConfig.retry_base_delay_s (default from YAML or 2.0)",
    )
    run.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Override RunConfig.max_output_tokens for fan-out providers (default from YAML or 16000)",
    )
    run.add_argument(
        "--verifier-max-output-tokens",
        type=int,
        default=None,
        help="Override RunConfig.verifier_max_output_tokens (iterative verifier only; default 16384)",
    )
    run.add_argument(
        "--verifier-provider",
        default=None,
        help="Override RunConfig.verifier_provider (iterative verify step; default gemini)",
    )
    run.add_argument(
        "--verifier-model",
        default=None,
        help="Override RunConfig.verifier_model (optional API model id for the verifier)",
    )
    run.add_argument(
        "--synthesizer-max-input-tokens",
        type=int,
        default=None,
        help="Override RunConfig.synthesizer_max_input_tokens (default from YAML or 100000)",
    )
    run.add_argument(
        "--synthesizer-max-output-tokens",
        type=int,
        default=None,
        help="Override RunConfig.synthesizer_max_output_tokens (default from YAML or 24000)",
    )
    run.add_argument(
        "--no-summarize-oversized",
        action="store_true",
        help="Disable Gemini Flash pre-summarization of oversized provider bodies before synthesis.",
    )
    run.add_argument(
        "--summarize-threshold-tokens",
        type=int,
        default=None,
        help="Override RunConfig.summarize_threshold_input_tokens (default 8000 estimated tokens per body)",
    )
    run.add_argument(
        "--no-prompt-cache",
        action="store_true",
        help="Disable prompt caching for Anthropic (system/tools) and Gemini explicit context caches.",
    )
    run.add_argument(
        "--no-force-tool-use",
        action="store_true",
        help="Do not force Anthropic tool_choice when web search is enabled (default: force at least one tool).",
    )
    run.add_argument(
        "--upload-to-drive",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable Google Drive upload for this run (overrides YAML / DRIVE_UPLOAD_ENABLED).",
    )
    run.add_argument(
        "--drive-folder-id",
        default=None,
        help="Google Drive folder ID for this run's upload root (overrides drive_root_folder_id / env).",
    )
    run.add_argument(
        "--environment",
        "--env",
        dest="run_environment",
        choices=["production", "test"],
        default=None,
        help="Run environment for Drive uploads: ``production`` → child folder ``prod``; ``test`` → ``test``. "
        "Overrides run_environment / RUN_ENVIRONMENT when set.",
    )
    run.add_argument(
        "--drive-auth-mode",
        choices=["service_account", "oauth_user"],
        default=None,
        help="Google Drive auth mode (overrides drive_auth_mode / DRIVE_AUTH_MODE).",
    )
    run.add_argument(
        "--pdf",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="pdf_output_enabled",
        help="Emit PDF alongside primary analysis markdown (default on). Use --no-pdf to disable.",
    )
    run.add_argument(
        "--extract-predictions",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="After a successful run, extract prediction horizons from synthesis into Postgres "
        "(uses prediction_extract_* RunConfig; default off unless enabled in YAML).",
    )

    outcome = sub.add_parser("outcome-record", help="Record realized outcomes for a prior run")
    outcome.add_argument("--run-dir", required=True, help="Absolute or relative path to outputs/<run>/")
    outcome.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for missing fields on stdin (press Enter to skip a field).",
    )
    outcome.add_argument("--earnings-day-open", type=float, default=None)
    outcome.add_argument("--earnings-day-high", type=float, default=None)
    outcome.add_argument("--earnings-day-low", type=float, default=None)
    outcome.add_argument("--earnings-day-close", type=float, default=None)
    outcome.add_argument("--next-trading-day-open", type=float, default=None)
    outcome.add_argument("--next-trading-day-close", type=float, default=None)
    outcome.add_argument("--one-week-later-close", type=float, default=None)
    outcome.add_argument("--direction-vs-prior-close", choices=["up", "down", "flat"], default=None)
    outcome.add_argument("--notes", default=None)
    outcome.add_argument(
        "--source",
        choices=["manual", "yahoo_csv", "alpaca", "polygon"],
        default="manual",
        help="How the realized outcomes were sourced.",
    )
    outcome.add_argument(
        "--auto-fetch",
        action="store_true",
        help="Fetch earnings-day OHLC + next trading day OHLC + ~5 trading days later close "
        "from Yahoo Finance via yfinance. Explicit --earnings-day-* / --next-trading-day-* "
        "/ --one-week-later-close / --direction-vs-prior-close flags override fetched values.",
    )

    out_batch = sub.add_parser(
        "outcome-record-batch",
        help="Record outcomes for all runs in a batch (summary file) or a symbol list",
    )
    out_batch.add_argument(
        "--batch-dir",
        default=None,
        help="Shape A: path to outputs/batch_<ts>/ (reads batch_summary.txt for output_dir= lines)",
    )
    out_batch.add_argument(
        "--symbols",
        default=None,
        help="Shape B: comma-separated tickers (exclusive with --batch-dir)",
    )
    out_batch.add_argument(
        "--symbols-file",
        type=Path,
        default=None,
        help="Shape B: file with tickers (one per line or comma-separated; # comments allowed)",
    )
    out_batch.add_argument(
        "--since",
        default=None,
        help="Shape B: only runs with directory timestamp on or after this date (YYYY-MM-DD). "
        "Default: 7 calendar days ago (UTC midnight).",
    )
    out_batch.add_argument(
        "--newest-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shape B: only the newest matching run per symbol (default: true).",
    )
    out_batch.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Shape B: directory containing per-run folders (default: outputs/).",
    )
    out_batch.add_argument(
        "--auto-fetch",
        action="store_true",
        help="Same as outcome-record --auto-fetch (yfinance merge before persisting).",
    )
    out_batch.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions only; do not write outcome.json, registry, or DB.",
    )
    out_batch.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep processing after a per-run failure (default: true). Use --no-continue-on-error to stop.",
    )
    out_batch.add_argument(
        "--rate-limit-sleep-s",
        type=float,
        default=0.5,
        help="Sleep this many seconds between symbols after the first (default: 0.5; yfinance courtesy).",
    )
    out_batch.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level for the equity_analyst logger (default INFO).",
    )

    pred_ex = sub.add_parser(
        "predictions-extract",
        help="LLM-extract structured prediction horizons from a run's synthesis into Postgres",
    )
    pred_ex.add_argument("--run-dir", required=True, help="Path to outputs/<run_id>/")
    pred_ex.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level for the equity_analyst logger (default INFO).",
    )

    pred_batch = sub.add_parser(
        "predictions-extract-batch",
        help="Run predictions-extract for each run in a batch summary or a symbol list",
    )
    pred_batch.add_argument(
        "--batch-dir",
        default=None,
        help="Shape A: path to outputs/batch_<ts>/ (reads batch_summary.txt for output_dir= lines)",
    )
    pred_batch.add_argument(
        "--symbols",
        default=None,
        help="Shape B: comma-separated tickers (exclusive with --batch-dir)",
    )
    pred_batch.add_argument(
        "--symbols-file",
        type=Path,
        default=None,
        help="Shape B: file with tickers (one per line or comma-separated; # comments allowed)",
    )
    pred_batch.add_argument(
        "--since",
        default=None,
        help="Shape B: only runs with directory timestamp on or after this date (YYYY-MM-DD). "
        "Default: 7 calendar days ago (UTC midnight).",
    )
    pred_batch.add_argument(
        "--newest-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shape B: only the newest matching run per symbol (default: true).",
    )
    pred_batch.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Shape B: directory containing per-run folders (default: outputs/).",
    )
    pred_batch.add_argument(
        "--dry-run",
        action="store_true",
        help="Print run dirs only; do not call the extractor or write DB/JSON.",
    )
    pred_batch.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep processing after a per-run failure (default: true). Use --no-continue-on-error to stop.",
    )
    pred_batch.add_argument(
        "--rate-limit-sleep-s",
        type=float,
        default=0.0,
        help="Sleep this many seconds between runs after the first (default: 0).",
    )
    pred_batch.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level for the equity_analyst logger (default INFO).",
    )

    backfill = sub.add_parser(
        "db-backfill",
        help="Backfill existing outputs/<run-id>/ run.json artifacts into Postgres (idempotent).",
    )
    backfill.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Root directory containing per-run output folders (default: outputs/).",
    )
    backfill.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Backfill at most N runs (oldest first by default; use --newest-first to flip).",
    )
    backfill.add_argument(
        "--newest-first",
        action="store_true",
        help="Walk runs newest first instead of the default oldest-first ordering.",
    )
    backfill.add_argument(
        "--dry-run",
        action="store_true",
        help="List runs that would be backfilled without writing to the database.",
    )
    backfill.add_argument(
        "--symbol",
        default=None,
        help="Only backfill runs whose directory name starts with this symbol (case-insensitive).",
    )
    backfill.add_argument(
        "--since",
        default=None,
        help="Only backfill runs whose directory timestamp is >= this date (YYYY-MM-DD).",
    )
    backfill.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level for the equity_analyst logger (default INFO).",
    )

    return parser


def _default_since_utc_midnight() -> datetime:
    d = datetime.now(tz=UTC).date() - timedelta(days=7)
    return datetime(d.year, d.month, d.day, tzinfo=UTC)


def _load_batch_symbols_csv_and_file(symbols: str | None, symbols_file: Path | None) -> list[str]:
    out: list[str] = []
    if symbols:
        out.extend(p.strip() for p in symbols.split(",") if p.strip())
    if symbols_file is not None:
        if not symbols_file.is_file():
            raise SystemExit(f"--symbols-file not found: {symbols_file}")
        for line in symbols_file.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if "," in s:
                out.extend(p.strip() for p in s.split(",") if p.strip())
            else:
                out.append(s)
    return out


def _symbol_prefix_from_run_id(run_id: str) -> str:
    if "_" not in run_id:
        return run_id
    return run_id.rsplit("_", 1)[0]


def _format_earnings_close_for_line(v: float | None) -> str:
    if v is None:
        return "null"
    return f"{v:.2f}"


def _print_outcome_batch_ok_line(
    *,
    symbol: str,
    run_id: str,
    earnings_close: float | None,
    auto_fetch: bool,
    direction: str | None,
) -> None:
    auto_s = "(auto)" if auto_fetch else ""
    dir_s = direction if direction is not None else "null"
    sys.stdout.write(
        f"[OK]    {symbol:6} outcome_recorded run={run_id}  "
        f"earnings_close={_format_earnings_close_for_line(earnings_close)} {auto_s}  direction={dir_s}\n"
    )


def _print_outcome_batch_warn_line(*, symbol: str, message: str) -> None:
    sys.stdout.write(f"[WARN]  {symbol:6} {message}\n")


def _print_outcome_batch_fail_line(*, symbol: str, message: str) -> None:
    sys.stdout.write(f"[FAIL]  {symbol:6} {message}\n")


def _apply_cli_config_overrides(cfg: RunConfig, args: argparse.Namespace) -> RunConfig:
    patch: dict[str, Any] = {}
    if getattr(args, "no_db", False):
        patch["db_enabled"] = False
    if getattr(args, "run_profile", None) is not None:
        patch["run_profile"] = args.run_profile
    if getattr(args, "t0_blend_preset", None) is not None:
        patch["t0_blend_preset"] = normalize_t0_blend_preset(str(args.t0_blend_preset))
    if args.retry_max_attempts is not None:
        patch["retry_max_attempts"] = args.retry_max_attempts
    if args.retry_base_delay_s is not None:
        patch["retry_base_delay_s"] = args.retry_base_delay_s
    if args.max_output_tokens is not None:
        patch["max_output_tokens"] = args.max_output_tokens
    if args.verifier_max_output_tokens is not None:
        patch["verifier_max_output_tokens"] = args.verifier_max_output_tokens
    if getattr(args, "verifier_provider", None) is not None:
        patch["verifier_provider"] = str(args.verifier_provider)
    if getattr(args, "verifier_model", None) is not None:
        patch["verifier_model"] = str(args.verifier_model)
    if args.synthesizer_max_input_tokens is not None:
        patch["synthesizer_max_input_tokens"] = args.synthesizer_max_input_tokens
    if args.synthesizer_max_output_tokens is not None:
        patch["synthesizer_max_output_tokens"] = args.synthesizer_max_output_tokens
    if getattr(args, "no_summarize_oversized", False):
        patch["summarize_oversized_providers"] = False
    if getattr(args, "summarize_threshold_tokens", None) is not None:
        patch["summarize_threshold_input_tokens"] = int(args.summarize_threshold_tokens)
    if getattr(args, "no_prompt_cache", False):
        patch["prompt_cache_enabled"] = False
    if getattr(args, "no_force_tool_use", False):
        patch["anthropic_force_tool_use"] = False
    if getattr(args, "upload_to_drive", None) is not None:
        patch["drive_upload_enabled"] = bool(args.upload_to_drive)
    if getattr(args, "drive_folder_id", None):
        patch["drive_root_folder_id"] = str(args.drive_folder_id)
    if getattr(args, "drive_auth_mode", None) is not None:
        patch["drive_auth_mode"] = str(args.drive_auth_mode)
    if getattr(args, "pdf_output_enabled", None) is not None:
        patch["pdf_output_enabled"] = bool(args.pdf_output_enabled)
    if getattr(args, "run_environment", None) is not None:
        patch["run_environment"] = str(args.run_environment)
    if getattr(args, "extract_predictions", None) is not None:
        patch["prediction_extract_enabled"] = bool(args.extract_predictions)
    if getattr(args, "keep_checkpoint", False):
        patch["delete_checkpoint_after_success"] = False
    if getattr(args, "facts_packet", None) is not None:
        patch["facts_packet_enabled"] = bool(args.facts_packet)
    if getattr(args, "conditional_fanout", None) is not None:
        patch["conditional_fanout_enabled"] = bool(args.conditional_fanout)
    return cfg if not patch else cfg.model_copy(update=patch)


def _load_cfg(args: argparse.Namespace) -> RunConfig:
    if args.config:
        return load_config(args.config)
    if args.iterative and args.resume:
        run_json = Path("outputs") / args.resume / "run.json"
        data = json.loads(run_json.read_text(encoding="utf-8"))
        return RunConfig.model_validate(data["config"])
    raise SystemExit("--config is required (unless --iterative --resume with run.json)")


async def _run_iterative_cli(
    args: argparse.Namespace,
    cfg: RunConfig,
    prompt_path: Path | None,
) -> tuple[str, Path, dict[str, Any]]:
    pp = prompt_path or Path("prompts/equity_analyst.j2")
    rendered = render_prompt(cfg, pp)
    reg = ProviderRegistry.default()
    if args.dry_run:
        logger.info(
            "Iterative dry-run: no output directory is created; per-run agent.log is not written "
            "(see README logging section).",
        )
        nodes = dry_run_compile_only(registry=reg)
        return (
            "# Iterative dry-run\n\n"
            f"Graph nodes: {', '.join(nodes)}\n\n"
            "## Rendered prompt (excerpt)\n\n"
            + rendered.text[:8000]
        ), Path("."), {}
    out_dir: Path
    thread_id: str
    resume = bool(args.resume)
    if resume:
        out_dir = Path("outputs") / args.resume
        if not out_dir.is_dir():
            raise FileNotFoundError(f"missing output dir {out_dir}")
        thread_id = args.resume
    else:
        ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        out_dir = Path("outputs") / f"{cfg.symbol}_{ts}"
        out_dir.mkdir(parents=True, exist_ok=False)
        thread_id = out_dir.name
    attach_run_file_logging(out_dir / "agent.log")
    cfg = log_drive_upload_plan_from_config(cfg)
    ckpt = out_dir / "checkpoint.sqlite"
    logger.info(
        "Iterative CLI output_dir=%s resume=%s thread_id=%s checkpoint=%s",
        str(out_dir.resolve()),
        resume,
        thread_id,
        str(ckpt.resolve()),
    )
    it_dir = out_dir / "iterations"
    it_dir.mkdir(parents=True, exist_ok=True)
    started_at_utc = datetime.now(tz=UTC).replace(microsecond=0).isoformat()
    meta = {
        "iterative": True,
        "thread_id": thread_id,
        "template_path": rendered.template_path,
        "config": cfg.model_dump(),
        "run_profile": cfg.run_profile,
        "started_at_utc": started_at_utc,
        "max_iterations": args.max_iterations,
        "confidence_threshold": args.confidence_threshold,
        "errors": [],
    }
    if not resume:
        (out_dir / "run.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    async with AsyncSqliteSaver.from_conn_string(str(ckpt)) as saver:
        app = compile_refinement_workflow(registry=reg, checkpointer=saver)
        config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
        with use_prompt_exporter(out_dir):
            if resume:
                final_state = await app.ainvoke(None, config=config)
            else:
                st = build_initial_refinement_state(cfg=cfg, rendered=rendered, output_dir=out_dir)
                st["max_iterations"] = args.max_iterations
                st["confidence_threshold"] = args.confidence_threshold
                st["enable_web_search"] = args.enable_web_search
                final_state = await app.ainvoke(st, config=config)
    logger.info("Iterative run finished output_dir=%s", str(out_dir.resolve()))
    return str(final_state.get("final_report", "")), out_dir, final_state


def _run_outcome_record_batch_cli(args: argparse.Namespace) -> int:
    configure_cli_logging(getattr(logging, str(args.log_level)))
    sym_list = _load_batch_symbols_csv_and_file(args.symbols, args.symbols_file)
    if args.batch_dir and sym_list:
        raise SystemExit("--batch-dir cannot be combined with --symbols / --symbols-file")
    if not args.batch_dir and not sym_list:
        raise SystemExit(
            "Provide --batch-dir (Shape A) or --symbols / --symbols-file (Shape B).",
        )

    work: list[tuple[str, Path | None]] = []
    since_note = ""
    if args.batch_dir:
        batch_dir = Path(str(args.batch_dir)).expanduser()
        summary = batch_dir / "batch_summary.txt"
        try:
            dirs = parse_output_dirs_from_batch_summary(summary)
        except OSError as exc:
            raise SystemExit(f"Could not read batch summary: {exc}") from exc
        for d in dirs:
            run_dir = d.expanduser()
            rid = run_dir.name
            work.append((_symbol_prefix_from_run_id(rid), run_dir))
    else:
        since = _parse_since_date(str(args.since)) if args.since else _default_since_utc_midnight()
        since_s = since.date().isoformat()
        since_note = since_s
        outputs_dir = Path(str(args.outputs_dir)).expanduser()
        work = plan_shape_b_run_directories(
            outputs_dir,
            sym_list,
            since,
            newest_only=bool(args.newest_only),
        )

    attempted = len(work)
    recorded = 0
    partial = 0
    skipped = 0
    failed = 0
    any_fail_line = False

    for idx, (sym_hint, run_dir_raw) in enumerate(work):
        if idx > 0 and float(args.rate_limit_sleep_s) > 0:
            time.sleep(float(args.rate_limit_sleep_s))

        if run_dir_raw is None:
            skipped += 1
            any_fail_line = True
            since_msg = f"since {since_note}" if since_note else "since cutoff"
            _print_outcome_batch_fail_line(
                symbol=sym_hint,
                message=f"no run dir found {since_msg}              → skipped",
            )
            continue

        run_dir = run_dir_raw.expanduser()
        sym = sym_hint
        try:
            if not run_dir.is_dir():
                raise FileNotFoundError(f"missing run directory: {run_dir}")
            res = record_outcome_for_run_dir(
                run_dir=run_dir,
                auto_fetch=bool(args.auto_fetch),
                dry_run=bool(args.dry_run),
                source="manual",
                db_upsert=not bool(args.dry_run),
            )
        except Exception as exc:
            failed += 1
            any_fail_line = True
            _print_outcome_batch_fail_line(
                symbol=sym,
                message=f"{type(exc).__name__}: {exc}              → skipped",
            )
            if not bool(args.continue_on_error):
                break
            continue

        o = res.outcome
        sym = o.symbol
        run_id = run_dir.name
        artifact_note = "dry-run: no files written" if args.dry_run else "outcome.json written with nulls; rerun later"
        partial_note = "dry-run: no files written" if args.dry_run else "outcome.json written; rerun later if needed"
        if res.auto_fetch_used and res.yfinance_empty:
            partial += 1
            _print_outcome_batch_warn_line(
                symbol=sym,
                message=f"auto-fetch returned no data (yfinance empty)  → {artifact_note}",
            )
        elif res.auto_fetch_partial:
            partial += 1
            _print_outcome_batch_warn_line(
                symbol=sym,
                message=f"auto-fetch returned partial data  → {partial_note}",
            )
        else:
            recorded += 1
            _print_outcome_batch_ok_line(
                symbol=sym,
                run_id=run_id,
                earnings_close=o.earnings_day_close,
                auto_fetch=res.auto_fetch_used,
                direction=o.direction_vs_prior_close,
            )

    summary_lines = [
        "Batch outcome record summary",
        f"  Attempted:  {attempted}",
        f"  Recorded:   {recorded}",
        f"  Partial:    {partial}   (auto-fetch returned partial or empty data)",
        f"  Skipped:    {skipped}   (no matching run dir)",
    ]
    if failed:
        summary_lines.append(f"  Failed:     {failed}   (processing errors)")
    summary_lines.append("")
    sys.stdout.write("\n".join(summary_lines) + "\n")
    return 1 if any_fail_line else 0


def _run_predictions_extract_batch_cli(args: argparse.Namespace) -> int:
    configure_cli_logging(getattr(logging, str(args.log_level)))
    sym_list = _load_batch_symbols_csv_and_file(args.symbols, args.symbols_file)
    if args.batch_dir and sym_list:
        raise SystemExit("--batch-dir cannot be combined with --symbols / --symbols-file")
    if not args.batch_dir and not sym_list:
        raise SystemExit(
            "Provide --batch-dir (Shape A) or --symbols / --symbols-file (Shape B).",
        )

    work: list[tuple[str, Path | None]] = []
    if args.batch_dir:
        batch_dir = Path(str(args.batch_dir)).expanduser()
        summary = batch_dir / "batch_summary.txt"
        try:
            dirs = parse_output_dirs_from_batch_summary(summary)
        except OSError as exc:
            raise SystemExit(f"Could not read batch summary: {exc}") from exc
        for d in dirs:
            run_dir = d.expanduser()
            rid = run_dir.name
            work.append((_symbol_prefix_from_run_id(rid), run_dir))
    else:
        since = _parse_since_date(str(args.since)) if args.since else _default_since_utc_midnight()
        outputs_dir = Path(str(args.outputs_dir)).expanduser()
        work = plan_shape_b_run_directories(
            outputs_dir,
            sym_list,
            since,
            newest_only=bool(args.newest_only),
        )

    attempted = len(work)
    ok_runs = 0
    skipped = 0
    failed = 0
    any_fail_line = False

    for idx, (sym_hint, run_dir_raw) in enumerate(work):
        if idx > 0 and float(args.rate_limit_sleep_s) > 0:
            time.sleep(float(args.rate_limit_sleep_s))

        if run_dir_raw is None:
            skipped += 1
            any_fail_line = True
            sys.stdout.write(f"[SKIP]  {sym_hint:6} no run dir found              → skipped\n")
            continue

        run_dir = run_dir_raw.expanduser()
        sym = sym_hint
        run_json = run_dir / "run.json"
        if args.dry_run:
            sys.stdout.write(f"[DRY]   {sym:6} would_extract run_dir={run_dir}\n")
            ok_runs += 1
            continue

        try:
            if not run_dir.is_dir():
                raise FileNotFoundError(f"missing run directory: {run_dir}")
            if not run_json.is_file():
                raise FileNotFoundError(f"missing run.json: {run_json}")
            data = json.loads(run_json.read_text(encoding="utf-8"))
            cfg_raw = data.get("config")
            if not isinstance(cfg_raw, dict):
                raise ValueError("run.json missing config snapshot")
            effective_rp = run_profile_from_persisted_run_json(data)
            cfg = RunConfig.model_validate(cfg_raw).model_copy(update={"run_profile": effective_rp})
            rows = asyncio.run(run_prediction_extract_for_run_dir(run_dir=run_dir, cfg=cfg))
        except Exception as exc:
            failed += 1
            any_fail_line = True
            sys.stdout.write(
                f"[FAIL]  {sym:6} {type(exc).__name__}: {exc}              → skipped\n",
            )
            if not bool(args.continue_on_error):
                break
            continue

        ok_runs += 1
        sys.stdout.write(
            f"[OK]    {sym:6} predictions_extracted run={run_dir.name}  rows={len(rows)}\n",
        )

    summary_lines = [
        "Batch predictions extract summary",
        f"  Attempted:  {attempted}",
        f"  Extracted:  {ok_runs}",
        f"  Skipped:    {skipped}   (no matching run dir)",
    ]
    if failed:
        summary_lines.append(f"  Failed:     {failed}   (processing errors)")
    summary_lines.append("")
    sys.stdout.write("\n".join(summary_lines) + "\n")
    return 1 if any_fail_line else 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv(override=False)
    args = _build_parser().parse_args(argv)

    if args.command == "run":
        configure_cli_logging(getattr(logging, str(args.log_level)))
        if args.resume and not args.iterative:
            raise SystemExit("--resume requires --iterative")
        cfg = _apply_cli_config_overrides(_load_cfg(args), args)
        if args.symbol:
            cfg.symbol = args.symbol

        prompt_path = Path(args.prompt_file) if args.prompt_file else None
        if args.iterative:
            text, out_dir, final_state = asyncio.run(_run_iterative_cli(args, cfg, prompt_path))
            if not args.dry_run:
                with contextlib.suppress(Exception):
                    run_json = out_dir / "run.json"
                    if run_json.is_file():
                        data = json.loads(run_json.read_text(encoding="utf-8"))
                        data["finished_at_utc"] = datetime.now(tz=UTC).replace(microsecond=0).isoformat()
                        data["run_profile"] = cfg.run_profile
                        run_json.write_text(
                            json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                        )
                        run_json_data = json.loads(run_json.read_text(encoding="utf-8"))
                    else:
                        run_json_data = {}

                    # Build per-provider rows from in-memory final_state (avoids parsing files).
                    provider_map = {pc.name: pc for pc in cfg.providers}
                    pr_rows: list[tuple[int | None, str, Any, str, bool]] = []
                    resp_rounds = (
                        final_state.get("provider_responses")
                        if isinstance(final_state, dict)
                        else None
                    )
                    if isinstance(resp_rounds, list):
                        for i, round_blob in enumerate(resp_rounds, start=1):
                            if not isinstance(round_blob, dict):
                                continue
                            raw = round_blob.get("responses")
                            if not isinstance(raw, dict):
                                continue
                            try:
                                response_path = str(
                                    (out_dir / "iterations" / f"iteration_{i}_providers.md").relative_to(
                                        out_dir.parent
                                    )
                                )
                            except Exception:
                                response_path = str(out_dir / "iterations" / f"iteration_{i}_providers.md")
                            for prov_name, d in raw.items():
                                if not isinstance(d, dict):
                                    continue
                                resp = _dict_to_provider_response(d)
                                pc = provider_map.get(prov_name)
                                ws_enabled = (
                                    bool(pc.web_search) if pc and pc.web_search is not None else bool(args.enable_web_search)
                                )
                                pr_rows.append((i, prov_name, resp, response_path, ws_enabled))

                    started_at = run_json_data.get("started_at_utc")
                    finished_at = run_json_data.get("finished_at_utc")
                    asyncio.run(
                        best_effort_upsert_run_and_responses(
                            cfg=cfg,
                            run_dir=out_dir,
                            run_json_data=run_json_data,
                            started_at_utc=_parse_dt_iso(started_at),
                            finished_at_utc=_parse_dt_iso(finished_at),
                            provider_responses=pr_rows,
                            synthesis_path=out_dir / "synthesis.md",
                            database_url=cfg.database_url,
                        )
                    )
        else:
            if not args.config:
                raise SystemExit("--config is required for non-iterative runs")
            orch = Orchestrator(config=cfg, prompt_path=prompt_path)
            text = orch.run_sync(dry_run=args.dry_run, enable_web_search=args.enable_web_search)
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")
        return 0

    if args.command == "outcome-record":
        configure_cli_logging(logging.INFO)
        run_dir = Path(str(args.run_dir))

        def _prompt_float(label: str, cur: float | None) -> float | None:
            if cur is not None:
                return cur
            raw = input(f"{label} (blank to skip): ").strip()
            if not raw:
                return None
            return float(raw)

        def _prompt_str(label: str, cur: str | None) -> str | None:
            if cur is not None:
                return cur
            raw = input(f"{label} (blank to skip): ").strip()
            return raw or None

        def _prompt_choice(label: str, cur: str | None, choices: list[str]) -> str | None:
            if cur is not None:
                return cur
            raw = input(f"{label} {choices} (blank to skip): ").strip().lower()
            if not raw:
                return None
            if raw not in choices:
                raise SystemExit(f"Invalid value for {label}: {raw!r} (choices: {choices})")
            return raw

        try:
            earnings_day_open = args.earnings_day_open
            earnings_day_high = args.earnings_day_high
            earnings_day_low = args.earnings_day_low
            earnings_day_close = args.earnings_day_close
            next_trading_day_open = args.next_trading_day_open
            next_trading_day_close = args.next_trading_day_close
            one_week_later_close = args.one_week_later_close
            direction_vs_prior_close = args.direction_vs_prior_close
            notes = args.notes

            auto_fetch_cli = bool(getattr(args, "auto_fetch", False))
            if args.interactive:
                if auto_fetch_cli:
                    merged, _fetched = merge_auto_fetch_into_cli_fields(
                        run_dir,
                        earnings_day_open=earnings_day_open,
                        earnings_day_high=earnings_day_high,
                        earnings_day_low=earnings_day_low,
                        earnings_day_close=earnings_day_close,
                        next_trading_day_open=next_trading_day_open,
                        next_trading_day_close=next_trading_day_close,
                        one_week_later_close=one_week_later_close,
                        direction_vs_prior_close=direction_vs_prior_close,
                    )
                    earnings_day_open = merged["earnings_day_open"]
                    earnings_day_high = merged["earnings_day_high"]
                    earnings_day_low = merged["earnings_day_low"]
                    earnings_day_close = merged["earnings_day_close"]
                    next_trading_day_open = merged["next_trading_day_open"]
                    next_trading_day_close = merged["next_trading_day_close"]
                    one_week_later_close = merged["one_week_later_close"]
                    direction_vs_prior_close = merged["direction_vs_prior_close"]
                earnings_day_open = _prompt_float("earnings_day_open", earnings_day_open)
                earnings_day_high = _prompt_float("earnings_day_high", earnings_day_high)
                earnings_day_low = _prompt_float("earnings_day_low", earnings_day_low)
                earnings_day_close = _prompt_float("earnings_day_close", earnings_day_close)
                next_trading_day_open = _prompt_float("next_trading_day_open", next_trading_day_open)
                next_trading_day_close = _prompt_float("next_trading_day_close", next_trading_day_close)
                one_week_later_close = _prompt_float("one_week_later_close", one_week_later_close)
                direction_vs_prior_close = _prompt_choice(
                    "direction_vs_prior_close", direction_vs_prior_close, ["up", "down", "flat"]
                )
                notes = _prompt_str("notes", notes)

            result = record_outcome_for_run_dir(
                run_dir=run_dir,
                auto_fetch=False if args.interactive else auto_fetch_cli,
                dry_run=False,
                earnings_day_open=earnings_day_open,
                earnings_day_high=earnings_day_high,
                earnings_day_low=earnings_day_low,
                earnings_day_close=earnings_day_close,
                next_trading_day_open=next_trading_day_open,
                next_trading_day_close=next_trading_day_close,
                one_week_later_close=one_week_later_close,
                direction_vs_prior_close=direction_vs_prior_close,
                notes=notes,
                source=cast(Literal["manual", "yahoo_csv", "alpaca", "polygon"], args.source),
                db_upsert=True,
            )
            outcome = result.outcome
        except KeyboardInterrupt as exc:
            raise SystemExit(130) from exc

        sys.stdout.write(json.dumps(outcome.model_dump(), indent=2, sort_keys=True) + "\n")
        return 0

    if args.command == "outcome-record-batch":
        return _run_outcome_record_batch_cli(args)

    if args.command == "predictions-extract":
        configure_cli_logging(getattr(logging, str(args.log_level)))
        run_dir = Path(str(args.run_dir)).expanduser().resolve()
        run_json = run_dir / "run.json"
        if not run_json.is_file():
            raise SystemExit(f"missing run.json at {run_json}")
        data = json.loads(run_json.read_text(encoding="utf-8"))
        cfg_raw = data.get("config")
        if not isinstance(cfg_raw, dict):
            raise SystemExit("run.json missing config snapshot")
        cfg = RunConfig.model_validate(cfg_raw)
        rows = asyncio.run(run_prediction_extract_for_run_dir(run_dir=run_dir, cfg=cfg))
        sys.stdout.write(json.dumps({"rows": [asdict(r) for r in rows]}, indent=2, sort_keys=True) + "\n")
        return 0

    if args.command == "predictions-extract-batch":
        return _run_predictions_extract_batch_cli(args)

    if args.command == "db-backfill":
        return _run_db_backfill_cli(args)

    raise AssertionError("unreachable")


def _parse_since_date(raw: str) -> datetime:
    try:
        return datetime.strptime(raw, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError as exc:
        raise SystemExit(f"--since must be YYYY-MM-DD (got {raw!r}): {exc}") from exc


def _run_db_backfill_cli(args: argparse.Namespace) -> int:
    from equity_analyst.db_backfill import iter_run_directories

    configure_cli_logging(getattr(logging, str(args.log_level)))
    outputs_dir = Path(str(args.outputs_dir))
    since: datetime | None = _parse_since_date(str(args.since)) if args.since else None

    dirs = iter_run_directories(
        outputs_dir,
        symbol=args.symbol,
        since=since,
        oldest_first=not args.newest_first,
    )
    if args.limit is not None and args.limit >= 0:
        dirs = dirs[: args.limit]

    if args.dry_run:
        logger.info(
            "db-backfill --dry-run: %d run dir(s) would be backfilled (outputs_dir=%s)",
            len(dirs),
            outputs_dir,
        )
        for d in dirs:
            sys.stdout.write(f"DRY-RUN run_id={d.name}\n")
        _print_backfill_summary(scanned=len(dirs), inserted=0, skipped=len(dirs), errors=0, dry_run=True)
        return 0

    return asyncio.run(_run_db_backfill_async(dirs, run_dirs_count=len(dirs)))


async def _run_db_backfill_async(dirs: list[Path], *, run_dirs_count: int) -> int:
    from equity_analyst.db import get_async_session, is_db_available
    from equity_analyst.db_backfill import backfill_run_directory

    if not await is_db_available():
        raise SystemExit(
            "db-backfill: DATABASE_URL is unreachable (DB unavailable). "
            "Set DATABASE_URL in .env or your shell and verify Postgres is running."
        )

    scanned = 0
    inserted = 0
    skipped = 0
    errors = 0

    async with get_async_session() as session:
        for d in dirs:
            scanned += 1
            try:
                result = await backfill_run_directory(session, d)
                await session.commit()
            except Exception as exc:
                logger.error("Backfill failed run_id=%s error=%r", d.name, exc)
                await session.rollback()
                errors += 1
                continue
            if result.skipped_reasons:
                skipped += 1
                logger.info(
                    "Backfill skipped run_id=%s reasons=%s",
                    result.run_id,
                    result.skipped_reasons,
                )
            elif result.inserted:
                inserted += 1
            else:
                skipped += 1

    _print_backfill_summary(
        scanned=scanned, inserted=inserted, skipped=skipped, errors=errors, dry_run=False
    )
    return 0 if errors == 0 else 1


def _print_backfill_summary(
    *, scanned: int, inserted: int, skipped: int, errors: int, dry_run: bool
) -> None:
    header = "Backfill summary (dry-run)" if dry_run else "Backfill summary"
    skipped_note = "  (dry-run: nothing written)" if dry_run else "  (already up to date)"
    sys.stdout.write(
        "\n".join(
            [
                header,
                f"  Scanned:   {scanned}",
                f"  Inserted:  {inserted}",
                f"  Skipped:   {skipped}{skipped_note}",
                f"  Errors:    {errors}",
                "",
            ]
        )
    )
