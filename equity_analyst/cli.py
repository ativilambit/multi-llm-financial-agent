from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from equity_analyst.config import RunConfig, load_config
from equity_analyst.iterative import (
    build_initial_refinement_state,
    compile_refinement_workflow,
    dry_run_compile_only,
)
from equity_analyst.logging_setup import attach_run_file_logging, configure_cli_logging
from equity_analyst.orchestrator import Orchestrator
from equity_analyst.prompting import render_prompt
from equity_analyst.providers.registry import ProviderRegistry

logger = logging.getLogger(__name__)


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
    run.add_argument("--max-iterations", type=int, default=3)
    run.add_argument("--confidence-threshold", type=float, default=0.85)
    run.add_argument(
        "--resume",
        default=None,
        help="Output folder name under outputs/ (checkpoint at outputs/<id>/checkpoint.sqlite)",
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
        help="Override RunConfig.verifier_max_output_tokens (iterative verifier only; default 1536)",
    )
    run.add_argument(
        "--synthesizer-max-input-tokens",
        type=int,
        default=None,
        help="Override RunConfig.synthesizer_max_input_tokens (default from YAML or 20000)",
    )
    run.add_argument(
        "--synthesizer-max-output-tokens",
        type=int,
        default=None,
        help="Override RunConfig.synthesizer_max_output_tokens (default from YAML or 24000)",
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

    return parser


def _apply_cli_config_overrides(cfg: RunConfig, args: argparse.Namespace) -> RunConfig:
    patch: dict[str, Any] = {}
    if args.retry_max_attempts is not None:
        patch["retry_max_attempts"] = args.retry_max_attempts
    if args.retry_base_delay_s is not None:
        patch["retry_base_delay_s"] = args.retry_base_delay_s
    if args.max_output_tokens is not None:
        patch["max_output_tokens"] = args.max_output_tokens
    if args.verifier_max_output_tokens is not None:
        patch["verifier_max_output_tokens"] = args.verifier_max_output_tokens
    if args.synthesizer_max_input_tokens is not None:
        patch["synthesizer_max_input_tokens"] = args.synthesizer_max_input_tokens
    if args.synthesizer_max_output_tokens is not None:
        patch["synthesizer_max_output_tokens"] = args.synthesizer_max_output_tokens
    if getattr(args, "no_prompt_cache", False):
        patch["prompt_cache_enabled"] = False
    if getattr(args, "no_force_tool_use", False):
        patch["anthropic_force_tool_use"] = False
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
) -> str:
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
        )
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
    meta = {
        "iterative": True,
        "thread_id": thread_id,
        "template_path": rendered.template_path,
        "config": cfg.model_dump(),
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
        if resume:
            final_state = await app.ainvoke(None, config=config)
        else:
            st = build_initial_refinement_state(cfg=cfg, rendered=rendered, output_dir=out_dir)
            st["max_iterations"] = args.max_iterations
            st["confidence_threshold"] = args.confidence_threshold
            st["enable_web_search"] = args.enable_web_search
            final_state = await app.ainvoke(st, config=config)
    logger.info("Iterative run finished output_dir=%s", str(out_dir.resolve()))
    return str(final_state.get("final_report", ""))


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
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
            text = asyncio.run(_run_iterative_cli(args, cfg, prompt_path))
        else:
            if not args.config:
                raise SystemExit("--config is required for non-iterative runs")
            orch = Orchestrator(config=cfg, prompt_path=prompt_path)
            text = orch.run_sync(dry_run=args.dry_run, enable_web_search=args.enable_web_search)
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")
        return 0

    raise AssertionError("unreachable")
