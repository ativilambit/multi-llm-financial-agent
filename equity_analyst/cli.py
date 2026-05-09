from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from equity_analyst.config import load_config
from equity_analyst.orchestrator import Orchestrator


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="equity_analyst")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run multi-provider analysis and synthesis")
    run.add_argument("--symbol", required=False, help="Override symbol from config")
    run.add_argument("--config", required=True, help="YAML config path or '-' for stdin")
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
        "--enable-web-search",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable providers' web search tools where supported",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = _build_parser().parse_args(argv)

    if args.command == "run":
        cfg = load_config(args.config)
        if args.symbol:
            cfg.symbol = args.symbol

        prompt_path = Path(args.prompt_file) if args.prompt_file else None
        orch = Orchestrator(config=cfg, prompt_path=prompt_path)
        synthesis = orch.run_sync(dry_run=args.dry_run, enable_web_search=args.enable_web_search)
        sys.stdout.write(synthesis)
        if not synthesis.endswith("\n"):
            sys.stdout.write("\n")
        return 0

    raise AssertionError("unreachable")

