#!/usr/bin/env bash
# Scheduled post-earnings outcome refresh: discovers tickers from outputs/*_<TS>/
# directories (with run.json) and runs:
#   python -m equity_analyst outcome-record-batch --auto-fetch ...
#
# Postgres: the CLI loads repo-root .env via python-dotenv (equity_analyst.cli main:
# load_dotenv(override=False)) before connecting. Realized OHLC / direction are upserted
# into the ``outcomes`` table (run_id PK, FK to ``runs``) by db_ops.best_effort_upsert_outcome;
# connection string is ``DATABASE_URL`` (see equity_analyst/db.py, .env.example).
#
# launchd StartCalendarInterval uses the Mac system clock timezone; plist + this script
# assume 1:30pm US/Pacific weekdays (see scripts/launchd/*.plist).
#
# Environment (optional):
#   POST_EVENT_SINCE_DAYS  Calendar days back for --since (default: 30). macOS date(1).
#   POST_EVENT_NEWEST_ONLY When set to 0, passes --no-newest-only (all runs in window).
#   POST_EVENT_RATE_LIMIT  Seconds between symbols after the first (default: 0.5).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export TZ="${TZ:-America/Los_Angeles}"

if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi

mkdir -p "$REPO_ROOT/logs"

since_days="${POST_EVENT_SINCE_DAYS:-30}"
# BSD date on macOS
since_date="$(date -j -v-"${since_days}"d +%Y-%m-%d)"

newest_only_args=(--newest-only)
if [[ "${POST_EVENT_NEWEST_ONLY:-1}" == "0" ]]; then
  newest_only_args=(--no-newest-only)
fi

rate_limit="${POST_EVENT_RATE_LIMIT:-0.5}"

outputs_dir="$REPO_ROOT/outputs"
if [[ ! -d "$outputs_dir" ]]; then
  echo "post_event_update: no outputs/ directory; exiting."
  exit 0
fi

symbols_tmp="$(mktemp -t post_event_symbols.XXXXXX)"
cleanup() { rm -f "$symbols_tmp"; }
trap cleanup EXIT

# Run id pattern: <SYMBOL>_<YYYYMMDDTHHMMSSZ> (see db_backfill._parse_run_dir_timestamp).
{
  for dir in "$outputs_dir"/*; do
    [[ -d "$dir" && -f "$dir/run.json" ]] || continue
    base="$(basename "$dir")"
    if [[ "$base" =~ ^(.+)_[0-9]{8}T[0-9]{6}Z$ ]]; then
      printf '%s\n' "${BASH_REMATCH[1]}"
    fi
  done
} | sort -u >"$symbols_tmp"

if ! [[ -s "$symbols_tmp" ]]; then
  echo "post_event_update: no run directories matched; nothing to do."
  exit 0
fi

echo "post_event_update: since=$since_date symbols_file=$symbols_tmp (lines=$(wc -l <"$symbols_tmp" | tr -d ' '))"

"$PYTHON_BIN" -m equity_analyst outcome-record-batch \
  --symbols-file "$symbols_tmp" \
  --since "$since_date" \
  --auto-fetch \
  "${newest_only_args[@]}" \
  --rate-limit-sleep-s "$rate_limit" \
  --outputs-dir "$outputs_dir"
