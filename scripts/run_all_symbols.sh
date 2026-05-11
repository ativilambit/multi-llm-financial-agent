#!/usr/bin/env bash
# Batch runner for the 10 symbol configs created on 2026-05-10.
#
# Sequential by default (one symbol at a time) so provider rate limits and
# long web-search runs do not stack; each symbol’s Python output is tee’d to
# the terminal and its log file. Use --parallel to launch every symbol as
# a background job; that warning is intentional — every provider key is shared
# across symbols, and Anthropic/Gemini/OpenAI/Grok will rate-limit aggressive
# fan-out. See the README "Running multiple symbols" subsection.
#
# Compatibility: macOS /bin/bash (Bash 3.2). No mapfile, no ${var,,}, no
# associative arrays. Avoid Bash 4+ features.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  echo "ERROR: $PYTHON_BIN not found or not executable. Create .venv per README." >&2
  exit 2
fi

# Fixed symbol order — keep aligned with the README table.
SYMBOLS="ASTS FIGR HIMS RGTI GTM PLUG STE ACHR IX QUBT"
CONFIG_DATE="2026_05_10"

# Defaults.
MODE="sequential"
ITERATIVE=1
MAX_ITERATIONS=3
LOG_LEVEL="INFO"

usage() {
  cat <<'USAGE'
Usage: scripts/run_all_symbols.sh [--parallel] [--no-iterative] [--max-iterations N] [--log-level LEVEL]

Options:
  --parallel              Launch every symbol as a background job and wait.
                          Warning: shares one set of API keys across all symbols;
                          providers may rate-limit. Sequential mode is safer.
  --no-iterative          Skip --iterative; run a single fan-out + synthesis pass.
  --max-iterations N      Forward to --max-iterations N (default 3, iterative only).
  --log-level LEVEL       Forward to --log-level (DEBUG|INFO|WARNING|ERROR; default INFO).
  -h, --help              Show this help and exit.

Symbols are run in this fixed order:
  ASTS FIGR HIMS RGTI GTM PLUG STE ACHR IX QUBT

A per-batch summary is written to outputs/batch_<timestamp>/batch_summary.txt
along with one combined log per symbol.
USAGE
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --parallel)
      MODE="parallel"
      shift
      ;;
    --no-iterative)
      ITERATIVE=0
      shift
      ;;
    --max-iterations)
      if [ "$#" -lt 2 ]; then
        echo "ERROR: --max-iterations requires a value" >&2
        exit 2
      fi
      MAX_ITERATIONS="$2"
      shift 2
      ;;
    --max-iterations=*)
      MAX_ITERATIONS="${1#--max-iterations=}"
      shift
      ;;
    --log-level)
      if [ "$#" -lt 2 ]; then
        echo "ERROR: --log-level requires a value" >&2
        exit 2
      fi
      LOG_LEVEL="$2"
      shift 2
      ;;
    --log-level=*)
      LOG_LEVEL="${1#--log-level=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

BATCH_TS="$(date -u +%Y%m%dT%H%M%SZ)"
BATCH_DIR="outputs/batch_${BATCH_TS}"
mkdir -p "$BATCH_DIR"
SUMMARY_FILE="$BATCH_DIR/batch_summary.txt"

{
  echo "Batch run started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "Mode: $MODE"
  echo "Iterative: $ITERATIVE  Max iterations: $MAX_ITERATIONS  Log level: $LOG_LEVEL"
  echo "Symbols (in order): $SYMBOLS"
  echo "Repo root: $REPO_ROOT"
  echo "----"
} >"$SUMMARY_FILE"

# Build the equity_analyst argv suffix (everything after the config path).
build_args() {
  # echoes a single shell-quote-safe argument string (we control all values).
  local _args="--log-level $LOG_LEVEL"
  if [ "$ITERATIVE" -eq 1 ]; then
    _args="$_args --iterative --max-iterations $MAX_ITERATIONS"
  fi
  echo "$_args"
}

EXTRA_ARGS="$(build_args)"

lowercase() {
  # Bash 3.2-safe lowercase conversion (no ${var,,}).
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

run_one() {
  # $1 = symbol (e.g. ASTS); writes per-symbol log under $BATCH_DIR and appends
  # an [OK] / [FAIL] line to $SUMMARY_FILE. Returns the underlying exit code.
  local symbol="$1"
  local lower
  lower="$(lowercase "$symbol")"
  local config="configs/${lower}_${CONFIG_DATE}.yaml"
  local log_file="$BATCH_DIR/${lower}.log"

  if [ ! -f "$config" ]; then
    {
      echo "[FAIL] $symbol  config_missing=$config"
    } >>"$SUMMARY_FILE"
    echo "[FAIL] $symbol (missing $config)" >&2
    return 1
  fi

  local started_epoch
  started_epoch="$(date +%s)"
  echo "[START] $symbol  config=$config  $(date -u +%Y-%m-%dT%H:%M:%SZ)" >&2

  local rc
  set +e
  # word-split EXTRA_ARGS intentionally — they are space-separated flags we control.
  # shellcheck disable=SC2086
  if [ "$MODE" = "sequential" ]; then
    echo "[STREAM] $symbol  log=$log_file" >&2
    "$PYTHON_BIN" -m equity_analyst run --config "$config" $EXTRA_ARGS 2>&1 | tee "$log_file"
    # Do not use `local rc=$PIPESTATUS[0]` — `local` resets PIPESTATUS before the assignment reads it.
    rc="${PIPESTATUS[0]}"
  else
    "$PYTHON_BIN" -m equity_analyst run --config "$config" $EXTRA_ARGS >"$log_file" 2>&1
    rc=$?
  fi
  set -e

  local ended_epoch
  ended_epoch="$(date +%s)"
  local duration=$((ended_epoch - started_epoch))

  # Try to recover the per-run output_dir from the captured log. Both
  # orchestrator.py and iterative finalize emit "output_dir=<absolute path>".
  local output_dir
  output_dir="$(grep -E 'output_dir=' "$log_file" 2>/dev/null | tail -n1 | sed -E 's/.*output_dir=([^ ]+).*/\1/')"
  if [ -z "$output_dir" ]; then
    output_dir="see outputs/"
  fi

  if [ "$rc" -eq 0 ]; then
    echo "[OK]   $symbol  duration=${duration}s  output_dir=${output_dir}" >>"$SUMMARY_FILE"
    echo "[OK]   $symbol (${duration}s)" >&2
  else
    echo "[FAIL] $symbol  duration=${duration}s  exit=${rc}  output_dir=${output_dir}  log=${log_file}" >>"$SUMMARY_FILE"
    echo "[FAIL] $symbol (exit ${rc}, ${duration}s, log: ${log_file})" >&2
  fi

  return "$rc"
}

OVERALL_RC=0

if [ "$MODE" = "sequential" ]; then
  for sym in $SYMBOLS; do
    if ! run_one "$sym"; then
      OVERALL_RC=1
    fi
  done
else
  echo "WARNING: --parallel mode shares one API key per provider across $(echo "$SYMBOLS" | wc -w | tr -d ' ') symbols." >&2
  echo "         Expect rate-limit pushback on Anthropic/OpenAI/Gemini/Grok and longer per-symbol wall time." >&2
  echo "[INFO] --parallel: per-symbol output goes to log files only; tail ${BATCH_DIR}/<SYMBOL>.log to follow a specific run" >&2

  PIDS=""
  for sym in $SYMBOLS; do
    run_one "$sym" &
    PIDS="$PIDS $!:$sym"
  done

  for entry in $PIDS; do
    pid="${entry%%:*}"
    sym="${entry##*:}"
    set +e
    wait "$pid"
    rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then
      OVERALL_RC=1
      echo "[FAIL-WAIT] $sym  pid=$pid  exit=$rc" >>"$SUMMARY_FILE"
    fi
  done
fi

{
  echo "----"
  echo "Batch run finished: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "Overall exit: $OVERALL_RC"
} >>"$SUMMARY_FILE"

echo "" >&2
echo "Batch summary: $SUMMARY_FILE" >&2

exit "$OVERALL_RC"
