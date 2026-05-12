#!/usr/bin/env bash
# Batch runner: iterates `python -m equity_analyst run` over a symbol list and a
# shared config date suffix (e.g. configs/asts_2026_05_10.yaml). With an explicit
# --date (or leading positional DATE) and no --symbols/--symbols-file, tickers are
# auto-discovered from every configs/*_<date>.yaml (sorted).
#
# Sequential by default (one symbol at a time) so provider rate limits and
# long web-search runs do not stack; each symbol’s Python output is tee’d to
# the terminal and its log file. Use --parallel to run symbols in the background
# with bounded concurrency (default 2 at a time; override with --jobs N). Every
# provider key is shared across symbols, and Anthropic/Gemini/OpenAI/Grok will
# rate-limit aggressive fan-out. See the README "Running multiple symbols" subsection.
#
# Compatibility: macOS /bin/bash (Bash 3.2). No mapfile, no ${var,,}, no
# associative arrays. Avoid Bash 4+ features.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Collapse runs of spaces/tabs to one space and trim ends (portable; avoids xargs).
squeeze_trim() {
  printf '%s' "$1" | tr -s '[:blank:]' ' ' | sed 's/^ //;s/ $//'
}

PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  echo "ERROR: $PYTHON_BIN not found or not executable. Create .venv per README." >&2
  exit 2
fi

# Defaults (2026-05-10 batch).
SYMBOLS_DEFAULT="ASTS FIGR HIMS RGTI GTM PLUG STE ACHR IX QUBT"
CONFIG_DATE_DEFAULT="2026_05_10"

# Defaults.
MODE="sequential"
JOBS=""
ITERATIVE=1
MAX_ITERATIONS=3
LOG_LEVEL="INFO"

HAVE_DATE=0
RAW_DATE=""
HAVE_SYMBOLS=0
SYMBOLS_ARG=""
HAVE_SYMBOLS_FILE=0
SYMBOLS_FILE_PATH=""

normalize_config_date() {
  local d="$1"
  d="$(printf '%s' "$d" | tr '-' '_')"
  case "$d" in
    [0-9][0-9][0-9][0-9]_[0-9][0-9]_[0-9][0-9]) printf '%s' "$d"; return 0 ;;
    *)
      echo "ERROR: --date must be YYYY-MM-DD or YYYY_MM_DD (got: $1)" >&2
      return 1
      ;;
  esac
}

# True if the token normalizes to configs/*_<token>.yaml style suffix.
is_config_date_token() {
  local d="$1"
  d="$(printf '%s' "$d" | tr '-' '_')"
  case "$d" in
    [0-9][0-9][0-9][0-9]_[0-9][0-9]_[0-9][0-9]) return 0 ;;
    *) return 1 ;;
  esac
}

# Uppercase tickers from configs/*_<date_suffix>.yaml, sorted uniquely (deterministic order).
discover_symbols_for_date() {
  local d="$1"
  local f base sym acc
  acc=""
  for f in "$REPO_ROOT/configs/"*"_${d}.yaml"; do
    [ -f "$f" ] || continue
    base="${f##*/}"
    sym="${base%_${d}.yaml}"
    [ -z "$sym" ] && continue
    sym="$(printf '%s' "$sym" | tr '[:lower:]' '[:upper:]')"
    acc="$acc $sym"
  done
  acc="$(squeeze_trim "$acc")"
  if [ -z "$acc" ]; then
    echo "ERROR: no configs matching configs/*_${d}.yaml" >&2
    return 1
  fi
  local sorted line
  sorted=""
  for line in $(printf '%s\n' $acc | sort -u); do
    [ -z "$line" ] && continue
    sorted="${sorted}${sorted:+ }$line"
  done
  printf '%s' "$sorted"
}

# Fail fast before any run when the user supplied an explicit symbol list.
validate_explicit_configs() {
  local d="$1"
  local syms="$2"
  local sym lower path had_missing
  had_missing=0
  for sym in $syms; do
    lower="$(printf '%s' "$sym" | tr '[:upper:]' '[:lower:]')"
    path="configs/${lower}_${d}.yaml"
    if [ ! -f "$path" ]; then
      if [ "$had_missing" -eq 0 ]; then
        echo "ERROR: missing config file(s) for explicit symbol list:" >&2
        had_missing=1
      fi
      echo "  $path" >&2
    fi
  done
  if [ "$had_missing" -eq 1 ]; then
    exit 2
  fi
}

load_symbols_from_file() {
  local path="$1"
  local line acc
  acc=""
  if [ ! -f "$path" ]; then
    echo "ERROR: --symbols-file not a readable file: $path" >&2
    return 1
  fi
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line%%#*}"
    line="$(squeeze_trim "$(printf '%s' "$line" | tr ',' ' ')")"
    [ -z "$line" ] && continue
    acc="$acc $line"
  done <"$path"
  squeeze_trim "$acc"
}

validate_jobs() {
  local j="$1"
  case "$j" in
    ''|*[!0-9]*)
      echo "ERROR: --jobs must be a positive integer between 1 and 10 (got: $j)" >&2
      exit 2
      ;;
  esac
  if [ "$j" -lt 1 ] || [ "$j" -gt 10 ]; then
    echo "ERROR: --jobs must be between 1 and 10 (got: $j)" >&2
    exit 2
  fi
}

finalize_parallel_jobs() {
  # Apply --jobs / --parallel defaults after argv parsing.
  if [ -n "$JOBS" ]; then
    validate_jobs "$JOBS"
    if [ "$JOBS" -gt 1 ]; then
      MODE="parallel"
    fi
  fi
  if [ "$MODE" = "parallel" ]; then
    if [ -z "$JOBS" ]; then
      JOBS=2
    elif [ "$JOBS" -lt 1 ]; then
      echo "ERROR: internal: invalid JOBS" >&2
      exit 2
    fi
  fi
}

usage() {
  cat <<'USAGE'
Usage: scripts/run_all_symbols.sh [positional] [options]

Positional shorthand (optional, must come first):
  DATE [SYM ...]        Same as --date DATE; if no tickers follow, symbols are
                        auto-discovered from every configs/*_DATE.yaml (sorted).
                        Ticker arguments may continue until the first --flag.

Batch options:
  --date YYYY-MM-DD       Config filename suffix (underscores in paths). Default:
                          2026-05-10 → configs/<sym>_2026_05_10.yaml
                        If you pass --date (or use the positional DATE form) and
                        do not pass --symbols / --symbols-file, tickers are read
                        from all matching configs/*_<suffix>.yaml files.
  --symbols A,B,C       Comma-separated tickers (overrides default list and
                          --symbols-file if both are passed).
  --symbols-file PATH   One or more tickers per line (# comments allowed).
                          Ignored if --symbols is also passed.
                        With --symbols or --symbols-file, every
                        configs/<sym_lower>_<date>.yaml must exist before the
                        batch starts (fail fast with a list of missing paths).

Run options:
  --parallel              Run symbols as background jobs with bounded concurrency
                          (default: 2 at a time). Per-symbol logs only; see README.
                          Warning: shares one set of API keys across symbols;
                          providers may rate-limit. Sequential mode is safer.
  --jobs N, -j N          With --parallel: max concurrent symbols (1–10; default 2).
                          Without --parallel: if N>1, enables parallel mode with
                          this cap; if N is 1, ignored (stay sequential).
  --no-iterative          Skip --iterative; run a single fan-out + synthesis pass.
  --max-iterations N      Forward to --max-iterations N (default 3, iterative only).
  --log-level LEVEL       Forward to --log-level (DEBUG|INFO|WARNING|ERROR; default INFO).
  -h, --help              Show this help and exit.

Default symbol order (with default --date 2026-05-10):
  ASTS FIGR HIMS RGTI GTM PLUG STE ACHR IX QUBT

Examples:
  scripts/run_all_symbols.sh 2026_05_12
  scripts/run_all_symbols.sh 2026_05_13 NBIS BABA
  scripts/run_all_symbols.sh --date 2026-05-12 --parallel

A per-batch summary is written to outputs/batch_<timestamp>/batch_summary.txt
along with one combined log per symbol.
USAGE
}

# Optional leading positional: DATE [SYM ...] then flags (Bash 3.2).
if [ "$#" -gt 0 ] && is_config_date_token "$1"; then
  HAVE_DATE=1
  RAW_DATE="$1"
  shift
  pos_syms=""
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --*) break ;;
      *)
        pos_syms="$pos_syms $1"
        shift
        ;;
    esac
  done
  pos_syms="$(squeeze_trim "$pos_syms")"
  if [ -n "$pos_syms" ]; then
    HAVE_SYMBOLS=1
    SYMBOLS_ARG="$pos_syms"
  fi
fi

while [ "$#" -gt 0 ]; do
  case "$1" in
    --date)
      if [ "$#" -lt 2 ]; then
        echo "ERROR: --date requires a value" >&2
        exit 2
      fi
      HAVE_DATE=1
      RAW_DATE="$2"
      shift 2
      ;;
    --date=*)
      HAVE_DATE=1
      RAW_DATE="${1#--date=}"
      shift
      ;;
    --symbols)
      if [ "$#" -lt 2 ]; then
        echo "ERROR: --symbols requires a value" >&2
        exit 2
      fi
      HAVE_SYMBOLS=1
      SYMBOLS_ARG="$2"
      shift 2
      ;;
    --symbols=*)
      HAVE_SYMBOLS=1
      SYMBOLS_ARG="${1#--symbols=}"
      shift
      ;;
    --symbols-file)
      if [ "$#" -lt 2 ]; then
        echo "ERROR: --symbols-file requires a path" >&2
        exit 2
      fi
      HAVE_SYMBOLS_FILE=1
      SYMBOLS_FILE_PATH="$2"
      shift 2
      ;;
    --symbols-file=*)
      HAVE_SYMBOLS_FILE=1
      SYMBOLS_FILE_PATH="${1#--symbols-file=}"
      shift
      ;;
    --parallel)
      MODE="parallel"
      shift
      ;;
    --jobs|-j)
      if [ "$#" -lt 2 ]; then
        echo "ERROR: $1 requires a value" >&2
        exit 2
      fi
      JOBS="$2"
      shift 2
      ;;
    --jobs=*)
      JOBS="${1#--jobs=}"
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

finalize_parallel_jobs

if [ "$HAVE_DATE" -eq 1 ]; then
  CONFIG_DATE="$(normalize_config_date "$RAW_DATE")" || exit 2
else
  CONFIG_DATE="$CONFIG_DATE_DEFAULT"
fi

EXPLICIT_SYMBOL_SOURCE=0
if [ "$HAVE_SYMBOLS" -eq 1 ]; then
  EXPLICIT_SYMBOL_SOURCE=1
  SYMBOLS="$(squeeze_trim "$(printf '%s' "$SYMBOLS_ARG" | tr ',' ' ')")"
elif [ "$HAVE_SYMBOLS_FILE" -eq 1 ]; then
  EXPLICIT_SYMBOL_SOURCE=1
  SYMBOLS="$(load_symbols_from_file "$SYMBOLS_FILE_PATH")" || exit 2
elif [ "$HAVE_DATE" -eq 1 ]; then
  SYMBOLS="$(discover_symbols_for_date "$CONFIG_DATE")" || exit 2
else
  SYMBOLS="$SYMBOLS_DEFAULT"
fi

if [ -z "$SYMBOLS" ]; then
  echo "ERROR: resolved symbol list is empty (check --symbols / --symbols-file / defaults)" >&2
  exit 2
fi

if [ "$EXPLICIT_SYMBOL_SOURCE" -eq 1 ]; then
  validate_explicit_configs "$CONFIG_DATE" "$SYMBOLS"
fi

BATCH_TS="$(date -u +%Y%m%dT%H%M%SZ)"
BATCH_DIR="outputs/batch_${BATCH_TS}"
mkdir -p "$BATCH_DIR"
SUMMARY_FILE="$BATCH_DIR/batch_summary.txt"

{
  echo "Batch run started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "Mode: $MODE"
  if [ "$MODE" = "parallel" ]; then
    echo "Parallel jobs (max concurrent): $JOBS"
  fi
  echo "Iterative: $ITERATIVE  Max iterations: $MAX_ITERATIONS  Log level: $LOG_LEVEL"
  echo "Config date suffix: $CONFIG_DATE"
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

# Parallel worker: run_one then persist exit code for parent aggregation (bash 3.2-safe).
run_one_parallel_worker() {
  local sym="$1"
  local lower
  lower="$(lowercase "$sym")"
  local rc
  set +e
  run_one "$sym"
  rc=$?
  set -e
  echo "$rc" >"$BATCH_DIR/.exit.$lower"
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
  echo "[INFO] --parallel: running up to $JOBS symbols concurrently. Per-symbol output goes to log files only; tail ${BATCH_DIR}/<SYMBOL>.log to follow a specific run" >&2

  # Bash 3.2 + `set -u` treats `"${arr[@]}"` on an empty array as an unbound
  # variable, so wrap every expansion in the `${arr[@]+"${arr[@]}"}` idiom
  # (expands to the elements if set, to nothing if empty/unset).
  running_pids=()
  for sym in $SYMBOLS; do
    while [ "${#running_pids[@]}" -ge "$JOBS" ]; do
      new_pids=()
      for p in ${running_pids[@]+"${running_pids[@]}"}; do
        if kill -0 "$p" 2>/dev/null; then
          new_pids=(${new_pids[@]+"${new_pids[@]}"} "$p")
        fi
      done
      running_pids=(${new_pids[@]+"${new_pids[@]}"})
      [ "${#running_pids[@]}" -ge "$JOBS" ] && sleep 1
    done
    run_one_parallel_worker "$sym" &
    running_pids=(${running_pids[@]+"${running_pids[@]}"} "$!")
  done

  wait

  for sym in $SYMBOLS; do
    lower="$(lowercase "$sym")"
    ef="$BATCH_DIR/.exit.$lower"
    if [ ! -f "$ef" ]; then
      OVERALL_RC=1
      echo "[FAIL-EXITFILE] $sym  missing_exit_record=$ef" >>"$SUMMARY_FILE"
      continue
    fi
    rc="$(tr -d ' \t\n\r' <"$ef")"
    case "$rc" in
      ''|*[!0-9]*)
        OVERALL_RC=1
        echo "[FAIL-EXITFILE] $sym  invalid_exit_record=$ef" >>"$SUMMARY_FILE"
        ;;
      *)
        if [ "$rc" -ne 0 ]; then
          OVERALL_RC=1
        fi
        ;;
    esac
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
