#!/usr/bin/env bash
# Quick OpenAI prompt-cache verifier.
#
# Runs configs/cache_test_openai.yaml twice back-to-back (standard, non-iterative)
# at --log-level DEBUG so the OpenAI provider emits both:
#   * `OpenAI request prefix ... hash=<...> instructions_sha16=<16hex> len=<...>`
#   * `OpenAI cache stats cache_read=<N> input=<...> output=<...>`
#
# After both runs complete it greps the two newest output dirs and prints a
# single PASS/FAIL verdict:
#   * `instructions_sha16` must match between the two runs (prompt prefix is stable).
#   * Run 2's max `cache_read` must be > 0 (caching is actually firing).
#
# Bash 3.2 compatible (macOS /bin/bash): no mapfile, no ${var,,}, no associative
# arrays. Exits 0 on PASS, 1 on FAIL.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CONFIG="configs/cache_test_openai.yaml"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "ERROR: $PYTHON_BIN not found or not executable. Create .venv per README." >&2
  exit 2
fi
if [ ! -f "$CONFIG" ]; then
  echo "ERROR: missing $CONFIG" >&2
  exit 2
fi

run_index() {
  # Run equity_analyst with the cache-test config at DEBUG. The standard
  # (non-iterative) orchestrator is enough: fan-out + synthesizer.
  n="$1"
  echo ""
  echo "========== Run ${n} (CONFIG=${CONFIG} --no-web-search --log-level DEBUG) =========="
  "$PYTHON_BIN" -m equity_analyst run \
    --config "$CONFIG" \
    --no-web-search \
    --log-level DEBUG
}

run_index 1
run_index 2

# Pick the two newest output dirs. The repo writes one per run under
# outputs/<SYMBOL>_<TS>/ and we just produced exactly two.
dirs=""
count=0
for d in $(ls -td "$REPO_ROOT"/outputs/*/ 2>/dev/null); do
  dirs="$dirs $d"
  count=$((count + 1))
  if [ "$count" -ge 2 ]; then
    break
  fi
done

# shellcheck disable=SC2086
set -- $dirs
if [ "$#" -lt 2 ]; then
  echo "[FAIL] expected at least two output dirs under outputs/; found $#" >&2
  exit 1
fi

# After `set --` above, $1 is the newest dir (= run 2) and $2 is run 1.
RUN2_DIR="$1"
RUN1_DIR="$2"

extract_sha16() {
  # Echo space-separated unique instructions_sha16 values from agent.log (DEBUG only).
  log_path="$1"
  if [ ! -f "$log_path" ]; then
    return 0
  fi
  grep 'OpenAI request prefix' "$log_path" 2>/dev/null \
    | sed -n 's/.*instructions_sha16=\([a-f0-9]\{16\}\).*/\1/p' \
    | sort -u \
    | tr '\n' ' '
}

extract_max_cache_read() {
  # Echo the largest cache_read=<N> integer in agent.log, or 0 if none seen.
  log_path="$1"
  if [ ! -f "$log_path" ]; then
    echo 0
    return 0
  fi
  max=0
  for v in $(grep 'OpenAI cache stats' "$log_path" 2>/dev/null \
    | sed -n 's/.*cache_read=\([0-9][0-9]*\).*/\1/p'); do
    if [ "$v" -gt "$max" ]; then
      max="$v"
    fi
  done
  echo "$max"
}

LOG1="${RUN1_DIR}agent.log"
LOG2="${RUN2_DIR}agent.log"

if [ ! -f "$LOG1" ] || [ ! -f "$LOG2" ]; then
  echo "[FAIL] missing agent.log in one of:"
  echo "  $LOG1"
  echo "  $LOG2"
  exit 1
fi

SHA1="$(extract_sha16 "$LOG1")"
SHA2="$(extract_sha16 "$LOG2")"
CR1="$(extract_max_cache_read "$LOG1")"
CR2="$(extract_max_cache_read "$LOG2")"

# Normalize whitespace so two equivalent single-hash strings compare equal.
SHA1_TRIM="$(printf '%s' "$SHA1" | tr -s ' ' | sed 's/^ *//;s/ *$//')"
SHA2_TRIM="$(printf '%s' "$SHA2" | tr -s ' ' | sed 's/^ *//;s/ *$//')"

echo ""
echo "---------- OpenAI cache validation summary ----------"
echo "Run 1 dir:         ${RUN1_DIR}"
echo "Run 2 dir:         ${RUN2_DIR}"
echo "Run 1 cache_read:  ${CR1}"
echo "Run 2 cache_read:  ${CR2}"
echo "Run 1 sha16(s):    ${SHA1_TRIM:-<none>}"
echo "Run 2 sha16(s):    ${SHA2_TRIM:-<none>}"

if [ -z "$SHA1_TRIM" ] || [ -z "$SHA2_TRIM" ]; then
  echo "[FAIL] No instructions_sha16 lines found. Did the runs use --log-level DEBUG and OpenAI fan-out?"
  exit 1
fi

if [ "$SHA1_TRIM" != "$SHA2_TRIM" ]; then
  echo "[FAIL] instructions_sha16 MISMATCH between runs."
  echo "       run1=${SHA1_TRIM}"
  echo "       run2=${SHA2_TRIM}"
  exit 1
fi

if [ "$CR2" -le 0 ]; then
  echo "[FAIL] Run 2 cache_read=${CR2}; expected >0 (cache did not warm)."
  echo "       instructions_sha16 matched: ${SHA1_TRIM}"
  exit 1
fi

echo "[PASS] Run 1 cache_read=${CR1}, Run 2 cache_read=${CR2} (N>0). instructions_sha16 matched: ${SHA1_TRIM}"
exit 0
