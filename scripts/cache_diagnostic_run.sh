#!/usr/bin/env bash
# Back-to-back equity_analyst runs with DEBUG logging to verify OpenAI prompt-cache
# prefix stability (hash lines) and cache hits (cache_read) on the second run.
#
# Usage:
#   cd /path/to/multi-llm-equity-analyst
#   source .venv/bin/activate
#   CONFIG=/path/to/your.yaml ./scripts/cache_diagnostic_run.sh
#
# Optional:
#   EXTRA_CLI_ARGS="--max-iterations 1"   # cheaper; iteration-2+ still uses same static prefix
#   CONFIG=... ./scripts/cache_diagnostic_run.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  echo "error: .venv not found under $ROOT" >&2
  exit 1
fi
# shellcheck source=/dev/null
source .venv/bin/activate

if [[ -z "${CONFIG:-}" ]]; then
  echo "error: set CONFIG to your YAML path, e.g. CONFIG=configs/crcl.yaml $0" >&2
  exit 1
fi

EXTRA_CLI_ARGS=${EXTRA_CLI_ARGS:-}

run_index() {
  local n="$1"
  echo ""
  echo "========== Run ${n} (same CONFIG, --log-level DEBUG) =========="
  python -m equity_analyst run --config "$CONFIG" --iterative --log-level DEBUG ${EXTRA_CLI_ARGS}
}

echo "Repo: $ROOT"
echo "OpenAI prefix hashes appear only at DEBUG: lines matching 'OpenAI request prefix'"
echo "Cache hits: lines matching 'OpenAI cache stats' (cache_read > 0 on run 2 if prefix warmed)"

run_index 1
run_index 2

echo ""
echo "---------- Prefix hashes (last two output dirs, OpenAI only) ----------"
mapfile -t dirs < <(ls -td "$ROOT"/outputs/*/ 2>/dev/null | head -2)
if [[ ${#dirs[@]} -lt 2 ]]; then
  echo "warning: expected at least two output dirs under outputs/" >&2
  exit 0
fi

hashes=()
for d in "${dirs[@]}"; do
  echo "--- ${d} ---"
  if [[ -f "${d}agent.log" ]]; then
    grep 'OpenAI request prefix' "${d}agent.log" || echo "(no DEBUG prefix lines; confirm --log-level DEBUG)"
    h="$(grep 'OpenAI request prefix' "${d}agent.log" | sed -n 's/.*hash=\([a-f0-9]\{16\}\).*/\1/p' | sort -u | tr '\n' ' ')"
    hashes+=("$h")
  else
    echo "missing agent.log"
  fi
  echo "OpenAI cache stats:"
  grep 'OpenAI cache stats' "${d}agent.log" || true
done

echo ""
if [[ ${#hashes[@]} -eq 2 && -n "${hashes[0]}" && -n "${hashes[1]}" ]]; then
  if [[ "${hashes[0]}" == "${hashes[1]}" ]]; then
    echo "Prefix hash set: MATCH between the two newest output dirs."
  else
    echo "Prefix hash set: DIFFER (investigate config or request shape between runs)."
  fi
fi
echo "Success for caching: run 2 shows cache_read>0 on at least one OpenAI fan-out line (after idle, run 1 may still be 0)."
