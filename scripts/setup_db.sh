#!/usr/bin/env bash
set -euo pipefail

if [ -f "$(dirname "$0")/../.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$(dirname "$0")/../.env"
  set +a
fi

# Create the database if it doesn't exist
psql -h localhost -U college_brain -d postgres -tc "SELECT 1 FROM pg_database WHERE datname='multi_llm_equity_runs'" \
  | grep -q 1 || \
  psql -h localhost -U college_brain -d postgres -c "CREATE DATABASE multi_llm_equity_runs"

# Run migrations
cd "$(dirname "$0")/.."
.venv/bin/alembic upgrade head

echo "DB ready."
