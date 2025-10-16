#!/usr/bin/env bash
set -euo pipefail

if ! command -v uvicorn >/dev/null 2>&1; then
  echo "uvicorn is required. Install dependencies with 'python -m pip install -r backend/requirements.txt'." >&2
  exit 1
fi

exec uvicorn backend.app:create_app --factory --reload --host 0.0.0.0 --port 8000
