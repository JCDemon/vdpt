#!/usr/bin/env bash
set -euo pipefail

exec uvicorn backend.app.main:create_app --factory --reload --host 0.0.0.0 --port 8000
