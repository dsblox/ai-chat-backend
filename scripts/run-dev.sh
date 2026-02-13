#!/usr/bin/env bash
set -euo pipefail

# Restrict reload watching to app code and ignore virtualenv/git noise.
exec uvicorn app.main:app \
  --reload \
  --host 0.0.0.0 \
  --port 8000 \
  --reload-dir app \
  --reload-exclude ".venv/*" \
  --reload-exclude ".git/*"
