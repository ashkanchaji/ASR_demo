#!/usr/bin/env bash
# Start the demo: http://127.0.0.1:8000
set -euo pipefail
cd "$(dirname "$0")"

# Bare metal only, and only for WHISPER_DEVICE=cuda. ctranslate2 dlopens
# libcudnn.so.9 / libcublas.so.12 rather than linking them, and this machine has
# neither system-wide, so point the loader at the wheels in the venv. The image
# is CPU-only and needs none of this.
libs=$(printf '%s:' .venv/lib/python*/site-packages/nvidia/*/lib 2>/dev/null || true)
export LD_LIBRARY_PATH="${libs}${LD_LIBRARY_PATH:-}"

if [ "${1:-}" = "--selfcheck" ]; then
  exec .venv/bin/python backend/selfcheck.py
fi

exec .venv/bin/uvicorn --app-dir backend main:app --host 127.0.0.1 --port "${PORT:-8000}" "$@"
