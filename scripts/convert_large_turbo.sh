#!/usr/bin/env bash
# One-time: convert openai/whisper-large-v3-turbo to CTranslate2 int8.
#
# No official Systran/faster-whisper-* pre-conversion exists for the turbo size
# (unlike small/medium -- see fetch_model.sh), so it goes through the converter
# like the Persian fine-tune. Unlike that fine-tune, this repo already ships a
# fast tokenizer.json, so no tokenizer fix-up step is needed here.
#
#   uv venv --python 3.12 .venv-convert
#   VIRTUAL_ENV=$PWD/.venv-convert uv pip install -r requirements-convert.txt
#   ./scripts/convert_large_turbo.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# Pinned: consistent with convert_model.sh -- bump deliberately.
REVISION=41f01f3fe87f28c78e2fbf8b568835947dd65ed9
OUT=models/whisper-large-v3-turbo-ct2
SRC=${TMPDIR:-/tmp}/whisper-large-v3-turbo-$REVISION

if [ -d "$OUT" ]; then
  echo "$OUT already exists. Delete it to re-convert."
  exit 0
fi

PY=${PYTHON:-.venv-convert/bin/python}
CONVERTER=${CONVERTER:-.venv-convert/bin/ct2-transformers-converter}

"$PY" - "$REVISION" "$SRC" <<'PY'
import sys
from huggingface_hub import snapshot_download
revision, dest = sys.argv[1], sys.argv[2]
snapshot_download("openai/whisper-large-v3-turbo", revision=revision, local_dir=dest,
                  allow_patterns=["*.json", "*.txt", "model.safetensors"])
PY

"$CONVERTER" --model "$SRC" --output_dir "$OUT" --quantization int8

echo "converted openai/whisper-large-v3-turbo@$REVISION -> $OUT"
