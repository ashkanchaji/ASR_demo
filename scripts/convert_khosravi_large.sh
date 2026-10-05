#!/usr/bin/env bash
# One-time: convert MohammadKhosravi/whisper-large-v3-Persian to CTranslate2 int8.
#
# Default Persian transcription model; use a stock model for language ID.
# This Persian-only fine-tune belongs in the fa slot, not the English slot
# or detector. See README.md for model preparation and role separation.
#
#   uv venv --python 3.12 .venv-convert
#   VIRTUAL_ENV=$PWD/.venv-convert uv pip install -r requirements-convert.txt
#   ./scripts/convert_khosravi_large.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# Pinned: no release tags, so "main" would silently change what ships.
REVISION=7804ebb2f91dd7efcca8e337c892c8e94e76a9b2
OUT=models/whisper-large-v3-persian-khosravi-ct2
SRC=${TMPDIR:-/tmp}/whisper-large-v3-persian-khosravi-$REVISION

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
snapshot_download("MohammadKhosravi/whisper-large-v3-Persian", revision=revision, local_dir=dest,
                  ignore_patterns=["runs/*", "training_args.bin"])
PY

"$CONVERTER" --model "$SRC" --output_dir "$OUT" --quantization int8 \
  --copy_files preprocessor_config.json tokenizer_config.json \
               special_tokens_map.json added_tokens.json normalizer.json

# Same situation as steja/whisper-small-persian: only the slow-tokenizer files
# (vocab.json + merges.txt) ship here, and without a tokenizer.json faster-whisper
# quietly fetches openai/whisper-tiny's from Hugging Face at runtime.
"$PY" - "$SRC" "$OUT" <<'PY'
import sys
from transformers import WhisperTokenizerFast
WhisperTokenizerFast.from_pretrained(sys.argv[1]).save_pretrained(sys.argv[2])
PY

echo "converted MohammadKhosravi/whisper-large-v3-Persian@$REVISION -> $OUT"
