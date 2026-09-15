#!/usr/bin/env bash
# One-time: convert the Persian fine-tune to CTranslate2 int8.
#
# Only this model needs converting. steja/whisper-small-persian is published as
# PyTorch weights and has no official CTranslate2 release, unlike the plain
# Systran/faster-whisper-* sizes -- fetch those with ./scripts/fetch_model.sh
# instead of running them through here.
#
#   uv venv --python 3.12 .venv-convert
#   VIRTUAL_ENV=$PWD/.venv-convert uv pip install -r requirements-convert.txt
#   ./scripts/convert_model.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# Pinned: the repo has no release tags, so a floating "main" would silently
# change what ships. Bump this deliberately, and re-check English (see README).
REVISION=8c600b6b69f552730abab77a6a46311d673ea584
OUT=models/whisper-small-persian-ct2
SRC=${TMPDIR:-/tmp}/whisper-small-persian-$REVISION

if [ -d "$OUT" ]; then
  echo "$OUT already exists. Delete it to re-convert."
  exit 0
fi

PY=${PYTHON:-.venv-convert/bin/python}
CONVERTER=${CONVERTER:-.venv-convert/bin/ct2-transformers-converter}

# Root files only. The repo also carries ten training checkpoints of optimizer
# and RNG state -- tens of GB the converter never opens.
"$PY" - "$REVISION" "$SRC" <<'PY'
import sys
from huggingface_hub import snapshot_download
revision, dest = sys.argv[1], sys.argv[2]
snapshot_download("steja/whisper-small-persian", revision=revision, local_dir=dest,
                  allow_patterns=["*.json", "*.txt", "pytorch_model.bin"],
                  ignore_patterns=["checkpoint-*/*"])
PY

"$CONVERTER" --model "$SRC" --output_dir "$OUT" --quantization int8 \
  --copy_files preprocessor_config.json tokenizer_config.json \
               special_tokens_map.json added_tokens.json normalizer.json

# faster-whisper loads tokenizer.json from the model directory. This repo ships
# only the slow-tokenizer files (vocab.json + merges.txt), and without a
# tokenizer.json faster-whisper silently falls back to fetching openai/whisper-tiny's
# from Hugging Face -- which is a startup crash under HF_HUB_OFFLINE=1, and a
# runtime network call without it. Build it once, here.
"$PY" - "$SRC" "$OUT" <<'PY'
import sys
from transformers import WhisperTokenizerFast
WhisperTokenizerFast.from_pretrained(sys.argv[1]).save_pretrained(sys.argv[2])
PY

echo "converted steja/whisper-small-persian@$REVISION -> $OUT"
