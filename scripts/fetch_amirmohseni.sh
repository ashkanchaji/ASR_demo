#!/usr/bin/env bash
# One-time: download AmirMohseni/whisper-large-v3-persian-ct2-int8. Already a
# CTranslate2 int8 directory (model.bin + tokenizer.json), so this is a plain
# fetch, same shape as fetch_model.sh -- no conversion step.
#
# COMPARISON MODEL, not a vetted default -- see AGENTS.md. Third-hand: a
# fine-tune (MohammadGholizadeh) reworked (AmirMohseni's bf16) then converted
# here by a third individual, empty README, no WER/CER or English numbers
# published anywhere in that chain. This is the "third-party mirror,
# unmaintained re-upload" case the rest of this file's model choices avoid --
# fetched for comparison only, on request.
set -euo pipefail
cd "$(dirname "$0")/.."

REPO="AmirMohseni/whisper-large-v3-persian-ct2-int8"
REVISION=5f850f99dc4db15a526e42b73027481a9f24327d
OUT="models/whisper-large-v3-persian-amirmohseni-ct2"

if [ -d "$OUT" ]; then
  echo "$OUT already exists. Delete it to re-download."
  exit 0
fi

.venv/bin/hf download "$REPO" --revision "$REVISION" --local-dir "$OUT"

echo "fetched $REPO@$REVISION -> $OUT"
