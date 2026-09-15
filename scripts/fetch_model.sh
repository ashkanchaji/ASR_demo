#!/usr/bin/env bash
# One-time: download an official pre-converted CTranslate2 Whisper model.
#
# These are published by the faster-whisper maintainers themselves, so there is
# no conversion step and no third-party mirror in the chain. `hf` comes with
# huggingface_hub, which faster-whisper already depends on -- no extra install.
set -euo pipefail
cd "$(dirname "$0")/.."

SIZE=${1:-medium}
REPO="Systran/faster-whisper-${SIZE}"
OUT="models/faster-whisper-${SIZE}"

if [ -d "$OUT" ]; then
  echo "$OUT already exists. Delete it to re-download."
  exit 0
fi

# --revision main: these repos are stable and small enough to re-pull; pin a
# commit here if a rebuild ever needs to be byte-identical.
.venv/bin/hf download "$REPO" --local-dir "$OUT"

echo "fetched $REPO -> $OUT"
