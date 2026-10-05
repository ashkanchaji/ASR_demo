# syntax=docker/dockerfile:1

# CPU-only: no CUDA base image and no cuDNN pinning, because production has no
# GPU budget and ctranslate2's CPU backend is self-contained in its wheel.
#
# Two stages, not one. The Systran sizes were pre-converted, so a plain download
# was the whole build and one stage was enough. The Persian fine-tune is
# published as PyTorch weights only, so conversion is back -- and torch plus
# transformers are ~1 GB of build-time tooling that must not ship in the image
# that serves requests.

FROM python:3.12-slim AS convert

WORKDIR /build
ENV PIP_NO_CACHE_DIR=1

COPY requirements-convert.txt .
RUN pip install --no-cache-dir -r requirements-convert.txt

# The same script developers run on the host, so the pinned revision and the
# tokenizer.json fix-up cannot drift between a local model and the image's.
COPY scripts/convert_model.sh scripts/convert_large_turbo.sh scripts/
RUN PYTHON=python CONVERTER=ct2-transformers-converter scripts/convert_model.sh
RUN PYTHON=python CONVERTER=ct2-transformers-converter scripts/convert_large_turbo.sh
# Stock English and wake models are already converted; fetch them directly.
RUN python - <<'PY'
from huggingface_hub import snapshot_download
for size in ("small.en", "tiny"):
    snapshot_download(f"Systran/faster-whisper-{size}",
                      local_dir=f"models/faster-whisper-{size}")
PY


FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Weights baked in, so the container makes no network call at request time.
# Above the source COPYs on purpose: ordinary code edits must not re-run the
# conversion stage's ~1 GB download.
COPY --from=convert /build/models/whisper-small-persian-ct2 \
                    /app/models/whisper-small-persian-ct2
COPY --from=convert /build/models/whisper-large-v3-turbo-ct2 \
                    /app/models/whisper-large-v3-turbo-ct2
COPY --from=convert /build/models/faster-whisper-small.en \
                    /app/models/faster-whisper-small.en
COPY --from=convert /build/models/faster-whisper-tiny \
                    /app/models/faster-whisper-tiny

COPY backend/ backend/
COPY frontend/ frontend/
COPY config/ config/

ENV WHISPER_FA_MODEL=./models/whisper-small-persian-ct2 \
    WHISPER_EN_MODEL=./models/faster-whisper-small.en \
    WHISPER_DETECT_MODEL=./models/whisper-large-v3-turbo-ct2 \
    WHISPER_WAKE_MODEL=./models/faster-whisper-tiny \
    WHISPER_DEVICE=cpu \
    WHISPER_COMPUTE=int8 \
    WHISPER_CPU_THREADS=0 \
    WHISPER_BEAM_SIZE=5 \
    WHISPER_HOTWORDS_FILE=./config/hotwords.json \
    PORT=8000 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 8000

# The app's lifespan hook loads the model before uvicorn serves anything, so a
# broken or missing model directory kills the container at boot with a clear
# message rather than failing on someone's first request.
CMD ["sh", "-c", "exec python3 -m uvicorn --app-dir backend main:app --host 0.0.0.0 --port ${PORT}"]
