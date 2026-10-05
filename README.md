# Persian & English ASR Demo

A local speech-to-text demo with a browser microphone interface and a Python backend powered by **faster-whisper** and **CTranslate2**. Speak Persian or English, then see the transcript, language, and processing time.

This project demonstrates automatic speech recognition (ASR). It does not resolve commands into URLs or call a navigation service on another device.

## Features

- Tap to record; capture stops after a pause or a five-second limit.
- Optional “hey PUM” / “سلام پام” wake-word mode, using a separate tiny Whisper model.
- Automatic Persian/English routing through a stock multilingual detector, or a manual language choice.
- Separate Persian and English transcription models, selectable in the browser.
- Persian text normalization and vocabulary-based spelling correction.
- Local model inference on CPU, with optional CUDA support and CPU fallback.
- One HTML frontend, no JavaScript build step, and a FastAPI backend.

Model preparation needs internet access. The backend uses local model directories at runtime and does not forward transcripts to another service. The browser loads an optional Google Fonts stylesheet; fallback system fonts work without it.

## Quick start

Use **Python 3.12** and [uv](https://docs.astral.sh/uv/). The default models require several GB of disk space and enough RAM to keep four models resident. Large models can be slow on CPU; CUDA is optional.

```bash
git clone https://github.com/ashkanchaji/ASR_demo.git
cd ASR_demo

uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r backend/requirements.txt

# Separate environment: conversion tools are not runtime dependencies.
uv venv --python 3.12 .venv-convert
uv pip install --python .venv-convert/bin/python -r requirements-convert.txt

# Prepare all four default model roles once.
./scripts/convert_khosravi_large.sh
./scripts/convert_large_turbo.sh
./scripts/fetch_model.sh small.en
./scripts/fetch_model.sh tiny

cp .env.example .env
./run.sh
```

Open **http://127.0.0.1:8000**, allow microphone access, and tap the microphone button. Try “show me the settings page” or “صفحه تنظیمات رو نشون بده”. The demo displays what you said; it does not open a page.

For wake-word mode, enable **Always listening for “hey PUM”**. Say the name followed by your phrase, or pause after the name and speak when prompted. The browser resumes listening after processing each clip. Tap the microphone while processing to abandon that request and record again.

Microphone access requires localhost or HTTPS. This demo has no authentication and binds to localhost by default.

### Model roles

| Role | Default model | Preparation |
| --- | --- | --- |
| Persian transcription | `MohammadKhosravi/whisper-large-v3-Persian` | `./scripts/convert_khosravi_large.sh` |
| English transcription | `Systran/faster-whisper-small.en` | `./scripts/fetch_model.sh small.en` |
| Persian/English language detection | `openai/whisper-large-v3-turbo` | `./scripts/convert_large_turbo.sh` |
| Wake-word probe | `Systran/faster-whisper-tiny` | `./scripts/fetch_model.sh tiny` |

All roles load at startup. Model instances are shared when roles point to the same directory. Conversion scripts pin source revisions and write int8 CTranslate2 weights; model files are not committed.

The detector is a stock multilingual model because Persian-only fine-tunes can misclassify English as Persian. An English-only `.en` model cannot serve as the detector or wake probe. Accuracy depends on accent, recording quality, and vocabulary; automated smoke checks do not measure recognition accuracy.

For a lighter Persian transcription model, prepare `./scripts/convert_model.sh` and set:

```dotenv
WHISPER_FA_MODEL=./models/whisper-small-persian-ct2
```

Other available models appear in the browser selectors. A selection changes that language slot for subsequent requests until changed again or the server restarts. Boot defaults come from `.env`.

### Optional CUDA

Install the local GPU libraries, then set these values in `.env`:

```bash
uv pip install --python .venv/bin/python -r backend/requirements-local-cuda.txt
```

```dotenv
WHISPER_DEVICE=cuda
WHISPER_COMPUTE=float16
```

`run.sh` adds the installed NVIDIA libraries to the loader path. If CUDA model loading fails, the backend logs a warning and retries on CPU with int8. The default large Persian model and detector can consume substantial VRAM together.

## Configuration

Copy [.env.example](.env.example) to `.env`; restart the server after changing it. Key settings:

| Setting | Purpose |
| --- | --- |
| `WHISPER_FA_MODEL`, `WHISPER_EN_MODEL` | Local transcription model directories |
| `WHISPER_DETECT_MODEL`, `WHISPER_WAKE_MODEL` | Local detector and wake probe directories |
| `WHISPER_MODELS_DIR` | Directory scanned for selectable CTranslate2 models |
| `WHISPER_DEVICE`, `WHISPER_COMPUTE` | CPU/CUDA and compute type |
| `WHISPER_CPU_THREADS` | CPU thread count; `0` uses the library default |
| `WHISPER_BEAM_SIZE` | Decoding beam width; `1` is cheaper than `5` |
| `WHISPER_FA_MIN_PROB` | Persian share of Persian/English probability mass required to route to Persian |
| `WHISPER_HOTWORDS_FILE` | JSON vocabulary hints for each language |
| `SPELL_CORRECTION_ENABLED`, `SPELL_CORRECTION_THRESHOLD` | Persian spelling correction controls |
| `SPELL_CORRECTION_LEXICON` | Optional local Persian word list |
| `WHISPER_SAVE_CLIPS` | Debug audio recording; off by default |
| `WHISPER_SAMPLES_DIR` | Debug recording directory; defaults to `eval/samples` |

`PORT=9000 ./run.sh` changes the listening port. The launcher reads `PORT` from the shell; the Python application reads model settings from `.env`.

Edit [config/hotwords.json](config/hotwords.json) to bias decoding toward your vocabulary. Persian spelling correction uses those terms plus [config/persian-words.txt](config/persian-words.txt), when present. Disabling correction still preserves Persian normalization. English output is unchanged.

## Docker

```bash
docker build -t asr-demo .
docker run --rm -p 127.0.0.1:8000:8000 asr-demo
```

The CPU-only image prepares and includes all four model roles. It uses the smaller `steja/whisper-small-persian` for Persian transcription, plus `small.en`, the stock turbo detector, and `tiny`. Local runs use the larger Khosravi Persian model by default. The initial build downloads model weights and conversion dependencies, so expect a large image and a slow first build. Conversion tools stay in the build stage.

## API

- `GET /` serves the browser interface.
- `GET /health` returns active models, model allow-lists, language choices, and decoding settings.
- `POST /transcribe` accepts multipart `audio` and optional `language`, `fa_model`, `en_model`, and `wake` fields.

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/transcribe \
  -F 'audio=@sample.webm' -F 'language=en'
```

Example response (timing is illustrative):

```json
{
  "text": "show me the settings page",
  "detected_language": "en",
  "processing_time_ms": 820,
  "model": "faster-whisper-small.en"
}
```

`language` accepts `auto`, `fa`, or `en`. Automatic routing also returns `language_probs` and `detector`. `wake=true` checks whether the clip begins with the wake name and returns `woken`; a rejected wake probe returns empty text before language detection. Successful wake transcripts omit the wake phrase.

Empty or speechless clips return `422` with `no_speech`. Invalid model/language choices return `422`; inference failures return `500` with `transcribe_failed`.

## Checks and evaluation

The smoke check needs prepared models, `httpx` for FastAPI's test client, and an `ffmpeg` executable to generate a test WebM clip. Serving audio itself uses PyAV and does not need an ffmpeg subprocess.

```bash
uv pip install --python .venv/bin/python httpx
SPELL_CORRECTION_ENABLED=true SPELL_CORRECTION_THRESHOLD=0.85 \
  WHISPER_FA_MIN_PROB=0.5 ./run.sh --selfcheck
SPELL_CORRECTION_ENABLED=true SPELL_CORRECTION_THRESHOLD=0.85 \
  .venv/bin/python backend/spellcheck.py
.venv/bin/python scripts/evaluate_model.py --check
SPELL_CORRECTION_ENABLED=true SPELL_CORRECTION_THRESHOLD=0.85 \
  .venv/bin/python scripts/evaluate_model.py --spell
```

The self-check exercises language routing, model loading, wake-word parsing, cancellation, input validation, and transcript responses with outbound connections blocked. Run with the default spelling settings and `WHISPER_FA_MIN_PROB=0.5` for its default-setting assertions.

To evaluate your voice, start with `WHISPER_SAVE_CLIPS=1 ./run.sh`, record examples, then stop saving. Set each filename's language and reference text in [eval/reference.json](eval/reference.json). Audio files are intentionally excluded from Git.

```bash
.venv/bin/python scripts/evaluate_model.py --detect \
  --models models/whisper-large-v3-turbo-ct2
.venv/bin/python scripts/evaluate_model.py \
  --models models/whisper-small-persian-ct2 --beams 1 5
```

Debug recording has no rotation or storage cap. Keep it off outside deliberate evaluation. The supplied reference JSON illustrates the format; its recordings are not included.

## Project layout

```text
backend/
  main.py                       FastAPI, model cache, routing, wake probe
  spellcheck.py                 Persian normalization and correction
  selfcheck.py                  Smoke and regression checks
  requirements.txt             Runtime dependencies
  requirements-local-cuda.txt   Optional local NVIDIA libraries
frontend/index.html            Microphone UI, capture, silence detection
config/                        Vocabulary hints and Persian word list
scripts/                       Model preparation and evaluation tools
eval/                          Reference text and spelling test cases
graphify-out/                  Code graph and architecture report
Dockerfile                     CPU image with prepared models
run.sh                         Local launcher
.env.example                   Documented settings
requirements-convert.txt       Separate model conversion dependencies
```

## Code graph

[Graphify's report](graphify-out/GRAPH_REPORT.md) and [graph.json](graphify-out/graph.json) map the code's symbols and calls. The initial graph used local `--code-only` extraction; `graphify update .` also added Markdown structure and symbol references. Inline browser JavaScript, configuration, and Docker behavior were reviewed directly in source.

```bash
uvx --from graphifyy graphify extract . --code-only --max-workers 2
uvx --from graphifyy graphify cluster-only . --no-label --no-viz
uvx --from graphifyy graphify query 'pick_model_and_language' --context call
# After code edits:
uvx --from graphifyy graphify update .
```

The main hubs are `transcribe`, `get_model`, `pick_model_and_language`, and `spell_correct`. Model preparation, browser capture, inference, spelling correction, and evaluation remain separate parts of the demo.

## Credits

Built with [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [CTranslate2](https://github.com/OpenNMT/CTranslate2), [FastAPI](https://github.com/fastapi/fastapi), and Whisper models distributed through Hugging Face. Model weights and dependencies retain their respective upstream licenses; check the selected model's terms before redistribution.
