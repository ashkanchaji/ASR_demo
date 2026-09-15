"""Phase-1 voice-to-text demo: one POST endpoint in front of faster-whisper.

No navigation, no persistence, no auth. It exists to prove the capture ->
transcribe -> display loop is fast and accurate enough in Persian and English.

The service never reaches the network: the model is a converted CTranslate2
directory on disk, and a missing one is a startup error, not a silent download.

Two models are resident for actual transcription, one per language slot (fa/en)
-- see _active below. WHISPER_FA_MODEL/WHISPER_EN_MODEL in .env set which model
fills each slot at boot; the frontend can swap either slot at runtime (a select
per language, backed by /health), which is a sticky change -- it updates the
active slot for every request after it, not just the one that sent it. The
client sends a model *name* that must appear in the list /health publishes for
that slot; a path from the client is never opened. Two more models are
resident but not swappable from the GUI: WHISPER_DETECT_MODEL (fa/en language
ID) and WHISPER_WAKE_MODEL (a cheap always-on wake-word probe, see the "Wake
word" section below).
"""

import asyncio
import inspect
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from difflib import get_close_matches
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio

from spellcheck import (ENABLED as SPELL_ENABLED, THRESHOLD as SPELL_THRESHOLD,
                        normalize, spell_correct)

load_dotenv()  # before every os.getenv below, so a local .env actually counts

# Boot-time default for each language slot -- see the module docstring and
# _active below. Both are env-driven so a deploy can pin a different pair
# without a code change; MODELS_DIR is where _discover_models() looks for
# everything else the GUI may swap a slot to.
MODELS_DIR = Path(os.getenv("WHISPER_MODELS_DIR", "./models"))
# Resolved to an absolute path immediately: "./models/x" (a typical .env value)
# and "models/x" (what Path.iterdir() below yields for the same directory) are
# different strings for the same directory otherwise, which broke dict lookups
# and set-membership checks that assume one canonical spelling per model.
def _resolved(value: str) -> str:
    return str(Path(value).resolve())


# large-v3 Persian fine-tune -- heavier than the old small-persian default (see
# CLAUDE.md's WER/vetting note), chosen anyway because accuracy is the point.
FA_MODEL = _resolved(os.getenv("WHISPER_FA_MODEL", str(MODELS_DIR / "whisper-large-v3-persian-khosravi-ct2")))
# Official Systran CT2, English-only by construction -- no conversion step,
# and no detect_language() either, see pick_model_and_language().
EN_MODEL = _resolved(os.getenv("WHISPER_EN_MODEL", str(MODELS_DIR / "faster-whisper-small.en")))
# Language ID is its own job, done by a STOCK model -- never by a Persian
# fine-tune, including the small one. Fine-tuning on Persian-only data retrains
# the language-ID head along with the decoder, and every fine-tune here is
# saturated at fa as a result. Measured on two real clips from this machine
# (scripts/evaluate_model.py --detect), fa share of the fa+en mass:
#
#   detector                      fa clip   en clip ("show me the settings page")
#   whisper-small-persian          1.000     1.000  <- P(fa)=.9996 vs .9995, identical
#   khosravi large-v3 fa           1.000     0.881
#   amirmohseni large-v3 fa        1.000     0.844
#   whisper-large-v3-turbo         0.930     0.000  <- the only one that separates
#
# The fine-tunes do not answer "fa" too *often*, they answer it unconditionally,
# which no FA_MIN_PROB threshold can rescue -- that is why this must be a stock
# model and why the knob below is a trim, not the fix. Accented English is
# exactly the case README flagged as untested for the fine-tune; it fails hard.
DETECT_MODEL = _resolved(os.getenv("WHISPER_DETECT_MODEL",
                                   str(MODELS_DIR / "whisper-large-v3-turbo-ct2")))
# The wake-word check gets its own tiny, isolated model -- see the "Wake word"
# section below. Never the fa/en slot or DETECT_MODEL: the whole point is a
# fast, cheap pass that filters out ambient noise and false triggers before
# any of the big models run. Stock and multilingual (never a fine-tune, never
# ".en") for the same reason as DETECT_MODEL -- the wake phrase has Persian
# spellings too ("سلام پام").
WAKE_MODEL = _resolved(os.getenv("WHISPER_WAKE_MODEL", str(MODELS_DIR / "faster-whisper-tiny")))
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE", "float16" if DEVICE == "cuda" else "int8")
# 0 is ctranslate2's own "use every core". Overridable because throughput on CPU
# tracks core count closely and the target machine's is not known here.
CPU_THREADS = int(os.getenv("WHISPER_CPU_THREADS", "0"))
# Beam search is the most expensive decoding knob on CPU. Tunable so the eval's
# beam_size=1 vs 5 verdict can be applied here without a code change.
BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "5"))

# The closed set. Every language decision in this file is a choice between these
# two and nothing else -- see closest_language().
LANGUAGES = ("fa", "en")
# Share of the fa+en mass that must be fa before a clip routes to Persian. 0.5
# is exactly argmax (ties included). On the measured clips the stock detector
# splits 0.930 (fa) against 0.000 (en), so anything from 0.05 to 0.9 scores the
# same and a fancier default would only bake in a guess. This is a trim for a
# detector that is already close, NOT a repair for a biased one -- a saturated
# fine-tune reads 1.000 on both languages and no threshold can separate those.
FA_MIN_PROB = float(os.getenv("WHISPER_FA_MIN_PROB", "0.5"))
# No env pin: boot always auto-detects, a per-request "language" field is the
# only way to force one -- see resolve_language().
FORCED_LANGUAGE = None

# Vocabulary biasing: app-specific terms faster-whisper otherwise mishears.
# Optional -- a missing or empty file just means no biasing, not a startup error.
HOTWORDS_FILE = os.getenv("WHISPER_HOTWORDS_FILE", "./config/hotwords.json")
try:
    HOTWORDS = json.loads(Path(HOTWORDS_FILE).read_text())
except FileNotFoundError:
    HOTWORDS = {}
# hotwords= landed in faster-whisper after some 1.x releases; initial_prompt is the
# same idea (a string folded into the decoder prompt) and has always existed.
HOTWORDS_PARAM = "hotwords" if "hotwords" in inspect.signature(WhisperModel.transcribe).parameters \
    else "initial_prompt"

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"

# A webm/opus clip shorter than this is a mis-fire, not speech. MediaRecorder
# emits a ~1KB container even when nothing was said.
MIN_CLIP_BYTES = 2000

# Off by default. On, every upload is kept in eval/samples/ so the evaluation
# harness finally has real audio to score -- this repo ships no clips, which is
# why the fa/en routing bug got fixed blind twice before anyone could measure it.
# Recording through the demo itself captures exactly the browser's webm/opus.
# ponytail: no rotation, no cap, no cleanup -- a debug switch, not a feature.
# If it ever needs to be on in a real deployment, that is when it needs those.
SAVE_CLIPS = os.getenv("WHISPER_SAVE_CLIPS", "").lower() in ("1", "true", "yes")
SAMPLES_DIR = Path(os.getenv("WHISPER_SAMPLES_DIR", "./eval/samples"))

# Optional: mirror each transcript to a paired PC's /api/external/messages
# (same Wi-Fi network) -- see .env.example. Empty host disables the push
# outright, no network hop attempted.
EXTERNAL_API_HOST = os.getenv("EXTERNAL_API_HOST", "")
EXTERNAL_API_PORT = os.getenv("EXTERNAL_API_PORT", "8000")
EXTERNAL_API_URL = (f"http://{EXTERNAL_API_HOST}:{EXTERNAL_API_PORT}/api/external/messages"
                    if EXTERNAL_API_HOST else None)
# The nav service resolves through an LLM (+ MCP calls), so this needs real
# headroom -- 5s was timing out requests the other side went on to answer,
# which read here as "unreachable" when it was actually just slow.
EXTERNAL_API_TIMEOUT = int(os.getenv("EXTERNAL_API_TIMEOUT", "30"))


def _post_external(text: str) -> dict:
    """Blocking POST + the nav service's own contract for reporting failure:
    business-logic outcomes (ambiguous/not_found/insufficient_context) come
    back as HTTP 200 and are not errors here; a 503 means one of *its*
    dependencies (model/registry/DB) is down and carries {"detail": ...}."""
    body = json.dumps({"text": text}).encode()
    req = urllib.request.Request(EXTERNAL_API_URL, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=EXTERNAL_API_TIMEOUT) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = json.loads(exc.read()).get("detail", exc.reason)
        raise RuntimeError(detail) from exc


async def push_external(text: str) -> tuple[dict | None, str | None]:
    """Best-effort call to the paired PC's /api/external/messages. Returns
    (parsed response, None) on success or (None, error string) on any
    failure -- never raises, so the GUI keeps showing the transcript even
    when the other machine is off, unreachable, or 503s."""
    if not EXTERNAL_API_URL:
        return None, None
    try:
        return await asyncio.to_thread(_post_external, text), None
    except Exception as exc:
        print(f"[warn] external push to {EXTERNAL_API_URL} failed: {exc}")
        return None, str(exc)


def save_clip(raw: bytes, filename: str | None) -> None:
    """Best-effort: a failed debug write must never cost someone a transcript."""
    try:
        SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        suffix = Path(filename or "clip.webm").suffix or ".webm"
        path = SAMPLES_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 1000:03d}{suffix}"
        path.write_bytes(raw)
        print(f"[clip] saved {path}")
    except OSError as exc:
        print(f"[warn] could not save clip: {exc}")

for _path, _var in ((FA_MODEL, "WHISPER_FA_MODEL"), (EN_MODEL, "WHISPER_EN_MODEL"),
                    (DETECT_MODEL, "WHISPER_DETECT_MODEL"), (WAKE_MODEL, "WHISPER_WAKE_MODEL")):
    if not Path(_path).is_dir():
        sys.exit(
            f"{_path!r} ({_var}) is not a directory on disk.\n"
            "This service never downloads weights at runtime. Build/fetch it with the "
            "matching scripts/convert_*.sh or scripts/fetch_*.sh (see CLAUDE.md), or "
            f"point {_var} at an existing CT2 model directory."
        )

# Caught at boot rather than on the first auto-detect request: faster-whisper
# refuses detect_language() on an English-only model, and a detector that cannot
# detect is a 500 per clip, not a degraded answer.
if Path(DETECT_MODEL).name.endswith(".en"):
    sys.exit(
        f"{DETECT_MODEL!r} (WHISPER_DETECT_MODEL) is English-only, so it has no "
        "detect_language() to call.\nPoint WHISPER_DETECT_MODEL at a multilingual "
        "model -- whisper-small-persian-ct2 is the default and the measured choice."
    )

# The wake word has Persian spellings too -- an English-only model here would
# silently mangle every "سلام پام", not fail loudly the way the check above does.
if Path(WAKE_MODEL).name.endswith(".en"):
    sys.exit(
        f"{WAKE_MODEL!r} (WHISPER_WAKE_MODEL) is English-only.\nPoint "
        "WHISPER_WAKE_MODEL at a multilingual model -- faster-whisper-tiny is the default."
    )


def _discover_models() -> dict[str, str]:
    """Every CT2 model directory found in MODELS_DIR, by bare name -- the pool
    either language slot's select can be swapped to. model.bin is what makes a
    directory a CTranslate2 model, so a half-finished conversion or a stray
    folder is not offered.
    """
    found = {p.name: _resolved(str(p)) for p in sorted(MODELS_DIR.iterdir())
             if p.is_dir() and (p / "model.bin").exists()}
    found.setdefault(Path(FA_MODEL).name, FA_MODEL)  # the configured ones are
    found.setdefault(Path(EN_MODEL).name, EN_MODEL)  # always offered, even if elsewhere on disk
    return found


MODELS = _discover_models()
# small.en-style directories are English-only by construction (no
# detect_language(), no fa tokens) -- Systran's own naming convention for that
# ("small.en", "base.en", ...) is the one signal we have, and it's reliable.
FA_MODELS = {k: v for k, v in MODELS.items() if not Path(v).name.endswith(".en")}
EN_MODELS = MODELS  # any multilingual model can be forced to English; see do_transcribe()


# The resident set: a model per language slot, plus the language detector.
# Everything here stays loaded and nothing else does -- get_model() evicts
# anything not in _active.values() before building a new entry, so a
# GUI-triggered slot swap frees the old model's VRAM the same way boot does.
#
# "detect" rides in the same dict purely to inherit that pinning. It costs
# nothing extra when it duplicates a slot: the set() collapses and _CACHE is
# keyed by path, so one instance serves both roles. It is deliberately not
# evictable -- rebuilding a detector from disk mid-request costs seconds, and
# the VRAM lever is choosing a small detector, not unpinning it.
# set_active_model() only ever writes "fa"/"en", so the wire cannot reach this.
_active = {"fa": FA_MODEL, "en": EN_MODEL, "detect": DETECT_MODEL, "wake": WAKE_MODEL}
_CACHE: dict[str, WhisperModel] = {}


def _build(path: str) -> WhisperModel:
    try:
        return WhisperModel(path, device=DEVICE, compute_type=COMPUTE_TYPE,
                            cpu_threads=CPU_THREADS)
    except Exception as exc:  # no GPU, no cuDNN, VRAM taken by something else
        print(f"[warn] {DEVICE}/{COMPUTE_TYPE} unavailable ({exc}); using CPU int8")
        return WhisperModel(path, device="cpu", compute_type="int8",
                            cpu_threads=CPU_THREADS)


def get_model(path: str) -> WhisperModel:
    if path not in _CACHE:
        pinned = set(_active.values())
        for other in [p for p in _CACHE if p not in pinned]:
            del _CACHE[other]  # CPython drops the refcount, VRAM frees itself
        _CACHE[path] = _build(path)
    return _CACHE[path]


def reload_model(path: str) -> WhisperModel:
    """A model broke mid-use. Drop only that one instance and rebuild it from
    disk -- never a standby spare, we're low on VRAM (requirement 1)."""
    _CACHE.pop(path, None)
    return get_model(path)


def set_active_model(lang: str, name: str) -> str:
    """Point the fa or en slot at a different model, by name from /health's
    fa_models/en_models. Sticky: this is the slot's model for every request
    after this one, not just the one that changed it (a GUI select, not a
    per-clip override)."""
    allowed = FA_MODELS if lang == "fa" else EN_MODELS
    if name not in allowed:
        raise HTTPException(422, "unknown_model")
    _active[lang] = allowed[name]
    return _active[lang]


def resolve_language(value: str) -> str | None:
    """"" keeps whatever .env pinned, "auto" detects per clip, else fa or en."""
    if not value:
        return FORCED_LANGUAGE
    if value == "auto":
        return None
    if value not in LANGUAGES:
        raise HTTPException(422, "unknown_language")
    return value


def closest_language(scores: dict, threshold: float | None = None) -> str:
    """The constraint itself, kept pure so it can be checked without a model.

    Whisper's own detection ranges over ~99 languages and is badly unreliable on
    a two-second clip -- it has returned Dutch for Persian speech. So we take the
    whole probability distribution and read off only these two entries, letting
    the other 97 lose by construction rather than by luck. A third language is
    unrepresentable here even if detection is nonsense.

    The fa/en call is on the share of the fa+en mass rather than on raw P(fa), so
    the threshold means the same thing after a detector swap -- models differ
    wildly in how much mass they leave on the other 97 languages, and a raw cutoff
    would silently change meaning with them. At FA_MIN_PROB=0.5 this is exactly
    argmax, ties to fa; no fa+en mass at all also falls to fa, which keeps a
    Persian-first deployment Persian when detection says nothing useful.
    """
    fa, en = scores.get("fa", 0.0), scores.get("en", 0.0)
    total = fa + en
    cutoff = FA_MIN_PROB if threshold is None else threshold
    return "fa" if not total or fa / total >= cutoff else "en"


async def with_fresh_retry(path: str, op):
    """Run async op(model) against the cached model at path. If it raises, that
    one instance is assumed broken (CUDA fault, driver hiccup, OOM) and gets
    rebuilt from disk once before retrying -- never a standby spare (requirement 1).
    """
    try:
        return await op(get_model(path))
    except Exception as exc:
        print(f"[warn] model at {path!r} failed ({exc}); reloading and retrying once")
        return await op(reload_model(path))


async def do_transcribe(path: str, samples, language: str, terms: str,
                        request: Request, vad_filter: bool = True) -> str:
    async def run(model):
        segments, _ = model.transcribe(
            samples,
            language=language,
            vad_filter=vad_filter,  # trims the dead air around a push-to-talk clip
            beam_size=BEAM_SIZE,
            condition_on_previous_text=False,  # each clip is an independent command
            **({HOTWORDS_PARAM: terms} if terms else {}),
        )
        pieces = []
        for segment in segments:
            # Checked between segments, not just once: a client-side abort (the
            # user reprompting mid-"thinking" -- requirement 3, see the
            # frontend's reprompt()/send()) stops the decode here instead of
            # running it to completion for a response nobody will read.
            if await request.is_disconnected():
                return ""
            pieces.append(segment.text)
        return "".join(pieces).strip()
    return await with_fresh_retry(path, run)


async def pick_model_and_language(fa_path: str, en_path: str, samples,
                                  request: Request) -> tuple[str, str, str, dict]:
    """Route a clip to the fa or en slot, by asking a dedicated detector.

    Two earlier versions of this got it wrong, both in ways worth not repeating:

      1. Raced both slot models (each forced to its own language) on mean
         avg_logprob. A Persian fine-tune transliterating English is *confident*,
         so decode confidence cannot separate "confidently Persian" from
         "confidently wrong".
      2. Called detect_language() on the fa-slot model. Right mechanism, wrong
         model: the fa slot defaults to a full large-v3 fine-tune trained on
         Persian-only data, whose language-ID head now answers fa to nearly
         anything, so English kept routing to Persian.

    Hence DETECT_MODEL: language ID is a separate job from transcription, and the
    model that does it is chosen for being measured in both directions rather
    than for being good at Persian. Only the winning slot then transcribes.
    """
    async def detect(model):
        _, _, probabilities = model.detect_language(samples)
        return dict(probabilities)
    scores = await with_fresh_retry(_active["detect"], detect)

    lang = closest_language(scores)
    fa, en = scores.get("fa", 0.0), scores.get("en", 0.0)
    probs = {"fa": round(fa, 4), "en": round(en, 4)}
    # Logged every time because this decision was guessed at twice for want of
    # exactly these four numbers. Cheap, and the only record of *why* a clip
    # routed the way it did.
    print(f"[detect] {Path(_active['detect']).name} P(fa)={fa:.4f} P(en)={en:.4f} "
          f"fa_share={fa / (fa + en):.3f} (>= {FA_MIN_PROB} is fa) -> {lang}"
          if fa + en else
          f"[detect] {Path(_active['detect']).name} no fa/en mass at all -> {lang}")

    path = fa_path if lang == "fa" else en_path
    terms = " ".join(HOTWORDS.get(lang, []))
    text = await do_transcribe(path, samples, lang, terms, request)
    return path, lang, text, probs


# ---------------------------------------------------------------------------
# Wake word. "hey PUM" / "سلام پام" / a bare "PUM" or "پام".
#
# The browser's energy VAD decides a clip contains speech; WAKE_MODEL (a tiny,
# isolated model -- see above) transcribes it cheaply just to check for the
# name, and this reads the name off the front. Whisper has never heard of PUM,
# so it comes back spelled a dozen ways -- hence a list of the spellings
# actually seen plus difflib for the rest, rather than an equality test that
# would never fire.
# ---------------------------------------------------------------------------
WAKE_LEAD = ("hey", "hi", "ok", "okay", "hey,", "سلام", "های", "هی")
WAKE_NAME = ("pum", "pam", "pom", "poom", "pumm", "pom,", "bum", "bam", "boom",
             "پام", "پم", "پوم", "پان", "بام", "بم", "بوم")
WAKE_CUTOFF = 0.75
# Words. Past this it is a sentence that happens to rhyme, not someone calling.
WAKE_WINDOW = 3
# Hotword-biases the tiny wake probe toward its own vocabulary -- it has
# nothing to do with config/hotwords.json (app vocabulary for the real
# pipeline), it's just nudging decode toward the name it's listening for.
WAKE_TERMS = " ".join(WAKE_LEAD + WAKE_NAME)


def wake_command(text: str) -> str | None:
    """Split a woken utterance into whatever was said after the name.

    None -- the name is not in the opening words; the clip was not addressed to
            us and is discarded without being shown.
    ""   -- woken, but nothing followed it, so the *next* clip is the command.
    """
    raw = normalize(text).split()
    words = [w.strip(".,!?;:؟،…\"'«»").lower() for w in raw]
    for i, word in enumerate(words[:WAKE_WINDOW]):
        if word in WAKE_LEAD:
            continue
        if word in WAKE_NAME or get_close_matches(word, WAKE_NAME, n=1, cutoff=WAKE_CUTOFF):
            return " ".join(raw[i + 1:])
        return None  # first real word is not the name
    return None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_model(_active["fa"])      # a broken model dies at boot, not on someone's first request
    get_model(_active["en"])      # both slots load together, see _active above
    get_model(_active["detect"])  # free when it duplicates a slot -- same _CACHE entry
    get_model(_active["wake"])    # the always-on wake-word probe, resident like the rest
    yield


app = FastAPI(title="Voice-to-text demo", lifespan=lifespan)


@app.post("/transcribe")
async def transcribe(request: Request, audio: UploadFile = File(...),
                     fa_model: str = Form(""),
                     en_model: str = Form(""),
                     language: str = Form(""),
                     wake: bool = Form(False)):
    """wake=true means this clip is still armed, listening for the wake word
    -- not "wake mode is on" in general (the frontend only ever sends it as
    the current wasArmed). Once actually woken, the follow-up command clip
    is sent with wake=false and falls straight into the ordinary path below.
    """
    raw = await audio.read()
    if len(raw) < MIN_CLIP_BYTES:
        raise HTTPException(422, "no_speech")

    forced = resolve_language(language)
    probs = None
    started = time.perf_counter()
    if SAVE_CLIPS:
        save_clip(raw, audio.filename)
    try:
        # A slot pick from the GUI is sticky -- see set_active_model() -- so an
        # unchanged select (the common case) is a no-op dict lookup, not a reload.
        fa_path = set_active_model("fa", fa_model) if fa_model else _active["fa"]
        en_path = set_active_model("en", en_model) if en_model else _active["en"]

        # Decode once. detect_language(), transcribe() and the wake probe all
        # want the same waveform. PyAV (bundled with faster-whisper) needs no
        # ffmpeg step.
        samples = decode_audio(io.BytesIO(raw))

        if wake:
            # Cheap, isolated wake-word check first -- see WAKE_MODEL above.
            # Most VAD-triggered clips here are ambient noise or not addressed
            # to us; this is what keeps the big fa/en pipeline from running on
            # every one of them.
            # vad_filter=False here on purpose: the browser's own VAD already
            # cut this clip, and Silero re-trimming a ~1s "hey PUM" has nothing
            # left to keep -- an empty probe reads as "not woken" and the clip
            # is dropped in silence, which is indistinguishable from a mishear.
            rough = await do_transcribe(_active["wake"], samples, None, WAKE_TERMS,
                                        request, vad_filter=False)
            command = wake_command(rough)
            # Logged every time for the same reason the [detect] line exists:
            # a wake that never fires is otherwise invisible, and guessing at
            # why is what cost this directory two shipped bugs already.
            print(f"[wake] {Path(_active['wake']).name} heard {rough!r} -> "
                  f"{'not woken' if command is None else 'woken' + (f', command {command!r}' if command else ', no command yet')}")
            if not command:
                # None -> not woken. "" -> woken, nothing followed in this
                # clip yet (the client records a follow-up for the command).
                # Either way the probe's own text is never shown or acted on.
                return {
                    "text": "", "woken": command is not None,
                    "detected_language": None,
                    "processing_time_ms": round((time.perf_counter() - started) * 1000),
                    "model": Path(_active["wake"]).name,
                }
            # Said in one breath (name + command, no pause): the probe's text
            # is too rough to act on, so this clip falls through to the real
            # pipeline below for an accurate transcript, exactly like any
            # other request -- wake_command() runs again on that text further
            # down to strip the name back off.

        if forced:
            path = fa_path if forced == "fa" else en_path
            detected = forced
            terms = " ".join(HOTWORDS.get(detected, []))
            text = await do_transcribe(path, samples, detected, terms, request)
        else:
            # The detector decides, then the matching slot transcribes -- see
            # pick_model_and_language() for the two ways this was got wrong before.
            path, detected, text, probs = await pick_model_and_language(
                fa_path, en_path, samples, request)

        if detected == "fa":
            # Persian only. Whisper spells Persian homophones wrong in a closed,
            # predictable way (ز/ذ/ض/ظ are one sound); English has no equivalent
            # failure and is returned exactly as decoded.
            text = spell_correct(text)
    except HTTPException:
        raise  # set_active_model()/resolve_language() reject bad input on purpose
    except Exception as exc:
        print(f"[error] transcribe failed: {exc}")
        raise HTTPException(500, "transcribe_failed")

    answer = {
        "detected_language": detected,
        "processing_time_ms": round((time.perf_counter() - started) * 1000),
        "model": Path(path).name,
    }
    if probs is not None:
        # Auto-detect only: a forced language never asked the detector anything.
        # Not shown in the UI (no jargon on screen) -- this is for the devtools
        # Network tab and for choosing FA_MIN_PROB from real numbers.
        answer |= {"language_probs": probs, "detector": Path(_active["detect"]).name}

    if wake:
        # Always-on mode: a clip without the name is a normal, expected outcome,
        # so it is a 200 with woken=false rather than an error the UI must dodge.
        command = wake_command(text)
        nav, push_error = await push_external(command) if command else (None, None)
        result = {"text": command or "", "woken": command is not None, **answer}
        if nav is not None:
            result |= {"nav_status": nav.get("status"), "nav_url": nav.get("url"),
                      "nav_message": nav.get("message")}
        if push_error is not None:
            result["external_error"] = push_error
        return result

    if not text:
        raise HTTPException(422, "no_speech")
    nav, push_error = await push_external(text)
    if nav is not None:
        answer |= {"nav_status": nav.get("status"), "nav_url": nav.get("url"),
                  "nav_message": nav.get("message")}
    if push_error is not None:
        answer["external_error"] = push_error
    return {"text": text, **answer}


@app.get("/health")
def health():
    return {"models": {"fa": Path(_active["fa"]).name, "en": Path(_active["en"]).name},
            "fa_models": sorted(FA_MODELS), "en_models": sorted(EN_MODELS),
            "detector": Path(_active["detect"]).name, "wake_model": Path(_active["wake"]).name,
            "fa_min_prob": FA_MIN_PROB,
            "device": DEVICE, "compute_type": COMPUTE_TYPE,
            "cpu_threads": CPU_THREADS, "beam_size": BEAM_SIZE,
            "language": FORCED_LANGUAGE or "auto",
            "languages": list(LANGUAGES),
            "spell_correction": {"enabled": SPELL_ENABLED, "threshold": SPELL_THRESHOLD}}


# Mounted last so it does not shadow the routes above.
app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
