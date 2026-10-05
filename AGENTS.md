# voice_assistant_demo

Phase 1 of a voice-navigation assistant: a local demo that proves the Persian/English
voice-to-text loop works. Push-to-talk in the browser, faster-whisper on the CPU, transcript
on screen. **No navigation or intent logic — that is phase 2.**

**Two language slots, not one model.** `backend/main.py` keeps a Persian (`fa`) and an English
(`en`) model resident at once — `WHISPER_FA_MODEL`/`WHISPER_EN_MODEL` in `.env` set which model
boots into each slot, and the frontend has a select per slot (backed by `GET /health`) to swap
either one at runtime — a sticky change, not a per-clip override. See the dual-model/pinned-slot
bullets below for why two, and the runtime-switches bullet for how the swap works.

- **fa slot default: `MohammadKhosravi/whisper-large-v3-Persian`**
  (`models/whisper-large-v3-persian-khosravi-ct2`). A full `large-v3` fine-tune on Common Voice 11
  fa only — chosen for accuracy over the older `steja/whisper-small-persian` default. Read the
  vetting caveat below before assuming its numbers generalize past that benchmark: no English
  numbers published, WER 26.87 on Common Voice, card is unreviewed Trainer boilerplate. Still
  large, so this is materially heavier than the old small-persian default — plan VRAM/CPU
  accordingly, especially alongside the always-resident en slot.
- **en slot default: `Systran/faster-whisper-small.en`** (`models/faster-whisper-small.en`).
  Official, pre-converted, English-only by construction. Exists specifically so English speech
  isn't decoded by a Persian-biased model — see the dual-model bullet below.
- `steja/whisper-small-persian` (`models/whisper-small-persian-ct2`): the original fa-slot
  default, still selectable. Cheap enough for CPU — **production is CPU-only** — and keeps its
  English, so it can fill either slot in a pinch.
- `openai/whisper-large-v3-turbo` (`models/whisper-large-v3-turbo-ct2`): bigger, more accurate,
  too slow on CPU for push-to-talk — pair it with `WHISPER_DEVICE=cuda`.
- `AmirMohseni/whisper-large-v3-persian-ct2-int8` (`models/whisper-large-v3-persian-amirmohseni-ct2`):
  comparison-only, not vetted the way the others are — see the bites-you bullet below before
  trusting it for anything.

Do not add a second *inference engine* here — that was tried (a LoRA fine-tune via a
`transformers`/`peft` path) and removed on purpose. A new model directory is a config change;
a new engine is architecture. Re-justify the latter from scratch before resurrecting it.

This directory is self-contained. It sits inside a larger `PUM/` tree that has its own
`.venv`, `AGENTS.md` and `.claude/`; **none of that applies here.** Use this folder's `.venv`
and nothing above it.

## Layout

| Path | What it is |
| --- | --- |
| `backend/main.py` | The whole backend: `POST /transcribe`, `GET /health`, serves `frontend/` |
| `backend/requirements.txt` | Runtime only — no torch, no transformers |
| `backend/requirements-local-cuda.txt` | cuDNN/cuBLAS wheels for optional local GPU runs; **not** used by Docker |
| `backend/spellcheck.py` | Persian-only post-processing: normalization + phonetic-bucket spell correction |
| `backend/selfcheck.py` | Smoke check: closed language set, model load, webm decode, `no_speech` |
| `frontend/index.html` | The whole frontend: markup, CSS and JS in one file, no build step |
| `scripts/convert_model.sh` | One-time: converts the pinned Persian fine-tune (steja) to CT2 int8 |
| `scripts/convert_large_turbo.sh` | One-time: converts `openai/whisper-large-v3-turbo` to CT2 int8 |
| `scripts/convert_khosravi_large.sh` | One-time: converts the Khosravi large-v3 Persian fine-tune to CT2 int8. `WHISPER_FA_MODEL`'s default |
| `scripts/fetch_amirmohseni.sh` | One-time download of the AmirMohseni pre-converted CT2 model. Comparison only |
| `scripts/fetch_model.sh` | One-time download of a stock Systran CT2 size. `small.en` is the pinned `EN_MODEL`, `tiny` is the pinned `WAKE_MODEL`; other sizes are for comparison |
| `requirements-convert.txt` | torch + transformers, build-time only. Never in the image's runtime stage |
| `config/hotwords.json` | `{"fa": [...], "en": [...]}` terms. Biases decoding **and** is the spell-correction vocabulary |
| `eval/spell_cases.json` | Text cases for `evaluate_model.py --spell`: injected misspellings + clean sentences |
| `Dockerfile` | CPU-only. Converter stage, then a runtime stage with the model baked in |
| `.env.example` | Every setting, documented. `.env` is gitignored |

## Running

```bash
./scripts/convert_khosravi_large.sh  # once -- fa-slot default, needs network and .venv-convert
./scripts/fetch_model.sh small.en    # once -- en-slot default, official Systran
./scripts/fetch_model.sh tiny        # once -- wake-word probe default, official Systran
./scripts/convert_model.sh        # once -- small-persian, alternate fa/en pick
./scripts/convert_large_turbo.sh  # once -- large-v3-turbo, alternate pick
./scripts/fetch_amirmohseni.sh       # once -- comparison only, see AGENTS.md
./run.sh                          # http://127.0.0.1:8000
./run.sh --selfcheck              # end-to-end smoke check
docker build -t voice-demo . && docker run --rm -p 8000:8000 voice-demo
```

## Things that will bite you

**The fine-tune is here for language detection at least as much as for WER.** Stock `small`
called four of six short Persian clips English — P(fa)=0.0001 against P(en)=0.77 on one — and
then transliterated them into plausible-looking English. A wrong language does not degrade a
transcript, it replaces it. The fine-tune scores P(fa)>0.99 on all six. Do not "simplify" back
to a stock size on the grounds that WER is close; check the wrong-language column too.

**The fine-tune shifted the detection prior toward Persian, and English detection paid for
it.** P(fa)=0.239 on the JFK clip, where stock `small` gives 0.0001. Still correct, but the
margin is much narrower. Accented or noisy English is the untested case. Pinning `language=fa`
per request removes the question entirely for a Persian-first deployment, and skips an encoder pass.

**Only Persian and English exist.** `pick_language()` takes Whisper's full ~99-language
distribution and reads off just `fa` and `en`, returning `max(("fa","en"), ...)`. A third
language is unrepresentable, not merely unlikely. This is the fix for real observed
failures — Dutch and Arabic coming back for Persian speech on short clips. **Do not
"simplify" it back to `language=None`.** The per-request `language` field is validated against
the same tuple in `resolve_language()`.

**Khosravi is the fa-slot default despite being the weaker-vetted of the two large-v3 Persian
fine-tunes — a deliberate call, not an oversight.** Its WER 26.87 / no-English-numbers / unreviewed
model card are all still true; it was promoted anyway because Persian accuracy mattered more than
the caution that kept it comparison-only before. If it underperforms on real clips, that caveat is
why, and reverting `WHISPER_FA_MODEL` to `steja/whisper-small-persian` is the fallback.
`AmirMohseni/whisper-large-v3-persian-ct2-int8` stays comparison-only — it's three hops removed
from any original fine-tuner (`MohammadGholizadeh`'s fine-tune → `AmirMohseni`'s bf16 rework →
this int8 CT2 conversion), empty README, no WER/CER published anywhere in that chain, the
"third-party mirror, unmaintained re-upload" case the others were chosen to avoid. Do not
default either slot to it without first checking WER on real clips and the wrong-language /
lost-English failure modes documented above.

**The service never touches the network.** `FA_MODEL`/`EN_MODEL` in `main.py` (from
`WHISPER_FA_MODEL`/`WHISPER_EN_MODEL`, defaulting to khosravi and small.en) must each be a
directory that already exists; `main.py` calls `sys.exit()` at import if either is not, rather
than letting faster-whisper download from Hugging Face. That check is the whole reason a bare
model name like `small` no longer works — run the matching `scripts/convert_*.sh` /
`scripts/fetch_*.sh`, or point the env var at a local directory.

**Convert the fine-tune, download the stock sizes.** `steja/whisper-small-persian` is
published as PyTorch weights with no official CTranslate2 release, so it goes through
`ct2-transformers-converter` (`scripts/convert_model.sh`, revision pinned in the script —
the repo has no tags, so "main" would silently change what ships). `openai/whisper-large-v3-turbo`
is the same situation — official, but no pre-converted CT2 release either, so
`scripts/convert_large_turbo.sh` converts it too (revision pinned, same reasoning; no
tokenizer fix-up needed there, unlike the Persian fine-tune, since that repo already ships a
fast tokenizer). The plain small/medium sizes are a different case: `Systran/faster-whisper-*`
are pre-converted by the faster-whisper maintainers, so **do not** run those through the
converter, and do not swap in a third-party mirror (`deepdml/...` and friends) — unmaintained
re-uploads.

**The converter does not always produce a `tokenizer.json`, and faster-whisper needs one.**
`steja/whisper-small-persian` and `MohammadKhosravi/whisper-large-v3-Persian` both ship only
the slow-tokenizer files (`vocab.json` + `merges.txt`). Without a `tokenizer.json` in the model
directory faster-whisper quietly fetches `openai/whisper-tiny`'s from Hugging Face — a runtime
network call, and a hard startup failure under `HF_HUB_OFFLINE=1`. `convert_model.sh` and
`convert_khosravi_large.sh` both build it with `WhisperTokenizerFast.save_pretrained`. Do not
drop that step from either.

**Python 3.12, not the system 3.14.** `ctranslate2` publishes no cp314 wheels.

**cuDNN only matters on bare metal now.** ctranslate2 4.8.2 `dlopen`s `libcudnn.so.9` and
`libcublas.so.12` instead of linking them, so a local `cuda` run needs them on the loader
path — that is all `run.sh`'s `LD_LIBRARY_PATH` line and `requirements-local-cuda.txt` are
for. The image is CPU-only and needs neither; ctranslate2's CPU backend is self-contained in
its wheel.

**Docker is two stages again, and the layer order is load-bearing.** Conversion needs torch
and transformers — about 1 GB of tooling that must not ship in the stage that serves
requests, which is the whole reason the single-stage build did not survive the move to a
fine-tune. The `COPY --from=convert` of the model sits above `COPY backend/ frontend/`, so
ordinary code edits do not re-run the download and conversion.

**No ffmpeg subprocess, deliberately.** faster-whisper's `decode_audio` uses PyAV, which
bundles ffmpeg. The endpoint decodes once and hands the same array to both
`detect_language()` and `transcribe()`.

**Spell correction is Persian-only and vocabulary-bound.** `spell_correct()` runs on the
transcript when the detected language is `fa`, never on English. It collapses each word onto
its sounds using the homophone table at the top of `backend/spellcheck.py` (ز/ذ/ض/ظ, س/ص/ث,
ت/ط, ه/ح, ق/غ, and the ا/آ/ء/ع hamze-alef family) and matches against `config/hotwords.json`.
Two things to know before touching it. **Do not merge پ/ب, گ/ک, چ/ج or ژ/ز into the table** —
they are distinct phonemes and merging them manufactures false positives. And **`_is_derived()`
is not optional**: «راهنمای» and «کاربری» are similar enough to «راهنما» and «کاربر» to beat any
threshold that still catches «سفه» → «صفحه», so the guard against rewriting inflected forms is
structural, not a matter of tuning. `SPELL_CORRECTION_ENABLED=false` disables the correction
only; normalization always runs on Persian output. Measure changes with
`scripts/evaluate_model.py --spell`, and read the false-positive column, not just the fix rate.

**No spell-correction dependency, deliberately.** Hazm would pull nltk/numpy/gensim to do a
`str.maketrans`, and RapidFuzz buys nothing over stdlib `difflib` against a 32-word vocabulary.
Both were considered and skipped; a general Persian lexicon is a pluggable file
(`SPELL_CORRECTION_LEXICON`), not a shipped artifact, because this service never downloads
anything.

**Text direction is `dir="auto"`, not our code.** An earlier version counted Arabic-script
letters against Latin ones and got "برو به dashboard" wrong — a Persian sentence with more
Latin characters than Persian ones. `dir="auto"` is the Unicode first-strong rule, which is
both correct and zero code. Do not reintroduce a script-counting heuristic.

**Hotwords bias decoding, they don't guarantee it.** `config/hotwords.json` terms get folded
into faster-whisper's `hotwords` (or `initial_prompt` on older faster-whisper) per detected
language. Known side effect: on unclear or very short audio the model can hallucinate a
hotword verbatim even when it wasn't said. `vad_filter` is already on for this reason — if a
hotword shows up in a transcript of a clip that didn't contain it, that's this, not a new bug.

**Model choice is per-language-slot, and it's sticky, not per-clip.** `/transcribe` takes
`fa_model`/`en_model`/`language` form fields; the frontend has a `<select>` per language slot,
filled from `/health`'s `fa_models`/`en_models`. `set_active_model(lang, name)` validates `name`
against that slot's allow-list (a bare directory name — a path off the wire is a path traversal)
and, if it differs from the current pick, **changes the slot for every request after this one**,
not just this clip — it's a settings change sent piggybacked on a transcribe call, not a
per-utterance override. `.env`'s `WHISPER_FA_MODEL`/`WHISPER_EN_MODEL` only set the *boot*
picks; nothing in `.env` can be changed without a restart, which is what still keeps a large or
unvetted model from ending up active by accident outside of a deliberate GUI swap.

**Three models are pinned resident, not swapped: the fa slot, the en slot, and the detector.**
`_active` holds all three and `get_model()` only evicts cache entries **not** in
`set(_active.values())`. Swapping a slot (`set_active_model()`) changes what's pinned, which is
what lets ordinary eviction free the old model's VRAM on the next call. `"detect"` rides in the
same dict purely to inherit that pinning, and costs nothing extra when it duplicates a slot — the
`set()` collapses and `_CACHE` is keyed by path, so one instance serves both roles.
`set_active_model()` only ever writes `"fa"`/`"en"`, so the wire cannot repoint the detector.

**Language ID is a separate job from transcription, done by `WHISPER_DETECT_MODEL` — never by
whatever sits in the fa slot.** This is the single most expensive lesson in this directory; it was
got wrong twice, and *both* wrong versions shipped behind a green `selfcheck`.

- **Attempt 1** ran both slot models fully and kept whichever decode had the higher mean
  `avg_logprob`. A Persian fine-tune forced to `language="fa"` on English speech does not fail
  loudly — it transliterates fluently into Persian script ("open my profile" → "اپن مای پروفایل"),
  which `avg_logprob` reads as *confident*, not wrong. Decode confidence cannot separate
  "confidently Persian" from "confidently transliterating English".
- **Attempt 2** moved to `detect_language()` on the fa-slot model. Right mechanism, wrong model.
  The fa slot defaults to `whisper-large-v3-persian-khosravi-ct2`, a **full** `large-v3` fine-tune
  on Persian-only Common Voice — fine-tuning on a 100% `<|fa|>` label distribution retrains the
  language-ID head too, so `P(fa)` is close to unconditional and English kept routing to Persian.

- **Attempt 3** made detection its own slot but defaulted it to `steja/whisper-small-persian`, on
  the strength of README's table. Also wrong, and it is the most instructive failure of the three.

**The detector must be a *stock* model. Every Persian fine-tune here is saturated at `fa`.**
Measured with `scripts/evaluate_model.py --detect` on two real clips recorded on this machine —
one Persian ("صفحه تنظیمات را نشان بده"), one accented English ("show me the settings page") —
as the fa share of the fa+en mass:

| detector | fa clip | en clip | |
| --- | --- | --- | --- |
| `whisper-small-persian-ct2` | 1.000 | **1.000** | P(fa)=0.9996 vs 0.9995 — identical |
| `whisper-large-v3-persian-khosravi-ct2` | 1.000 | **0.881** | |
| `whisper-large-v3-persian-amirmohseni-ct2` | 1.000 | **0.844** | |
| `whisper-large-v3-turbo-ct2` (stock) | 0.930 | 0.000 | the default, for this reason |

The lesson is sharper than "fine-tunes are biased": they do not merely answer `fa` too *often*,
they answer it **unconditionally**, so no `FA_MIN_PROB` threshold can rescue one — steja's two
numbers are identical to four decimal places. Fine-tuning on Persian-only data retrains the
language-ID head along with the decoder. README's detection table belongs to steja and describes
*clean American English* (JFK, P(fa)=0.239); it explicitly flagged accented English as untested,
and accented English is where it collapses to 0.9995. Do not re-derive a detector from that table.

The cost is real: the detector is a third resident model and a large one. Re-measure with
`--detect` before trading it for anything cheaper — the stock *small* sizes are not a safe swap,
since README shows stock `small` getting four of six short Persian clips wrong in the other
direction. A detector whose directory name ends `.en` is rejected at boot: faster-whisper refuses
`detect_language()` on English-only models.

**`closest_language()` decides on the fa *share* of the fa+en mass, not raw `P(fa)`.** Models
differ wildly in how much mass they leave on the other 97 languages, so a raw cutoff would
silently change meaning when the detector is swapped. `WHISPER_FA_MIN_PROB` defaults to `0.5`,
which is exactly argmax with ties to fa — deliberately not higher, because the only measured
English clip normalizes to 0.257 against Persian's >0.99, so anything from 0.3 to 0.9 is identical
on every clip that exists and a fancier default would just bake in a guess. Raise it from the
logged numbers if accented English misroutes. No fa+en mass at all falls to fa, keeping a
Persian-first deployment Persian when detection says nothing useful.

**Every auto-detect request logs `P(fa)`/`P(en)`/share/route, and returns them in the JSON.**
`[detect] …` in the server log, plus `language_probs` and `detector` in the `/transcribe` response
(auto branch only — a forced language never asked the detector anything, so it must not claim to
have). This existed nowhere for the first two attempts, which is precisely why they were guesses.
Not shown in the UI — "no jargon in the UI", and the devtools Network tab already has it.

**A `selfcheck` test cannot validate language routing, and must not be trusted to.** The routing
tests stub `detect_language()` with hardcoded probabilities — the stub *is* the assumption under
test. They prove the router reads the detector, honours the threshold, and never calls the fa
slot for an English clip; they prove nothing about whether a given model gets *your voice* right.
The only thing that can is `scripts/evaluate_model.py --detect` over real clips, which prints
`P(fa)`/`P(en)`/route per clip per model and needs each clip's *language* only, not a transcript.
Record clips by running the demo with `WHISPER_SAVE_CLIPS=1` (off by default, `ponytail:` no
rotation, debug only) — every upload lands in `eval/samples/` as the exact webm/opus the browser
sent. This repo ships no clips, and that gap is the reason this bug shipped twice.

**A model that breaks mid-clip is rebuilt from disk once, never kept as a standby spare.**
`with_fresh_retry()` wraps every `detect_language()`/`transcribe()` call; on any exception it drops
that one cached instance and retries once against a freshly built one — we're too low on VRAM to
keep a second copy of anything just in case. A second consecutive failure propagates as the
ordinary `transcribe_failed` 500.

**A reprompt mid-"thinking" is a client-side abort, not a server-side cancel token.**
`do_transcribe()` checks `await request.is_disconnected()` between segments (not just once) — the
lazy part is that faster-whisper's `segments` is already a generator, decoding one segment per
`next()`, so this is free cooperative cancellation, no threading or cancel-token plumbing needed.
The frontend's `reprompt()` (button-click reprompt, not-VAD) and `send()`'s unconditional
`controller?.abort()` (VAD/wake mode: a fresh clip always supersedes whatever was still being
thought about) are what actually close the connection. In wake mode the next clip starts recording
immediately after capture, not after the previous fetch resolves — `render()` deliberately never
forces `state` back to `"idle"` in that path, since a newer "listening" cycle may already be
running; only the not-VAD branch of `send()` does that.

**The wake word gets its own tiny, isolated model — never the fa/en or detect models.**
`WHISPER_WAKE_MODEL` (default `faster-whisper-tiny`, stock and multilingual for the same reason
as `WHISPER_DETECT_MODEL` — the wake phrase has Persian spellings too) is a fourth resident model
that runs a cheap probe on every armed (`wake=true`) clip before any big model does. This was the
upgrade this file used to say to reach for "if [the single-engine version] stops keeping up" —
it did: running the full detect+transcribe pass on every VAD-triggered clip, including ambient
noise and false triggers, made always-on feel like it needed a beat to "activate" and dropped the
first words of the real command while the heavy pass ran. `wake_command()` still reads the name
off the front of a transcript exactly as before, and `WAKE_NAME`/`difflib` still cover Whisper's
spellings — do not replace that with an equality test, it would never fire. What changed is whose
transcript: the probe's own text is never shown or acted on, it only answers yes/no. `None` from
`wake_command()` (name not found) short-circuits before any big model runs — `200` with
`"woken": false`, never a `422`, the loop must not paint errors on screen for every stray noise in
the room. `""` (woken, nothing followed in this clip) also short-circuits, just as fast — the
client records a follow-up clip for the command. Only when the probe finds a name *and* trailing
speech in the same breath does the clip fall through to the real fa/en pipeline, exactly like any
other request, and `wake_command()` runs a second time on that accurate transcript to strip the
name back off. The probe is never trusted for real content, only for "was the name said".

**Both slot models, the detector, and the wake probe all load at startup, not lazily.** A FastAPI
lifespan hook calls `get_model()` on `_active["fa"]`, `_active["en"]`, `_active["detect"]` and
`_active["wake"]` so a broken image or a missing model dies at boot instead of on someone's first
request.

## Conventions

- Fewest files. The backend is one module, the frontend is one HTML file.
- Comments explain *why*, not what.
- No jargon in the UI. Errors are sentences a non-technical customer can act on; the
  machine-readable codes (`no_speech`, `transcribe_failed`) never reach the screen.
- Deliberate shortcuts get a `ponytail:` comment naming the ceiling and the upgrade path.
- Out of scope for this phase: navigation, accounts, persistence, auth.
