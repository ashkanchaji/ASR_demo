# ASR — voice-to-text demo (Persian + English)

Phase 1 of a voice-navigation assistant: prove the capture → transcribe → display loop is
accurate and fast enough. **It does not act on what you say** — it only shows the text.
Push the button, talk, stop talking (or turn on **Always listening** and just say the wake
word). Recording ends itself after a short silence and the transcript, the language and the
transcription time appear below the button.

The service recognizes **Persian or English and nothing else**, by construction — see
*Language handling* below. It keeps **four small models resident at once**, not one:

| slot | role | default |
| --- | --- | --- |
| `fa` | transcribes Persian | `MohammadKhosravi/whisper-large-v3-Persian` (CT2 int8) |
| `en` | transcribes English | `Systran/faster-whisper-small.en` |
| detect | decides fa vs en per clip, never transcribes | `openai/whisper-large-v3-turbo` (CT2 int8) |
| wake | cheap "was the wake word said" probe | `Systran/faster-whisper-tiny` |

The `fa`/`en` slots are swappable at runtime from the page (see *Model choice*); the
detector and wake probe are fixed at boot only. *Language handling* below explains why
detection needs its own model instead of reusing the `fa` slot.

---

## Setup

### 1. Install the service

```bash
uv venv --python 3.12 .venv
VIRTUAL_ENV=$PWD/.venv uv pip install -r backend/requirements.txt
cp .env.example .env
```

The system Python here is 3.14, which has no `ctranslate2` wheels — 3.12 is not optional.
Add `-r backend/requirements-local-cuda.txt` only if you want to run on this machine's GPU;
production is CPU-only and the Docker image needs none of it.

### 2. Build the four models (once)

The service will not start without all four model directories already on disk — `fa`, `en`,
detector and wake probe — and it never downloads anything at runtime (`main.py` calls
`sys.exit()` at import if one is missing). Build the ones that back the defaults:

```bash
uv venv --python 3.12 .venv-convert
VIRTUAL_ENV=$PWD/.venv-convert uv pip install -r requirements-convert.txt

./scripts/convert_khosravi_large.sh   # fa slot  -> models/whisper-large-v3-persian-khosravi-ct2
./scripts/fetch_model.sh small.en     # en slot  -> models/faster-whisper-small.en
./scripts/convert_large_turbo.sh      # detector -> models/whisper-large-v3-turbo-ct2
./scripts/fetch_model.sh tiny         # wake     -> models/faster-whisper-tiny
```

[steja]: https://huggingface.co/steja/whisper-small-persian

`convert_khosravi_large.sh` and `convert_large_turbo.sh` pin a revision inside the script —
neither source repo has release tags, so tracking `main` would silently change what ships.
Conversion needs `torch` and `transformers`, which is why they live in their own venv and
their own requirements file — the running service never imports either.

`fetch_model.sh` pulls **official pre-converted CTranslate2 releases published by the
faster-whisper maintainers** (`Systran/faster-whisper-*`), so they need no conversion.

Two more scripts exist for swapping the `fa` slot at runtime, not required to boot:

```bash
./scripts/convert_model.sh        # alternate fa/en pick: steja/whisper-small-persian
./scripts/fetch_amirmohseni.sh    # comparison only -- see CLAUDE.md before trusting it
```

Do **not** run a `Systran/faster-whisper-*` size through the converter (it's already
converted), and do not substitute a third-party mirror for any of these — see CLAUDE.md for
why each specific model was picked.

## Run

```bash
./run.sh                 # http://127.0.0.1:8000
./run.sh --selfcheck     # end-to-end smoke check
```

FastAPI serves the page and the API from one port, so there is no second server and no CORS.
The model loads at startup, not on the first request, so a broken setup fails immediately.

## Configuration

Settings come from `.env` (loaded by `python-dotenv`) or the shell environment; the shell
wins. Every setting is documented in [.env.example](.env.example). `.env` only sets the
**boot** pick for each slot — swapping a model from the page afterwards is a runtime change
that does not touch `.env` and does not survive a restart.

| Variable | Default | Notes |
| --- | --- | --- |
| `WHISPER_FA_MODEL` | khosravi large-v3 fa | Boot pick for the `fa` slot |
| `WHISPER_EN_MODEL` | `faster-whisper-small.en` | Boot pick for the `en` slot |
| `WHISPER_DETECT_MODEL` | `whisper-large-v3-turbo-ct2` | Language-ID only, never transcribes. Must be a stock, multilingual model — see *Language handling* |
| `WHISPER_WAKE_MODEL` | `faster-whisper-tiny` | Wake-word probe. Must not be an `.en`-suffixed model |
| `WHISPER_FA_MIN_PROB` | `0.5` | fa-share-of-(fa+en) cutoff for auto language routing |
| `WHISPER_DEVICE` | `cpu` | `cuda` for local GPU runs; falls back to CPU with a warning |
| `WHISPER_COMPUTE` | `int8` on CPU, `float16` on CUDA | Follows `WHISPER_DEVICE` when unset |
| `WHISPER_CPU_THREADS` | `0` | `0` = every core. Set it on a shared box |
| `WHISPER_BEAM_SIZE` | `5` | `1` is greedy decoding: cheaper on CPU, see *Model choice* |
| `WHISPER_SAVE_CLIPS` | `false` | Debug only: dumps every uploaded clip to `eval/samples/`, no rotation |
| `SPELL_CORRECTION_ENABLED` | `true` | Persian spell correction. Normalization runs either way |
| `SPELL_CORRECTION_THRESHOLD` | `0.85` | Phonetic similarity needed before a word is rewritten |
| `SPELL_CORRECTION_LEXICON` | unset | Optional path to extend the hotwords vocabulary |
| `PORT` | `8000` | |

## Persian spell correction

Whisper's Persian mistakes are not random noise. Persian inherited several Arabic letters that
merged into a single sound — ز ذ ض ظ are all /z/, س ص ث are all /s/, ه and ح are both /h/ — so
the model has four ways to spell a syllable and only one of them is the word we want. It
returns «گذارش» for «گزارش», «سفه» for «صفحه». The confusion is closed and enumerable, which is
why [backend/spellcheck.py](backend/spellcheck.py) is a lookup table rather than a language model.

It runs on Persian output only, after decoding, in two tiers:

1. **Normalization**, always. Arabic ي/ك rewritten to Persian ی/ک, diacritics dropped,
   whitespace collapsed, half-spaces tidied. Correct regardless of tier 2, so
   `SPELL_CORRECTION_ENABLED=false` does not disable it.
2. **Phonetic-bucket correction.** Every word is collapsed onto the sounds it is made of, and
   matched against `config/hotwords.json` — the same file that biases decoding, not a copy.
   After bucketing, «گذارش» and «گزارش» are the *identical string*, so the common error is an
   exact dictionary hit that never goes near a similarity threshold. Fuzzy matching
   (`difflib`, stdlib) only handles the rarer case where a letter was also dropped or added.

The bucket table is six lines at the top of `spellcheck.py`, one per sound, and is the intended
edit point. پ/ب, گ/ک, چ/ج and ژ/ز are deliberately *not* merged: those are genuinely distinct
phonemes, and merging them would invent false positives rather than fix real errors.

Four guards keep correct words intact: an exact vocabulary hit skips fuzzy matching entirely;
a phonetic form owned by two vocabulary words is declined rather than guessed; tokens with no
Persian-script character are skipped, so `dashboard` inside a Persian sentence survives; and a
token that already contains the vocabulary word spelled correctly is treated as an inflected
form, not a misspelling — «راهنمای» and «کاربری» score above any usable threshold, and only
that rule saves them.

```bash
.venv/bin/python scripts/evaluate_model.py --spell                  # fix rate and false positives
.venv/bin/python scripts/evaluate_model.py --spell --threshold 0.9  # sweep it
```

Measured on the 24 cases in `eval/spell_cases.json` — 16 injected homophone misspellings across
the app's own vocabulary, plus clean Persian sentences that must survive untouched:

| threshold | fixed | false positives | Persian WER |
| --- | --- | --- | --- |
| off | 0 / 16 | 0 / 86 | 15.7% |
| 0.70 | 16 / 16 | 3 / 86 | |
| 0.75 | 16 / 16 | 2 / 86 | |
| **0.80–0.85** | **16 / 16** | **0 / 86** | **0.0%** |
| 0.90–0.95 | 15 / 16 | 0 / 86 | |

0.85 sits at the top of a flat plateau: 0.90 loses «سفه» → «صفحه» (it scores 0.857), and 0.75
and below starts rewriting correct words. Re-run the sweep after editing the hotwords — a
larger vocabulary means more phonetic neighbours and a higher false-positive floor.

**These are synthetic misspellings, not recorded audio.** They are the error pattern observed
in real transcripts, injected into text, which measures the correction layer honestly but says
nothing about how often Whisper actually produces them. The audio harness reports Persian WER
with correction on and off as soon as clips exist in `eval/samples/`.

Not built: the SymSpell + KenLM general-language tier, and any seq2seq corrector. Vocabulary
matching covers the technical terms this app navigates by, which is the priority; revisit only
if coverage beyond that vocabulary turns out to matter.

## Language handling

Whisper's own detection ranges over ~99 languages and is unreliable on a two-second clip —
it has returned Dutch and Arabic for Persian speech. The fix is not a better threshold, it is
a smaller set:

1. `detect_language()` returns the full probability distribution.
2. We read off **only** the `fa` and `en` entries and take the larger.
3. That language is passed explicitly as `language=` to `transcribe()` — never left on auto.

`pick_language()` returns `max(("fa", "en"), key=...)`, so a third language is
*unrepresentable*, not merely unlikely, even if detection is complete nonsense. The
self-check asserts this against a detector that scores Dutch at 0.62.

Closing the set was necessary but not sufficient: it stops Dutch winning, it does not stop
English winning on Persian speech, which a stock `small` does on most short Persian clips.
The Persian fine-tune is what fixed the remaining half — see *Known limitations*.

Mixed utterances ("«فایل رو `commit` کن»") are fine: the clip is transcribed under whichever
of the two scores higher, and Whisper still emits the embedded technical term.
`condition_on_previous_text=False` keeps one clip from dragging context into the next.

## API

`POST /transcribe`, multipart field `audio` (webm/opus straight from MediaRecorder), plus
four optional form fields:

| Field | Values | Meaning |
| --- | --- | --- |
| `fa_model` | a name from `/health`'s `fa_models` | Swaps the `fa` slot, **sticky** for every request after this one. Empty = leave it |
| `en_model` | a name from `/health`'s `en_models` | Same, for the `en` slot |
| `language` | `auto`, `fa`, `en` | Empty/`auto` runs the detector; `fa`/`en` pins it and skips detection |
| `wake` | `true` / `false` | Run the cheap wake-word probe first; only fall through to fa/en on a match |

```json
{ "text": "صفحه تنظیمات رو نشون بده", "detected_language": "fa",
  "processing_time_ms": 1240, "model": "whisper-large-v3-persian-khosravi-ct2",
  "language_probs": {"fa": 0.9996, "en": 0.0001}, "detector": "whisper-large-v3-turbo-ct2" }
```

`language_probs`/`detector` only appear on the auto-detect path — a forced `language` never
asked the detector anything. `detected_language` is always `fa` or `en`. `422 no_speech` when
the clip is empty or Whisper returns nothing; `422 unknown_model` / `422 unknown_language` when
a form field is outside the allowed set; `500 transcribe_failed` when decoding fails twice in a
row (a single failure retries once against a freshly loaded model). The UI turns all of them
into plain-language sentences — no codes on screen.

`fa_model`/`en_model` are **names**, never paths: each is checked against that slot's own
allow-list of converted directories, so nothing the client sends is ever opened directly.
`GET /health` reports the active model per slot, the detector, the wake model, and the
per-slot lists the UI may switch to (`fa_models`, `en_models`).

## Wake word

With **Always listening** on, the browser stops pushing a button and starts looping: its energy
VAD cuts each burst of speech into a clip and sends it with `wake=true`. The server runs the
tiny wake probe (`WHISPER_WAKE_MODEL`, `faster-whisper-tiny` by default) first — a cheap,
isolated model that is never shown or acted on, it only answers "was the name said". Only a
clip where it finds the name falls through to the real fa/en pipeline.

Recognized: `hey PUM`, `PUM`, `سلام پام`, `پام` — Whisper has never heard the name, so
`WAKE_NAME` in `backend/main.py` lists the spellings it actually returns (`pam`, `pom`, `بام`,
`پوم`, …) and `difflib` covers the rest at a 0.75 cutoff. A clip without the name comes back
`200` with `"woken": false` and never reaches the screen; a clip with the name and nothing else
also short-circuits with `"woken": true` and empty `text`, and the client records a follow-up
clip for the command.

Say `hey PUM open settings` and the command in the same clip is the transcript, run back
through `wake_command()` a second time to strip the name off the accurate transcript. Say
`hey PUM` and stop, and the *next* clip is taken as the command without repeating the name.

The probe adds one small extra pass per clip so that ambient noise and false triggers don't
cost a full fa/en decode — see CLAUDE.md for why running detect+transcribe on every VAD trigger
was tried first and dropped.

## Model choice

The table below is the original measurement that picked `steja/whisper-small-persian` as the
first `fa`-slot default; the slot has since moved to the larger, more accurate khosravi
fine-tune (see *Known limitations* for why, and its known weaker vetting). It's kept here
because the *shape* of the comparison — fine-tune vs. stock, and the wrong-language column in
particular — is what to re-run whenever a new `fa` candidate is considered, via
`scripts/evaluate_model.py`. Measured on 13 clips — 6 Persian (Common Voice 17 `fa` test) and 7 English (LibriSpeech
dev-clean plus the JFK clip from `openai/whisper`'s own tests) — all re-encoded to webm/opus
16 kHz mono, which is what the browser sends. CPU int8 on a 16-core Ryzen 7 6800H, decoded
through the service's own path: closed fa/en set, `condition_on_previous_text=False`, VAD on.

| model | beam | WER fa | WER en | mean s | max s | wrong language | peak RSS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **whisper-small-persian-ct2** | **5** | **25.0%** | 8.7% | **2.48** | 4.75 | **0 / 13** | **712 MB** |
| whisper-small-persian-ct2 | 1 | 33.3% | 8.7% | 2.29 | 4.05 | 0 / 13 | 705 MB |
| faster-whisper-small | 5 | 116.7% | 7.7% | 2.27 | 4.27 | 4 / 13 | 1303 MB |
| faster-whisper-small | 1 | 116.7% | 7.7% | 2.22 | 3.71 | 4 / 13 | 1303 MB |
| faster-whisper-medium | 5 | 75.0% | 9.2% | 6.92 | 13.65 | 3 / 13 | 3312 MB |
| faster-whisper-medium | 1 | 83.3% | 6.6% | 7.27 | 11.76 | 3 / 13 | 3312 MB |

The fine-tune wins on Persian by a wide margin, matches the stock models on English to within
about a point, runs at `small` speed, and is the only one of the three that never picked the
wrong language. Beam 5 buys 8 points of Persian WER for 0.19 s, so it stays the default.
`medium` is both slower and worse here and is not a candidate.

**English is not degraded in any way that matters.** 8.7% against `small`'s 7.7% on
LibriSpeech, and the errors are the same proper-noun and homophone slips the stock model
makes ("Layton" for "Leighton", "idols" for "idylls"). It never emitted Persian script for
English audio. Even so, the current design keeps a dedicated English-only model in the `en`
slot (`faster-whisper-small.en`) rather than leaning on the Persian fine-tune for both
languages — see the dual-model bullets in CLAUDE.md for why a Persian-biased model transcribing
English, even accurately here, is not a bet worth making across every future `fa` pick.

**The 116.7% Persian WER for stock `small` is not a typo.** WER counts insertions, so a model
that hears a four-word Persian clip, decides it is English and emits a longer English sentence
can score above 100%.

### What these numbers do not cover

- **13 clips is a sanity check, not an evaluation.** The Persian side is 6 clips of 3–5 words,
  so a single word error moves WER by about four points.
- **The Persian clips favour the fine-tune.** It was trained on Common Voice 11 `fa`; these are
  Common Voice 17 `fa` test — a different release and an unseen split, but the same corpus,
  the same recording conditions and the same read-prompt style. Expect a smaller margin on
  spontaneous command speech.
- **The English clips are read audiobook prose, not commands.** Short imperative utterances are
  the actual workload and are not represented here.
- The Common Voice references use colloquial spellings (`مهمونداری`) where every model returns
  the formal one (`مهمانداری`). That inflates Persian WER equally for all three.

Re-run it with your own clips before trusting the margin:

```bash
.venv/bin/python scripts/evaluate_model.py                       # shipped model, beam 1 and 5
.venv/bin/python scripts/evaluate_model.py --models \
    models/whisper-small-persian-ct2 models/faster-whisper-small --beams 5 --threads 8
```

**It needs clips, and this repository ships none.** Record them with the demo's own
push-to-talk button (webm/opus is what the frontend really sends), drop them in
`eval/samples/`, and write `eval/reference.json`:

```json
{"clip-01.webm": {"language": "fa", "text": "برو به داشبورد"},
 "clip-02.webm": {"language": "en", "text": "open the dashboard"}}
```

Cover the short-utterance case deliberately — one- and two-word commands are where language
detection failed before, and short clips are most of what push-to-talk produces.

## Docker

```bash
docker build -t voice-demo .
docker run --rm -p 8000:8000 voice-demo
```

`python:3.12-slim`, no CUDA base image and no cuDNN pinning: ctranslate2's CPU backend needs
nothing outside its own wheel, and production has no GPU.

Two stages. The first installs `requirements-convert.txt` and runs a conversion script — the
same one a developer runs on the host, so the pinned revision cannot drift between a local
model and the image's. The second copies only the converted directory across — `torch` and
`transformers` are about 1 GB of build-time tooling and have no business in the stage that
serves requests.

The weights are baked in, so the container makes **no network calls** at startup or during a
request; `HF_HUB_OFFLINE=1` turns any attempt into a loud failure rather than a silent stall.

> **The Dockerfile currently bakes in only the original single-model setup**
> (`whisper-small-persian-ct2` via `convert_model.sh`, `WHISPER_MODEL` env var) and has not
> been updated for the four-slot fa/en/detect/wake architecture described above. Building it
> as-is will not boot `backend/main.py`, which now requires `WHISPER_FA_MODEL`,
> `WHISPER_EN_MODEL`, `WHISPER_DETECT_MODEL` and `WHISPER_WAKE_MODEL` to each already exist on
> disk. Until it's updated, run locally with `./run.sh` (*Run* above) instead of Docker, or
> extend the `convert`/copy stages to bake in all four models before relying on the image.

```bash
docker run --rm voice-demo python3 -c "import torch"   # must fail, once the image builds
```

---

## Known limitations

**Every clip costs about 2.5 s on CPU, almost none of it decoding.** Mean 2.48 s per clip at
beam 5 on 16 cores, worst case 4.75 s on a 29 s clip. Whisper's encoder always runs over a
fixed 30 s window, and the service runs it **twice** — once for `detect_language()`, once to
transcribe — so a one-word command costs nearly what a ten-second sentence does. That fixed
cost, not the decoding, is what decides whether push-to-talk feels immediate. On a smaller
box it will be worse: this is 16 cores, and CPU throughput here tracks core count closely.

Two levers, if 2.5 s is too slow: pinning `language=fa` per request skips the detection pass
and removes roughly half the fixed cost, at the price of transliterating embedded English
terms into Persian script. `WHISPER_BEAM_SIZE=1` saves about 0.19 s and costs 8 points of
Persian WER — a poor trade at this workload.

**Language detection was the worst failure mode, and the fine-tune is what fixed it.** Forcing
the choice into {fa, en} removed the Dutch-and-Arabic failure entirely, but it never made the
fa-vs-en call reliable on short Persian: stock `small` gets **four of six** Persian clips wrong,
confidently, and then transliterates them into plausible-looking English. `P(fa)` on the same
six clips (2.4–4.6 s each):

| clip | length | stock `small` P(fa) | P(en) | fine-tune P(fa) | P(en) |
| --- | --- | --- | --- | --- | --- |
| fa-00 | 2.7 s | 0.377 | 0.099 | **0.9999** | 0.000 |
| fa-01 | 3.7 s | 0.038 | **0.079** ❌ | **0.9996** | 0.000 |
| fa-02 | 4.6 s | 0.092 | **0.227** ❌ | **1.0000** | 0.000 |
| fa-03 | 2.4 s | 0.0001 | **0.771** ❌ | **0.9901** | 0.001 |
| fa-04 | 4.1 s | 0.862 | 0.020 | **0.9999** | 0.000 |
| fa-05 | 3.8 s | 0.002 | **0.693** ❌ | **0.9995** | 0.000 |

The fine-tune is above 0.99 on every one. This is a bigger practical win than the WER
difference: a wrong language does not degrade the transcript, it replaces it.

> **These numbers are `steja/whisper-small-persian`'s, and they do not transfer to any other
> fine-tune.** When the Persian slot default later moved to
> `MohammadKhosravi/whisper-large-v3-Persian`, this table was silently read as if it still
> applied, and language detection was routed through that model. It does not apply: khosravi is a
> *full* `large-v3` fine-tune on Persian-only Common Voice, so its language-ID head was retrained
> on a 100% `<|fa|>` label distribution and answers `fa` to nearly anything — English came back
> transliterated into Persian script. Language ID is now a separate job done by
> `WHISPER_DETECT_MODEL` (defaulting to the model actually measured here), and any candidate
> detector must be re-measured with `scripts/evaluate_model.py --detect` before it is trusted.

**It moved the prior, though, and that is the residual risk.** English detection is not free
of it: on the JFK clip the fine-tune scores P(fa)=0.239 against P(en)=0.689, where stock
`small` scores P(fa)=0.0001. It still chose English on all seven English clips, but the margin
is far narrower than the stock model's. Heavily accented or noisy English is the case to watch,
and it is not represented in these clips. If a Persian-first deployment can accept it, pinning
`language=fa` per request removes the question and the latency of asking it.

**Persian errors are now word-level, not script-level.** The fine-tune returns fluent Persian
script and gets short content words right; what it misses are colloquial forms and
lower-frequency words — «دو بستر لمتفاوت داریم» for «دو استایل متفاوت دارین». It also
normalizes colloquial speech to formal spelling («مهمانداری» for «مهمونداری»), which is
arguably correct and still scores as an error against a verbatim reference.

**Client-side stop is an energy threshold, not real VAD.** It measures the room for the first
300 ms and treats 3× that as speech. It stops early on a long mid-sentence pause and late in
a noisy room. The "I'm done" button covers those cases and a hard 15 s cap catches a
threshold that never trips. Silero via `onnxruntime-web` is the upgrade if user testing says
this is the weak link.

**Text direction is the browser's call.** The transcript element is `dir="auto"`, so the
Unicode first-strong rule decides. That is more accurate than counting glyphs — "برو به
dashboard" is a Persian sentence with more Latin characters than Persian ones. It gets the
direction wrong only when a Persian sentence opens with an English word.

**Browser tab must stay visible.** The silence detector runs on `requestAnimationFrame`,
which browsers throttle in background tabs.

**Browser support.** Chrome, Edge and Firefox record Opus; Safari records AAC in MP4, which
the backend also decodes but which is the least tested path here. `getUserMedia` needs
`localhost` or HTTPS — a LAN IP will silently fail.

**No hardening.** No auth, no rate limit, no upload size cap, no persistence. Local demo
only, as scoped.
