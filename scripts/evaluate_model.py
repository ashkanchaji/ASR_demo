"""Compare Whisper models and beam widths on CPU: WER per language, latency per clip.

Runs the same decoding path as the service -- closed fa/en language set,
condition_on_previous_text=False, VAD on -- so the numbers describe what a user
would actually get, not a bare model benchmark.

    .venv/bin/python scripts/evaluate_model.py                  # shipped model, beam 1 and 5
    .venv/bin/python scripts/evaluate_model.py \\
        --models models/whisper-small-persian-ct2 models/faster-whisper-medium --beams 1
    .venv/bin/python scripts/evaluate_model.py --detect \\      # which model should ROUTE
        --models models/whisper-small-persian-ct2 models/whisper-large-v3-turbo-ct2

--detect is the one that picks WHISPER_DETECT_MODEL and WHISPER_FA_MIN_PROB: it
prints P(fa)/P(en) per clip per model, and needs only each clip's language, not
its transcript. Record clips through the demo with WHISPER_SAVE_CLIPS=1.

Needs eval/reference.json next to the clips it names:

    {"clip-01.webm": {"language": "fa", "text": "برو به داشبورد"},
     "clip-02.webm": {"language": "en", "text": "open the dashboard"}}

Clips live in eval/samples/ and are whatever the browser records -- webm/opus is
what the real frontend sends, so evaluating on it keeps the decode path honest.
"""

import argparse
import json
import re
import statistics
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "eval" / "samples"
REFERENCE = ROOT / "eval" / "reference.json"
SPELL_CASES = ROOT / "eval" / "spell_cases.json"

sys.path.insert(0, str(ROOT / "backend"))

from faster_whisper import WhisperModel  # noqa: E402
from faster_whisper.audio import decode_audio  # noqa: E402

import main  # noqa: E402  -- reuses the service's own two-language constraint
import spellcheck  # noqa: E402

# Arabic yeh/kaf for the Persian ones, and the zero-width non-joiner. Real
# transcripts mix all of these; scoring them as errors would measure the
# keyboard the reference was typed on, not the model.
NORMALIZE = str.maketrans({"ي": "ی", "ك": "ک", "‌": " "})


def normalize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).translate(NORMALIZE).lower()
    return re.sub(r"[^\w\s]", " ", text).split()


def word_errors(reference: list[str], hypothesis: list[str]) -> int:
    """Levenshtein distance over words. Exact, and shorter than importing jiwer."""
    previous = list(range(len(hypothesis) + 1))
    for i, want in enumerate(reference, 1):
        current = [i]
        for j, got in enumerate(hypothesis, 1):
            current.append(min(previous[j] + 1,          # deletion
                               current[j - 1] + 1,       # insertion
                               previous[j - 1] + (want != got)))
        previous = current
    return previous[-1]


def detect(model: WhisperModel, samples) -> tuple[dict, str]:
    """The service's own language call, minus the service. Returns the raw
    fa/en probabilities as well, because those -- not the WER column -- are what
    tells you whether a model can be trusted to route a clip at all."""
    _, _, probabilities = model.detect_language(samples)
    scores = dict(probabilities)
    return scores, main.closest_language(scores)


def transcribe(model: WhisperModel, samples, beam_size: int) -> tuple[str, str]:
    _, language = detect(model, samples)
    segments, _ = model.transcribe(
        samples,
        language=language,
        vad_filter=True,
        beam_size=beam_size,
        condition_on_previous_text=False,
    )
    return "".join(s.text for s in segments).strip(), language


def evaluate(path: str, beam_size: int, clips: dict, threads: int) -> list[dict]:
    """One decode per clip, scored with spell correction off and on.

    Correction is a pure text transform, so re-running the model to measure it
    would double the eval's runtime and tell us nothing the second pass of
    scoring does not.
    """
    model = WhisperModel(str(ROOT / path), device="cpu", compute_type="int8",
                         cpu_threads=threads)
    errors = {spell: {"fa": [0, 0], "en": [0, 0]}  # [edits, reference words]
              for spell in (False, True)}
    latencies, wrong_language = [], 0

    for name, expected in sorted(clips.items()):
        audio = decode_audio(str(SAMPLES / name))
        started = time.perf_counter()
        text, language = transcribe(model, audio, beam_size)
        latencies.append(time.perf_counter() - started)

        reference = normalize(expected["text"])
        for spell in (False, True):
            # Same gate as the endpoint: Persian only, English untouched.
            scored = spellcheck.spell_correct(text, enabled=spell) if language == "fa" else text
            errors[spell][expected["language"]][0] += word_errors(reference, normalize(scored))
            errors[spell][expected["language"]][1] += len(reference)
        wrong_language += language != expected["language"]
        corrected = spellcheck.spell_correct(text) if language == "fa" else text
        print(f"  {name:24} {language}  {latencies[-1]:6.2f}s  {text}"
              + (f"\n  {'':24}    corrected  {corrected}" if corrected != text else ""))

    return [{
        "model": Path(path).name,
        "beam_size": beam_size,
        "spell": spell,
        "wer": {lang: (edits / words if words else None)
                for lang, (edits, words) in errors[spell].items()},
        "latency_mean_s": statistics.fmean(latencies),
        "latency_max_s": max(latencies),
        "wrong_language": wrong_language,
    } for spell in (False, True)]


def detect_report(paths: list[str], clips: dict, threads: int) -> None:
    """Per clip, per model: P(fa), P(en), the fa share, and the resulting route.

    This is the only thing in the repo that can actually *choose* a detector or
    a threshold, because it is the only one that puts real speech through a real
    language-ID head. It needs no reference transcripts -- only each clip's true
    language -- so it is usable the moment clips exist. Everything else (a unit
    test stubbing detect_language, a WER table) assumes the answer it is meant
    to be checking; that is how English-routed-to-Persian shipped twice.

    English-only model directories are skipped: faster-whisper refuses
    detect_language() on them, so they cannot be detectors by construction.
    """
    audio = {name: decode_audio(str(SAMPLES / name)) for name in sorted(clips)}
    for path in paths:
        if Path(path).name.endswith(".en"):
            print(f"\n{path}\n  skipped -- English-only, has no detect_language()")
            continue
        model = WhisperModel(str(ROOT / path), device="cpu", compute_type="int8",
                             cpu_threads=threads)
        print(f"\n{path}")
        print(f"  {'clip':24} {'want':>4} {'P(fa)':>8} {'P(en)':>8} {'fa share':>9} "
              f"{'route':>6}")
        wrong = 0
        for name, samples in audio.items():
            scores, route = detect(model, samples)
            fa, en = scores.get("fa", 0.0), scores.get("en", 0.0)
            want = clips[name]["language"]
            wrong += route != want
            print(f"  {name:24} {want:>4} {fa:8.4f} {en:8.4f} "
                  f"{(fa / (fa + en) if fa + en else 1.0):9.3f} "
                  f"{route:>6}{'' if route == want else '   <- WRONG'}")
        print(f"  {wrong}/{len(audio)} misrouted at "
              f"WHISPER_FA_MIN_PROB={main.FA_MIN_PROB}")
        del model


def spell_report(threshold: float | None) -> None:
    """Score the spell-correction layer on text alone -- no clips, no model.

    The audio eval above cannot run until someone records clips, but the
    correction layer is a pure function and its two failure modes are countable
    today. Fixes and false positives are reported as separate numbers on
    purpose: a layer that repairs nine words and breaks three is not "six good".
    """
    cases = json.loads(SPELL_CASES.read_text(encoding="utf-8"))["cases"]
    counts = {}
    for spell in (False, True):
        fixed = missed = broken = kept = 0
        edits = words = 0
        for case in cases:
            got = spellcheck.spell_correct(case["in"], enabled=spell,
                                           threshold=threshold).split()
            was, want = case["in"].split(), case["want"].split()
            for before, after, truth in zip(was, got, want):
                if before != truth:                      # a known misspelling
                    fixed += after == truth
                    missed += after != truth
                else:                                    # was already correct
                    broken += after != truth
                    kept += after == truth
                if before != after:
                    print(f"  {'on' if spell else 'off':>3}  {before} -> {after}"
                          f"{'' if after == truth else f'   WRONG, wanted {truth}'}")
            reference = normalize(case["want"])
            edits += word_errors(reference, normalize(" ".join(got)))
            words += len(reference)
        counts[spell] = (fixed, missed, broken, kept, edits / words)

    print(f"\n{'spell':>6} {'fixed':>7} {'missed':>7} {'false pos':>10} "
          f"{'left ok':>8} {'WER fa':>8}")
    for spell, (fixed, missed, broken, kept, wer) in counts.items():
        print(f"{'on' if spell else 'off':>6} {fixed:>7} {missed:>7} {broken:>10} "
              f"{kept:>8} {wer:>7.1%}")
    known = counts[True][0] + counts[True][1]
    print(f"\n{counts[True][0]}/{known} known misspellings fixed; "
          f"{counts[True][2]} false positives out of {counts[True][2] + counts[True][3]} "
          f"already-correct words; threshold "
          f"{spellcheck.THRESHOLD if threshold is None else threshold}")


def main_() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", help="model directories, relative to the repo",
                        default=["models/whisper-small-persian-ct2"])
    parser.add_argument("--beams", nargs="+", type=int, default=[1, 5])
    parser.add_argument("--threads", type=int, default=0, help="0 = all cores")
    parser.add_argument("--check", action="store_true",
                        help="check the scoring itself, no model needed")
    parser.add_argument("--spell", action="store_true",
                        help="score spell correction on eval/spell_cases.json, no model needed")
    parser.add_argument("--detect", action="store_true",
                        help="language-ID only: P(fa)/P(en) and route per clip per model. "
                             "Needs each clip's language in reference.json, not its transcript")
    parser.add_argument("--threshold", type=float, default=None,
                        help="override SPELL_CORRECTION_THRESHOLD for --spell")
    args = parser.parse_args()

    if args.check:
        assert word_errors(["a", "b", "c"], ["a", "b", "c"]) == 0
        assert word_errors(["a", "b", "c"], ["a", "x", "c"]) == 1   # substitution
        assert word_errors(["a", "b", "c"], ["a", "c"]) == 1        # deletion
        assert word_errors(["a", "c"], ["a", "b", "c"]) == 1        # insertion
        assert word_errors([], ["a"]) == 1 and word_errors(["a"], []) == 1
        # Arabic yeh and kaf must score as the Persian ones, punctuation as nothing.
        assert normalize("برو به داشبورد!") == normalize("برو بة داشبورد".replace("ة", "ه"))
        assert normalize("كي") == normalize("کی")
        assert normalize("Open THE dashboard.") == ["open", "the", "dashboard"]
        print("scoring ok")
        return

    if args.spell:
        spell_report(args.threshold)
        return

    if not REFERENCE.exists():
        sys.exit(f"{REFERENCE} is missing. It maps each clip in {SAMPLES}/ to its\n"
                 'language and true transcript:\n'
                 '    {"clip-01.webm": {"language": "fa", "text": "برو به داشبورد"}}\n'
                 '--detect needs only the "language" key, not "text".\n'
                 "Record the clips with the demo's own push-to-talk button (run it with\n"
                 "WHISPER_SAVE_CLIPS=1 and they land in eval/samples/) so the audio\n"
                 "matches what the service really receives.")

    clips = json.loads(REFERENCE.read_text(encoding="utf-8"))
    missing = [name for name in clips if not (SAMPLES / name).exists()]
    if missing:
        sys.exit(f"reference.json names clips that are not in {SAMPLES}/: {missing}")

    if args.detect:
        detect_report(args.models, clips, args.threads)
        return

    results = []
    for path in args.models:
        for beam in args.beams:
            print(f"\n{path} / beam_size={beam}")
            results.extend(evaluate(path, beam, clips, args.threads))

    print(f"\n{'model':30} {'beam':>4} {'spell':>6} {'WER fa':>8} {'WER en':>8} "
          f"{'mean s':>8} {'max s':>8} {'lang err':>9}")
    for r in results:
        fa, en = r["wer"]["fa"], r["wer"]["en"]
        print(f"{r['model']:30} {r['beam_size']:>4} {'on' if r['spell'] else 'off':>6} "
              f"{'--' if fa is None else format(fa, '7.1%')} "
              f"{'--' if en is None else format(en, '7.1%')} "
              f"{r['latency_mean_s']:8.2f} {r['latency_max_s']:8.2f} "
              f"{r['wrong_language']:>9}")


if __name__ == "__main__":
    main_()
