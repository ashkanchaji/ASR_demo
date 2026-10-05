"""End-to-end smoke check.

Covers the parts that can only fail at runtime: that the language choice is
structurally confined to Persian and English, that ctranslate2 finds cuDNN and
loads the converted model, and that a webm/opus clip decodes without an ffmpeg
step. Accuracy is not checked here -- that needs a microphone and a human.

    ./run.sh --selfcheck
"""

import asyncio
import io
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient
from faster_whisper.audio import decode_audio

import main
import spellcheck


def check_language_is_closed() -> None:
    """The constraint must hold whatever the detector says."""
    cases = [
        # Real symptom: Dutch winning outright on Persian speech.
        ({"nl": 0.62, "ar": 0.20, "fa": 0.09, "en": 0.04}, "fa"),
        ({"nl": 0.62, "ar": 0.20, "en": 0.09, "fa": 0.04}, "en"),
        ({"de": 0.99}, "fa"),  # neither present: still cannot answer "de"
        ({}, "fa"),            # empty distribution must not blow up
    ]
    for scores, expected in cases:
        got = main.closest_language(scores)
        assert got in main.LANGUAGES, f"escaped the closed set: {got!r}"
        assert got == expected, f"{scores} -> {got!r}, wanted {expected!r}"

    # No env pin left -- boot always auto-detects.
    assert main.FORCED_LANGUAGE is None


def check_fa_threshold() -> None:
    """The knob that absorbs a detector's Persian prior.

    Decided on the fa share of the fa+en mass, not raw P(fa), so the same
    threshold keeps its meaning when the detector is swapped for one that
    leaves a different amount of mass on the other 97 languages.
    """
    # The default is exactly argmax, ties to fa -- i.e. the old behaviour.
    assert main.FA_MIN_PROB == 0.5, "default must not bake in an unmeasured claim"
    for scores in ({"fa": 0.6, "en": 0.4}, {"fa": 0.5, "en": 0.5}):
        assert main.closest_language(scores, 0.5) == "fa"
    assert main.closest_language({"fa": 0.4, "en": 0.6}, 0.5) == "en"

    # The README's measured numbers, the two that matter. Persian must survive
    # any threshold anyone would plausibly dial in; the one measured English
    # clip must lose at the default and stay losing as the knob is raised.
    persian = {"fa": 0.9901, "en": 0.001}
    english = {"fa": 0.239, "en": 0.689}   # JFK, the narrow-margin case
    for threshold in (0.3, 0.5, 0.9):
        assert main.closest_language(persian, threshold) == "fa", threshold
        assert main.closest_language(english, threshold) == "en", threshold

    # Raising the knob moves accented-English-shaped input off Persian; that is
    # the whole point of it being tunable rather than hardcoded.
    accented = {"fa": 0.55, "en": 0.40}
    assert main.closest_language(accented, 0.5) == "fa"
    assert main.closest_language(accented, 0.8) == "en"


def check_detector_can_actually_detect() -> None:
    """The one check here that is not a stub, and the one that would have caught
    this bug class at all.

    It cannot prove the detector is *right* -- see
    check_detection_routes_to_the_en_slot -- only that the configured model is
    physically capable of language ID. That matters because faster-whisper
    refuses detect_language() outright on an English-only model, which would be
    a 500 on every auto-detect request rather than a bad answer.
    """
    assert main.DETECT_MODEL in main._active.values(), "detector must be pinned resident"
    assert not Path(main.DETECT_MODEL).name.endswith(".en"), \
        "an English-only model has no detect_language()"

    _, _, probabilities = main.get_model(main._active["detect"]).detect_language(
        decode_audio(io.BytesIO(clip())))
    scores = dict(probabilities)
    for code in main.LANGUAGES:
        assert code in scores, f"detector returned no P({code}) at all"
    assert main.closest_language(scores) in main.LANGUAGES


def check_wake_word() -> None:
    """The wake word is read off the front of an ordinary transcript, so the
    only thing that can break is this split. None means the clip was never
    addressed to us; "" means woken with the command still to come."""
    cases = [
        ("hey PUM open settings", "open settings"),
        ("Hey, Pam.", ""),
        ("پام", ""),
        ("سلام پام برو به داشبورد", "برو به داشبورد"),
        ("پوم صفحه تنظیمات", "صفحه تنظیمات"),
        # Not addressed to us: no name in the opening words.
        ("open settings", None),
        ("برو به داشبورد", None),
        ("سلام خوبی", None),
        ("", None),
    ]
    for text, expected in cases:
        got = main.wake_command(text)
        assert got == expected, f"{text!r} -> {got!r}, wanted {expected!r}"


def check_runtime_switches() -> None:
    """fa_model/en_model/language come off the wire now, so the allow-list is a
    trust boundary: a name outside it must be refused, not opened."""
    assert main.MODELS, "no CTranslate2 model directories discovered"
    assert main.FA_MODEL in main.FA_MODELS.values()
    assert main.EN_MODEL in main.EN_MODELS.values()
    # small.en-style dirs are English-only by construction -- must never be
    # offered as a Persian slot pick, see FA_MODELS' filter in main.py.
    assert Path(main.EN_MODEL).name not in main.FA_MODELS, \
        "an English-only model must not be selectable for the fa slot"
    for lang in ("fa", "en"):
        for bad in ("../etc", "/etc/passwd", "small", ""):
            try:
                main.set_active_model(lang, bad)
            except HTTPException as exc:
                assert exc.status_code == 422
            else:
                raise AssertionError(f"set_active_model({lang!r}, {bad!r}) accepted it")
    # A real swap must actually move the slot, then move back cleanly.
    was = main._active["fa"]
    name = next(iter(main.FA_MODELS))
    assert main.set_active_model("fa", name) == main.FA_MODELS[name]
    assert main._active["fa"] == main.FA_MODELS[name]
    main._active["fa"] = was  # leave global state as later checks expect it

    assert main.resolve_language("auto") is None
    assert main.resolve_language("fa") == "fa"
    assert main.resolve_language("") == main.FORCED_LANGUAGE
    try:
        main.resolve_language("nl")
    except HTTPException:
        pass
    else:
        raise AssertionError("resolve_language accepted a third language")


def check_fault_tolerance() -> None:
    """Requirement 1: a model that breaks mid-use gets rebuilt from disk once
    and retried -- not left dead, not doubled up in VRAM as a standby."""
    calls = {"n": 0}

    async def flaky(_model):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("simulated fault")
        return "ok"

    assert asyncio.run(main.with_fresh_retry(main.FA_MODEL, flaky)) == "ok"
    assert calls["n"] == 2, "must retry exactly once, not loop or give up early"


class _FakeSegment:
    def __init__(self, text):
        self.text = text


class _NeverDisconnected:
    async def is_disconnected(self):
        return False


def check_detection_routes_to_the_en_slot() -> None:
    """English must reach the en slot even though the fa slot is the "main" model.

    WHAT THIS CAN PROVE: that the routing reads the *detector*, that a
    below-threshold fa share sends the clip to the en slot, and that the fa
    slot's transcribe() is never even called in that case.

    WHAT IT CANNOT PROVE, AND THIS IS THE POINT: that the detector answers
    correctly on real speech. The probabilities below are stubbed, and the
    stub *is* the assumption under test -- this bug shipped twice behind a
    green test exactly like this one, because the repo has no speech clips.
    Only `scripts/evaluate_model.py --detect` over recorded clips can choose a
    detector or a threshold. Record some with WHISPER_SAVE_CLIPS=1.
    """
    detector = main.get_model(main._active["detect"])
    fa_model, en_model = main.get_model(main.FA_MODEL), main.get_model(main.EN_MODEL)
    # The reported symptom: the fa model would have transliterated English into
    # Persian script. If routing works, this stub is never reached.
    with patch.object(detector, "detect_language",
                      return_value=(None, None, [("fa", 0.1), ("en", 0.9)])), \
         patch.object(fa_model, "transcribe",
                      return_value=([_FakeSegment("اپن مای پروفایل")], None)), \
         patch.object(en_model, "transcribe",
                      return_value=([_FakeSegment("open my profile")], None)):
        path, lang, text, probs = asyncio.run(main.pick_model_and_language(
            main.FA_MODEL, main.EN_MODEL, object(), _NeverDisconnected()))
    assert (path, lang, text) == (main.EN_MODEL, "en", "open my profile"), (path, lang, text)
    assert probs == {"fa": 0.1, "en": 0.9}, probs

    # And the reverse, so this is a router and not a hardcoded "always en".
    with patch.object(detector, "detect_language",
                      return_value=(None, None, [("fa", 0.97), ("en", 0.01)])), \
         patch.object(fa_model, "transcribe",
                      return_value=([_FakeSegment("برو به داشبورد")], None)):
        path, lang, _, _ = asyncio.run(main.pick_model_and_language(
            main.FA_MODEL, main.EN_MODEL, object(), _NeverDisconnected()))
    assert (path, lang) == (main.FA_MODEL, "fa"), (path, lang)


def check_detector_is_not_the_fa_slot_by_accident() -> None:
    """Regression guard for the actual root cause.

    The fa slot defaults to a large-v3 fine-tune trained on Persian-only data,
    whose language-ID head answers fa to nearly anything. Detection must not be
    routed through whatever model happens to occupy that slot, so swapping the
    fa slot must leave the detector where it is.
    """
    was = main._active["fa"]
    detector_was = main._active["detect"]
    other = next((n for n, p in main.FA_MODELS.items() if p != was), None)
    if other:
        main.set_active_model("fa", other)
        assert main._active["detect"] == detector_was, \
            "a fa-slot swap must not move the detector"
        main._active["fa"] = was


def check_transcribe_can_be_aborted() -> None:
    """Requirement 3: a disconnected client stops the decode instead of paying
    for a response nobody will read -- what a mid-"thinking" reprompt relies on."""
    class FakeRequest:
        async def is_disconnected(self):
            return True

    async def check():
        model = main.get_model(main.FA_MODEL)
        with patch.object(model, "transcribe",
                          return_value=([_FakeSegment("should not be collected")], None)):
            text = await main.do_transcribe(main.FA_MODEL, object(), "fa", "", FakeRequest())
        assert text == "", "must bail out instead of collecting segments after disconnect"

    asyncio.run(check())


def check_hotwords_param_picked() -> None:
    """The feature-detected param name must be one faster-whisper's transcribe()
    actually accepts, whichever faster-whisper version is installed."""
    import inspect
    assert main.HOTWORDS_PARAM in inspect.signature(main.WhisperModel.transcribe).parameters


def check_spell_correction_loads() -> None:
    """The vocabulary is read from disk at import, so a bad path or a malformed
    hotwords file must fail here rather than silently correcting nothing."""
    assert spellcheck.HOTWORDS, f"no Persian hotwords loaded from {spellcheck.HOTWORDS_FILE}"
    spellcheck.demo()


def clip() -> bytes:
    """A short webm/opus clip, encoded the way MediaRecorder encodes one."""
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=220:duration=2",
         "-c:a", "libopus", "-b:a", "24k", "-f", "webm", "pipe:1"],
        check=True, stdout=subprocess.PIPE,
    ).stdout


def check_transcripts_stay_local() -> None:
    """Both capture modes return transcripts without an outbound connection."""
    client = TestClient(main.app)
    audio = {"audio": ("clip.webm", b"x" * main.MIN_CLIP_BYTES, "audio/webm")}
    with patch("socket.socket.connect", side_effect=AssertionError("outbound connection")), \
         patch.object(main, "decode_audio", return_value=object()), \
         patch.object(main, "do_transcribe", new_callable=AsyncMock) as transcribe, \
         patch.object(main, "pick_model_and_language", new_callable=AsyncMock) as route:
        for language in ("en", "fa", "auto"):
            for wake in (False, True):
                text = "hey PUM open settings" if wake else "open settings"
                transcribe.return_value = text
                route.return_value = (main.EN_MODEL, "en", text, {"fa": 0.1, "en": 0.9})
                response = client.post("/transcribe", files=audio,
                                       data={"language": language, "wake": str(wake).lower()})
                assert response.status_code == 200, response.text
                body = response.json()
                assert body["text"] == "open settings", body
                assert not any(key.startswith("nav_") or key == "external_error" for key in body), body
                if wake:
                    assert body["woken"] is True, body


def run() -> None:
    check_transcripts_stay_local()
    check_language_is_closed()
    check_wake_word()
    check_runtime_switches()
    check_hotwords_param_picked()
    check_spell_correction_loads()
    check_fa_threshold()
    check_fault_tolerance()
    check_detector_can_actually_detect()
    check_detection_routes_to_the_en_slot()
    check_detector_is_not_the_fa_slot_by_accident()
    check_transcribe_can_be_aborted()

    client = TestClient(main.app)  # the lifespan hook loads the model here
    assert client.get("/health").status_code == 200
    assert client.get("/").status_code == 200, "frontend is not being served"

    # An empty MediaRecorder container must read as speechless, not crash.
    r = client.post("/transcribe", files={"audio": ("clip.webm", b"", "audio/webm")})
    assert r.status_code == 422 and r.json()["detail"] == "no_speech", r.text

    # A real container. A tone is not speech, so 422 is the expected answer; a 500
    # would mean the decode or the model load broke, which is the point of this.
    tone = clip()
    r = client.post("/transcribe", files={"audio": ("clip.webm", tone, "audio/webm")})
    assert r.status_code in (200, 422), r.text
    if r.status_code == 200:
        body = r.json()
        assert body["detected_language"] in main.LANGUAGES
        assert body["detector"] == Path(main._active["detect"]).name, body
        assert set(body["language_probs"]) == set(main.LANGUAGES), body

    # Wake mode: a clip nobody addressed to us is a 200 the UI can ignore, not
    # an error. A tone has no wake word in it by construction.
    r = client.post("/transcribe", files={"audio": ("clip.webm", tone, "audio/webm")},
                    data={"wake": "true"})
    assert r.status_code == 200 and r.json()["woken"] is False, r.text
    # A rejected wake probe exits before language detection runs.
    assert "detector" not in r.json() and "language_probs" not in r.json(), r.text

    # A forced language never consults the detector, so it must NOT claim to have.
    r = client.post("/transcribe", files={"audio": ("clip.webm", tone, "audio/webm")},
                    data={"wake": "true", "language": "fa"})
    assert r.status_code == 200 and "language_probs" not in r.json(), r.text

    # The switchable model lists are allow-lists, over the wire as well as in unit form.
    r = client.post("/transcribe", files={"audio": ("clip.webm", tone, "audio/webm")},
                    data={"fa_model": "../../etc"})
    assert r.status_code == 422 and r.json()["detail"] == "unknown_model", r.text
    r = client.post("/transcribe", files={"audio": ("clip.webm", tone, "audio/webm")},
                    data={"en_model": "/etc/passwd"})
    assert r.status_code == 422 and r.json()["detail"] == "unknown_model", r.text

    print(f"ok -- {main.FA_MODEL} + {main.EN_MODEL} on {main.get_model(main.FA_MODEL).model.device}")


if __name__ == "__main__":
    try:
        run()
    except AssertionError as exc:
        sys.exit(f"FAILED: {exc}")
