"""Persian-only spell correction for the transcript, after decoding.

Whisper's Persian errors are not random: it emits a real word that sounds right
and is spelled wrong -- "گذارش" for "گزارش", "سفه" for "صفحه". Persian inherited
several Arabic letters that merged into one sound (ز ذ ض ظ are all /z/), so the
model has four ways to spell a syllable and only one of them is the word we
want. That is a closed, enumerable confusion, which is why a lookup table beats
a language model here.

Two tiers, both Persian-only -- English output never reaches this module:

  1. normalize()  -- always. Arabic yeh/kaf for the Persian ones, diacritics
     dropped, whitespace collapsed. Runs whether or not tier 2 is enabled.
  2. correction   -- phonetic-bucket match against the app's own vocabulary.
     Off with SPELL_CORRECTION_ENABLED=false.

The bias is heavily toward leaving text alone: a word that is already correct
must never be "corrected" into something else, so an exact vocabulary hit skips
the fuzzy step entirely and an ambiguous match is declined rather than guessed.

    .venv/bin/python backend/spellcheck.py     # self-check
"""

import json
import os
import re
import unicodedata
from difflib import get_close_matches
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# The phonetic bucket table. THIS IS THE EDIT POINT -- one line per sound.
#
# Each string is a group of letters that are pronounced identically in standard
# Iranian Persian; the first letter of the group stands for the whole group in
# a phonetic key. Adding a merger is one character; removing one is the same.
#
# Deliberately NOT merged: پ/ب, گ/ک, چ/ج, ژ/ز. Those are distinct phonemes --
# merging them would invent false positives rather than fix real ones.
# ك (Arabic kaf), ي (Arabic yeh) and ة are not here because normalize() has
# already rewritten them to their Persian equivalents.
# ---------------------------------------------------------------------------
HOMOPHONES = (
    "زذضظ",        # /z/  -- the big one: گزارش vs گذارش
    "سصث",         # /s/  -- صفحه vs سفحه
    "تط",          # /t/
    "هح",          # /h/  -- صفحه vs صفهه
    "قغ",          # /q/  -- merged in Iranian Persian, distinct in Dari
    "اآأإءعئؤ",    # glottal stop / hamze / alef. ع is not a consonant in
                   # Persian, so مسئله / مساله / مسله are one spoken word.
)
BUCKET = {letter: group[0] for group in HOMOPHONES for letter in group}

ZWNJ = "‌"          # half-space: part of a word, not a separator
TATWEEL = "ـ"       # decorative letter-stretching, carries no sound
PERSIAN_BLOCK = ("؀", "ۿ")

# Arabic forms of Persian letters. Whisper emits both; only the Persian ones are
# correct Persian orthography, and unifying them is what makes the vocabulary
# lookups below meet at all.
ARABIC_TO_PERSIAN = str.maketrans({"ي": "ی", "ى": "ی", "ك": "ک", "ة": "ه", "ۀ": "ه"})

HOTWORDS_FILE = os.getenv("WHISPER_HOTWORDS_FILE", "./config/hotwords.json")
# Optional newline-delimited Persian word list. Serves two jobs: words in it are
# known-good and never corrected, and they are correction targets of last resort
# after the hotwords. Absent file is the normal case, not an error -- the
# hotwords are the vocabulary this app actually depends on.
LEXICON_FILE = os.getenv("SPELL_CORRECTION_LEXICON", "./config/persian-words.txt")

ENABLED = os.getenv("SPELL_CORRECTION_ENABLED", "true").lower() not in ("false", "0", "no")
# 0.85, not 0.8, and the two clips either side of it are why: "سفه" -> "صفحه"
# scores 0.857 and must pass; "کاربران" -> "کاربر" scores 0.833 and must not.
THRESHOLD = float(os.getenv("SPELL_CORRECTION_THRESHOLD", "0.85"))

# Below this many letters every Persian word is a near-neighbour of every other
# one -- به, که, را, نه. Exact bucket matches are still allowed at any length,
# because those require an identical phonetic form rather than a close one.
MIN_FUZZY_LENGTH = 3


def normalize(text: str) -> str:
    """Tier 1. Orthographic cleanup that is correct regardless of tier 2.

    NFKC alone is not enough: it leaves ي/ك and the harakat in place. The
    half-space is preserved -- it is a letter-joiner inside words like ثبت‌نام,
    and turning it into a space would split one word into two.
    """
    text = unicodedata.normalize("NFKC", text).translate(ARABIC_TO_PERSIAN)
    text = "".join(c for c in text if not unicodedata.combining(c) and c != TATWEEL)
    text = re.sub(rf"\s*{ZWNJ}\s*", ZWNJ, text)   # a spaced half-space is still one
    return re.sub(r"[^\S\n]+", " ", text).strip()


def phonetic_key(word: str) -> str:
    """Collapse a word onto the sounds it is made of.

    The whole point: گزارش and گذارش produce the same key, so the common error
    is an exact dictionary hit and never has to go near a similarity threshold.
    """
    return "".join(BUCKET.get(c, c) for c in word if c != ZWNJ)


def _load_vocabulary() -> tuple[set, dict, set, dict]:
    try:
        hotwords = set(json.loads(Path(HOTWORDS_FILE).read_text(encoding="utf-8")).get("fa", []))
    except (OSError, ValueError, AttributeError):
        hotwords = set()
    try:
        lexicon = set(Path(LEXICON_FILE).read_text(encoding="utf-8").split())
    except OSError:
        lexicon = set()

    hotwords = {normalize(w) for w in hotwords if w}
    lexicon = {normalize(w) for w in lexicon if w} - hotwords

    def by_key(words: set) -> dict:
        keys: dict = {}
        for word in words:
            keys.setdefault(phonetic_key(word), []).append(word)
        return keys

    return hotwords, by_key(hotwords), lexicon, by_key(lexicon)


HOTWORDS, HOTWORD_KEYS, LEXICON, LEXICON_KEYS = _load_vocabulary()


def _candidate(key: str, keys: dict, cutoff: float) -> str | None:
    """One vocabulary tier's answer for a phonetic key, or None to decline.

    Declines on ambiguity: if two vocabulary words sound the same there is no
    evidence here for choosing between them, and a coin flip is worse than the
    word the model already produced.
    """
    exact = keys.get(key)
    if exact:
        return exact[0] if len(exact) == 1 else None
    if len(key) < MIN_FUZZY_LENGTH:
        return None
    match = get_close_matches(key, keys, n=1, cutoff=cutoff)
    if not match:
        return None
    words = keys[match[0]]
    return words[0] if len(words) == 1 else None


def _is_derived(word: str, candidate: str) -> bool:
    """Is the word an inflected form of the candidate rather than a misspelling?

    راهنمای and کاربری both score above any threshold that still catches سفه ->
    صفحه, so the threshold cannot separate them -- but it does not have to. They
    contain the vocabulary word spelled *correctly*, and a homophone confusion by
    definition misspells it. So a token that carries the candidate intact has
    nothing here to fix, whatever the similarity says. This is the guard that
    keeps plurals and adjectives from being flattened onto the singular noun.
    """
    return candidate in word or candidate.replace(ZWNJ, "") in word.replace(ZWNJ, "")


def _correct_word(word: str, cutoff: float) -> str:
    # Already a word we know. This is the single most important guard: it means
    # a correctly spelled term can never be fuzzed into a different one.
    if word in HOTWORDS or word in LEXICON:
        return word
    key = phonetic_key(word)
    # Hotwords first at every stage -- page and feature names are what this app
    # depends on, so they outrank a generic lexicon hit rather than tying with it.
    match = _candidate(key, HOTWORD_KEYS, cutoff) or _candidate(key, LEXICON_KEYS, cutoff)
    if match is None or _is_derived(word, match):
        return word
    return match


def spell_correct(text: str, enabled: bool | None = None,
                  threshold: float | None = None) -> str:
    """Normalize Persian text, then correct homophone misspellings in it.

    enabled/threshold override the environment so the eval harness can run the
    same text through both settings without reimporting the module.
    """
    text = normalize(text)
    if not (ENABLED if enabled is None else enabled):
        return text
    cutoff = THRESHOLD if threshold is None else threshold

    def replace(match: re.Match) -> str:
        word = match.group()
        # Latin words and bare numbers inside a Persian sentence are left alone;
        # the vocabulary and the bucket table only describe Persian script.
        if not any(PERSIAN_BLOCK[0] <= c <= PERSIAN_BLOCK[1] for c in word):
            return word
        return _correct_word(word, cutoff)

    # ZWNJ is Cf, not \w, so it has to be named explicitly or ثبت‌نام tokenizes
    # as two words and can never match the one-word hotword.
    return re.sub(rf"[\w{ZWNJ}]+", replace, text)


def demo() -> None:
    assert normalize("كي") == "کی" and normalize("مدرسة") == "مدرسه"
    assert normalize("سَلام") == "سلام", "harakat must not survive"
    assert normalize("ثبت ‌ نام") == "ثبت‌نام", "spaced half-space is one word"
    assert normalize("  برو   به  ") == "برو به"

    # The mergers that matter, and the ones that must not happen.
    assert phonetic_key("گذارش") == phonetic_key("گزارش")
    assert phonetic_key("مسئله") == phonetic_key("مساله")
    assert phonetic_key("پل") != phonetic_key("بل"), "پ/ب are different sounds"
    assert phonetic_key("گل") != phonetic_key("کل"), "گ/ک are different sounds"

    # Correction against the real hotwords file.
    assert "گزارش" in HOTWORDS, "hotwords did not load"
    assert spell_correct("گذارش") == "گزارش"          # exact bucket hit
    assert spell_correct("سفه") == "صفحه"             # needs the fuzzy step
    assert spell_correct("داشبرد") == "داشبورد"
    assert spell_correct("برو به گذارش") == "برو به گزارش"

    # False positives are the failure mode that matters more than the fix rate.
    assert spell_correct("گزارش") == "گزارش", "a correct word must survive"
    assert spell_correct("کاربران") == "کاربران", "0.833 is below the threshold"
    # Inflected forms score far above the threshold; only _is_derived saves them.
    assert spell_correct("کاربری") == "کاربری", "an adjective is not a misspelling"
    assert spell_correct("راهنمای") == "راهنمای", "an ezafe form is not a misspelling"
    assert spell_correct("open the dashboard") == "open the dashboard"
    assert spell_correct("برو به dashboard") == "برو به dashboard"

    # Disabled means tier 1 only -- normalization still runs.
    assert spell_correct("گذارش", enabled=False) == "گذارش"
    assert spell_correct("كي", enabled=False) == "کی"

    print(f"spellcheck ok -- {len(HOTWORDS)} hotwords, {len(LEXICON)} lexicon words")


if __name__ == "__main__":
    demo()
