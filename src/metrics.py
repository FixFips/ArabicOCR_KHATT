# src/metrics.py
import re
import unicodedata as ud
from rapidfuzz.distance import Levenshtein


_re_diac = re.compile(r"[\u064B-\u0652]")


def norm_ar(s: str) -> str:
    """Collapse orthographic variants that don't change meaning/phonetics.

    Applies: NFKC, strip tatweel and diacritics, fold alef-variants (أ إ آ → ا),
    fold ى → ي, collapse whitespace. KHATT labels mix these forms (e.g. 102
    words appear with both ى and ي forms in train), so normalizing avoids
    penalizing the model for arbitrary scribe choices.
    """
    s = ud.normalize("NFKC", s)
    s = s.replace("\u0640", "")
    s = _re_diac.sub("", s)
    s = (s.replace("\u0623", "\u0627")
           .replace("\u0625", "\u0627")
           .replace("\u0622", "\u0627")
           .replace("\u0649", "\u064A"))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def cer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    return Levenshtein.distance(list(ref), list(hyp)) / len(ref)


def cer_norm(ref: str, hyp: str) -> float:
    """CER after Arabic orthographic normalization (see norm_ar)."""
    return cer(norm_ar(ref), norm_ar(hyp))


def wer(ref: str, hyp: str) -> float:
    r = ref.split()
    h = hyp.split()
    if not r:
        return 0.0 if not h else 1.0
    return Levenshtein.distance(r, h) / len(r)


# Arabic dot-group characters: letters that share the same base stroke
# and differ ONLY by dot count/position.
_DOT_GROUPS = {
    # ba/ta/tha group (same tooth shape)
    "\u0628",  # ba  — 1 dot below
    "\u062A",  # ta  — 2 dots above
    "\u062B",  # tha — 3 dots above
    # nun/ya group (similar final forms)
    "\u0646",  # nun — 1 dot above
    "\u064A",  # ya  — 2 dots below
    # jim/ha/kha group (same bowl shape)
    "\u062C",  # jim — 1 dot below
    "\u062D",  # ha  — no dot
    "\u062E",  # kha — 1 dot above
    # fa/qaf group (same base stroke; 30 qaf→fa subs seen in v2 val analysis 2026-04-23)
    "\u0641",  # fa  — 1 dot above
    "\u0642",  # qaf — 2 dots above
}


def dot_group_cer(refs: list[str], hyps: list[str]) -> float:
    """
    CER measured only on dot-differentiated Arabic letter groups.

    Extracts characters belonging to dot-ambiguous groups from both
    reference and hypothesis, then computes edit distance on those
    subsequences.  This directly measures the #1 error source in
    Arabic handwriting OCR.
    """
    total_ref_len = 0
    total_distance = 0

    for ref, hyp in zip(refs, hyps):
        ref_dots = [ch for ch in ref if ch in _DOT_GROUPS]
        hyp_dots = [ch for ch in hyp if ch in _DOT_GROUPS]
        if not ref_dots:
            continue
        total_ref_len += len(ref_dots)
        total_distance += Levenshtein.distance(ref_dots, hyp_dots)

    if total_ref_len == 0:
        return 0.0
    return total_distance / total_ref_len
