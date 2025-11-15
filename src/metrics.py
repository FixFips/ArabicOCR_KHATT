# src/metrics.py
from rapidfuzz.distance import Levenshtein

def cer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0 if not hyp else 1.0
    return Levenshtein.distance(list(ref), list(hyp)) / len(ref)

def wer(ref: str, hyp: str) -> float:
    r = ref.split()
    h = hyp.split()
    if not r:
        return 0.0 if not h else 1.0
    return Levenshtein.distance(r, h) / len(r)