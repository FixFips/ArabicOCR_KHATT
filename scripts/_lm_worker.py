"""Torch-free worker for lm_decode_sweep.py (spawned processes import only this)."""
import importlib.util
import pickle
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "arabicocr_lm", Path(__file__).resolve().parents[1] / "arabicocr_khatt" / "lm.py")
lm = importlib.util.module_from_spec(_spec)
sys.modules["arabicocr_lm"] = lm  # so pickled LM objects resolve their class
_spec.loader.exec_module(lm)

_W = {}


def bidi_flip(text: str) -> str:
    rev = text[::-1]
    return lm.re.sub(r"[0-9٠-٩]+(?:[.,:/][0-9٠-٩]+)*|[A-Za-z]+",
                     lambda m: m.group(0)[::-1], rev)


def init(path):
    # LM classes were pickled under this module's name, so unpickling works here
    with open(path, "rb") as f:
        _W.update(pickle.load(f))
    _W["flip"] = bidi_flip if _W["label_order"] == "bidi" else (lambda t: t[::-1])


def decode(args):
    i, cfg = args
    lp = _W["lps"][i]
    if cfg is None:
        hyp = lm.ctc_greedy_decode_np(lp, _W["id2char"])
    else:
        hyp = lm.ctc_lm_beam_decode(lp, _W["id2char"], _W["clm"], _W["wlm"], beam_width=_W["beam"], **cfg)
    return _W["flip"](hyp)
