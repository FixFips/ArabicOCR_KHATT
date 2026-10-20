"""Pure-Python n-gram language models + LM-fused CTC beam search.

KenLM needs a C++ toolchain this Windows box doesn't have, so the models are
implemented here (counts in dicts; fine up to ~30M training characters).

Everything operates in the model's DECODING direction: the CTC emits visual
left-to-right order, so LM training text must be passed through the same
flip the checkpoint uses (``model.label_flipper(state)``) before training.

Why not the old bigram (model.build_bigram_lm)? Run-2 sweep: it hurt at
weight >= 0.3. Literature (PyLaia, arXiv 2404.18722) gets -0.9 CER / -3.3 WER
on average with a 6-gram char LM *plus an insertion bonus*; without the bonus
an LM shortens outputs, and our model already under-transcribes (31% of edit
ops are deletions). The word lexicon prior targets space errors: merged or
wrongly split words are usually out-of-vocabulary.
"""
import math
import re
from collections import Counter, defaultdict

import numpy as np

# No torch / package imports: decode workers load this file standalone
# (scripts/lm_decode_sweep.py) so each process stays light on Windows.

_PUNCT = re.compile(r"[^ء-ي]")


def ids_to_text(ids, id2char: dict) -> str:
    """Same as model.ids_to_text (kept local so this module is torch-free)."""
    out = [id2char.get(int(i), "") for i in ids]
    out = [c for c in out if c and not (c.startswith("<") and c.endswith(">"))]
    return re.sub(r"\s+", " ", "".join(out)).strip()


def ctc_greedy_decode_np(log_probs: np.ndarray, id2char: dict) -> str:
    """Greedy CTC decode for one line, ``log_probs``: [T, C]."""
    best = log_probs.argmax(-1)
    seq, last = [], -1
    for p in best:
        if p != last and p != 0:
            seq.append(int(p))
        last = p
    return ids_to_text(seq, id2char)


class CharNgramLM:
    """Interpolated Kneser-Ney character n-gram LM (single discount per order).

    Text is wrapped in a start marker so line-initial context is modelled.
    ``logp(context, ch)`` returns natural-log probability.
    """

    BOS = "\x02"

    def __init__(self, order: int = 6, discount: float = 0.75):
        self.order = order
        self.d = discount
        self.vocab = set()
        # per order k (1..N): gram[k][h+c] count, tot[k][h], types[k][h]
        self.gram = [None] + [Counter() for _ in range(order)]
        self.tot = [None] + [Counter() for _ in range(order)]
        self.types = [None] + [Counter() for _ in range(order)]
        self._cache = {}

    def fit(self, texts):
        n = self.order
        top = self.gram[n]
        for t in texts:
            s = self.BOS * (n - 1) + t
            self.vocab.update(t)
            for i in range(n - 1, len(s)):
                top[s[i - n + 1:i + 1]] += 1
        # lower orders use continuation counts: distinct left extensions
        for k in range(n - 1, 0, -1):
            g = self.gram[k]
            for gram in self.gram[k + 1]:
                g[gram[1:]] += 1
        for k in range(1, n + 1):
            tot, typ = self.tot[k], self.types[k]
            for gram, c in self.gram[k].items():
                tot[gram[:-1]] += c
                typ[gram[:-1]] += 1
        self.V = len(self.vocab) + 1
        return self

    def _p(self, h: str, c: str, k: int) -> float:
        if k == 0:
            return 1.0 / self.V
        h = h[len(h) - (k - 1):] if k > 1 else ""
        lower = self._p(h[1:] if h else "", c, k - 1)
        tot = self.tot[k].get(h, 0)
        if tot == 0:
            return lower
        cnt = self.gram[k].get(h + c, 0)
        return max(cnt - self.d, 0.0) / tot + self.d * self.types[k][h] / tot * lower

    def logp(self, context: str, ch: str) -> float:
        h = (self.BOS * (self.order - 1) + context)[-(self.order - 1):]
        key = (h, ch)
        v = self._cache.get(key)
        if v is None:
            v = math.log(self._p(h, ch, self.order))
            if len(self._cache) < 5_000_000:
                self._cache[key] = v
        return v


class WordLexicon:
    """Unigram word prior with an OOV penalty (natural-log scores).

    Tokens are compared with punctuation/digits stripped; tokens that are
    pure punctuation or numbers score 0 (neutral).
    """

    def __init__(self, min_count: int = 2):
        self.min_count = min_count
        self.logp = {}

    def fit(self, texts):
        c = Counter()
        for t in texts:
            for w in t.split():
                w = _PUNCT.sub("", w)
                if w:
                    c[w] += 1
        total = sum(c.values())
        self.logp = {w: math.log(n / total) for w, n in c.items() if n >= self.min_count}
        self.floor = math.log(self.min_count / total)
        return self

    def score(self, word: str, oov: float) -> float:
        w = _PUNCT.sub("", word)
        if not w:
            return 0.0
        lp = self.logp.get(w)
        # relative to the floor so in-vocab words get a bonus >= 0, OOV pays `oov`
        return (lp - self.floor) / 10.0 if lp is not None else -oov


def _lse(a, b):
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    m = max(a, b)
    return m + math.log1p(math.exp(-abs(a - b)))


def ctc_lm_beam_decode(log_probs: np.ndarray, id2char: dict, char_lm=None, word_lm=None,
                       alpha: float = 0.5, beta: float = 1.0, gamma: float = 0.0,
                       oov: float = 3.0, beam_width: int = 10,
                       blank_skip: float = 0.999) -> str:
    """CTC prefix beam search with char-LM shallow fusion, insertion bonus and
    word prior, for ONE line. ``log_probs``: [T, C] natural-log softmax.

    score = log P_ctc + alpha * log P_charLM + beta * n_chars + gamma * word_prior
    Frames whose blank prob >= blank_skip are skipped (pure speed-up).
    """
    space_id = next((i for i, c in id2char.items() if c == " "), None)
    is_char = np.array([not (id2char.get(i, "<").startswith("<") and id2char.get(i, "").endswith(">"))
                        for i in range(log_probs.shape[1])])
    is_char[0] = False
    # beam entry: prefix(tuple) -> [pb, pnb, lm_score, text, last_word_start]
    beams = {(): [0.0, -math.inf, 0.0, "", 0]}
    blank_lp = log_probs[:, 0]
    top_k = min(beam_width, log_probs.shape[1])

    def ext_lm(entry, c):
        """LM score delta for appending char id c to entry."""
        ch = id2char[c]
        text = entry[3]
        d = 0.0
        if char_lm is not None:
            d += alpha * char_lm.logp(text, ch)
        d += beta
        if word_lm is not None and gamma and c == space_id:
            d += gamma * word_lm.score(text[entry[4]:], oov)
        return d, text + ch, (len(text) + 1 if c == space_id else entry[4])

    for t in range(log_probs.shape[0]):
        if blank_lp[t] >= math.log(blank_skip):
            # blank-dominated frame: carry beams through on blank
            for e in beams.values():
                p = _lse(e[0], e[1]) + blank_lp[t]
                e[0], e[1] = p, -math.inf
            continue
        lp_t = log_probs[t]
        cand = np.argpartition(-lp_t, top_k - 1)[:top_k]
        nb = {}
        for prefix, e in beams.items():
            pb, pnb = e[0], e[1]
            ptot = _lse(pb, pnb)
            for c in cand:
                c = int(c)
                lp = float(lp_t[c])
                if c == 0:
                    n = nb.setdefault(prefix, [-math.inf, -math.inf, e[2], e[3], e[4]])
                    n[0] = _lse(n[0], ptot + lp)
                    continue
                if not is_char[c]:
                    continue
                if prefix and prefix[-1] == c:
                    n = nb.setdefault(prefix, [-math.inf, -math.inf, e[2], e[3], e[4]])
                    n[1] = _lse(n[1], pnb + lp)
                    src = pb
                else:
                    src = ptot
                key = prefix + (c,)
                n = nb.get(key)
                if n is None:
                    d, txt, ws = ext_lm(e, c)
                    n = nb[key] = [-math.inf, -math.inf, e[2] + d, txt, ws]
                n[1] = _lse(n[1], src + lp)
        ranked = sorted(nb.items(), key=lambda kv: _lse(kv[1][0], kv[1][1]) + kv[1][2], reverse=True)
        beams = dict(ranked[:beam_width])

    def final(e):
        s = _lse(e[0], e[1]) + e[2]
        if word_lm is not None and gamma:
            s += gamma * word_lm.score(e[3][e[4]:], oov)
        return s

    best = max(beams.items(), key=lambda kv: final(kv[1]))
    return ids_to_text(list(best[0]), id2char)
