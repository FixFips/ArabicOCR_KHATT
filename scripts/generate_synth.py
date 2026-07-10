# scripts/generate_synth.py
"""
Generate the Option A synthetic training set (default 100k lines).

Text sources (train split ONLY — val/test text must never leak into synth):
  natural  55%  — KHATT train labels verbatim (~6x renders each at 100k)
  dotmix   25%  — constructed lines from train vocabulary, words weighted
                  toward dot-differentiated letters (ب ت ث ن ي / ج ح خ / ف ق /
                  د ذ / ر ز ...) — DotCER is the #1 substitution source
  short    10%  — 1-3 word lines (worst real CER bucket)
  hamza     5%  — words rich in أ إ آ ء ؤ ئ ة ى (orthographic confusions)
  punct     5%  — constructed lines with . ، ؟ ! placed between words

Output (drop-in compatible with KHATTDataset via inline `label` column):
  <out>/images/synth_XXXXXX.jpg
  <out>/synth.csv   columns: filename,label,family,kind

Usage:
  python scripts/generate_synth.py --n 100000 --procs 8
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from arabicocr_khatt.dataset import read_label            # noqa: E402
from arabicocr_khatt.synthetic import (                   # noqa: E402
    LineRenderer, clean_synth_text, render_sample,
)

DOT_LETTERS = set("بتثنيىجحخفقدذرزسشصضطظعغة")
HAMZA_LETTERS = set("أإآءؤئةى")
PUNCT = [".", "،", "؟", "!", ":"]


# ----------------------------------------------------------------------------
# Text pool construction
# ----------------------------------------------------------------------------

def load_natural_pool(train_csv: Path) -> list[str]:
    df = pd.read_csv(train_csv)
    pool = []
    for lp in df["label_path"]:
        p = ROOT / str(lp).replace("\\", "/").lstrip("./")
        try:
            t = clean_synth_text(read_label(str(p)))
        except Exception:
            continue
        if t:
            pool.append(t)
    return pool


def build_vocab(pool: list[str]) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Return (words, freq_weights, dot_weights, hamza_weights)."""
    counts = Counter(w for line in pool for w in line.split(" ")
                     if 2 <= len(w) <= 14)
    words = list(counts)
    freq = np.array([counts[w] for w in words], dtype=np.float64)
    dot = np.array([1.0 + 2.0 * sum(ch in DOT_LETTERS for ch in w) for w in words])
    ham = np.array([sum(ch in HAMZA_LETTERS for ch in w) for w in words])
    return words, freq, dot, ham


def make_texts(n: int, pool: list[str], rng: np.random.Generator) -> list[tuple[str, str]]:
    """Build (kind, text) list according to the source mix."""
    words, freq, dot, ham = build_vocab(pool)
    p_freq = freq / freq.sum()
    p_dot = (freq * dot) / (freq * dot).sum()
    p_ham_raw = freq * ham
    p_ham = p_ham_raw / p_ham_raw.sum()

    def sample_words(k: int, p: np.ndarray) -> str:
        idx = rng.choice(len(words), size=k, p=p)
        return " ".join(words[i] for i in idx)

    kinds = rng.choice(
        ["natural", "dotmix", "short", "hamza", "punct"],
        size=n, p=[0.55, 0.25, 0.10, 0.05, 0.05],
    )
    out: list[tuple[str, str]] = []
    for kind in kinds:
        if kind == "natural":
            text = pool[int(rng.integers(len(pool)))]
        elif kind == "dotmix":
            text = sample_words(int(rng.integers(4, 11)), p_dot)
        elif kind == "short":
            text = sample_words(int(rng.integers(1, 4)), p_freq)
        elif kind == "hamza":
            text = sample_words(int(rng.integers(3, 9)), p_ham)
        else:  # punct
            k = int(rng.integers(4, 10))
            toks = sample_words(k, p_freq).split(" ")
            for _ in range(int(rng.integers(1, 4))):
                pos = int(rng.integers(1, len(toks) + 1))
                toks.insert(pos, PUNCT[int(rng.integers(len(PUNCT)))])
            text = " ".join(toks)
        out.append((str(kind), text))
    return out


# ----------------------------------------------------------------------------
# Multiprocess rendering (Windows spawn-safe: module-level worker + globals)
# ----------------------------------------------------------------------------

_RENDERER: LineRenderer | None = None
_IMAGES_DIR: Path | None = None


def _init_worker(images_dir: str) -> None:
    global _RENDERER, _IMAGES_DIR
    _RENDERER = LineRenderer()
    _RENDERER._glyph_cache_cap = 12000  # informational; cap lives in synthetic.py
    _IMAGES_DIR = Path(images_dir)


def _render_one(task: tuple[int, str, str, int]) -> tuple[str, str, str, str] | None:
    idx, kind, text, seed = task
    rng = np.random.default_rng(seed)
    out = render_sample(text, _RENDERER, rng)
    if out is None:
        return None
    img, label, family = out
    fn = f"synth_{idx:06d}.jpg"
    img.save(_IMAGES_DIR / fn, quality=int(rng.integers(82, 93)))
    return fn, label, family, kind


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--out", default=str(ROOT / "archive" / "synth"))
    ap.add_argument("--train-csv", default=str(ROOT / "archive" / "splits" / "train.csv"))
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    out_dir = Path(args.out)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    pool = load_natural_pool(Path(args.train_csv))
    print(f"natural pool: {len(pool)} lines")
    # 5% headroom for font-coverage rejects, trimmed back to n after rendering
    texts = make_texts(int(args.n * 1.05), pool, rng)
    tasks = [(i, k, t, int(rng.integers(0, 2**31))) for i, (k, t) in enumerate(texts)]

    rows: list[tuple[str, str, str, str]] = []
    t0 = time.time()
    from multiprocessing import Pool
    with Pool(args.procs, initializer=_init_worker, initargs=(str(images_dir),)) as p:
        for res in p.imap_unordered(_render_one, tasks, chunksize=64):
            if res is not None:
                rows.append(res)
            done = len(rows)
            if done % 5000 == 0 and done:
                rate = done / (time.time() - t0)
                print(f"  {done}/{args.n} ({rate:.0f} lines/s, "
                      f"eta {(args.n - done) / rate / 60:.0f} min)", flush=True)
            if done >= args.n:
                p.terminate()
                break

    rows = rows[: args.n]
    rows.sort()
    with open(out_dir / "synth.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label", "family", "kind"])
        w.writerows(rows)

    dt = time.time() - t0
    kinds = Counter(r[3] for r in rows)
    fams = Counter(r[2] for r in rows)
    print(f"\nwrote {len(rows)} lines in {dt/60:.1f} min -> {out_dir / 'synth.csv'}")
    print("kinds:", dict(kinds))
    print("families:", dict(fams.most_common()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
