# scripts/build_correction_pairs.py
"""Build (noisy, clean) text pairs for OCR post-correction training.

Learns a character-level error model from a real eval TSV (label/pred columns,
e.g. eval_val output) — per-character substitution/deletion probabilities and
confusion targets, plus insertion rates — then applies that stochastic
corruption to clean text lines. The injected noise matches the OCR model's
real error distribution by construction (including space del/ins, the #1
category), so a corrector trained on these pairs sees realistic inputs.

This needs no GPU and no OCR inference. ~5k real (pred, label) dev pairs come
directly from the eval TSVs; this script provides the >=100k training pairs
the literature says seq2seq correctors need (arXiv 2502.01205).

Usage:
    python scripts/build_correction_pairs.py \
        --error-tsv runs/exp2_ft/eval_val_samples.tsv \
        --n 100000 --out archive/correction/pairs_train.tsv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rapidfuzz.distance import Levenshtein  # noqa: E402

from arabicocr_khatt.dataset import read_label  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class CharErrorModel:
    """Per-character sub/del probabilities + confusion targets + insertions."""

    def __init__(self):
        self.total = Counter()          # gt char -> occurrences
        self.subs = defaultdict(Counter)  # gt char -> Counter(pred char)
        self.dels = Counter()           # gt char -> deletions
        self.ins = Counter()            # inserted char -> count
        self.n_positions = 0            # total gt chars (insertion rate base)

    def fit(self, rows: list[tuple[str, str]]) -> "CharErrorModel":
        for gt, pr in rows:
            for ch in gt:
                self.total[ch] += 1
            self.n_positions += len(gt)
            for op in Levenshtein.editops(gt, pr):
                if op.tag == "replace":
                    self.subs[gt[op.src_pos]][pr[op.dest_pos]] += 1
                elif op.tag == "delete":
                    self.dels[gt[op.src_pos]] += 1
                else:
                    self.ins[pr[op.dest_pos]] += 1
        return self

    def stats(self) -> str:
        n_sub = sum(sum(c.values()) for c in self.subs.values())
        n_del = sum(self.dels.values())
        n_ins = sum(self.ins.values())
        cer = (n_sub + n_del + n_ins) / max(self.n_positions, 1)
        return (f"fitted on {self.n_positions:,} chars: sub={n_sub} del={n_del} "
                f"ins={n_ins} (source CER~{cer*100:.2f}%)")

    def corrupt(self, text: str, rng: np.random.Generator, scale: float = 1.0) -> str:
        """Apply the learned noise to a clean line."""
        ins_rate = scale * sum(self.ins.values()) / max(self.n_positions, 1)
        ins_chars = list(self.ins)
        ins_p = np.array([self.ins[c] for c in ins_chars], dtype=np.float64)
        ins_p = ins_p / ins_p.sum() if ins_p.sum() else ins_p

        out = []
        for ch in text:
            n = self.total.get(ch, 0)
            if n:
                p_sub = scale * sum(self.subs[ch].values()) / n
                p_del = scale * self.dels[ch] / n
                r = rng.random()
                if r < p_del:
                    pass  # deleted
                elif r < p_del + p_sub and self.subs[ch]:
                    cands = list(self.subs[ch])
                    w = np.array([self.subs[ch][c] for c in cands], dtype=np.float64)
                    out.append(cands[rng.choice(len(cands), p=w / w.sum())])
                else:
                    out.append(ch)
            else:
                out.append(ch)
            if ins_chars and rng.random() < ins_rate:
                out.append(ins_chars[int(rng.choice(len(ins_chars), p=ins_p))])
        return "".join(out)


def load_pairs_tsv(path: str) -> list[tuple[str, str]]:
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            gt, pr = r.get("label") or "", r.get("pred") or ""
            if gt:
                rows.append((gt, pr))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--error-tsv", required=True,
                    help="Eval TSV (label/pred) to learn the error model from. "
                         "Use a VAL tsv — never test.")
    ap.add_argument("--source-csv", default=str(ROOT / "archive/splits/train.csv"),
                    help="Split CSV whose labels provide clean text (default train).")
    ap.add_argument("--text-file", default=None,
                    help="Optional extra clean-text file (one line per row) to mix in.")
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--noise-scale", type=float, default=1.0,
                    help="Multiply learned error rates (1.0 = match source CER).")
    ap.add_argument("--clean-frac", type=float, default=0.15,
                    help="Fraction of identity (clean->clean) pairs — teaches the "
                         "corrector to leave correct text alone (anti-hallucination).")
    ap.add_argument("--out", default=str(ROOT / "archive/correction/pairs_train.tsv"))
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args()

    model = CharErrorModel().fit(load_pairs_tsv(args.error_tsv))
    print(model.stats())

    # clean text pool (tatweel-stripped, matching the OCR label policy)
    pool: list[str] = []
    df = pd.read_csv(args.source_csv)
    for lp in df["label_path"]:
        try:
            t = read_label(str(ROOT / str(lp).replace("\\", "/").lstrip("./")))
            t = t.replace("ـ", "").strip()
            if t:
                pool.append(t)
        except Exception:
            continue
    if args.text_file:
        for line in Path(args.text_file).read_text(encoding="utf-8").splitlines():
            line = line.replace("ـ", "").strip()
            if len(line) >= 8:
                pool.append(line)
    print(f"clean text pool: {len(pool):,} lines")

    rng = np.random.default_rng(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    inj_dist = inj_len = 0
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["noisy", "clean"])
        while n_written < args.n:
            clean = pool[int(rng.integers(len(pool)))]
            if rng.random() < args.clean_frac:
                noisy = clean  # identity pair: learn to abstain on correct text
            else:
                noisy = model.corrupt(clean, rng, scale=args.noise_scale)
                if not noisy.strip():
                    continue
            w.writerow([noisy, clean])
            inj_dist += Levenshtein.distance(clean, noisy)
            inj_len += len(clean)
            n_written += 1

    print(f"wrote {n_written:,} pairs -> {out_path}")
    print(f"injected CER: {inj_dist / max(inj_len, 1) * 100:.2f}% "
          f"(should track source CER x noise-scale)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
