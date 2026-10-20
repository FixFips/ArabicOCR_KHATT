"""Build a writer-disjoint, text-deduplicated KHATT split.

The legacy split (archive/splits, random 80/10/10 over line images) shares
every test writer with train and ~22% of test lines have their exact text in
train: KHATT's fixed paragraphs are copied by every writer. That inflates CER
and makes it incomparable with the unique-text protocol used in the literature
(e.g. arXiv 2601.16713, 2410.02179).

This script:
  1. assigns WRITERS (AHTD3A<id>) to train/val/test 80/10/10,
  2. drops val/test lines whose text overlaps train — measured as the share of
     the line's word 3-grams that occur anywhere in train text (exact-line
     matching misses overlap because writers break lines at different places),
  3. drops known mislabels (flag_suspects TSVs) from TRAIN only; val/test are
     kept whole so reported numbers need no exclusion footnote.

Usage:
    python scripts/build_strict_splits.py \
        --exclude-train runs/exp1/train_suspects.tsv runs/exp1/val_suspects.tsv \
                        runs/exp1/test_suspects.tsv
"""
import argparse
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from arabicocr_khatt.dataset import read_label  # noqa: E402
from arabicocr_khatt.metrics import norm_ar  # noqa: E402

IMAGES_DIR = "./archive/images"
LABELS_DIR = "./archive/labels"


def writer_of(fname: str) -> str:
    return fname.split("_")[0]


def ngrams(text: str, n: int = 3) -> set:
    w = norm_ar(text).split()
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./archive/splits_strict")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overlap", type=float, default=0.5,
                    help="Drop a val/test line if >= this share of its word 3-grams occur in train.")
    ap.add_argument("--exclude-train", nargs="*", default=[],
                    help="flag_suspects TSVs; their filenames are removed from train only.")
    args = ap.parse_args()

    rows = []
    for fname in sorted(os.listdir(IMAGES_DIR)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        label_path = os.path.join(LABELS_DIR, os.path.splitext(fname)[0] + ".txt")
        if os.path.exists(label_path):
            rows.append({"filename": fname, "label_path": label_path,
                         "text": read_label(label_path)})
    df = pd.DataFrame(rows)
    df["writer"] = df["filename"].map(writer_of)

    writers = sorted(df["writer"].unique())
    random.Random(args.seed).shuffle(writers)
    n_val = n_test = round(len(writers) * 0.1)
    role = {w: "test" for w in writers[:n_test]}
    role.update({w: "val" for w in writers[n_test:n_test + n_val]})
    role.update({w: "train" for w in writers[n_test + n_val:]})
    df["split"] = df["writer"].map(role)

    bad = set()
    for p in args.exclude_train:
        bad.update(pd.read_csv(p, sep="\t")["filename"].astype(str))
    train = df[(df["split"] == "train") & ~df["filename"].isin(bad)]
    train_grams = set().union(*(ngrams(t) for t in train["text"]))

    def overlap(text):
        g = ngrams(text)
        return sum(x in train_grams for x in g) / len(g) if g else 0.0

    os.makedirs(args.out, exist_ok=True)
    train[["filename", "label_path"]].to_csv(Path(args.out, "train.csv"), index=False)
    print(f"writers: {len(writers)} total -> {len(writers) - 2 * n_val} train / {n_val} val / {n_test} test")
    print(f"train: {len(train)} lines ({(df['split'] == 'train').sum() - len(train)} suspects dropped)")
    for split in ["val", "test"]:
        part = df[df["split"] == split].copy()
        part["overlap"] = part["text"].map(overlap)
        keep = part[part["overlap"] < args.overlap]
        para = Counter(re.search(r"_(Para\d)", f).group(1) for f in part.loc[part["overlap"] >= args.overlap, "filename"])
        keep[["filename", "label_path"]].to_csv(Path(args.out, f"{split}.csv"), index=False)
        print(f"{split}: {len(keep)} kept / {len(part)} (dropped for train-text overlap: {dict(sorted(para.items()))})")


if __name__ == "__main__":
    main()
