"""Write {val,test}_clean.csv: the strict split minus lines whose label belongs
to a different image (verified by eye, listed in research/strict_label_errors.tsv).

Only objectively wrong labels are listed (shifted paragraphs, "#" labels) —
never lines that are merely hard. Report both raw and clean numbers.

Usage:
    python scripts/make_clean_splits.py --splits-dir archive/splits_strict
"""
import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits-dir", default="./archive/splits_strict")
    ap.add_argument("--errors", default="./research/strict_label_errors.tsv")
    args = ap.parse_args()

    bad = set(pd.read_csv(args.errors, sep="\t")["filename"])
    for split in ["val", "test"]:
        df = pd.read_csv(Path(args.splits_dir, f"{split}.csv"))
        clean = df[~df["filename"].isin(bad)]
        clean.to_csv(Path(args.splits_dir, f"{split}_clean.csv"), index=False)
        print(f"{split}: {len(df)} -> {len(clean)}")


if __name__ == "__main__":
    main()
