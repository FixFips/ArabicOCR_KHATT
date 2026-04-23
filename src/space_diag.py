"""Diagnose space-deletion pattern in val predictions.

Reads the full-val TSV (written by eval_val.py) and reports:
  * total spaces in GT vs PR
  * per-line deletions
  * alignment classification of where spaces are lost (word-boundary vs other)
  * sample lines with the biggest space deletions

Usage:
    python -m src.space_diag
    python -m src.space_diag --tsv runs/exp1/val_epoch_999_samples.tsv
"""
import argparse
import csv
import os
import sys
from collections import Counter

# Windows console defaults to cp1252 and chokes on Arabic. Force UTF-8 stdout.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from rapidfuzz.distance import Levenshtein as _Lev

DEFAULT_TSV = os.path.join("runs", "exp1", "val_epoch_999_samples.tsv")


def classify_deletion(gt: str, pos: int) -> str:
    """Describe the context around a deleted space in gt at position `pos`."""
    left = gt[pos - 1] if pos > 0 else ""
    right = gt[pos + 1] if pos + 1 < len(gt) else ""
    if not left or not right:
        return "edge"
    # Arabic letter unicode range roughly \u0600-\u06FF
    def is_ar(c):
        return "\u0600" <= c <= "\u06FF"
    if is_ar(left) and is_ar(right):
        return "ar_ar"            # between two Arabic letters (word gap)
    if left.isdigit() or right.isdigit():
        return "digit_adj"
    if left in ".,،؛:!؟\"'()" or right in ".,،؛:!؟\"'()":
        return "punct_adj"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default=DEFAULT_TSV)
    ap.add_argument("--top", type=int, default=10, help="show top-N lines with most deleted spaces")
    args = ap.parse_args()

    if not os.path.exists(args.tsv):
        raise SystemExit(f"TSV not found: {args.tsv}")

    gt_space_total = 0
    pr_space_total = 0
    deleted_total = 0
    inserted_total = 0
    context_counts = Counter()
    per_line = []  # (deletions, gt, pr)

    with open(args.tsv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            gt = row.get("label", "")
            pr = row.get("pred", "")
            g_sp = gt.count(" ")
            p_sp = pr.count(" ")
            gt_space_total += g_sp
            pr_space_total += p_sp

            # Align and count where GT spaces are dropped / extra spaces inserted
            ops = _Lev.editops(gt, pr)
            line_dels = 0
            for op, i1, i2 in ops:
                if op == "delete" and i1 < len(gt) and gt[i1] == " ":
                    deleted_total += 1
                    line_dels += 1
                    context_counts[classify_deletion(gt, i1)] += 1
                elif op == "insert" and i2 < len(pr) and pr[i2] == " ":
                    inserted_total += 1
                elif op == "replace" and i1 < len(gt) and gt[i1] == " ":
                    deleted_total += 1
                    line_dels += 1
                    context_counts[classify_deletion(gt, i1)] += 1
            per_line.append((line_dels, gt, pr))

    n_lines = len(per_line)
    print(f"Val lines       : {n_lines}")
    print(f"GT spaces       : {gt_space_total}")
    print(f"PR spaces       : {pr_space_total}  (diff {pr_space_total - gt_space_total:+d})")
    print(f"Deleted spaces  : {deleted_total}")
    print(f"Inserted spaces : {inserted_total}")
    print(f"Avg GT spaces/line: {gt_space_total / max(n_lines, 1):.1f}")
    print(f"Avg deletions/line: {deleted_total / max(n_lines, 1):.2f}")

    print("\nContext of deleted spaces:")
    for ctx, n in context_counts.most_common():
        pct = 100 * n / max(deleted_total, 1)
        print(f"  {ctx:12s}: {n:5d}  ({pct:4.1f}%)")

    print(f"\nTop {args.top} lines with most space deletions:")
    per_line.sort(reverse=True, key=lambda x: x[0])
    for i, (n, gt, pr) in enumerate(per_line[: args.top], 1):
        if n == 0:
            break
        gt_show = gt if len(gt) <= 120 else gt[:120] + "..."
        pr_show = pr if len(pr) <= 120 else pr[:120] + "..."
        print(f"\n[{i}] deleted={n}")
        print(f"  GT: {gt_show}")
        print(f"  PR: {pr_show}")


if __name__ == "__main__":
    main()
