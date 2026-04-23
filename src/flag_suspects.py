"""Flag suspect val samples likely caused by KHATT dataset errors, not model errors.

Reads a val samples TSV (written by eval_val.py) plus the source images,
and flags lines that are probably corrupt/mislabeled:

  * blank        -- almost no ink in the image
  * non_line_dim -- image height far outside single-line range (multi-line / page scan)
  * high_cer     -- CER >= threshold despite non-blank image (likely label mismatch)

Output: a TSV listing every flagged sample with its flag, CER, image dims,
and ink ratio, sorted worst-first. Use this list to:
  - visually audit the worst offenders
  - feed to --exclude in future eval runs (not wired up yet)

Usage:
    python -m src.flag_suspects
    python -m src.flag_suspects --tsv runs/exp1/val_epoch_999_samples.tsv
"""
import argparse
import csv
import os
import sys
from collections import Counter

import numpy as np
from PIL import Image
from rapidfuzz.distance import Levenshtein as _Lev

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_TSV    = os.path.join("runs", "exp1", "val_epoch_999_samples.tsv")
DEFAULT_OUT    = os.path.join("runs", "exp1", "val_suspects.tsv")
DEFAULT_IMGDIR = os.path.join("archive", "images")

# Thresholds
INK_BLANK_MAX        = 0.002   # < 0.2% dark pixels -> blank
LINE_HEIGHT_MAX      = 300     # line images in KHATT are ~100-200px; >300 = not a line
HIGH_CER_THRESHOLD   = 0.5     # CER >= 50% on a non-blank image -> likely mislabel


def ink_ratio(img: Image.Image) -> float:
    """Fraction of dark pixels (ink) after a simple threshold."""
    arr = np.asarray(img.convert("L"))
    return float((arr < 128).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default=DEFAULT_TSV)
    ap.add_argument("--images", default=DEFAULT_IMGDIR)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--cer-threshold", type=float, default=HIGH_CER_THRESHOLD)
    args = ap.parse_args()

    if not os.path.exists(args.tsv):
        raise SystemExit(f"TSV not found: {args.tsv}")

    flags_counter = Counter()
    rows_out = []
    total = 0
    missing_img = 0

    with open(args.tsv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            total += 1
            fn = row.get("filename") or ""
            gt = row.get("label") or ""
            pr = row.get("pred") or ""

            img_path = os.path.join(args.images, fn) if fn else ""
            if not fn or not os.path.exists(img_path):
                missing_img += 1
                continue

            with Image.open(img_path) as im:
                w, h = im.size
                ratio = ink_ratio(im)

            line_cer = _Lev.distance(gt, pr) / max(len(gt), 1)

            flags = []
            if ratio < INK_BLANK_MAX:
                flags.append("blank")
            if h > LINE_HEIGHT_MAX:
                flags.append("non_line_dim")
            if line_cer >= args.cer_threshold and ratio >= INK_BLANK_MAX:
                flags.append("high_cer")

            if flags:
                for fl in flags:
                    flags_counter[fl] += 1
                rows_out.append({
                    "filename": fn,
                    "flags": ",".join(flags),
                    "cer": round(line_cer, 4),
                    "ink_ratio": round(ratio, 4),
                    "width": w,
                    "height": h,
                    "label": gt,
                    "pred": pr,
                })

    # Sort worst-first: non-blank high-CER before blanks (since blanks are obvious)
    rows_out.sort(key=lambda r: (-r["cer"], -r["height"]))

    with open(args.out, "w", encoding="utf-8", newline="") as fo:
        w = csv.DictWriter(fo, delimiter="\t",
                           fieldnames=["filename", "flags", "cer", "ink_ratio",
                                       "width", "height", "label", "pred"])
        w.writeheader()
        for r in rows_out:
            r["label"] = r["label"].replace("\t", " ").replace("\n", " ")
            r["pred"] = r["pred"].replace("\t", " ").replace("\n", " ")
            w.writerow(r)

    n_flagged = len(rows_out)
    print(f"Total samples  : {total}")
    if missing_img:
        print(f"Missing images : {missing_img}")
    print(f"Flagged        : {n_flagged}  ({100 * n_flagged / max(total, 1):.1f}%)")
    print("\nFlag breakdown (samples may carry >1 flag):")
    for fl, n in flags_counter.most_common():
        print(f"  {fl:14s}: {n}")

    print(f"\nWrote: {args.out}")
    if rows_out:
        print(f"\nTop 10 suspects (by CER, then height):")
        for i, r in enumerate(rows_out[:10], 1):
            gt_disp = (r["label"][:80] + "...") if len(r["label"]) > 80 else r["label"]
            pr_disp = (r["pred"][:80] + "...") if len(r["pred"]) > 80 else r["pred"]
            print(f"\n[{i}] {r['filename']}  flags={r['flags']}  "
                  f"cer={r['cer']:.2f}  ink={r['ink_ratio']:.4f}  dims={r['width']}x{r['height']}")
            print(f"  GT: {gt_disp}")
            print(f"  PR: {pr_disp}")


if __name__ == "__main__":
    main()
