# scripts/eval_correction.py
"""Apply a trained AraBART corrector to OCR predictions and measure the delta.

Reads a TSV with real (noisy, clean) pairs — or an eval TSV with label/pred
columns — corrects the noisy side, and reports CER / CER(n) / perfect lines /
space-ops before vs after.

Guardrail: a correction that edits more than --guardrail of the line's chars
is rejected (keep the OCR output) — hallucination protection per the
"No Free Lunches in OCR Post-Correction" findings.

Usage:
    python scripts/eval_correction.py --model-dir runs/arabart_corr/best \
        --tsv archive/correction/pairs_dev_real.tsv
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata

import numpy as np
import torch
from rapidfuzz.distance import Levenshtein

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_re_diac = re.compile(r"[ً-ْ]")


def norm_ar(s: str) -> str:
    """Same orthographic normalization as eval_val's CER(n)."""
    s = unicodedata.normalize("NFKC", s).replace("ـ", "")
    s = _re_diac.sub("", s)
    s = (s.replace("أ", "ا").replace("إ", "ا")
          .replace("آ", "ا").replace("ى", "ي"))
    return re.sub(r"\s+", " ", s).strip()


def load_rows(path: str) -> list[tuple[str, str]]:
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            clean = r.get("clean") or r.get("label") or ""
            noisy = r.get("noisy") if r.get("noisy") is not None else r.get("pred")
            if clean:
                rows.append((noisy or "", clean))
    return rows


def report(name: str, hyps: list[str], refs: list[str]) -> None:
    d = sum(Levenshtein.distance(r, h) for r, h in zip(refs, hyps))
    L = sum(len(r) for r in refs)
    dn = sum(Levenshtein.distance(norm_ar(r), norm_ar(h)) for r, h in zip(refs, hyps))
    Ln = sum(len(norm_ar(r)) for r in refs)
    perfect = sum(1 for r, h in zip(refs, hyps) if r == h)
    space = 0
    for r, h in zip(refs, hyps):
        for op in Levenshtein.editops(r, h):
            if ((op.tag == "replace" and (r[op.src_pos] == " " or h[op.dest_pos] == " "))
                    or (op.tag == "delete" and r[op.src_pos] == " ")
                    or (op.tag == "insert" and h[op.dest_pos] == " ")):
                space += 1
    print(f"{name:12s} CER={d/L*100:5.2f}%  CER(n)={dn/Ln*100:5.2f}%  "
          f"perfect={perfect}/{len(refs)}  space-ops={space}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--guardrail", type=float, default=0.15,
                    help="Reject corrections editing > this fraction of chars.")
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--out", default=None, help="Optional corrected TSV output")
    args = ap.parse_args()

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir).to(device).eval()

    rows = load_rows(args.tsv)
    noisy = [n for n, _ in rows]
    clean = [c for _, c in rows]
    print(f"{len(rows)} lines | guardrail {args.guardrail:.0%} | beams {args.beams}")

    corrected: list[str] = []
    rejected = 0
    with torch.no_grad():
        for i in range(0, len(noisy), args.batch):
            chunk = noisy[i:i + args.batch]
            enc = tokenizer(chunk, max_length=args.max_len, truncation=True,
                            padding=True, return_tensors="pt").to(device)
            gen = model.generate(**enc, num_beams=args.beams,
                                 max_length=args.max_len)
            outs = tokenizer.batch_decode(gen, skip_special_tokens=True)
            for src, out in zip(chunk, outs):
                edit = Levenshtein.distance(src, out) / max(len(src), 1)
                if edit > args.guardrail or not out.strip():
                    corrected.append(src)  # too aggressive -> keep OCR output
                    rejected += 1
                else:
                    corrected.append(out)

    print(f"guardrail rejections: {rejected}/{len(rows)}\n")
    report("uncorrected", noisy, clean)
    report("corrected", corrected, clean)

    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["noisy", "corrected", "clean"])
            w.writerows(zip(noisy, corrected, clean))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
