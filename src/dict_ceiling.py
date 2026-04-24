"""Measure the oracle CER ceiling from a word-dictionary correction step.

Builds a vocabulary from train labels, then examines val predictions and asks:
how much CER could we save if we could perfectly spell-correct non-word
predictions back to the nearest in-vocab word?

This gives an *upper bound* on what any lexicon-constrained decoder could
achieve — no fancy tooling (no morphological analyzer, no context scoring).

Usage:
    python -m src.dict_ceiling
    python -m src.dict_ceiling --tsv runs/exp1/val_epoch_999_samples.tsv
"""
import argparse
import csv
import os
import re
import sys
import unicodedata as ud
from collections import Counter
from pathlib import Path

import pandas as pd
from rapidfuzz import process, distance as _dist
from rapidfuzz.distance import Levenshtein

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_re_diac = re.compile(r"[\u064B-\u0652]")
def norm_ar(s: str) -> str:
    s = ud.normalize("NFKC", s)
    s = s.replace("\u0640", "")
    s = _re_diac.sub("", s)
    s = (s.replace("\u0623", "\u0627")
           .replace("\u0625", "\u0627")
           .replace("\u0622", "\u0627")
           .replace("\u0649", "\u064A"))
    return re.sub(r"\s+", " ", s).strip()


def read_label(path):
    for enc in ["windows-1256", "utf-8", "utf-8-sig"]:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except Exception:
            pass
    return ""


def is_word_token(tok: str) -> bool:
    """True if the token looks like an Arabic word we'd want to check against a dict.
    Excludes bare punctuation, digits, single letters, empties."""
    if not tok or len(tok) < 2:
        return False
    if tok.isdigit():
        return False
    # Must contain at least one Arabic letter
    if not any("\u0600" <= c <= "\u06FF" for c in tok):
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", default="runs/exp1/val_epoch_999_samples.tsv",
                    help="Val predictions TSV from eval_val.py")
    ap.add_argument("--train-csv", default="archive/splits/train.csv")
    ap.add_argument("--topk", type=int, default=1,
                    help="Consider top-K dict candidates by edit distance; mark fixable if any match GT")
    ap.add_argument("--normalize", action="store_true",
                    help="Apply Arabic orthographic normalization (ى→ي, alef-variants, diacritics) before comparing")
    args = ap.parse_args()

    # --- Build vocabulary from train labels ---
    train_df = pd.read_csv(args.train_csv)
    vocab_counter = Counter()
    for _, row in train_df.iterrows():
        text = read_label(row["label_path"])
        if args.normalize:
            text = norm_ar(text)
        for tok in text.split():
            if is_word_token(tok):
                vocab_counter[tok] += 1
    vocab = set(vocab_counter.keys())
    print(f"Train vocabulary: {len(vocab)} unique word types (min-len 2, Arabic letters, tokens only)")
    print(f"  token occurrences: {sum(vocab_counter.values())}")
    print(f"  top-10 most frequent: {vocab_counter.most_common(10)}")
    print(f"  normalization: {'ON (ى→ي, alef variants, diacritics)' if args.normalize else 'OFF'}")

    vocab_list = list(vocab)  # for rapidfuzz.process

    # --- Scan val predictions ---
    sub_count = 0            # number of word-level substitutions
    sub_nonword = 0          # subs where PR is not in vocab (dict *could* help)
    sub_valid_word = 0       # subs where PR IS in vocab (dict can't help — model produced a real different word)
    fixable = 0              # subs where PR→nearest-in-vocab == GT
    chars_saved = 0          # character edits saved by perfect correction
    total_char_edits = 0     # total char edits across all val samples (for CER context)
    total_chars = 0          # total GT chars
    gt_word_tokens = 0
    gt_word_oov = 0          # GT words not in train vocab (upper bound on coverage)

    with open(args.tsv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            gt = row["label"] or ""
            pr = row["pred"]  or ""
            if args.normalize:
                gt = norm_ar(gt)
                pr = norm_ar(pr)

            total_char_edits += Levenshtein.distance(gt, pr)
            total_chars      += len(gt)

            gt_tokens = gt.split()
            pr_tokens = pr.split()

            for t in gt_tokens:
                if is_word_token(t):
                    gt_word_tokens += 1
                    if t not in vocab:
                        gt_word_oov += 1

            # Align tokens via word-level editops
            ops = Levenshtein.editops(gt_tokens, pr_tokens)
            for op, i1, i2 in ops:
                if op != "replace":
                    continue
                gt_w = gt_tokens[i1]
                pr_w = pr_tokens[i2]
                if not is_word_token(gt_w) or not is_word_token(pr_w):
                    continue
                sub_count += 1
                if pr_w in vocab:
                    sub_valid_word += 1
                    continue
                sub_nonword += 1
                # Find nearest in-vocab word to pr_w (bounded edit distance)
                candidates = process.extract(
                    pr_w, vocab_list, scorer=_dist.Levenshtein.distance,
                    limit=args.topk,
                )
                # Candidates are (word, distance, index)
                if any(c[0] == gt_w for c in candidates):
                    fixable += 1
                    # chars saved = dist(gt,pr) since the corrected pr would match gt
                    chars_saved += Levenshtein.distance(gt_w, pr_w)

    # --- Report ---
    cer_now    = total_char_edits / max(total_chars, 1)
    cer_oracle = (total_char_edits - chars_saved) / max(total_chars, 1)

    print()
    print(f"=== GT coverage (what a perfect dict could possibly fix) ===")
    print(f"GT word-tokens (Arabic, len>=2):  {gt_word_tokens}")
    print(f"GT tokens OOV vs train vocab:      {gt_word_oov}  ({100*gt_word_oov/max(gt_word_tokens,1):.1f}%)")
    print(f"    → floor: any unigram dictionary misses these by design")

    print()
    print(f"=== Word-level substitutions (GT ≠ PR) ===")
    print(f"Substitutions total:                 {sub_count}")
    print(f"  PR is a valid vocab word:          {sub_valid_word}  (dict can't help — model picked a real wrong word)")
    print(f"  PR is a non-word:                  {sub_nonword}  (dict *could* help if nearest match == GT)")
    if sub_nonword:
        print(f"  of those non-word subs, fixable by top-{args.topk} lookup: {fixable}  ({100*fixable/sub_nonword:.1f}%)")

    print()
    print(f"=== Character-level CER impact ===")
    print(f"Total char edits today:              {total_char_edits}")
    print(f"Char edits saved by oracle:          {chars_saved}  ({100*chars_saved/max(total_char_edits,1):.2f}% of all edits)")
    print(f"CER now:                             {cer_now*100:.4f}%")
    print(f"CER with oracle top-{args.topk} dict correction: {cer_oracle*100:.4f}%   (saves {(cer_now-cer_oracle)*10000:.1f} bp)")
    print()
    print(f"  This is an ORACLE ceiling — real dictionary correction will be lower.")
    print(f"  In practice you'd also need: morphology-aware matching, beam scoring, OOV handling.")


if __name__ == "__main__":
    main()
