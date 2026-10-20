"""Build an external Arabic text corpus for LM decoding (arabicocr_khatt.lm).

Downloads Arabic Wikipedia shards (wikimedia/wikipedia, 20231101.ar, CC BY-SA)
and writes one sentence-ish chunk per line, Arabic-dominant only.
Test/val leakage filtering happens later in lm_decode_sweep.py.

Usage:
    python scripts/prepare_lm_corpus.py --shards 2 --out archive/lm/ar_wiki.txt
"""
import argparse
import os
import re

import pyarrow.parquet as pq  # NOTE: install pyarrow in a separate env — in the training venv it breaks "import pandas; import torch" (WinError 1114)
from huggingface_hub import hf_hub_download

_SPLIT = re.compile(r"(?<=[.!؟?:؛])\s+|\n+")
_AR = re.compile(r"[ء-ي]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=2, help="How many of the 7 parquet shards to use.")
    ap.add_argument("--out", default="archive/lm/ar_wiki.txt")
    ap.add_argument("--raw-dir", default="archive/lm/raw")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n_lines = n_chars = 0
    with open(args.out, "w", encoding="utf-8") as fo:
        for i in range(args.shards):
            path = hf_hub_download("wikimedia/wikipedia", f"20231101.ar/train-{i:05d}-of-00007.parquet",
                                   repo_type="dataset", local_dir=args.raw_dir)
            for batch in pq.ParquetFile(path).iter_batches(columns=["text"], batch_size=2000):
                for doc in batch.column(0).to_pylist():
                    for s in _SPLIT.split(doc):
                        s = " ".join(s.split())
                        if not 20 <= len(s) <= 300:
                            continue
                        if len(_AR.findall(s)) < 0.7 * len(s.replace(" ", "")):
                            continue
                        fo.write(s + "\n")
                        n_lines += 1; n_chars += len(s)
            print(f"shard {i}: {n_lines} lines, {n_chars/1e6:.1f}M chars so far", flush=True)


if __name__ == "__main__":
    main()
