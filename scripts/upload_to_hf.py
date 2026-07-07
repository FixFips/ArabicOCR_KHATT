#!/usr/bin/env python
"""Upload the trained checkpoint, bigram LM and model card to the Hugging Face Hub.

Run this on the training machine (where runs/exp1/crnn_best.pt and the KHATT
archive live):

    pip install -e .
    hf auth login          # or: huggingface-cli login
    python scripts/upload_to_hf.py --run-dir runs/exp1 --repo-id FixFips/arabicocr-khatt

What it does:
  1. Validates the checkpoint (vocab + arch_version 2).
  2. Builds the Arabic character bigram LM from the training split (if present)
     and serializes it to bigram_lm.json.
  3. Fills the model card template with the best-epoch metrics from metrics.csv.
  4. Creates the Hub repo (if needed) and uploads everything.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # allow running without pip install

from arabicocr_khatt.pipeline import load_checkpoint, save_bigram_lm_json  # noqa: E402


def best_metrics_row(csv_path: Path):
    """Return the row with the lowest validation CER, or None."""
    if not csv_path.exists():
        return None
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("cer")]
    if not rows:
        return None
    return min(rows, key=lambda r: float(r["cer"]))


def build_lm(train_csv: Path, char2id: dict):
    from arabicocr_khatt.dataset import read_label
    from arabicocr_khatt.model import build_bigram_lm

    texts = []
    with open(train_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                texts.append(read_label(row["label_path"]))
            except Exception:
                pass
    if not texts:
        return None
    return build_bigram_lm(texts, char2id)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", default="runs/exp1", help="training run directory")
    parser.add_argument("--repo-id", default="FixFips/arabicocr-khatt")
    parser.add_argument("--train-csv", default="archive/splits/train.csv",
                        help="training split CSV used to build the bigram LM")
    parser.add_argument("--private", action="store_true", help="create the repo as private")
    parser.add_argument("--dry-run", action="store_true", help="prepare files but do not upload")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    ckpt_path = run_dir / "crnn_best.pt"
    if not ckpt_path.exists():
        print(f"error: checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 1

    print(f"Validating {ckpt_path} ...")
    state = load_checkpoint(ckpt_path)
    vocab = state["vocab"]
    char2id = {c: i for i, c in enumerate(vocab)}
    print(f"  OK — vocab size {len(vocab)}, arch_version {state['arch_version']}")

    # ---- bigram LM ----
    lm_path = run_dir / "bigram_lm.json"
    train_csv = Path(args.train_csv)
    if train_csv.exists():
        print(f"Building bigram LM from {train_csv} ...")
        lm = build_lm(train_csv, char2id)
        if lm:
            save_bigram_lm_json(lm, lm_path)
            print(f"  OK — {len(lm)} entries -> {lm_path}")
    else:
        print(f"  skipping bigram LM ({train_csv} not found)")

    # ---- model card ----
    template = (REPO_ROOT / "scripts" / "model_card_template.md").read_text(encoding="utf-8")
    row = best_metrics_row(run_dir / "metrics.csv")

    def pct(key):
        return f"{float(row[key]) * 100:.2f}%" if row and row.get(key) else "TBD"

    card = template.format(
        repo_id=args.repo_id,
        epoch=row["epoch"] if row else "TBD",
        cer=pct("cer"), wer=pct("wer"), wer_norm=pct("wer_norm"), dot_cer=pct("dot_cer"),
    )
    card_path = run_dir / "README.md"
    card_path.write_text(card, encoding="utf-8")
    print(f"Model card written to {card_path}"
          + ("" if row else "  (metrics.csv missing — fill in TBD values!)"))

    if args.dry_run:
        print("--dry-run: skipping upload.")
        return 0

    # ---- upload ----
    from huggingface_hub import HfApi

    api = HfApi()
    repo_url = api.create_repo(args.repo_id, exist_ok=True, private=args.private)
    print(f"Uploading to {repo_url} ...")
    uploads = [(ckpt_path, "crnn_best.pt"), (card_path, "README.md")]
    if lm_path.exists():
        uploads.append((lm_path, "bigram_lm.json"))
    charset = REPO_ROOT / "arabicocr_khatt" / "charset_arabic.txt"
    if charset.exists():
        uploads.append((charset, "charset_arabic.txt"))
    for local, remote in uploads:
        print(f"  {local} -> {remote}")
        api.upload_file(path_or_fileobj=str(local), path_in_repo=remote, repo_id=args.repo_id)

    print(f"\nDone: https://huggingface.co/{args.repo_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
