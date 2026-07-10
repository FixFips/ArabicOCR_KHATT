# scripts/train_arabart_correction.py
"""Fine-tune AraBART as an OCR post-corrector (Option 3).

Trains moussaKam/AraBART (BART-base, 139M) on (noisy, clean) pairs from
build_correction_pairs.py. Model selection is by corrected CER on the REAL
dev pairs (val-split OCR predictions), not on synthetic noise.

Usage:
    python scripts/train_arabart_correction.py \
        --pairs archive/correction/pairs_train.tsv \
        --dev archive/correction/pairs_dev_real.tsv \
        --out-dir runs/arabart_corr
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from rapidfuzz.distance import Levenshtein

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class PairsDataset(Dataset):
    def __init__(self, path: str):
        self.rows: list[tuple[str, str]] = []
        with open(path, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                noisy, clean = r.get("noisy") or "", r.get("clean") or ""
                if clean:
                    self.rows.append((noisy, clean))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        noisy, clean = self.rows[i]
        return {"noisy": noisy, "clean": clean}


def cer(ref: str, hyp: str) -> float:
    return Levenshtein.distance(ref, hyp) / max(len(ref), 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", default="archive/correction/pairs_train.tsv")
    ap.add_argument("--dev", default="archive/correction/pairs_dev_real.tsv")
    ap.add_argument("--model", default="moussaKam/AraBART")
    ap.add_argument("--out-dir", default="runs/arabart_corr")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=192, help="max tokens per side")
    ap.add_argument("--eval-steps", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                              Seq2SeqTrainer, Seq2SeqTrainingArguments)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    print(f"{args.model}: {sum(p.numel() for p in model.parameters()):,} params")

    train_ds = PairsDataset(args.pairs)
    dev_ds = PairsDataset(args.dev)
    noop_cer = float(np.mean([cer(c, n) for n, c in dev_ds.rows]))
    print(f"train pairs: {len(train_ds):,} | dev pairs: {len(dev_ds):,} "
          f"| dev no-op CER (uncorrected): {noop_cer*100:.2f}%")

    def collate(batch):
        noisy = [b["noisy"] for b in batch]
        clean = [b["clean"] for b in batch]
        enc = tokenizer(noisy, text_target=clean, max_length=args.max_len,
                        truncation=True, padding=True, return_tensors="pt")
        enc["labels"][enc["labels"] == tokenizer.pad_token_id] = -100
        return enc

    def compute_metrics(eval_pred):
        preds, _ = eval_pred
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        hyps = tokenizer.batch_decode(preds, skip_special_tokens=True)
        cers = [cer(clean, hyp) for (noisy, clean), hyp in zip(dev_ds.rows, hyps)]
        return {"cer": float(np.mean(cers))}

    targs = Seq2SeqTrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch * 2,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        weight_decay=0.01,
        bf16=torch.cuda.is_available(),
        logging_steps=200,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=args.max_len,
        generation_num_beams=1,      # greedy during training evals (speed);
                                     # final eval uses beams via eval_correction.py
        dataloader_num_workers=args.workers,
        report_to=[],
        seed=42,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collate,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    best_dir = Path(args.out_dir) / "best"
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))
    final = trainer.evaluate()
    print(f"\nbest model saved to {best_dir}")
    print(f"dev corrected CER: {final['eval_cer']*100:.2f}%  "
          f"(no-op baseline {noop_cer*100:.2f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
