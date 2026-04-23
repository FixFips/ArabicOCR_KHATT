"""Replay a checkpoint over a split with multiple decode configurations.

Compares greedy, beam search at various widths, and beam search with
bigram-LM at various weights — all on the same checkpoint, same data,
same images. Prints a summary table of CER / WER / WER(n) / DotCER.

Usage:
    # Default: greedy + beam(10) + beam(10)+lm(0.3)
    python -m src.replay_checkpoint

    # Sweep beam widths (lm disabled)
    python -m src.replay_checkpoint --beam-widths 5,10,20 --no-lm

    # Sweep LM weights at fixed beam=10
    python -m src.replay_checkpoint --beam-widths 10 --lm-weights 0.0,0.1,0.3,0.5,0.7

    # Against cleaned val
    python -m src.replay_checkpoint --exclude runs/exp1/val_suspects.tsv
"""
import argparse
import os
import re
import sys
import tempfile
import time
import unicodedata as ud
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from .dataset import KHATTDataset, read_label
from .metrics import cer as cer_raw, wer as wer_raw, dot_group_cer
from .model import (CRNN, text_to_ids, ctc_greedy_decode, ctc_beam_decode,
                    build_bigram_lm)

IMAGES_DIR = "./archive/images"
SPLITS_DIR = "./archive/splits"
RUN_DIR    = "./runs/exp1"
CKPT_PATH  = os.path.join(RUN_DIR, "crnn_best.pt")
HEIGHT     = 96
MAX_W      = 1536
BATCH_SIZE = 16

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


class CRNNCollate:
    def __init__(self, char2id, unk_id):
        self.char2id = char2id
        self.unk_id = unk_id
        self.to_tensor = transforms.ToTensor()

    def __call__(self, batch):
        imgs, texts_orig = zip(*batch)
        imgs = torch.stack([self.to_tensor(im) for im in imgs], dim=0)
        return imgs, list(texts_orig)


def _parse_float_list(s):
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_int_list(s):
    return [int(x) for x in s.split(",") if x.strip()]


def _filter_split(csv_path, exclude_paths):
    df = pd.read_csv(csv_path)
    before = len(df)
    bad = set()
    for p in exclude_paths:
        excl = pd.read_csv(p, sep="\t")
        bad.update(excl["filename"].astype(str))
    df = df[~df["filename"].isin(bad)].reset_index(drop=True)
    print(f"--exclude: dropped {before - len(df)} / {before}")
    tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8")
    df.to_csv(tmp.name, index=False)
    tmp.close()
    return tmp.name


def _evaluate(decode_fn, refs, loader, device, id2char):
    """Run decode_fn (takes logits -> list[str]) over loader; return metrics + time."""
    vcer = []; vwer = []; vwer_n = []
    all_refs = []; all_hyps = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for imgs, texts_ref in loader:
            imgs = imgs.to(device, non_blocking=True)
            logits = None  # will be filled by decode_fn
            # decode_fn is a closure that runs the model and decodes
            hyps_rtl = decode_fn(imgs)
            for r, h in zip(texts_ref, hyps_rtl):
                vcer.append(cer_raw(r, h))
                vwer.append(wer_raw(r, h))
                all_refs.append(r); all_hyps.append(h)
            refs_n = [norm_ar(r) for r in texts_ref]
            hyps_n = [norm_ar(h) for h in hyps_rtl]
            for r, h in zip(refs_n, hyps_n):
                vwer_n.append(wer_raw(r, h))
    elapsed = time.perf_counter() - t0
    return {
        "cer": float(np.mean(vcer)) if vcer else 1.0,
        "wer": float(np.mean(vwer)) if vwer else 1.0,
        "wer_n": float(np.mean(vwer_n)) if vwer_n else 1.0,
        "dot_cer": dot_group_cer(all_refs, all_hyps),
        "time": elapsed,
        "samples": len(vcer),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=CKPT_PATH)
    ap.add_argument("--split", default="val", choices=["val", "test", "train"])
    ap.add_argument("--exclude", action="append", default=[])
    ap.add_argument("--beam-widths", default="10",
                    help="Comma-separated beam widths (default: 10). Use empty string to skip beam.")
    ap.add_argument("--lm-weights", default="0.3",
                    help="Comma-separated LM weights. Use 0.0 to effectively disable LM.")
    ap.add_argument("--no-lm", action="store_true", help="Skip all LM-weighted configs")
    ap.add_argument("--no-greedy", action="store_true", help="Skip greedy config")
    ap.add_argument("--out-tsv", default=None, help="Optional: save best-config predictions here")
    args = ap.parse_args()

    beam_widths = _parse_int_list(args.beam_widths) if args.beam_widths.strip() else []
    lm_weights  = _parse_float_list(args.lm_weights) if args.lm_weights.strip() else []

    csv_path = Path(SPLITS_DIR, f"{args.split}.csv")
    if not csv_path.exists():
        raise SystemExit(f"Split CSV not found: {csv_path}")
    if args.exclude:
        csv_path = Path(_filter_split(str(csv_path), args.exclude))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}{' ' + torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}")

    state = torch.load(args.ckpt, map_location=device, weights_only=False)
    vocab = state["vocab"]
    num_classes = len(vocab)
    char2id = {c: i for i, c in enumerate(vocab)}
    id2char = {i: c for i, c in enumerate(vocab)}
    unk_id = char2id.get("<unk>", 1)
    print(f"Checkpoint: {args.ckpt}  (arch_version={state.get('arch_version', '?')}, vocab={num_classes})")

    model = CRNN(num_classes).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    # Build bigram LM from the training split (unfiltered — LM is reusable)
    print("Building bigram LM from train split...")
    train_df = pd.read_csv(Path(SPLITS_DIR, "train.csv"))
    train_texts = []
    for _, row in train_df.iterrows():
        try:
            train_texts.append(read_label(row["label_path"]))
        except Exception:
            pass
    bigram_lm = build_bigram_lm(train_texts, char2id)
    print(f"  {len(bigram_lm)} entries")

    ds = KHATTDataset(csv_path, IMAGES_DIR, mode="crnn", crnn_h=HEIGHT, crnn_max_w=MAX_W)
    collate = CRNNCollate(char2id, unk_id)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
                        pin_memory=True, collate_fn=collate)
    print(f"Split: {args.split}  ({len(ds)} samples, {len(loader)} batches)\n")

    # Build the list of configs to run
    configs = []
    if not args.no_greedy:
        configs.append(("greedy", None))
    for bw in beam_widths:
        configs.append((f"beam({bw})", {"beam_width": bw, "bigram_lm": None, "lm_weight": 0.0}))
        if not args.no_lm:
            for lw in lm_weights:
                if lw == 0.0:
                    continue
                configs.append((f"beam({bw})+lm({lw})",
                                {"beam_width": bw, "bigram_lm": bigram_lm, "lm_weight": lw}))

    results = []
    for name, cfg in configs:
        print(f"  Running {name}...", end=" ", flush=True)
        if cfg is None:
            def run(imgs, m=model, i2c=id2char):
                return [h[::-1] for h in ctc_greedy_decode(m(imgs), i2c)]
        else:
            bw, lm, lw = cfg["beam_width"], cfg["bigram_lm"], cfg["lm_weight"]
            def run(imgs, m=model, i2c=id2char, bw=bw, lm=lm, lw=lw):
                return [h[::-1] for h in ctc_beam_decode(m(imgs), i2c,
                        beam_width=bw, bigram_lm=lm, lm_weight=lw)]
        r = _evaluate(run, None, loader, device, id2char)
        r["config"] = name
        results.append(r)
        print(f"CER={r['cer']*100:.2f}%  time={r['time']:.1f}s")

    # Summary table
    print("\n" + "=" * 72)
    print(f"{'Config':<22} {'CER':>8} {'WER':>8} {'WER(n)':>8} {'DotCER':>8} {'Time':>8}")
    print("-" * 72)
    best = min(results, key=lambda r: r["cer"])
    for r in results:
        mark = " *" if r is best else "  "
        print(f"{r['config']:<22} "
              f"{r['cer']*100:>7.2f}% {r['wer']*100:>7.2f}% "
              f"{r['wer_n']*100:>7.2f}% {r['dot_cer']*100:>7.2f}% "
              f"{r['time']:>7.1f}s{mark}")
    print("=" * 72)
    print(f"Best: {best['config']} — CER {best['cer']*100:.2f}%")


if __name__ == "__main__":
    main()
