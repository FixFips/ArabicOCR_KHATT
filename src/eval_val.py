"""Standalone val-pass evaluator for an existing CRNN checkpoint.

Loads runs/exp1/crnn_best.pt, runs greedy CTC decode over the val split,
prints CER/WER/WER(n)/DotCER, and writes a full samples TSV.

Usage:
    python -m src.eval_val
"""
import os
import re
import time
import argparse
import tempfile
import unicodedata as ud
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from .dataset import KHATTDataset
from .metrics import cer as cer_raw, wer as wer_raw, dot_group_cer
from .model import CRNN, text_to_ids, ids_to_text, ctc_greedy_decode

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
    s = re.sub(r"\s+", " ", s).strip()
    return s


class CRNNCollate:
    def __init__(self, char2id, unk_id):
        self.char2id = char2id
        self.unk_id = unk_id
        self.to_tensor = transforms.ToTensor()

    def __call__(self, batch):
        imgs, texts_orig = zip(*batch)
        texts_ltr = [t[::-1] for t in texts_orig]
        imgs = torch.stack([self.to_tensor(im) for im in imgs], dim=0)
        targets = [torch.tensor(text_to_ids(t, self.char2id, self.unk_id), dtype=torch.long) for t in texts_ltr]
        target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
        targets = torch.cat(targets) if len(targets) else torch.tensor([], dtype=torch.long)
        return imgs, targets, target_lengths, list(texts_orig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val", choices=["val", "test", "train"])
    parser.add_argument("--ckpt", default=CKPT_PATH)
    parser.add_argument("--out", default=None, help="TSV output path (default: RUN_DIR/eval_<split>_samples.tsv)")
    parser.add_argument("--exclude", default=None,
                        help="TSV with a 'filename' column (e.g. runs/exp1/val_suspects.tsv) whose rows to skip")
    args = parser.parse_args()

    csv_path = Path(SPLITS_DIR, f"{args.split}.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {csv_path}. Run training once to build splits.")

    # If --exclude is set, write a filtered split CSV to a temp file and use it
    tmp_csv = None
    if args.exclude:
        excl_df = pd.read_csv(args.exclude, sep="\t")
        if "filename" not in excl_df.columns:
            raise SystemExit(f"--exclude TSV has no 'filename' column: {args.exclude}")
        bad = set(excl_df["filename"].astype(str))
        split_df = pd.read_csv(csv_path)
        before = len(split_df)
        split_df = split_df[~split_df["filename"].isin(bad)].reset_index(drop=True)
        dropped = before - len(split_df)
        print(f"--exclude: dropped {dropped} / {before} samples from {args.split} split")
        tmp_csv = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8")
        split_df.to_csv(tmp_csv.name, index=False)
        tmp_csv.close()
        csv_path = Path(tmp_csv.name)

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

    ds = KHATTDataset(csv_path, IMAGES_DIR, mode="crnn", crnn_h=HEIGHT, crnn_max_w=MAX_W)
    collate = CRNNCollate(char2id, unk_id)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
                        pin_memory=True, collate_fn=collate)
    filenames = pd.read_csv(csv_path)["filename"].tolist()
    print(f"Split: {args.split}  ({len(ds)} samples, {len(loader)} batches)")

    vcer = []; vwer = []; vwer_norm = []
    all_refs = []; all_hyps = []
    samples_to_save = []

    t0 = time.perf_counter()
    with torch.no_grad():
        for imgs, _, _, texts_ref in loader:
            imgs = imgs.to(device, non_blocking=True)
            logits = model(imgs)
            hyps_ltr = ctc_greedy_decode(logits, id2char)
            hyps_rtl = [h[::-1] for h in hyps_ltr]

            for r, h in zip(texts_ref, hyps_rtl):
                vcer.append(cer_raw(r, h))
                vwer.append(wer_raw(r, h))
                all_refs.append(r); all_hyps.append(h)
                samples_to_save.append((r, h))

            refs_n = [norm_ar(r) for r in texts_ref]
            hyps_n = [norm_ar(h) for h in hyps_rtl]
            for r, h in zip(refs_n, hyps_n):
                vwer_norm.append(wer_raw(r, h))

    mcer   = float(np.mean(vcer)) if vcer else 1.0
    mwer   = float(np.mean(vwer)) if vwer else 1.0
    mwer_n = float(np.mean(vwer_norm)) if vwer_norm else 1.0
    mdot   = dot_group_cer(all_refs, all_hyps)
    elapsed = time.perf_counter() - t0

    default_name = "val_epoch_999_samples.tsv" if args.split == "val" else f"eval_{args.split}_samples.tsv"
    out_path = args.out or os.path.join(RUN_DIR, default_name)
    if len(filenames) != len(samples_to_save):
        print(f"WARN: filename count ({len(filenames)}) != sample count ({len(samples_to_save)}); omitting filenames")
        filenames = [""] * len(samples_to_save)
    with open(out_path, "w", encoding="utf-8") as fo:
        fo.write("filename\tlabel\tpred\n")
        for fn, (r, h) in zip(filenames, samples_to_save):
            fo.write(f"{fn}\t{r.replace(chr(9),' ').replace(chr(10),' ')}\t{h.replace(chr(9),' ').replace(chr(10),' ')}\n")

    print("")
    print(f"CER     = {mcer:.4f}")
    print(f"WER     = {mwer:.4f}")
    print(f"WER(n)  = {mwer_n:.4f}")
    print(f"DotCER  = {mdot:.4f}")
    print(f"Samples = {len(samples_to_save)}  |  Time = {elapsed:.1f}s")
    print(f"Saved TSV: {out_path}")

    if tmp_csv is not None:
        try:
            os.unlink(tmp_csv.name)
        except OSError:
            pass


if __name__ == "__main__":
    main()
