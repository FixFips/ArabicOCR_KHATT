"""Tune LM-fused CTC decoding (arabicocr_khatt.lm) on a split.

1. Runs the checkpoint once over the split and caches log-probs (.npz).
2. Trains a char n-gram LM + word lexicon on KHATT train labels and,
   optionally, an external corpus (one sentence per line). Corpus lines that
   share any word 4-gram with val/test labels are dropped (no test leakage).
3. Decodes every grid point in parallel and prints CER / CER(n) / WER.

Usage:
    python scripts/lm_decode_sweep.py --ckpt runs/exp4_strict_s42/crnn_best.pt \
        --splits-dir archive/splits_strict --split val \
        --corpus archive/lm/ar_wiki.txt --order 6 \
        --alpha 0 0.3 0.5 0.8 --beta 0 0.5 1 2 --gamma 0 0.5 1
"""
import argparse
import itertools
import os
import pickle
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _lm_worker  # noqa: E402  (torch-free; the only import spawned workers need)

# Heavy imports (torch, the package) live in main(): on Windows every Pool
# worker re-imports this module, and 8+ extra torch/CUDA loads exhausted the
# commit limit next to a training run (WinError 1114).

IMAGES_DIR = "./archive/images"
_DIAC = re.compile(r"[ٌ-ْ]")
def cache_logprobs(ckpt, csv_path, out):
    if os.path.exists(out):
        d = np.load(out, allow_pickle=True)
        return list(d["lps"]), list(d["refs"])
    import torch
    from torch.utils.data import DataLoader
    from arabicocr_khatt.dataset import KHATTDataset
    from arabicocr_khatt.eval_val import CRNNCollate
    from arabicocr_khatt.model import CRNN, label_flipper
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(ckpt, map_location=device, weights_only=False)
    vocab = state["vocab"]
    char2id = {c: i for i, c in enumerate(vocab)}
    model = CRNN(len(vocab), use_attention=bool(state.get("use_attention", False))).to(device)
    model.load_state_dict(state["model"]); model.eval()
    ds = KHATTDataset(csv_path, IMAGES_DIR, mode="crnn", crnn_h=96, crnn_max_w=1536, strip_tatweel=True)
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0,
                    collate_fn=CRNNCollate(char2id, char2id["<unk>"], label_flipper(state)))
    lps, refs = [], []
    with torch.no_grad():
        for imgs, _, _, texts in dl:
            lp = model(imgs.to(device)).float().log_softmax(-1).cpu().numpy()  # [T,B,C]
            for b in range(lp.shape[1]):
                lps.append(lp[:, b].astype(np.float32)); refs.append(texts[b])
    np.savez_compressed(out, lps=np.array(lps, dtype=object), refs=np.array(refs, dtype=object))
    return lps, refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--splits-dir", default="./archive/splits")
    ap.add_argument("--split", default="val")
    ap.add_argument("--corpus", default=None, help="External text, one sentence per line (UTF-8).")
    ap.add_argument("--corpus-max-chars", type=int, default=30_000_000)
    ap.add_argument("--no-indomain", action="store_true", help="Don't add KHATT train labels to the LM.")
    ap.add_argument("--order", type=int, default=6)
    ap.add_argument("--alpha", type=float, nargs="+", default=[0.0, 0.3, 0.5, 0.8])
    ap.add_argument("--beta", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    ap.add_argument("--gamma", type=float, nargs="+", default=[0.0])
    ap.add_argument("--oov", type=float, nargs="+", default=[3.0])
    ap.add_argument("--beam", type=int, default=10)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--work", default="./runs/lm_cache")
    ap.add_argument("--out-tsv", default=None, help="Write per-line preds of the best config.")
    args = ap.parse_args()

    import pandas as pd
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from arabicocr_khatt.dataset import read_label
    from arabicocr_khatt.metrics import cer, cer_norm, wer, norm_ar
    from arabicocr_khatt.model import label_flipper
    CharNgramLM, WordLexicon = _lm_worker.lm.CharNgramLM, _lm_worker.lm.WordLexicon

    os.makedirs(args.work, exist_ok=True)
    tag = Path(args.ckpt).parent.name + "_" + Path(args.splits_dir).name + "_" + args.split
    csv_path = Path(args.splits_dir, f"{args.split}.csv")
    lps, refs = cache_logprobs(args.ckpt, csv_path, os.path.join(args.work, f"lp_{tag}.npz"))

    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    flip = label_flipper(state)
    id2char = {i: c for i, c in enumerate(state["vocab"])}
    vocab_chars = set(state["vocab"])

    # ---- LM training text (decode direction) ----
    texts = []
    if not args.no_indomain:
        tr = pd.read_csv(Path(args.splits_dir, "train.csv"))
        texts += [read_label(p).replace("ـ", "") for p in tr["label_path"]]
    n_in = len(texts)
    if args.corpus:
        held = set()
        for s in ["val", "test"]:
            for p in pd.read_csv(Path(args.splits_dir, f"{s}.csv"))["label_path"]:
                w = norm_ar(read_label(p)).split()
                held.update(tuple(w[i:i + 4]) for i in range(len(w) - 3))
        n_chars = dropped = 0
        with open(args.corpus, encoding="utf-8") as f:
            for line in f:
                # KHATT writers rarely mark harakat; Wikipedia does. Keep only
                # tanween-fath (common in handwriting, e.g. "أيضاً").
                line = _DIAC.sub("", line.replace("ـ", ""))
                line = " ".join(line.split())
                if len(line) < 10 or any(ch not in vocab_chars for ch in line):
                    continue
                w = norm_ar(line).split()
                if any(tuple(w[i:i + 4]) in held for i in range(len(w) - 3)):
                    dropped += 1
                    continue
                texts.append(line); n_chars += len(line)
                if n_chars >= args.corpus_max_chars:
                    break
        print(f"corpus: {len(texts) - n_in} lines / {n_chars/1e6:.1f}M chars (dropped {dropped} overlapping val/test)")
    flipped = [flip(t) for t in texts]
    t0 = time.time()
    clm = CharNgramLM(args.order).fit(flipped)
    wlm = WordLexicon().fit(flipped)
    print(f"LM: {n_in} in-domain + {len(texts) - n_in} external lines, order {args.order}, "
          f"{len(wlm.logp)} lexicon words, built in {time.time() - t0:.0f}s")

    lm_path = os.path.join(args.work, f"workerstate_{os.getpid()}.pkl")
    with open(lm_path, "wb") as f:
        pickle.dump({"lps": lps, "id2char": id2char, "clm": clm, "wlm": wlm, "beam": args.beam,
                     "label_order": state.get("label_order", "reverse")}, f)

    grid = [None] + [dict(alpha=a, beta=b, gamma=g, oov=o) for a, b, g, o in
                     itertools.product(args.alpha, args.beta, args.gamma, args.oov)]
    results = []
    with Pool(args.procs, initializer=_lm_worker.init, initargs=(lm_path,)) as pool:
        for cfg in grid:
            t0 = time.time()
            hyps = pool.map(_lm_worker.decode, [(i, cfg) for i in range(len(lps))], chunksize=8)
            m = (np.mean([cer(r, h) for r, h in zip(refs, hyps)]),
                 np.mean([cer_norm(r, h) for r, h in zip(refs, hyps)]),
                 np.mean([wer(r, h) for r, h in zip(refs, hyps)]))
            results.append((cfg, m, hyps))
            name = "greedy" if cfg is None else " ".join(f"{k}={v}" for k, v in cfg.items())
            print(f"{name:45s} CER={m[0]:.4f} CER(n)={m[1]:.4f} WER={m[2]:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    os.unlink(lm_path)

    best = min(results, key=lambda r: r[1][1])
    print(f"\nBEST by CER(n): {best[0]}  CER={best[1][0]:.4f} CER(n)={best[1][1]:.4f} WER={best[1][2]:.4f}")
    if args.out_tsv:
        with open(args.out_tsv, "w", encoding="utf-8") as f:
            f.write("label\tpred\n")
            for r, h in zip(refs, best[2]):
                f.write(f"{r}\t{h}\n")


if __name__ == "__main__":
    main()
