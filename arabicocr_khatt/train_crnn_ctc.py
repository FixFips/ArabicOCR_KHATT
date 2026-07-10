import argparse
import os
import re
import sys
import math
import tempfile
import unicodedata as ud
import time
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path

# Windows console defaults to cp1252 and crashes when printing Arabic sample lines.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

from .dataset import KHATTDataset
from .metrics import cer as cer_raw, wer as wer_raw, dot_group_cer
from .model import CRNN, text_to_ids, ids_to_text, ctc_greedy_decode, ctc_beam_decode
from .augment import ArabicAugment

# ---------------- Config ----------------
IMAGES_DIR = "./archive/images"
LABELS_DIR = "./archive/labels"
SPLITS_DIR = "./archive/splits"
RUN_DIR    = "./runs/exp1"

HEIGHT     = 96       # was 64 — Arabic needs 7 vertical zones for dots/diacritics
MAX_W      = 1536     # was 1024 — at H=96 mean image width becomes ~1416px
BATCH_SIZE = 16       # was 32 — compensated by gradient accumulation
GRAD_ACCUM = 2        # effective batch = 16 × 2 = 32
EPOCHS     = 120      # was 70 — augmentation + larger model needs more time
LR         = 1e-3
SEED       = 42
PATIENCE   = 15       # early stopping patience (epochs without CER improvement)

CHARSET_PATH = os.path.join(os.path.dirname(__file__), "charset_arabic.txt")

SHOW_SAMPLES_PRINT = 5

LOG_FILE  = os.path.join(RUN_DIR, "metrics.csv")
CKPT_PATH = os.path.join(RUN_DIR, "crnn_best.pt")

# ------------- Utils -------------
def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _load_exclude_filenames(paths):
    """Read one or more TSVs with a 'filename' column; return the union set."""
    bad = set()
    for p in paths:
        df = pd.read_csv(p, sep="\t")
        if "filename" not in df.columns:
            raise SystemExit(f"--exclude TSV has no 'filename' column: {p}")
        bad.update(df["filename"].astype(str).tolist())
    return bad


def _filter_split_csv(src_csv: str, bad_filenames: set) -> str:
    """Write a filtered copy of src_csv (rows whose filename is in bad_filenames removed).
    Returns path to a temp CSV file."""
    df = pd.read_csv(src_csv)
    before = len(df)
    df = df[~df["filename"].isin(bad_filenames)].reset_index(drop=True)
    dropped = before - len(df)
    tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8")
    df.to_csv(tmp.name, index=False)
    tmp.close()
    print(f"  {os.path.basename(src_csv)}: dropped {dropped} / {before}")
    return tmp.name

def build_splits():
    rows = []
    for fname in os.listdir(IMAGES_DIR):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            label_file = os.path.splitext(fname)[0] + ".txt"
            label_path = os.path.join(LABELS_DIR, label_file)
            if os.path.exists(label_path):
                rows.append({"filename": fname, "label_path": label_path})
    df = pd.DataFrame(rows)
    train_df, temp = train_test_split(df, test_size=0.2, random_state=SEED, shuffle=True)
    val_df, test_df = train_test_split(temp, test_size=0.5, random_state=SEED, shuffle=True)
    os.makedirs(SPLITS_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(SPLITS_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(SPLITS_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(SPLITS_DIR, "test.csv"), index=False)

def load_charset(path):
    vocab = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.rstrip("\n")
            if not s:
                continue
            if s.lstrip().startswith("# "):  # only "# text" is a comment, bare "#" is the character
                continue
            vocab.append(s)
    required = ["<pad>", "<unk>", "<bos>", "<eos>", " "]
    have = {t: (t in vocab) for t in required}
    for t in reversed(required):
        if not have[t]:
            vocab.insert(0, t)
        else:
            vocab.remove(t); vocab.insert(0, t)
    char2id = {c: i for i, c in enumerate(vocab)}
    id2char = {i: c for i, c in enumerate(vocab)}
    return vocab, char2id, id2char, char2id["<unk>"]

# ---------- Arabic normalization for WER ----------
_re_diac = re.compile(r"[\u064B-\u0652]")
def norm_ar(s: str) -> str:
    s = ud.normalize("NFKC", s)
    s = s.replace("\u0640", "")   # tatweel
    s = _re_diac.sub("", s)       # diacritics
    s = (s.replace("\u0623", "\u0627")   # أ → ا
           .replace("\u0625", "\u0627")   # إ → ا
           .replace("\u0622", "\u0627")   # آ → ا
           .replace("\u0649", "\u064A"))  # ى → ي
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ---- CSV helpers ----
def _init_metrics_file():
    os.makedirs(RUN_DIR, exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", encoding="utf-8", newline="") as f:
            f.write("timestamp,epoch,train_loss,cer,wer,wer_norm,dot_cer,train_batches,val_batches,ckpt_saved,lr\n")

def _append_metrics(epoch, train_loss, cer, wer, wer_n, dot_cer, train_batches, val_batches, ckpt_saved, lr):
    with open(LOG_FILE, "a", encoding="utf-8", newline="") as f:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{ts},{epoch},{train_loss:.6f},{cer:.6f},{wer:.6f},{wer_n:.6f},{dot_cer:.6f},{train_batches},{val_batches},{int(ckpt_saved)},{lr:.2e}\n")

# ------------- Collate (must be top-level for Windows multiprocessing) -------------
class CRNNCollate:
    """Picklable collate function for Windows DataLoader workers."""
    def __init__(self, char2id, unk_id):
        self.char2id = char2id
        self.unk_id = unk_id
        self.to_tensor = transforms.ToTensor()

    def __call__(self, batch):
        imgs, texts_orig = zip(*batch)
        texts_ltr = [t[::-1] for t in texts_orig]     # LTR time-axis
        imgs = torch.stack([self.to_tensor(im) for im in imgs], dim=0)
        targets = [torch.tensor(text_to_ids(t, self.char2id, self.unk_id), dtype=torch.long) for t in texts_ltr]
        target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
        targets = torch.cat(targets) if len(targets) else torch.tensor([], dtype=torch.long)
        return imgs, targets, target_lengths, list(texts_orig)

# ------------- Train -------------
def main():
    global RUN_DIR, LOG_FILE, CKPT_PATH
    parser = argparse.ArgumentParser(description="Train Arabic CRNN-CTC on KHATT")
    parser.add_argument("--exclude", action="append", default=[],
                        help="TSV with 'filename' column; rows are dropped from all splits. Repeatable.")
    parser.add_argument("--run-dir", default=RUN_DIR,
                        help="Output directory for this training run (default: ./runs/exp1).")
    parser.add_argument("--archive-old", action="store_true",
                        help="If --run-dir already contains a metrics.csv, rename the old dir "
                             "to <dir>_<timestamp> before starting a fresh run.")
    parser.add_argument("--attention", action="store_true",
                        help="Arch v3: add one transformer encoder layer after BiLSTM for global context.")
    parser.add_argument("--synth-csv", default=None,
                        help="Synthetic data CSV (filename,label,...); images resolved in "
                             "<csv dir>/images. Mixed into training via --synth-ratio.")
    parser.add_argument("--synth-ratio", type=float, default=0.35,
                        help="Expected fraction of synthetic samples per batch (default 0.35).")
    parser.add_argument("--warm-start", default=None,
                        help="Checkpoint to initialize model weights from (e.g. runs/exp1/crnn_best.pt).")
    parser.add_argument("--max-lr", type=float, default=3e-3,
                        help="OneCycleLR max_lr (default 3e-3; use 3e-4 for warm-start, 1e-5 for fine-tune).")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Number of epochs (default {EPOCHS}).")
    parser.add_argument("--save-topk", type=int, default=1,
                        help="Keep the K best checkpoints by val CER as crnn_epNNN.pt "
                             "(for weight averaging). crnn_best.pt is always maintained.")
    parser.add_argument("--amp", action="store_true",
                        help="bf16 autocast for model forward passes (CTC loss stays fp32). "
                             "~1.5-2x faster epochs on RTX 5080; use for new long runs, not "
                             "mid-experiment recipe comparisons.")
    parser.add_argument("--workers", type=int, default=4,
                        help="DataLoader workers (default 4, persistent). Smooths JPG-decode/"
                             "augment stalls; set 2 to reproduce the exp1/exp2 recipe exactly.")
    parser.add_argument("--strip-tatweel", action="store_true",
                        help="Strip tatweel (U+0640) from all labels. The model never learns to "
                             "emit it (109 forced deletions on train in run 2); removing it removes "
                             "label noise. Raw CER is then not directly comparable to older runs — "
                             "compare with CER(n) from eval_val.")
    args = parser.parse_args()

    # Apply --run-dir override to module globals so downstream code picks it up
    RUN_DIR = args.run_dir
    LOG_FILE = os.path.join(RUN_DIR, "metrics.csv")
    CKPT_PATH = os.path.join(RUN_DIR, "crnn_best.pt")

    # Archive any existing run before we start overwriting it
    if args.archive_old and os.path.exists(os.path.join(RUN_DIR, "metrics.csv")):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        archived = f"{RUN_DIR.rstrip(os.sep)}_{ts}"
        os.rename(RUN_DIR, archived)
        print(f"Archived previous run: {RUN_DIR} -> {archived}")

    os.makedirs(RUN_DIR, exist_ok=True)
    _init_metrics_file()
    set_seed(SEED)
    if not Path(SPLITS_DIR, "train.csv").exists():
        build_splits()

    # Resolve split CSV paths, filtering if --exclude was given
    train_csv = Path(SPLITS_DIR, "train.csv")
    val_csv   = Path(SPLITS_DIR, "val.csv")
    test_csv  = Path(SPLITS_DIR, "test.csv")
    tmp_csvs = []
    if args.exclude:
        bad = _load_exclude_filenames(args.exclude)
        print(f"Excluding {len(bad)} filenames from all splits (from {len(args.exclude)} TSV(s)):")
        train_csv = Path(_filter_split_csv(str(train_csv), bad)); tmp_csvs.append(train_csv)
        val_csv   = Path(_filter_split_csv(str(val_csv),   bad)); tmp_csvs.append(val_csv)
        if test_csv.exists():
            test_csv = Path(_filter_split_csv(str(test_csv), bad)); tmp_csvs.append(test_csv)

    vocab, char2id, id2char, unk_id = load_charset(CHARSET_PATH)
    num_classes = len(vocab)
    print(f"Vocabulary: {num_classes} classes")

    # --- Data augmentation (training only) ---
    aug = ArabicAugment(training=True)

    train_ds = KHATTDataset(
        train_csv, IMAGES_DIR,
        mode="crnn", crnn_h=HEIGHT, crnn_max_w=MAX_W, augment=aug,
        strip_tatweel=args.strip_tatweel,
    )
    val_ds = KHATTDataset(
        val_csv, IMAGES_DIR,
        mode="crnn", crnn_h=HEIGHT, crnn_max_w=MAX_W,
        strip_tatweel=args.strip_tatweel,
    )

    collate = CRNNCollate(char2id, unk_id)

    # --- Optional synthetic mixing (Option A) ---
    # ConcatDataset + WeightedRandomSampler so each batch is an iid mix with
    # P(synth) = --synth-ratio. num_samples keeps the expected number of REAL
    # samples per epoch at len(train_ds), so "epoch" stays comparable.
    if args.synth_csv:
        from torch.utils.data import ConcatDataset, WeightedRandomSampler
        synth_dir = os.path.dirname(os.path.abspath(args.synth_csv))
        synth_ds = KHATTDataset(
            args.synth_csv, os.path.join(synth_dir, "images"),
            mode="crnn", crnn_h=HEIGHT, crnn_max_w=MAX_W, augment=aug,
            strip_tatweel=args.strip_tatweel,
        )
        r = args.synth_ratio
        if not 0.0 < r < 1.0:
            raise SystemExit(f"--synth-ratio must be in (0,1), got {r}")
        weights = torch.cat([
            torch.full((len(train_ds),), (1.0 - r) / len(train_ds), dtype=torch.double),
            torch.full((len(synth_ds),), r / len(synth_ds), dtype=torch.double),
        ])
        num_samples = int(round(len(train_ds) / (1.0 - r)))
        sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)
        mixed_ds = ConcatDataset([train_ds, synth_ds])
        train_loader = DataLoader(mixed_ds, batch_size=BATCH_SIZE, sampler=sampler,
                                  num_workers=args.workers, pin_memory=True,
                                  persistent_workers=args.workers > 0, collate_fn=collate)
        print(f"Mixed training: {len(train_ds)} real + {len(synth_ds)} synth, "
              f"P(synth)={r:.2f}, {num_samples} samples/epoch "
              f"({math.ceil(num_samples / BATCH_SIZE)} batches)")
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=args.workers, pin_memory=True,
                                  persistent_workers=args.workers > 0, collate_fn=collate)

    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=args.workers, pin_memory=True,
                              persistent_workers=args.workers > 0, collate_fn=collate)

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
    model = CRNN(num_classes, use_attention=args.attention).to(device)
    arch_version = 3 if args.attention else 2
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}  |  arch_version={arch_version}"
          f"{' (with transformer encoder layer)' if args.attention else ''}")

    # --- Warm start from an existing checkpoint (Option A blend training) ---
    if args.warm_start:
        state = torch.load(args.warm_start, map_location=device, weights_only=False)
        if state.get("vocab") != vocab:
            raise SystemExit(f"--warm-start vocab mismatch: checkpoint has "
                             f"{len(state.get('vocab', []))} classes, charset has {len(vocab)}")
        model.load_state_dict(state["model"])
        print(f"Warm-started from {args.warm_start} "
              f"(arch_version={state.get('arch_version', '?')})")

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    # --- OneCycleLR scheduler ---
    # ceil division: accounts for the tail batch when len(train_loader) % GRAD_ACCUM != 0
    epochs = args.epochs
    steps_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.max_lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=10,
        final_div_factor=100,
    )

    # bf16 autocast: model forward only. logits are cast back to fp32 before
    # log_softmax/CTC (CTC is numerically fragile below fp32). bf16 needs no
    # GradScaler (fp32-range exponent).
    use_amp = args.amp and device.type == "cuda"

    def autocast_ctx():
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp)

    # Parsed by the monitor dashboard for progress/ETA — keep the format stable.
    print(f"Epochs: {epochs} | max_lr={args.max_lr:g} | steps/epoch={steps_per_epoch} | "
          f"amp={'bf16' if use_amp else 'off'} | workers={args.workers}")

    best_cer = float("inf")
    patience_counter = 0
    topk_ckpts: list[tuple[float, str]] = []  # (cer, path), worst last

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()

        # ---- train ----
        model.train()
        tr_loss = 0.0
        optimizer.zero_grad()

        for step, (imgs, targets, target_lengths, _) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            with autocast_ctx():
                logits = model(imgs)
            log_probs = logits.float().log_softmax(2)
            T, B, _ = logits.shape
            input_lengths = torch.full((B,), T, dtype=torch.long, device=device)
            targets_d = targets.to(device, non_blocking=True)
            target_lengths_d = target_lengths.to(device, non_blocking=True)

            loss = criterion(log_probs, targets_d, input_lengths, target_lengths_d)
            loss = loss / GRAD_ACCUM
            loss.backward()
            tr_loss += float(loss.item()) * GRAD_ACCUM

            if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        tr_loss /= max(1, len(train_loader))
        current_lr = optimizer.param_groups[0]["lr"]  # capture LR after all scheduler steps

        # ---- val ----
        model.eval()
        vcer = []; vwer = []; vwer_norm = []
        all_refs = []; all_hyps = []
        shown = 0
        samples_to_save = []

        with torch.no_grad():
            for imgs, _, _, texts_ref in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                with autocast_ctx():
                    logits = model(imgs)
                hyps_ltr = ctc_greedy_decode(logits.float(), id2char)
                hyps_rtl = [h[::-1] for h in hyps_ltr]

                for r, h in zip(texts_ref, hyps_rtl):
                    vcer.append(cer_raw(r, h)); vwer.append(wer_raw(r, h))
                    all_refs.append(r); all_hyps.append(h)

                refs_n = [norm_ar(r) for r in texts_ref]
                hyps_n = [norm_ar(h) for h in hyps_rtl]
                for r, h in zip(refs_n, hyps_n):
                    vwer_norm.append(wer_raw(r, h))

                for r, h in zip(texts_ref, hyps_rtl):
                    if shown < SHOW_SAMPLES_PRINT:
                        r_disp = (r[:120] + "...") if len(r) > 120 else r
                        h_disp = (h[:120] + "...") if len(h) > 120 else h
                        print(f"GT: {r_disp}")
                        print(f"PR: {h_disp}\n")
                        shown += 1
                    samples_to_save.append((r, h))

        if samples_to_save:
            tsv_path = os.path.join(RUN_DIR, f"val_epoch_{epoch:03d}_samples.tsv")
            with open(tsv_path, "w", encoding="utf-8") as fo:
                fo.write("label\tpred\n")
                for r, h in samples_to_save:
                    fo.write(f"{r.replace(chr(9),' ').replace(chr(10),' ')}\t{h.replace(chr(9),' ').replace(chr(10),' ')}\n")

        mcer   = float(np.mean(vcer)) if vcer else 1.0
        mwer   = float(np.mean(vwer)) if vwer else 1.0
        mwer_n = float(np.mean(vwer_norm)) if vwer_norm else 1.0
        mdot   = dot_group_cer(all_refs, all_hyps)

        elapsed = time.perf_counter() - t0
        print(f"Epoch {epoch:03d} | loss={tr_loss:.3f} | CER={mcer:.4f} | WER={mwer:.4f} | "
              f"WER(n)={mwer_n:.4f} | DotCER={mdot:.4f} | LR={current_lr:.2e} | {elapsed:.0f}s")

        # ---- checkpoint & early stopping ----
        ckpt_saved = False
        should_stop = False
        ckpt_payload = {"model": model.state_dict(), "vocab": vocab,
                        "arch_version": arch_version, "use_attention": args.attention}
        if mcer < best_cer:
            best_cer = mcer
            patience_counter = 0
            torch.save(ckpt_payload, os.path.join(RUN_DIR, "crnn_best.pt"))
            ckpt_saved = True
            print(f"  -> Saved best model (CER={best_cer:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  -> Early stopping after {PATIENCE} epochs without improvement")
                should_stop = True

        # ---- top-K checkpoint pool (for post-hoc weight averaging) ----
        if args.save_topk > 1 and (len(topk_ckpts) < args.save_topk or mcer < topk_ckpts[-1][0]):
            path = os.path.join(RUN_DIR, f"crnn_ep{epoch:03d}.pt")
            torch.save(ckpt_payload, path)
            topk_ckpts.append((mcer, path))
            topk_ckpts.sort(key=lambda t: t[0])
            while len(topk_ckpts) > args.save_topk:
                _, worst_path = topk_ckpts.pop()
                try:
                    os.unlink(worst_path)
                except OSError:
                    pass

        # Always log metrics (even the final epoch before early stop)
        _append_metrics(
            epoch=epoch, train_loss=tr_loss,
            cer=mcer, wer=mwer, wer_n=mwer_n, dot_cer=mdot,
            train_batches=len(train_loader), val_batches=len(val_loader),
            ckpt_saved=ckpt_saved, lr=current_lr,
        )

        if should_stop:
            break

    # ---- Final test evaluation ----
    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION")
    print("=" * 60)

    if test_csv.exists():
        # Reload best checkpoint
        state = torch.load(os.path.join(RUN_DIR, "crnn_best.pt"), map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        model.eval()

        test_ds = KHATTDataset(test_csv, IMAGES_DIR, mode="crnn", crnn_h=HEIGHT, crnn_max_w=MAX_W,
                               strip_tatweel=args.strip_tatweel)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=args.workers, pin_memory=True, collate_fn=collate)

        # Greedy + beam(5) without LM: the run-2 decoder sweep showed beam>=5
        # saturates and the train-only bigram LM hurts (see project memory).
        for decode_name, decode_fn in [("Greedy", None), ("Beam(5)", "beam")]:
            tcer = []; twer = []; twer_norm = []
            t_refs = []; t_hyps = []

            with torch.no_grad():
                for imgs, _, _, texts_ref in test_loader:
                    imgs = imgs.to(device, non_blocking=True)
                    with autocast_ctx():
                        logits = model(imgs)
                    logits = logits.float()

                    if decode_fn is None:
                        hyps_ltr = ctc_greedy_decode(logits, id2char)
                    else:
                        hyps_ltr = ctc_beam_decode(logits, id2char, beam_width=5,
                                                   bigram_lm=None, lm_weight=0.0)

                    hyps_rtl = [h[::-1] for h in hyps_ltr]
                    for r, h in zip(texts_ref, hyps_rtl):
                        tcer.append(cer_raw(r, h)); twer.append(wer_raw(r, h))
                        t_refs.append(r); t_hyps.append(h)
                    refs_n = [norm_ar(r) for r in texts_ref]
                    hyps_n = [norm_ar(h) for h in hyps_rtl]
                    for r, h in zip(refs_n, hyps_n):
                        twer_norm.append(wer_raw(r, h))

            test_cer = float(np.mean(tcer))
            test_wer = float(np.mean(twer))
            test_wer_n = float(np.mean(twer_norm))
            test_dot = dot_group_cer(t_refs, t_hyps)

            print(f"\n[{decode_name}] Test CER={test_cer:.4f} | WER={test_wer:.4f} | "
                  f"WER(n)={test_wer_n:.4f} | DotCER={test_dot:.4f}")
    else:
        print("No test.csv found — skipping test evaluation.")

    # Clean up any temp filtered split CSVs
    for p in tmp_csvs:
        try:
            os.unlink(str(p))
        except OSError:
            pass

if __name__ == "__main__":
    main()
