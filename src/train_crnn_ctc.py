import os
import re
import unicodedata as ud
import time
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

from .dataset import KHATTDataset
from .metrics import cer as cer_raw, wer as wer_raw

# ---------------- Config ----------------
IMAGES_DIR = "./data/images"
LABELS_DIR = "./data/labels"
SPLITS_DIR = "./data/splits"
RUN_DIR    = "./runs/exp1"

HEIGHT     = 64
MAX_W      = 1024
BATCH_SIZE = 32
EPOCHS     = 70
LR         = 1e-3
SEED       = 42

CHARSET_PATH = "./src/charset_arabic.txt"

SHOW_SAMPLES_PRINT = 5
SHOW_SAMPLES_FILE  = 200

# ---- NEW: metrics log file path ----
LOG_FILE = os.path.join(RUN_DIR, "metrics.csv")

# ---- NEW: path to resume from (fine-tune) ----
CKPT_PATH = os.path.join(RUN_DIR, "crnn_best.pt")   # change if your best model lives elsewhere

# ------------- Utils -------------
def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

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
            if s.lstrip().startswith("#"):
                continue
            vocab.append(s)
    # enforce required specials at the top
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
    s = s.replace("\u0640", "")  # tatweel
    s = _re_diac.sub("", s)      # diacritics
    s = (s.replace("أ", "ا")
           .replace("إ", "ا")
           .replace("آ", "ا")
           .replace("ى", "ي"))
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ------------- Model -------------
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2,2),(2,1),(0,1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(), nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(), nn.MaxPool2d((2,2),(2,1),(0,1)),
            nn.Conv2d(512, 512, 2, 1, 0), nn.ReLU(),
        )
        self.rnn = nn.LSTM(512, 256, bidirectional=True, num_layers=2, batch_first=True)
        self.fc  = nn.Linear(512, num_classes)

    def forward(self, x):                   # [B,1,H,W]
        x = self.cnn(x)                     # [B,512,H',W']
        x = F.adaptive_avg_pool2d(x, (1, x.size(-1)))   # [B,512,1,W']
        x = x.squeeze(2).permute(0, 2, 1)               # [B,W',512]
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x.permute(1, 0, 2)           # [T=W', B, C]

# ------------- CTC helpers -------------
def text_to_ids(text, char2id, unk_id):
    return [char2id.get(ch, unk_id) for ch in text]

def ids_to_text(ids, id2char):
    out = []
    for i in ids:
        ch = id2char.get(i, "")
        if not ch:
            continue
        if ch.startswith("<") and ch.endswith(">"):
            continue
        out.append(ch)
    return re.sub(r"\s+", " ", "".join(out)).strip()

def ctc_decode(logits, id2char):
    pred = logits.argmax(-1).detach().cpu().numpy()  # [T, B]
    T, B = pred.shape
    texts = []
    for b in range(B):
        seq, last = [], -1
        for t in range(T):
            p = pred[t, b]
            if p != last and p != 0:  # 0 = CTC blank
                seq.append(p)
            last = p
        texts.append(ids_to_text(seq, id2char))
    return texts

# ---- CSV helpers ----
def _init_metrics_file():
    os.makedirs(RUN_DIR, exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", encoding="utf-8", newline="") as f:
            f.write("timestamp,epoch,train_loss,cer,wer,wer_norm,train_batches,val_batches,ckpt_saved\n")

def _append_metrics(epoch, train_loss, cer, wer, wer_n, train_batches, val_batches, ckpt_saved):
    with open(LOG_FILE, "a", encoding="utf-8", newline="") as f:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{ts},{epoch},{train_loss:.6f},{cer:.6f},{wer:.6f},{wer_n:.6f},{train_batches},{val_batches},{int(ckpt_saved)}\n")

# ------------- Train -------------
def main():
    os.makedirs(RUN_DIR, exist_ok=True)
    _init_metrics_file()
    set_seed(SEED)
    if not Path(SPLITS_DIR, "train.csv").exists():
        build_splits()

    vocab, char2id, id2char, unk_id = load_charset(CHARSET_PATH)
    num_classes = len(vocab)

    train_ds = KHATTDataset(Path(SPLITS_DIR, "train.csv"), IMAGES_DIR, mode="crnn", crnn_h=HEIGHT, crnn_max_w=MAX_W)
    val_ds   = KHATTDataset(Path(SPLITS_DIR, "val.csv"),   IMAGES_DIR, mode="crnn", crnn_h=HEIGHT, crnn_max_w=MAX_W)

    to_tensor = transforms.ToTensor()

    def collate(batch):
        imgs, texts_orig = zip(*batch)
        texts_ltr = [t[::-1] for t in texts_orig]     # LTR time-axis
        imgs = torch.stack([to_tensor(im) for im in imgs], dim=0)  # [B,1,H,W]
        targets = [torch.tensor(text_to_ids(t, char2id, unk_id), dtype=torch.long) for t in texts_ltr]
        target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
        targets = torch.cat(targets) if len(targets) else torch.tensor([], dtype=torch.long)
        return imgs, targets, target_lengths, list(texts_orig)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True, collate_fn=collate)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True, collate_fn=collate)

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
    model = CRNN(num_classes).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    # -------- NEW: resume & fine-tune from best checkpoint --------
    if os.path.exists(CKPT_PATH):
        state = torch.load(CKPT_PATH, map_location=device)
        if "model" in state:
            model.load_state_dict(state["model"])
            print(f"Resumed weights from {CKPT_PATH}")
            # lower LR for safer fine-tuning
            for g in optimizer.param_groups:
                g["lr"] = 3e-4
            print(f"Fine-tune LR set to {optimizer.param_groups[0]['lr']}")
        else:
            print(f"Checkpoint found at {CKPT_PATH} but missing 'model' key. Skipping resume.")
    # ---------------------------------------------------------------

    best_cer = float("inf")
    for epoch in range(1, EPOCHS+1):
        t0 = time.perf_counter()
        # ---- train ----
        model.train(); tr_loss = 0.0
        for imgs, targets, target_lengths, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            logits = model(imgs)                          # [T, B, C]
            log_probs = logits.log_softmax(2)
            T, B, _ = logits.shape
            input_lengths  = torch.full((B,), T, dtype=torch.long, device=device)
            targets        = targets.to(device, non_blocking=True)
            target_lengths = target_lengths.to(device, non_blocking=True)

            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            tr_loss += float(loss.item())
        tr_loss /= max(1, len(train_loader))

        # ---- val ----
        model.eval(); vcer=[]; vwer=[]; vwer_norm=[]
        shown = 0
        samples_to_save = []
        with torch.no_grad():
            for imgs, _, _, texts_ref in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                logits = model(imgs)
                hyps_ltr = ctc_decode(logits, id2char)   # LTR
                hyps_rtl = [h[::-1] for h in hyps_ltr]   # back to RTL

                for r, h in zip(texts_ref, hyps_rtl):
                    vcer.append(cer_raw(r, h)); vwer.append(wer_raw(r, h))
                refs_n = [norm_ar(r) for r in texts_ref]
                hyps_n = [norm_ar(h) for h in hyps_rtl]
                for r, h in zip(refs_n, hyps_n):
                    vwer_norm.append(wer_raw(r, h))

                for r, h in zip(texts_ref, hyps_rtl):
                    if shown < SHOW_SAMPLES_PRINT:
                        r_disp = (r[:120] + "…") if len(r) > 120 else r
                        h_disp = (h[:120] + "…") if len(h) > 120 else h
                        print(f"GT: {r_disp}")
                        print(f"PR: {h_disp}\n")
                        shown += 1
                    if len(samples_to_save) < SHOW_SAMPLES_FILE:
                        samples_to_save.append((r, h))

        if samples_to_save:
            os.makedirs(RUN_DIR, exist_ok=True)
            tsv_path = os.path.join(RUN_DIR, f"val_epoch_{epoch:02d}_samples.tsv")
            with open(tsv_path, "w", encoding="utf-8") as fo:
                fo.write("label\tpred\n")
                for r, h in samples_to_save:
                    fo.write(f"{r.replace('\t',' ').replace('\n',' ')}\t{h.replace('\t',' ').replace('\n',' ')}\n")

        mcer   = float(np.mean(vcer)) if vcer else 1.0
        mwer   = float(np.mean(vwer)) if vwer else 1.0
        mwer_n = float(np.mean(vwer_norm)) if vwer_norm else 1.0
        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.3f} | CER={mcer:.3f} | WER={mwer:.3f} | WER(norm)={mwer_n:.3f}")

        # save best & log
        ckpt_saved = False
        if mcer < best_cer:
            best_cer = mcer
            torch.save({"model": model.state_dict(), "vocab": vocab},
                       os.path.join(RUN_DIR, "crnn_best.pt"))
            ckpt_saved = True
            print("✔ Saved best model")

        _append_metrics(
            epoch=epoch,
            train_loss=tr_loss,
            cer=mcer, wer=mwer, wer_n=mwer_n,
            train_batches=len(train_loader),
            val_batches=len(val_loader),
            ckpt_saved=ckpt_saved
        )

        t1 = time.perf_counter()
        # optional: print(f"Epoch time: {t1 - t0:.1f}s")

if __name__ == "__main__":
    main()