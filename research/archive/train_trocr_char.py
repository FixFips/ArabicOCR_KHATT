import os
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from transformers import (
    VisionEncoderDecoderModel,
    AutoImageProcessor,
    PreTrainedTokenizerFast,
    AdamW,
)

from sklearn.model_selection import train_test_split

from src.dataset import KHATTDataset
from src.metrics import cer, wer

IMAGES_DIR = "./data/images"
LABELS_DIR = "./data/labels"
SPLITS_DIR = "./data/splits"
RUN_DIR    = "./runs/exp1"

BATCH_SIZE = 8
EPOCHS     = 10
LR         = 5e-5
SIDE       = 384
SEED       = 42

ENCODER_NAME = "microsoft/trocr-small-stage1"  # ViT encoder
CHARSET_PATH = "./src/charset_arabic.txt"


def load_charset():
    with open(CHARSET_PATH, "r", encoding="utf-8") as f:
        vocab = [t.strip("\n") for t in f.readlines()]
    bos_id = vocab.index("<bos>")
    eos_id = vocab.index("<eos>")
    pad_id = vocab.index("<pad>")
    unk_id = vocab.index("<unk>")
    return vocab, bos_id, eos_id, pad_id, unk_id


def make_tokenizer(vocab):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    vocab_dict = {tok: i for i, tok in enumerate(vocab)}
    tok = Tokenizer(WordLevel(vocab=vocab_dict, unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<bos>", eos_token="<eos>", pad_token="<pad>", unk_token="<unk>")
    return fast


def collate(tokenizer):
    def _fn(batch):
        imgs, texts = zip(*batch)
        pixel_values = torch.stack([transforms.ToTensor()(im).repeat(3,1,1) for im in imgs])  # 3-ch
        labels = [tokenizer.convert_tokens_to_ids(list(t)) for t in texts]
        labels = [[tokenizer.convert_tokens_to_ids("<bos>")] + l + [tokenizer.convert_tokens_to_ids("<eos>")] for l in labels]
        max_len = max(len(l) for l in labels)
        pad_id = tokenizer.convert_tokens_to_ids("<pad>")
        label_ids = torch.full((len(labels), max_len), pad_id, dtype=torch.long)
        for i,l in enumerate(labels):
            label_ids[i,:len(l)] = torch.tensor(l)
        return {"pixel_values": pixel_values, "labels": label_ids}
    return _fn


def main():
    os.makedirs(RUN_DIR, exist_ok=True)

    if not Path(os.path.join(SPLITS_DIR, "train.csv")).exists():
        rows=[]
        for fname in os.listdir(IMAGES_DIR):
            if fname.lower().endswith((".png",".jpg",".jpeg")):
                lab = os.path.join("./data/labels", os.path.splitext(fname)[0]+".txt")
                if os.path.exists(lab):
                    rows.append({"filename": fname, "label_path": lab})
        df = pd.DataFrame(rows)
        from sklearn.model_selection import train_test_split
        train_df, temp = train_test_split(df, test_size=0.2, random_state=SEED, shuffle=True)
        val_df, test_df = train_test_split(temp, test_size=0.5, random_state=SEED, shuffle=True)
        os.makedirs(SPLITS_DIR, exist_ok=True)
        train_df.to_csv(os.path.join(SPLITS_DIR,"train.csv"), index=False)
        val_df.to_csv(os.path.join(SPLITS_DIR,"val.csv"), index=False)
        test_df.to_csv(os.path.join(SPLITS_DIR,"test.csv"), index=False)

    vocab, bos_id, eos_id, pad_id, unk_id = load_charset()
    tokenizer = make_tokenizer(vocab)

    image_processor = AutoImageProcessor.from_pretrained(ENCODER_NAME)
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(ENCODER_NAME, ENCODER_NAME)
    model.decoder.resize_token_embeddings(len(vocab))

    train_ds = KHATTDataset(os.path.join(SPLITS_DIR, "train.csv"), IMAGES_DIR, mode="trocr", trocr_side=SIDE)
    val_ds   = KHATTDataset(os.path.join(SPLITS_DIR, "val.csv"),   IMAGES_DIR, mode="trocr", trocr_side=SIDE)

    loader_train = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=collate(tokenizer))
    loader_val   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, collate_fn=collate(tokenizer))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optim = AdamW(model.parameters(), lr=LR)

    def decode(ids):
        inv = {i:t for i,t in enumerate(vocab)}
        txt = "".join(inv[i] for i in ids if i not in (pad_id, bos_id, eos_id))
        return txt

    best = 1e9
    for epoch in range(1, EPOCHS+1):
        model.train(); tr_loss=0
        for batch in loader_train:
            inp = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            out = model(pixel_values=inp, labels=labels)
            loss = out.loss
            optim.zero_grad(); loss.backward(); optim.step()
            tr_loss += loss.item()
        tr_loss /= max(1,len(loader_train))

        # val (greedy generate)
        model.eval(); vc=[]; vw=[]
        with torch.no_grad():
            for batch in loader_val:
                inp = batch["pixel_values"].to(device)
                out = model.generate(pixel_values=inp, max_length=128)
                hyps = [decode(seq.tolist()) for seq in out]
                # quick refs (same order)
                _, refs = zip(*[val_ds[i] for i in range(len(hyps))])
                for r,h in zip(refs, hyps):
                    vc.append(cer(r,h)); vw.append(wer(r,h))
        print(f"Epoch {epoch:02d} | train_loss={tr_loss:.3f} | CER={np.mean(vc):.3f} | WER={np.mean(vw):.3f}")
        if np.mean(vc) < best:
            best = float(np.mean(vc))
            model.save_pretrained(os.path.join(RUN_DIR, "trocr_char_best"))
            tokenizer.save_pretrained(os.path.join(RUN_DIR, "trocr_char_best_tok"))
            print("✔ Saved best transformer model")

if __name__ == "__main__":
    main()