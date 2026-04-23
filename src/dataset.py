import os
import re
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from .preprocess import (
    to_grayscale, binarize, normalize,
    resize_keep_ratio_height, pad_width, pad_to_square
)

_WS_RUN = re.compile(r"\s+")


def read_label(path: str) -> str:
    for enc in ["windows-1256", "utf-8", "utf-8-sig"]:
        try:
            with open(path, "r", encoding=enc) as f:
                raw = f.read()
            return _WS_RUN.sub(" ", raw).strip()
        except Exception:
            pass
    raise RuntimeError(f"Cannot read label file: {path}")


class KHATTDataset(Dataset):
    def __init__(self, csv_path: str, images_dir: str,
                 mode: str = "crnn", crnn_h: int = 96, crnn_max_w: int = 1536,
                 trocr_side: int = 384, augment=None):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.mode = mode
        self.crnn_h = crnn_h
        self.crnn_max_w = crnn_max_w
        self.trocr_side = trocr_side
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row["filename"])
        label_path = row["label_path"]

        img = Image.open(img_path)
        img = to_grayscale(img)
        img = binarize(img)
        img = normalize(img)

        # Arabic-safe augmentation (training only, before resize/pad)
        if self.augment is not None:
            img = self.augment(img)

        if self.mode == "crnn":
            img = resize_keep_ratio_height(img, self.crnn_h)
            if img.width > self.crnn_max_w:
                img = img.resize((self.crnn_max_w, self.crnn_h), Image.LANCZOS)
            img = pad_width(img, self.crnn_max_w)
        else:
            img = pad_to_square(img, self.trocr_side)

        label = read_label(label_path)
        return img, label
