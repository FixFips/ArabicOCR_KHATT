import os # build a file path
import pandas as pd  # able to make u read the csv files in ur system into a dataframe
from PIL import Image # opens the image files in ur system
from torch.utils.data import Dataset
from .preprocess import (  # we import the below items from our preprocess
    to_grayscale, binarize, normalize,
    resize_keep_ratio_height, pad_width, pad_to_square
)

def read_label(path: str) -> str:
    for enc in ["windows-1256", "utf-8", "utf-8-sig"]: # the data label is encoded so we need to try some common encoding
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read().strip() #returns the file content with strip which removes any white spaces.
        except Exception:
            pass
    raise RuntimeError(f"Cannot read label file: {path}")

class KHATTDataset(Dataset): #so the pytorch can know where is ur path and report the data length + fetch a single item for training
    def __init__(self, csv_path: str, images_dir: str,
                 mode: str = "crnn", crnn_h: int = 64, crnn_max_w: int = 1024, trocr_side: int = 384):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.mode = mode
        self.crnn_h = crnn_h
        self.crnn_max_w = crnn_max_w
        self.trocr_side = trocr_side

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

        if self.mode == "crnn":
            img = resize_keep_ratio_height(img, self.crnn_h)
            if img.width > self.crnn_max_w:
                img = img.resize((self.crnn_max_w, self.crnn_h), Image.BILINEAR)
            img = pad_width(img, self.crnn_max_w)
        else:
            img = pad_to_square(img, self.trocr_side)

        label = read_label(label_path)
        return img, label
