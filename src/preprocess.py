# src/preprocess.py
import cv2
import numpy as np
from PIL import Image

def to_grayscale(img: Image.Image) -> Image.Image:
    return img.convert("L")

def binarize(img: Image.Image) -> Image.Image:
    # 1) grayscale + local contrast (handles colored backgrounds)
    g = np.array(img.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(g)

    # 2) Otsu both polarities
    _, b1 = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, b2 = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 3) choose the more "text-like" (≈ 15–35% black + horizontal variability)
    def score(bw):
        br = (bw == 0).mean()
        hp = (bw == 0).sum(axis=1).astype(np.float32)
        return -(abs(br - 0.25)) + (hp.var() / 1e6)

    th = b1 if score(b1) > score(b2) else b2

    # 4) fallback to adaptive if Otsu fails (nearly-all black/white)
    br = (th == 0).mean()
    if br < 0.03 or br > 0.90:
        th = cv2.adaptiveThreshold(
            g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 15
        )

    # 5) FORCE final polarity: background white, ink black
    if th.mean() < 127:            # background mostly dark → invert
        th = 255 - th

    # 6) light cleanup
    th = cv2.medianBlur(th, 3)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)

    return Image.fromarray(th)

def normalize(img: Image.Image) -> Image.Image:
    return img  # normalization happens in ToTensor()

# ---------- CRNN helpers ----------
def resize_keep_ratio_height(img: Image.Image, target_h: int) -> Image.Image:
    w, h = img.size
    new_w = max(1, int(w * (target_h / h)))
    return img.resize((new_w, target_h), Image.BILINEAR)

def pad_width(img: Image.Image, max_w: int) -> Image.Image:
    bg = Image.new("L", (max_w, img.height), 255)
    bg.paste(img, (0, 0))
    return bg

# ---------- TrOCR helper (kept) ----------
def pad_to_square(img: Image.Image, side: int = 384) -> Image.Image:
    w, h = img.size
    scale = min(side / w, side / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    img = img.resize((nw, nh), Image.BILINEAR)
    bg = Image.new("L", (side, side), 255)
    bg.paste(img, ((side - nw) // 2, (side - nh) // 2))
    return bg