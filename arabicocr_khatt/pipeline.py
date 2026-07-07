# arabicocr_khatt/pipeline.py
"""High-level inference API for Arabic handwritten OCR.

Usage (Python):

    from arabicocr_khatt import ArabicOCR

    ocr = ArabicOCR.from_pretrained()          # downloads weights from the HF Hub
    text = ocr.recognize("page.jpg")           # str, Path, PIL.Image or np.ndarray

Usage (CLI):

    arabicocr page.jpg
    arabicocr line.png --greedy --no-segment
    arabicocr scan.jpg --checkpoint runs/exp1/crnn_best.pt

This module is the single source for the inference pipeline (preprocessing
constants, line segmentation, polarity handling, decoding).  The Gradio test
bench (webocr.py) imports from here — never duplicate these functions.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from .model import CRNN, ctc_beam_decode, ctc_greedy_decode
from .preprocess import binarize, normalize, pad_width, resize_keep_ratio_height, to_grayscale

# ---------------- Config (must match training) ----------------
HEIGHT = 96
MAX_W = 1536

DEFAULT_REPO_ID = os.environ.get("ARABICOCR_REPO", "FixFips/arabicocr-khatt")
DEFAULT_CKPT_FILENAME = "crnn_best.pt"
BIGRAM_LM_FILENAME = "bigram_lm.json"

ImageLike = Union[str, Path, Image.Image, np.ndarray]


# ---------------- Line preprocessing (same pipeline as training) ----------------

def prep_line(img: Image.Image, upscale: float = 1.0, force_invert: bool = False) -> Image.Image:
    if upscale and upscale != 1.0:
        w, h = img.size
        img = img.resize((max(1, int(w * upscale)), max(1, int(h * upscale))), Image.BICUBIC)

    if force_invert:
        img = ImageOps.invert(img.convert("RGB"))

    # Same pipeline as training: grayscale -> CLAHE+Otsu binarize -> normalize
    img = to_grayscale(img)
    img = binarize(img)

    # Enforce black text on white bg
    if np.asarray(img).mean() < 127:
        img = ImageOps.invert(img)

    img = normalize(img)
    img = resize_keep_ratio_height(img, HEIGHT)
    if img.width > MAX_W:
        img = img.resize((MAX_W, HEIGHT), Image.LANCZOS)
    img = pad_width(img, MAX_W)
    return img


# ---------------- Multi-line segmentation (morphology) ----------------

def _robust_binarize(pil_img: Image.Image) -> np.ndarray:
    g = np.array(pil_img.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = clahe.apply(g)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (bw == 0).mean() < 0.12:
        bw = 255 - bw
    return bw


def segment_into_lines(
    pil_img: Image.Image,
    min_h: int = 14,
    min_width_ratio: float = 0.35,
    remove_ruled_lines: bool = True
) -> list[Image.Image]:
    bw = _robust_binarize(pil_img)
    text = 255 - bw
    H, W = text.shape

    if remove_ruled_lines:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(W // 8, 80), 1))
        rules = cv2.morphologyEx(text, cv2.MORPH_OPEN, k, iterations=1)
        text = cv2.subtract(text, rules)

    kx = max(W // 40, 20)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, 3))
    smooth = cv2.dilate(text, kernel, iterations=1)

    tiny_k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(W // 90, 8), 3))
    smooth = cv2.morphologyEx(smooth, cv2.MORPH_OPEN, tiny_k, iterations=1)

    cc = (smooth > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(cc, connectivity=8)

    boxes = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if h < min_h:
            continue
        if w < int(W * min_width_ratio):
            continue
        if h > H * 0.35 and w < W * 0.60:
            continue
        boxes.append((y, x, w, h))

    boxes.sort(key=lambda b: b[0])

    merged = []
    for y, x, w, h in boxes:
        if merged and y - (merged[-1][0] + merged[-1][3]) < int(H * 0.02):
            y0, x0, w0, h0 = merged[-1]
            nx = min(x, x0); ny = min(y, y0)
            nx2 = max(x + w, x0 + w0); ny2 = max(y + h, y0 + h0)
            merged[-1] = (ny, nx, nx2 - nx, ny2 - ny)
        else:
            merged.append((y, x, w, h))

    lines = [pil_img.crop((x, y, x + w, y + h)) for (y, x, w, h) in merged]
    return lines or [pil_img]


# ---------------- Checkpoint / LM loading ----------------

def load_checkpoint(path: Union[str, Path]) -> dict:
    """Load a checkpoint dict {'model': state_dict, 'vocab': list[str], 'arch_version': 2}.

    Loads with weights_only=True (v2 checkpoints contain only tensors and
    primitives). Unpickling arbitrary objects is a code-execution risk, so the
    unsafe path requires the ARABICOCR_UNSAFE_LOAD=1 environment variable and
    should only be used with checkpoints you trained yourself.
    """
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        if os.environ.get("ARABICOCR_UNSAFE_LOAD") == "1":
            state = torch.load(path, map_location="cpu", weights_only=False)
        else:
            raise RuntimeError(
                f"Could not load {path} with weights_only=True: {e}. "
                "If this is a legacy checkpoint you trained yourself, retry with "
                "ARABICOCR_UNSAFE_LOAD=1 (unpickles arbitrary objects — unsafe for "
                "untrusted files)."
            ) from e
    if "vocab" not in state or "model" not in state:
        raise RuntimeError(
            "Checkpoint missing 'vocab'/'model'. Expected format "
            "{'model': state_dict, 'vocab': list[str], 'arch_version': 2}."
        )
    if state.get("arch_version") != 2:
        raise RuntimeError(
            f"Checkpoint arch_version={state.get('arch_version')!r} is not supported "
            "(this release requires the v2 multi-scale vertical CRNN)."
        )
    return state


def save_bigram_lm_json(lm: dict, path: Union[str, Path]) -> None:
    """Serialize a bigram LM (built by model.build_bigram_lm) to JSON."""
    default = lm.get(("_default",))
    bigrams = {
        f"{k[0]},{k[1]}": v for k, v in lm.items() if k != ("_default",)
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"default": default, "bigrams": bigrams}, f)


def load_bigram_lm_json(path: Union[str, Path]) -> dict:
    """Load a bigram LM serialized by save_bigram_lm_json."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    lm = {tuple(int(p) for p in k.split(",")): v for k, v in data["bigrams"].items()}
    if data.get("default") is not None:
        lm[("_default",)] = data["default"]
    return lm


def _to_tensor(img: Image.Image) -> torch.Tensor:
    """L-mode PIL image -> [1, 1, H, W] float tensor in [0, 1]."""
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None, None]


def _ensure_pil(image: ImageLike) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return Image.fromarray(image.astype(np.uint8), mode="L")
        return Image.fromarray(image.astype(np.uint8)[..., :3], mode="RGB")
    if isinstance(image, (str, Path)):
        return Image.open(image)
    raise TypeError(f"Unsupported image type: {type(image)!r}")


# ---------------- Public API ----------------

class ArabicOCR:
    """Line-level Arabic handwritten text recognizer (CRNN-CTC, KHATT).

    Args:
        checkpoint: path to a crnn_best.pt checkpoint.
        device: torch device string ("cuda", "cpu", "mps"); auto-detected if None.
        bigram_lm: optional Arabic character bigram LM dict for beam search
            (as built by model.build_bigram_lm or loaded via load_bigram_lm_json).
    """

    def __init__(
        self,
        checkpoint: Union[str, Path],
        device: Optional[str] = None,
        bigram_lm: Optional[dict] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        state = load_checkpoint(checkpoint)
        self.vocab: List[str] = list(state["vocab"])
        self.id2char = {i: c for i, c in enumerate(self.vocab)}
        self.char2id = {c: i for i, c in enumerate(self.vocab)}
        self.bigram_lm = bigram_lm

        self.model = CRNN(num_classes=len(self.vocab)).to(self.device)
        self.model.load_state_dict(state["model"])
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = DEFAULT_REPO_ID,
        filename: str = DEFAULT_CKPT_FILENAME,
        device: Optional[str] = None,
        revision: Optional[str] = None,
    ) -> "ArabicOCR":
        """Download weights (and the bigram LM, if published) from the HF Hub."""
        from huggingface_hub import hf_hub_download

        ckpt_path = hf_hub_download(repo_id, filename, revision=revision)
        bigram_lm = None
        try:
            lm_path = hf_hub_download(repo_id, BIGRAM_LM_FILENAME, revision=revision)
            bigram_lm = load_bigram_lm_json(lm_path)
        except Exception:
            pass  # LM is optional — beam search still works without it
        return cls(ckpt_path, device=device, bigram_lm=bigram_lm)

    # ---- inference ----

    def _forward(self, prepped: Image.Image) -> torch.Tensor:
        return self.model(_to_tensor(prepped).to(self.device))  # [T, 1, C]

    def _decode(self, logits: torch.Tensor, beam_width: int, lm_weight: float) -> str:
        if beam_width > 1:
            lm = self.bigram_lm if lm_weight > 0 else None
            hyp_ltr = ctc_beam_decode(
                logits, self.id2char,
                beam_width=beam_width, bigram_lm=lm, lm_weight=lm_weight,
            )[0]
        else:
            hyp_ltr = ctc_greedy_decode(logits, self.id2char)[0]
        return hyp_ltr[::-1]  # model works LTR; reverse back to Arabic RTL

    @torch.inference_mode()
    def recognize_lines(
        self,
        image: ImageLike,
        segment: bool = True,
        beam_width: int = 10,
        lm_weight: float = 0.3,
        polarity: str = "auto",
        upscale: float = 1.0,
    ) -> List[str]:
        """Recognize an image and return one string per detected line."""
        pil = _ensure_pil(image)
        lines = segment_into_lines(pil) if segment else [pil]

        texts: List[str] = []
        for ln in lines:
            if polarity == "normal":
                logits = self._forward(prep_line(ln, upscale=upscale, force_invert=False))
            elif polarity == "invert":
                logits = self._forward(prep_line(ln, upscale=upscale, force_invert=True))
            else:  # auto: run both polarities, keep the one that reads more text
                ln_ = self._forward(prep_line(ln, upscale=upscale, force_invert=False))
                li_ = self._forward(prep_line(ln, upscale=upscale, force_invert=True))
                tn = self._decode(ln_, beam_width=1, lm_weight=0.0)
                ti = self._decode(li_, beam_width=1, lm_weight=0.0)
                logits = li_ if len(ti.strip()) > len(tn.strip()) else ln_
            texts.append(self._decode(logits, beam_width=beam_width, lm_weight=lm_weight))
        return texts

    def recognize(self, image: ImageLike, **kwargs: Any) -> str:
        """Recognize an image and return the full text (lines joined by newlines).

        Accepts the same keyword arguments as recognize_lines.
        """
        return "\n".join(self.recognize_lines(image, **kwargs))


# ---------------- CLI ----------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="arabicocr",
        description="Arabic handwritten text recognition (CRNN-CTC, KHATT).",
    )
    parser.add_argument("images", nargs="+", help="image file(s) to recognize")
    parser.add_argument("--checkpoint", help="local crnn_best.pt (default: download from HF Hub)")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="HF Hub repo for weights")
    parser.add_argument("--device", default=None, help='torch device (e.g. "cpu", "cuda")')
    parser.add_argument("--greedy", action="store_true", help="greedy decode (default: beam search)")
    parser.add_argument("--beam-width", type=int, default=10)
    parser.add_argument("--lm-weight", type=float, default=0.3)
    parser.add_argument("--no-segment", action="store_true", help="treat input as a single line")
    parser.add_argument("--polarity", choices=["auto", "normal", "invert"], default="auto")
    parser.add_argument("--upscale", type=float, default=1.0, help="upscale factor for tiny text")
    args = parser.parse_args(argv)

    if args.checkpoint:
        ocr = ArabicOCR(args.checkpoint, device=args.device)
    else:
        ocr = ArabicOCR.from_pretrained(args.repo_id, device=args.device)

    for path in args.images:
        text = ocr.recognize(
            path,
            segment=not args.no_segment,
            beam_width=1 if args.greedy else args.beam_width,
            lm_weight=args.lm_weight,
            polarity=args.polarity,
            upscale=args.upscale,
        )
        if len(args.images) > 1:
            print(f"{path}\t{text}")
        else:
            print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
