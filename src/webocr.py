# src/webocr.py
import os
import random
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageOps

import torch
from torchvision import transforms
import gradio as gr

from .dataset import read_label
from .metrics import cer as cer_raw, wer as wer_raw, dot_group_cer
from .model import CRNN, ctc_greedy_decode, ctc_beam_decode, build_bigram_lm
from .preprocess import to_grayscale, binarize, normalize, resize_keep_ratio_height, pad_width


# ---------------- Config (match training) ----------------
HEIGHT = 96
MAX_W  = 1536
CKPT   = "./runs/exp1/crnn_best.pt"
SPLITS_DIR  = "./archive/splits"
IMAGES_DIR  = "./archive/images"


# ---------------- Checkpoint / charset helpers ----------------
def load_vocab_from_ckpt(path: str):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "vocab" not in state:
        raise RuntimeError("Checkpoint missing 'vocab'. Save as {'model':..., 'vocab':...}.")
    vocab = state["vocab"]
    id2char = {i: c for i, c in enumerate(vocab)}
    return vocab, id2char, state["model"]


# ---------- Line preprocessing (uses same pipeline as training) ----------

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


# ---------------- Robust ImageEditor -> PIL conversion ----------------
def _np_to_pil(arr: np.ndarray) -> Image.Image:
    if arr.ndim == 2:
        return Image.fromarray(arr.astype(np.uint8), mode="L")
    if arr.ndim == 3 and arr.shape[2] == 3:
        return Image.fromarray(arr.astype(np.uint8), mode="RGB")
    if arr.ndim == 3 and arr.shape[2] == 4:
        rgba = Image.fromarray(arr.astype(np.uint8), mode="RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[-1])
        return bg
    raise ValueError("Unsupported ndarray shape for image.")

def ensure_pil_from_editor(x: Any, use_cropped: bool = True) -> Image.Image:
    if isinstance(x, dict):
        cand = None
        if use_cropped and "image" in x:
            cand = x["image"]
        elif "background" in x:
            cand = x["background"]
        elif "layers" in x and isinstance(x["layers"], list) and len(x["layers"]) > 0:
            cand = x["layers"][-1]

        if isinstance(cand, Image.Image):
            return cand
        if isinstance(cand, np.ndarray):
            return _np_to_pil(cand)
        if isinstance(cand, dict) and "data" in cand and isinstance(cand["data"], np.ndarray):
            return _np_to_pil(cand["data"])
        raise ValueError("Unsupported image format inside editor dict.")

    if isinstance(x, Image.Image):
        return x
    if isinstance(x, np.ndarray):
        return _np_to_pil(x)
    if isinstance(x, str) and os.path.exists(x):
        return Image.open(x)
    raise ValueError("Unsupported image format.")


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


def stack_preview(imgs: List[Image.Image]) -> Image.Image:
    if len(imgs) == 1:
        return imgs[0]
    widths = [im.width for im in imgs]
    maxw = max(widths)
    totalh = sum(im.height for im in imgs) + 4 * (len(imgs) - 1)
    canvas = Image.new("L", (maxw, totalh), 255)
    y = 0
    for im in imgs:
        canvas.paste(im, (0, y))
        y += im.height + 4
    return canvas


# ---------------- Utilities: rotation ----------------
def rotate_if_needed(pil: Image.Image, angle_deg: float) -> Image.Image:
    if not angle_deg:
        return pil
    mode = pil.mode
    if mode not in ("L", "RGB"):
        pil = pil.convert("RGB")
        mode = "RGB"
    fill = 255 if mode == "L" else (255, 255, 255)
    return pil.rotate(angle_deg, resample=Image.BICUBIC, expand=True, fillcolor=fill)


# ---------------- Model init ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab, id2char, state_dict = load_vocab_from_ckpt(CKPT)
char2id = {c: i for i, c in enumerate(vocab)}
model = CRNN(num_classes=len(vocab)).to(device)
model.load_state_dict(state_dict)
model.eval()
to_tensor = transforms.ToTensor()


# ---------------- Optional bigram LM for beam search ----------------
_bigram_lm: Optional[dict] = None
def get_bigram_lm():
    """Lazy-build the bigram LM from the training split on first use."""
    global _bigram_lm
    if _bigram_lm is not None:
        return _bigram_lm
    train_csv = Path(SPLITS_DIR, "train.csv")
    if not train_csv.exists():
        return None
    df = pd.read_csv(train_csv)
    texts = []
    for _, row in df.iterrows():
        try:
            texts.append(read_label(row["label_path"]))
        except Exception:
            pass
    _bigram_lm = build_bigram_lm(texts, char2id)
    return _bigram_lm


# ---------------- Sample loader (KHATT test/val) ----------------
def load_random_sample(split: str = "test") -> Tuple[Optional[Image.Image], str, str]:
    csv_path = Path(SPLITS_DIR, f"{split}.csv")
    if not csv_path.exists():
        return None, "", f"split not found: {csv_path}"
    df = pd.read_csv(csv_path)
    if not len(df):
        return None, "", "split is empty"
    row = df.sample(1).iloc[0]
    img_path = os.path.join(IMAGES_DIR, row["filename"])
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        return None, "", f"cannot open {img_path}: {e}"
    try:
        gt = read_label(row["label_path"])
    except Exception:
        gt = ""
    return img, gt, f"{split}: {row['filename']}"


# ---------------- Decode helpers ----------------
def _decode_one(im_pil: Image.Image, device: torch.device, id2char,
                use_beam: bool = False, beam_width: int = 10,
                lm_weight: float = 0.0) -> str:
    x = to_tensor(im_pil).unsqueeze(0).to(device)
    logits = model(x)
    if use_beam:
        lm = get_bigram_lm() if lm_weight > 0 else None
        hyp_ltr = ctc_beam_decode(logits, id2char,
                                  beam_width=beam_width,
                                  bigram_lm=lm,
                                  lm_weight=lm_weight)[0]
    else:
        hyp_ltr = ctc_greedy_decode(logits, id2char)[0]
    return hyp_ltr[::-1]  # back to RTL


@torch.inference_mode()
def recognize_image(
    pil_img: Image.Image,
    upscale: float = 1.0,
    force_multiline: bool = True,
    polarity_mode: str = "auto",
    use_beam: bool = False,
    beam_width: int = 10,
    lm_weight: float = 0.0,
) -> Tuple[str, Image.Image]:
    lines = segment_into_lines(pil_img) if force_multiline else [pil_img]

    selected_prepped: List[Image.Image] = []
    texts: List[str] = []

    def _dec(img):
        return _decode_one(img, device, id2char,
                           use_beam=use_beam, beam_width=beam_width,
                           lm_weight=lm_weight)

    for ln in lines:
        if polarity_mode == "normal":
            prep_n = prep_line(ln, upscale=upscale, force_invert=False)
            prep_best, txt_best = prep_n, _dec(prep_n)

        elif polarity_mode == "invert":
            prep_i = prep_line(ln, upscale=upscale, force_invert=True)
            prep_best, txt_best = prep_i, _dec(prep_i)

        else:  # auto: try both, choose longer non-blank
            prep_n = prep_line(ln, upscale=upscale, force_invert=False)
            txt_n = _dec(prep_n)
            prep_i = prep_line(ln, upscale=upscale, force_invert=True)
            txt_i = _dec(prep_i)
            if len(txt_i.strip()) > len(txt_n.strip()):
                prep_best, txt_best = prep_i, txt_i
            else:
                prep_best, txt_best = prep_n, txt_n

        selected_prepped.append(prep_best)
        texts.append(txt_best)

    stacked = stack_preview(selected_prepped)
    return "\n".join(texts), stacked


# ---------------- Gradio UI ----------------
def _map_polarity(label: str) -> str:
    if label.startswith("Normal"): return "normal"
    if label.startswith("Invert"): return "invert"
    return "auto"


def _fmt_metrics(gt: str, pred: str) -> str:
    if not gt.strip():
        return ""
    c = cer_raw(gt, pred) * 100
    w = wer_raw(gt, pred) * 100
    d = dot_group_cer([gt], [pred]) * 100
    color = "#3fb950" if c < 10 else "#d29922" if c < 25 else "#f85149"
    return (f'<div style="font-family:ui-monospace,monospace;font-size:0.95em">'
            f'<span style="color:{color}"><b>CER {c:.2f}%</b></span> &nbsp; '
            f'WER {w:.2f}% &nbsp; DotCER {d:.2f}%</div>')


def infer(editor_payload: Any, use_cropped: bool, angle: float, upscale: float,
          force_multiline: bool, polarity_label: str, use_beam: bool,
          beam_width: int, lm_weight: float, gt_text: str):
    if editor_payload is None:
        return "", None, "", ""
    try:
        pil = ensure_pil_from_editor(editor_payload, use_cropped=use_cropped)
    except Exception as e:
        return f"Unsupported image format ({type(editor_payload)}): {e}", None, "", ""

    pil = rotate_if_needed(pil, angle)
    mode = _map_polarity(polarity_label)
    t0 = time.perf_counter()
    txt, prev = recognize_image(
        pil, upscale=upscale, force_multiline=force_multiline,
        polarity_mode=mode, use_beam=use_beam,
        beam_width=int(beam_width), lm_weight=float(lm_weight),
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    info = f'<span style="color:#8b949e">decoded in {elapsed_ms:.0f} ms &middot; device={device.type}</span>'
    metrics_html = _fmt_metrics(gt_text or "", txt)
    return txt, prev, metrics_html, info


def load_sample_cb(split: str):
    img, gt, info = load_random_sample(split)
    # Gradio ImageEditor accepts None or a dict-like; passing a PIL directly works for upload display.
    return img, gt, f'<span style="color:#8b949e">loaded: {info}</span>'


with gr.Blocks(title="Arabic OCR — Test Bench") as demo:
    gr.Markdown(
        "## Arabic OCR — Test Bench\n"
        "- Upload, crop, rotate, segment paragraphs, pick polarity.\n"
        "- Optional **Ground Truth** field enables live CER / WER / DotCER scoring.\n"
        "- **Load Random Sample** picks a KHATT test/val image so you can benchmark on known-labeled data."
    )
    with gr.Row():
        editor = gr.ImageEditor(
            label="Upload / crop image",
            sources=["upload", "clipboard", "webcam"],
            image_mode="RGB",
            height=420,
            show_share_button=False,
        )
        with gr.Column():
            with gr.Accordion("Image options", open=True):
                use_crop = gr.Checkbox(value=True, label="Use cropped/edited image")
                angle    = gr.Slider(-30.0, 30.0, value=0.0, step=0.5, label="Rotate (deg)")
                upscale  = gr.Slider(1.0, 3.0, value=1.3, step=0.1, label="Upscale")
                force_multi = gr.Checkbox(value=True, label="Auto-segment into lines")
                polarity = gr.Radio(
                    choices=["Auto (try both)", "Normal (black on white)", "Invert (white on black)"],
                    value="Auto (try both)", label="Polarity",
                )
            with gr.Accordion("Decoder", open=True):
                use_beam = gr.Checkbox(value=True, label="Beam search (else greedy)")
                beam_width = gr.Slider(1, 30, value=10, step=1, label="Beam width")
                lm_weight  = gr.Slider(0.0, 1.0, value=0.3, step=0.05,
                                       label="Bigram LM weight (0 = disabled)")
            with gr.Accordion("Test bench", open=True):
                split = gr.Radio(choices=["test", "val", "train"], value="test", label="KHATT split")
                load_btn = gr.Button("Load random sample", size="sm")
                load_info = gr.HTML()
                gt_text = gr.Textbox(label="Ground truth (optional, enables CER/WER)",
                                     lines=3, rtl=True)
            run_btn = gr.Button("Recognize", variant="primary")

    with gr.Row():
        out_text = gr.Textbox(label="Prediction (RTL)", lines=6, show_copy_button=True, rtl=True)
    metrics = gr.HTML()
    timing  = gr.HTML()
    prev = gr.Image(label="What the model saw (preprocessed line(s))", type="pil")

    run_btn.click(
        fn=infer,
        inputs=[editor, use_crop, angle, upscale, force_multi, polarity,
                use_beam, beam_width, lm_weight, gt_text],
        outputs=[out_text, prev, metrics, timing],
    )
    load_btn.click(
        fn=load_sample_cb,
        inputs=[split],
        outputs=[editor, gt_text, load_info],
    )


if __name__ == "__main__":
    print("Device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
    demo.launch(server_name="127.0.0.1", server_port=7860)
