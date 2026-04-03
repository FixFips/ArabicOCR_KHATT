# src/webocr.py
import os
from typing import Any, List, Tuple

import numpy as np
import cv2
from PIL import Image, ImageOps

import torch
from torchvision import transforms
import gradio as gr

from .model import CRNN, ctc_greedy_decode, ctc_beam_decode
from .preprocess import to_grayscale, binarize, normalize, resize_keep_ratio_height, pad_width


# ---------------- Config (match training) ----------------
HEIGHT = 96
MAX_W  = 1536
CKPT   = "./runs/exp1/crnn_best.pt"


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
model = CRNN(num_classes=len(vocab)).to(device)
model.load_state_dict(state_dict)
model.eval()
to_tensor = transforms.ToTensor()


# ---------------- Decode helpers ----------------
def _decode_one(im_pil: Image.Image, device: torch.device, id2char, use_beam: bool = False) -> str:
    x = to_tensor(im_pil).unsqueeze(0).to(device)
    logits = model(x)
    if use_beam:
        hyp_ltr = ctc_beam_decode(logits, id2char, beam_width=10)[0]
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
) -> Tuple[str, Image.Image]:
    lines = segment_into_lines(pil_img) if force_multiline else [pil_img]

    selected_prepped: List[Image.Image] = []
    texts: List[str] = []

    for ln in lines:
        if polarity_mode == "normal":
            prep_n = prep_line(ln, upscale=upscale, force_invert=False)
            txt_n = _decode_one(prep_n, device, id2char, use_beam=use_beam)
            prep_best, txt_best = prep_n, txt_n

        elif polarity_mode == "invert":
            prep_i = prep_line(ln, upscale=upscale, force_invert=True)
            txt_i = _decode_one(prep_i, device, id2char, use_beam=use_beam)
            prep_best, txt_best = prep_i, txt_i

        else:  # auto: try both, choose longer non-blank
            prep_n = prep_line(ln, upscale=upscale, force_invert=False)
            txt_n = _decode_one(prep_n, device, id2char, use_beam=use_beam)

            prep_i = prep_line(ln, upscale=upscale, force_invert=True)
            txt_i = _decode_one(prep_i, device, id2char, use_beam=use_beam)

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

def infer(editor_payload: Any, use_cropped: bool, angle: float, upscale: float,
          force_multiline: bool, polarity_label: str, use_beam: bool):
    if editor_payload is None:
        return "", None
    try:
        pil = ensure_pil_from_editor(editor_payload, use_cropped=use_cropped)
    except Exception as e:
        return f"Unsupported image format ({type(editor_payload)}): {e}", None

    pil = rotate_if_needed(pil, angle)
    mode = _map_polarity(polarity_label)
    txt, prev = recognize_image(pil, upscale=upscale, force_multiline=force_multiline,
                                polarity_mode=mode, use_beam=use_beam)
    return txt, prev


with gr.Blocks(title="Arabic OCR — Upload / Crop / Recognize") as demo:
    gr.Markdown(
        "## Arabic OCR — Upload / Crop / Recognize\n"
        "- Uses **CLAHE + dual-polarity Otsu** preprocessing (same as training).\n"
        "- Multi-scale vertical encoding preserves Arabic dot positions.\n"
        "- You can **crop**, **rotate**, **upscale** small text, **auto-split** paragraphs, "
        "and pick **polarity** (Auto / Normal / Invert)."
    )
    with gr.Row():
        editor = gr.ImageEditor(
            label="Upload & (optional) crop selection",
            sources=["upload", "clipboard", "webcam"],
            image_mode="RGB",
            height=480,
            show_download_button=False,
            show_share_button=False,
        )
        with gr.Column():
            use_crop = gr.Checkbox(value=True, label="Use cropped/edited image (uncheck = use original/background)")
            angle    = gr.Slider(-30.0, 30.0, value=0.0, step=0.5, label="Rotate (degrees)")
            upscale  = gr.Slider(1.0, 3.0, value=1.3, step=0.1, label="Upscale before OCR (helps tiny text)")
            force_multi = gr.Checkbox(value=True, label="Auto-segment into lines (paragraphs)")
            polarity = gr.Radio(
                choices=["Auto (try both)", "Normal (black on white)", "Invert (white on black)"],
                value="Auto (try both)",
                label="Polarity"
            )
            use_beam = gr.Checkbox(value=True, label="Use beam search (slower but more accurate)")
            run_btn = gr.Button("Recognize", variant="primary")
            out_text = gr.Textbox(label="Prediction (RTL)", lines=8, show_copy_button=True)
    prev = gr.Image(label="What the model saw (preprocessed line(s))", type="pil")

    run_btn.click(
        fn=infer,
        inputs=[editor, use_crop, angle, upscale, force_multi, polarity, use_beam],
        outputs=[out_text, prev]
    )


if __name__ == "__main__":
    print("Device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
    demo.launch(server_name="127.0.0.1", server_port=7860)
