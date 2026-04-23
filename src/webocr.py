# src/webocr.py
import html as _html
import os
import random
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageOps
from rapidfuzz.distance import Levenshtein as _Lev

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
    mode = "RGB" if any(im.mode != "L" for im in imgs) else "L"
    bg = (255, 255, 255) if mode == "RGB" else 255
    maxw = max(im.width for im in imgs)
    totalh = sum(im.height for im in imgs) + 4 * (len(imgs) - 1)
    canvas = Image.new(mode, (maxw, totalh), bg)
    y = 0
    for im in imgs:
        if im.mode != mode:
            im = im.convert(mode)
        canvas.paste(im, (0, y))
        y += im.height + 4
    return canvas


# ---------------- Confidence heatmap ----------------
def _confidence_strip(logits: torch.Tensor, width: int, height: int = 10) -> Image.Image:
    """Build an RGB strip where each CTC timestep's max softmax probability
    is colorized (red=low, yellow=mid, green=high). Width is repeated to
    match the prepped-image pixel width."""
    probs = logits.softmax(-1).detach().cpu().numpy()
    conf = probs[:, 0, :].max(axis=-1)
    T = conf.shape[0]

    strip = np.zeros((height, width, 3), dtype=np.uint8)
    tile_w = width // T
    extra  = width - tile_w * T
    x = 0
    for t in range(T):
        w = tile_w + (1 if t < extra else 0)
        c = float(conf[t])
        if c < 0.5:
            r, g = 255, int(510 * c)
        else:
            r, g = int(510 * (1.0 - c)), 255
        strip[:, x:x + w] = [r, g, 0]
        x += w
    return Image.fromarray(strip, mode="RGB")


def _attach_strip(prep: Image.Image, strip: Image.Image, gap: int = 2) -> Image.Image:
    base = prep.convert("RGB")
    if strip.width != base.width:
        strip = strip.resize((base.width, strip.height), Image.NEAREST)
    canvas = Image.new("RGB", (base.width, base.height + gap + strip.height), (255, 255, 255))
    canvas.paste(base, (0, 0))
    canvas.paste(strip, (0, base.height + gap))
    return canvas


# ---------------- Preprocessing stages (for the "Pipeline" tab) ----------------
def preprocessing_stages(pil: Image.Image, upscale: float = 1.0,
                         force_invert: bool = False) -> dict:
    """Return each intermediate image in the preprocessing pipeline as a PIL.

    Useful for showing reviewers exactly what the model ingests.
    """
    stages = {"raw": pil.convert("RGB")}
    if upscale and upscale != 1.0:
        w, h = pil.size
        pil = pil.resize((max(1, int(w * upscale)), max(1, int(h * upscale))), Image.BICUBIC)
        stages["upscaled"] = pil.convert("RGB")
    if force_invert:
        pil = ImageOps.invert(pil.convert("RGB"))
        stages["inverted"] = pil.convert("RGB")
    gray = to_grayscale(pil)
    stages["grayscale"] = gray.convert("RGB")
    binar = binarize(gray)
    if np.asarray(binar).mean() < 127:
        binar = ImageOps.invert(binar)
    stages["binarized"] = binar.convert("RGB")
    resized = resize_keep_ratio_height(binar, HEIGHT)
    if resized.width > MAX_W:
        resized = resized.resize((MAX_W, HEIGHT), Image.LANCZOS)
    stages[f"resized H={HEIGHT}"] = resized.convert("RGB")
    padded = pad_width(resized, MAX_W)
    stages[f"padded W={MAX_W}"] = padded.convert("RGB")
    return stages


# ---------------- Save-as-PNG export ----------------
def _render_report_png(pred: str, gt: str, preview: Optional[Image.Image],
                       metrics_txt: str, config_txt: str) -> Optional[str]:
    """Compose preview + prediction + GT + metrics into a single PNG.
    Returns a temp file path (None if preview is missing).
    Arabic glyphs are rasterized via PIL's default font; the goal here is a
    shareable screenshot, not typographic perfection."""
    if preview is None:
        return None
    try:
        from PIL import ImageDraw, ImageFont
    except Exception:
        return None

    try:
        font = ImageFont.truetype("arial.ttf", 22)
        small = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
        small = font

    margin = 18
    text_block = []
    if config_txt:   text_block.append(("cfg", config_txt, small, "#8b949e"))
    if metrics_txt:  text_block.append(("metric", metrics_txt, font, "#e6edf3"))
    if gt.strip():   text_block.append(("gt", "GT: " + gt.strip(), small, "#7ee787"))
    if pred.strip(): text_block.append(("pr", "PR: " + pred.strip(), small, "#f78166"))

    # Measure
    line_h = 28
    text_h = margin * 2 + line_h * len(text_block)
    w = max(preview.width, 800) + margin * 2
    h = preview.height + text_h + margin * 3

    canvas = Image.new("RGB", (w, h), (13, 17, 23))
    canvas.paste(preview.convert("RGB"), (margin, margin))

    draw = ImageDraw.Draw(canvas)
    y = preview.height + margin * 2
    for _tag, txt, fnt, color in text_block:
        draw.text((margin, y), txt, font=fnt, fill=color)
        y += line_h

    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    canvas.save(tmp.name, "PNG")
    tmp.close()
    return tmp.name


# ---------------- Curated KHATT examples ----------------
def get_example_list() -> list:
    """Pre-curated KHATT samples useful for demos: dot-confusion, multi-line, etc.

    Returns a list of [filename, label, description] triples drawn from the
    KHATT val split. Only items whose image exists locally are returned.
    """
    curated = [
        ("AHTD3A0019_Para1_1.jpg", "dot-group heavy sample"),
        ("AHTD3A0023_Para1_1.jpg", "typical line"),
        ("AHTD3A0040_Para1_1.jpg", "multi-word"),
        ("AHTD3A0055_Para1_1.jpg", "dense handwriting"),
        ("AHTD3A0089_Para1_1.jpg", "faint scan"),
    ]
    rows = []
    for fname, note in curated:
        img_path = os.path.join(IMAGES_DIR, fname)
        lbl_path = os.path.join("./archive/labels", os.path.splitext(fname)[0] + ".txt")
        if not os.path.exists(img_path):
            continue
        try:
            gt = read_label(lbl_path)
        except Exception:
            gt = ""
        rows.append([fname, gt, note])
    return rows


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
def _forward_logits(im_pil: Image.Image) -> torch.Tensor:
    x = to_tensor(im_pil).unsqueeze(0).to(device)
    return model(x)  # [T, 1, C]


def _decode_from_logits(logits: torch.Tensor, use_beam: bool = False,
                        beam_width: int = 10, lm_weight: float = 0.0) -> str:
    if use_beam:
        lm = get_bigram_lm() if lm_weight > 0 else None
        hyp_ltr = ctc_beam_decode(logits, id2char,
                                  beam_width=beam_width,
                                  bigram_lm=lm,
                                  lm_weight=lm_weight)[0]
    else:
        hyp_ltr = ctc_greedy_decode(logits, id2char)[0]
    return hyp_ltr[::-1]  # back to RTL


def _decode_one(im_pil: Image.Image, device: torch.device, id2char,
                use_beam: bool = False, beam_width: int = 10,
                lm_weight: float = 0.0) -> str:
    """Forward + decode in one call. Kept for call sites that don't need the logits."""
    return _decode_from_logits(_forward_logits(im_pil), use_beam, beam_width, lm_weight)


@torch.inference_mode()
def recognize_image(
    pil_img: Image.Image,
    upscale: float = 1.0,
    force_multiline: bool = True,
    polarity_mode: str = "auto",
    use_beam: bool = False,
    beam_width: int = 10,
    lm_weight: float = 0.0,
    show_heatmap: bool = True,
) -> Tuple[str, Image.Image]:
    lines = segment_into_lines(pil_img) if force_multiline else [pil_img]

    previews: List[Image.Image] = []
    texts: List[str] = []

    for ln in lines:
        if polarity_mode == "normal":
            prep_best = prep_line(ln, upscale=upscale, force_invert=False)
            logits_best = _forward_logits(prep_best)

        elif polarity_mode == "invert":
            prep_best = prep_line(ln, upscale=upscale, force_invert=True)
            logits_best = _forward_logits(prep_best)

        else:  # auto: forward both, pick longer greedy
            prep_n = prep_line(ln, upscale=upscale, force_invert=False)
            prep_i = prep_line(ln, upscale=upscale, force_invert=True)
            ln_ = _forward_logits(prep_n); li_ = _forward_logits(prep_i)
            tn = _decode_from_logits(ln_, use_beam=False)
            ti = _decode_from_logits(li_, use_beam=False)
            if len(ti.strip()) > len(tn.strip()):
                prep_best, logits_best = prep_i, li_
            else:
                prep_best, logits_best = prep_n, ln_

        txt_best = _decode_from_logits(logits_best, use_beam=use_beam,
                                       beam_width=beam_width, lm_weight=lm_weight)

        if show_heatmap:
            strip = _confidence_strip(logits_best, width=prep_best.width)
            preview = _attach_strip(prep_best, strip)
        else:
            preview = prep_best

        previews.append(preview)
        texts.append(txt_best)

    stacked = stack_preview(previews)
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


_DIFF_CSS = """
<style>
  .diff-wrap { direction:rtl; text-align:right; font-size:1.35em; line-height:1.8;
               font-family:'Geeza Pro','Arabic Typesetting','Segoe UI',sans-serif;
               background:#0d1117; color:#e6edf3; padding:12px 14px;
               border:1px solid #30363d; border-radius:8px; margin-top:6px; }
  .diff-wrap .row { padding:2px 0; }
  .diff-wrap .lab { direction:ltr; display:inline-block; font-family:ui-monospace,monospace;
                     font-size:0.55em; color:#8b949e; background:#21262d;
                     border-radius:3px; padding:2px 6px; margin-inline-end:8px;
                     vertical-align:middle; }
  .diff-wrap .eq { }
  .diff-wrap .sub { background:rgba(210,153,34,0.30); color:#d29922;
                     border-radius:3px; padding:0 2px; }
  .diff-wrap .del { background:rgba(248,81,73,0.30); color:#f85149;
                     border-radius:3px; padding:0 2px; text-decoration:line-through; }
  .diff-wrap .ins { background:rgba(248,81,73,0.30); color:#f85149;
                     border-radius:3px; padding:0 2px; }
  .diff-wrap .legend { direction:ltr; text-align:left; font-size:0.55em;
                        color:#8b949e; margin-top:8px; font-family:ui-monospace,monospace; }
</style>
"""


def _char_html(ch: str, cls: str) -> str:
    """Escape and wrap a single char. Render space as a visible middle-dot when not equal."""
    esc = _html.escape(ch) if ch != " " else "&nbsp;"
    return f'<span class="{cls}">{esc}</span>'


def _diff_html(gt: str, pred: str) -> str:
    """Character-aligned diff of GT vs PR — green=equal, yellow=sub, red=del/ins."""
    if not gt.strip():
        return ""

    ops = _Lev.editops(gt, pred)
    gi = pi = 0
    gt_spans: List[str] = []
    pr_spans: List[str] = []

    for tag, i1, i2 in ops:
        # Equal run before this op — by editops invariant, both advance in lockstep.
        while gi < i1 and pi < i2:
            gt_spans.append(_char_html(gt[gi], "eq"))
            pr_spans.append(_char_html(pred[pi], "eq"))
            gi += 1; pi += 1
        if tag == "replace":
            gt_spans.append(_char_html(gt[i1], "sub"))
            pr_spans.append(_char_html(pred[i2], "sub"))
            gi += 1; pi += 1
        elif tag == "delete":
            gt_spans.append(_char_html(gt[i1], "del"))
            gi += 1
        elif tag == "insert":
            pr_spans.append(_char_html(pred[i2], "ins"))
            pi += 1

    # Trailing equal tail
    while gi < len(gt) and pi < len(pred):
        gt_spans.append(_char_html(gt[gi], "eq"))
        pr_spans.append(_char_html(pred[pi], "eq"))
        gi += 1; pi += 1
    while gi < len(gt):
        gt_spans.append(_char_html(gt[gi], "del"))
        gi += 1
    while pi < len(pred):
        pr_spans.append(_char_html(pred[pi], "ins"))
        pi += 1

    return (
        _DIFF_CSS
        + '<div class="diff-wrap">'
        + f'<div class="row"><span class="lab">GT</span>{"".join(gt_spans)}</div>'
        + f'<div class="row"><span class="lab">PR</span>{"".join(pr_spans)}</div>'
        + '<div class="legend">'
          'green = match &middot; '
          '<span style="color:#d29922">yellow = substitution</span> &middot; '
          '<span style="color:#f85149">red = deletion/insertion</span></div>'
        + '</div>'
    )


def _err(msg: str):
    return (f'<div style="color:#f85149;padding:8px 10px;background:#2a1414;'
            f'border:1px solid #5d2324;border-radius:6px">⚠ {msg}</div>')


def infer(editor_payload: Any, use_cropped: bool, angle: float, upscale: float,
          force_multiline: bool, polarity_label: str, use_beam: bool,
          beam_width: int, lm_weight: float, show_heatmap: bool, gt_text: str):
    if editor_payload is None:
        return "", None, _err("No image — upload, paste, or use Load Random Sample."), "", ""
    try:
        pil = ensure_pil_from_editor(editor_payload, use_cropped=use_cropped)
    except Exception as e:
        return "", None, _err(f"Unsupported image format: {e}"), "", ""
    if pil is None or pil.size == (0, 0):
        return "", None, _err("Image is empty."), "", ""

    pil = rotate_if_needed(pil, angle)
    mode = _map_polarity(polarity_label)
    t0 = time.perf_counter()
    txt, prev = recognize_image(
        pil, upscale=upscale, force_multiline=force_multiline,
        polarity_mode=mode, use_beam=use_beam,
        beam_width=int(beam_width), lm_weight=float(lm_weight),
        show_heatmap=bool(show_heatmap),
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    info = f'<span style="color:#8b949e">decoded in {elapsed_ms:.0f} ms &middot; device={device.type}</span>'
    metrics_html = _fmt_metrics(gt_text or "", txt)
    diff_html = _diff_html(gt_text or "", txt)
    return txt, prev, metrics_html, info, diff_html


# ---------------- Decoder comparison ----------------
def _decode_with(prepped_lines: List[Image.Image], use_beam: bool,
                 beam_width: int, lm_weight: float) -> str:
    texts = []
    for img in prepped_lines:
        texts.append(_decode_one(img, device, id2char,
                                 use_beam=use_beam,
                                 beam_width=beam_width,
                                 lm_weight=lm_weight))
    return "\n".join(texts)


@torch.inference_mode()
def compare_decoders(editor_payload: Any, use_cropped: bool, angle: float, upscale: float,
                     force_multiline: bool, polarity_label: str,
                     beam_width: int, lm_weight: float, gt_text: str):
    if editor_payload is None:
        return "_upload an image first_"
    try:
        pil = ensure_pil_from_editor(editor_payload, use_cropped=use_cropped)
    except Exception as e:
        return f"unsupported image: {e}"

    pil = rotate_if_needed(pil, angle)
    mode = _map_polarity(polarity_label)
    lines = segment_into_lines(pil) if force_multiline else [pil]

    # Prep once per line using the greedy-longer-polarity pick (stable choice)
    prepped: List[Image.Image] = []
    for ln in lines:
        if mode == "normal":
            prepped.append(prep_line(ln, upscale=upscale, force_invert=False))
        elif mode == "invert":
            prepped.append(prep_line(ln, upscale=upscale, force_invert=True))
        else:
            pn = prep_line(ln, upscale=upscale, force_invert=False)
            pi = prep_line(ln, upscale=upscale, force_invert=True)
            tn = _decode_one(pn, device, id2char, use_beam=False)
            ti = _decode_one(pi, device, id2char, use_beam=False)
            prepped.append(pi if len(ti.strip()) > len(tn.strip()) else pn)

    configs = [
        ("greedy", dict(use_beam=False, beam_width=1, lm_weight=0.0)),
        (f"beam({int(beam_width)})",
            dict(use_beam=True, beam_width=int(beam_width), lm_weight=0.0)),
        (f"beam({int(beam_width)})+lm({float(lm_weight):.2f})",
            dict(use_beam=True, beam_width=int(beam_width), lm_weight=float(lm_weight))),
    ]

    rows = []
    best_cer = None
    gt = (gt_text or "").strip()
    for name, kw in configs:
        t0 = time.perf_counter()
        txt = _decode_with(prepped, **kw)
        ms = (time.perf_counter() - t0) * 1000
        if gt:
            c = cer_raw(gt, txt) * 100
            best_cer = c if best_cer is None else min(best_cer, c)
            rows.append((name, txt, c, ms))
        else:
            rows.append((name, txt, None, ms))

    # Markdown table with RTL-safe prediction cells
    header = "| Config | Prediction | CER | Time |\n|---|---|---|---|\n"
    body_lines = []
    for name, txt, c, ms in rows:
        safe = txt.replace("|", "\\|").replace("\n", " ⏎ ")
        pred_cell = f'<span style="direction:rtl; unicode-bidi:embed">{safe}</span>'
        if c is None:
            cer_cell = "—"
        else:
            mark = " ⭐" if c == best_cer else ""
            cer_cell = f"{c:.2f}%{mark}"
        body_lines.append(f"| `{name}` | {pred_cell} | {cer_cell} | {ms:.0f} ms |")
    return header + "\n".join(body_lines)


def load_sample_cb(split: str):
    img, gt, info = load_random_sample(split)
    # Gradio ImageEditor accepts None or a dict-like; passing a PIL directly works for upload display.
    return img, gt, f'<span style="color:#8b949e">loaded: {info}</span>'


# ---------------- Preprocessing tab handler ----------------
def show_pipeline_stages(editor_payload: Any, use_cropped: bool, angle: float,
                         upscale: float, polarity_label: str):
    if editor_payload is None:
        return [], _err("Upload an image first.")
    try:
        pil = ensure_pil_from_editor(editor_payload, use_cropped=use_cropped)
    except Exception as e:
        return [], _err(f"Unsupported image: {e}")
    pil = rotate_if_needed(pil, angle)
    pol = _map_polarity(polarity_label)
    force_invert = (pol == "invert")
    stages = preprocessing_stages(pil, upscale=upscale, force_invert=force_invert)
    # Gradio Gallery accepts list of (image, caption)
    gallery = [(img, name) for name, img in stages.items()]
    return gallery, ""


# ---------------- Save-report callback ----------------
def save_report_cb(pred: str, gt: str, preview: Optional[Image.Image],
                   metrics_html: str, timing_html: str):
    # Strip tags from HTML for a plain-text summary line
    import re as _re
    def _plain(h): return _re.sub(r"<[^>]+>", "", h or "").strip()
    metrics_txt = _plain(metrics_html)
    cfg_txt = _plain(timing_html)
    path = _render_report_png(pred or "", gt or "", preview, metrics_txt, cfg_txt)
    if path is None:
        return None
    return path


# ---------------- Batch-mode handler ----------------
@torch.inference_mode()
def run_batch(files: list, use_beam: bool, beam_width: int, lm_weight: float,
              force_multiline: bool, polarity_label: str, upscale: float):
    import zipfile
    import tempfile
    import csv as _csv

    if not files:
        return None, _err("Select image files or a ZIP — then click Run batch.")

    # Collect (filename, image_bytes) pairs. Accept: single images, multiple images, ZIPs.
    items: List[Tuple[str, Image.Image]] = []
    for f in files:
        # Gradio passes filepath strings for type="filepath"
        path = f if isinstance(f, str) else getattr(f, "name", None)
        if not path or not os.path.exists(path):
            continue
        low = path.lower()
        if low.endswith(".zip"):
            try:
                with zipfile.ZipFile(path, "r") as z:
                    for name in z.namelist():
                        nlow = name.lower()
                        if not nlow.endswith((".png", ".jpg", ".jpeg")):
                            continue
                        data = z.read(name)
                        try:
                            im = Image.open(__import__("io").BytesIO(data)).convert("RGB")
                        except Exception:
                            continue
                        items.append((os.path.basename(name), im))
            except Exception:
                continue
        elif low.endswith((".png", ".jpg", ".jpeg")):
            try:
                items.append((os.path.basename(path), Image.open(path).convert("RGB")))
            except Exception:
                continue

    if not items:
        return None, _err("No supported images found (.png/.jpg/.jpeg, directly or inside .zip).")

    pol = _map_polarity(polarity_label)
    rows = []
    total_cer = 0.0; n_with_gt = 0
    t0 = time.perf_counter()
    for fname, im in items:
        try:
            txt, _ = recognize_image(
                im, upscale=float(upscale), force_multiline=bool(force_multiline),
                polarity_mode=pol, use_beam=bool(use_beam),
                beam_width=int(beam_width), lm_weight=float(lm_weight),
                show_heatmap=False,
            )
        except Exception as e:
            rows.append({"filename": fname, "prediction": f"[error: {e}]",
                         "gt": "", "cer": ""})
            continue

        # Try to pair with a KHATT label file if this filename exists in archive/labels
        gt = ""
        base = os.path.splitext(fname)[0]
        lbl = os.path.join("./archive/labels", base + ".txt")
        if os.path.exists(lbl):
            try:
                gt = read_label(lbl)
            except Exception:
                gt = ""
        cer_s = ""
        if gt:
            c = cer_raw(gt, txt)
            total_cer += c; n_with_gt += 1
            cer_s = f"{c:.4f}"
        rows.append({"filename": fname, "prediction": txt, "gt": gt, "cer": cer_s})
    elapsed = time.perf_counter() - t0

    tmp = tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False,
                                      encoding="utf-8", newline="")
    writer = _csv.DictWriter(tmp, delimiter="\t",
                             fieldnames=["filename", "prediction", "gt", "cer"])
    writer.writeheader()
    for r in rows:
        r["prediction"] = (r["prediction"] or "").replace("\t", " ").replace("\n", " ")
        r["gt"] = (r["gt"] or "").replace("\t", " ").replace("\n", " ")
        writer.writerow(r)
    tmp.close()

    summary_bits = [f"{len(rows)} samples &middot; {elapsed:.1f}s total"]
    if n_with_gt:
        summary_bits.append(f"avg CER {100 * total_cer / n_with_gt:.2f}% over {n_with_gt} labeled")
    summary = ('<div style="color:#3fb950;font-family:ui-monospace,monospace;font-size:0.9em">✓ '
               + ' &middot; '.join(summary_bits) + '</div>')
    return tmp.name, summary


with gr.Blocks(title="Arabic OCR — Test Bench") as demo:
    gr.Markdown(
        "## Arabic OCR — Test Bench\n"
        "Single image, batch folder/ZIP, KHATT sample loader, live CER/WER, "
        "character diff, confidence heatmap, pipeline preview, decoder comparison."
    )
    with gr.Tabs():
        # =============== Single image tab ===============
        with gr.Tab("Single"):
            with gr.Row():
                editor = gr.ImageEditor(
                    label="Upload / crop image",
                    sources=["upload", "clipboard", "webcam"],
                    image_mode="RGB",
                    height=420,
                )
                with gr.Column():
                    with gr.Accordion("Image options", open=True):
                        use_crop = gr.Checkbox(value=True, label="Use cropped/edited image")
                        angle    = gr.Slider(-30.0, 30.0, value=0.0, step=0.5, label="Rotate (deg)")
                        upscale  = gr.Slider(1.0, 3.0, value=1.3, step=0.1, label="Upscale")
                        force_multi = gr.Checkbox(value=True, label="Auto-segment into lines")
                        polarity = gr.Radio(
                            choices=["Auto (try both)", "Normal (black on white)",
                                     "Invert (white on black)"],
                            value="Auto (try both)", label="Polarity",
                        )
                    with gr.Accordion("Decoder", open=True):
                        use_beam = gr.Checkbox(value=True, label="Beam search (else greedy)")
                        beam_width = gr.Slider(1, 30, value=10, step=1, label="Beam width")
                        lm_weight  = gr.Slider(0.0, 1.0, value=0.3, step=0.05,
                                               label="Bigram LM weight (0 = disabled)")
                        show_heatmap = gr.Checkbox(value=True, label="Show confidence heatmap")
                    with gr.Accordion("Test bench", open=True):
                        split = gr.Radio(choices=["test", "val", "train"], value="test",
                                         label="KHATT split")
                        load_btn = gr.Button("Load random sample", size="sm")
                        load_info = gr.HTML()
                        gt_text = gr.Textbox(label="Ground truth (optional, enables CER/WER)",
                                             lines=3, rtl=True)
                    with gr.Row():
                        run_btn = gr.Button("Recognize", variant="primary")
                        compare_btn = gr.Button("Compare decoders", variant="secondary")
                        save_btn = gr.Button("Save as PNG", variant="secondary")

            with gr.Row():
                out_text = gr.Textbox(label="Prediction (RTL)", lines=6, rtl=True)
            metrics = gr.HTML()
            timing  = gr.HTML()
            diff_view = gr.HTML()
            compare_table = gr.Markdown()
            prev = gr.Image(label="What the model saw (preprocessed line(s) + "
                                  "confidence heatmap)", type="pil")
            saved_file = gr.File(label="Report PNG", visible=True)

        # =============== Pipeline tab ===============
        with gr.Tab("Pipeline"):
            gr.Markdown("Shows each intermediate step the preprocessing pipeline applies. "
                        "Use the same image from the Single tab (upload above first).")
            pipe_btn = gr.Button("Show pipeline stages", variant="primary")
            pipe_err = gr.HTML()
            pipe_gallery = gr.Gallery(label="Pipeline", columns=2, rows=4,
                                      object_fit="contain", height=520)

        # =============== Batch tab ===============
        with gr.Tab("Batch"):
            gr.Markdown(
                "Upload multiple images, or a single ZIP containing images. "
                "If a matching KHATT label exists in `archive/labels/` for each "
                "filename, per-sample CER and an average CER are included in the TSV."
            )
            with gr.Row():
                batch_files = gr.File(label="Images or ZIP", file_count="multiple",
                                      file_types=[".png", ".jpg", ".jpeg", ".zip"])
            with gr.Row():
                batch_use_beam = gr.Checkbox(value=True, label="Beam search")
                batch_beam_w = gr.Slider(1, 30, value=10, step=1, label="Beam width")
                batch_lm_w = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="LM weight")
            with gr.Row():
                batch_multi = gr.Checkbox(value=True, label="Auto-segment lines")
                batch_pol = gr.Radio(
                    choices=["Auto (try both)", "Normal (black on white)",
                             "Invert (white on black)"],
                    value="Auto (try both)", label="Polarity",
                )
                batch_upscale = gr.Slider(1.0, 3.0, value=1.3, step=0.1, label="Upscale")
            batch_run = gr.Button("Run batch", variant="primary")
            batch_status = gr.HTML()
            batch_tsv = gr.File(label="Results TSV")

        # =============== Examples tab ===============
        with gr.Tab("Examples"):
            gr.Markdown("Pre-curated KHATT samples. Click a row to load its image + GT "
                        "into the Single tab.")
            example_rows = get_example_list()
            if example_rows:
                example_df = gr.Dataframe(
                    headers=["filename", "ground truth", "notes"],
                    value=example_rows, wrap=True, interactive=False,
                )
            else:
                gr.Markdown("_No curated samples found locally._")
                example_df = gr.Dataframe(visible=False)

    # ---- wiring ----
    run_btn.click(
        fn=infer,
        inputs=[editor, use_crop, angle, upscale, force_multi, polarity,
                use_beam, beam_width, lm_weight, show_heatmap, gt_text],
        outputs=[out_text, prev, metrics, timing, diff_view],
    )
    compare_btn.click(
        fn=compare_decoders,
        inputs=[editor, use_crop, angle, upscale, force_multi, polarity,
                beam_width, lm_weight, gt_text],
        outputs=[compare_table],
    )
    save_btn.click(
        fn=save_report_cb,
        inputs=[out_text, gt_text, prev, metrics, timing],
        outputs=[saved_file],
    )
    load_btn.click(
        fn=load_sample_cb,
        inputs=[split],
        outputs=[editor, gt_text, load_info],
    )
    pipe_btn.click(
        fn=show_pipeline_stages,
        inputs=[editor, use_crop, angle, upscale, polarity],
        outputs=[pipe_gallery, pipe_err],
    )
    batch_run.click(
        fn=run_batch,
        inputs=[batch_files, batch_use_beam, batch_beam_w, batch_lm_w,
                batch_multi, batch_pol, batch_upscale],
        outputs=[batch_tsv, batch_status],
    )

    # Click a row in Examples → load that sample into the Single tab
    def _example_to_editor(evt: gr.SelectData):
        if evt is None or evt.index is None:
            return gr.update(), gr.update()
        row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
        rows = get_example_list()
        if row_idx >= len(rows):
            return gr.update(), gr.update()
        fname, gt, _ = rows[row_idx]
        img_path = os.path.join(IMAGES_DIR, fname)
        try:
            im = Image.open(img_path).convert("RGB")
        except Exception:
            return gr.update(), gr.update()
        return im, gt

    if example_rows:
        example_df.select(
            fn=_example_to_editor,
            inputs=None,
            outputs=[editor, gt_text],
        )


if __name__ == "__main__":
    print("Device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")
    demo.launch(server_name="127.0.0.1", server_port=7860)
