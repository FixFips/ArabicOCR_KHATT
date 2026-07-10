# arabicocr_khatt/synthetic.py
"""
Synthetic Arabic handwriting line renderer (Option A).

Renders a text line with real OpenType shaping (uharfbuzz) + rasterization
(freetype-py), then applies handwriting-izing degradations:

  render words at ~4x scale (H≈400) with per-word baseline jitter and
  randomized inter-word gaps  ->  slant shear  ->  pen-width morph  ->
  elastic warp  ->  downscale to scan resolution (H 120-200)  ->
  noise / blur / JPEG round-trip

Design notes (tied to the run-2 error decomposition):
  - Inter-word gap width is *heavily* randomized (tight to wide, sometimes
    near-touching): space del/ins is 24% of all test edit ops.
  - Per-word rendering is shaping-safe: Arabic joining never crosses a space,
    so shaping words independently equals shaping the whole line.
  - Elastic warp runs at ~4x scale where dots are 12-20 px, so it cannot
    destroy them (they are 2-4 px only at H=96; see augment.py BANNED list).
  - Lines containing digits or Latin are rejected: HarfBuzz does not do bidi
    reordering, and mixed-direction lines would render in the wrong visual
    order. Real KHATT data still covers digits.

Standalone module: numpy, cv2, PIL, uharfbuzz, freetype only (no torch).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import freetype
import uharfbuzz as hb

# ----------------------------------------------------------------------------
# Fonts
# ----------------------------------------------------------------------------

FONTS_DIR = Path(__file__).resolve().parent.parent / "archive" / "fonts"

# Family pick weight: emphasize hand-like styles (ruqaa, Kano hand), keep
# print-like naskh for coverage, de-emphasize display/nastaliq.
FAMILY_WEIGHTS = {
    "ArefRuqaa": 3.0,
    "ArefRuqaaInk": 2.0,
    "Alkalami": 3.0,
    "Mirza": 2.5,
    "Lateef": 2.0,
    "ScheherazadeNew": 2.0,
    "Harmattan": 1.5,
    "Katibeh": 1.0,
    "Rakkas": 1.0,
    "NotoNastaliqUrdu": 0.75,
}


def font_family(path: Path) -> str:
    """'ArefRuqaaInk-Bold.ttf' -> 'ArefRuqaaInk'; 'Noto[wght].ttf' -> 'Noto'."""
    stem = path.stem
    return stem.split("-")[0].split("[")[0]


def list_fonts(fonts_dir: Path = FONTS_DIR) -> dict[str, list[Path]]:
    """Map family name -> list of .ttf paths."""
    families: dict[str, list[Path]] = {}
    for p in sorted(Path(fonts_dir).glob("*.ttf")):
        families.setdefault(font_family(p), []).append(p)
    return families


# ----------------------------------------------------------------------------
# Text eligibility / cleanup
# ----------------------------------------------------------------------------

_WS_RUN = re.compile(r"\s+")
_REJECT = re.compile(r"[A-Za-z0-9٠-٩۰-۹]")  # Latin + digits


def clean_synth_text(s: str) -> str | None:
    """Normalize a KHATT label into renderable synth text, or None to reject.

    - strips tatweel (elongation comes from image-space kashida augment
      instead, so labels stay tatweel-free)
    - collapses whitespace
    - rejects lines with Latin/digits (single-run RTL shaping assumption)
    """
    s = s.replace("ـ", "")
    s = _WS_RUN.sub(" ", s).strip()
    if not s or _REJECT.search(s):
        return None
    # must contain at least a few Arabic letters
    n_arab = sum(1 for ch in s if "؀" <= ch <= "ۿ")
    if n_arab < 3:
        return None
    return s


# ----------------------------------------------------------------------------
# Core renderer
# ----------------------------------------------------------------------------


@dataclass
class _WordInk:
    alpha: np.ndarray  # uint8 [H, W] coverage (0 transparent, 255 full ink)
    baseline: int      # row index of the baseline inside `alpha`


class LineRenderer:
    """Shape + rasterize Arabic lines. One instance per process (has caches)."""

    def __init__(self, fonts_dir: Path = FONTS_DIR):
        self.families = list_fonts(fonts_dir)
        if not self.families:
            raise FileNotFoundError(
                f"No .ttf fonts in {fonts_dir} — run scripts/download_fonts.py"
            )
        self._hb: dict[Path, tuple[hb.Font, int]] = {}  # path -> (font, upem)
        self._ft: dict[Path, freetype.Face] = {}
        self._glyphs: dict[tuple, tuple] = {}  # (path, px, gid) -> (arr, left, top)

    # -- font handles --------------------------------------------------------
    def _hb_font(self, path: Path) -> tuple[hb.Font, int]:
        f = self._hb.get(path)
        if f is None:
            blob = hb.Blob.from_file_path(str(path))
            face = hb.Face(blob)
            f = (hb.Font(face), face.upem)
            self._hb[path] = f
        return f

    def _ft_face(self, path: Path) -> freetype.Face:
        f = self._ft.get(path)
        if f is None:
            f = freetype.Face(str(path))
            self._ft[path] = f
        return f

    def pick_font(self, rng: np.random.Generator) -> Path:
        fams = list(self.families)
        w = np.array([FAMILY_WEIGHTS.get(f, 1.0) for f in fams], dtype=np.float64)
        fam = fams[rng.choice(len(fams), p=w / w.sum())]
        files = self.families[fam]
        return files[int(rng.integers(len(files)))]

    # -- shaping / rasterizing ----------------------------------------------
    def _shape(self, text: str, path: Path):
        buf = hb.Buffer()
        buf.add_str(text)
        buf.direction = "rtl"
        buf.script = "Arab"
        buf.language = "ar"
        hb.shape(self._hb_font(path)[0], buf, None)
        return buf.glyph_infos, buf.glyph_positions

    def supports(self, text: str, path: Path) -> bool:
        """True if the font has real glyphs (no .notdef) for every char."""
        infos, _ = self._shape(text.replace(" ", ""), path)
        return bool(infos) and all(i.codepoint != 0 for i in infos)

    def _glyph_bitmap(self, path: Path, px: int, gid: int):
        key = (path, px, gid)
        hit = self._glyphs.get(key)
        if hit is not None:
            return hit
        face = self._ft_face(path)
        face.set_pixel_sizes(0, px)
        face.load_glyph(gid, freetype.FT_LOAD_RENDER | freetype.FT_LOAD_NO_HINTING)
        slot = face.glyph
        bmp = slot.bitmap
        if bmp.rows == 0 or bmp.width == 0:
            arr = np.zeros((0, 0), dtype=np.uint8)
        else:
            arr = np.array(bmp.buffer, dtype=np.uint8).reshape(bmp.rows, bmp.pitch)
            arr = arr[:, : bmp.width].copy()
        val = (arr, slot.bitmap_left, slot.bitmap_top)
        if len(self._glyphs) > 30000:  # crude cap; generation uses many procs
            self._glyphs.clear()
        self._glyphs[key] = val
        return val

    def _render_word(self, word: str, path: Path, px: int) -> _WordInk | None:
        """Rasterize one space-free chunk onto an alpha canvas."""
        infos, poss = self._shape(word, path)
        if not infos:
            return None
        s = px / self._hb_font(path)[1]
        placed = []  # (arr, x, y_top_rel_baseline)
        pen_x = pen_y = 0.0
        min_x = min_y = 1e9
        max_x = max_y = -1e9
        for info, pos in zip(infos, poss):
            arr, left, top = self._glyph_bitmap(path, px, info.codepoint)
            gx = pen_x + pos.x_offset * s + left
            gy = pen_y - pos.y_offset * s - top  # y down, relative to baseline
            if arr.size:
                placed.append((arr, gx, gy))
                min_x = min(min_x, gx)
                min_y = min(min_y, gy)
                max_x = max(max_x, gx + arr.shape[1])
                max_y = max(max_y, gy + arr.shape[0])
            pen_x += pos.x_advance * s
            pen_y -= pos.y_advance * s
        if not placed:
            return None
        w = int(np.ceil(max_x - min_x)) + 2
        h = int(np.ceil(max_y - min_y)) + 2
        canvas = np.zeros((h, w), dtype=np.uint8)
        for arr, gx, gy in placed:
            x0 = int(round(gx - min_x))
            y0 = int(round(gy - min_y))
            roi = canvas[y0 : y0 + arr.shape[0], x0 : x0 + arr.shape[1]]
            np.maximum(roi, arr[: roi.shape[0], : roi.shape[1]], out=roi)
        return _WordInk(alpha=canvas, baseline=int(round(-min_y)))

    # -- line composition -----------------------------------------------------
    def render_line(
        self,
        text: str,
        rng: np.random.Generator,
        font_path: Path | None = None,
        px: int | None = None,
    ) -> tuple[np.ndarray, Path] | None:
        """Compose a full line at render scale. Returns (grayscale u8, font)."""
        if font_path is None:
            for _ in range(4):  # retry until a font covers the charset
                cand = self.pick_font(rng)
                if self.supports(text, cand):
                    font_path = cand
                    break
            else:
                return None
        elif not self.supports(text, font_path):
            return None

        px = px or int(rng.integers(95, 150))
        words = text.split(" ")
        inks: list[_WordInk] = []
        for wtext in words:
            wi = self._render_word(wtext, font_path, px)
            if wi is None:
                return None
            inks.append(wi)

        # Inter-word gaps: wide variance on purpose (space errors = #1 category).
        n_gaps = len(inks) - 1
        gaps = np.clip(rng.normal(0.32, 0.15, size=max(n_gaps, 0)), 0.07, 0.95) * px
        near_touch = rng.random(max(n_gaps, 0)) < 0.07
        gaps[near_touch] = 0.04 * px

        asc = max(w.baseline for w in inks)
        desc = max(w.alpha.shape[0] - w.baseline for w in inks)
        jitter = int(round(0.05 * px))
        margin_y = int(round(0.10 * px)) + jitter
        margin_x = int(round(0.15 * px))
        H = asc + desc + 2 * margin_y
        W = sum(w.alpha.shape[1] for w in inks) + int(gaps.sum()) + 2 * margin_x
        alpha = np.zeros((H, W), dtype=np.uint8)
        baseline = margin_y + asc

        # RTL: first logical word sits rightmost -> place in reversed order LTR.
        # Visual gap after placing word idx (before word idx-1) is logical
        # gap idx-1 (the one between words idx-1 and idx).
        x = margin_x
        for idx in range(len(inks) - 1, -1, -1):
            wi = inks[idx]
            dy = int(rng.integers(-jitter, jitter + 1))
            y0 = max(baseline - wi.baseline + dy, 0)
            roi = alpha[y0 : y0 + wi.alpha.shape[0], x : x + wi.alpha.shape[1]]
            np.maximum(roi, wi.alpha[: roi.shape[0], : roi.shape[1]], out=roi)
            x += wi.alpha.shape[1]
            if idx >= 1:
                x += int(gaps[idx - 1])
        # composite ink over paper
        paper = float(rng.uniform(238, 255))
        ink = float(rng.uniform(20, 90))
        a = alpha.astype(np.float32) / 255.0
        img = paper * (1.0 - a) + ink * a
        return img.astype(np.uint8), font_path


# ----------------------------------------------------------------------------
# Degradations (render scale first, then scan scale)
# ----------------------------------------------------------------------------


def _shear(img: np.ndarray, deg: float) -> np.ndarray:
    h, w = img.shape
    t = np.tan(np.radians(deg))
    pad = int(abs(t) * h) + 1
    img = cv2.copyMakeBorder(img, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=255)
    M = np.array([[1, t, -t * h / 2], [0, 1, 0]], dtype=np.float32)
    out = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                         flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=255)
    return out


def _elastic(img: np.ndarray, rng: np.random.Generator,
             alpha: float, sigma: float) -> np.ndarray:
    h, w = img.shape
    kx = int(sigma * 4) | 1
    dx = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32), (kx, kx), sigma) * alpha
    dy = cv2.GaussianBlur(rng.uniform(-1, 1, (h, w)).astype(np.float32), (kx, kx), sigma) * alpha
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, xx + dx, yy + dy, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def _crop_to_ink(img: np.ndarray, margin_frac: float = 0.04) -> np.ndarray:
    ys, xs = np.where(img < 200)
    if len(xs) == 0:
        return img
    m = int(margin_frac * img.shape[0]) + 2
    y0, y1 = max(ys.min() - m, 0), min(ys.max() + m, img.shape[0])
    x0, x1 = max(xs.min() - m, 0), min(xs.max() + m, img.shape[1])
    return img[y0:y1, x0:x1]


def degrade(img: np.ndarray, rng: np.random.Generator) -> Image.Image:
    """Full handwriting-izing pipeline; input at render scale, white bg."""
    # 1) writer slant
    if rng.random() < 0.7:
        img = _shear(img, float(rng.uniform(-8, 8)))
    # 2) page skew
    if rng.random() < 0.5:
        h, w = img.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), float(rng.uniform(-1.5, 1.5)), 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    # 3) pen width (safe at ~4x scale: dots are 12-20 px here)
    r = rng.random()
    if r < 0.30:
        img = cv2.erode(img, np.ones((2, 2), np.uint8))      # thicker ink
    elif r < 0.50:
        img = cv2.dilate(img, np.ones((2, 2), np.uint8))     # thinner / dry pen
    # 4) elastic wobble — the core "not printed" signal
    if rng.random() < 0.85:
        img = _elastic(img, rng, alpha=float(rng.uniform(6, 14)),
                       sigma=float(rng.uniform(4.5, 8.0)))
    # 5) mild defocus at render scale
    if rng.random() < 0.8:
        s = float(rng.uniform(0.4, 1.1))
        img = cv2.GaussianBlur(img, (0, 0), s)
    img = _crop_to_ink(img)
    # 6) downscale to scan resolution
    save_h = int(rng.integers(120, 200))
    scale = save_h / img.shape[0]
    pil = Image.fromarray(img).resize(
        (max(int(img.shape[1] * scale), 8), save_h), Image.LANCZOS
    )
    img = np.asarray(pil, dtype=np.uint8)
    # 7) paper gradient
    if rng.random() < 0.3:
        g = np.linspace(rng.uniform(0.95, 1.0), rng.uniform(0.97, 1.03), img.shape[1])
        img = np.clip(img.astype(np.float32) * g[None, :], 0, 255).astype(np.uint8)
    # 8) sensor noise
    if rng.random() < 0.6:
        noise = rng.normal(0, float(rng.uniform(1, 6)), img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    # 9) JPEG round-trip (KHATT is JPG scans)
    if rng.random() < 0.7:
        q = int(rng.integers(55, 91))
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            img = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
    return Image.fromarray(img, mode="L")


def render_sample(
    text: str, renderer: LineRenderer, rng: np.random.Generator,
    font_path: Path | None = None,
) -> tuple[Image.Image, str, str] | None:
    """Clean text -> rendered+degraded PIL image.

    Returns (image, label_text, font_family) or None if the line is
    ineligible / no font covers it.
    """
    cleaned = clean_synth_text(text)
    if cleaned is None:
        return None
    out = renderer.render_line(cleaned, rng, font_path=font_path)
    if out is None:
        return None
    img, used_font = out
    return degrade(img, rng), cleaned, font_family(used_font)
