# ArabicOCR-KHATT — Arabic Handwritten Text Recognition (CRNN-CTC)

[![PyPI](https://img.shields.io/pypi/v/arabicocr-khatt)](https://pypi.org/project/arabicocr-khatt/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Model on HF](https://img.shields.io/badge/%F0%9F%A4%97%20Hub-FixFips%2Farabicocr--khatt-blue)](https://huggingface.co/FixFips/arabicocr-khatt)

**Arabic handwritten text recognition** — line-level OCR (a line or paragraph
image → Arabic text), trained on the **KHATT** dataset.

What makes it Arabic-specific rather than a generic CRNN:

- **3-zone vertical pooling** — preserves *where* dots sit (above vs below the
  baseline), the only difference between ب/ت/ث and ن/ي
- **Input height 96** — keeps diacritic dots large enough (3–5 px) for 3×3
  convolutions to detect
- **Dot-safe augmentation** — shear, kashida stretch, mild rotation; erosion /
  dilation / elastic are banned because they destroy 2–4 px dots
- **Beam-search decoding with an Arabic character bigram LM**
- **Dot-group CER metric** — tracks errors on dot-differentiated letter groups,
  the #1 Arabic OCR error source

---

## Install

```bash
pip install arabicocr-khatt
```

## Quickstart

```python
from arabicocr_khatt import ArabicOCR

ocr = ArabicOCR.from_pretrained()        # downloads weights from the HF Hub
text = ocr.recognize("handwritten_page.jpg")   # multi-line pages segmented automatically
print(text)
```

Or from the command line:

```bash
arabicocr handwritten_page.jpg
arabicocr line.png --no-segment --greedy       # single line, fastest decode
arabicocr scan.jpg --checkpoint runs/exp1/crnn_best.pt   # local weights
```

Try it in the browser: **[demo Space on Hugging Face](https://huggingface.co/spaces/FixFips/arabicocr-khatt-demo)**.

---

## Architecture

```
Input [B, 1, 96, 1536] grayscale
  → CNN (7 conv layers, full BatchNorm, Dropout2d)
  → adaptive pool to 3 vertical zones (dot position preserved)
  → 2-layer BiLSTM(1536 → 384)
  → FC → CTC loss
  → greedy or beam-search decode (+ Arabic bigram LM), reversed back to RTL
```

Preprocessing (shared between training and inference): grayscale → CLAHE →
dual-polarity Otsu binarization → LANCZOS resize to H=96 → pad to W=1536.

## Project structure

```text
ArabicOCR_KHATT/
├─ arabicocr_khatt/            # the installable package
│  ├─ pipeline.py              # ArabicOCR API + CLI (inference single-source)
│  ├─ model.py                 # CRNN + CTC decoders + bigram LM builder
│  ├─ preprocess.py            # CLAHE, binarization, resize, padding
│  ├─ augment.py               # Arabic-safe augmentation
│  ├─ dataset.py               # KHATTDataset
│  ├─ metrics.py               # CER / WER / dot-group CER
│  ├─ train_crnn_ctc.py        # training script
│  ├─ webocr.py                # Gradio test bench (rich debugging UI)
│  ├─ show_metrics.py, monitor.py, compare_runs.py, eval_val.py, ...
│  └─ charset_arabic.txt       # 75-class charset
├─ scripts/upload_to_hf.py     # publish weights + model card to the HF Hub
├─ space/                      # Hugging Face Space demo app
├─ archive/                    # (gitignored) KHATT images + labels + splits
├─ runs/                       # (gitignored) checkpoints + metrics.csv
└─ pyproject.toml
```

---

## Training your own model

Get the KHATT dataset (request access at [khatt.ideas2serve.net](https://khatt.ideas2serve.net/))
and lay it out as:

```text
archive/
├─ images/   *.jpg            # line images
├─ labels/   *.txt            # Arabic text (Windows-1256 / UTF-8)
└─ splits/                    # auto-created train/val/test CSVs (80/10/10)
```

Then, with the training extras installed:

```bash
pip install -e ".[train]"
python -m arabicocr_khatt.train_crnn_ctc                 # trains to runs/exp1
python -m arabicocr_khatt.show_metrics --run ./runs/exp1 # inspect metrics
python -m arabicocr_khatt.monitor                        # live training dashboard
```

Training uses OneCycleLR, gradient accumulation, early stopping, and logs
per-epoch CER / WER / WER(norm) / dot-group CER to `runs/<exp>/metrics.csv`.
Checkpoints are saved as `{"model": state_dict, "vocab": [...], "arch_version": 2}`.

## Web test bench (Gradio)

A rich debugging UI with crop/rotate, polarity control, decoder comparison,
confidence heatmaps, char-level diff against ground truth, and batch evaluation:

```bash
pip install -e ".[demo]"
python -m arabicocr_khatt.webocr
```

## Publishing weights to the Hub

After training (on the machine that has `runs/` and `archive/`):

```bash
hf auth login
python scripts/upload_to_hf.py --run-dir runs/exp1
```

This validates the checkpoint, builds the bigram LM from the training split,
fills the model card with your best-epoch metrics, and uploads everything to
[FixFips/arabicocr-khatt](https://huggingface.co/FixFips/arabicocr-khatt).

---

## Notes and limitations

- Trained on KHATT: performance is best on handwriting similar to that dataset.
  Printed fonts, very noisy backgrounds, or strongly curved text may fail.
- Line-level model: pages are segmented into lines with classical morphology;
  complex layouts may segment poorly.
- Labels use Windows-1256 encoding (KHATT standard); the reader falls back to
  UTF-8 automatically.

## Contributing

Contributions are very welcome — Arabic OCR is an underserved area and there is
plenty to do. See [CONTRIBUTING.md](CONTRIBUTING.md) and the
[good first issues](https://github.com/FixFips/ArabicOCR_KHATT/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).

## License

MIT — see [LICENSE](LICENSE).

If you use this project in research, please also cite the KHATT dataset:

> Mahmoud, S. A., et al. "KHATT: An open Arabic offline handwritten text
> database." Pattern Recognition 47.3 (2014): 1096–1112.
