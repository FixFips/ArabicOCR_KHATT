# Contributing to ArabicOCR-KHATT

Thanks for your interest! Arabic handwritten OCR is an underserved area, and
contributions of any size are welcome — code, docs, evaluation on new data, or
just a well-written bug report with a sample image.

## Getting started

```bash
git clone https://github.com/FixFips/ArabicOCR_KHATT.git
cd ArabicOCR_KHATT
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[demo,train]"
```

You do **not** need the KHATT dataset or a GPU for most contributions:

- Inference-side work (pipeline, CLI, demo, docs) only needs the published
  weights: `ArabicOCR.from_pretrained()` downloads them automatically.
- Training-side work can be smoke-tested with a handful of synthetic
  image/label pairs in `archive/`.

## Where to help

Check the issues labeled
[`good first issue`](https://github.com/FixFips/ArabicOCR_KHATT/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
and [`help wanted`](https://github.com/FixFips/ArabicOCR_KHATT/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22).
Ideas that are always welcome:

- Evaluation on other Arabic handwriting datasets (IFN/ENIT, Muharaf, …)
- Better line/page segmentation
- Speed: batched line inference, ONNX export, quantization
- Docs and examples, especially in Arabic

## Ground rules

- **One source of truth**: the CRNN lives only in `arabicocr_khatt/model.py`,
  inference preprocessing only in `pipeline.py`/`preprocess.py`. Never duplicate them.
- **Arabic-safe transforms only** in augmentation: no erosion, dilation, or
  elastic distortion — they destroy the 2–4 px dots that distinguish letters.
- Match the existing code style; keep functions small and typed.
- Before opening a PR, make sure `python -m arabicocr_khatt.webocr` still starts
  and `arabicocr --help` works.

## Reporting recognition errors

The most valuable bug report for an OCR project is an image the model gets
wrong. Please open an issue with the image (or a crop), the expected text, the
model output, and your decoding settings (greedy/beam, LM weight).
