# ArabicOCR_KHATT (CRNN-CTC)

End-to-end Arabic handwritten text recognition (line-level) using a CRNN (CNN + BiLSTM) trained with CTC, plus a small Gradio web demo. Trained/evaluated on KHATT. Metrics logged per epoch (CER, WER, normalized WER) to `runs/<exp>/metrics.csv`.

## Features
- Robust preprocessing (CLAHE + adaptive/Otsu, auto polarity, safe fallback).
- Train/val split autogeneration.
- Per-epoch logging to CSV + `show_metrics.py` summary.
- Gradio demo with **invert** and **rotation** controls so white-on-color or skewed text works.

## Install
```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

## Show Metric
```bash
python -m src.show_metrics --run ./runs/exp1
python -m src.show_metrics --all         
```

## WebDemo
```bash
python -m src.webocr
```
