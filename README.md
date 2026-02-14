# ArabicOCR_KHATT (CRNN-CTC)

This project is an **Arabic handwritten text recognition** system.

- Model: CRNN (CNN + BiLSTM) with CTC loss  
- Level: line-based OCR (one line or paragraph image → text)  
- Dataset: **KHATT** handwritten Arabic (images + text labels)  
- Extras: a small **Gradio web demo** for testing on your own images

Metrics per epoch (CER, WER, normalized WER) are saved to  
`runs/<exp_name>/metrics.csv`.

---

## 1. Features

- **CRNN-CTC model** for Arabic text (right-to-left handling inside the code).
- **Preprocessing**:
  - grayscale + CLAHE (local contrast)
  - Otsu / adaptive threshold
  - automatic polarity (tries to keep black text on white background)
- **Automatic splits**: builds `train/val/test` CSVs from the `data` folder.
- **Metrics logger**:
  - per-epoch `train_loss`, `CER`, `WER`, `WER(norm)`
  - `show_metrics.py` script to summarize runs
- **Web demo** (`src/webocr.py`) with:
  - image upload / paste / webcam
  - crop region
  - rotate slider
  - upscale slider (for tiny text)
  - polarity mode: **Auto / Normal / Invert**
  - preview of “what the model saw” after preprocessing

---

## 2. Project structure

```text
ArabicOCR_KHATT/
├─ src/
│  ├─ __init__.py
│  ├─ charset_arabic.txt      # charset used for labels (includes <pad>, <unk>, etc.)
│  ├─ dataset.py              # KHATTDataset (images + labels + preprocessing)
│  ├─ metrics.py              # CER / WER functions (Levenshtein)
│  ├─ preprocess.py           # image preprocessing (CLAHE, binarize, padding...)
│  ├─ show_metrics.py         # small script to print best/last metrics from CSV
│  ├─ train_crnn_ctc.py       # main training script
│  └─ webocr.py               # Gradio web demo
├─ data/                      # (ignored by git) images + labels + splits
├─ runs/                      # (ignored by git) checkpoints + metrics.csv
├─ research/                  # optional experiments / notes
├─ requirements.txt
├─ LICENSE
└─ README.md
```

Note: data/ and runs/ are not tracked by git (see .gitignore),
so you need to create them locally.

⸻

## 3. Installation

Create a virtual environment and install dependencies.

**Windows (PowerShell)**

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` includes:

- torch>=2.2  
- torchvision>=0.17  
- numpy>=1.24  
- pandas>=2.0  
- scikit-learn>=1.3  
- pillow>=10.0  
- opencv-python>=4.8  
- rapidfuzz>=3.6  
- gradio>=4.0  

⸻

## 4. Dataset preparation (KHATT)

The KHATT dataset itself is not included in this repo.

Expected structure:

```text
data/
├─ images/
│  ├─ 000001.png
│  ├─ 000002.png
│  └─ ...
├─ labels/
│  ├─ 000001.txt   # Arabic text (Windows-1256 / UTF-8 / UTF-8-SIG)
│  ├─ 000002.txt
│  └─ ...
└─ splits/         # will be auto-created if missing
   ├─ train.csv
   ├─ val.csv
   └─ test.csv
```

On the first run, `train_crnn_ctc.py` will:
- scan `data/images` + `data/labels`
- build `train.csv`, `val.csv`, `test.csv` under `data/splits/`

Each CSV row contains: `filename,label_path`.

⸻

## 5. Training

From the project root (with the venv activated):

```bash
python -m src.train_crnn_ctc
```

The script will:
- create the splits (if they don’t exist)
- train the CRNN-CTC model
- log metrics to `runs/exp1/metrics.csv`
- save the best checkpoint (by CER) to `runs/exp1/crnn_best.pt`

You can change basic settings at the top of `src/train_crnn_ctc.py`:

```python
IMAGES_DIR = "./data/images"
LABELS_DIR = "./data/labels"
SPLITS_DIR = "./data/splits"
RUN_DIR    = "./runs/exp1"

HEIGHT     = 64
MAX_W      = 1024
BATCH_SIZE = 32
EPOCHS     = 70
LR         = 1e-3
```

There is also a resume/fine-tune option: if `crnn_best.pt` already exists,
the script loads the weights and continues with a smaller learning rate.

⸻

## 6. Viewing metrics

You can quickly see the best epoch and the last epoch using:

```bash
# show metrics for one run
python -m src.show_metrics --run ./runs/exp1
```

Or scan all run folders:

```bash
python -m src.show_metrics --all
```

Example output (just an idea):

```text
=== ./runs/exp1 ===
epochs: 70 | best epoch (CER): 22 | best CER: 0.120 | WER: 0.419 | WER(norm): 0.409
last  : epoch 70 | train_loss: 0.002 | CER: 0.121 | WER: 0.423 | WER(norm): 0.414

 epoch | train_loss |   CER  |  WER  | WER(n) | saved
-------+------------+--------+-------+--------+------
    61 |      0.043 |  0.128 | 0.439 |  0.429 |  0
    62 |      0.038 |  0.126 | 0.435 |  0.425 |  0
    ...
```

⸻

## 7. Web demo (Gradio)

To run the web interface:

```bash
python -m src.webocr
```

A local URL will appear in the terminal (for example `http://127.0.0.1:7860`).

The UI lets you:
- upload, paste, or capture an image
- crop the region that contains text
- rotate the image (small angles)
- upscale small text before OCR
- choose a polarity mode:
  - Auto (try both): normal + inverted, keeps the better one
  - Normal (black on white)
  - Invert (white on black)
- see:
  - the recognized text (RTL)
  - a preview of the preprocessed line(s) the model actually used

This is useful for testing on signs, notebook photos, product labels, etc.

⸻

## 8. Notes and limitations

- The model is trained on KHATT, so performance is best on handwriting
  similar to that dataset (line-level Arabic writing).
- Printed fonts, very noisy backgrounds, or extremely curved text may fail.
- The web demo preprocessing is similar to training, but we added a few
  extra tricks (contrast, polarity, rotation) to handle real-world images.

⸻

## 9. Git quick start 

If you clone this repo or want to reuse the template:

```bash
git init
git add .
git commit -m "Initial commit: CRNN-CTC Arabic OCR + web demo"
git branch -M main
```

Then create an empty GitHub repo named `ArabicOCR_KHATT`, and:

```bash
git remote add origin https://github.com/<your-user>/ArabicOCR_KHATT.git
git push -u origin main
```

`.gitignore` already ignores:
- `data/`, `runs/`, checkpoints (`*.pt`, `*.pth`, `*.onnx`)
- local envs (`.venv/`, `.env/`, `.gradio/`)
- editor/OS files (`.idea/`, `.vscode/`, `.DS_Store`, etc.)

⸻

## 10. License

This project is released under the terms in `LICENSE`.

