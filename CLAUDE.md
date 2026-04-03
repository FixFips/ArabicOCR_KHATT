# CLAUDE.md — Project Guide for Arabic Handwritten OCR (KHATT)

## Project Overview

Arabic handwritten text recognition using CRNN-CTC trained on the KHATT dataset.
Line-level OCR: one handwritten line image -> Arabic text output.

- **Branch**: `Osama-Sharaf/improvement`
- **Dataset**: KHATT, 11,375 line images in `archive/images/` + `archive/labels/`
- **Vocabulary**: 75 classes (Arabic letters, digits, punctuation, special tokens)
- **Framework**: PyTorch
- **Training hardware**: Windows PC with RTX 5080 16GB / 32GB RAM
- **Dev machine**: macOS M2 (no GPU training here)

## Baseline vs Current

| Metric | Old Baseline (v1, H=64) | Target (v2, H=96) |
|--------|------------------------|-------------------|
| CER | 12.2% | 4-7% |
| WER | 42.4% | 20-30% |
| DotCER | not measured | 8-12% |

The old v1 model had: no augmentation, no LR scheduler, only 1 BatchNorm, greedy-only decoding, H=64, MAX_W=1024, incomplete charset (2,624 chars lost to `<unk>`). All of these are fixed in v2.

## Architecture (v2 — Multi-Scale Vertical CRNN-CTC)

```
Input: [B, 1, 96, 1536] grayscale
  |
CNN: 7 conv layers (full BatchNorm, Dropout2d after pools)
  |
Adaptive pool to 3 vertical zones (above-baseline / baseline / below-baseline)
  -> preserves Arabic dot POSITION (above vs below)
  -> critical for ba/ta/tha/nun/ya disambiguation
  |
2-layer BiLSTM(1536 -> 384) with dropout=0.2
  |
Dropout(0.3) -> FC(768 -> 75) -> CTC loss
  |
Output: [T=385, B, 75] -- decoded via greedy or beam search with Arabic bigram LM
```

**Tensor shape trace through CNN (input H=96, W=1536):**
```
Block 1: Conv(1,64) BN ReLU Pool(2,2) Drop     -> [B, 64, 48, 768]
Block 2: Conv(64,128) BN ReLU Pool(2,2) Drop    -> [B, 128, 24, 384]
Block 3: Conv(128,256) BN ReLU                   -> [B, 256, 24, 384]
Block 4: Conv(256,256) BN ReLU Pool(asym) Drop   -> [B, 256, 12, 385]
Block 5: Conv(256,512) BN ReLU                   -> [B, 512, 12, 385]
Block 6: Conv(512,512) BN ReLU Pool(asym)        -> [B, 512, 6, 386]
Block 7: Conv(512,512,k=2) BN ReLU              -> [B, 512, 5, 385]
adaptive_avg_pool2d(3, 385)                      -> [B, 512, 3, 385]
view + permute                                   -> [B, 385, 1536]
BiLSTM(1536, 384, bidirectional)                 -> [B, 385, 768]
Dropout(0.3) + FC(768, 75)                       -> [B, 385, 75]
permute                                          -> [385, B, 75] (T, B, C for CTC)
```

T=385 timesteps vs max label length 132 chars = 2.9x ratio (CTC needs T >= label_len).

## Key Design Decisions (Arabic-Specific)

1. **Height=96** (not 64): Arabic has 7 vertical zones. At H=64, dots are 2-3px (below 3x3 conv detection threshold). At H=96, dots become 3-5px.

2. **MAX_W=1536** (not 1024): At H=96, mean image width becomes ~1416px. Without increasing MAX_W, images would be horizontally squished.

3. **3-zone vertical pooling** (not 1): `adaptive_avg_pool2d(x, (3, W'))` preserves WHERE dots sit vertically. Pooling to 1 row destroys the distinction between ta (dots above) and ba (dot below).

4. **Arabic-safe augmentation only**: Erosion, dilation, and elastic distortion are BANNED -- they destroy 2-4px dots and break cursive connections. Only shear, kashida stretch, mild rotation, brightness, and noise are used.

5. **LANCZOS resize**: Preserves dot sharpness during downscaling (bilinear blurs small features).

6. **Horizontal-only morphological kernel** (1,2): Removes noise specks without destroying vertically-compact dots.

7. **RTL handling**: Ground-truth labels are reversed for CTC (which processes left->right), then predictions are reversed back to Arabic RTL for evaluation.

8. **Charset coverage**: Expanded to cover 99.996% of label characters (was ~97% before). Comment detection uses `"# "` (with space) so bare `#` is treated as the character.

## File Structure

```
src/
  model.py            -- CRNN architecture + CTC decoders (greedy + beam) + bigram LM builder
  augment.py          -- ArabicAugment class (dot-safe transforms only)
  train_crnn_ctc.py   -- Training loop (OneCycleLR, grad accum, early stopping, test eval)
  dataset.py          -- KHATTDataset (PyTorch Dataset, augment hook)
  preprocess.py       -- CLAHE + dual-polarity Otsu binarization, LANCZOS resize, padding
  metrics.py          -- CER, WER, dot-group CER
  webocr.py           -- Gradio web demo (beam search, line segmentation)
  charset_arabic.txt  -- 70 characters + 5 special tokens = 75 classes
  show_metrics.py     -- CLI tool to view training metrics from CSV
  __init__.py         -- Package init (empty)
archive/              -- (gitignored) KHATT dataset
  images/             -- 11,375 line images (JPG, RGB, mean 2006x136px)
  labels/             -- 11,375 text labels (TXT, windows-1256 encoding)
  splits/             -- Auto-generated train/val/test CSV files (80/10/10)
runs/exp1/            -- (gitignored) Training outputs
  crnn_best.pt        -- Best model checkpoint (by val CER)
  metrics.csv         -- Per-epoch training log
research/archive/
  train_trocr_char.py -- Archived TrOCR experiment (not used)
```

## Import Dependency Graph (no cycles)

```
model.py          -- standalone (torch only)
augment.py        -- standalone (cv2, numpy, PIL)
metrics.py        -- standalone (rapidfuzz)
preprocess.py     -- standalone (cv2, numpy, PIL)
dataset.py        -- imports preprocess
train_crnn_ctc.py -- imports dataset, metrics, model, augment
webocr.py         -- imports model, preprocess
```

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Train (creates splits automatically on first run)
python -m src.train_crnn_ctc

# View training metrics
python -m src.show_metrics --run ./runs/exp1

# Launch web demo (requires trained checkpoint at runs/exp1/crnn_best.pt)
python -m src.webocr
```

## Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Height | 96 | Arabic dots need 3-5px to be detectable by 3x3 convolutions |
| Max Width | 1536 | Fits mean image (1416px at H=96) without forced compression |
| Batch Size | 16 | With grad accumulation=2, effective batch=32 |
| Grad Accumulation | 2 | Compensates for smaller batch size |
| Epochs | 120 | Early stopping (patience=15) prevents waste |
| Optimizer | AdamW | lr=1e-3 initial |
| Scheduler | OneCycleLR | max_lr=3e-3, 10% warmup, cosine anneal |
| Loss | CTCLoss | blank=0 (=pad token), zero_infinity=True |
| Grad Clip | 5.0 | Max norm |
| Augmentation | Arabic-safe only | Shear, kashida, rotation, baseline shift, brightness, noise |
| VRAM estimate | ~3.5-5 GB | Fits comfortably in 16GB RTX 5080 |

## Metrics

- **CER**: Character Error Rate (Levenshtein distance / ref length)
- **WER**: Word Error Rate (word-level Levenshtein / ref word count)
- **WER_norm**: WER after removing diacritics, tatweel, normalizing alef/ya forms
- **DotCER**: CER measured only on dot-differentiated letter groups (ba/ta/tha, jim/ha/kha, nun/ya) -- directly tracks the #1 Arabic OCR error source

## Checkpoint Format

```python
{
    "model": state_dict,          # OrderedDict of tensors
    "vocab": list[str],           # ordered vocabulary (75 entries)
    "arch_version": 2,            # v2 = multi-scale vertical CRNN
}
```

## Execution Flow: Training

```
1. INITIALIZATION
   +-- Set seed=42 for reproducibility
   +-- Build train/val/test CSV splits (80/10/10) in archive/splits/
   |   9,101 train / 1,138 val / 1,139 test lines
   +-- Load charset (75 classes)
   +-- Create ArabicAugment (training-only)
   +-- Create DataLoaders (batch=16, workers=2)

2. MODEL CREATION
   +-- CRNN(75): CNN(7 layers, full BN) -> 3-zone pool -> BiLSTM(1536->384) -> FC(768->75)
   +-- CTCLoss(blank=0)
   +-- AdamW optimizer
   +-- OneCycleLR scheduler (285 steps/epoch, 120 epochs = 34,200 total steps)

3. TRAINING LOOP (up to 120 epochs, early stop patience=15)
   For each epoch:
   +-- TRAIN PHASE
   |   +-- For each batch of 16 images:
   |   |   +-- Load image -> grayscale -> CLAHE+Otsu binarize -> augment -> resize H=96 -> pad W=1536
   |   |   +-- Forward: [16,1,96,1536] -> CNN -> 3-zone pool -> BiLSTM -> FC -> [385,16,75]
   |   |   +-- CTC loss (labels reversed for LTR alignment)
   |   |   +-- Accumulate gradients (every 2 batches -> clip -> optimizer.step -> scheduler.step)
   |   +-- Average training loss
   |
   +-- VALIDATION PHASE (no augmentation, no gradient)
   |   +-- Forward -> greedy CTC decode -> reverse back to RTL
   |   +-- Compute: CER, WER, WER_norm, DotGroupCER
   |   +-- Save 200 sample predictions to TSV file
   |
   +-- LOGGING
   |   +-- Print: epoch, loss, CER, WER, WER_norm, DotCER, LR, time
   |   +-- Append to runs/exp1/metrics.csv
   |   +-- Save checkpoint if CER improved (crnn_best.pt)
   |
   +-- EARLY STOP CHECK
       +-- If no CER improvement for 15 epochs -> stop

4. FINAL TEST EVALUATION (after training completes)
   +-- Load best checkpoint (crnn_best.pt)
   +-- Build Arabic character bigram LM from training labels (~5,625 bigrams)
   +-- Evaluate on held-out test set (1,139 lines):
   |   +-- Greedy decode -> CER, WER, WER_norm, DotCER
   |   +-- Beam search (width=10) + bigram LM -> CER, WER, WER_norm, DotCER
   +-- Print final comparison results
```

## Execution Flow: Web Demo Inference

```
1. Load checkpoint -> CRNN model in eval mode
2. User uploads image via Gradio UI
3. Optional: crop, rotate, adjust polarity
4. Segment into lines (morphological connected components)
5. For each line:
   a. Grayscale -> CLAHE+Otsu binarize (same as training)
   b. Polarity: try normal + inverted, pick longer output
   c. Resize H=96 -> Pad W=1536
   d. Forward -> beam search CTC decode -> reverse to RTL
6. Concatenate line texts -> display with preprocessed preview
```

## Data Flow Diagram

```
              TRAINING                             INFERENCE (Web Demo)

KHATT image (JPG, RGB, ~2000x136)          User upload (any format)
      |                                           |
      v                                           v
Grayscale -> CLAHE+Otsu Binarize            Segment into lines
      |                                           |
      v                                           v
Arabic-safe Augmentation (train only)       Grayscale -> CLAHE+Otsu Binarize
(shear, kashida, rotation, noise)                 |
      |                                           v
      v                                     Resize H=96 -> Pad W=1536
Resize H=96 (LANCZOS) -> Pad W=1536              |
      |                                           v
      v                                     CRNN forward pass
ToTensor [1, 96, 1536]                           |
      |                                           v
      v                                     Beam search + bigram LM
CRNN: CNN -> 3-zone pool -> BiLSTM -> FC         |
      |                                           v
      v                                     Reverse LTR -> RTL
CTC Loss (labels reversed LTR)                   |
      |                                           v
      v                                     Arabic text output
Greedy decode -> Reverse -> CER/WER
```

## Important Notes

- The CRNN class lives ONLY in `src/model.py`. Both training and web demo import from there. Never duplicate it.
- Labels use **windows-1256** encoding (KHATT standard). The dataset reader tries this first, then falls back to UTF-8.
- Charset comment lines must start with `# ` (hash + space). A bare `#` is the hash character itself.
- Diacritics are extremely rare in KHATT (<40 total across 713K characters). WER_norm is nearly equal to WER.
- Data lives in `archive/` (gitignored). The `data/` directory is unused.
- All preprocessing (binarize, resize, pad) is shared between training and web demo via `src/preprocess.py`.
- The old v1 checkpoint format (without `arch_version`) is NOT compatible with v2 architecture. Must retrain from scratch.
- `num_workers=2` works on Windows because `collate_fn` runs in the main process and `KHATTDataset` + `ArabicAugment` are picklable.
- Python 3.9+ required (PEP 585 type hints used). Python 3.10+ recommended.

## What Changed from v1 to v2 (Summary)

| Component | v1 (old) | v2 (current) |
|-----------|----------|-------------|
| Input height | 64 | 96 (dots: 2-3px -> 3-5px) |
| Input max width | 1024 | 1536 (fits H=96 images) |
| Batch size | 32 | 16 + grad_accum=2 |
| BatchNorm | 1 of 7 conv layers | All 7 conv layers |
| Dropout | None | Dropout2d(0.1) + LSTM(0.2) + FC(0.3) |
| Vertical pooling | pool to 1 row | Pool to 3 zones (dot position preserved) |
| LSTM | BiLSTM(512, 256) | BiLSTM(1536, 384) |
| FC | Linear(512, N) | Linear(768, N) |
| Augmentation | None | Arabic-safe (6 transforms, 4 banned) |
| LR scheduler | None (constant) | OneCycleLR (warmup + cosine) |
| Early stopping | None | patience=15 |
| Epochs | 70 | 120 (with early stop) |
| CTC decoding | Greedy only | Greedy + Beam(10) + Arabic bigram LM |
| Resize interpolation | BILINEAR | LANCZOS |
| Morph kernel | (2,2) square | (1,2) horizontal-only |
| Charset coverage | ~97% (2,624 chars -> unk) | 99.996% (only 32 rare symbols -> unk) |
| Test evaluation | Not implemented | Greedy + beam on held-out test set |
| Dot-group CER | Not measured | Tracked per epoch |
| CRNN location | Duplicated in train + webocr | Single source in model.py |
| Preprocessing | Different in train vs webocr | Shared via preprocess.py |
