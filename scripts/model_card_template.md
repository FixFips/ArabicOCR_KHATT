---
language:
- ar
license: mit
library_name: pytorch
pipeline_tag: image-to-text
tags:
- ocr
- handwritten-text-recognition
- arabic
- khatt
- crnn
- ctc
---

# ArabicOCR-KHATT — Arabic Handwritten Text Recognition (CRNN-CTC)

Line-level Arabic handwritten text recognition, trained on the
[KHATT](https://khatt.ideas2serve.net/) dataset (11,375 handwritten line images).

The architecture is a CRNN (CNN + BiLSTM) with CTC loss, with **Arabic-specific
design choices**: input height 96 so diacritic dots stay detectable, 3-zone
vertical pooling that preserves *where* dots sit (the only difference between
ba/ta/tha/nun/ya), dot-safe augmentation, and beam-search decoding with an
Arabic character bigram LM.

- **Code / training pipeline:** https://github.com/FixFips/ArabicOCR_KHATT
- **Python package:** `pip install arabicocr-khatt`

## Usage

```python
from arabicocr_khatt import ArabicOCR

ocr = ArabicOCR.from_pretrained("{repo_id}")
text = ocr.recognize("handwritten_page.jpg")   # segments lines automatically
print(text)
```

Or from the command line:

```bash
pip install arabicocr-khatt
arabicocr handwritten_page.jpg
```

## Validation metrics (KHATT, best epoch {epoch})

| Metric | Value |
|--------|-------|
| CER | {cer} |
| WER | {wer} |
| WER (normalized) | {wer_norm} |
| Dot-group CER | {dot_cer} |

Dot-group CER measures errors only on dot-differentiated letter groups
(ba/ta/tha, jim/ha/kha, nun/ya) — the #1 error source in Arabic OCR.

## Files

| File | Purpose |
|------|---------|
| `crnn_best.pt` | Model checkpoint: `{{"model": state_dict, "vocab": list[str], "arch_version": 2}}` |
| `bigram_lm.json` | Arabic character bigram LM for beam-search decoding |
| `charset_arabic.txt` | 75-class character set (70 characters + 5 special tokens) |

## Limitations

- Line-level model: full pages are segmented into lines with classical
  morphology before recognition; complex layouts may segment poorly.
- Trained only on KHATT handwriting; printed text, historical manuscripts, and
  heavily diacritized text are out of domain.
- No word-level language model — output is not spell-corrected.

## Citation

If you use this model, please also cite the KHATT dataset:

> Mahmoud, S. A., et al. "KHATT: An open Arabic offline handwritten text database."
> Pattern Recognition 47.3 (2014): 1096-1112.
