---
title: Arabic Handwritten OCR (KHATT)
emoji: ✍️
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
short_description: Arabic handwriting recognition (CRNN-CTC, KHATT)
models:
- FixFips/arabicocr-khatt
---

# Arabic Handwritten OCR (KHATT)

Line-level Arabic handwritten text recognition. Upload an image of Arabic
handwriting and get the recognized text — multi-line pages are segmented
automatically.

- **Model:** CRNN (CNN + BiLSTM) + CTC with Arabic-specific 3-zone vertical
  pooling (preserves dot position, the only difference between ba/ta/tha/nun/ya)
- **Weights:** [FixFips/arabicocr-khatt](https://huggingface.co/FixFips/arabicocr-khatt)
- **Code:** https://github.com/FixFips/ArabicOCR_KHATT
- **Install:** `pip install arabicocr-khatt`

## Deploying this Space

```bash
# from the repo root
hf auth login
hf repo create arabicocr-khatt-demo --repo-type space --space-sdk gradio
git clone https://huggingface.co/spaces/FixFips/arabicocr-khatt-demo
cp space/* arabicocr-khatt-demo/
cd arabicocr-khatt-demo && git add . && git commit -m "Deploy demo" && git push
```
