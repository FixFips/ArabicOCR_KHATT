# Open-weight VLM and OCR foundation models for Arabic handwritten line recognition (KHATT), compared with a small CRNN-CTC

Project baseline for reference: CRNN+BiLSTM+CTC (~15M params), about 9k real KHATT train lines, test CER 5.79%, WER 26.4%, train CER 0.08% (memorizes, so it is data-bound). Single RTX 5080 16GB, bf16.

IMPORTANT caveat for all comparisons: KHATT numbers across papers use different splits (paragraph vs line, train vs train+val), different normalization, and some KHATT mirrors store ground truth in visual (reversed) order, which silently breaks CER if not fixed ([mabdulaziz499/arabic-htr-fanar-7b-lora card](https://huggingface.co/mabdulaziz499/arabic-htr-fanar-7b-lora)). Re-evaluate any candidate on your own test split with your own CER(n) script before trusting a number.

## Q1. Arabic-specific OCR VLMs (Qari-OCR, Baseer, AIN, Arabic-Nougat, 2025-26 releases): do any report KHATT handwriting results?

### Takeaway
The only public, line-level KHATT numbers for fine-tuned VLMs come from a community model card: Fanar-2-Oryx-IVU (a Qwen2.5-VL-7B derivative) with QLoRA reaches 3.77-4.11% CER on the KHATT test lines, and Qwen2.5-VL-3B with LoRA reaches 6.12%. These were trained on about 16.9k train+val lines, so they are not directly comparable to your 9k-line 5.79%. The printed-text Arabic OCR VLMs (Qari, Arabic-Nougat) fail badly on handwriting. AIN-7B's 0.07 KHATT CER on KITAB-Bench is suspiciously good and may reflect training overlap. Baseer__Nakba (3B/4B) won the NAKBA 2026 historical handwriting task, but it is CC-BY-NC-SA.

### Cited Findings
**Community KHATT line benchmark (KHATT v1.0 test split, 1,038 lines)** ([mabdulaziz499/arabic-htr-fanar-7b-lora](https://huggingface.co/mabdulaziz499/arabic-htr-fanar-7b-lora))
- Fanar-2-Oryx-IVU + LoRA, generalist mix (KHATT + Muharaf + synthetic), 8.29B: CER 3.77%, WER 16.50%.
- Fanar-2-Oryx-IVU + LoRA, KHATT-only: CER 4.11%, WER 15.96%.
- `sherif1313/Arabic-English-handwritten-OCR-v3` (3.75B), zero-shot on KHATT: CER 5.86%, WER 18.30%.
- `sherif1313/Arabic-handwritten-OCR-4bit-Qwen2.5-VL-3B-v3` (3.75B), zero-shot: CER 5.96%, WER 18.88%. The card says it was trained on Muharaf, not KHATT. Note, however, that the sherif1313 card itself lists KHATT among its training sources, so this is a possible contradiction; see below.
- Qwen2.5-VL-3B + LoRA fine-tuned on KHATT train: CER 6.12%, WER 18.43%.
- Zero-shot base models: Qwen2.5-VL-7B-Instruct CER 50.85% / WER 82.56%; Fanar-2-Oryx-IVU base CER 50.98%; Qwen2.5-VL-3B-Instruct CER 73.35% / WER 93.92%.
- Qari-OCR-v0.3 (2.21B) on KHATT: CER 235.55% / WER 126.10%, because it is built for printed text.

**Fanar recipe** (same card)
- QLoRA with a 4-bit NF4 base, LoRA r=16, alpha=32, dropout 0.05, applied to the LLM attention and MLP projections with the vision tower frozen. 40.4M trainable parameters.
- 3,000 steps (about 2.8 epochs), effective batch 16, LR 1e-4, cosine schedule, warmup 0.03. Took 2.6 h on a single 96GB GPU. License Apache-2.0.
- Training data was KHATT v1.0 train+val, 16,882 lines after leakage filtering. About 1,700 train lines whose exact text appears in test were removed.

**Failure notes from the same card**
- Multi-line input is out of distribution and the model "may fall back to a memorized KHATT sentence".
- Numbers and dates are weak, and output is undiacritized.

**sherif1313 Qwen2.5-VL-3B-v3**
- Base Qwen2.5-VL-3B-Instruct, Apache-2.0, 47,842 samples, 3 epochs, 69 h of training ([card](https://huggingface.co/sherif1313/Arabic-handwritten-OCR-4bit-Qwen2.5-VL-3B-v3)).
- Data sources are listed as Muharaf, KHATT, Arabic OCR images, historical manuscripts and English handwriting. Self-reported CER is inconsistent (3.82% vs 2.5%).
- Speed: 0.31 s/image unquantized, 0.57 s/image in 4-bit (hardware unspecified).
- If KHATT train was used, its "zero-shot" KHATT number above is not truly zero-shot. Leakage status is unknown.

**sherif1313 Qwen3-VL-4B**
- `sherif1313/Arabic-English-handwritten-OCR-Qwen3-VL-4B`, LoRA via PEFT on Qwen3-VL-4B-Instruct, Apache-2.0 ([card](https://huggingface.co/sherif1313/Arabic-English-handwritten-OCR-Qwen3-VL-4B)).
- No CER is reported. The author says it is about 10% less accurate than v3 in practice and "very slow" (about 400% slower).

**KITAB-Bench (arXiv 2502.14949), Table 9, handwriting datasets** (CER / WER) ([KITAB-Bench](https://arxiv.org/html/2502.14949v2))

| Model | KHATT | Muharaf |
|---|---|---|
| GPT-4o | 0.45 / 0.74 | 0.56 / 0.90 |
| Gemini-2.0-Flash | 0.19 / 0.45 | 0.33 / 0.69 |
| Qwen2-VL-7B | 1.12 / 5.04 | 3.57 / 2.87 |
| Qwen2.5-VL-7B | 1.60 / 1.80 | 2.40 / 2.74 |
| AIN-7B | 0.07 / 0.22 | 0.61 / 0.96 |
| Tesseract | 0.67 / 1.06 | 0.77 / 1.22 |
| PaddleOCR | 0.76 / 1.00 | 0.80 / 1.01 |

- The handwriting datasets in KITAB-Bench are KHATT, ADAB, Muharaf and OnlineKhatt.
- Arabic-Nougat averages CER 4.37 over all datasets (no per-dataset number).

**AIN** (arXiv 2502.00094, `MBZUAI/AIN`, Qwen2-VL-7B based)
- Shows KHATT CER 0.07 but Muharaf CER 0.61 in the same table ([KITAB-Bench](https://arxiv.org/html/2502.14949v2)).

**QARI-OCR** (arXiv 2506.02295)
- Qwen2-VL-2B-Instruct fine-tuned on synthetic Arabic. v0.2 reports CER 0.061 / WER 0.160 on diacritic-heavy printed text, and handwriting is left as future work ([HF paper page](https://huggingface.co/papers/2506.02295)).
- v0.4 exists as `NAMAA-Space/Qari-OCR-0.4.0-VL-4B-Instruct`, a PEFT adapter on Qwen3-VL-4B, Apache-2.0, trained on books OCR data ([Mixpeek listing](https://mixpeek.com/model/NAMAA-Space/Qari-OCR-0.4.0-VL-4B-Instruct)).
- Zero-shot on Muharaf lines, QARI v0.4 gets CER 167.2%, with 62.4% of its errors classed as hallucination ([When Do VLMs Help Arabic Manuscript OCR?, arXiv 2608.22366](https://www.alphaxiv.org/abs/2608.22366.md)).

**Baseer** (arXiv 2509.18174, Misraj AI)
- Qwen2.5-VL-3B-Instruct fine-tuned on 500k document pairs (300k synthetic, 200k real) for document-to-Markdown. Reports WER 0.25 on its Misraj-DocOCR benchmark ([arXiv 2509.18174](https://arxiv.org/abs/2509.18174)).
- The handwriting variant `Misraj/Baseer__Nakba` (4B BF16 on the Hub, license **cc-by-nc-sa-4.0**) won NAKBA NLP 2026 HTR with CER 0.0790 / WER 0.2440 on historical manuscript pages ([card](https://huggingface.co/Misraj/Baseer__Nakba)).
  - Recipe: decoder-only SFT on Muharaf, then full encoder+decoder tuning with the vision encoder unfrozen (encoder LR 9e-6, decoder LR 1e-4), then a SLERP merge of the epoch-1 and epoch-5 checkpoints. Trained on 2x H100, batch 128, 644x644 input.
  - Other teams (CER): Oblevit 0.0925, 3reeq 0.0938, Latent Narratives 0.1050, Al-Warraq 0.1142, Not Gemma 0.1217, NAMAA-Qari 0.1950, baseline 0.3683.
- NAKBA 2026 used the Omar Al-Saleh memoir collection (6,395 pages, 1951-1965), and Misraj ranked first in the systems track ([acrps.ai](https://acrps.ai/nakba-nlp-manu-understanding-2026)).

**Historic-Arabic-OCR (NakbaNLP 2026)**
- A LoRA fine-tune of Qari-OCR with CLAHE preprocessing reports CER 0.10 / WER 0.28 on historical text ([ACL Anthology 2026.nakbanlp-1.43](https://aclanthology.org/2026.nakbanlp-1.43/)).

**"Not Gemma" team (NakbaNLP 2026)**
- The Mubsiir OCR paper references the sherif1313 Qwen3-VL-4B and Qwen2.5-VL-3B-v3 models ([ACL Anthology 2026.nakbanlp-1.42](https://aclanthology.org/2026.nakbanlp-1.42/)). Search snippets attribute a 3B Qwen2.5-VL fine-tune at 8.59% CER / 25.87% WER on historical validation data to a NakbaNLP 2026 paper (unverified; the PDF was not text-extractable).

### Inferences
- The Fanar result shows that a 7B Arabic-adapted VLM with QLoRA can beat a CRNN on KHATT lines by a wide margin in CER (3.8-4.1% vs your 5.79%) and especially in WER (16% vs 26%).
  - However, it used about 1.9x more real lines (16.9k train+val) than your 9k. Their removal of about 1.7k text-overlapping lines also differs from the standard split.
  - The 3B LoRA run (6.12% CER) is roughly on par with your CRNN in CER but much better in WER (18.4% vs 26.4%). This points to the LLM decoder's language prior fixing word-level errors.
- The large WER gap at similar CER suggests VLMs make fewer but "whole-word" errors, while CRNN-CTC spreads character errors across many words.
- AIN-7B's KHATT CER of 0.07 alongside Muharaf 0.61 suggests KHATT-like data was in AIN's SFT mix. Treat it as possible contamination, not evidence of zero-shot skill.
- Printed-text Arabic OCR VLMs (Qari, Arabic-Nougat, Baseer base) are not useful zero-shot for handwriting. They are useful only as Arabic-adapted starting points.

### Gaps
- No official KHATT-line numbers from Qari, Baseer or AIN papers. No independent replication of the community Fanar/sherif1313 numbers.
- Exact KHATT split used by the community card is "v1.0 test split, 1,038 lines". Whether this matches the user's preprocessing and CER(n) normalization is unknown.
- AIN's training data composition regarding KHATT was not verified.

## Q2. General OCR VLMs (Qwen2.5-VL / Qwen3-VL, olmOCR, GOT-OCR2, Florence-2, PaddleOCR-VL, dots.ocr, DeepSeek-OCR, MinerU, Nanonets, Mistral OCR, SmolDocling, Granite-Docling): Arabic handwriting zero-shot and fine-tuned

### Takeaway
Zero-shot, every general open VLM is poor on Arabic handwriting: KHATT CER is about 41-73% for Qwen2.5-VL 3B/7B. None of the document-parsing OCR VLMs (PaddleOCR-VL, dots.ocr, DeepSeek-OCR, olmOCR, etc.) report Arabic handwriting results. Only the Qwen-VL family has demonstrated fine-tuned KHATT results, so it is the practical base.

### Cited Findings
- **Qwen2.5-VL zero-shot on KHATT test lines:** 7B gets CER 50.85%, 3B gets CER 73.35% ([fanar-lora card](https://huggingface.co/mabdulaziz499/arabic-htr-fanar-7b-lora)).
- **Zero-shot study (arXiv 2608.22366):** Qwen2.5-VL-7B gets KHATT CER 40.9% (paragraphs, 1,038 test samples) versus Tesseract 67.3%, and Muharaf lines CER 70.4%. Fanar-2 gets Muharaf 70.8% ([alphaxiv 2608.22366](https://www.alphaxiv.org/abs/2608.22366.md)).
  - The same study found Qwen omits 11.9% and hallucinates 15.9% (heuristic) on Muharaf. It also found repetition loops: Qwen's corpus CER on student essays was 199.0%, falling to 66.6% after repetition control.
- **PaddleOCR-VL** (0.9B, arXiv 2510.14528) reports handwriting only for English (edit distance 0.118) and Chinese (0.034), with no Arabic handwriting results ([search summary of arXiv 2510.14528](https://arxiv.org/pdf/2510.14528)).
- **DeepSeek-OCR and dots.ocr:** no controlled Arabic handwriting evaluation was found. One informal blog test calls DeepSeek-OCR weak on Arabic ([sonusahani blog](https://sonusahani.com/blogs/deepseek-ocr)).
- **Manchu low-resource OCR VLM study** ([arXiv 2507.06761](https://arxiv.org/html/2507.06761v1)):
  - Qwen2.5-VL-7B fine-tuned was clearly weaker (CER 0.254 on 218 handwritten word images) than LLaMA-3.2-11B-Vision (CER 0.0219).
  - A follow-up compares LLaMA-3.2-11B, Pixtral-12B and Qwen3-VL-8B fine-tuned with LoRA on both vision and language modules ([arXiv 2609.11495](https://arxiv.org/pdf/2609.11495)).

### Inferences
- Shortlist bases: Qwen2.5-VL-3B-Instruct, Qwen2.5-VL-7B-Instruct / QCRI Fanar-2-Oryx-IVU, Qwen3-VL-4B/8B-Instruct, plus Arabic-handwriting-adapted checkpoints (sherif1313 v3, Baseer__Nakba) as warm starts.
- Document-parsing models (PaddleOCR-VL, dots.ocr, DeepSeek-OCR, MinerU, olmOCR, SmolDocling / Granite-Docling, Nanonets-OCR) are layout/Markdown-oriented and Latin/CJK-focused. With no Arabic handwriting evidence, they are low priority.
- Florence-2 and GOT-OCR2 lack meaningful Arabic training signal (unverified, no source found).
- Mistral OCR is API-only as far as found, so it does not meet the open-weights criterion (not verified in this session).

### Gaps
- No published Arabic handwriting CER for PaddleOCR-VL, dots.ocr, DeepSeek-OCR, olmOCR, GOT-OCR2, Florence-2, MinerU, Nanonets-OCR, SmolDocling or Granite-Docling.
- No fine-tuned Qwen3-VL KHATT number found. The IEEE BigData 2025 "Arabic OCR in the Age of Multimodal Models" paper ([Lamarr](https://lamarr-institute.org/publication/arabic-ocr-in-the-age-of-multimodal-models-a-comprehensive-comparative-evaluation/)) may have handwriting numbers but was not accessed.

## Q3. Encoder-decoder HTR (TrOCR Arabic, HATFormer, DTrOCR) and pretrained encoders with CTC heads: do they beat a CRNN at 9k lines?

### Takeaway
HATFormer (TrOCR-base, 334M) did **not** beat a CRNN on KHATT. It got 15.4% CER versus 14.1% for the Saeed et al. CRNN under its own split, while beating CRNN on the larger Muharaf set. That is evidence that, at KHATT scale, transformer encoder-decoders need heavy synthetic pretraining and still may not win. No direct evidence was found for a DINOv2/ViT + CTC head beating a CRNN on Arabic at about 9k lines.

### Cited Findings
**HATFormer** (arXiv 2410.02179) ([arXiv html v2](https://arxiv.org/html/2410.02179v2))
- Initialized from `microsoft/trocr-base-stage1` (334M, BEiT encoder + RoBERTa decoder).
- Stage 1: pretraining on 1,000,065 synthetic Arabic lines (54 fonts, 130 backgrounds). Stage 2: fine-tuning.
- Uses a BlockProcessor: horizontal flip, height 64, and up to six 384x384 blocks (width up to 2,304 px).
- CER results:
  - Muharaf Arabic-only 8.6% (vs CRNN Saeed et al. 17.6%)
  - KHATT 15.4% (vs CRNN Saeed et al. 14.1%; Lamtougui CRNN 19.9%; Momeni Transformer 18.5%)
  - MADCAT 4.2% (vs Rawls CRNN 1.52%)
- Ablation: removing synthetic pretraining raised Muharaf CER from 10.4% to 14.6%. Random initialization reached 86.0%.
- Trained on 2-4 A100/H100. Code DOI: 10.5281/zenodo.13936253. The authors "will release" weights, but weight availability was not verified.
- The HATFormer KHATT split used about 6.7k lines (72/14/14).

**HTR-VT** (arXiv 2409.08573)
- A CNN-stem ViT encoder + CTC with SAM and span masking, trained from scratch without pretraining or synthetic data. Reports IAM line CER 4.7 / WER 14.9 ([HTR-VT](https://arxiv.org/pdf/2409.08573)).
- HTR-ConvText (arXiv 2512.05021) gets IAM CER 4.0 / WER 12.9 at 65.9M params. A plain ViT in that table gets 32.4 CER ([HTR-ConvText](https://arxiv.org/pdf/2512.05021)).
- A 2026 paper reports a CRNN at 8.45% CER on KHATT and 10.11% on Muharaf ([arXiv 2503.15023 search result](https://arxiv.org/pdf/2503.15023)).

### Inferences
- Your CRNN at 5.79% CER on KHATT is already stronger than every published TrOCR/HATFormer-style KHATT number found (14-19%), although splits and normalization likely differ.
- TrOCR-style models are unlikely to help without a 1M-line synthetic Arabic pretraining stage, which is expensive on one GPU.
- Plain ViT encoders underfit small HTR data (32.4 CER for vanilla ViT on IAM). The data-efficient variants that work (HTR-VT, HTR-ConvText) keep a CNN stem. This is consistent with your observation that adding a transformer layer underfit.
- A DINOv2 backbone is pretrained on natural images, not text, so it may transfer poorly to Arabic strokes. There is no evidence either way.

### Gaps
- No published DINOv2/generic-ViT + CTC Arabic HTR results found. DTrOCR Arabic results were not found. HATFormer weights' public availability was not confirmed.

## Q4. Published evidence of fine-tuned VLMs beating CRNN-CTC on low-resource HTR at about 10k lines

### Takeaway
The best direct evidence is the KHATT community card: a 7B VLM with QLoRA gets 3.77-4.11% CER / about 16% WER on about 16.9k lines, versus your CRNN at 5.79% / 26.4% on 9k lines. A 3B LoRA gets 6.12% / 18.4%. Elsewhere the evidence is mixed: VLMs win clearly on Manchu real-world words and Muharaf-scale historical data, but a 2026 study found kraken CRNNs beat VLMs on several historical Latin HTR tasks.

### Cited Findings
- **KHATT lines:** Fanar-2 7B + QLoRA gets CER 3.77-4.11%, WER 15.96-16.50%. Qwen2.5-VL-3B + LoRA gets CER 6.12%, WER 18.43% ([fanar-lora card](https://huggingface.co/mabdulaziz499/arabic-htr-fanar-7b-lora)).
- **Manchu (arXiv 2507.06761):** the CRNN dropped from 99.8% accuracy on synthetic data to 72.5% on real documents. Fine-tuned LLaMA-3.2-11B reached CER 0.0219 / 93.1% word accuracy on real handwritten word images ([arXiv 2507.06761](https://arxiv.org/html/2507.06761v1)).
  - In a later version, the CRNN got 56.8% on scanned historical documents versus 87.4% for the VLM on 753 real word images ([Cambridge CHR](https://www.cambridge.org/core/journals/computational-humanities-research/article/finetuning-visionlanguage-models-as-ocr-systems-for-lowresource-languages-a-case-study-of-manchu/D67751520CFEC6E538F4FF5E1EAF5AA4)).
- **Counterpoint (arXiv 2609.20064):** on historical Latin-script HTR, a kraken CRNN had lower CER/WER than PP-OCRv6 on several tasks. Fine-tuned PP-OCRv6 could outperform a large HTR-tailored VLM ([arXiv 2609.20064](https://arxiv.org/pdf/2609.20064)).
- **NAKBA 2026 historical Arabic:** all top systems were VLM fine-tunes, with the best CER at 0.079 ([Baseer__Nakba card](https://huggingface.co/Misraj/Baseer__Nakba)).

### Inferences
- At about 9k lines, expect a 3-4B VLM LoRA to land near your CRNN in CER (about 6%) but well below it in WER. A 7B Arabic-adapted VLM with QLoRA could plausibly reach about 4-5% CER.
- An ensemble or rescoring approach is low-risk: CRNN for character fidelity, VLM for word-level correction. Use the CRNN CTC output to detect VLM hallucinations, e.g. by flagging large edit distances between the two.

### Gaps
- No controlled same-split study of VLM vs CRNN at exactly 9k Arabic lines was found. No data-size crossover study was found (arXiv 2608.22366 explicitly does not analyse it).

## Q5. Practical: fine-tuning recipes (Unsloth / HF PEFT), VRAM on 16GB, inference speed, hallucination risk

### Takeaway
On 16GB, two setups are realistic:
- bf16 LoRA on a 2-4B model (Qwen2.5-VL-3B, Qwen3-VL-4B), or
- 4-bit QLoRA on a 7-8B model (Qwen2.5-VL-7B / Fanar-2-Oryx-IVU, Qwen3-VL-8B).

Line images are short, so visual-token counts are small. The main risks are hallucinated or memorized sentences, repetition loops, and skipped words. These need constrained decoding (greedy, a max_new_tokens tied to line width, a repetition penalty) and line-level inputs only.

### Cited Findings
- **Unsloth Qwen3-VL:** Unsloth supports Qwen3-VL fine-tuning and has a free Colab notebook for the 8B model. It claims "1.7x faster with 60% less VRAM" (vendor claim) ([Unsloth docs](https://docs.unsloth.ai/models/qwen3-vl-run-and-fine-tune)).
- **Unsloth generic minimum VRAM for 8B:** QLoRA about 6GB, 16-bit LoRA about 22GB. The table is for text models and is described as an "absolute minimum" ([Unsloth requirements](https://docs.unsloth.ai/get-started/beginner-start-here/unsloth-requirements)).
- **Fanar KHATT recipe** (proven): QLoRA NF4, r=16, alpha=32, LLM attention+MLP only, vision frozen, LR 1e-4, effective batch 16, about 2.8 epochs. Ran on a 96GB GPU, but the 4-bit 7B config itself is small ([card](https://huggingface.co/mabdulaziz499/arabic-htr-fanar-7b-lora)).
- **Baseer__Nakba recipe** (stronger, heavier): decoder-only SFT, then unfreeze the vision encoder with a low LR (9e-6) and a decoder LR of 1e-4, then SLERP-merge checkpoints ([card](https://huggingface.co/Misraj/Baseer__Nakba)).
- **Inference speed:**
  - sherif1313 Qwen2.5-VL-3B: 0.31 s/image bf16, 0.57 s/image 4-bit (hardware unspecified) ([card](https://huggingface.co/sherif1313/Arabic-handwritten-OCR-4bit-Qwen2.5-VL-3B-v3)).
  - Its Qwen3-VL-4B successor is reported as about 4x slower ([card](https://huggingface.co/sherif1313/Arabic-English-handwritten-OCR-Qwen3-VL-4B)).
  - Baseer__Nakba supports vLLM / SGLang serving ([card](https://huggingface.co/Misraj/Baseer__Nakba)).
- **Hallucination and omission risks:**
  - The Fanar KHATT model "may fall back to a memorized KHATT sentence" on out-of-distribution input ([card](https://huggingface.co/mabdulaziz499/arabic-htr-fanar-7b-lora)).
  - Zero-shot Qwen2.5-VL on Muharaf showed 11.9% omission and 15.9% hallucination. QARI got 62.4% hallucination. Repetition loops inflated corpus CER to 199% on essays ([arXiv 2608.22366](https://www.alphaxiv.org/abs/2608.22366.md)).

### Inferences (VRAM and speed estimates, not measured)
- **Qwen2.5-VL-3B / Qwen3-VL-4B, bf16 LoRA:** weights about 7-9GB. With gradient checkpointing and short line images (about 100-400 visual tokens at H=96, width up to about 1,500 px with the 28 px patch merge) plus short targets (under 100 tokens), it should fit in 16GB at micro-batch 4-8.
- **Qwen2.5-VL-7B / Fanar-2 / Qwen3-VL-8B, 4-bit QLoRA:** weights about 5-6GB. Should fit at micro-batch 2-4 with gradient checkpointing. Full fine-tuning of any of these does not fit in 16GB.
- **Windows:** bitsandbytes and Unsloth support on Windows has historically been fragile. WSL2 is safer (not verified in this session).
- **Throughput:** about 0.2-0.5 s/line for HF generate at 3-4B, and faster with vLLM (Linux/WSL). This is still 100x+ slower than a CRNN, but fine for 1-2k test lines.
- **Training time:** 9k lines x 3 epochs at about 2-4 it/s effective should take a few hours on an RTX 5080 (estimate).
- **Line-level guardrails:**
  - Greedy decoding with repetition_penalty about 1.05.
  - max_new_tokens proportional to image width.
  - Reject outputs whose length ratio vs the CRNN hypothesis is below 0.5 or above 2.
  - Fix the KHATT reversed-order labels before training and before evaluation.

### Ranked shortlist (synthesis)

| Rank | Model ID | Params | License | Arabic HW evidence | Recipe on 16GB | Fits 16GB? |
|---|---|---|---|---|---|---|
| 1 | `QCRI/Fanar-2-Oryx-IVU` (Qwen2.5-VL-7B based); adapter ref `mabdulaziz499/arabic-htr-fanar-7b-lora` | 8.29B | Apache-2.0 (per card) | KHATT CER 3.77-4.11% after QLoRA | QLoRA NF4 r16, vision frozen | Yes (4-bit) |
| 2 | `sherif1313/Arabic-handwritten-OCR-4bit-Qwen2.5-VL-3B-v3` / `...-OCR-v3` (warm start) | 3.75B | Apache-2.0 | KHATT CER about 5.9% "zero-shot" (possible KHATT leakage) | bf16 LoRA on top | Yes |
| 3 | `Qwen/Qwen2.5-VL-3B-Instruct` | 3.75B | Qwen research license (verify) | KHATT CER 6.12% after LoRA | bf16 LoRA | Yes |
| 4 | `Qwen/Qwen3-VL-4B-Instruct` / `Qwen3-VL-8B-Instruct` | 4B / 8B | Apache-2.0 (verify) | No KHATT number; base of Qari v0.4 and sherif Qwen3 | Unsloth LoRA (4B) / QLoRA (8B) | Yes |
| 5 | `Misraj/Baseer__Nakba` | about 3-4B | CC-BY-NC-SA-4.0 (non-commercial) | NAKBA 2026 winner, CER 0.079 on historical pages | LoRA or partial unfreeze | Yes (LoRA) |
| 6 | HATFormer (TrOCR-base) | 334M | per Zenodo | KHATT 15.4% (worse than CRNN) | full FT | Yes, but low expected gain |
| Not recommended | Qari-OCR (all), Arabic-Nougat, AIN-7B zero-shot, PaddleOCR-VL, dots.ocr, DeepSeek-OCR | various | various | Printed-only or no Arabic HW evidence | n/a | n/a |

### Gaps
- No measured peak VRAM for Qwen-VL OCR LoRA on a 16GB card was found. The Qwen2.5-VL-3B license (Qwen Research) and the Qwen3-VL license were not verified in this session.
