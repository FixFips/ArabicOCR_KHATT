# Arabic HTR SOTA (KHATT and other benchmarks) and open Arabic handwriting line datasets, as of Oct 2026

Project reference point: CRNN+BiLSTM+CTC (~15M params), KHATT line split with ~9k real training lines, test CER(n) 5.79% and WER 26.4%, with ~1.5% of samples treated as mislabeled and excluded.

## Q1. Best reported KHATT line-level CER/WER in 2024-2026, and which methods reach it

### Takeaway
No 2024-2026 paper I found reports a KHATT line CER below about 8.4%. The best published numbers come from plain CRNN/CNN models trained only on real data: 8.45% for CER-HV's CRNN (Al-azzawi et al., 2026), 8.77% for FCN (Yousef et al.) and 8.9% for DAN (Aljishi et al., 2024). Transformer and TrOCR-style models do worse on KHATT: HATFormer gets 15.4% and the cross-attention Transformer 18.45%. Zero-shot VLMs are far worse again, at 19-160% CER. Almost no paper reports KHATT line WER. The project's 5.79% is not directly comparable to these numbers (see Q2).

### Cited findings: dense KHATT results table

| # | Method (paper) | Arch / params | Extra data / pretraining | KHATT split and granularity | CER % | WER % | Source |
|---|---|---|---|---|---|---|---|
| 1 | CER-HV CRNN baseline (Al-azzawi et al., 2026) | 7x7 conv, then ResNet blocks (64/128/256), then 3x BiLSTM-256 and CTC (greedy). Auxiliary CTC shortcut (weight 0.1) at train time. Params not stated | None (real KHATT only, no synthetic) | Unique-text subset only: 4,837 / 938 / 967 lines (6,742 total) | 8.45 ± 0.14 raw. Cleaned train+eval: test 8.3, val 7.93 | n/r | [arXiv 2601.16713](https://arxiv.org/abs/2601.16713) |
| 2 | Same CRNN, "Performance gap Latin vs Arabic" (2026) | CRNN-CTC, AdamW | None | 4,791 / 938 / 967 (cleaned) | 8.72 (Table 3) vs 8.45 (Table 7); the paper contradicts itself | n/r | [arXiv 2606.18884](https://arxiv.org/html/2606.18884) |
| 3 | HTR-VT (same group, 2026) | CNN + 4-layer ViT + CTC, SAM, EMA, span masking | None | Same unique-text split | 8.89-8.91 | n/r | [arXiv 2605.02089](https://arxiv.org/html/2605.02089); [arXiv 2606.18884](https://arxiv.org/html/2606.18884) |
| 4 | Cross-lingual joint training (2026) | CRNN (as above) | Aux. Urdu NUST-UHWR + Persian PHTD (J=2) | Unique-text split, K=4,500 | 8.80 (J=0), then 8.59 (J=2). Gains are large in low resource: K=100 goes 25.95 to 19.77 | n/r | [arXiv 2605.02089](https://arxiv.org/html/2605.02089) |
| 5 | DAN (Aljishi, Mughaus, Luqman, Parvez, ISI 29(6), Dec 2024) | Document Attention Network. Also compared FCN, GFN, VAN | Not verified | Not verified | 8.9 (as cited by CER-HV and 2606.18884) | DAN is best on WER, value not verified | [KFUPM record](https://pure.kfupm.edu.sa/en/publications/a-comparative-study-of-four-handwritten-text-recognition-models-i/); [arXiv 2601.16713](https://arxiv.org/abs/2601.16713) |
| 6 | FCN (Yousef, Hussain, Mohammed, Pattern Recognition 108, 2020) | Fully convolutional, gated, CTC | None | Not verified | 8.77 (as cited by CER-HV) | n/r | [arXiv 1812.11894](https://arxiv.org/pdf/1812.11894); cited in [arXiv 2601.16713](https://arxiv.org/abs/2601.16713) |
| 7 | EfficientNetB4 + BLSTM | CNN-BLSTM | Synthetic data | Not verified | 13.34 (as cited by CER-HV) | n/r | [arXiv 2601.16713](https://arxiv.org/abs/2601.16713) Table 2 |
| 8 | CNN+BiLSTM+CTC (same ref [5]) | CRNN | Not verified | Not verified | 15.45 (as cited) | n/r | [arXiv 2601.16713](https://arxiv.org/abs/2601.16713) |
| 9 | Muharaf baseline "SFR"/Start-Follow-Read CRNN (Saeed et al., NeurIPS 2024 D&B) | CRNN | Synthetic data | 72/14/14 of 6,687 lines | 14.1 | n/r | [arXiv 2406.09630](https://arxiv.org/abs/2406.09630); [arXiv 2410.02179](https://arxiv.org/html/2410.02179) |
| 10 | HATFormer (Chan et al., 2024, rev. 2025) | TrOCR-base (BEiT encoder + RoBERTa decoder), ~334M. BlockProcessor + Arabic BPE tokenizer | TrOCR English pretraining + 1M synthetic Arabic lines | 72/14/14 of 6,687 lines (the dataset authors' recommendation) | 15.4. With KHATT+Muharaf+MADCAT combined: 15.3 | n/r | [arXiv 2410.02179](https://arxiv.org/html/2410.02179) |
| 11 | Transformer cross-attention seq2seq (Momeni & BabaAli, SIViP 2024) | Transformer enc-dec. The Transformer-Transducer variant got 19.76 | Synthetic data | Not verified | 18.45 | n/r | [arXiv 2307.15045](https://arxiv.org/pdf/2307.15045) |
| 12 | Lamtougui et al. 2023 hybrid | CRNN | Not stated | Not stated | 19.9 | n/r | cited in [arXiv 2410.02179](https://arxiv.org/html/2410.02179) |
| 13 | KITAB-Bench zero-shot (ACL Findings 2025), 200 KHATT line samples | Gemini-2.0-Flash / GPT-4o / AIN / Qari / Azure / Tesseract | Zero-shot | 200 sampled lines (pool of 13.4k line annots) | Gemini 19, GPT-4o 45, GPT-4o-mini 64, Tesseract 67, Azure 83, Qari 61, Qwen2-VL 112, Qwen2.5-VL 160. AIN 7 (suspect, see Q2) | Gemini 45, GPT-4o 74, AIN 22 | [arXiv 2502.14949](https://arxiv.org/html/2502.14949) |
| 14 | "When do VLMs help Arabic manuscript OCR?" (Aug 2026) | Qwen2.5-VL-7B zero-shot, Tesseract, Tesseract→Qwen corrector | Zero-shot | 1,038 KHATT test **paragraphs** | Qwen2.5-VL-7B 40.9 (diacritics-normalized). Tesseract 67.3. Corrector 42.1. Qwen's diacritics penalty on KHATT is +20.2 pts | n/r | [arXiv 2608.22366](https://arxiv.org/html/2608.22366v1) |
| 15 | Invizo (Feb 2025), **Online-KHATT** (not offline KHATT) | 3-layer CNN + 6-layer Transformer encoder + 6-layer decoder, d=256 | 84k synthetic printed lines (Naqsh Arabi) | Online-KHATT 6,974 / 767 / 766 | 7.91 | 31.41 | [arXiv 2502.05277](https://arxiv.org/html/2502.05277v1) |

Cross-dataset (zero-shot transfer) HATFormer numbers, useful as domain-gap baselines ([arXiv 2410.02179](https://arxiv.org/html/2410.02179)):
- Train Muharaf, test KHATT: 22.8 (CRNN baseline 38.5).
- Train MADCAT, test KHATT: 16.3.
- Train KHATT, test Muharaf: 40.7.

### Inferences
- On KHATT, plain CRNN/FCN-style CTC models trained only on real data (about 8.4-8.9%) beat large pretrained transformers (15-18%), even with synthetic pretraining. This is the reverse of the Muharaf picture, where HATFormer wins (Q3). The likely reason is that KHATT is small and writer-diverse, and image resizing and tokenization for ViT hurt there.
- Cross-script auxiliary data (Urdu/Persian) helps most in low-data regimes: about 6 CER points at K=100, and only about 0.2 at full data ([arXiv 2605.02089](https://arxiv.org/html/2605.02089)). So "more real lines" probably helps the project less than people expect, unless the lines are in-domain modern handwriting.

### Gaps
- I could not verify the exact KHATT CER/WER, split, or normalization of DAN (Aljishi 2024) or FCN (Yousef 2020): the PDFs could not be parsed. Treat 8.9 and 8.77 as numbers cited by others.
- No 2024-2026 paper reported KHATT line WER for a trained model, so the project's WER of 26.4% has no published comparator. The only proxy is Online-KHATT WER of 31.4% (Invizo).
- I found no published fine-tuned VLM (Qwen2.5-VL/Qari/AIN LoRA) result on the full KHATT line test set.
- The AMIDDA paper (HIP@ICDAR 2026) evaluates a "foundation VLM for Arabic HTR" on a mix that includes KHATT. The HAL page was bot-blocked, so I have no numbers.

## Q2. How comparable are these numbers (split, normalization, CER with or without spaces)?

### Takeaway
They are poorly comparable. At least three different KHATT line protocols are in use:
- the unique-text subset of about 6.7k lines (4,837/938/967, or 72/14/14 of 6,687);
- the full release of about 9.5k train / 1.9k val lines, which includes the fixed-text paragraphs;
- tiny 200-sample or paragraph-level VLM subsets.

Normalization ranges from none, to whitespace collapse (HATFormer), to full Arabic normalization (alef/ya/ta-marbuta unification and diacritic stripping). A training set of ~9k lines means the project almost certainly uses the full split with fixed-text lines. That split is easier because the same fixed-paragraph text appears for many writers across splits. So 5.79% CER(n) is not evidence of beating the 8.45% SOTA.

### Cited findings
- CER-HV and 2606.18884 use only the KHATT unique-text subset: 4,837/938/967 (6,742 lines) and 4,791/938/967 after cleaning ([arXiv 2601.16713](https://arxiv.org/html/2601.16713v1); [arXiv 2606.18884](https://arxiv.org/html/2606.18884)).
- HATFormer uses a 72/14/14 split of 6,687 lines, described as the dataset authors' recommendation ([arXiv 2410.02179](https://arxiv.org/html/2410.02179)).
- 2608.22316 decodes the full KHATT v1.0 release into 9,497 train and 1,901 validation lines. That count matches the project's ~9k figure ([arXiv 2608.22316](https://arxiv.org/html/2608.22316)).
- The KHATT database has 1,000 writers, 2,000 random paragraphs and 9,327 lines. Each writer also wrote fixed-text paragraphs (the original paper; summarized in [KHATT paper PDF](https://faculty.kfupm.edu.sa/ics/alshayeb/publications/J20-KHATT.pdf); [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0031320313003300)).
- KITAB-Bench draws its KHATT samples from a pool of 13.4k line and 2.72k paragraph annotations, and evaluates only 200 per granularity ([arXiv 2502.14949](https://arxiv.org/html/2502.14949)).
- HATFormer CER is Levenshtein (S+D+I)/N over characters. Consecutive whitespace is collapsed at test time, as in the HF `evaluate` v0.4.2 CER, so spaces are counted. Collapsing at train time gave another −0.2 CER. Normalization (removing diacritics and similar) reduces Muharaf CER from 8.6 to 6.7, which shows how much it shifts scores ([arXiv 2410.02179](https://arxiv.org/html/2410.02179)).
- HATFormer notes that Rawls et al.'s MADCAT 1.52% is low partly because of NIST OpenHART normalization ([arXiv 2410.02179](https://arxiv.org/html/2410.02179)).
- 2608.22316 applies NFKC, removes tatweel and diacritics, collapses whitespace, and unifies alef/hamza/ya/ta-marbuta. It reports raw and normalized CER side by side ([arXiv 2608.22316](https://arxiv.org/html/2608.22316)).
- 2608.22366 strips diacritic ranges U+0610-061A, U+064B-065F, U+0670 and U+06D6-06ED, unifies alef, removes tatweel, and computes CER/WER at corpus level ([arXiv 2608.22366](https://arxiv.org/html/2608.22366v1)).
- CER-HV found label errors in 0.9% of KHATT train lines (40 transcription + 4 segmentation errors), 5 in validation and none in test. Cleaning improved test CER by only 0.10-0.15 ([arXiv 2601.16713](https://arxiv.org/html/2601.16713v1)).
- Several 2026 papers say outright that external KHATT results "are not directly comparable" because training setups differ ([arXiv 2606.18884](https://arxiv.org/html/2606.18884)).
- KITAB-Bench's handwriting table contains implausible values, for example a Surya KHATT row identical to its ISI-PPT row, and AIN's ADAB CER of 0.00. The AIN KHATT number (7% CER) should be treated with caution ([arXiv 2502.14949](https://arxiv.org/html/2502.14949)).

### Inferences
- To compare fairly with the literature, the project should re-run on the unique-text split (4,837/938/967 per CER-HV) with HATFormer's whitespace-collapse-only metric, or report raw and normalized CER as 2608.22316 does. If the full split has fixed-text overlap, the 5.79% probably reflects partial text memorization: identical sentences written by different writers in train and test.
- The project excludes 1.5% of samples as mislabeled. CER-HV removed none from test, so excluding test samples also breaks comparability. Excluding from train only is fine.

### Gaps
- Most papers do not say whether CER is computed with or without spaces. HATFormer is the clearest (spaces counted, runs collapsed).
- I did not confirm whether CER-HV's or HATFormer's splits are writer-independent.

## Q3. New Arabic HTR datasets and benchmarks in 2024-2026, and their suitability for modern handwriting like KHATT

### Takeaway
2025-2026 produced several large line-level Arabic sets, but almost all are historical manuscripts:
- Muharaf, 24.5k lines;
- AraMS-28k, 28.6k lines;
- the Omar al-Saleh memoirs, about 20.7k lines;
- AMIDDA, a 66.8k-line aggregate;
- OpenITI-Makhzan, about 23k lines.

KITAB-Bench and 2608.22366 are evaluation benchmarks, not training sets. I found no new large modern-script line dataset that is open like KHATT. The Omar al-Saleh memoirs (1951-65 personal handwriting, CC BY 4.0) are the closest stylistically.

### Cited findings: dataset table

| Dataset | Size (lines) | Style / domain | License | Commercial OK? | URL | Fit for KHATT-like modern script |
|---|---|---|---|---|---|---|
| KHATT (2014) | 9,327 lines per the original paper. 9,497 train + 1,901 val in the full release. ~6.7k unique-text. 11,398 per the aggregator | Modern unconstrained handwriting, 1,000 writers from 18 countries | Free for research. Also sold by LDC under the Non-Member User Agreement (fee) | Not stated; assume no | [LDC2015T23](https://catalog.ldc.upenn.edu/LDC2015T23); [paper](https://faculty.kfupm.edu.sa/ics/alshayeb/publications/J20-KHATT.pdf) | — (baseline) |
| Muharaf (NeurIPS 2024 D&B, rev. Feb 2025) | 24,495 lines (22,091 / 1,069 / 1,334 HF split), ~1,600 pages | Historical (19th-21st c.) Levantine letters, journals, poetry, church and legal records; very diverse hands | Public part (1,216 images) CC BY-NC-SA 4.0. Restricted part (428 images) research-only on request from the Phoenix Center | No (NC) | [HF aamijar/muharaf-public](https://huggingface.co/datasets/aamijar/muharaf-public); [arXiv 2406.09630](https://arxiv.org/abs/2406.09630); [license discussion](https://huggingface.co/datasets/aamijar/muharaf-public/discussions/2) | Medium. Some 20th-c. personal handwriting, but HATFormer cross-tests show a large domain gap (Muharaf→KHATT 22.8 CER) |
| AraMS-28k (arXiv Aug 2026) | 28,600 lines (27,971 main + 629 margin), 14 books, 3,043 pages. Book-level split 19,739 / 1,486 / 6,746 | Historical manuscripts: Naskh, Ruq'ah, Maghrebi, plus 1 lithograph | CC BY-NC-SA 4.0 | No | [arXiv 2608.26921](https://huggingface.co/papers/2608.26921); HF "Archatext/AraMS-28k-HTR"; Zenodo DOI 10.5281/zenodo.22095333 (via the aggregator card) | Low-medium. The Ruq'ah books are closest to everyday handwriting. Baselines: Kraken 11.65 / HATFormer 13.26 on Ruq'ah |
| Omar al-Saleh memoirs (NAKBA-NLP 2026 shared task, LREC 2026) | 15,969 train + 2,095 test + 2,671 blind test images (about 18-20.7k lines) | Single-author handwritten memoirs, 1951-1965 | CC BY 4.0 (gated: accept conditions) | Yes (CC BY) | [HF U4RASD/omar-al-saleh-manuscripts-segments](https://huggingface.co/datasets/U4RASD/omar-al-saleh-manuscripts-segments) | High style fit (20th-c. personal Ruq'ah/Naskh-like hand), but one writer means low writer diversity |
| AMIDDA (HIP@ICDAR 2026) | 66,779 lines (53,420 / 6,675 / 6,684). Aggregates RASAM-1 7,331, RASAM-2 3,720, TARIMA 2,285, BAYBARS 15,567, ISKANDAR 5,221, KHATT 6,673, Muharaf 24,494 (RASM 1,488 not redistributed) | Mixed Maghrebi/Oriental manuscripts + modern | Per-source (mixed) | Depends on subset | [HF calfa-ai/AMIDDA](https://huggingface.co/datasets/calfa-ai/AMIDDA); [HAL hal-05693582](https://enc.hal.science/hal-05693582) | The non-KHATT/Muharaf parts are historical. Useful for pretraining diversity. Caution: it contains KHATT lines, which could leak test data |
| Arabic Manuscript Collection (TheSeniorTeam, HF) | 82,561 labelled images, aggregating Omar al-Saleh, Muharaf, AraMs-28k, AHDQ (642 Quran pages), optional KHATT, AMIDDA and OpenITI-Makhzan | Mostly historical | No single license; the NC subsets make combined use non-commercial | No (as a whole) | [HF card](https://huggingface.co/datasets/TheSeniorTeam/Arabic_Manuscript_Collection_Dataset) | Convenience loader; check for duplicates and KHATT leakage |
| OpenITI-Makhzan | 23,079 lines, 1,497 pages, 208 documents | Multilingual manuscripts (Arabic, Persian, Ottoman Turkish, Urdu), six scripts | Unverified (the aggregator says to treat as all rights reserved) | Unknown | [AKU record](https://scholars.aku.edu/en/publications/openiti-makhzan-an-open-annotated-dataset-of-arabic-persian-ottom/) | Low |
| RASAM / RASAM 2 | ~7.3k + ~3.7k lines, ~300 pages (RASAM 2) | Maghribi manuscripts, PAGE-XML | Public (open; exact license not verified here) | Not verified | [RASAM paper](https://halshs.archives-ouvertes.fr/halshs-03430697v1/file/RASAM_A_Dataset_for_the_Recognition_and.pdf); [arXiv 2211.16147](https://arxiv.org/pdf/2211.16147) | Low (Maghribi script differs strongly) |
| Online-KHATT | 6,974 / 767 / 766 line images (rendered from pen trajectories) | Modern, same source text as KHATT unique paragraphs | Research | Not stated | [arXiv 2502.05277](https://arxiv.org/html/2502.05277v1) | High style fit, but the renderings come from online strokes and the **text overlaps KHATT** (leakage risk) |
| Arabic Handwritten Text Line Dataset (2023) | Historical, word-position annotations | Historical | See paper | — | [arXiv 2312.07573](https://arxiv.org/abs/2312.07573) | Low |
| KITAB-Bench (ACL Findings 2025): benchmark only | 8,809 samples, 9 domains. Handwriting = 200 each from KHATT line, KHATT paragraph, ADAB, Muharaf, OnlineKhatt | Eval | Paper CC BY 4.0; data license not stated | — | [arXiv 2502.14949](https://arxiv.org/abs/2502.14949); [project](https://mbzuai-oryx.github.io/KITAB-Bench/) | Eval only |
| VLM manuscript OCR study (Aug 2026): benchmark only | Muharaf 1,334 lines, RASAM 2 159 pages, KHATT 1,038 paragraphs, Madinah 40 pages, SARD, and in-house QNL Books and student essays (unreleased) | Eval | Not stated | — | [arXiv 2608.22366](https://arxiv.org/html/2608.22366v1) | Eval only. The student-essay set (Grades 4-12) would be ideal but is unreleased |
| Cross-script Arabic-script line sets used by the Al-azzawi group | NUST-UHWR (Urdu) 10,605 lines; PHTI (Pashto) ~36k; PHTD (Persian) 1,787; Ajami (Hausa/Fulfulde) 6,132. The CER-HV paper releases cleaned versions | Modern (except Ajami) | Varies (not verified) | Not verified | [arXiv 2601.16713](https://arxiv.org/abs/2601.16713); [arXiv 2605.02089](https://arxiv.org/html/2605.02089) | Medium as auxiliary joint-training data (+0.2 CER gain at full data) |

Other benchmark results (2024-2026):
- **Muharaf line:**
  - HATFormer: 8.6 (Arabic-only), 11.7 (full), 6.7 with normalization post-processing. CRNN baseline: 17.6 / 14.9 ([arXiv 2410.02179](https://arxiv.org/html/2410.02179)).
  - CRNN (Al-azzawi): 10.11-10.30. HTR-VT: 11.79 ([arXiv 2606.18884](https://arxiv.org/html/2606.18884)).
  - Qwen2.5-VL-7B + LoRA: 45.0 ([arXiv 2608.22316](https://arxiv.org/html/2608.22316)).
  - KITAB-Bench zero-shot: Gemini-2.0-Flash 33, Qari 38, GPT-4o 56 ([arXiv 2502.14949](https://arxiv.org/html/2502.14949)).
- **MADCAT line (LDC, not open):** HATFormer 4.2, Shtaiwi 2022 4.0, Rawls 2018 1.52 (normalized) ([arXiv 2410.02179](https://arxiv.org/html/2410.02179)).
- **AraMS-28k test books:** Kraken 11.65 / 22.62 / 32.71 and HATFormer 13.26 / 25.37 / 37.88 (Ruq'ah / Naskh / Maghrebi) ([HF paper page 2608.26921](https://huggingface.co/papers/2608.26921)).
- **IFN/ENIT (word level, not line):** reported accuracies of 94.7-99.9% across non-comparable protocols ([survey arXiv 2312.11812](https://arxiv.org/pdf/2312.11812); [BEEI 2024](https://beei.org/index.php/EEI/article/download/8178/3983)).
- **Staged KHATT→Muharaf warm-up (2608.22316):** under compute-matched conditions, a KHATT warm-up made Muharaf *worse* (+2.42 CER, CI [+0.60, +4.25]) ([arXiv 2608.22316](https://arxiv.org/html/2608.22316)).

### Inferences
- For a modern-script KHATT model, the best open additions are:
  - Omar al-Saleh (CC BY, 20th-c. hand);
  - Muharaf, Arabic-only lines, for diverse hands, though NC;
  - Urdu/Persian/Pashto line sets for cross-script joint training.

  The evidence says historical data transfers asymmetrically and modestly. Synthetic handwriting-font lines (HATFormer used 1M) are probably at least as valuable.
- Deduplicate any aggregator (AMIDDA, TheSeniorTeam, Online-KHATT) against the KHATT test set before training.

### Gaps
- I did not find a 2025-26 open dataset of *modern* multi-writer Arabic handwriting lines comparable to KHATT. Unreleased in-house sets exist (student essays in 2608.22366).
- BADAM was not checked in detail. It is a Zenodo-hosted historical *baseline-detection* set, not a transcription set.
- I did not check AHTD or Hijja. Hijja is children's isolated characters, so it is not line-level.
- I did not verify the RASAM license, the OpenITI-Makhzan license, or the exact AraMS-28k download URL.

## Q4. Which datasets allow research-only versus commercial use?

### Takeaway
Only CC BY-licensed sets clearly allow commercial use. Of the line-level Arabic handwriting sets, that means the Omar al-Saleh memoirs (CC BY 4.0, gated) and AHDQ (MIT, page-level Quran). KHATT is research-only or LDC-licensed. Muharaf and AraMS-28k are CC BY-NC-SA 4.0, which means non-commercial use and share-alike. Aggregates inherit the most restrictive license.

### Cited findings
- KHATT is free for research ([KHATT paper](https://faculty.kfupm.edu.sa/ics/alshayeb/publications/J20-KHATT.pdf)), and LDC distributes it under the Non-Member User Agreement for a fee. Commercial terms are not stated ([LDC2015T23](https://catalog.ldc.upenn.edu/LDC2015T23)).
- Muharaf: the public part is CC BY-NC-SA 4.0 and the restricted part is research-only. A user question about commercial use and model weights had no official answer ([HF discussion #2](https://huggingface.co/datasets/aamijar/muharaf-public/discussions/2)).
- AraMS-28k: CC BY-NC-SA 4.0 ([HF paper page](https://huggingface.co/papers/2608.26921)).
- Omar al-Saleh: CC BY 4.0, gated ([HF](https://huggingface.co/datasets/U4RASD/omar-al-saleh-manuscripts-segments)).
- AMIDDA: mixed per-source licenses ([HF](https://huggingface.co/datasets/calfa-ai/AMIDDA)).
- TheSeniorTeam collection: no single license. Its NC subsets make combined use non-commercial. OpenITI-Makhzan should be treated as all-rights-reserved until verified. AHDQ is MIT ([HF](https://huggingface.co/datasets/TheSeniorTeam/Arabic_Manuscript_Collection_Dataset)).

### Inferences
- Commercial deployment is effectively limited to:
  - Omar al-Saleh, plus synthetic data and any data the project collects itself;
  - KHATT only if LDC or KFUPM grant commercial terms, which needs confirmation.

  Models trained on NC data (Muharaf, AraMS-28k) carry legal ambiguity for commercial use.

### Gaps
- Whether model weights trained on CC BY-NC-SA data count as derivative works is unresolved in the sources.
- KHATT's commercial terms are not published.
