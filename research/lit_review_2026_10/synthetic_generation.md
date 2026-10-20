# Synthetic handwritten text-line generation (Arabic focus) vs font rendering for HTR

Context: CRNN-CTC on KHATT (~9k real lines, test CER(n) 5.79%). 100k uharfbuzz/10-font renders with elastic warp and Augraphy, blended at P(synth)=0.35 then fine-tuned on real only, gave 5.94 -> 5.79 (-0.15pp). Dot errors improved slightly; space errors did not change.

## Q1. Arabic-capable handwriting generators: which exist, code/weights, line vs word level, trainable on KHATT?

### Takeaway
No open Arabic **line-level** handwriting generator with weights turned up. The only Arabic-specific open generator found is **Arabic-One-DM**, which works at word level, was trained on IFN/ENIT, AlexU-Word, AHAWP and IESK-ArDB, and does not use KHATT. Line-level generators with code (DiffBrush, Emuru/Eruku, DiffusionPen paragraph mode) are trained on English/Latin, plus Chinese for DiffBrush. Using one for Arabic means retraining it on KHATT lines, and that costs a lot of compute.

### Cited Findings
- **Arabic-One-DM** ("One Stroke, One Shot: Diffusing a New Era in Arabic Handwriting Generation", Abushahla et al.) adapts One-DM to Arabic. It pretrains a ResNet-18 style encoder on a synthetic Arabic word set called Khat² (inspired by Font²) and pretrains an OCR network. Real data comes from IFN/ENIT, AlexU-Word, the AHAWP words and IESK-ArDB: 62,437 words, 1,412 unique words, 1,417 writers. It generates words, not lines. KHATT is not used. The README does not clearly say whether weights are released (it has `model_zoo/` paths). The README links an IEEE Xplore issue, but the venue and DOI fields are blank and no arXiv ID is given. — [GitHub 7abushahla/Arabic-One-DM](https://github.com/7abushahla/Arabic-One-DM)
- **One-DM** (arXiv 2409.04004, ECCV 2024) is a one-shot diffusion mimicker. It works at word level in its public release, which says "can generate handwritten words". A trained checkpoint, a pretrained OCR model and a ResNet18 are released (Google Drive/Baidu). The paper covers English, Chinese and Japanese, but only English code and data are released. The example training command uses 4 GPUs. — [GitHub dailenson/One-DM](https://github.com/dailenson/One-DM); [arXiv 2409.04004](https://arxiv.org/pdf/2409.04004)
- **DiffBrush** ("Beyond Isolated Words: Diffusion Brush for Handwritten Text-Line Generation", arXiv 2508.03256) generates whole lines and explicitly models vertical alignment and **inter-word spacing**. Its style masking is column- and row-wise, and it uses a line-level 3D-CNN discriminator plus a word-level discriminator. It is built on One-DM with the SD-1.5 VAE and samples with 50 DDIM steps. It was trained on IAM (13,353 lines), CVL (11,007 lines) and CASIA-HWDB 2.0–2.2 Chinese (52,230 lines). Code: github.com/dailenson/DiffBrush; I did not confirm whether weights are released. Training took 800 epochs on **8x RTX 4090, about 4 days**. — [arXiv 2508.03256](https://arxiv.org/html/2508.03256v1)
- **DiffusionPen** (arXiv 2409.06065, ECCV 2024) is a 5-shot latent-diffusion generator with a style encoder that combines metric learning and classification. Weights are on HF (`konnik/DiffusionPen`): IAM style encoders (class and triplet) plus the DiffusionPen IAM model, with the VAE/DDIM from SD-1.5. For a new dataset the README just says to "adjust the data loader". It has a char map (`letter2index.json`) that an Arabic alphabet would need to replace. It supports single-image and paragraph sampling. — [GitHub koninik/DiffusionPen](https://github.com/koninik/DiffusionPen); [arXiv 2409.06065](https://www.arxiv.org/pdf/2409.06065)
- **DiffusionPen was retrained on a new non-Latin script (Ukrainian, 2026)** "without architectural modification". The Ukrainian set had 126,177 word images from 308 writers. The pipeline used connected-component segmentation, quality filtering and oversampling of rare characters. The authors state they release the dataset, models and protocol. This shows a recipe for porting the model to a new script, but at word level. — [arXiv 2605.27487](https://arxiv.org/abs/2605.27487)
- **Emuru** (arXiv 2503.17074, CVPR 2025) is a VAE plus an autoregressive Transformer. It was trained only on synthetic English text rendered in more than 100k fonts, and it imitates unseen handwriting styles zero-shot without background artifacts. It is strongest at **line level**: one review reports IAM-lines BFID 6.19 and HWD 1.87, the best among compared models, with weaker word-level results. Its follow-up **Eruku** ("Autoregressive Styled Text Image Generation, but Make it Reliable", arXiv 2510.23240, WACV 2026) is listed as a line-level AR model. — [arXiv 2503.17074](https://arxiv.org/html/2503.17074v2); [arXiv 2510.23240](https://arxiv.org/pdf/2510.23240); [awesome-handwritten-text-generation list](https://github.com/koninik/awesome-handwritten-text-generation)
- Other 2025–2026 HTG models, none of them shown to support Arabic: GL-GAN (COLING 2025, beats VATr++ on FID/HWD/KID); Dual Orthogonal Guidance diffusion (arXiv 2508.17017); WriteViT (arXiv 2505.13235); FW-GAN (arXiv 2508.21040); SpiS-GAN (arXiv 2607.06949); DiffHTR (SSRN 2025, evaluated on IAM/RIMES/CASIA); CA-DIFF (IJDAR 2026). — [GL-GAN](https://aclanthology.org/2025.coling-main.166.pdf); [2508.17017](https://arxiv.org/pdf/2508.17017); [WriteViT](https://arxiv.org/pdf/2505.13235); [FW-GAN](https://arxiv.org/pdf/2508.21040); [SpiS-GAN](https://arxiv.org/pdf/2607.06949); [DiffHTR](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5612670); [CA-DIFF](https://link.springer.com/article/10.1007/s10032-026-00623-4)
- Arabic GAN augmentation exists but is small-scale. Rabi & Amrouche (Springer 2025) use GANs to augment training for a CNN-BLSTM Arabic recognizer; I did not see numbers. A 2025 review of Arabic calligraphy generation found 19 papers (2009–2024) and notes there are no standardized datasets. — [Springer chapter](https://link.springer.com/chapter/10.1007/978-3-031-80438-0_11); [IJACSA 16(3) review](https://thesai.org/Downloads/Volume16No3/Paper_81-Handwritten_Arabic_Calligraphy_Generation.pdf)
- KHATT itself provides line segmentation: 4,000 paragraphs from 1,000 writers in 18 countries. One recent split uses 4,837/938/967 lines for the unique-text subset. That is enough lines and writers to train a line generator. — [KHATT (ResearchGate)](https://www.researchgate.net/publication/259092955_KHATT_An_open_Arabic_offline_handwritten_text_database); [arXiv 2605.02089](https://arxiv.org/html/2605.02089)

### Inferences
- The most practical generator path for Arabic lines is to **fine-tune or retrain a line-level model (DiffBrush or Emuru/Eruku) on KHATT lines**, with an Arabic char map and an Arabic OCR/content loss. A cheaper path is to generate Arabic words with Arabic-One-DM or DiffusionPen and stitch them into lines. Stitching would bring back exactly the spacing problem that DiffBrush was built to fix.
- Compute on a 16GB RTX 5080: DiffBrush took about 32 RTX-4090 GPU-days. Retraining it from scratch on one 5080 would take weeks and probably needs smaller batches or gradient accumulation; this is an estimate, not a reported figure. Inference (SD-1.5 VAE plus U-Net, 50 DDIM steps) should fit in 16GB comfortably. Producing 100k lines takes hours to a day. Fine-tuning DiffusionPen (SD-1.5-scale latent U-Net) at small batch with fp16 is the most feasible on 16GB.
- Emuru was trained only on rendered fonts yet generalizes to handwriting styles zero-shot. For Arabic, that suggests the AR model could be pretrained on the existing font renders and then adapted to KHATT. This is speculative: no Arabic Emuru checkpoint was found.

### Gaps
- I could not confirm released weights for DiffBrush, Arabic-One-DM, or Eruku. Check the repos and HF directly.
- I found no 2024–2026 paper on diffusion **line-level** Arabic generation trained on KHATT. Note that "AraHWGAN" did not appear in searches.
- I found no GPU memory figures for DiffusionPen, One-DM, or Emuru training.

## Q2. Evidence that generated data improves HTR more than font rendering (IAM/RIMES/KHATT/low-resource)

### Takeaway
On IAM, learned generators added to real data give **relative CER reductions of about 7–20%** with CRNN-type recognizers (DiffBrush: 5.78 -> 4.62). That is real but modest, and **no controlled study pits them against a strong font-rendering pipeline**. The largest systematic study (Quo Vadis HTG-for-HTR, 2025) finds that generated data helps mainly when there are **fewer than about 130 real lines**. With full training sets, synthetic pretraining often ends slightly **worse** than real-only. With 9k real lines, the project is in the regime where little gain is expected from either source.

### Cited Findings
- **DiffBrush, Appendix K (IAM lines, synthetic plus full real IAM, real test):** CER/WER for each training mix.
  - Real only: 5.78/21.76.
  - +CSA-GAN: 5.39/19.89 (-6.7% relative).
  - +VATr: 5.08/19.31 (-12.1%).
  - +One-DM: 4.99/18.51 (-13.7%).
  - +DiffBrush: **4.62/16.86 (-20.1%)**.
  - Synthetic volumes are not stated. — [arXiv 2508.03256](https://arxiv.org/html/2508.03256v1)
- **Quo Vadis HTG for HTR (arXiv 2508.09936, Pippi, Nikolaidou, Cascianelli, Retsinas, Sfikas, Cucchiara, Liwicki, 2025):**
  - Setup: three off-the-shelf generators, VATr++ (GAN), DiffusionPen (diffusion) and Emuru (AR). Each generated about 23k–88k unfiltered synthetic lines per dataset. The recognizer was a **DefCRNN** (deformable-conv CRNN, CTC, no LM). Datasets were single-writer historical: Leopardi (1,303 train lines), Washington (526) and Saint Gall (468).
  - Result: with 100% real fine-tuning, unfiltered Emuru pretraining **hurt** on two of three datasets (Leopardi 6.8 vs 4.2 real-only; Washington 4.8 vs 3.6) and was close on the third (Saint Gall 5.4 vs 5.0).
  - Synthetic pretraining mattered mainly with **fewer than about 130 real lines**.
  - Emuru is most style-faithful and best for zero-shot and very-low-data use. With more fine-tuning data, the more varied DiffusionPen data generalized better.
  - Zero-shot direct transfer CER (unfiltered): Washington Emuru 16.1, DiffPen 67.0, VATr++ 84.8.
  - **CER-filtering** synthetic samples at about CER < 0.30 helped. HWD filtering did not. Too-strict filtering hurt convergence.
  - No font-rendered baseline was reported.
  - [arXiv 2508.09936](https://arxiv.org/html/2508.09936v1)
- **Glyph-conditional DDPM (arXiv 2305.19543):** a printed-glyph-to-handwriting DDPM with progressive confidence filtering. It gives about **45% relative WER reduction** on IAM over real-only. A search summary of the paper cites font-based IAM line CERs from other works (about 5.70 and 4.76) against 4.13 with DDPM-filtered synthetic data. Those numbers come from different architectures and papers, so the comparison is not controlled. — [arXiv 2305.19543](https://arxiv.org/abs/2305.19543)
- **WordStylist (arXiv 2303.16576):** synthetic plus real "considerably outperforms all other settings". Synthetic-only training reaches low error but is "not on par with the real data". — [arXiv 2303.16576](https://arxiv.org/pdf/2303.16576)
- **WriteViT / FW-GAN protocol:** TrOCR-style recognizer on 5k real IAM plus 25k synthetic. Generators are compared with each other and differ by about 0.01pp CER (3.14 vs 3.13). They are not compared with fonts. — [WriteViT](https://arxiv.org/pdf/2505.13235); [FW-GAN](https://arxiv.org/pdf/2508.21040)
- **Font rendering can be strong if it is large and diverse.** HATFormer (arXiv 2410.02179) rendered **1,000,065 Arabic lines**: 1–20 words sampled from an 8.2M-word corpus, **54 fonts**, 130 paper backgrounds and 8 geometric augmentations. On Muharaf, removing that synthetic stage raised CER from 10.4% to **14.6%** (+4.2pp). Synthetic sets of 500k–1M gave the best results. This is a transformer (TrOCR-334M) result. For reference, HATFormer's KHATT CER is **15.4%**. — [arXiv 2410.02179v2](https://arxiv.org/html/2410.02179v2)
- **Synthetic-only pretraining underperforms real pretraining for CRNN adaptation.** CRNNs pretrained on synthetic data generally did worse than those pretrained on real data when fine-tuned to single writers. — [arXiv 2305.02593](https://arxiv.org/pdf/2305.02593)
- **KHATT reference CERs:** CRNN 8.80% (8.59% with Urdu/Persian auxiliary data), HTR-VT 8.89%; cited literature includes CRNN 8.45%, TrOCR 15.40% and SRF 14.10%. Those use the unique-text split with 4,837 training lines. Another source reports 9.5% for a CNN model. The project's 5.79% CER(n) probably uses a different split or normalization and is not directly comparable. — [arXiv 2605.02089](https://arxiv.org/html/2605.02089); [arXiv 1812.11894](https://arxiv.org/pdf/1812.11894)

### Inferences
- The "3–5pp" expectation looks like it came from transformer or low-data settings, such as HATFormer's +4.2pp on Muharaf for a TrOCR model that needs Arabic visual pretraining. For a CRNN-CTC trained from scratch on 9k lines, published HTG gains on full-data IAM are about **0.4–1.2pp absolute** (DiffBrush 5.78 -> 4.62 is the best case). Quo Vadis shows they can be zero or negative.
- The unchanged **space errors** fit what is known. Font rendering, and word-level generators stitched into lines, use font or uniform spacing, while KHATT writers space irregularly. Inter-word spacing is exactly the gap DiffBrush targets with line-level training. A line-level generator trained on KHATT is the only synthetic route that could plausibly move space errors. Cheaper experiments could test this first: randomized inter-word gap jitter in the renderer, and spacing sampled from KHATT statistics.
- The 10 fonts used are far fewer than HATFormer's 54. Font and text diversity, plus volume (they used 1M lines), may matter more than the jump to diffusion for the font route.

### Gaps
- I found no controlled same-recognizer, same-budget comparison of font-rendered vs learned-generator data for any script, and none at all for Arabic.
- I found no published HTR gain from generated **Arabic** handwriting on KHATT.

## Q3. Best practices: volume, mixing, pretrain-then-fine-tune, ratios; transformers vs CRNN

### Takeaway
The evidence favors a **large synthetic pretraining stage followed by real fine-tuning**, with filtering of bad synthetic samples, over light blending. Transformers/VLMs depend far more on synthetic pretraining than CRNNs do. No paper sweeps the synthetic:real ratio for CRNN vs transformer.

### Cited Findings
- **HATFormer** recipe: Stage 1 on 1M synthetic lines, then Stage 2 fine-tuning on real data. The synthetic stage was worth 4.2pp CER, and 500k–1M lines were best. The authors argue an English OCR initialization alone is not enough for Arabic. Without pretrained weights, CER was 86%. Compute was **2–4 A100/H100** for a 334M model at batch 15/30. — [arXiv 2410.02179v2](https://arxiv.org/html/2410.02179v2)
- **Quo Vadis:** pretrain on synthetic, then fine-tune on real fractions from 1.25% to 100%. The benefit shrinks as real data grows and is significant below about 130 lines. Filter with CER < about 0.30 using an HTR model, but do not over-filter. Style similarity to the target matters less as real data grows. — [arXiv 2508.09936](https://arxiv.org/html/2508.09936v1)
- **Mixing ratios used in practice:**
  - WriteViT/FW-GAN: 5k real + 25k synthetic, a 1:5 mix. — [arXiv 2505.13235](https://arxiv.org/pdf/2505.13235)
  - Arabic transformer HTR (arXiv 2307.15045): pretrain on **500k synthetic printed lines**, then fine-tune on KHATT. — [arXiv 2307.15045](https://arxiv.org/pdf/2307.15045)
  - No ratio sweep was found.
- **CRNN vs transformer data efficiency on KHATT:** at K=100 labeled lines, CRNN scored 25.95% CER vs HTR-VT 45.46%; at K=1000, 13.25 vs 20.21. The CRNN is much more data-efficient, so the ViT-based model is the one that needs extra (synthetic or auxiliary) data. — [arXiv 2605.02089](https://arxiv.org/html/2605.02089)
- **Real auxiliary Arabic-script data helps CRNNs in low data:** joint training with Urdu NUST-UHWR and Persian PHTD dropped KHATT CRNN CER from 25.95 to 19.77 at K=100 and from 13.25 to 11.62 at K=1000. With full KHATT the gain was only 8.80 -> 8.59. Balanced batch sampling gave no consistent gain over proportional mixing. Characters shared across scripts improved by about 2pp; unique characters got worse by about 2pp. — [arXiv 2605.02089](https://arxiv.org/html/2605.02089)
- A 2026 study ("Zero-Shot Synthetic-to-Real HTR via Task Analogies", arXiv 2604.09713) compares CRNN, VAN, TrOCR and HTR-VT on synthetic-to-real transfer. — [arXiv 2604.09713](https://arxiv.org/pdf/2604.09713)

### Inferences
- P(synth)=0.35 blending at 100k lines with only 10 fonts is a light synthetic signal. If synthetic data is retried, the literature-aligned recipe has three parts:
  - pretrain fully on 500k–1M diverse lines (more fonts, corpus text that matches KHATT's domain, spacing jitter);
  - filter with the current KHATT model at about CER < 0.3;
  - fine-tune on real only, possibly keeping a small synthetic fraction (about 10–20%) as regularization; this last part is untested.
- Expect bigger payoffs if the project moves to a transformer/ViT recognizer (TrOCR/HTR-VT/VLM). For the CRNN at 9k lines, the ceiling from synthetic data is likely under 1pp.
- Real Arabic-script auxiliary data (NUST-UHWR, PHTD, plus Arabic word datasets) is a proven alternative scaling lever, though its gains shrink at full KHATT size.

### Gaps
- I found no published synthetic:real ratio sweep or curriculum schedule for line-level HTR.
- I found no VLM-specific (Qwen-VL and similar) Arabic HTR synthetic-data study in my searches.

## Q4. Other ways to scale data: self-training/pseudo-labeling, test-time augmentation

### Takeaway
Self-training with confidence filtering is well supported for HTR. It gives roughly 26–42% relative CER reductions over the seed model when there is a sizeable pool of unlabeled in-domain lines. I did not research TTA in depth.

### Cited Findings
- **Wolf & Fink (ICPR 2022, arXiv 2206.03149):** start from a model trained on font-rendered synthetic data, then iteratively pseudo-label real unlabeled data and retrain. This yields "considerable adaptation" and nearly closes the gap to full supervision on four benchmarks (word level). The **IJDAR 2024** extension adds confidence measures that drop wrong pseudo-labels and reports significant gains. — [arXiv 2206.03149](https://arxiv.org/pdf/2206.03149); [Wolf 2024 abstract](https://web.patrec.cs.tu-dortmund.de/pubs/abstracts/Wolf2024-SWR.htm)
- **Brno FIT thesis** (seed model + LM pseudo-labeling, line-level, ICFHR 2014 Bentham): CER 3.70% (-42% relative vs the seed trained on the smaller labeled set) and 1.90% (-26% relative) with the larger set. The year is not shown on the page. — [FIT VUT thesis](https://www.fit.vut.cz/study/thesis/24175/.en)
- **Unsupervised writer adaptation (Kang et al., WACV 2020, arXiv 1909.08473):** a synthetically pretrained model reaches about 26% CER on GW before adaptation and about 3% after supervised fine-tuning. — [arXiv 1909.08473](https://arxiv.org/pdf/1909.08473)

### Inferences
- For KHATT, the unused paragraph text (the fixed-text paragraphs) and other unlabeled Arabic handwriting pages could be line-segmented, pseudo-labeled with the current 5.79% model plus an Arabic LM, and filtered by CTC confidence. Self-training on CRNN-CTC is cheap on a 16GB GPU.

### Gaps
- I found no Arabic-specific self-training numbers on KHATT.
- I did not search TTA for HTR; that area is uncovered here.
