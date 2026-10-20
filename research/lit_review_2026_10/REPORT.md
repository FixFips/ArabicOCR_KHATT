# Decoding and VLMs now beat more synthetic data

The three most promising levers for this KHATT project come from 2024-2026 evidence, not from more font rendering. The first is a properly built CTC beam-search language model: a 6-gram character model with an insertion bonus, plus a word model trained on external Arabic text. The second is a QLoRA fine-tune of an Arabic-adapted 7B vision-language model. The third is a set of cheap, HTR-tested CRNN recipe fixes: an auxiliary CTC shortcut head, 128 px aspect-preserving input and column max-pooling. The single most important caveat comes before any of these. **The 5.79% CER(n) is not evidence of beating the published state of the art.** The best peer-reviewed KHATT line CER is about **8.45%**, and it is measured on a different, harder unique-text split with no test exclusions. On the full-release split this project appears to use, the closest comparator is a community model card. There, a QLoRA-tuned Fanar-2 7B reaches **4.11% CER and 15.96% WER** using KHATT data only. That card is unreplicated. It used about 1.9x more real lines but removed train lines whose text appears in test. The project's WER of 26.4% is the clearest weakness: VLM decoders with a language prior land at 16-18% WER even when their CER is no better. The failed experiments (light font synthesis, a bigram LM, AraBART on iid noise, an extra transformer layer) all match what the literature predicts, which shows where effort should not go next. This report separates measured numbers from projections throughout. Every expected gain for this specific project is an inference, because no paper has tested these methods on this exact split.

## 5.79% sits on an easier split than the 8.45% state of the art

At least three incompatible KHATT line protocols are in circulation. The 2026 papers from the Al-azzawi group use only the **unique-text subset: 4,837 / 938 / 967 lines**. Their CRNN baseline reports **8.45 ± 0.14% CER** with no synthetic data. Cleaning the labels improved test CER by only 0.10-0.15, and the authors found **no label errors in the test set** ([arXiv 2601.16713](https://arxiv.org/html/2601.16713v1)). HATFormer uses a 72/14/14 split of 6,687 lines and gets 15.4% ([arXiv 2410.02179](https://arxiv.org/html/2410.02179)). The full KHATT v1.0 release, by contrast, decodes to **9,497 train and 1,901 validation lines** ([arXiv 2608.22316](https://arxiv.org/html/2608.22316)). That matches this project's roughly 9k training lines.

The full release includes the fixed-text paragraphs that every writer copied. A model card that trained on this release filtered out about **1,700 train lines whose exact text appears in test** ([mabdulaziz499/arabic-htr-fanar-7b-lora](https://huggingface.co/mabdulaziz499/arabic-htr-fanar-7b-lora)). That filter is direct evidence of large text overlap between train and test. The project's train CER of 0.08% shows the model memorizes training lines. So part of the 5.79% very likely reflects recognizing sentences the model has already seen written by other people.

Three more factors shift the number. **Normalization** moves scores substantially: on Muharaf, diacritic and letter-form normalization alone took HATFormer from 8.6% to 6.7% ([arXiv 2410.02179](https://arxiv.org/html/2410.02179)). **Excluding about 1.5% of test samples** as suspects breaks comparability, since the reference papers excluded nothing from test. Excluding suspects from train is fine. Finally, some KHATT mirrors store labels in **visual (reversed) order**, which silently corrupts CER if not fixed ([Fanar card](https://huggingface.co/mabdulaziz499/arabic-htr-fanar-7b-lora)).

The project's 26.4% WER has **no peer-reviewed comparator** for a trained model. The only same-split reference is the community card's table, which is not peer-reviewed. Its numbers, measured on 1,038 KHATT v1.0 test lines, are:

| System (same card) | Training data | CER % | WER % |
|---|---|---|---|
| Fanar-2-Oryx-IVU 7B + QLoRA | KHATT train+val, about 16.9k lines, test-overlap filtered | 4.11 | 15.96 |
| Fanar-2-Oryx-IVU 7B + QLoRA | KHATT + Muharaf + synthetic | 3.77 | 16.50 |
| Qwen2.5-VL-3B + LoRA | KHATT train | 6.12 | 18.43 |
| sherif1313 Qwen2.5-VL-3B v3, "zero-shot" | Card lists KHATT among sources, so leakage is possible | 5.86 | 18.30 |
| Qwen2.5-VL-7B base, zero-shot | none | 50.85 | 82.56 |

This project: CRNN, about 9k lines, suspects excluded from test: **5.79% CER, 26.4% WER.**

The honest reading has three parts. In CER, the CRNN is competitive with a 3B VLM LoRA. In WER it is about 8 points behind every fine-tuned VLM. On a fair protocol it would probably score above 5.79%. One discrepancy remains open: a separate 2026 study describes 1,038 KHATT test items as *paragraphs*, not lines ([arXiv 2608.22366](https://arxiv.org/html/2608.22366v1)). Check what the card's test set actually contains before treating it as directly comparable.

**Action before anything else:** report four numbers for every model:

| Metric | Split | Normalization | Test exclusions |
|---|---|---|---|
| (a) Current full-split CER(n) | Full release | CER(n) | none |
| (b) Raw CER | Full release | Whitespace collapse only, as HATFormer does ([arXiv 2410.02179](https://arxiv.org/html/2410.02179)) | none |
| (c) CER and WER | Unique-text 4,837 / 938 / 967 | CER(n) | none |
| (d) No-space CER | Full release | CER(n) | none |

The no-space CER separates the 1.5pp of word-boundary errors from character errors. The CER-HV seed spread of ±0.14 ([arXiv 2601.16713](https://arxiv.org/html/2601.16713v1)) also sets the noise floor. A single-seed change smaller than about 0.3pp is not a result.

## The WER gap points at language modeling, not optics

The error profile points to word-level knowledge. About 1.5pp of CER comes from space insertions and deletions, and about 0.8pp from dot groups. Deletions make up 31% of edit operations. Whether a gap after a non-connecting letter (ا د ذ ر ز و) is a word boundary depends on the whole word, not on the previous character. That is why a bigram character LM cannot fix it. Two further mechanisms explain why the bigram actively hurt. A beam LM without an insertion bonus biases decoding toward shorter outputs, which adds deletions to a model that already under-transcribes. And the weight was swept upward from 0.3 without that compensating term. Both explanations are inferences from the notes; neither was measured here.

The counter-evidence that LM decoding works at this scale is strong. PyLaia (2024) built n-gram LMs **only from each dataset's training transcriptions**, the same in-domain setup that failed here. It improved **11 of 12 datasets** using a **6-gram character KenLM with weight 1.5**. Results were IAM (6,482 train lines) CER 8.44 → 7.50 and WER 24.51 → 20.98, and RIMES CER 4.55 → 3.82 and WER 14.39 → 10.53. The average gain was about **0.9pp CER and 3.3pp WER** ([arXiv 2404.18722](https://arxiv.org/html/2404.18722v1)). The one failure, Esposalles, already had a CER below 1% on only 2,328 lines. The project is not in that regime. On Arabic, a Muharaf CNN-Transformer-CTC model card reports greedy CER of about 17.0, about 14.2 with a char 5-gram LM, and 13.31 with neural rerankers ([HF sdkv2/muharaf-arabic-ocr](https://huggingface.co/sdkv2/muharaf-arabic-ocr)). That card is not peer-reviewed, but it is the same script and the same decoder family.

The fine-tuned VLM numbers above make the same point from the other direction. A Qwen2.5-VL-3B LoRA at 6.12% CER, *worse* than the CRNN, still reaches 18.4% WER. Its autoregressive decoder makes fewer, whole-word errors where CTC spreads character errors across many words. That is an inference from the card's numbers. A word LM, a VLM, or the two combined are the levers aimed at the 26.4% WER.

## Ranked recommendations, with measured and projected gains kept apart

The ranking below orders by **strength of evidence that the lever transfers to this setup**, weighed against effort. The "measured" column reports what a cited paper actually observed, on its own dataset. The "projected" column is this report's estimate for this project, and none of those estimates has been measured on KHATT.

| Rank | Lever | Measured evidence (source, dataset) | Projected for this project (inference) | Effort | Risk |
|---|---|---|---|---|---|
| 0 | Fix evaluation: unique-text split, raw + normalized CER, no-space CER, no test exclusions, 3 seeds | CER-HV seed std ±0.14; test has no label errors ([2601.16713](https://arxiv.org/html/2601.16713v1)) | No CER gain. Reported CER likely rises on the fair split. Makes every later result trustworthy | 1-2 days | None |
| 1 | CTC beam search: 6-gram char KenLM (weight about 1-1.5) **plus insertion bonus**, then word 3/4-gram KenLM on external Arabic text via pyctcdecode | PyLaia: 11/12 datasets improved, average -0.9pp CER / -3.3pp WER, IAM WER -14% relative ([2404.18722](https://arxiv.org/html/2404.18722v1)); Muharaf card 17.0 → 14.2 → 13.3 ([HF](https://huggingface.co/sdkv2/muharaf-arabic-ocr)) | -0.3 to -0.8pp CER, -2 to -5pp WER, mostly on spaces and dot variants that form non-words | 3-5 days | Low-medium: earlier bigram failure; LM text leakage on the full split |
| 2 | VLM fine-tune: Qwen2.5-VL-3B LoRA (pipeline check), then Fanar-2-Oryx-IVU 7B QLoRA; CRNN as hallucination guard | Same-dataset card: 7B QLoRA 4.11 / 15.96, 3B LoRA 6.12 / 18.43 ([card](https://huggingface.co/mabdulaziz499/arabic-htr-fanar-7b-lora)); single unreplicated source | CER about 4.5-5.5%, WER about 17-20% on 9k lines; the largest WER lever available | 1-2 weeks (WSL2 setup, guardrails) | Medium-high: hallucinated or memorized sentences, about 100x slower inference, Windows bitsandbytes fragility (unverified) |
| 3 | CRNN recipe: CTC shortcut head (weight 0.1), 128 px height with padding instead of resize, column max-pool | IAM: shortcut 4.93 → 4.62 CER and 5.93 → 5.11; padding 5.28 → 4.93 ([2404.11339](https://arxiv.org/html/2404.11339v1)); the shortcut is part of the 8.45% KHATT CRNN ([2601.16713](https://arxiv.org/abs/2601.16713)) | -0.2 to -0.6pp CER; resolution specifically targets the 0.8pp of dot errors if current height is ≤64 px | 2-4 days | Low |
| 4 | CR-CTC consistency: two augmented views, symmetric KL on CTC posteriors, alpha 0.2 | Speech only: LibriSpeech test-other 6.02 → 4.61, Aishell 5.26 → 4.12 ([2410.05101](https://arxiv.org/html/2410.05101v2)) | Unknown for HTR; plausibly -0.2 to -0.5pp; also better-calibrated posteriors for LM fusion | 3-5 days | Medium (untested modality) |
| 5 | N-best rescoring with a 7-12B Arabic LLM as a *scorer* (log-likelihood only), gated on CTC confidence | No Arabic HTR numbers. Confidence gating limits harmful edits in speech ([2509.25048](https://arxiv.org/pdf/2509.25048)) | Small extra WER gain on top of rank 1; unmeasured | 1 week | Medium |
| 6 | Auxiliary rasm (dotless) CTC head for dot groups | Classical letter-body decomposition only ([Ahmad 2015](https://faculty.kfupm.edu.sa/ics/irfanics/docs/Ahmad2015-MSH.pdf)); no modern numbers | Unknown; targets the 0.8pp of dot errors | 2-3 days | Medium (no evidence) |
| 7 | Cross-script auxiliary real data (Urdu NUST-UHWR, Persian PHTD) | KHATT CRNN at full data: 8.80 → 8.59 ([2605.02089](https://arxiv.org/html/2605.02089)) | About -0.2pp, within seed noise | 1 week | Low |
| 8 | Large font-synthetic pretraining (500k-1M lines, 50+ fonts, CER < 0.3 filtering), then real fine-tune | HATFormer synthetic stage worth 4.2pp, but on a 334M transformer on Muharaf ([2410.02179](https://arxiv.org/html/2410.02179v2)); full-data generated pretraining hurt a CRNN on 2 of 3 sets ([2508.09936](https://arxiv.org/html/2508.09936v1)) | Under 1pp for the CRNN; matters only if the recognizer becomes a transformer | 1-2 weeks | Medium |
| 9 | Learned line generator (DiffBrush / Emuru) retrained on KHATT | IAM, CRNN + full real data: 5.78 → 4.62 with DiffBrush ([2508.03256](https://arxiv.org/html/2508.03256v1)); took about 32 RTX-4090 GPU-days | Possibly the only synthetic route that moves space errors; under 1pp CER | Weeks of GPU time | High |

Rank 1 comes ahead of rank 2, even though the VLM shows the larger measured number. The LM evidence spans 12 datasets and a peer-reviewed toolkit. The VLM evidence is one community card that used twice the training data. The LM also costs days, not weeks.

Rank 3 sits just below the VLM because its measured gains come from IAM, not Arabic. Still, it is the best-evidenced change to the optical model itself. It is also the recipe behind the strongest peer-reviewed KHATT CRNN.

## Stop what the literature already predicted would fail

**Stop free-form post-correction**, whether AraBART, ByT5 or LLM rewriting, at this CER. On KHATT specifically, a 2026 study found that VLM correction of OCR output did not help: Tesseract → Qwen gave 42.1% against 40.9% for Qwen alone. Correction made handwritten student essays worse, with median CER going from 77.4 to 90.9 ([arXiv 2608.22366](https://arxiv.org/html/2608.22366v1)). Arabic chat models are the worst offenders. On Judeo-Arabic correction, Jais-13B-Chat **halved word accuracy** relative to uncorrected input (31.1 against a 64.9 baseline), and Fanar also degraded it ([arXiv 2507.04746](https://arxiv.org/html/2507.04746v2)). Models under 8B do not reliably beat the OCR baseline ([arXiv 2609.21595](https://arxiv.org/abs/2609.21595)).

The AraBART regression also has a known cause. Synthetic pairs whose error distribution does not match the deployed OCR teach the corrector to fix the wrong errors; render→OCR pairs failed for exactly this reason in a controlled comparison ([arXiv 2408.02253](https://arxiv.org/html/2408.02253v1)). If correction is revisited at all, follow three rules. Generate pairs by k-fold decoding of KHATT train with the deployed CRNN. Constrain outputs to the CTC N-best. Run an edit-penalty λ sweep on the 1.1k real validation pairs first: a curve that only decreases toward pure copying means the edits have no value ([arXiv 2609.29680](https://arxiv.org/abs/2609.29680)).

**Deprioritize light font synthesis and spacing-jitter augmentation.** The project's -0.15pp is in line with the evidence that synthetic data matters mainly below about 130 real lines ([arXiv 2508.09936](https://arxiv.org/html/2508.09936v1)). Space errors did not move with synthetic data, which fits the view that word boundaries are a language problem, not a visual one.

**Stop adding plain transformer layers and avoid TrOCR/HATFormer.** On KHATT, the CRNN beats HTR-VT at every data size: 25.95% vs 45.46% CER at 100 lines, 13.25% vs 20.21% at 1,000 lines, and 8.80% vs 8.89% at full data ([arXiv 2605.02089](https://arxiv.org/html/2605.02089)). HATFormer's 334M model reaches only 15.4% on KHATT despite 1M synthetic pretraining lines ([arXiv 2410.02179](https://arxiv.org/html/2410.02179v2)).

**Skip zero-shot VLMs, printed-text Arabic OCR VLMs and document parsers.** Qari-OCR scores **235.55% CER** on KHATT, and Qwen2.5-VL-7B scores 50.85% zero-shot ([Fanar card](https://huggingface.co/mabdulaziz499/arabic-htr-fanar-7b-lora)). PaddleOCR-VL reports no Arabic handwriting results ([arXiv 2510.14528](https://arxiv.org/pdf/2510.14528)). AIN-7B's 0.07 KHATT CER on KITAB-Bench sits next to 0.61 on Muharaf, which looks like contamination rather than skill ([arXiv 2502.14949](https://arxiv.org/html/2502.14949v2)).

**Stop excluding test samples, and audit aggregators for leakage.** AMIDDA contains 6,673 KHATT lines ([HF calfa-ai/AMIDDA](https://huggingface.co/datasets/calfa-ai/AMIDDA)), and Online-KHATT shares KHATT's source text. Deduplicate both against test before using them.

## Protocols for the top three experiments

All three protocols share one setup. Evaluate on both the full-split test set and the unique-text split. Report raw CER, CER(n), no-space CER, WER, and space and dot error buckets. Use the same normalization script throughout, run 3 seeds where training is involved, and count only gains of 0.3pp or more as real.

### Protocol 1: LM-fused CTC decoding (no retraining)

This protocol tests whether a correctly configured LM fixes the space errors that the bigram could not.

**Step 1: dump log-probabilities.** Save CTC log-probs once for validation and test, so every decode reuses the same optical outputs.

**Step 2: build the character LM.** Train a **6-gram character KenLM with modified Kneser-Ney**, following PyLaia ([arXiv 2404.18722](https://arxiv.org/html/2404.18722v1)). Use training transcriptions with every sentence that appears in the test set removed. This is essential on the full split, because otherwise the LM inflates test scores through the fixed-text overlap. Encode spaces as an explicit token.

**Step 3: build the word LM.** Train a 3- or 4-gram word KenLM on an external Arabic corpus (Wikipedia, OSCAR or news), normalized to KHATT's character set and deduplicated against KHATT test text. Decode with pyctcdecode.

**Step 4: grid-search on validation only.** Search beam width {25, 50, 100}, LM weight α ∈ {0.3, 0.5, 0.8, 1.0, 1.5, 2.0}, and insertion bonus β ∈ {−1, 0, 1, 2, 3}. The β term is the one missing from the earlier bigram experiment, and it should counter the 31% deletion share. Run the character LM alone, the word LM alone, and both.

**Step 5: diagnose.** Report the deletion share before and after, plus the space-insertion and space-deletion counts. A rising deletion share means β is too low.

**Success criterion:** at least 0.3pp CER or 2pp WER gain on validation that holds on test. **Kill criterion:** no α/β pair beats greedy decoding on validation. If that happens, inspect the space labeling conventions in the ground truth before abandoning LMs.

### Protocol 2: VLM fine-tune with a CRNN guard

This protocol tests whether the WER gap closes with a pretrained language-vision decoder at 9k lines.

**Step 1: prepare the environment and data.** Use WSL2, given that bitsandbytes on native Windows is fragile (unverified). Verify the logical-order labels.

**Step 2: run a pipeline check.** LoRA-tune **Qwen2.5-VL-3B-Instruct in bf16** on the project's exact training lines. Use the vision tower frozen, LoRA on the LLM attention and MLP projections, r=16, alpha=32, dropout 0.05, LR 1e-4 with cosine schedule and 3% warmup, effective batch 16, and about 3 epochs ([Fanar card recipe](https://huggingface.co/mabdulaziz499/arabic-htr-fanar-7b-lora)). The card's 6.12% CER and 18.43% WER is the replication target.

**Step 3: run the main experiment.** Apply the same recipe as **4-bit NF4 QLoRA to Fanar-2-Oryx-IVU (7B)**. The card's KHATT-only result is 4.11% / 15.96% on about 16.9k lines; expect worse with 9k. Train two variants: the project's training set as is, and one with test-overlapping text removed, to measure memorization.

**Step 4: decode with guardrails.** Use greedy decoding, repetition_penalty about 1.05, and max_new_tokens proportional to image width. Reject any output whose length ratio against the CRNN hypothesis falls outside 0.5-2 and fall back to the CRNN. These are failure modes documented in [arXiv 2608.22366](https://arxiv.org/html/2608.22366v1).

**Step 5: report.** Give CER and WER with and without the fallback, the share of lines with CER above 50% (hallucination proxy), and seconds per line.

**Success criterion:** test WER at or below 21% with CER no worse than the CRNN's.

**Optional follow-up:** the Baseer__Nakba recipe, which unfreezes the vision encoder at LR 9e-6 and SLERP-merges checkpoints, won NAKBA 2026 ([card](https://huggingface.co/Misraj/Baseer__Nakba)). Those weights are CC-BY-NC-SA, so use the recipe, not the weights, if commercial use matters.

### Protocol 3: CRNN recipe ablation

This protocol tests whether HTR-tested architecture and preprocessing choices transfer to Arabic dots and spaces ([arXiv 2404.11339](https://arxiv.org/html/2404.11339v1); [code](https://github.com/georgeretsi/HTR-best-practices/)). Start from the current model and add one change at a time, 3 seeds each:

| Run | Change | IAM reference result | What to watch |
|---|---|---|---|
| A | Auxiliary CTC shortcut: 1D conv, kernel 3, output size = number of classes, on the max-pooled CNN features; its own CTC loss at weight 0.1; dropped at inference | 4.93 → 4.62 CER | Overall CER |
| B | A + input height 128 px with median-value padding to preserve aspect ratio (about 128x1024 for typical lines) instead of resizing | Padding: 5.28 → 4.93 CER | Dot-group errors specifically |
| C | B + column-wise max-pool flattening instead of concatenation | Part of the full-recipe result | Overall CER |
| D | C + CR-CTC: two different augmentations per line (mild affine, plus horizontal masking stripes as the image analogue of time masking); loss = mean of both CTC losses + 0.2 × symmetric KL between frame posteriors, with stop-gradient on the target side ([arXiv 2410.05101](https://arxiv.org/html/2410.05101v2)) | Speech only; no HTR result | Overall CER, and whether calibration improves LM fusion |

Re-run Protocol 1's best decoder on the winning checkpoint. CR-CTC is expected to make posteriors less peaky, which would compound with LM fusion.

**Success criterion:** run C beats the baseline by at least 0.3pp averaged over seeds, with the dot bucket down by at least 0.2pp.

## Conclusion

The project's bottleneck is not missing visual training signal. It lacks word-level knowledge, and its benchmark position is uncertain. Each failed experiment approached the problem as a shortage of pixels: more rendered lines, a bigger encoder, noise-trained correction. The evidence instead favors putting knowledge into decoding, where it can be checked against the image. Proper LM fusion does that cheaply. A fine-tuned VLM does it with a full language model. Both resolve the visual ambiguity at word boundaries that drives the 1.5pp of space errors and most of the 26.4% WER.

The less comfortable implication is that, on the unique-text split without test exclusions, the project may land closer to the published 8.45% than 5.79% suggests. Do that measurement first. It decides whether the right claim is "competitive CRNN" or "state of the art". It also determines whether any later gain reflects generalization or deeper memorization of fixed-text sentences. The largest open uncertainty is the VLM result itself. It rests on one unreplicated, non-peer-reviewed card, so replicating its 3B number on this project's split is the cheapest way to resolve the most consequential question in this report.
