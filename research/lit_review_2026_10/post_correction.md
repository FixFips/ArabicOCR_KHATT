# OCR/HTR Post-Correction (2025 to Oct 2026): Arabic and Low-Data, Hallucination-Limited Settings

Project context: the model is a CRNN-CTC on KHATT with test CER(n) 5.79%. An AraBART corrector trained on 200k pairs with iid noise regressed dev CER from 5.85 to 6.07. Only about 1.1k real val pairs exist. The next planned round uses render→OCR pairs.

## Q1. LLM post-correction on Arabic OCR/HTR (zero-shot vs fine-tuned): when it helps and when it hurts

### Takeaway
The 2025-2026 evidence is consistent. LLM and VLM post-correction helps when the input CER is high but the OCR output can still be recovered (aged print, mixed-domain pages). It does not help, or it hurts, on handwriting, and that includes KHATT. Small and Arabic-specific chat LLMs (Jais, Fanar) can damage text badly. At a 5-6% CER starting point, there is no published Arabic evidence that a free-rewriting LLM corrector gives a net gain.

### Cited Findings
- **When Do VLMs Help Arabic Manuscript OCR? (arXiv 2608.22366, 2026)**: covers 8 Arabic datasets. It compares Tesseract 5, Qwen2.5-VL-7B (zero-shot), Fanar-2-Oryx-IVU, and QARI-OCR, plus a corrector that takes the Tesseract text and the image and passes both to the VLM. CER results:
  - Muharaf: 69.2 → 60.6 (Tesseract→Qwen).
  - KITAB-Bench: 40.2 → 32.7.
  - QNL Books: 33.1 → 15.5 (T→Fanar-2).
  - SARD: 4.7 → 2.6.
  - RASAM 2 got worse: 72.2 → 75.8.
  - **KHATT: Tesseract 67.3, Qwen standalone 40.9, T→Qwen 42.1 (−2.9% relative, so no help).**
  - Students Essays (handwriting) got worse: median 77.4 → 90.9. — [arXiv 2608.22366](https://arxiv.org/html/2608.22366v1)
  - The authors' decision rule is "OCR-prior recoverability". They give no explicit CER threshold, and the gap between the OCR and the VLM alone does not predict benefit. — [arXiv 2608.22366](https://arxiv.org/html/2608.22366v1)
  - Line-level risk on Muharaf: the corrector improved 36.4% of lines by more than 1% and worsened 44.2%, so the average hides how often lines are damaged. — [arXiv 2608.22366](https://arxiv.org/html/2608.22366v1)
  - Hallucination made up 15.9% of Qwen errors and 16.5% of Fanar-2 errors, against 0.1% for Tesseract. QARI-OCR errors were 62.4% hallucination, and its standalone CER on Muharaf was 167.2%. About 28% of Qwen outputs on student essays were repetition loops under greedy decoding. — [arXiv 2608.22366](https://arxiv.org/html/2608.22366v1)
  - On KHATT, Qwen paid a diacritics penalty of +20.2 points, the largest in the study. — [arXiv 2608.22366](https://arxiv.org/html/2608.22366v1)
- **A Tale of Two Scripts: Judeo-Arabic post-correction (arXiv 2507.04746, EACL 2026)**: post-correction was framed as MSA GEC on the output of a character-mapping step. Results on dotted input (F0.5 / word accuracy):
  - CharMapper baseline: 40.4 / 64.9.
  - GPT-4o: 63.1 / 90.4 (best).
  - sweet (Alhafni & Habash 2025): 62.3 / 86.0.
  - Seq2Seq+MG: 61.8 / 86.9.
  - GPT-3.5: 57.4 / 82.6.
  - **Fanar: 46.0 / 74.0.**
  - **Jais-13B-Chat: 16.0 / 31.1, which halved accuracy relative to the uncorrected input.**
  - Fanar also hurt the Mitelman outputs: 70.0 → 61.0 accuracy, dotless. — [arXiv 2507.04746](https://arxiv.org/html/2507.04746v2)
  - Even for the best system (GPT-4o), "unnecessary change" made up 21% of residual errors. — [arXiv 2507.04746](https://arxiv.org/html/2507.04746v2)
- **No Free Lunches (arXiv 2502.01205, RESOURCEFUL 2025)**: open-weight LLMs reduced CER on historical English. For Finnish, "a practically useful performance … was not reached". This supports the view that the gains are tied to high-resource languages. — [arXiv 2502.01205](https://arxiv.org/abs/2502.01205)
- **Benchmarking LLMs for HTR (arXiv 2503.15195, 2025)**: MLLMs show "limited ability" to correct their own zero-shot transcriptions. Proprietary models (Claude 3.5 Sonnet) beat open-source ones, and the models favour English. — [arXiv 2503.15195](https://arxiv.org/abs/2503.15195)
  - The same study, as summarised in search results, reports that open-source post-corrections reduced accuracy overall. The best case was GPT-4o on ICDAR2017 at roughly −8% CER. — [Emerald J. Documentation](https://www.emerald.com/jd/article/81/7/334/1275080/Benchmarking-large-language-models-for-handwritten) (search snippet only)
- LLM post-correction of handwriting (SOICT 2024, published 2025) reports up to −30% CER, but "results varied significantly depending on the prompt and the LLM". — [Springer](https://link.springer.com/chapter/10.1007/978-981-96-4285-4_9) (search snippet only)
- Devanagari ICL post-OCR (arXiv 2609.21595, 2026):
  - Models below 8B "do not reliably improve over the OCR baseline".
  - 3B models degrade more Marathi sentences than they improve.
  - At 12B and above the gains are reliable. Gemma-3-27B with 5 retrieved examples (CharBM25) cut WER by 55.0% on Hindi and 33.3% on Marathi.
  - Retrieving examples by character-trigram BM25 beats random selection by 2.8-4.0 pp WER. — [arXiv 2609.21595](https://arxiv.org/abs/2609.21595)

### Inferences
- KHATT is the one dataset where a 2026 study directly tested VLM correction on Arabic handwriting, and it did not help, although that started from a far worse Tesseract baseline. The project's 5.79% CER is far below any regime where these papers show gains. With a stronger base, the share of tokens that are already correct is larger, which raises the overcorrection risk.
- Do not use Jais or Fanar chat models as free-form correctors. If an LLM is tried, it should be at least 12B (or an API model such as GPT-4o or Claude), use retrieved few-shot examples taken from the 1.1k real val pairs, and be gated (see Q2).

### Gaps
- I found no 2025-2026 paper that reports fine-tuned LLM correction on KHATT at single-digit CER.
- I found no published numbers for ALLaM as an OCR corrector.
- 2608.22366 did not test frontier API models such as GPT-4o or Claude.

## Q2. Approaches that limit hallucination

### Takeaway
The strongest signals are these: give the corrector the evidence (image or N-best), restrict its output space (constrain it to edits or hypotheses), and measure the copy-to-edit trade-off as an explicit knob. Confidence gating is intuitively right, but a 2026 study warns that model confidence may not separate good edits from bad ones. ByT5/mT5 beat mBART for character-level OCR correction.

### Cited Findings
- **ByT5 vs mBART (Guan & Greene, Findings ACL 2024)**: ByT5 and mT5 were similar, and both clearly beat mBART and from-scratch models.
  - Best CER: English, ByT5 2.96 vs mT5 3.00 vs mBART 3.54. Russian, 2.14 vs 2.27 vs 2.78.
  - Even so, after correction 25.37% of Irish sentences and 8.66% of Russian sentences had higher CER. — [arXiv 2408.02253](https://arxiv.org/html/2408.02253v1)
- **Constrained decoding as a diagnostic (Confident but Wrong, arXiv 2609.29680, 2026)**, on low-resource automatic post-editing:
  - The method sweeps an edit-distance penalty λ that runs from free editing to copying the input.
  - A U-shaped TER-vs-λ curve means some edits are useful. A curve that only decreases means no edits help.
  - Failure modes: "Binary collapse", where the model either copies or makes off-target edits, and "confident miscalibration", where confidence does not separate useful edits from unnecessary ones.
  - When the curve is favourable, a static constraint gives a free gain at inference time. — [arXiv 2609.29680](https://arxiv.org/abs/2609.29680)
- **Confidence-gated correction (speech, arXiv 2509.25048, 2025)**: the LLM attempted corrections on 74.7% of low-confidence utterances against 20.3% of high-confidence ones, and harmful corrections stayed low on the high-confidence input. — [arXiv 2509.25048](https://arxiv.org/pdf/2509.25048)
- Confidence-aware OCR error detection injects OCR confidence into a BERT detector's token embeddings. This is a detection-then-correct design. — [arXiv 2409.04117](https://arxiv.org/pdf/2409.04117)
- **N-best / constrained correction**: N-best T5 (ASR) feeds several hypotheses to the corrector and constrains the decoding space. — [arXiv 2303.00456](https://arxiv.org/pdf/2303.00456)
  - Rewriting the lattice of a non-autoregressive ASR model is a related approach, but enumerating paths does not scale to dense CTC lattices. — [arXiv 2409.16469](https://arxiv.org/pdf/2409.16469)
  - A 2025 ASR-refinement survey describes LLMs used for N-best reranking with in-context demonstrations. — [arXiv 2508.07285](https://arxiv.org/pdf/2508.07285)
- **Arabic HTR with rescoring instead of rewriting**: a Muharaf CNN-Transformer-CTC model card reports:
  - greedy decoding CER about 0.170;
  - beam search with a char 5-gram LM about 0.142;
  - adding a neural LM reranker and a second reranker 0.1331.
  - This is a model card, not peer-reviewed. — [HF sdkv2/muharaf-arabic-ocr](https://huggingface.co/sdkv2/muharaf-arabic-ocr)
- The project's own KHATT card reports CER 6.09% on val with a char bigram LM beam search and no word LM. — [HF FixFips/arabicocr-khatt](https://huggingface.co/FixFips/arabicocr-khatt)
- **Edit-based Arabic correction**: text-editing (tagging) approaches for Arabic GEC are an alternative to seq2seq rewriting. See Alhafni & Habash 2025 ([arXiv 2503.00985](https://arxiv.org/pdf/2503.00985)) and STAGEET, stage-wise typed edit tagging ([arXiv 2608.28614](https://arxiv.org/pdf/2608.28614)). I did not retrieve their numbers.
- 2608.22366 used repetition control (no_repeat_ngram_size=4), which cut the failed-case CER on student essays from 199 to 66.6. It also recommends checks on length ratio, repetition, and hallucination. — [arXiv 2608.22366](https://arxiv.org/html/2608.22366v1)

### Inferences
- **Constrain the output to the CTC hypothesis space.** For a 5.8% CER CTC model, the safest corrector reranks or selects among CTC beam hypotheses (top-k per line, or per-word alternatives from the beam) using a strong Arabic LM or a small fine-tuned scorer. It should not free-generate. This removes, by construction, the failure where a correct rare word that is still valid vocabulary gets rewritten into a frequent word.
- **Run the λ sweep from 2609.29680 on the 1.1k real val pairs before any new training.** If the CER-vs-λ curve for AraBART or ByT5 only decreases toward pure copying, the corrector's edits have no value and more data from the same distribution is unlikely to fix that.
- **Gate on CTC posteriors at span level.** Allow edits only within words whose min-char or mean posterior is below a threshold tuned on val. Validate the gate on the real pairs, because 2609.29680 found that confidence can be miscalibrated.
- If a seq2seq model is kept, ByT5-small or base is preferred to AraBART. Byte-level inputs suit Arabic dot and letter-form confusions, and the 2024 comparison favoured ByT5/mT5 over mBART.

### Gaps
- I found no 2025-2026 paper that feeds CTC top-k or posteriors into an LLM corrector for HTR, Arabic or otherwise. The evidence for this comes from ASR.
- I found no copy-mechanism OCR post-correction paper from 2025-2026.

## Q3. Generating realistic training pairs (render→OCR, k-fold, visual-similarity noise) and how many pairs are needed

### Takeaway
There is direct evidence that render→OCR pairs can underperform when the rendering and OCR pipeline does not match the test-time OCR's error distribution. Real-error injection (error statistics from held-out real OCR) was best when enough data existed. Glyph-similarity noise was a strong data-free option. Data scaling showed diminishing returns, plateauing around 4× replication.

### Cited Findings
- **Guan & Greene 2024 compared four synthetic-data methods (arXiv 2408.02253)**:
  - (1) Random injection.
  - (2) "Image creation", meaning render the text and OCR it. **This underperformed.** Training used Google Vision and test used Tesseract, and because Vision is more accurate, "models mostly learned to fix low-CER sentences" and "fail to correct sentences with varying degrees of CER".
  - (3) Real-world injection, using error distributions from an extra 20% of real OCR. This was best when enough data was available (English).
  - (4) Glyph similarity. Character-pair similarity comes from ORB/AKAZE/SIFT feature matching across fonts. Error rate p is sampled from 0-15% per chunk, with substitution, deletion, and insertion rates of 5/7·p, 1/7·p, and 1/7·p, and only 1:1 errors (87.9% of errors in ICDAR 2017/19). It generally matched or beat (1) and (2).
  - ByT5 with method (4) reduced CER by 12.4-48.2% across 8 languages. — [arXiv 2408.02253](https://arxiv.org/html/2408.02253v1)
- **Data size**:
  - Replicating clean text with fresh noise at 1×, 2×, 4×, and 8× showed diminishing returns, and 4× to 8× was marginal.
  - Corpus subsets of 1, ⅔, ⅓, and ⅙ were tested, with corpora of about 5.8M to 27.9M characters.
  - Pair counts are not reported. — [arXiv 2408.02253](https://arxiv.org/html/2408.02253v1)
- RoundTripOCR (ICON 2024) releases render→OCR-style post-OCR datasets for Hindi, Marathi, Bodo, Nepali, Konkani, and Sanskrit, and frames correction as MT. I could not retrieve its pair counts or results from the abstract. — [arXiv 2412.15248](https://arxiv.org/abs/2412.15248)
- PreP-OCR (2025) trained language-specific ByT5 models on synthetic pairs, reaching about 1.53% CER (French) and 2.57% (Spanish) after post-processing. — [arXiv 2505.20429](https://arxiv.org/html/2505.20429) (search snippet only)
- Arabic GEC synthetic data: an error-tagging model plus a synthetic generation model produced balanced synthetic data, and AraT5 reached F1 79.36% on QALB-14. — [arXiv 2502.05312](https://arxiv.org/html/2502.05312v1)
- OCR-EDR (arXiv 2609.03445, 2026) renders the OCR prediction itself and compares it with the source image to diagnose and repair errors:
  - 94.78% diagnostic accuracy on its benchmark of real OCR errors (OCRErrBench);
  - 86.23% of erroneous inputs repaired to visual consistency.
  - It separates genuine errors from "rendering-equivalent" positives. — [arXiv 2609.03445](https://arxiv.org/abs/2609.03445)

### Inferences (concrete guidance for the render→OCR round)
- **The OCR model in the loop must be the exact deployed CRNN-CTC, with the same decoder and LM.** The method (2) failure in 2408.02253 came from a train/test OCR mismatch. A second failure path is a domain gap: printed or font-rendered Arabic is much easier than KHATT handwriting, so the CRNN will produce far fewer errors than on real lines and the error structure will differ. Calibrate the rendering so the CER on rendered lines matches the real val CER (about 5-6%) and the error-burst statistics match. Check the share of adjacent errors (target 27%) and the share of corrupted words that remain valid vocabulary (target 42%) against the real val pairs before training.
- **Prefer k-fold OCR on KHATT train over pure rendering.** Train k CRNNs on k−1 folds and decode the held-out fold. This gives real handwriting errors from a matched model at nearly the same CER, which matches the "real-world injection" setting that was best when data was sufficient in 2408.02253. Render→OCR can then add text diversity, used as augmentation with handwriting-like fonts and degradations.
- **Include a large share of identity pairs (input already correct)** at the rate seen in real data, and many lines with errors only in rare valid words. This teaches copying and directly targets the rare-word rewriting failure.
- **Use glyph-similarity-conditioned noise as the cheap baseline.** Build a confusion model over Arabic letter forms (dot-count and position confusions, connected forms), insert errors in bursts, and allow real-word substitutions.
- **Pair counts**: no paper gives a pair-count threshold. The 4× replication plateau suggests that text diversity and error realism matter more than pair volume. 200k pairs from a mismatched distribution did not help in this project.
- Use the 1.1k real pairs only for validation, threshold tuning, and few-shot retrieval. Do not use them for training.

### Gaps
- I found no study that measures how many render→OCR pairs an Arabic HTR corrector needs.
- I found no Arabic study comparing k-fold OCR with rendering.
- I could not retrieve RoundTripOCR's numbers.

## Q4. Multimodal correction (image + OCR text)

### Takeaway
Giving a VLM both the image and the OCR text clearly beats text-only or image-only input when the base OCR is weak, and both inputs are needed. On Arabic handwriting (KHATT, student essays) with 7B open VLMs, it did not beat the VLM alone and added hallucination.

### Cited Findings
- 2608.22366 ablation on Muharaf: the full corrector scored 60.6% CER, shuffling the OCR prior gave 81.6%, and withholding the image gave 73.9%. "Both signals are needed." — [arXiv 2608.22366](https://arxiv.org/html/2608.22366v1)
- The KHATT result was T→Qwen 42.1 against Qwen alone 40.9, and student essays got worse. — [arXiv 2608.22366](https://arxiv.org/html/2608.22366v1)
- Multimodal LLMs for historical OCR, post-correction, and NER (arXiv 2504.00414, 2025, German print): normalized CER was Transkribus 3.67% → Gemini 2.0 Flash correction 0.84% (GPT-4o 1.00%), against direct Gemini OCR at 1.27%. The paper gives no controlled text-only comparison. Having the model correct its own prior transcription gave no noteworthy gain. — [arXiv 2504.00414](https://arxiv.org/html/2504.00414)
- QARI-OCR (Qwen2-VL fine-tuned for Arabic OCR, arXiv 2506.02295) is a recognizer, not a corrector. In 2608.22366 it hallucinated heavily on manuscripts (CER 167.2%). — [arXiv 2506.02295](https://arxiv.org/abs/2506.02295); [arXiv 2608.22366](https://arxiv.org/html/2608.22366v1)
- OCR-EDR verifies a prediction by rendering it and comparing it with the image (94.78% diagnosis accuracy). This is a non-generative check that can gate edits. — [arXiv 2609.03445](https://arxiv.org/abs/2609.03445)

### Inferences
- For a corrector that fits in 16GB VRAM, a multimodal LLM is unlikely to beat a 5.8% CTC on KHATT, given the 40.9% standalone Qwen2.5-VL-7B CER. A cheaper way to use the image is to rescore candidate words with the CTC model itself. Compute the forced-alignment CTC likelihood of each candidate correction against the line image, and reject any edit the image does not support. This acts as a visual-consistency veto in the spirit of OCR-EDR.

### Gaps
- I found no 2025-2026 result for frontier API VLMs (GPT-4o, Claude, Gemini) as image+text correctors on KHATT.

## Q5. Arabic-specific correction resources and small LMs that fit 16GB

### Takeaway
Arabic GEC/spelling has moved to system combination and edit-tagging. QALB remains the benchmark. Among small LMs, SILMA-9B, ALLaM-7B, Fanar-1-9B, and Gemma-3-12B all fit in 16GB when quantized. As free-form OCR correctors, though, the Arabic chat models (Fanar, Jais) have documented failures.

### Cited Findings
- ArbESC+ (Nov 2025) combines several Arabic GEC systems: SOTA F0.5 of 82.63% on QALB-14 and 65.55% on QALB-15 L2. — [arXiv 2511.14230](https://arxiv.org/html/2511.14230v1)
- Transfer learning for Arabic spelling correction (IJICC 2025): BERT2GPT2 reached WER 6.86% against a baseline of 8.02%, and the authors release about 3.2M samples with dynamic error synthesis. — [Emerald IJICC](https://www.emerald.com/ijicc/article-abstract/doi/10.1108/IJICC-08-2025-0558/1335420/Transfer-learning-for-Arabic-spelling-correction-a) (abstract only)
- In an LLM study on Arabic spelling (QALB-based), AraBERT reached the highest top-k accuracy at 89.5%. — [Springer](https://link.springer.com/chapter/10.1007/978-3-032-10895-1_11) (snippet only)
- sweet, Alhafni & Habash's 2025 text-editing GEC, got 62.3 F0.5 in the Judeo-Arabic setting. Edit tagging suits a copy-first regime. — [arXiv 2503.00985](https://arxiv.org/pdf/2503.00985); [arXiv 2507.04746](https://arxiv.org/html/2507.04746v2)
- AraT5 trained on synthetic error data reached F1 79.36% on QALB-14. — [arXiv 2502.05312](https://arxiv.org/html/2502.05312v1)
- Small Arabic LLMs:
  - SILMA-9B led the aiXplain June 2025 Arabic benchmark together with GPT-4.1. — [aiXplain](https://aixplain.com/arabic-llm-benchmark-report) (summary only)
  - A 2026 Arabic SLM evaluation ranked Gemma-3-12B top, followed by Aya and Command Arabic. — [arXiv 2606.21460](https://arxiv.org/pdf/2606.21460)
  - ALLaM-7B: [arXiv 2407.15390](https://arxiv.org/html/2407.15390v1).
  - Fanar-1-9B and SILMA-9B were evaluated in SAHM 2026. — [arXiv 2604.19098](https://arxiv.org/pdf/2604.19098)
- For context, KITAB-Bench (ACL Findings 2025) notes that Arabic OCR still lags. — [ACL Anthology](https://aclanthology.org/2025.findings-acl.1135.pdf)

### Inferences
- **Recommended stack on 16GB.** Use (a) CTC beam search with a stronger word-level or neural Arabic LM through shallow fusion or N-best reranking. Score candidates with a 7-12B model (SILMA-9B, ALLaM-7B, or Gemma-3-12B, 4-bit), using log-likelihood only and no generation. Then (b) optionally run ByT5-small/base or an edit-tagging corrector gated to low-confidence spans, with a CTC forced-alignment veto. Use the LLMs as scorers, not as generators. That avoids the Fanar and Jais rewriting failures while still using their Arabic knowledge.
- The QALB-trained GEC models target human spelling and grammar errors, not visual confusions. They are useful as an LM-ish prior but should not be applied as is to OCR output.

### Gaps
- I did not verify CAMeL Tools 2025 releases or their spelling or morphology-based validity checks.
- I found no published evaluation of SILMA, ALLaM, or Fanar as rescorers for Arabic HTR N-best lists.
- I did not retrieve numbers for STAGEET (2608.28614) or for Alhafni & Habash 2025.
