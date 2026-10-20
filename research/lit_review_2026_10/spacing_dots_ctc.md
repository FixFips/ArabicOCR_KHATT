# Reducing space/word-boundary errors, dot confusions, and improving CTC training/decoding for Arabic HTR (KHATT, CRNN+BiLSTM+CTC)

Context: CRNN+BiLSTM+CTC, about 9k KHATT training lines, test CER 5.79%, WER 26.4%. Space ins/del is about 1.5pp of CER. Dot-group substitutions are about 0.8pp. Ops split 50% sub, 31% del, 19% ins. An in-domain bigram char LM hurt at weight 0.3 or more.

Research budget note: 16 tool calls. Several items (inter-CTC, self-conditioned CTC, focal CTC, SAM/EMA, PAW-unit CTC on KHATT, KHATT space conventions) could not be checked against primary sources in this pass. They are listed in Gaps, not stated as facts.

## Q1. Word-boundary handling in Arabic HTR (PAW, space tokens, lexicon/WFST, external word LMs, space-aware CER, KHATT labeling)

### Takeaway
I found no 2023-2026 primary source that directly targets the "non-connecting letter gap vs. word space" ambiguity on KHATT with measured gains. The best-evidenced lever for word-boundary errors is LM-assisted decoding. The LM needs more capacity than a bigram (6-gram char) or must be word-level/lexicon-based, which carries the word-segmentation knowledge that the image cannot. One cheap, HTR-tested labeling trick: pad transcriptions with leading and trailing spaces during training (Retsinas et al.).

### Cited Findings
- Retsinas et al. add spaces before and after every transcription during training and strip them at test time. Their system is a CNN+3xBiLSTM+CTC with greedy decoding and no LM/lexicon on IAM (HTR-tested; no isolated ablation for the trick). — [arXiv 2404.11339](https://arxiv.org/html/2404.11339v1)
- PyLaia (Tarride et al., 2024) decodes with an n-gram LM built only from each dataset's training transcriptions, with no external data. This is the same "in-domain" setup that hurt in this project, yet it helped on 11 of 12 datasets. It used a 6-gram char LM with modified Kneser-Ney smoothing, a fixed LM weight of 1.5 (tuned on NorHand v3), and the gain was larger on WER than on CER. HTR-tested, no Arabic dataset. — [arXiv 2404.18722](https://arxiv.org/html/2404.18722v1)
  - IAM (6,482 train lines, the closest scale to KHATT's 9k): CER 8.44 to 7.50, WER 24.51 to 20.98 (-14.4% relative WER).
  - RIMES (10,188 train lines): CER 4.55 to 3.82, WER 14.39 to 10.53 (-26.8% rel).
  - Average gain: about 0.9pp CER and 3.3pp WER absolute (11.9% / 12.9% relative).
  - Counter-example: Esposalles (2,328 lines, CER already 0.76) got worse (CER 0.76 to 1.04, WER 2.62 to 3.38). The LM hurts when the optical model is already very accurate or the data is tiny.
  - The paper does not describe how spaces or word delimiters are represented in the char LM.
- Lamtougui et al. (2023) used CNN+BLSTM+CTC with a Word Beam Search decoder on KHATT and reported 80.15% accuracy. WBS constrains words to a dictionary or LM while allowing free non-word characters. These figures come from a search summary; the primary source was not opened. — [search summary pointing to KHATT literature](https://arxiv.org/abs/2406.15329v1)
- Aabed and Khairaldin (2024), arXiv:2406.15329: DCNN+BLSTM+CTC trained from scratch on KHATT. They report 84% character and 71% word recognition on test, with no evidence of an LM in the abstract. — [arXiv 2406.15329](https://arxiv.org/abs/2406.15329v1)
- Older KHATT work reported HMM baselines with a char-bigram LM at about 46% char accuracy. A comparison table lists HMM at 53.87% CER, MDLSTM at 24.20% CER, and CNN-based at about 8.7–9.9% CER. These come from search snippets of secondary sources and could not be verified. — [search result incl. arXiv 2207.04651](https://arxiv.org/pdf/2207.04651)

### Inferences
- The bigram LM likely hurt for three reasons. (a) A bigram is far too weak to model word boundaries: whether a space follows ا/د/ر/و depends on the whole word, not one previous char. (b) Weight was swept upward from 0.3, but beam CTC LMs usually also need an insertion bonus (word/char bonus "beta") to counter the LM's bias toward shorter outputs. Without it, an LM worsens deletions, and this model already under-transcribes (31% del). (c) The LM was trained on the train set it was optimizing against. Recommendation: build a 6-gram char KenLM (PyLaia-style, weight around 1–1.5, plus a tuned insertion bonus). Separately, build a word 3/4-gram KenLM on external Arabic text (OSCAR/Wikipedia/news, normalized to KHATT's character set) and decode with pyctcdecode (word LM, alpha/beta grid on validation). From PyLaia's IAM/RIMES numbers, expect roughly 10–25% relative WER and about 10% relative CER reduction. Space errors are the error class a word LM most directly fixes, because it scores in-vocabulary words. This last point is an inference, not measured on Arabic.
- An explicit "word-boundary" token is effectively what CTC's space char already is. A lexicon/WFST (or WBS) adds a vocabulary of valid words, so gaps after non-connectors inside a known word are not split. This is the principled fix for the visual ambiguity the project hypothesizes.
- Labeling-convention check (cheap, do first): audit KHATT ground truth for inconsistent spaces, e.g. spaces after non-connectors or before و conjunctions written attached, and double spaces. Also report CER with and without spaces (a "no-space CER") so you can see how much of the 1.5pp is boundary ambiguity versus character errors.

### Gaps
- I found no primary source describing KHATT's space-labeling convention or a KHATT-specific "space-aware CER" protocol. I also found no 2023–2026 CTC paper using PAW (piece-of-Arabic-word) units on KHATT with numbers.
- No Arabic HTR paper with numbers for pyctcdecode + KenLM on external text (OSCAR/Wikipedia) was found. Gains are inferred from Latin-script PyLaia results.
- The Muharaf paper (arXiv 2406.09630) reportedly evaluated KHATT and Muharaf with CTC + char n-gram decoding (search snippet), but the abstract page has no numbers. The full PDF should be checked. — [arXiv 2406.09630](https://arxiv.org/abs/2406.09630)

## Q2. External LMs for CTC (large n-gram, neural shallow fusion, LLM N-best rescoring)

### Takeaway
n-gram LM decoding is well evidenced in HTR at this data scale, even when trained in-domain (PyLaia: IAM WER -14% rel, RIMES -27% rel). Evidence that LLM post-correction or rescoring helps *Arabic* OCR is weak or negative. Treat LLM rescoring as low priority and N-best-constrained only.

### Cited Findings
- PyLaia n-gram decoding numbers: see Q1 (6-gram char, weight 1.5, in-domain training text only; 11/12 datasets improved; CASIA Chinese CER 4.61 to 1.53). HTR-tested. — [arXiv 2404.18722](https://arxiv.org/html/2404.18722v1)
- A 2026 Kazakh post-OCR correction study (Arabic, Cyrillic and Latin scripts) reports that Arabic-script post-OCR correction remained unsuccessful across all setups tested (search summary). — [ACL Anthology 2026.cawl-1.8](https://aclanthology.org/2026.cawl-1.8/)
- An arXiv 2026 study on VLM OCR of Ancient Greek editions also tested Arabic. For Tesseract-ara, LLM post-correction harmed more lines than it helped (1,377 vs 629), with a reported median CER shift of about +2.24 points. The table was hard to parse in the summary, so verify. — [arXiv 2605.27750](https://arxiv.org/pdf/2605.27750)
- For handwritten Latin-script (BRESSAY, RIMES, IAM), closed LLMs (Gemini/GPT) consistently reduced CER more than open models in post-OCR correction. — [STIL/SBC 2025](https://sol.sbc.org.br/index.php/stil/article/view/37859)
- Qalam (arXiv 2407.13559) is a multimodal LLM for Arabic OCR/HTR that includes KHATT-type handwriting. It shows the end-to-end LLM route exists for Arabic, but its numbers were not extracted here. — [arXiv 2407.13559](https://arxiv.org/html/2407.13559)

### Inferences
- If an LLM is used, restrict it to *rescoring the CTC N-best* (e.g. top-10 from pyctcdecode), interpolating LLM log-prob with the CTC score. Do not use free-form correction, which hallucinates and normalizes Arabic. Free-form correction is where negative Arabic results come from. Space errors are a plausible win for rescoring, because word-segmentation alternatives appear in the N-best list. This is unmeasured.
- Neural shallow fusion (char-level LSTM/Transformer LM on large Arabic text) is speech-proven. I found no Arabic HTR numbers. A KenLM word model is a cheaper first step with the same intent.

### Gaps
- No primary source found reporting shallow fusion with a neural LM, or LLM N-best rescoring, on Arabic handwritten lines with numbers.

## Q3. CTC training improvements (joint CTC/attention, inter-CTC, self-conditioned CTC, label smoothing/focal, CR-CTC, SAM/EMA)

### Takeaway
Two cheap, well-documented additions. (1) The HTR-tested "CTC shortcut" auxiliary head gives -0.3 to -0.8pp CER on IAM. (2) CR-CTC consistency regularization is speech-only, but its two-view augmentation design maps directly onto HTR image augmentation. Gains from joint CTC/attention in HTR are small.

### Cited Findings
- **CTC shortcut (HTR-tested):** a 1D conv (kernel 3, output = #classes) on the max-pooled CNN features, trained with its own CTC loss weighted 0.1 and dropped at inference. IAM line test, padded + max-pool: CER/WER 4.93/16.81 without the shortcut, 4.62/15.89 with it. Resized + concat: 5.93/19.57 to 5.11/16.96. Column-wise max-pool flattening and aspect-preserving padding each also helped (resized+concat no-shortcut 5.93 vs padded+max-pool+shortcut 4.62 CER). Setup: 128x1024 line input, Adam lr 1e-3, 240 epochs. — [arXiv 2404.11339](https://arxiv.org/html/2404.11339v1)
- **CR-CTC (speech-only, ICLR 2025, arXiv 2410.05101):** two independently augmented views pass through the model. Loss = mean of the two CTC losses + alpha x bidirectional KL between per-frame posteriors (stop-grad on the target side), alpha = 0.2 (swept 0.1/0.2/0.3). Time masking is increased 2.5x. — [arXiv 2410.05101](https://arxiv.org/html/2410.05101v2)
  - LibriSpeech test-clean/other: Zipformer-M CTC 2.52/6.02 to CR-CTC 2.10/4.61. Zipformer-L 2.50/5.72 to 2.02/4.35.
  - Aishell-1 test: Zipformer-S 5.26 to 4.12. GigaSpeech test: XL 10.87 to 10.28.
  - SR-CTC (CTC + smoothing KL to suppress peaky outputs), a related ablation, also helps: 2.51/6.02 to 2.32/5.22 on LibriSpeech, versus CR-CTC's 2.12/4.62.
  - Not tested on any non-speech data.
- **Joint CTC + attention/CE in HTR:** the 3D-attention HTR (arXiv 2410.18374) trains with CTC + CE but decodes with CTC only. Reported IAM gain over CTC-only is negligible (about 94.41 vs 94.37% accuracy). — [arXiv 2410.18374](https://arxiv.org/pdf/2410.18374). In WSRNet (word-level), joint CTC + Seq2Seq gave WER 16.5% vs 21.6% for Seq2Seq alone. The benefit there flows to the attention branch, not to CTC. — [arXiv 2008.07109](https://arxiv.org/pdf/2008.07109). An earlier seq2seq HTR study found an auxiliary CTC loss "not necessary". — [arXiv 1903.07377](https://arxiv.org/pdf/1903.07377)

### Inferences
- Priority 1: add the CTC shortcut. It is about 10 lines of code, HTR-tested, and gives -0.3 to -0.8pp CER on IAM.
- Priority 2: CR-CTC adapted to images. Make two different augmentations per line (affine/elastic plus horizontal masking stripes as the analogue of time masking), with symmetric KL on CTC posteriors, alpha about 0.2. Speech shows 5–25% relative WER reduction. HTR transfer is unverified. Its effect of reducing over-confident, peaky CTC distributions could also make LM fusion work better, because posteriors become better calibrated. PyLaia also stresses calibration.
- The 31% deletion share suggests peaky/blank-dominant CTC. SR-CTC/CR-CTC-style smoothing and an insertion bonus at decode time are the natural counters.

### Gaps
- Not verified this pass, so no numbers are given: inter-CTC (Lee & Watanabe 2021) and self-conditioned CTC (Nozaki & Komatsu 2021). Both are speech papers, and I found no HTR evidence. Focal CTC and label-smoothed CTC in HTR, and SAM/EMA weight averaging for HTR, also had no primary source located.

## Q4. Dot-confusion fixes (resolution, dot-aware augmentation, auxiliary losses, base-shape + dot decomposition)

### Takeaway
Base-shape ("letter-body") plus dot decomposition is an established idea in classical Arabic HTR (HMM era), but I found no 2023–2026 CTC/attention paper with a decomposed output vocabulary and numbers. The most actionable evidence-backed lever is input resolution and aspect preservation: Retsinas uses 128 px line height and padding instead of resizing.

### Cited Findings
- Menasri et al. (cited by Ahmad, KFUPM) mapped Arabic letters sharing a core shape and differing only in dots into one "letter-body" class in an HMM/NN hybrid. Ahmad's multi-stage system recognizes core shapes first and then adds diacritic and dot information. — [Ahmad 2015, KFUPM](https://faculty.kfupm.edu.sa/ics/irfanics/docs/Ahmad2015-MSH.pdf)
- Ahmad and Fink (2015, 2019) studied separating dots from rasms during OCR, as cited in the dotless-Arabic NLP work. — [arXiv 2312.16104](https://arxiv.org/html/2312.16104v1)
- Dotless (rasm-only) Arabic text gives NLP performance comparable to dotted text with up to 50% smaller vocabularies, and a dot-restoration system is presented (Al-Shaibani and Ahmad, Computational Linguistics 2025). — [ACL Anthology 2025.cl-2.5](https://aclanthology.org/2025.cl-2.5)
- Wshah et al. (ICPR 2010) used dot descriptors for lexicon reduction: 87% reduction at >93% accuracy on IFN/ENIT. — [Buffalo](https://researchconnect.buffalo.edu/en/publications/a-novel-lexicon-reduction-method-for-arabic-handwriting-recogniti/)
- Al-Jawfi (IAJIT 2009) used a two-stage network: main shape first, then dot recognition. — [IAJIT](https://www.ccis2k.org/iajit/PDF/vol.6,no.3/15.pdf)
- HTR-tested resolution evidence: Retsinas uses 128 px line height (128x1024). Padding to preserve aspect ratio, instead of resizing, cut CER from 5.28 to 4.93 (max-pool, no shortcut). — [arXiv 2404.11339](https://arxiv.org/html/2404.11339v1)

### Inferences
- A practical modern version of the decomposition idea: keep the normal CTC char head and add an auxiliary CTC head over rasm (dotless) classes, collapsing ب/ت/ث/ن/ي to one skeleton and ج/ح/خ, ف/ق, د/ذ, ر/ز to theirs. Optionally add a per-frame dot-count/position head (0/1/2/3, above/below). The deterministic char-to-rasm mapping makes the rasm labels free. The rasm head gives the CNN a clean shape signal, and the char head must then learn the dots. This is untested; I found no paper with numbers.
- Dot errors are small-blob errors, so (a) line height of 64 px or below plus aggressive downsampling probably destroys dots; try 96–128 px. (b) Augmentations that blur, erode or erase can delete dots and teach label noise. Keep morphological erosion mild, and avoid random erasing of small patches above the baseline. A word LM (Q1) also resolves many dot confusions, since most dot variants of a skeleton are non-words.

### Gaps
- No 2023–2026 CTC/attention Arabic HTR paper with base-shape+dot output units or dot-count auxiliary losses and reported numbers was found.

## Q5. Stronger augmentation at small scale and "best practices"

### Takeaway
The HTR-tested best-practice recipe (Retsinas et al., 2024) uses only *mild* augmentation: small-rotation/skew affine plus Gaussian noise. Its gains come mainly from architecture and preprocessing (padding instead of resizing, column max-pool, CTC shortcut). Heavier augmentation (elastic/TPS, erasing, mixup) was not confirmed with HTR numbers in this pass.

### Cited Findings
- Retsinas et al. (arXiv 2404.11339, the "best practices" paper; the prompt's "2022" refers to the DAS 2022 presentation): global affine with small rotation/skew + Gaussian noise; 128x1024 lines with median-value padding; ResNet backbone (7x7 conv32, then 2x64, 4x128, 4x256 blocks), column max-pool, 3 BiLSTM (256), greedy CTC. IAM line test CER 4.62 / WER 15.89 with all practices, versus 5.93 / 19.57 for the naive variant. — [arXiv 2404.11339](https://arxiv.org/html/2404.11339v1); [DAS 2022 slides](https://iapr.org/archives/das2022/wp-content/uploads/OS-slides/74.pdf); code: https://github.com/georgeretsi/HTR-best-practices/
- Multi-scale training is proposed alongside 3D attention for HTR (arXiv 2410.18374). — [arXiv 2410.18374](https://arxiv.org/pdf/2410.18374)
- In speech, CR-CTC shows that stronger masking (2.5x) pays off *when paired with a consistency loss*. — [arXiv 2410.05101](https://arxiv.org/html/2410.05101v2)

### Inferences
- Since word-space jitter and synthetic spacing variation did not help, the spacing problem is unlikely to be fixed by augmentation alone. That supports the hypothesis that the problem is labeling or semantics, so the fixes belong in the LM/lexicon side (Q1).
- Combine stronger augmentation with CR-CTC rather than using it alone. Two different augmentations per line plus a KL term is the speech-proven way to benefit from heavy augmentation.

### Gaps
- No primary HTR source with numbers was located in this pass for elastic/TPS distortion, random erasing, or mixup variants on Arabic/KHATT. Older HTR work (e.g. Wigington et al. 2017 elastic augmentation; the CNN-only HTR of arXiv 1812.11894) exists but was not verified here.
