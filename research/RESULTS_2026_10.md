# Results — October 2026 (strict split + LM decoding)

## Evaluation setup

- **Split:** `archive/splits_strict` (`scripts/build_strict_splits.py`): writer-disjoint 80/10/10 (seed 42), val/test lines with ≥50% word-3-gram overlap with train removed.
- **Clean sets:** `val_clean` / `test_clean` additionally drop 19 lines whose label belongs to a different image (verified by eye; `research/strict_label_errors.tsv`, `scripts/make_clean_splits.py`). One `#` label alone added ~12pp to mean test CER.
- **Metrics:** mean per-line CER, CER(n) (Arabic-normalized), WER. Numbers are *not* comparable to the old 5.79% (legacy split: 22% of test texts and 100% of test writers were in train).
- **Model:** CRNN-CTC (15M params), `--strip-tatweel --amp`, labels in bidi visual order (digit runs kept left-to-right).

## Headline: CRNN + n-gram LM decoding, strict `test_clean` (n=573)

| Seed | Greedy CER(n) | Greedy WER | + LM CER(n) | + LM WER |
|---|---|---|---|---|
| 42 | 9.14 | 40.9 | 7.30 | 31.9 |
| 43 | 9.32 | 41.4 | 7.25 | 32.6 |
| 44 | 9.10 | 40.0 | 7.16 | 32.3 |
| **mean** | **9.19** | **40.8** | **7.24** | **32.3** |

The LM lowers CER(n) by 1.95pp and WER by 8.5pp, consistently across seeds (seed spread ~0.1pp CER(n)).

- **LM:** 6-gram interpolated Kneser-Ney character LM plus a word-lexicon prior (`arabicocr_khatt/lm.py`), trained on strict-train labels plus 10M characters of Arabic Wikipedia. Wikipedia lines sharing a word 4-gram with val/test are removed.
- **Fusion:** shallow fusion in CTC prefix beam search (beam 10).
- **Tuned on `val_clean` (seed 42) only:** alpha=1.0, beta=3.0 (insertion bonus), gamma=0.5 (lexicon), oov=4. These are the defaults in `pipeline.py`.
- **Wikipedia-only LM** (no KHATT text, seed 42): 7.51 / 32.7.
- **Reference point:** the best peer-reviewed plain-CRNN line CER on the unique-text KHATT split is ~8.45% (arXiv 2601.16713).

## Qwen2.5-VL-3B LoRA (stopped, negative)

- **Recipe:** LoRA r16 on the LLM only, vision tower frozen, lr 1e-4, effective batch 16. Run on the strict split with model selection on `val_clean`.
- **Low resolution** (lines resized to ~112px high): CER(n) was 31.4% at step 200, 26.3% at step 400 and 32.4% at step 600.
- **Higher resolution** (~native 140px): 31.4% at step 200 and 28.8% at step 400 (200-line validation subset).
- **Failure mode:** easy lines are read well (~4% CER). On hard lines the model writes fluent but wrong Arabic, sometimes memorised fixed-paragraph text.
- **Memory:** the process commits 22–27GB on Windows, so it needs a page file of 32GB or more. Two runs were killed for low memory.

## Reproduce

```
python -m arabicocr_khatt.train_crnn_ctc --run-dir runs/exp4_strict_s42 \
    --splits-dir archive/splits_strict --seed 42 --strip-tatweel --amp
python scripts/make_clean_splits.py
python -m arabicocr_khatt.eval_val --ckpt runs/exp4_strict_s42/crnn_best.pt \
    --split test_clean --splits-dir archive/splits_strict
python scripts/lm_decode_sweep.py --ckpt runs/exp4_strict_s42/crnn_best.pt \
    --splits-dir archive/splits_strict --split test_clean --corpus archive/lm/ar_wiki.txt \
    --corpus-max-chars 10000000 --alpha 1.0 --beta 3 --gamma 0.5 --oov 4
```
