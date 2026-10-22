"""LoRA fine-tune Qwen2.5-VL on KHATT lines (step 3 of the Oct 2026 plan).

Recipe follows the community KHATT card (mabdulaziz499/arabic-htr-fanar-7b-lora)
scaled to fit 16GB without 4-bit: bf16 base weights, LoRA r=16/alpha=32 on the
LLM attention + MLP projections, vision tower frozen, lr 1e-4, effective batch
16, gradient checkpointing. Same writer-disjoint split as the CRNN
(archive/splits_strict) so the comparison is fair.

Run in the separate VLM env (.venv-vlm) — never the CRNN training venv:
    .venv-vlm/Scripts/python.exe scripts/vlm/train_qwen_vl.py --run-dir runs/vlm_qwen3b
    .venv-vlm/Scripts/python.exe scripts/vlm/train_qwen_vl.py --run-dir runs/vlm_qwen3b \
        --eval-only --split test
"""
import argparse
import math
import os
import random
import re
import sys
import time
import unicodedata as ud
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from rapidfuzz.distance import Levenshtein

IMAGES_DIR = "./archive/images"
PROMPT = "اكتب النص العربي المكتوب بخط اليد في هذه الصورة كما هو."
_DIAC = re.compile(r"[ً-ْ]")
_WS = re.compile(r"\s+")


# ---- metrics (same definitions as arabicocr_khatt.metrics) ----
def norm_ar(s):
    s = ud.normalize("NFKC", s).replace("ـ", "")
    s = _DIAC.sub("", s)
    s = s.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ى", "ي")
    return _WS.sub(" ", s).strip()


def cer(r, h):
    return (0.0 if not h else 1.0) if not r else Levenshtein.distance(list(r), list(h)) / len(r)


def wer(r, h):
    r, h = r.split(), h.split()
    return (0.0 if not h else 1.0) if not r else Levenshtein.distance(r, h) / len(r)


def read_label(path):
    for enc in ["windows-1256", "utf-8", "utf-8-sig"]:
        try:
            with open(path, encoding=enc) as f:
                return _WS.sub(" ", f.read()).strip().replace("ـ", "")
        except Exception:
            pass
    raise RuntimeError(path)


def load_split(splits_dir, split):
    df = pd.read_csv(Path(splits_dir, f"{split}.csv"))
    return [(os.path.join(IMAGES_DIR, f), read_label(p)) for f, p in zip(df["filename"], df["label_path"])]


def messages(with_answer=None):
    m = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    if with_answer is not None:
        m.append({"role": "assistant", "content": [{"type": "text", "text": with_answer}]})
    return m


class Collate:
    def __init__(self, processor):
        self.p = processor
        tok = processor.tokenizer
        self.im_start = tok.convert_tokens_to_ids("<|im_start|>")

    def __call__(self, batch):
        imgs = [Image.open(f).convert("RGB") for f, _ in batch]
        texts = [self.p.apply_chat_template(messages(t), tokenize=False) for _, t in batch]
        enc = self.p(text=texts, images=imgs, return_tensors="pt", padding=True)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        for i, ids in enumerate(enc["input_ids"].tolist()):
            # mask everything up to and including "<|im_start|>assistant\n"
            last = max(j for j, x in enumerate(ids) if x == self.im_start)
            labels[i, : last + 3] = -100
        enc["labels"] = labels
        return enc


@torch.no_grad()
def evaluate(model, processor, items, bs=8, max_new=160, out_tsv=None, limit=None):
    model.eval()
    tok = processor.tokenizer
    tok.padding_side = "left"
    items = items[:limit] if limit else items
    prompt = processor.apply_chat_template(messages(), tokenize=False, add_generation_prompt=True)
    refs, hyps = [], []
    for i in range(0, len(items), bs):
        chunk = items[i:i + bs]
        imgs = [Image.open(f).convert("RGB") for f, _ in chunk]
        enc = processor(text=[prompt] * len(chunk), images=imgs, return_tensors="pt", padding=True).to(model.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, repetition_penalty=1.05)
        dec = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for (_, ref), h in zip(chunk, dec):
            refs.append(ref); hyps.append(_WS.sub(" ", h).strip())
    tok.padding_side = "right"
    m = {"CER": sum(map(cer, refs, hyps)) / len(refs),
         "CER(n)": sum(cer(norm_ar(r), norm_ar(h)) for r, h in zip(refs, hyps)) / len(refs),
         "WER": sum(map(wer, refs, hyps)) / len(refs),
         "WER(n)": sum(wer(norm_ar(r), norm_ar(h)) for r, h in zip(refs, hyps)) / len(refs),
         "n": len(refs)}
    if out_tsv:
        with open(out_tsv, "w", encoding="utf-8") as f:
            f.write("label\tpred\n")
            for r, h in zip(refs, hyps):
                f.write(f"{r}\t{h.replace(chr(9), ' ')}\n")
    model.train()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--run-dir", default="runs/vlm_qwen3b")
    ap.add_argument("--splits-dir", default="./archive/splits_strict")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-pixels", type=int, default=256 * 28 * 28,
                    help="Cap image area; a 136x2006 KHATT line becomes ~112x1792 = 256 visual tokens.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-every", type=int, default=200, help="optimizer steps between quick val evals")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--split", default="test")
    ap.add_argument("--val-split", default="val_clean", help="split used for adapter selection")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None, help="smoke test: stop after N optimizer steps")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, get_cosine_schedule_with_warmup

    random.seed(args.seed); torch.manual_seed(args.seed)
    os.makedirs(args.run_dir, exist_ok=True)
    processor = AutoProcessor.from_pretrained(args.model, min_pixels=32 * 28 * 28, max_pixels=args.max_pixels)
    processor.tokenizer.padding_side = "right"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()

    adapter = Path(args.run_dir, "adapter_best")
    if args.eval_only:
        model = PeftModel.from_pretrained(model, str(adapter)).cuda()
        items = load_split(args.splits_dir, args.split)
        t0 = time.time()
        m = evaluate(model, processor, items, out_tsv=Path(args.run_dir, f"eval_{args.split}.tsv"), limit=args.limit)
        print(f"[{args.split}] " + " ".join(f"{k}={v:.4f}" if k != "n" else f"n={v}" for k, v in m.items())
              + f"  ({time.time() - t0:.0f}s)")
        return

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    cfg = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, bias="none",
                     # language-model projections only; the vision tower stays frozen
                     target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    train = load_split(args.splits_dir, "train")
    val = load_split(args.splits_dir, args.val_split)
    collate = Collate(processor)
    loader = torch.utils.data.DataLoader(train, batch_size=args.bs, shuffle=True, collate_fn=collate,
                                         num_workers=args.workers, persistent_workers=args.workers > 0)
    steps = math.ceil(len(loader) * args.epochs / args.accum)
    if args.max_steps:
        steps = args.max_steps
        val = val[: args.limit or 16]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * steps), steps)
    log = open(Path(args.run_dir, "train_log.csv"), "a", encoding="utf-8")
    log.write("time,step,loss,val_cer,val_cer_n,val_wer\n")

    best = float("inf"); step = 0; micro = 0; run_loss = 0.0; t0 = time.time()
    model.train()
    done = False
    while not done:
        for batch in loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss / args.accum
            loss.backward()
            run_loss += loss.item(); micro += 1
            if micro % args.accum:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            if step % 20 == 0:
                print(f"step {step}/{steps} loss={run_loss / 20:.4f} lr={sched.get_last_lr()[0]:.2e} "
                      f"{(time.time() - t0) / step:.2f}s/step", flush=True)
                run_loss = 0.0
            if step % args.eval_every == 0 or step == steps:
                m = evaluate(model, processor, val)  # full val each time: consistent selection
                print(f"  val@{step}: CER={m['CER']:.4f} CER(n)={m['CER(n)']:.4f} WER={m['WER']:.4f} (n={m['n']})", flush=True)
                log.write(f"{time.strftime('%H:%M:%S')},{step},{loss.item() * args.accum:.4f},"
                          f"{m['CER']:.4f},{m['CER(n)']:.4f},{m['WER']:.4f}\n"); log.flush()
                if m["CER(n)"] < best:
                    best = m["CER(n)"]
                    model.save_pretrained(str(adapter))
                    print(f"  -> saved adapter (val CER(n)={best:.4f})", flush=True)
            if step >= steps:
                done = True
                break

    # the selected adapter is scored on test separately: --eval-only --split test
    print(f"training done; best val CER(n) = {best:.4f}; adapter at {adapter}")


if __name__ == "__main__":
    main()
