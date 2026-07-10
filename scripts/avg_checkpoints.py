# scripts/avg_checkpoints.py
"""Average the weights of several checkpoints into a "model soup".

Train with --save-topk K to keep the K best epoch checkpoints, then:

    python scripts/avg_checkpoints.py runs/exp2/crnn_ep*.pt --out runs/exp2/crnn_soup.pt
    python -m arabicocr_khatt.eval_val --ckpt runs/exp2/crnn_soup.pt --split val

Averaging checkpoints from nearby epochs of the same run typically buys
0.1-0.3 pp CER for zero inference cost. All checkpoints must share the same
vocab and arch_version.
"""
import argparse
import glob
import sys
from pathlib import Path

import torch


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ckpts", nargs="+", help="Checkpoint paths (globs ok)")
    ap.add_argument("--out", required=True, help="Output checkpoint path")
    args = ap.parse_args()

    paths: list[str] = []
    for pat in args.ckpts:
        hits = glob.glob(pat)
        paths.extend(hits if hits else [pat])
    paths = sorted(set(paths))
    if len(paths) < 2:
        print(f"Need >=2 checkpoints to average, got {len(paths)}: {paths}")
        return 1

    states = [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    ref = states[0]
    for p, s in zip(paths, states):
        if s.get("vocab") != ref.get("vocab") or s.get("arch_version") != ref.get("arch_version"):
            print(f"Checkpoint mismatch (vocab/arch): {p}")
            return 1

    avg = {}
    for k, v in ref["model"].items():
        if v.dtype.is_floating_point:
            avg[k] = sum(s["model"][k].to(torch.float64) for s in states) / len(states)
            avg[k] = avg[k].to(v.dtype)
        else:  # int buffers (e.g. BatchNorm num_batches_tracked): keep first
            avg[k] = v.clone()

    out = {**ref, "model": avg}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"Averaged {len(paths)} checkpoints -> {args.out}")
    for p in paths:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
