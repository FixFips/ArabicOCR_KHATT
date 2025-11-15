# src/show_metrics.py
import os, argparse, csv
from pathlib import Path

def load_metrics_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            # cast numbers
            for k in ["epoch","train_loss","cer","wer","wer_norm","train_batches","val_batches","ckpt_saved"]:
                if k in row:
                    try:
                        row[k] = float(row[k]) if k not in ("epoch","train_batches","val_batches","ckpt_saved") else int(float(row[k]))
                    except Exception:
                        pass
            rows.append(row)
    return rows

def show_run(run_dir):
    mpath = Path(run_dir) / "metrics.csv"
    if not mpath.exists():
        print(f"[{run_dir}] metrics.csv not found.")
        return
    rows = load_metrics_csv(mpath)
    if not rows:
        print(f"[{run_dir}] no rows in metrics.csv.")
        return

    # best by CER
    best = min(rows, key=lambda r: r["cer"])
    last = rows[-1]
    print(f"\n=== {run_dir} ===")
    print(f"epochs: {len(rows)} | best epoch (CER): {best['epoch']} | best CER: {best['cer']:.3f} | WER: {best['wer']:.3f} | WER(norm): {best['wer_norm']:.3f}")
    print(f"last  : epoch {last['epoch']} | train_loss: {last['train_loss']:.3f} | CER: {last['cer']:.3f} | WER: {last['wer']:.3f} | WER(norm): {last['wer_norm']:.3f}")

    # show a short table tail
    tail = rows[-10:] if len(rows) > 10 else rows
    print("\n epoch | train_loss |   CER  |  WER  | WER(n) | saved")
    print("-------+------------+--------+-------+--------+------")
    for r in tail:
        print(f"{int(r['epoch']):5d} | {r['train_loss']:10.3f} | {r['cer']:6.3f} | {r['wer']:5.3f} | {r['wer_norm']:6.3f} |  {int(r['ckpt_saved'])}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="./runs/exp1", help="Path to a single run directory (has metrics.csv)")
    ap.add_argument("--all", action="store_true", help="Scan all subfolders in ./runs/")
    args = ap.parse_args()

    if args.all:
        root = Path("./runs")
        if not root.exists():
            print("No ./runs directory.")
            return
        for d in sorted([p for p in root.iterdir() if p.is_dir()]):
            show_run(str(d))
    else:
        show_run(args.run)

if __name__ == "__main__":
    main()