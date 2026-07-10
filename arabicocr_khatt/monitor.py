# arabicocr_khatt/monitor.py
"""
Live training monitor — serves a web dashboard that reads metrics.csv + sample predictions.
Access from any device on the same network.

Usage:
    python -m arabicocr_khatt.monitor                  # default port 8080
    python -m arabicocr_khatt.monitor --port 9090      # custom port
"""

import os
import csv
import json
import glob
import re
import argparse
import subprocess
from collections import defaultdict
from http.server import HTTPServer, BaseHTTPRequestHandler
from rapidfuzz.distance import Levenshtein as _Lev

METRICS_PATH = os.path.join("runs", "exp1", "metrics.csv")
RUN_DIR = os.path.join("runs", "exp1")
SYNTH_CSV = os.path.join("archive", "synth", "synth.csv")

# Dot-group definitions (letters sharing the same base stroke, differing only by dots)
_DOT_GROUPS_LIST = [
    {"name": "ba / ta / tha", "chars": ["\u0628", "\u062a", "\u062b"]},
    {"name": "jim / ha / kha", "chars": ["\u062c", "\u062d", "\u062e"]},
    {"name": "nun / ya", "chars": ["\u0646", "\u064a"]},
    {"name": "fa / qaf", "chars": ["\u0641", "\u0642"]},
]


def _find_latest_tsv():
    """Return (path, epoch_num) for the most recent val samples TSV, or (None, None)."""
    pattern = os.path.join(RUN_DIR, "val_epoch_*_samples.tsv")
    files = sorted(glob.glob(pattern))
    if not files:
        return None, None
    latest = files[-1]
    basename = os.path.basename(latest)
    epoch_str = basename.replace("val_epoch_", "").replace("_samples.tsv", "")
    try:
        epoch_num = int(epoch_str)
    except ValueError:
        epoch_num = 0
    return latest, epoch_num


# --- Cached analysis: one editops pass, recomputed only when the TSV changes ---
_analysis_cache = {"mtime": None, "path": None, "char_errors": [], "confusion": [],
                   "ops": {}}

_ALL_DOT_CHARS = set()
for _g in _DOT_GROUPS_LIST:
    _ALL_DOT_CHARS.update(_g["chars"])


def _refresh_analysis_cache():
    """Recompute analysis from latest TSV if the file has changed. Returns (epoch, cache)."""
    path, epoch = _find_latest_tsv()
    if path is None:
        return None, _analysis_cache

    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None, _analysis_cache

    if path == _analysis_cache["path"] and mtime == _analysis_cache["mtime"]:
        return epoch, _analysis_cache  # cache hit

    # --- Read samples ---
    samples = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                samples.append((row.get("label", ""), row.get("pred", "")))
    except Exception:
        return None, _analysis_cache

    # --- Single pass: compute editops once per sample, feed both analyses ---
    char_total = defaultdict(int)
    char_subs = defaultdict(lambda: defaultdict(int))
    char_dels = defaultdict(int)
    dot_confusion = defaultdict(lambda: defaultdict(int))
    dot_correct = defaultdict(int)
    dot_deleted = defaultdict(int)
    dot_total = defaultdict(int)

    n_sub = n_del = n_ins = n_space = 0
    for gt, pr in samples:
        for ch in gt:
            char_total[ch] += 1
        ops = _Lev.editops(gt, pr)
        consumed = set()
        for op, i1, i2 in ops:
            if op == "replace":
                consumed.add(i1)
                n_sub += 1
                if gt[i1] == " " or pr[i2] == " ":
                    n_space += 1
                char_subs[gt[i1]][pr[i2]] += 1
                if gt[i1] in _ALL_DOT_CHARS:
                    dot_total[gt[i1]] += 1
                    dot_confusion[gt[i1]][pr[i2]] += 1
            elif op == "delete":
                consumed.add(i1)
                n_del += 1
                if gt[i1] == " ":
                    n_space += 1
                char_dels[gt[i1]] += 1
                if gt[i1] in _ALL_DOT_CHARS:
                    dot_total[gt[i1]] += 1
                    dot_deleted[gt[i1]] += 1
            elif op == "insert":
                n_ins += 1
                if pr[i2] == " ":
                    n_space += 1
        for i, ch in enumerate(gt):
            if ch in _ALL_DOT_CHARS and i not in consumed:
                dot_total[ch] += 1
                dot_correct[ch] += 1

    # --- Build per-char result ---
    char_errors = []
    for ch in sorted(char_total, key=lambda c: char_total[c], reverse=True):
        total = char_total[ch]
        sub_total = sum(char_subs[ch].values())
        dels = char_dels[ch]
        errors = sub_total + dels
        top_subs = sorted(char_subs[ch].items(), key=lambda x: x[1], reverse=True)[:5]
        char_errors.append({
            "char": ch,
            "total": total,
            "correct": total - errors,
            "errors": errors,
            "error_rate": round(errors / total, 4) if total else 0,
            "top_subs": [[k, v] for k, v in top_subs],
            "deletions": dels,
        })

    # --- Build confusion matrices ---
    confusion_groups = []
    for g in _DOT_GROUPS_LIST:
        chars = g["chars"]
        matrix = []
        for gt_ch in chars:
            row = []
            for pr_ch in chars:
                if gt_ch == pr_ch:
                    row.append(dot_correct.get(gt_ch, 0))
                else:
                    row.append(dot_confusion.get(gt_ch, {}).get(pr_ch, 0))
            matrix.append(row)
        confusion_groups.append({
            "name": g["name"],
            "chars": chars,
            "matrix": matrix,
            "deletions": [dot_deleted.get(ch, 0) for ch in chars],
            "totals": [dot_total.get(ch, 0) for ch in chars],
        })

    _analysis_cache["mtime"] = mtime
    _analysis_cache["path"] = path
    _analysis_cache["char_errors"] = char_errors
    _analysis_cache["confusion"] = confusion_groups
    _analysis_cache["ops"] = {"sub": n_sub, "del": n_del, "ins": n_ins, "space": n_space}
    return epoch, _analysis_cache


# --- Run status: liveness, ETA, baseline delta, GPU ---------------------------
TOTAL_EPOCHS = None            # --total-epochs, else parsed from train.log
BASELINE_RUN = os.path.join("runs", "exp1")

_status_cache = {"gpu_t": 0.0, "gpu": None, "baseline": None, "baseline_key": None}


def _read_metric_rows():
    rows = []
    if not os.path.exists(METRICS_PATH):
        return rows
    with open(METRICS_PATH, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            reader.fieldnames = [n.strip() for n in reader.fieldnames]
        for row in reader:
            try:
                rows.append({"ts": row.get("timestamp", "").strip(),
                             "epoch": int(float(row["epoch"])),
                             "cer": float(row["cer"])})
            except (ValueError, KeyError, TypeError):
                pass
    return rows


def _gpu_stats():
    """nvidia-smi snapshot, cached 5s. None when unavailable."""
    import time as _time
    now = _time.time()
    if now - _status_cache["gpu_t"] < 5:
        return _status_cache["gpu"]
    gpu = None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,"
             "temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4)
        if out.returncode == 0 and out.stdout.strip():
            u, mu, mt, t, p = [v.strip() for v in out.stdout.strip().splitlines()[0].split(",")]
            gpu = {"util": int(float(u)), "mem_used": int(float(mu)),
                   "mem_total": int(float(mt)), "temp": int(float(t)),
                   "power": round(float(p))}
    except Exception:
        gpu = None
    _status_cache["gpu_t"] = now
    _status_cache["gpu"] = gpu
    return gpu


def _baseline_best():
    """Best CER of the baseline run (cached). None when unavailable."""
    path = os.path.join(BASELINE_RUN, "metrics.csv")
    key = (BASELINE_RUN, os.path.getmtime(path) if os.path.exists(path) else None)
    if _status_cache["baseline_key"] == key:
        return _status_cache["baseline"]
    best = None
    if key[1] is not None and os.path.abspath(BASELINE_RUN) != os.path.abspath(RUN_DIR):
        try:
            with open(path, "r", encoding="utf-8", newline="") as f:
                cers = [float(r["cer"]) for r in csv.DictReader(f) if r.get("cer")]
            if cers:
                best = {"name": os.path.basename(BASELINE_RUN.rstrip("/\\")),
                        "cer": min(cers)}
        except Exception:
            best = None
    _status_cache["baseline_key"] = key
    _status_cache["baseline"] = best
    return best


def _get_status():
    from datetime import datetime
    from statistics import median
    rows = _read_metric_rows()
    log_path = os.path.join(RUN_DIR, "train.log")
    head = ""
    if os.path.exists(log_path):
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as f:
                head = f.read(20000)
        except OSError:
            pass

    total = TOTAL_EPOCHS
    if total is None:
        m = re.search(r"Epochs: (\d+)", head)
        total = int(m.group(1)) if m else None
    dev = None
    m = re.search(r"Device: \S+ (.+)", head)
    if m:
        dev = m.group(1).strip()

    out = {"run_dir": RUN_DIR.replace("\\", "/"), "device": dev,
           "epoch": None, "total_epochs": total, "state": "waiting",
           "avg_epoch_s": None, "eta_s": None, "best_cer": None,
           "best_epoch": None, "baseline": _baseline_best(), "gpu": _gpu_stats()}
    if not rows:
        return out

    last = rows[-1]
    best = min(rows, key=lambda r: r["cer"])
    out["epoch"] = last["epoch"]
    out["best_cer"] = best["cer"]
    out["best_epoch"] = best["epoch"]

    # epoch durations from consecutive timestamps (same-run rows only)
    durs = []
    for a, b in zip(rows[:-1], rows[1:]):
        try:
            d = (datetime.strptime(b["ts"], "%Y-%m-%d %H:%M:%S")
                 - datetime.strptime(a["ts"], "%Y-%m-%d %H:%M:%S")).total_seconds()
            if 20 <= d <= 3600 and b["epoch"] == a["epoch"] + 1:
                durs.append(d)
        except ValueError:
            pass
    med = median(durs[-8:]) if durs else None
    out["avg_epoch_s"] = med

    try:
        age = (datetime.now() - datetime.strptime(last["ts"], "%Y-%m-%d %H:%M:%S")).total_seconds()
    except ValueError:
        age = None
    fresh_limit = max(600.0, 3.0 * med) if med else 600.0

    if total is not None and last["epoch"] >= total:
        out["state"] = "done"
    elif age is not None and age <= fresh_limit:
        out["state"] = "training"
        if total is not None and med:
            out["eta_s"] = max((total - last["epoch"]) * med - (age or 0), 0)
    else:
        out["state"] = "stale"
    return out


# --- Char-level diff for GT vs prediction samples ------------------------------
import html as _htmlmod


def _diff_html(gt: str, pr: str):
    """Escaped HTML for both strings with sub/del/ins spans marked."""
    g_cls = ["ok"] * len(gt)
    p_cls = ["ok"] * len(pr)
    for op, i1, i2 in _Lev.editops(gt, pr):
        if op == "replace":
            g_cls[i1] = "sub"
            p_cls[i2] = "sub"
        elif op == "delete":
            g_cls[i1] = "del"
        else:
            p_cls[i2] = "ins"

    def render(s, cls):
        parts = []
        i = 0
        while i < len(s):
            j = i
            while j < len(s) and cls[j] == cls[i]:
                j += 1
            seg = _htmlmod.escape(s[i:j])
            parts.append(seg if cls[i] == "ok"
                         else f'<span class="d-{cls[i]}">{seg}</span>')
            i = j
        return "".join(parts)

    return render(gt, g_cls), render(pr, p_cls)


# --- Data-source mix (real KHATT vs synthetic) --------------------------------
# Parsed from the run's train.log ("Mixed training: N real + M synth, ...")
# plus the synth.csv composition. Cached by file mtimes.
_datamix_cache = {"log_mtime": None, "csv_mtime": None, "data": None}
_MIX_RE = re.compile(
    r"Mixed training: (\d+) real \+ (\d+) synth, P\(synth\)=([\d.]+), "
    r"(\d+) samples/epoch \((\d+) batches\)")


def _get_datamix():
    log_path = os.path.join(RUN_DIR, "train.log")
    log_mtime = os.path.getmtime(log_path) if os.path.exists(log_path) else None
    csv_mtime = os.path.getmtime(SYNTH_CSV) if os.path.exists(SYNTH_CSV) else None
    c = _datamix_cache
    if c["data"] is not None and c["log_mtime"] == log_mtime and c["csv_mtime"] == csv_mtime:
        return c["data"]

    data = {"mode": "real", "real": None, "synth": 0, "p_synth": 0.0,
            "samples_per_epoch": None, "batches": None, "warm_start": None,
            "kinds": {}, "families": {}}
    if log_mtime is not None:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            head = f.read(20000)  # config lines are at the top
        ws = re.search(r"Warm-started from (\S+)", head)
        if ws:
            data["warm_start"] = ws.group(1)
        m = _MIX_RE.search(head)
        if m:
            data.update(mode="mixed", real=int(m.group(1)), synth=int(m.group(2)),
                        p_synth=float(m.group(3)), samples_per_epoch=int(m.group(4)),
                        batches=int(m.group(5)))
    if data["mode"] == "mixed" and csv_mtime is not None:
        kinds, fams = defaultdict(int), defaultdict(int)
        try:
            with open(SYNTH_CSV, "r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    kinds[row.get("kind") or "?"] += 1
                    fams[row.get("family") or "?"] += 1
            data["kinds"] = dict(sorted(kinds.items(), key=lambda kv: -kv[1]))
            data["families"] = dict(sorted(fams.items(), key=lambda kv: -kv[1]))
        except OSError:
            pass
    c.update(log_mtime=log_mtime, csv_mtime=csv_mtime, data=data)
    return data


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Arabic OCR Training Monitor</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0d1117; color: #e6edf3; padding: 20px; }
  .hdr { max-width: 1200px; margin: 0 auto 14px; display: flex; align-items: flex-start;
         justify-content: space-between; gap: 16px; flex-wrap: wrap; }
  h1 { font-size: 1.4em; margin-bottom: 2px; }
  .subtitle { color: #8b949e; font-size: 0.85em; }
  .hdr-right { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .pill { display: inline-flex; align-items: center; gap: 7px; font-size: 0.72em;
          font-weight: 700; letter-spacing: 0.06em; border-radius: 999px;
          padding: 4px 12px; border: 1px solid #30363d; color: #8b949e; }
  .pill::before { content: ''; width: 8px; height: 8px; border-radius: 50%;
                  background: currentColor; }
  .pill.training { color: #3fb950; border-color: #238636; background: #23863622; }
  .pill.training::before { animation: pulse 1.6s ease-in-out infinite; }
  .pill.stale { color: #d29922; border-color: #9e6a03; background: #9e6a0322; }
  .pill.done { color: #58a6ff; border-color: #1f6feb; background: #1f6feb22; }
  .pill.waiting { color: #8b949e; }
  @keyframes pulse { 50% { opacity: 0.25; } }
  @media (prefers-reduced-motion: reduce) { .pill.training::before { animation: none; } }
  .gpu-chips { display: flex; gap: 6px; flex-wrap: wrap; }

  .progress-wrap { max-width: 1200px; margin: 0 auto 18px; }
  .progress-bar { height: 8px; background: #21262d; border-radius: 4px; overflow: hidden; }
  .progress-fill { height: 100%; width: 0; background: linear-gradient(90deg, #1f6feb, #58a6ff);
                   border-radius: 4px; transition: width 0.6s ease; }
  .progress-text { color: #8b949e; font-size: 0.8em; margin-top: 5px;
                   font-variant-numeric: tabular-nums; }

  .stat-card .sub { color: #8b949e; font-size: 0.7em; margin-top: 3px;
                    font-variant-numeric: tabular-nums; }
  .stat-card .value.good { color: #3fb950; }
  .stat-card .value.bad { color: #f85149; }

  /* char-level diff highlighting */
  .d-sub { background: rgba(248, 81, 73, 0.35); border-radius: 2px; }
  .d-del { background: rgba(210, 153, 34, 0.50); border-radius: 2px; }
  .d-ins { background: rgba(88, 166, 255, 0.38); border-radius: 2px; }
  .diff-legend { direction: ltr; color: #8b949e; font-size: 0.75em; margin-bottom: 10px; }
  .diff-legend .sw { display: inline-block; width: 11px; height: 11px; border-radius: 2px;
                     vertical-align: -1px; margin: 0 4px 0 10px; }
  .cer-badge { direction: ltr; float: left; font-size: 0.7em; font-weight: 700;
               color: #8b949e; background: #21262d; border: 1px solid #30363d;
               border-radius: 4px; padding: 1px 7px; font-variant-numeric: tabular-nums; }
  .sort-btns { display: inline-flex; gap: 6px; margin-left: 12px; vertical-align: middle; }
  .sort-btns button { background: #21262d; color: #8b949e; border: 1px solid #30363d;
                      border-radius: 5px; padding: 2px 10px; font-size: 0.72em; cursor: pointer; }
  .sort-btns button.active { color: #e6edf3; border-color: #58a6ff; }
  .ops-chips { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 12px; }

  @media (max-width: 820px) {
    .charts { grid-template-columns: 1fr; }
    .analysis-grid { grid-template-columns: 1fr; }
    .hdr { flex-direction: column; }
  }
  .stats { display: flex; gap: 12px; justify-content: center; flex-wrap: wrap; margin-bottom: 20px; }
  .stat-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
               padding: 14px 22px; text-align: center; min-width: 130px; }
  .stat-card .label { color: #8b949e; font-size: 0.75em; text-transform: uppercase; }
  .stat-card .value { font-size: 1.6em; font-weight: 700; margin-top: 4px; }
  .stat-card .value.cer { color: #58a6ff; }
  .stat-card .value.wer { color: #f78166; }
  .stat-card .value.dot { color: #d2a8ff; }
  .stat-card .value.loss { color: #7ee787; }
  .stat-card .value.epoch { color: #e6edf3; }
  .stat-card .value.lr { color: #8b949e; }
  /* Training data sources */
  .datamix-section { max-width: 1200px; margin: 0 auto 20px; }
  .datamix-box { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                 padding: 14px 18px; }
  .datamix-box h3 { color: #8b949e; font-size: 0.95em; margin-bottom: 10px; }
  .mix-bar { display: flex; height: 26px; border-radius: 6px; overflow: hidden;
             font-size: 0.75em; font-weight: 700; }
  .mix-bar .real-seg { background: #238636; color: #fff; display: flex;
                       align-items: center; justify-content: center; white-space: nowrap; }
  .mix-bar .synth-seg { background: #9e6a03; color: #fff; display: flex;
                        align-items: center; justify-content: center; white-space: nowrap; }
  .mix-note { color: #8b949e; font-size: 0.8em; margin-top: 8px; line-height: 1.5; }
  .mix-note b { color: #e6edf3; }
  .mix-chips { margin-top: 10px; display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
  .chip { background: #21262d; border: 1px solid #30363d; color: #e6edf3;
          border-radius: 12px; padding: 2px 10px; font-size: 0.75em; }
  .chip b { color: #d2a8ff; font-variant-numeric: tabular-nums; }
  .chip-label { color: #8b949e; font-size: 0.75em; margin-right: 2px; }

  .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; max-width: 1200px; margin: 0 auto; }
  .chart-box { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
  .chart-box h3 { margin-bottom: 10px; font-size: 0.95em; color: #8b949e; }
  canvas { width: 100% !important; }

  /* Samples section */
  .samples-section { max-width: 1200px; margin: 24px auto 0; }
  .samples-section h2 { font-size: 1.1em; color: #8b949e; margin-bottom: 4px; }
  .samples-epoch { color: #58a6ff; font-size: 0.85em; margin-bottom: 12px; }
  .samples-list { display: flex; flex-direction: column; gap: 8px; }
  .sample-row { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                padding: 12px 16px; direction: rtl; text-align: right; }
  .sample-row .gt-line { color: #7ee787; font-size: 0.95em; margin-bottom: 6px;
                          font-family: 'Segoe UI', 'Geeza Pro', 'Arabic Typesetting', sans-serif; }
  .sample-row .pr-line { color: #f78166; font-size: 0.95em;
                          font-family: 'Segoe UI', 'Geeza Pro', 'Arabic Typesetting', sans-serif; }
  .sample-row .tag { direction: ltr; text-align: left; display: inline-block;
                      font-size: 0.7em; font-weight: 700; border-radius: 4px;
                      padding: 1px 6px; margin-left: 8px; vertical-align: middle; }
  .sample-row .tag.gt { background: #23352a; color: #7ee787; }
  .sample-row .tag.pr { background: #352320; color: #f78166; }
  .sample-row.match { border-color: #238636; }
  .sample-row.match .pr-line { color: #7ee787; }
  .sample-hidden { display: none; }
  .show-more-btn { display: block; margin: 12px auto; padding: 8px 24px;
                   background: #21262d; color: #8b949e; border: 1px solid #30363d;
                   border-radius: 6px; cursor: pointer; font-size: 0.9em; }
  .show-more-btn:hover { background: #30363d; color: #e6edf3; }
  .no-samples { color: #8b949e; text-align: center; padding: 20px; font-style: italic; }

  /* Analysis panels */
  .analysis-section { max-width: 1200px; margin: 28px auto 0; }
  .analysis-section h2 { font-size: 1.1em; color: #8b949e; margin-bottom: 4px; }
  .analysis-epoch { color: #58a6ff; font-size: 0.85em; margin-bottom: 12px; }
  .analysis-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .analysis-grid .chart-box { min-height: 360px; }
  .char-err-table-wrap { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                          padding: 16px; max-height: 420px; overflow-y: auto; }
  .char-err-detail { width: 100%; border-collapse: collapse; font-size: 0.85em; }
  .char-err-detail th { text-align: left; color: #8b949e; padding: 4px 8px;
                         border-bottom: 1px solid #30363d; position: sticky; top: 0;
                         background: #161b22; }
  .char-err-detail td { padding: 4px 8px; border-bottom: 1px solid #21262d; }
  .char-err-detail .ar { font-family: 'Geeza Pro', 'Arabic Typesetting', 'Segoe UI', sans-serif;
                          font-size: 1.15em; direction: rtl; }
  .char-err-detail .bar-cell { width: 90px; }
  .char-err-detail .mini-bar { height: 14px; border-radius: 3px; }
  .no-analysis { color: #8b949e; text-align: center; padding: 30px; font-style: italic; }

  /* Confusion matrices */
  .confusion-grid { display: flex; gap: 16px; flex-wrap: wrap; justify-content: center; }
  .confusion-box { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                    padding: 16px; min-width: 200px; flex: 1; }
  .confusion-box h3 { color: #8b949e; font-size: 0.9em; margin-bottom: 10px; text-align: center; }
  .confusion-matrix { border-collapse: collapse; margin: 0 auto; }
  .confusion-matrix th { color: #8b949e; padding: 6px 10px; font-size: 0.8em; }
  .confusion-matrix th.ar { font-family: 'Geeza Pro', 'Arabic Typesetting', 'Segoe UI', sans-serif;
                              font-size: 1.3em; color: #e6edf3; }
  .confusion-matrix td { text-align: center; padding: 8px 10px; border-radius: 4px;
                          font-size: 0.95em; font-weight: 600; min-width: 60px; }
  .confusion-matrix .pct { display: block; font-size: 0.7em; font-weight: 400; color: #8b949e; margin-top: 2px; }

  .status { text-align: center; margin-top: 16px; color: #8b949e; font-size: 0.8em; }
  .status .live { color: #3fb950; }
  .pull-btn { background: #21262d; color: #e6edf3; border: 1px solid #30363d;
              border-radius: 5px; padding: 3px 10px; font-size: 0.9em; cursor: pointer;
              font-family: inherit; }
  .pull-btn:hover { background: #30363d; }
  .pull-btn:disabled { opacity: 0.6; cursor: wait; }
  #pull-result { margin-left: 8px; font-family: ui-monospace, monospace; }
  #pull-result.ok { color: #3fb950; }
  #pull-result.err { color: #f85149; }
  @media (max-width: 700px) { .charts { grid-template-columns: 1fr; }
    .analysis-grid { grid-template-columns: 1fr; }
    .confusion-grid { flex-direction: column; } }
</style>
</head>
<body>
<header class="hdr">
  <div>
    <h1>Arabic OCR Training Monitor</h1>
    <p class="subtitle" id="subtitle">&mdash;</p>
  </div>
  <div class="hdr-right">
    <div class="gpu-chips" id="gpu-chips"></div>
    <span class="pill waiting" id="state-pill">WAITING</span>
  </div>
</header>

<div class="progress-wrap" id="progress-wrap" style="display:none">
  <div class="progress-bar"><div class="progress-fill" id="progress-fill"></div></div>
  <p class="progress-text" id="progress-text"></p>
</div>

<div class="stats">
  <div class="stat-card"><div class="label">Epoch</div><div class="value epoch" id="s-epoch">-</div></div>
  <div class="stat-card"><div class="label">Best CER</div><div class="value cer" id="s-cer">-</div>
    <div class="sub" id="s-cer-sub"></div></div>
  <div class="stat-card" id="delta-card" style="display:none"><div class="label">&Delta; vs baseline</div>
    <div class="value" id="s-delta">-</div><div class="sub" id="s-delta-sub"></div></div>
  <div class="stat-card"><div class="label">WER</div><div class="value wer" id="s-wer">-</div></div>
  <div class="stat-card"><div class="label">DotCER</div><div class="value dot" id="s-dot">-</div></div>
  <div class="stat-card"><div class="label">Train Loss</div><div class="value loss" id="s-loss">-</div></div>
  <div class="stat-card"><div class="label">LR</div><div class="value lr" id="s-lr">-</div></div>
</div>

<div class="datamix-section">
  <div class="datamix-box">
    <h3>Training Data Sources</h3>
    <div class="mix-bar" id="mix-bar"><div class="real-seg" style="width:100%">loading&hellip;</div></div>
    <p class="mix-note" id="mix-note"></p>
    <div class="mix-chips" id="mix-chips"></div>
  </div>
</div>

<div class="charts">
  <div class="chart-box"><h3>CER &amp; DotCER</h3><canvas id="chart-cer"></canvas></div>
  <div class="chart-box"><h3>WER &amp; WER(norm)</h3><canvas id="chart-wer"></canvas></div>
  <div class="chart-box"><h3>Train Loss</h3><canvas id="chart-loss"></canvas></div>
  <div class="chart-box"><h3>Learning Rate</h3><canvas id="chart-lr"></canvas></div>
</div>

<div class="samples-section">
  <h2>GT vs Prediction Samples
    <span class="sort-btns" id="sort-btns">
      <button data-sort="worst" class="active" onclick="setSort('worst')">worst first</button>
      <button data-sort="best" onclick="setSort('best')">best first</button>
      <button data-sort="order" onclick="setSort('order')">val order</button>
    </span>
  </h2>
  <p class="samples-epoch" id="samples-epoch"></p>
  <p class="diff-legend">diff:
    <span class="sw" style="background:rgba(248,81,73,0.55)"></span>substituted
    <span class="sw" style="background:rgba(210,153,34,0.65)"></span>missing in prediction (on GT)
    <span class="sw" style="background:rgba(88,166,255,0.55)"></span>inserted by model (on PR)
  </p>
  <div class="samples-list" id="samples-list">
    <p class="no-samples">No samples yet — predictions appear after epoch 1</p>
  </div>
  <button class="show-more-btn" id="show-more-btn" style="display:none;" onclick="showMore()">Show more samples</button>
</div>

<div class="analysis-section">
  <h2>Per-Character Error Breakdown</h2>
  <p class="analysis-epoch" id="char-err-epoch"></p>
  <div class="ops-chips" id="ops-chips"></div>
  <div class="analysis-grid">
    <div class="chart-box"><h3>Top Characters by Error Rate</h3><canvas id="chart-char-errors"></canvas></div>
    <div class="char-err-table-wrap" id="char-err-table">
      <p class="no-analysis">No analysis yet &mdash; appears after epoch 1</p>
    </div>
  </div>
</div>

<div class="analysis-section">
  <h2>Dot-Group Confusion Matrices</h2>
  <p class="analysis-epoch" id="confusion-epoch"></p>
  <div class="confusion-grid" id="confusion-grid">
    <p class="no-analysis">No analysis yet &mdash; appears after epoch 1</p>
  </div>
</div>

<p class="status">
  Auto-refresh every <strong>30s</strong> &middot;
  <span class="live" id="status-text">waiting...</span> &middot;
  <button id="pull-btn" class="pull-btn" onclick="doPull()">git pull</button>
  <span id="pull-result"></span>
</p>

<script>
const INITIAL_SHOW = 10;
let allSamples = [];
let visibleCount = INITIAL_SHOW;
let sortMode = 'worst';
let gBaseline = null;

function setSort(mode) {
  sortMode = mode;
  visibleCount = INITIAL_SHOW;
  document.querySelectorAll('#sort-btns button').forEach(b =>
    b.classList.toggle('active', b.dataset.sort === mode));
  renderSamples();
}

function fmtDur(s) {
  s = Math.round(s);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  if (h) return h + 'h ' + String(m).padStart(2, '0') + 'm';
  if (m) return m + 'm ' + String(s % 60).padStart(2, '0') + 's';
  return s + 's';
}

async function loadStatus() {
  try {
    const r = await fetch('/api/status?_=' + Date.now());
    const st = await r.json();
    gBaseline = st.baseline;

    const map = { training: ['TRAINING', 'training'], stale: ['STALLED?', 'stale'],
                  done: ['COMPLETE', 'done'], waiting: ['WAITING', 'waiting'] };
    const [txt, cls] = map[st.state] || map.waiting;
    const pill = document.getElementById('state-pill');
    pill.textContent = txt;
    pill.className = 'pill ' + cls;

    document.getElementById('subtitle').textContent =
      st.run_dir + (st.device ? ' · ' + st.device : '') + ' · CRNN-CTC v2';

    if (st.epoch != null && st.total_epochs) {
      document.getElementById('progress-wrap').style.display = '';
      const pct = Math.min(100 * st.epoch / st.total_epochs, 100);
      document.getElementById('progress-fill').style.width = pct + '%';
      let t = 'epoch ' + st.epoch + ' / ' + st.total_epochs;
      if (st.avg_epoch_s) t += ' · ' + fmtDur(st.avg_epoch_s) + '/epoch';
      if (st.state === 'training' && st.eta_s != null) {
        const fin = new Date(Date.now() + st.eta_s * 1000)
          .toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        t += ' · ETA ' + fmtDur(st.eta_s) + ' (~' + fin + ')';
      }
      if (st.state === 'done') t += ' · finished';
      document.getElementById('progress-text').textContent = t;
    }

    const g = st.gpu;
    document.getElementById('gpu-chips').innerHTML = !g ? '' :
      '<span class="chip">GPU <b>' + g.util + '%</b></span>' +
      '<span class="chip">VRAM <b>' + (g.mem_used / 1024).toFixed(1) + '/' +
        Math.round(g.mem_total / 1024) + 'G</b></span>' +
      '<span class="chip"><b>' + g.temp + '°C</b></span>' +
      '<span class="chip"><b>' + g.power + 'W</b></span>';
  } catch (e) { /* keep last state */ }
}

const chartOpts = (yLabel) => ({
  responsive: true,
  animation: { duration: 300 },
  scales: {
    x: { title: { display: true, text: 'Epoch', color: '#8b949e' },
         ticks: { color: '#8b949e' }, grid: { color: '#21262d' } },
    y: { title: { display: true, text: yLabel, color: '#8b949e' },
         ticks: { color: '#8b949e' }, grid: { color: '#21262d' },
         beginAtZero: true }
  },
  plugins: { legend: { labels: { color: '#e6edf3' } } }
});

const cerChart = new Chart(document.getElementById('chart-cer'), {
  type: 'line', data: { labels: [], datasets: [
    { label: 'CER', data: [], borderColor: '#58a6ff', borderWidth: 2, pointRadius: [],
      pointBackgroundColor: '#58a6ff', fill: false },
    { label: 'DotCER', data: [], borderColor: '#d2a8ff', borderWidth: 2, pointRadius: 1, fill: false },
    { label: 'baseline', data: [], borderColor: '#8b949e', borderWidth: 1.5,
      borderDash: [6, 4], pointRadius: 0, fill: false, hidden: true },
  ]}, options: chartOpts('Error Rate')
});
const werChart = new Chart(document.getElementById('chart-wer'), {
  type: 'line', data: { labels: [], datasets: [
    { label: 'WER', data: [], borderColor: '#f78166', borderWidth: 2, pointRadius: 1, fill: false },
    { label: 'WER(norm)', data: [], borderColor: '#ffa657', borderWidth: 2, pointRadius: 1, fill: false },
  ]}, options: chartOpts('Error Rate')
});
const lossOpts = chartOpts('Loss (log)');
lossOpts.scales.y.type = 'logarithmic';
lossOpts.scales.y.beginAtZero = false;
lossOpts.plugins.legend.display = false;   // single series — the title names it
const lossChart = new Chart(document.getElementById('chart-loss'), {
  type: 'line', data: { labels: [], datasets: [
    { label: 'Train Loss', data: [], borderColor: '#7ee787', borderWidth: 2, pointRadius: 1, fill: false },
  ]}, options: lossOpts
});
const lrOpts = chartOpts('LR');
lrOpts.plugins.legend.display = false;     // single series — the title names it
const lrChart = new Chart(document.getElementById('chart-lr'), {
  type: 'line', data: { labels: [], datasets: [
    { label: 'Learning Rate', data: [], borderColor: '#8b949e', borderWidth: 2, pointRadius: 1, fill: false },
  ]}, options: lrOpts
});

const charErrChart = new Chart(document.getElementById('chart-char-errors'), {
  type: 'bar', data: { labels: [], datasets: [
    { label: 'Substitutions %', data: [], backgroundColor: '#f8514980', borderColor: '#f85149', borderWidth: 1 },
    { label: 'Deletions %', data: [], backgroundColor: '#d2992280', borderColor: '#d29922', borderWidth: 1 },
  ]}, options: {
    indexAxis: 'y', responsive: true,
    animation: { duration: 300 },
    scales: {
      x: { stacked: true, title: { display: true, text: 'Error %', color: '#8b949e' },
           ticks: { color: '#8b949e' }, grid: { color: '#21262d' }, beginAtZero: true },
      y: { stacked: true, ticks: { color: '#e6edf3', font: { size: 13, family: "'Geeza Pro','Arabic Typesetting',sans-serif" } },
           grid: { display: false } }
    },
    plugins: { legend: { labels: { color: '#e6edf3' } } }
  }
});

function renderSamples() {
  const container = document.getElementById('samples-list');
  const btn = document.getElementById('show-more-btn');

  if (!allSamples.length) {
    container.innerHTML = '<p class="no-samples">No samples yet \u2014 predictions appear after epoch 1</p>';
    btn.style.display = 'none';
    return;
  }

  const order = allSamples.map((s, i) => i);
  if (sortMode === 'worst') order.sort((a, b) => allSamples[b].cer - allSamples[a].cer);
  else if (sortMode === 'best') order.sort((a, b) => allSamples[a].cer - allSamples[b].cer);

  let html = '';
  for (let k = 0; k < order.length; k++) {
    const s = allSamples[order[k]];
    const isMatch = s.gt.trim() === s.pr.trim();
    const hidden = k >= visibleCount ? ' sample-hidden' : '';
    const matchCls = isMatch ? ' match' : '';
    const badge = '<span class="cer-badge">CER ' + (s.cer * 100).toFixed(1) + '%</span>';
    const gtHtml = s.gt_html || escapeHtml(s.gt);
    const prHtml = s.pr_html || escapeHtml(s.pr);
    html += '<div class="sample-row' + matchCls + hidden + '">'
         + badge
         + '<div class="gt-line"><span class="tag gt">GT</span> ' + gtHtml + '</div>'
         + '<div class="pr-line"><span class="tag pr">PR</span> ' + (prHtml || '<em style="color:#484f58">(empty)</em>') + '</div>'
         + '</div>';
  }
  container.innerHTML = html;

  if (allSamples.length > visibleCount) {
    btn.style.display = 'block';
    btn.textContent = 'Show more (' + (allSamples.length - visibleCount) + ' remaining)';
  } else {
    btn.style.display = 'none';
  }
}

function showMore() {
  visibleCount = Math.min(visibleCount + 20, allSamples.length);
  const rows = document.querySelectorAll('.sample-row');
  rows.forEach((row, i) => {
    if (i < visibleCount) row.classList.remove('sample-hidden');
  });
  const btn = document.getElementById('show-more-btn');
  if (visibleCount >= allSamples.length) {
    btn.style.display = 'none';
  } else {
    btn.textContent = 'Show more (' + (allSamples.length - visibleCount) + ' remaining)';
  }
}

function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function renderCharErrors(chars) {
  const sorted = chars.filter(c => c.total >= 5)
    .sort((a, b) => b.error_rate - a.error_rate).slice(0, 30);

  charErrChart.data.labels = sorted.map(c => c.char === ' ' ? '(space)' : c.char);
  charErrChart.data.datasets[0].data = sorted.map(c => {
    const subPct = (c.errors - c.deletions) / c.total * 100;
    return Math.round(subPct * 10) / 10;
  });
  charErrChart.data.datasets[1].data = sorted.map(c => {
    return Math.round(c.deletions / c.total * 1000) / 10;
  });
  charErrChart.update();

  let html = '<table class="char-err-detail">';
  html += '<tr><th>Char</th><th>Total</th><th>Err</th><th>Rate</th><th>Top Confusions</th></tr>';
  for (const c of sorted) {
    const subs = c.top_subs.map(s => '<span class="ar">' + escapeHtml(s[0]) + '</span>(' + s[1] + ')').join(' ');
    const del_note = c.deletions > 0 ? ' <span style="color:#d29922">DEL(' + c.deletions + ')</span>' : '';
    const rateColor = c.error_rate > 0.3 ? '#f85149' : c.error_rate > 0.15 ? '#d29922' : '#7ee787';
    html += '<tr><td class="ar">' + (c.char === ' ' ? '(space)' : escapeHtml(c.char)) + '</td>';
    html += '<td>' + c.total + '</td>';
    html += '<td>' + c.errors + '</td>';
    html += '<td style="color:' + rateColor + '">' + (c.error_rate * 100).toFixed(1) + '%</td>';
    html += '<td>' + subs + del_note + '</td></tr>';
  }
  html += '</table>';
  document.getElementById('char-err-table').innerHTML = html;
}

function renderConfusion(groups) {
  let html = '';
  for (const g of groups) {
    html += '<div class="confusion-box">';
    html += '<h3>' + g.name + '</h3>';
    html += '<table class="confusion-matrix">';
    html += '<tr><th>GT \\ PR</th>';
    for (const ch of g.chars) html += '<th class="ar">' + ch + '</th>';
    html += '<th>DEL</th></tr>';

    for (let i = 0; i < g.chars.length; i++) {
      html += '<tr><th class="ar">' + g.chars[i] + '</th>';
      const rowTotal = g.totals[i] || 1;
      for (let j = 0; j < g.chars.length; j++) {
        const val = g.matrix[i][j];
        const pct = val / rowTotal;
        let bg;
        if (i === j) {
          bg = 'rgba(63,185,80,' + (Math.min(pct, 1) * 0.6 + 0.05) + ')';
        } else {
          bg = pct > 0 ? 'rgba(248,81,73,' + (Math.min(pct * 4, 1) * 0.6 + 0.05) + ')' : 'transparent';
        }
        html += '<td style="background:' + bg + '">' + val;
        html += '<span class="pct">' + (pct * 100).toFixed(1) + '%</span></td>';
      }
      const delVal = g.deletions[i];
      const delPct = delVal / rowTotal;
      const delBg = delVal > 0 ? 'rgba(210,168,34,' + (Math.min(delPct * 4, 1) * 0.6 + 0.05) + ')' : 'transparent';
      html += '<td style="background:' + delBg + '">' + delVal;
      html += '<span class="pct">' + (delPct * 100).toFixed(1) + '%</span></td>';
      html += '</tr>';
    }
    html += '</table></div>';
  }
  document.getElementById('confusion-grid').innerHTML = html;
}

async function refresh() {
  try {
    // Fetch metrics
    const r = await fetch('/api/metrics?_=' + Date.now());
    const rows = await r.json();
    if (!rows.length) { document.getElementById('status-text').textContent = 'no data yet'; return; }

    const epochs = rows.map(r => r.epoch);
    const last = rows[rows.length - 1];
    let bestRow = rows[0];
    for (const r of rows) if (r.cer < bestRow.cer) bestRow = r;

    document.getElementById('s-epoch').textContent = last.epoch;
    document.getElementById('s-cer').textContent = (bestRow.cer * 100).toFixed(2) + '%';
    document.getElementById('s-cer-sub').textContent = '@ epoch ' + bestRow.epoch;
    document.getElementById('s-wer').textContent = (last.wer * 100).toFixed(1) + '%';
    document.getElementById('s-dot').textContent = (last.dot_cer * 100).toFixed(2) + '%';
    document.getElementById('s-loss').textContent = last.train_loss.toFixed(3);
    document.getElementById('s-lr').textContent = last.lr.toExponential(2);

    // Δ vs baseline run (negative = better than baseline)
    if (gBaseline) {
      const d = (bestRow.cer - gBaseline.cer) * 100;
      const el = document.getElementById('s-delta');
      el.textContent = (d > 0 ? '+' : '') + d.toFixed(2) + 'pp';
      el.className = 'value ' + (d < 0 ? 'good' : d > 0 ? 'bad' : 'epoch');
      document.getElementById('s-delta-sub').textContent =
        gBaseline.name + ' best ' + (gBaseline.cer * 100).toFixed(2) + '%';
      document.getElementById('delta-card').style.display = '';
    }

    cerChart.data.labels = epochs;
    cerChart.data.datasets[0].data = rows.map(r => r.cer);
    // dots mark epochs where a checkpoint was saved
    cerChart.data.datasets[0].pointRadius = rows.map(r => r.ckpt_saved ? 3 : 0);
    cerChart.data.datasets[1].data = rows.map(r => r.dot_cer);
    if (gBaseline) {
      cerChart.data.datasets[2].data = epochs.map(() => gBaseline.cer);
      cerChart.data.datasets[2].label = gBaseline.name + ' best';
      cerChart.data.datasets[2].hidden = false;
    }
    cerChart.update();

    werChart.data.labels = epochs;
    werChart.data.datasets[0].data = rows.map(r => r.wer);
    werChart.data.datasets[1].data = rows.map(r => r.wer_norm);
    werChart.update();

    lossChart.data.labels = epochs;
    lossChart.data.datasets[0].data = rows.map(r => r.train_loss);
    lossChart.update();

    lrChart.data.labels = epochs;
    lrChart.data.datasets[0].data = rows.map(r => r.lr);
    lrChart.update();

    // Fetch samples
    const sr = await fetch('/api/samples?_=' + Date.now());
    const sdata = await sr.json();
    if (sdata.epoch) {
      document.getElementById('samples-epoch').textContent = 'Latest epoch: ' + sdata.epoch + ' (' + sdata.total + ' samples)';
      allSamples = sdata.samples;
      visibleCount = INITIAL_SHOW;
      renderSamples();
    }

    // Fetch per-character error analysis
    const ceResp = await fetch('/api/char_errors?_=' + Date.now());
    const ceData = await ceResp.json();
    if (ceData.epoch && ceData.chars.length) {
      document.getElementById('char-err-epoch').textContent = 'Epoch ' + ceData.epoch + ' \u2014 from ' + ceData.chars.reduce((s,c) => s + c.total, 0).toLocaleString() + ' characters';
      renderCharErrors(ceData.chars);
      const o = ceData.ops || {};
      const tot = (o.sub || 0) + (o.del || 0) + (o.ins || 0);
      document.getElementById('ops-chips').innerHTML = !tot ? '' :
        '<span class="chip">edit ops <b>' + tot.toLocaleString() + '</b></span>' +
        '<span class="chip">substitutions <b>' + Math.round(100 * o.sub / tot) + '%</b></span>' +
        '<span class="chip">deletions <b>' + Math.round(100 * o.del / tot) + '%</b></span>' +
        '<span class="chip">insertions <b>' + Math.round(100 * o.ins / tot) + '%</b></span>' +
        '<span class="chip">space-involved <b>' + Math.round(100 * o.space / tot) + '%</b> of all ops</span>';
    }

    // Fetch dot-group confusion matrices
    const cmResp = await fetch('/api/confusion?_=' + Date.now());
    const cmData = await cmResp.json();
    if (cmData.epoch && cmData.groups.length) {
      document.getElementById('confusion-epoch').textContent = 'Epoch ' + cmData.epoch;
      renderConfusion(cmData.groups);
    }

    const now = new Date().toLocaleTimeString();
    document.getElementById('status-text').textContent = 'updated ' + now + ' \u2014 epoch ' + last.epoch;
  } catch(e) {
    document.getElementById('status-text').textContent = 'error: ' + e.message;
  }
}

async function loadDatamix() {
  try {
    const r = await fetch('/api/datamix?_=' + Date.now());
    const d = await r.json();
    const bar = document.getElementById('mix-bar');
    const note = document.getElementById('mix-note');
    const chips = document.getElementById('mix-chips');
    if (d.mode === 'mixed') {
      const rp = Math.round((1 - d.p_synth) * 100), sp = 100 - rp;
      bar.innerHTML =
        '<div class="real-seg" style="width:' + rp + '%">Real KHATT \u00b7 ' + rp + '%</div>' +
        '<div class="synth-seg" style="width:' + sp + '%">Synthetic \u00b7 ' + sp + '%</div>';
      note.innerHTML = 'Each batch draws <b>~' + rp + '% real KHATT</b> handwriting (' +
        d.real.toLocaleString() + ' scanned lines) and <b>~' + sp + '% synthetic</b> rendered lines (' +
        d.synth.toLocaleString() + ' generated) \u2014 ' +
        (d.samples_per_epoch ? d.samples_per_epoch.toLocaleString() + ' samples/epoch. ' : '') +
        'All validation metrics, samples and confusion analyses on this page are measured on ' +
        '<b>100% real KHATT</b> lines.' +
        (d.warm_start ? ' Weights warm-started from <b>' + d.warm_start + '</b>.' : '');
      let html = '';
      if (Object.keys(d.kinds).length) {
        html += '<span class="chip-label">synthetic text mix:</span>';
        for (const [k, v] of Object.entries(d.kinds))
          html += '<span class="chip">' + k + ' <b>' + v.toLocaleString() + '</b></span>';
      }
      if (Object.keys(d.families).length) {
        html += '<span class="chip-label" style="margin-left:8px">fonts:</span>';
        for (const [k, v] of Object.entries(d.families))
          html += '<span class="chip">' + k + ' <b>' + v.toLocaleString() + '</b></span>';
      }
      chips.innerHTML = html;
    } else {
      bar.innerHTML = '<div class="real-seg" style="width:100%">Real KHATT \u00b7 100%</div>';
      note.textContent = 'This run trains on real KHATT scanned lines only (no synthetic mixing).' +
        (d.warm_start ? ' Weights warm-started from ' + d.warm_start + '.' : '');
      chips.innerHTML = '';
    }
  } catch (e) { /* panel stays in loading state */ }
}

async function doPull() {
  const btn = document.getElementById('pull-btn');
  const out = document.getElementById('pull-result');
  btn.disabled = true;
  out.className = '';
  out.textContent = 'pulling...';
  try {
    const r = await fetch('/api/pull', { method: 'POST' });
    const data = await r.json();
    if (data.ok) {
      out.className = 'ok';
      const clean = (data.stdout || '').trim().split('\n').pop() || 'ok';
      out.textContent = clean;
      refresh();
    } else {
      out.className = 'err';
      out.textContent = (data.stderr || 'error').trim().split('\n')[0];
    }
  } catch (e) {
    out.className = 'err';
    out.textContent = 'fetch failed: ' + e.message;
  } finally {
    btn.disabled = false;
  }
}

loadDatamix();
loadStatus().then(refresh);
setInterval(() => loadStatus().then(refresh), 30000);
</script>
</body>
</html>"""


class MonitorHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api/metrics"):
            self._serve_metrics()
        elif self.path.startswith("/api/samples"):
            self._serve_samples()
        elif self.path.startswith("/api/char_errors"):
            self._serve_char_errors()
        elif self.path.startswith("/api/confusion"):
            self._serve_confusion()
        elif self.path.startswith("/api/datamix"):
            self._json_response(_get_datamix())
        elif self.path.startswith("/api/status"):
            self._json_response(_get_status())
        else:
            self._serve_html()

    def do_POST(self):
        if self.path.startswith("/api/pull"):
            self._serve_pull()
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_pull(self):
        try:
            proc = subprocess.run(
                ["git", "pull"], capture_output=True, text=True, timeout=30,
            )
            body = {
                "ok": proc.returncode == 0,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "returncode": proc.returncode,
            }
        except subprocess.TimeoutExpired:
            body = {"ok": False, "stdout": "", "stderr": "git pull timed out after 30s",
                    "returncode": -1}
        except Exception as e:
            body = {"ok": False, "stdout": "", "stderr": f"{type(e).__name__}: {e}",
                    "returncode": -1}
        self._json_response(body)

    def _serve_html(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode("utf-8"))

    def _serve_metrics(self):
        rows = []
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                # Strip whitespace from header names (Windows CSV artifacts)
                if reader.fieldnames:
                    reader.fieldnames = [n.strip() for n in reader.fieldnames]
                for row in reader:
                    # Normalize keys: strip whitespace from values too.
                    # Skip None keys (restkey from DictReader when a row has more fields than the header).
                    row = {k.strip(): v.strip() if isinstance(v, str) else v
                           for k, v in row.items() if isinstance(k, str)}
                    try:
                        rows.append({
                            "epoch": int(float(row.get("epoch", 0))),
                            "train_loss": float(row.get("train_loss", 0)),
                            "cer": float(row.get("cer", 1)),
                            "wer": float(row.get("wer", 1)),
                            "wer_norm": float(row.get("wer_norm", 1)),
                            "dot_cer": float(row.get("dot_cer", 1)),
                            "lr": float(row.get("lr", 0)),
                            "ckpt_saved": int(float(row.get("ckpt_saved", 0))),
                        })
                    except (ValueError, KeyError):
                        pass
        self._json_response(rows)

    def _serve_samples(self):
        path, epoch_num = _find_latest_tsv()
        if path is None:
            self._json_response({"epoch": None, "total": 0, "samples": []})
            return

        samples = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    gt, pr = row.get("label", ""), row.get("pred", "")
                    gt_html, pr_html = _diff_html(gt, pr)
                    cer = round(_Lev.distance(gt, pr) / max(len(gt), 1), 4)
                    samples.append({"gt": gt, "pr": pr, "cer": cer,
                                    "gt_html": gt_html, "pr_html": pr_html})
        except Exception:
            pass

        self._json_response({
            "epoch": epoch_num,
            "total": len(samples),
            "samples": samples,
        })

    def _serve_char_errors(self):
        epoch, cache = _refresh_analysis_cache()
        if epoch is None:
            self._json_response({"epoch": None, "chars": [], "ops": {}})
            return
        self._json_response({"epoch": epoch, "chars": cache["char_errors"],
                             "ops": cache.get("ops", {})})

    def _serve_confusion(self):
        epoch, cache = _refresh_analysis_cache()
        if epoch is None:
            self._json_response({"epoch": None, "groups": []})
            return
        self._json_response({"epoch": epoch, "groups": cache["confusion"]})

    def _json_response(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def log_message(self, format, *args):
        pass  # suppress request logs


def main():
    global METRICS_PATH, RUN_DIR, TOTAL_EPOCHS, BASELINE_RUN
    ap = argparse.ArgumentParser(description="Live training monitor")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--run-dir", default=RUN_DIR,
                    help=f"Run directory to monitor (default: {RUN_DIR})")
    ap.add_argument("--total-epochs", type=int, default=None,
                    help="Planned epochs for progress/ETA (else parsed from train.log).")
    ap.add_argument("--baseline-run", default=BASELINE_RUN,
                    help=f"Run dir whose best CER is shown as a reference line "
                         f"(default: {BASELINE_RUN}; ignored if same as --run-dir).")
    args = ap.parse_args()

    RUN_DIR = args.run_dir
    METRICS_PATH = os.path.join(RUN_DIR, "metrics.csv")
    TOTAL_EPOCHS = args.total_epochs
    BASELINE_RUN = args.baseline_run

    server = HTTPServer(("0.0.0.0", args.port), MonitorHandler)
    print(f"Monitor running at http://0.0.0.0:{args.port}")
    print(f"  Watching: {RUN_DIR}")
    print(f"Access from your Mac: http://<PC-IP>:{args.port}")
    print("Press Ctrl+C to stop\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nMonitor stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
