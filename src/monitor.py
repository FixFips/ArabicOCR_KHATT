# src/monitor.py
"""
Live training monitor — serves a web dashboard that reads metrics.csv + sample predictions.
Access from any device on the same network.

Usage:
    python -m src.monitor                  # default port 8080
    python -m src.monitor --port 9090      # custom port
"""

import os
import csv
import json
import glob
import argparse
import subprocess
from collections import defaultdict
from http.server import HTTPServer, BaseHTTPRequestHandler
from rapidfuzz.distance import Levenshtein as _Lev

METRICS_PATH = os.path.join("runs", "exp1", "metrics.csv")
RUN_DIR = os.path.join("runs", "exp1")

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
_analysis_cache = {"mtime": None, "path": None, "char_errors": [], "confusion": []}

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

    for gt, pr in samples:
        for ch in gt:
            char_total[ch] += 1
        ops = _Lev.editops(gt, pr)
        consumed = set()
        for op, i1, i2 in ops:
            if op == "replace":
                consumed.add(i1)
                char_subs[gt[i1]][pr[i2]] += 1
                if gt[i1] in _ALL_DOT_CHARS:
                    dot_total[gt[i1]] += 1
                    dot_confusion[gt[i1]][pr[i2]] += 1
            elif op == "delete":
                consumed.add(i1)
                char_dels[gt[i1]] += 1
                if gt[i1] in _ALL_DOT_CHARS:
                    dot_total[gt[i1]] += 1
                    dot_deleted[gt[i1]] += 1
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
    return epoch, _analysis_cache


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
  h1 { text-align: center; margin-bottom: 8px; font-size: 1.5em; }
  .subtitle { text-align: center; color: #8b949e; margin-bottom: 20px; font-size: 0.9em; }
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
<h1>Arabic OCR Training Monitor</h1>
<p class="subtitle">CRNN-CTC v2 &middot; KHATT Dataset &middot; RTX 5080</p>

<div class="stats">
  <div class="stat-card"><div class="label">Epoch</div><div class="value epoch" id="s-epoch">-</div></div>
  <div class="stat-card"><div class="label">Best CER</div><div class="value cer" id="s-cer">-</div></div>
  <div class="stat-card"><div class="label">WER</div><div class="value wer" id="s-wer">-</div></div>
  <div class="stat-card"><div class="label">DotCER</div><div class="value dot" id="s-dot">-</div></div>
  <div class="stat-card"><div class="label">Train Loss</div><div class="value loss" id="s-loss">-</div></div>
  <div class="stat-card"><div class="label">LR</div><div class="value lr" id="s-lr">-</div></div>
</div>

<div class="charts">
  <div class="chart-box"><h3>CER &amp; DotCER</h3><canvas id="chart-cer"></canvas></div>
  <div class="chart-box"><h3>WER &amp; WER(norm)</h3><canvas id="chart-wer"></canvas></div>
  <div class="chart-box"><h3>Train Loss</h3><canvas id="chart-loss"></canvas></div>
  <div class="chart-box"><h3>Learning Rate</h3><canvas id="chart-lr"></canvas></div>
</div>

<div class="samples-section">
  <h2>GT vs Prediction Samples</h2>
  <p class="samples-epoch" id="samples-epoch"></p>
  <div class="samples-list" id="samples-list">
    <p class="no-samples">No samples yet — predictions appear after epoch 1</p>
  </div>
  <button class="show-more-btn" id="show-more-btn" style="display:none;" onclick="showMore()">Show more samples</button>
</div>

<div class="analysis-section">
  <h2>Per-Character Error Breakdown</h2>
  <p class="analysis-epoch" id="char-err-epoch"></p>
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
    { label: 'CER', data: [], borderColor: '#58a6ff', borderWidth: 2, pointRadius: 1, fill: false },
    { label: 'DotCER', data: [], borderColor: '#d2a8ff', borderWidth: 2, pointRadius: 1, fill: false },
  ]}, options: chartOpts('Error Rate')
});
const werChart = new Chart(document.getElementById('chart-wer'), {
  type: 'line', data: { labels: [], datasets: [
    { label: 'WER', data: [], borderColor: '#f78166', borderWidth: 2, pointRadius: 1, fill: false },
    { label: 'WER(norm)', data: [], borderColor: '#ffa657', borderWidth: 2, pointRadius: 1, fill: false },
  ]}, options: chartOpts('Error Rate')
});
const lossChart = new Chart(document.getElementById('chart-loss'), {
  type: 'line', data: { labels: [], datasets: [
    { label: 'Train Loss', data: [], borderColor: '#7ee787', borderWidth: 2, pointRadius: 1, fill: false },
  ]}, options: chartOpts('Loss')
});
const lrChart = new Chart(document.getElementById('chart-lr'), {
  type: 'line', data: { labels: [], datasets: [
    { label: 'Learning Rate', data: [], borderColor: '#8b949e', borderWidth: 2, pointRadius: 1, fill: false },
  ]}, options: chartOpts('LR')
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

  let html = '';
  for (let i = 0; i < allSamples.length; i++) {
    const s = allSamples[i];
    const isMatch = s.gt.trim() === s.pr.trim();
    const hidden = i >= visibleCount ? ' sample-hidden' : '';
    const matchCls = isMatch ? ' match' : '';
    const num = String(i + 1).padStart(3, ' ');
    html += '<div class="sample-row' + matchCls + hidden + '" data-idx="' + i + '">'
         + '<div class="gt-line"><span class="tag gt">GT</span> ' + escapeHtml(s.gt) + '</div>'
         + '<div class="pr-line"><span class="tag pr">PR</span> ' + (s.pr || '<em style="color:#484f58">(empty)</em>') + '</div>'
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
    const bestCer = Math.min(...rows.map(r => r.cer));

    document.getElementById('s-epoch').textContent = last.epoch;
    document.getElementById('s-cer').textContent = bestCer.toFixed(4);
    document.getElementById('s-wer').textContent = last.wer.toFixed(4);
    document.getElementById('s-dot').textContent = last.dot_cer.toFixed(4);
    document.getElementById('s-loss').textContent = last.train_loss.toFixed(3);
    document.getElementById('s-lr').textContent = last.lr.toExponential(2);

    cerChart.data.labels = epochs;
    cerChart.data.datasets[0].data = rows.map(r => r.cer);
    cerChart.data.datasets[1].data = rows.map(r => r.dot_cer);
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
    }

    // Fetch dot-group confusion matrices
    const cmResp = await fetch('/api/confusion?_=' + Date.now());
    const cmData = await cmResp.json();
    if (cmData.epoch && cmData.groups.length) {
      document.getElementById('confusion-epoch').textContent = 'Epoch ' + cmData.epoch;
      renderConfusion(cmData.groups);
    }

    const now = new Date().toLocaleTimeString();
    document.getElementById('status-text').textContent = 'updated ' + now + ' \u2014 epoch ' + last.epoch + '/120';
  } catch(e) {
    document.getElementById('status-text').textContent = 'error: ' + e.message;
  }
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

refresh();
setInterval(refresh, 30000);
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
                    samples.append({"gt": row.get("label", ""), "pr": row.get("pred", "")})
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
            self._json_response({"epoch": None, "chars": []})
            return
        self._json_response({"epoch": epoch, "chars": cache["char_errors"]})

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
    ap = argparse.ArgumentParser(description="Live training monitor")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    server = HTTPServer(("0.0.0.0", args.port), MonitorHandler)
    print(f"Monitor running at http://0.0.0.0:{args.port}")
    print(f"Access from your Mac: http://<PC-IP>:{args.port}")
    print("Press Ctrl+C to stop\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nMonitor stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
