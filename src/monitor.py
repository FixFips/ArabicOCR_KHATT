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
from http.server import HTTPServer, BaseHTTPRequestHandler

METRICS_PATH = os.path.join("runs", "exp1", "metrics.csv")
RUN_DIR = os.path.join("runs", "exp1")

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

  .status { text-align: center; margin-top: 16px; color: #8b949e; font-size: 0.8em; }
  .status .live { color: #3fb950; }
  @media (max-width: 700px) { .charts { grid-template-columns: 1fr; } }
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

<p class="status">Auto-refresh every <strong>30s</strong> &middot; <span class="live" id="status-text">waiting...</span></p>

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

    const now = new Date().toLocaleTimeString();
    document.getElementById('status-text').textContent = 'updated ' + now + ' \u2014 epoch ' + last.epoch + '/120';
  } catch(e) {
    document.getElementById('status-text').textContent = 'error: ' + e.message;
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
        else:
            self._serve_html()

    def _serve_html(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode("utf-8"))

    def _serve_metrics(self):
        rows = []
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
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
        # Find the latest val_epoch_*_samples.tsv file
        pattern = os.path.join(RUN_DIR, "val_epoch_*_samples.tsv")
        files = sorted(glob.glob(pattern))
        if not files:
            self._json_response({"epoch": None, "total": 0, "samples": []})
            return

        latest = files[-1]
        # Extract epoch number from filename like val_epoch_003_samples.tsv
        basename = os.path.basename(latest)
        epoch_str = basename.replace("val_epoch_", "").replace("_samples.tsv", "")
        try:
            epoch_num = int(epoch_str)
        except ValueError:
            epoch_num = 0

        samples = []
        try:
            with open(latest, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                for row in reader:
                    gt = row.get("label", "")
                    pr = row.get("pred", "")
                    samples.append({"gt": gt, "pr": pr})
        except Exception:
            pass

        self._json_response({
            "epoch": epoch_num,
            "total": len(samples),
            "samples": samples,
        })

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
