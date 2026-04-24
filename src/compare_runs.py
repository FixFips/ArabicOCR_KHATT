"""Compare multiple training runs side by side.

Reads metrics.csv from each given run directory, prints a CLI summary
table, and serves a small web dashboard that overlays CER / WER /
DotCER / Train-Loss curves.

Usage:
    python -m src.compare_runs runs/exp1 runs/exp2
    python -m src.compare_runs runs/exp1 runs/exp2 --port 8090
    python -m src.compare_runs runs/exp1 runs/exp2 --cli-only
"""
import argparse
import csv
import json
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _read_metrics(run_dir: str):
    path = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(path):
        return []
    # metrics.csv can accumulate multiple training runs (same dir, run again
    # without clearing). Keep the LAST occurrence of each epoch number so the
    # chart reflects the most recent run, not whatever was written first.
    by_epoch = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames:
            reader.fieldnames = [n.strip() for n in reader.fieldnames]
        for row in reader:
            row = {k.strip() if isinstance(k, str) else k:
                   v.strip() if isinstance(v, str) else v
                   for k, v in row.items() if isinstance(k, str)}
            try:
                ep = int(float(row.get("epoch", 0)))
                by_epoch[ep] = {
                    "epoch": ep,
                    "train_loss": float(row.get("train_loss", 0)),
                    "cer": float(row.get("cer", 1)),
                    "wer": float(row.get("wer", 1)),
                    "wer_norm": float(row.get("wer_norm", 1)),
                    "dot_cer": float(row.get("dot_cer", 1)),
                    "lr": float(row.get("lr", 0)),
                }
            except (ValueError, KeyError):
                pass
    return [by_epoch[e] for e in sorted(by_epoch)]


def _summary(run_dir: str, rows: list):
    if not rows:
        return {"run": run_dir, "epochs": 0}
    last = rows[-1]
    best_cer = min(r["cer"] for r in rows)
    best_epoch = next(r["epoch"] for r in rows if r["cer"] == best_cer)
    min_loss = min(r["train_loss"] for r in rows)
    return {
        "run": run_dir,
        "epochs": len(rows),
        "last_epoch": last["epoch"],
        "best_cer": best_cer,
        "best_epoch": best_epoch,
        "last_cer": last["cer"],
        "last_wer": last["wer"],
        "last_wer_norm": last["wer_norm"],
        "last_dot_cer": last["dot_cer"],
        "min_loss": min_loss,
    }


def _print_cli_summary(summaries):
    cols = [("Run", "run", 24), ("Epochs", "epochs", 6), ("Best CER", "best_cer", 10, "%"),
            ("@Epoch", "best_epoch", 7), ("Last CER", "last_cer", 10, "%"),
            ("Last WER", "last_wer", 10, "%"), ("Last WER(n)", "last_wer_norm", 12, "%"),
            ("Last DotCER", "last_dot_cer", 12, "%"), ("Min Loss", "min_loss", 10)]
    header = "  ".join(f"{c[0]:<{c[2]}}" for c in cols)
    print(header)
    print("-" * len(header))
    for s in summaries:
        cells = []
        for col in cols:
            key = col[1]
            w = col[2]
            v = s.get(key, "-")
            if len(col) == 4 and col[3] == "%" and isinstance(v, (float, int)):
                v = f"{v * 100:.2f}%"
            elif isinstance(v, float):
                v = f"{v:.4f}"
            elif isinstance(v, int):
                v = str(v)
            v = str(v)
            if len(v) > w:
                v = v[: w - 1] + "…"
            cells.append(f"{v:<{w}}")
        print("  ".join(cells))


# --- Web dashboard ----------------------------------------------------------

HTML_PAGE = r"""<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>Run Comparison</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0d1117; color: #e6edf3; padding: 20px; margin: 0; }
  h1 { text-align: center; margin-bottom: 16px; font-size: 1.4em; }
  table { margin: 0 auto 20px; border-collapse: collapse; background: #161b22;
          border: 1px solid #30363d; border-radius: 6px; overflow: hidden; }
  th, td { padding: 8px 14px; border-bottom: 1px solid #30363d; font-size: 0.85em; }
  th { background: #21262d; color: #8b949e; font-weight: 600; text-align: left; }
  td.run { font-family: ui-monospace, monospace; color: #58a6ff; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; max-width: 1300px; margin: 0 auto; }
  .chart-box { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
  .chart-box h3 { margin: 0 0 10px; font-size: 0.95em; color: #8b949e; }
  canvas { width: 100% !important; }
  @media (max-width: 800px) { .grid { grid-template-columns: 1fr; } }
</style></head><body>
<h1>Run Comparison</h1>
<div id="summary"></div>
<div id="h2h" style="margin:0 auto 20px; max-width:1300px;"></div>
<div class="grid">
  <div class="chart-box"><h3>CER</h3><canvas id="c-cer"></canvas></div>
  <div class="chart-box"><h3>DotCER</h3><canvas id="c-dot"></canvas></div>
  <div class="chart-box"><h3>WER</h3><canvas id="c-wer"></canvas></div>
  <div class="chart-box"><h3>Train Loss</h3><canvas id="c-loss"></canvas></div>
</div>
<script>
const COLORS = ['#58a6ff','#f78166','#7ee787','#d2a8ff','#ffa657','#f85149','#3fb950','#a371f7'];
const opts = (yLabel) => ({
  responsive: true, animation: { duration: 200 },
  scales: { x: { title:{display:true,text:'Epoch',color:'#8b949e'}, ticks:{color:'#8b949e'}, grid:{color:'#21262d'} },
            y: { title:{display:true,text:yLabel,color:'#8b949e'}, ticks:{color:'#8b949e'}, grid:{color:'#21262d'}, beginAtZero:true } },
  plugins: { legend: { labels: { color:'#e6edf3' } } }
});
function mkChart(id, yLabel) {
  return new Chart(document.getElementById(id), { type:'line', data:{labels:[],datasets:[]}, options: opts(yLabel) });
}
const cers = mkChart('c-cer','Error Rate');
const dots = mkChart('c-dot','Error Rate');
const wers = mkChart('c-wer','Error Rate');
const loss = mkChart('c-loss','Loss');

async function load() {
  const r = await fetch('/api/data?_=' + Date.now());
  const data = await r.json();
  renderSummary(data.summaries);
  renderHeadToHead(data.runs);
  renderCharts(data.runs);
}
function renderHeadToHead(runs) {
  // Use the run with the smallest last epoch as the "live" run — compare others at same epoch.
  const valid = runs.filter(r => r.rows && r.rows.length);
  if (valid.length < 2) { document.getElementById('h2h').innerHTML = ''; return; }
  const lastEps = valid.map(r => r.rows[r.rows.length - 1].epoch);
  const anchor = Math.min(...lastEps);
  const start = Math.max(1, anchor - 4);
  const metrics = [['cer','CER'], ['wer','WER'], ['dot_cer','DotCER'], ['train_loss','Loss']];
  let h = `<h3 style="text-align:center;color:#8b949e;margin-bottom:8px;font-size:0.95em;">Head-to-head @ epoch ${anchor} (and previous 4)</h3>`;
  h += '<table style="margin:0 auto;"><tr><th>Epoch</th>';
  for (const r of valid) for (const [k, lbl] of metrics) h += `<th>${r.name} ${lbl}</th>`;
  if (valid.length === 2) for (const [k, lbl] of metrics.slice(0, 3)) h += `<th>Δ ${lbl}</th>`;
  h += '</tr>';
  const fmt = (key, v) => v == null ? '-' : (key === 'train_loss' ? v.toFixed(3) : (v*100).toFixed(2) + '%');
  for (let ep = start; ep <= anchor; ep++) {
    h += `<tr><td class="run">${ep}</td>`;
    const rowsAtEp = valid.map(r => r.rows.find(x => x.epoch === ep));
    for (const row of rowsAtEp) for (const [k] of metrics) h += `<td>${row ? fmt(k, row[k]) : '-'}</td>`;
    if (valid.length === 2 && rowsAtEp[0] && rowsAtEp[1]) {
      for (const [k, lbl] of metrics.slice(0, 3)) {
        const d = (rowsAtEp[0][k] - rowsAtEp[1][k]) * 100;
        const col = d > 0 ? '#3fb950' : (d < 0 ? '#f85149' : '#8b949e');
        h += `<td style="color:${col};font-weight:600;">${d > 0 ? '+' : ''}${d.toFixed(2)}pp</td>`;
      }
    }
    h += '</tr>';
  }
  h += '</table>';
  h += '<div style="text-align:center;color:#8b949e;font-size:0.8em;margin-top:6px;">Δ = '
     + (valid.length === 2 ? `${valid[0].name} − ${valid[1].name}; positive = ${valid[0].name} worse (higher error)` : '')
     + '</div>';
  document.getElementById('h2h').innerHTML = h;
}
function renderSummary(summaries) {
  let h = '<table><tr><th>Run</th><th>Epochs</th><th>Best CER</th><th>@Ep</th><th>Last WER</th><th>Last DotCER</th><th>Min Loss</th></tr>';
  const best = Math.min(...summaries.map(s => s.best_cer ?? 1));
  for (const s of summaries) {
    const bc = s.best_cer == null ? '-' : (s.best_cer*100).toFixed(2) + '%';
    const wc = s.last_wer == null ? '-' : (s.last_wer*100).toFixed(2) + '%';
    const dc = s.last_dot_cer == null ? '-' : (s.last_dot_cer*100).toFixed(2) + '%';
    const ml = s.min_loss == null ? '-' : s.min_loss.toFixed(4);
    const highlight = s.best_cer === best ? ' style="color:#3fb950;font-weight:700"' : '';
    h += `<tr><td class="run">${s.run}</td><td>${s.epochs}</td><td${highlight}>${bc}</td><td>${s.best_epoch ?? '-'}</td><td>${wc}</td><td>${dc}</td><td>${ml}</td></tr>`;
  }
  h += '</table>';
  document.getElementById('summary').innerHTML = h;
}
function renderCharts(runs) {
  const maxEpoch = Math.max(...runs.map(r => Math.max(...r.rows.map(x => x.epoch), 0)), 0);
  const labels = Array.from({length: maxEpoch}, (_, i) => i + 1);
  function ds(key, color) {
    return runs.map((r, i) => ({
      label: r.name,
      data: labels.map(e => { const row = r.rows.find(x => x.epoch === e); return row ? row[key] : null; }),
      borderColor: COLORS[i % COLORS.length],
      backgroundColor: 'transparent',
      borderWidth: 2, pointRadius: 1, spanGaps: true, fill: false,
    }));
  }
  cers.data.labels = labels; cers.data.datasets = ds('cer'); cers.update();
  dots.data.labels = labels; dots.data.datasets = ds('dot_cer'); dots.update();
  wers.data.labels = labels; wers.data.datasets = ds('wer'); wers.update();
  loss.data.labels = labels; loss.data.datasets = ds('train_loss'); loss.update();
}
load();
setInterval(load, 30000);
</script></body></html>"""


def _serve(runs, port):
    cache = {"runs": runs}

    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/api/data"):
                runs_data = []
                summaries = []
                for run_dir in cache["runs"]:
                    rows = _read_metrics(run_dir)
                    runs_data.append({"name": os.path.basename(run_dir.rstrip("/\\")) or run_dir,
                                      "dir": run_dir, "rows": rows})
                    summaries.append(_summary(run_dir, rows))
                body = json.dumps({"runs": runs_data, "summaries": summaries},
                                  ensure_ascii=False).encode("utf-8")
                self._send(200, body, "application/json; charset=utf-8")
            else:
                self._send(200, HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")

        def log_message(self, *a, **kw):
            pass

    server = HTTPServer(("0.0.0.0", port), H)
    print(f"Run-comparison dashboard: http://0.0.0.0:{port}")
    print("Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="Run directories (each must contain metrics.csv)")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--cli-only", action="store_true", help="Print summary and exit (no web server)")
    args = ap.parse_args()

    summaries = []
    for run_dir in args.runs:
        rows = _read_metrics(run_dir)
        summaries.append(_summary(run_dir, rows))
        print(f"  {run_dir}: {len(rows)} epochs")

    print()
    _print_cli_summary(summaries)

    if args.cli_only:
        return
    _serve(args.runs, args.port)


if __name__ == "__main__":
    main()
