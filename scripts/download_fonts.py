"""Download OFL Arabic fonts for synthetic handwriting rendering.

Pulls TTFs from the google/fonts GitHub repo (all OFL-licensed) into
archive/fonts/ (gitignored, same as the KHATT data). Idempotent: skips
files that already exist.

Font selection rationale (see project memory, Option A plan): KHATT is
everyday cursive handwriting, closest to ruqaa/informal naskh. We want
style diversity across ruqaa, naskh, nastaliq, and hand-drawn faces.

Usage:
    python scripts/download_fonts.py
"""
import json
import sys
import urllib.request
from pathlib import Path

# google/fonts repo directory names under ofl/
FAMILIES = [
    "arefruqaa",        # ruqaa — closest match to KHATT everyday hand
    "arefruqaaink",     # ruqaa with ink texture
    "scheherazadenew",  # SIL traditional naskh
    "lateef",           # SIL informal-leaning naskh
    "notonastaliqurdu", # nastaliq (variable font, default instance ok)
    "mirza",            # nastaliq-flavored
    "katibeh",          # nastaliq-ish display
    "rakkas",           # ruqaa display
    "vibes",            # casual handwriting-style
    "harmattan",        # SIL warsh-style (West African hand)
    "alkalami",         # SIL Kano/Ajami — strongly hand-drawn
]

API = "https://api.github.com/repos/google/fonts/contents/ofl/{fam}"
OUT_DIR = Path(__file__).resolve().parent.parent / "archive" / "fonts"


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "arabicocr-khatt-fontdl"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ok, failed = [], []
    for fam in FAMILIES:
        try:
            listing = json.loads(fetch(API.format(fam=fam)))
            ttfs = [e for e in listing if e["name"].lower().endswith(".ttf")]
            if not ttfs:
                failed.append((fam, "no .ttf in listing"))
                continue
            for e in ttfs:
                dest = OUT_DIR / e["name"]
                if dest.exists():
                    ok.append((fam, e["name"] + " (cached)"))
                    continue
                dest.write_bytes(fetch(e["download_url"]))
                ok.append((fam, e["name"]))
                print(f"  {fam}: {e['name']} ({dest.stat().st_size // 1024} KB)")
        except Exception as exc:  # noqa: BLE001 — report and continue
            failed.append((fam, str(exc)))
            print(f"  {fam}: FAILED ({exc})", file=sys.stderr)

    print(f"\n{len(ok)} font files ready in {OUT_DIR}")
    if failed:
        print(f"{len(failed)} families failed: {[f[0] for f in failed]}")
    return 1 if failed and not ok else 0


if __name__ == "__main__":
    sys.exit(main())
