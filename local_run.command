#!/bin/bash
# VO Repair — double-click launcher for macOS
# Safe to run repeatedly: creates .venv on first run, updates deps, then launches GUI.

set -e

# ── Move to the repo root (same folder as this script) ───────────────────────
cd "$(dirname "$0")"

# ── Create virtual environment if it doesn't exist ───────────────────────────
if [ ! -d ".venv" ]; then
    echo "==> Creating virtual environment…"
    python3 -m venv .venv
    echo "    Done."
fi

# ── Activate ─────────────────────────────────────────────────────────────────
source .venv/bin/activate

# ── Install / sync dependencies ───────────────────────────────────────────────
echo "==> Checking dependencies…"
python3 -m pip install -r requirements.txt --quiet --quiet

# ── Launch GUI ────────────────────────────────────────────────────────────────
echo "==> Starting VO Repair…"
python3 run_app.py
