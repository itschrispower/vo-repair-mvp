#!/bin/bash
# VO Repair — Linux/Unix launcher
# Safe to run repeatedly: creates .venv on first run, updates deps, then launches GUI.

set -e

# Move to the repo root
cd "$(dirname "$0")"

# Create virtual environment if needed
if [ ! -d ".venv" ]; then
    echo "==> Creating virtual environment…"
    python3 -m venv .venv
    echo "    Done."
fi

# Activate
source .venv/bin/activate

# Install / sync dependencies
echo "==> Checking dependencies…"
python3 -m pip install -q -r requirements.txt

# Launch GUI
echo "==> Starting VO Repair…"
python3 run_app.py
