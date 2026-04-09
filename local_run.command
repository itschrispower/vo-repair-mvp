#!/bin/bash
# VO Repair — double-click launcher for macOS
# Robust Python & Tk runtime detection to avoid Apple CLT Tcl/Tk crash
# Safe to run repeatedly: creates .venv on first run, updates deps, then launches GUI.

set -e

# ── Move to the repo root (same folder as this script) ───────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
echo "Working directory: $SCRIPT_DIR"

# ── Detect a good Python interpreter (not Apple CLT's broken Tk) ──────────────
CHOSEN_PYTHON=""

# Priority 1: Homebrew Python (has working Tk on modern macOS)
if [ -x "/opt/homebrew/bin/python3" ]; then
    CHOSEN_PYTHON="/opt/homebrew/bin/python3"
    echo "✓ Using Homebrew Python: $CHOSEN_PYTHON"
# Priority 2: Intel Homebrew (for older Macs)
elif [ -x "/usr/local/bin/python3" ]; then
    CHOSEN_PYTHON="/usr/local/bin/python3"
    echo "✓ Using Intel Homebrew Python: $CHOSEN_PYTHON"
# Priority 3: System Python (might be broken, but fallback)
elif [ -x "/usr/bin/python3" ]; then
    CHOSEN_PYTHON="/usr/bin/python3"
    echo "⚠ Falling back to system Python (may have Tk issues): $CHOSEN_PYTHON"
else
    echo "✗ ERROR: No Python interpreter found"
    echo "Please install Python 3 via Homebrew: brew install python3"
    echo "Press Enter to close…"
    read -r
    exit 1
fi

# ── Create virtual environment if it doesn't exist ─────────────────────────────
if [ ! -d ".venv" ]; then
    echo "==> Creating virtual environment with $CHOSEN_PYTHON…"
    "$CHOSEN_PYTHON" -m venv .venv
    echo "    Done."
fi

# ── Activate ──────────────────────────────────────────────────────────────────
source .venv/bin/activate

# ── Verify venv Python is available ───────────────────────────────────────────
if [ ! -x ".venv/bin/python3" ]; then
    echo "✗ ERROR: Virtual environment creation failed"
    echo "Press Enter to close…"
    read -r
    exit 1
fi

# ── Install / sync dependencies ───────────────────────────────────────────────
echo "==> Checking dependencies…"
.venv/bin/python3 -m pip install -r requirements.txt --quiet --quiet 2>/dev/null || {
    echo "⚠ Pip install had warnings (may be network-related); continuing anyway…"
}

# ── Launch GUI ────────────────────────────────────────────────────────────────
echo "==> Starting VO Repair…"
.venv/bin/python3 run_app.py

# ── Keep terminal open on error ───────────────────────────────────────────────
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "=== VO REPAIR CRASHED ==="
    echo "Exit code: $EXIT_CODE"
    echo ""
    echo "Press Enter to close…"
    read -r
fi

exit $EXIT_CODE
