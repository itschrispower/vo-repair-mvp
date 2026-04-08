"""
run_app.py — primary entry point for the VO Repair desktop GUI.

Usage:
    python3 run_app.py
"""
import sys
from pathlib import Path

# Add src/ to path so gui_app.py can import run_job, engine, etc.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from gui_app import main

if __name__ == "__main__":
    main()
