#!/bin/bash
cd "$(dirname "$0")"
source .venv/bin/activate
python3 src/gui_app.py 2>&1