#!/bin/zsh
set -e

cd "$(dirname "$0")"

if ! command -v python3 >/dev/null 2>&1; then
  osascript -e 'display dialog "python3 not found. Install Python 3 first." buttons {"OK"} default button "OK"'
  exit 1
fi

python3 run_app.py