@echo off
REM VO Repair — Windows launcher
REM Safe to run repeatedly: creates .venv on first run, updates deps, then launches GUI.

cd /d "%~dp0"

if not exist ".venv" (
    echo =^> Creating virtual environment...
    python -m venv .venv
    echo     Done.
)

call .venv\Scripts\activate.bat

echo =^> Checking dependencies...
python -m pip install -q -r requirements.txt

echo =^> Starting VO Repair...
python run_app.py

pause
