import subprocess
import traceback
from pathlib import Path

import run_job

BASE = Path(__file__).resolve().parent.parent


def pick_folder():
    result = subprocess.run(
        [
            "osascript",
            "-e",
            'POSIX path of (choose folder with prompt "Select VO job folder")',
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return Path(result.stdout.strip()).resolve()


def run_selected_job(job_path: Path):
    try:
        error_log = job_path / "error.txt"
        if error_log.exists():
            error_log.unlink()

        run_job.process_job(job_path)
        return True
    except Exception:
        error_log = job_path / "error.txt"
        error_log.write_text(traceback.format_exc())
        return False


def open_output(job_path: Path):
    output_dir = job_path / "OUTPUT"
    if output_dir.exists():
        subprocess.run(["open", str(output_dir)])


def show_done():
    subprocess.run(
        [
            "osascript",
            "-e",
            'display dialog "VO Repair finished." buttons {"OK"} default button "OK"',
        ]
    )


def show_failed():
    subprocess.run(
        [
            "osascript",
            "-e",
            'display dialog "VO Repair failed. See error.txt in your job folder." buttons {"OK"} default button "OK"',
        ]
    )


def main():
    job_path = pick_folder()
    if not job_path:
        return

    ok = run_selected_job(job_path)

    if ok:
        open_output(job_path)
        show_done()
    else:
        show_failed()


if __name__ == "__main__":
    main()