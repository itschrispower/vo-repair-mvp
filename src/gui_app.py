import subprocess
from pathlib import Path

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
    return Path(result.stdout.strip())


def run_job(job_path: Path):
    job_name = job_path.name

    result = subprocess.run(
        ["python3", "src/run_job.py", job_name],
        cwd=BASE,
        text=True,
    )
    return result.returncode == 0


def open_deliverables(job_path: Path):
    path = job_path / "deliverables"
    if path.exists():
        subprocess.run(["open", str(path)])


def show_done():
    subprocess.run(
        [
            "osascript",
            "-e",
            'display dialog "VO Repair finished." buttons {"OK"} default button "OK"',
        ]
    )


def main():
    job_path = pick_folder()
    if not job_path:
        return

    ok = run_job(job_path)
    if ok:
        open_deliverables(job_path)
        show_done()


if __name__ == "__main__":
    main()