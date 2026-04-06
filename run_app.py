import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent

def list_jobs():
    return sorted([p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("VO_JOB")])

def choose_job(jobs):
    print("\nAvailable jobs:\n")
    for i, job in enumerate(jobs, 1):
        print(f"{i}. {job.name}")

    choice = input("\nSelect job number: ").strip()

    if not choice.isdigit():
        print("Invalid choice")
        return None

    idx = int(choice) - 1
    if idx < 0 or idx >= len(jobs):
        print("Invalid choice")
        return None

    return jobs[idx]

def show_summary(job_path: Path):
    summaries = sorted((job_path / "deliverables").glob("*_summary.txt"))
    if not summaries:
        print("\nNo summary file found.")
        return

    print("\n" + "=" * 60)
    print(f"SUMMARY: {summaries[0].name}")
    print("=" * 60)
    print(summaries[0].read_text(encoding="utf-8"))
    print("=" * 60)

def run_job(job_path: Path):
    result = subprocess.run(["python3", "src/run_job.py", job_path.name], cwd=ROOT)
    if result.returncode != 0:
        print("\nJOB FAILED")
        return
    show_summary(job_path)

def main():
    jobs = list_jobs()
    if not jobs:
        print("No job folders found.")
        return

    job = choose_job(jobs)
    if not job:
        return

    run_job(job)

if __name__ == "__main__":
    main()