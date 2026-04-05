import subprocess
from pathlib import Path

ROOT = Path(__file__).parent


def list_jobs():
    return [p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("VO_JOB")]


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
    deliverables = job_path / "deliverables"
    summaries = sorted(deliverables.glob("*_summary.txt"))

    if not summaries:
        print("\nNo summary file found.")
        return

    summary_path = summaries[0]

    print("\n" + "=" * 60)
    print(f"SUMMARY: {summary_path.name}")
    print("=" * 60)
    print(summary_path.read_text(encoding="utf-8"))
    print("=" * 60)


def run_job(job_path: Path):
    cmd = f"python3 src/run_job.py {job_path.name}"
    result = subprocess.run(cmd, shell=True)

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