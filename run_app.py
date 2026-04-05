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


def run_job(job_path: Path):
    cmd = f"python3 src/run_job.py {job_path.name}"
    subprocess.run(cmd, shell=True)


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