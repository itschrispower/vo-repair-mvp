import subprocess
import sys
from pathlib import Path


def run(cmd):
    print(f"\n> {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print("FAILED")
        sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 src/run_job.py VO_JOB_01")
        sys.exit(1)

    job = sys.argv[1]

    base = Path(__file__).resolve().parent.parent
    job_path = base / job

    if not job_path.exists():
        print(f"Job folder not found: {job}")
        sys.exit(1)

    run(f"python3 src/pt_to_positions.py {job}")
    run(f"python3 src/engine.py {job}")

    print("\nDONE — final.wav ready")


if __name__ == "__main__":
    main()