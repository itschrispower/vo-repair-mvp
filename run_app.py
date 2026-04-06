import os
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_jobs():
    return sorted([d for d in os.listdir(BASE_DIR) if d.startswith("VO_JOB")])

def run_job(job_name):
    print(f"\nRunning job: {job_name}\n")

    subprocess.run(["python3", "src/pt_to_positions.py", job_name], cwd=BASE_DIR)
    subprocess.run(["python3", "src/engine.py", job_name], cwd=BASE_DIR)

    # OPEN DELIVERABLES IN FINDER
    deliverables_path = os.path.join(BASE_DIR, job_name, "deliverables")
    if os.path.exists(deliverables_path):
        subprocess.run(["open", deliverables_path])

def main():
    jobs = get_jobs()

    if not jobs:
        print("No VO jobs found.")
        return

    run_job(jobs[0])

if __name__ == "__main__":
    main()