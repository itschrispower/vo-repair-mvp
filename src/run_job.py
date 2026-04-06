import subprocess
import sys
from pathlib import Path

import librosa

SR = 48000


def run(cmd: str):
    print(f"\n> {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print("FAILED")
        sys.exit(1)


def require(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"{label} missing: {path}")


def check_wav(path: Path, label: str):
    _, sr = librosa.load(str(path), sr=None, mono=True)
    if sr != SR:
        raise ValueError(f"{label} must be {SR} Hz (got {sr})")


def find_aaf(job: Path) -> Path:
    aafs = list(job.glob("*.aaf"))
    if len(aafs) != 1:
        raise ValueError("Exactly ONE AAF required in job folder")
    return aafs[0]


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 src/run_job.py VO_JOB_01")
        sys.exit(1)

    job_name = sys.argv[1]
    base = Path(__file__).resolve().parent.parent
    job = base / job_name

    audio = job / "audio"
    vo = audio / "VOBU_48k.wav"
    ref = audio / "aaf_reference.wav"

    try:
        require(job, "job folder")
        require(audio, "audio folder")
        require(vo, "VOBU")
        require(ref, "AAF reference")
        find_aaf(job)

        check_wav(vo, "VOBU")
        check_wav(ref, "AAF reference")

    except Exception as e:
        print(f"VALIDATION FAILED: {e}")
        sys.exit(1)

    # STEP 1 — AAF → positions
    run(f"python3 src/pt_to_positions.py {job_name}")

    # STEP 2 — build + match + rebuild
    run(f"python3 src/engine.py {job_name}")

    print("\nSUCCESS")
    print("Deliverables:")
    print(job / "deliverables")


if __name__ == "__main__":
    main()
    