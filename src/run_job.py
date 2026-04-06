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


def require_file(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def require_dir(path: Path, label: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{label} is not a folder: {path}")


def check_wav_48k(path: Path, label: str):
    _, sr = librosa.load(str(path), sr=None, mono=True)
    if sr != SR:
        raise ValueError(f"{label} must be {SR} Hz: {path} (got {sr})")


def find_single_aaf(job_path: Path) -> Path:
    aafs = sorted(job_path.glob("*.aaf"))
    if not aafs:
        raise FileNotFoundError(f"No AAF found in {job_path}")
    if len(aafs) > 1:
        raise ValueError(
            "Expected exactly one AAF in job folder, found: "
            + ", ".join(p.name for p in aafs)
        )
    return aafs[0]


def validate_positions(path: Path):
    require_file(path, "positions file")

    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise ValueError(f"positions.txt is empty: {path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 src/run_job.py VO_JOB_01")
        sys.exit(1)

    job = sys.argv[1]

    base = Path(__file__).resolve().parent.parent
    job_path = (base / job).resolve()

    if not job_path.exists():
        print(f"Job folder not found: {job_path}")
        sys.exit(1)

    audio_dir = job_path / "audio"
    match_dir = job_path / "match_check"
    rebuild_dir = job_path / "rebuild_audio"

    vo_path = audio_dir / "VOBU_48k.wav"
    ref_path = audio_dir / "aaf_reference.wav"

    try:
        require_dir(audio_dir, "audio folder")

        match_dir.mkdir(exist_ok=True)
        rebuild_dir.mkdir(exist_ok=True)

        require_file(vo_path, "VO WAV")
        require_file(ref_path, "AAF reference WAV")
        find_single_aaf(job_path)

        check_wav_48k(vo_path, "VO WAV")
        check_wav_48k(ref_path, "AAF reference WAV")

    except Exception as e:
        print(f"VALIDATION FAILED: {e}")
        sys.exit(1)

    run(f"python3 src/pt_to_positions.py {job}")

    try:
        validate_positions(job_path / "positions.txt")
    except Exception as e:
        print(f"VALIDATION FAILED AFTER POSITION BUILD: {e}")
        sys.exit(1)

    run(f"python3 src/engine.py {job}")

    print("\nDONE — final.wav ready")


if __name__ == "__main__":
    main()