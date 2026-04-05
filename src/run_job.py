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


def find_session_info_txt(job_path: Path) -> Path:
    candidates = sorted(
        p for p in job_path.glob("*.txt")
        if p.name.lower() != "positions.txt"
    )
    if not candidates:
        raise FileNotFoundError(f"No Pro Tools text export found in {job_path}")
    if len(candidates) > 1:
        raise ValueError(
            "Expected exactly one Pro Tools text export in job folder, found: "
            + ", ".join(p.name for p in candidates)
        )
    return candidates[0]


def parse_comp_clip_names(txt_path: Path) -> list[str]:
    clip_names = []
    in_comp_track = False

    with open(txt_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            stripped = line.strip()

            if stripped.startswith("TRACK NAME:"):
                track_name = stripped.split("TRACK NAME:", 1)[1].strip()
                in_comp_track = (track_name == "COMP")
                continue

            if not in_comp_track:
                continue

            if not stripped:
                continue

            if stripped.startswith(("COMMENTS:", "USER DELAY:", "STATE:", "PLUG-INS:")):
                continue

            if "CHANNEL" in stripped and "EVENT" in stripped and "CLIP NAME" in stripped:
                continue

            parts = stripped.split()
            if len(parts) < 6:
                continue

            if not parts[0].isdigit() or not parts[1].isdigit():
                continue

            clip_name = parts[2]
            clip_names.append(clip_name)

    if not clip_names:
        raise ValueError(f"No COMP clip rows found in {txt_path}")

    return clip_names


def candidate_ref_paths(ref_dir: Path, clip_name: str) -> list[Path]:
    return [
        ref_dir / f"{clip_name}.wav",
        ref_dir / f"{clip_name}_ref.wav",
        ref_dir / f"clip_{clip_name}.wav",
        ref_dir / f"clip_{clip_name}_ref.wav",
    ]


def validate_reference_clips(ref_dir: Path, clip_names: list[str]):
    missing = []

    for clip_name in clip_names:
        candidates = candidate_ref_paths(ref_dir, clip_name)
        found = None
        for p in candidates:
            if p.exists():
                found = p
                break

        if found is None:
            missing.append((clip_name, [p.name for p in candidates]))
            continue

        check_wav_48k(found, f"Reference clip for {clip_name}")

    if missing:
        lines = []
        for clip_name, tried in missing:
            lines.append(f"{clip_name}: tried {', '.join(tried)}")
        raise FileNotFoundError("Missing reference clips:\n" + "\n".join(lines))


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
    clips_dir = job_path / "clips"
    match_dir = job_path / "match_check"
    rebuild_dir = job_path / "rebuild_audio"

    vo_path = audio_dir / "VOBU_48k.wav"
    ref_path = audio_dir / "aaf_reference.wav"

    try:
        require_dir(audio_dir, "audio folder")
        require_dir(clips_dir, "clips folder")

        match_dir.mkdir(exist_ok=True)
        rebuild_dir.mkdir(exist_ok=True)

        require_file(vo_path, "VO WAV")
        require_file(ref_path, "AAF reference WAV")

        check_wav_48k(vo_path, "VO WAV")
        check_wav_48k(ref_path, "AAF reference WAV")

        txt_path = find_session_info_txt(job_path)
        clip_names = parse_comp_clip_names(txt_path)
        validate_reference_clips(clips_dir, clip_names)

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