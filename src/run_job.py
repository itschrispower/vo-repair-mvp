import shutil
import sys
from pathlib import Path

import librosa

from pt_to_positions import main as pt_to_positions_main
from engine import main as engine_main
from utils import SR, find_single_aaf as _find_single_aaf
DEBUG = False  # False = clean product mode


def run_step(label, func, argv):
    print(f"\n> {' '.join(argv)}")
    old_argv = sys.argv[:]
    try:
        sys.argv = argv
        try:
            func()
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 0
            if code not in (0, None):
                raise RuntimeError(f"{label} failed with exit code {code}")
    finally:
        sys.argv = old_argv


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


def find_single_aaf(input_dir: Path) -> Path:
    return _find_single_aaf(input_dir)


def copy_if_needed(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        try:
            same = src.resolve() == dst.resolve()
        except Exception:
            same = False
        if same:
            return
    shutil.copy2(src, dst)


def stage_inputs_for_pipeline(job_path: Path):
    input_dir = job_path / "INPUT"
    input_audio = input_dir / "audio"

    aaf_src = find_single_aaf(input_dir)
    vobu_src = input_audio / "VOBU_48k.wav"
    ref_src = input_audio / "aaf_reference.wav"

    legacy_audio = job_path / "audio"
    legacy_aaf = job_path / aaf_src.name

    copy_if_needed(vobu_src, legacy_audio / "VOBU_48k.wav")
    copy_if_needed(ref_src, legacy_audio / "aaf_reference.wav")
    copy_if_needed(aaf_src, legacy_aaf)

    return legacy_aaf, legacy_audio


def copy_outputs_to_user_output(job_path: Path):
    deliver_dir = job_path / "deliverables"
    output_dir = job_path / "OUTPUT"
    output_dir.mkdir(exist_ok=True)

    aaf_files = sorted(deliver_dir.glob("*_rebuilt.aaf"))
    wav_files = sorted(deliver_dir.glob("*_final.wav"))
    summary_files = sorted(deliver_dir.glob("*_summary.txt"))

    if not aaf_files:
        raise FileNotFoundError(f"No rebuilt AAF found in {deliver_dir}")
    if not wav_files:
        raise FileNotFoundError(f"No final WAV found in {deliver_dir}")
    if not summary_files:
        raise FileNotFoundError(f"No summary file found in {deliver_dir}")

    shutil.copy2(aaf_files[0], output_dir / aaf_files[0].name)
    shutil.copy2(wav_files[0], output_dir / wav_files[0].name)
    shutil.copy2(summary_files[0], output_dir / summary_files[0].name)

    return output_dir


def cleanup_internal(job_path: Path):
    for name in [
        "auto_refs",
        "clips",
        "match_check",
        "rebuild_audio",
        "deliverables",
    ]:
        p = job_path / name
        if p.exists():
            shutil.rmtree(p)


def cleanup_staging(job_path: Path, staged_aaf: Path, staged_audio: Path):
    positions = job_path / "positions.txt"

    if staged_aaf.exists():
        staged_aaf.unlink()

    if staged_audio.exists():
        shutil.rmtree(staged_audio)

    if positions.exists():
        positions.unlink()


def process_job(job_path: Path):
    job_path = Path(job_path).expanduser().resolve()

    input_dir = job_path / "INPUT"
    input_audio = input_dir / "audio"

    require_dir(job_path, "job folder")
    require_dir(input_dir, "INPUT folder")
    require_dir(input_audio, "INPUT/audio folder")

    aaf_src = find_single_aaf(input_dir)
    vobu_src = input_audio / "VOBU_48k.wav"
    ref_src = input_audio / "aaf_reference.wav"

    require_file(aaf_src, "AAF")
    require_file(vobu_src, "VOBU WAV")
    require_file(ref_src, "AAF reference WAV")

    check_wav_48k(vobu_src, "VOBU WAV")
    check_wav_48k(ref_src, "AAF reference WAV")

    staged_aaf, staged_audio = stage_inputs_for_pipeline(job_path)

    run_step("pt_to_positions", pt_to_positions_main, ["pt_to_positions.py", str(job_path)])
    run_step("engine", engine_main, ["engine.py", str(job_path)])

    output_dir = copy_outputs_to_user_output(job_path)

    if not DEBUG:
        cleanup_internal(job_path)
        cleanup_staging(job_path, staged_aaf, staged_audio)

    print("\nSUCCESS")
    print("Output:")
    print(output_dir)

    return output_dir


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 src/run_job.py /full/path/to/VO_JOB_01")
        sys.exit(1)

    try:
        process_job(Path(sys.argv[1]))
    except Exception as e:
        print(f"FAILED: {e}")
        raise


if __name__ == "__main__":
    main()