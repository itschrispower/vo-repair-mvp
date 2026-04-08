"""
src/run_job.py

Orchestrates the two-step pipeline (AAF → positions.txt → engine)
and manages job folder structure.

process_files() is the primary entry point called by the GUI.
process_job()   is the legacy folder-based CLI entry point.
"""
import shutil
import sys
import tempfile
from pathlib import Path

from pt_to_positions import main as pt_to_positions_main
from engine import main as engine_main
from utils import SR, find_single_aaf as _find_single_aaf

DEBUG = False  # False = clean product mode (removes temp working files after run)


def run_step(label, func, argv):
    """Invoke a pipeline stage function with a fake sys.argv."""
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
    """
    Fast sample-rate check using soundfile header read (no full file decode).
    Falls back to librosa if soundfile is unavailable.
    """
    try:
        import soundfile as sf
        info = sf.info(str(path))
        sr = info.samplerate
    except Exception:
        import librosa
        _, sr = librosa.load(str(path), sr=None, mono=True, duration=0.1)
    if sr != SR:
        raise ValueError(
            f"{label} must be {SR} Hz — got {sr} Hz.\n"
            f"Please export/resample to 48 kHz before processing: {path.name}"
        )


def find_single_aaf(input_dir: Path) -> Path:
    return _find_single_aaf(input_dir)


def copy_if_needed(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        same = src.resolve() == dst.resolve()
    except Exception:
        same = False
    if not same:
        shutil.copy2(src, dst)


# ── Folder-based job structure (legacy CLI flow) ──────────────────────────────

def stage_inputs_for_pipeline(job_path: Path):
    input_dir   = job_path / "INPUT"
    input_audio = input_dir / "audio"

    aaf_src  = find_single_aaf(input_dir)
    vobu_src = input_audio / "VOBU_48k.wav"
    ref_src  = input_audio / "aaf_reference.wav"

    legacy_audio = job_path / "audio"
    legacy_aaf   = job_path / aaf_src.name

    copy_if_needed(vobu_src, legacy_audio / "VOBU_48k.wav")
    copy_if_needed(ref_src,  legacy_audio / "aaf_reference.wav")
    copy_if_needed(aaf_src,  legacy_aaf)

    return legacy_aaf, legacy_audio


def copy_outputs_to_user_output(job_path: Path) -> Path:
    """
    Copy deliverables to OUTPUT/. Best-effort: copies whatever exists.
    Does not raise if the AAF is missing (engine may have skipped it on error).
    """
    deliver_dir = job_path / "deliverables"
    output_dir  = job_path / "OUTPUT"
    output_dir.mkdir(exist_ok=True)

    aaf_files     = sorted(deliver_dir.glob("*_rebuilt.aaf"))
    wav_files     = sorted(deliver_dir.glob("*_final.wav"))
    summary_files = sorted(deliver_dir.glob("*_summary.txt"))

    if not wav_files:
        raise FileNotFoundError(f"No final WAV found in {deliver_dir} — processing may have failed.")
    if not summary_files:
        raise FileNotFoundError(f"No summary file found in {deliver_dir}.")

    shutil.copy2(wav_files[0],     output_dir / wav_files[0].name)
    shutil.copy2(summary_files[0], output_dir / summary_files[0].name)

    # AAF is optional — engine may fail to write it without stopping the job
    if aaf_files:
        shutil.copy2(aaf_files[0], output_dir / aaf_files[0].name)

    # Copy JSON report so the GUI can display colour-coded results
    rebuild_dir = job_path / "rebuild_audio"
    for report in sorted(rebuild_dir.glob("*_report.json")):
        shutil.copy2(report, output_dir / report.name)

    return output_dir


def cleanup_internal(job_path: Path):
    for name in ["auto_refs", "clips", "match_check", "rebuild_audio", "deliverables"]:
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
    """
    Process a job folder (INPUT/ structure).
    Used by the CLI runner and local_run.command.
    """
    job_path = Path(job_path).expanduser().resolve()

    input_dir   = job_path / "INPUT"
    input_audio = input_dir / "audio"

    require_dir(job_path,    "job folder")
    require_dir(input_dir,   "INPUT folder")
    require_dir(input_audio, "INPUT/audio folder")

    aaf_src  = find_single_aaf(input_dir)
    vobu_src = input_audio / "VOBU_48k.wav"
    ref_src  = input_audio / "aaf_reference.wav"

    require_file(aaf_src,  "AAF")
    require_file(vobu_src, "VOBU WAV")
    require_file(ref_src,  "AAF reference WAV")

    check_wav_48k(vobu_src, "VOBU WAV")
    check_wav_48k(ref_src,  "AAF reference WAV")

    staged_aaf, staged_audio = stage_inputs_for_pipeline(job_path)

    run_step("pt_to_positions", pt_to_positions_main, ["pt_to_positions.py", str(job_path)])
    run_step("engine",          engine_main,           ["engine.py",          str(job_path)])

    output_dir = copy_outputs_to_user_output(job_path)

    if not DEBUG:
        cleanup_internal(job_path)
        cleanup_staging(job_path, staged_aaf, staged_audio)

    print("\nSUCCESS")
    print("Output:", output_dir)
    return output_dir


# ── GUI entry point: arbitrary file paths ─────────────────────────────────────

def process_files(
    aaf_path: Path,
    ref_wav_path: Path,
    vobu_wav_path: Path,
    output_dir: Path,
) -> Path:
    """
    Process explicitly provided files — no fixed folder structure required.

    Creates a temporary working directory, runs the full pipeline, then copies
    all output files (WAV, AAF, summary, JSON report) to output_dir.

    Always produces output. If some clips are low-confidence they are written
    as FORCED and clearly flagged in the report — the job never aborts.

    Parameters
    ----------
    aaf_path       Path to the Pro Tools AAF export (.aaf)
    ref_wav_path   Path to the full-session bounce (aaf_reference.wav)
    vobu_wav_path  Path to the VO backup recording (VOBU_48k.wav)
    output_dir     Destination for all output files

    Returns
    -------
    output_dir (Path)
    """
    aaf_path      = Path(aaf_path).resolve()
    ref_wav_path  = Path(ref_wav_path).resolve()
    vobu_wav_path = Path(vobu_wav_path).resolve()
    output_dir    = Path(output_dir).resolve()

    # ── Validate inputs ───────────────────────────────────────────────────────
    for p, label in [
        (aaf_path,      "AAF file"),
        (ref_wav_path,  "AAF Full Bounce WAV"),
        (vobu_wav_path, "VO BU WAV"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"Missing {label}: {p}")

    check_wav_48k(ref_wav_path,  "AAF Full Bounce WAV")
    check_wav_48k(vobu_wav_path, "VO BU WAV")

    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        audio_dir = tmp / "audio"
        audio_dir.mkdir()

        # Stage inputs into the expected pipeline structure
        shutil.copy2(aaf_path,      tmp / aaf_path.name)
        shutil.copy2(ref_wav_path,  audio_dir / "aaf_reference.wav")
        shutil.copy2(vobu_wav_path, audio_dir / "VOBU_48k.wav")

        # ── Run pipeline stages ───────────────────────────────────────────────
        run_step(
            "pt_to_positions", pt_to_positions_main,
            ["pt_to_positions.py", str(tmp)],
        )
        run_step(
            "engine", engine_main,
            ["engine.py", str(tmp)],
        )

        # ── Copy outputs — always best-effort ─────────────────────────────────
        deliver_dir = tmp / "deliverables"
        if not deliver_dir.exists():
            raise RuntimeError(
                "Processing failed — deliverables folder was not created.\n"
                "Check the log above for errors during the pipeline stages."
            )

        copied = []
        for f in sorted(deliver_dir.iterdir()):
            dst = output_dir / f.name
            shutil.copy2(f, dst)
            copied.append(f.name)

        if not copied:
            raise RuntimeError(
                "Processing produced no output files.\n"
                "Check the log above for pipeline errors."
            )

        # Copy JSON report so the GUI can show colour-coded results
        rebuild_dir = tmp / "rebuild_audio"
        for report in sorted(rebuild_dir.glob("*_report.json")):
            shutil.copy2(report, output_dir / report.name)

    print(f"\nSUCCESS — output in: {output_dir}")
    return output_dir


# ── CLI entry point ───────────────────────────────────────────────────────────

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
