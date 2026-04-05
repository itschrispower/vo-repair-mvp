import json
import re
import sys
from pathlib import Path
import shutil

import librosa
import numpy as np
import soundfile as sf

SR = 48000
FAIL_THRESHOLD = 0.75
REVIEW_THRESHOLD = 0.90


def tc_to_seconds(tc: str) -> float:
    secs, frames = tc.split(".")
    return int(secs) + (int(frames) / 25.0)


def tc_to_samples(tc: str) -> int:
    return int(tc_to_seconds(tc) * SR)


def norm(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x -= np.mean(x)
    peak = np.max(np.abs(x))
    if peak > 0:
        x /= peak
    return x


def match_clip(full: np.ndarray, clip: np.ndarray) -> tuple[int, float]:
    full_n = norm(full)
    clip_n = norm(clip)

    corr = np.correlate(full_n, clip_n, mode="valid")
    match_start = int(np.argmax(corr))

    matched_segment = full_n[match_start:match_start + len(clip_n)]
    denom = np.linalg.norm(matched_segment) * np.linalg.norm(clip_n)

    if denom == 0:
        confidence = 0.0
    else:
        confidence = float(corr[match_start] / denom)

    return match_start, confidence


def load_positions(path: Path):
    clips = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"Bad line in positions.txt: {line}")

            clip_name, start_tc, end_tc = parts
            clips.append((clip_name, start_tc, end_tc))

    return clips


def resolve_job_root() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).resolve()
    return Path(__file__).resolve().parent.parent


def sanitise_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return cleaned or "job"


def find_reference_clip(ref_dir: Path, clip_name: str) -> Path:
    candidates = [
        ref_dir / f"{clip_name}.wav",
        ref_dir / f"{clip_name}_ref.wav",
        ref_dir / f"clip_{clip_name}.wav",
        ref_dir / f"clip_{clip_name}_ref.wav",
        ref_dir / f"{sanitise_name(clip_name)}.wav",
        ref_dir / f"{sanitise_name(clip_name)}_ref.wav",
        ref_dir / f"clip_{sanitise_name(clip_name)}.wav",
        ref_dir / f"clip_{sanitise_name(clip_name)}_ref.wav",
    ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Missing reference clip for '{clip_name}'. Tried: "
        + ", ".join(str(p.name) for p in candidates)
    )


def detect_job_label(base: Path) -> str:
    aaf_files = sorted(base.glob("*.aaf"))
    if aaf_files:
        return sanitise_name(aaf_files[0].stem)

    txt_files = sorted(
        p for p in base.glob("*.txt")
        if p.name.lower() != "positions.txt"
    )
    if txt_files:
        return sanitise_name(txt_files[0].stem)

    return sanitise_name(base.name)


def write_summary(path: Path, report: list[dict], final_path: Path, failed: bool):
    lines = []

    lines.append("VO REPAIR SUMMARY")
    lines.append("")

    for item in report:
        lines.append(f"Clip {item['index']} — {item['status'].upper()}")
        lines.append(f"Name: {item['clip_name']}")
        lines.append(f"Ref: {item['ref_name']}")
        lines.append(f"Placed: {item['timeline_start_tc']} to {item['timeline_end_tc']}")
        lines.append(f"Matched in VOBU at: {item['source_match_sec']:.6f}s")
        lines.append(f"Confidence: {item['confidence']:.4f}")
        lines.append(f"Preview: {item['output_preview']}")
        lines.append("")

    if failed:
        lines.append("Final output was NOT written because one or more clips failed validation.")
    else:
        lines.append(f"Final output written: {final_path}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    base = resolve_job_root()
    job_label = detect_job_label(base)

    vo_path = base / "audio" / "VOBU_48k.wav"
    ref_dir = base / "clips"
    out_dir = base / "rebuild_audio"
    check_dir = base / "match_check"
    deliver_dir = base / "deliverables"
    positions_path = base / "positions.txt"

    final_path = out_dir / f"{job_label}_final.wav"
    report_path = out_dir / f"{job_label}_report.json"
    summary_path = out_dir / f"{job_label}_summary.txt"

    deliver_final_path = deliver_dir / f"{job_label}_final.wav"
    deliver_summary_path = deliver_dir / f"{job_label}_summary.txt"

    if not vo_path.exists():
        raise FileNotFoundError(f"Missing VO file: {vo_path}")

    if not ref_dir.exists():
        raise FileNotFoundError(f"Missing clips folder: {ref_dir}")

    if not positions_path.exists():
        raise FileNotFoundError(f"Missing positions file: {positions_path}")

    vo, vo_sr = librosa.load(str(vo_path), sr=SR)
    if vo_sr != SR:
        raise ValueError(f"VO file must be {SR} Hz")

    clips = load_positions(positions_path)

    out_dir.mkdir(exist_ok=True)
    check_dir.mkdir(exist_ok=True)
    deliver_dir.mkdir(exist_ok=True)

    max_end = max(tc_to_samples(end_tc) for _, _, end_tc in clips)
    output = np.zeros(max_end, dtype=np.float32)

    report = []
    failed = False

    for i, (clip_name, start_tc, end_tc) in enumerate(clips, start=1):
        ref_path = find_reference_clip(ref_dir, clip_name)

        ref_clip, ref_sr = librosa.load(str(ref_path), sr=SR)
        if ref_sr != SR:
            raise ValueError(f"Reference clip must be {SR} Hz: {ref_path.name}")

        match_start, confidence = match_clip(vo, ref_clip)

        out_start = tc_to_samples(start_tc)
        out_end = tc_to_samples(end_tc)
        target_len = out_end - out_start

        vo_clip = vo[match_start:match_start + target_len]

        if len(vo_clip) < target_len:
            padded = np.zeros(target_len, dtype=np.float32)
            padded[:len(vo_clip)] = vo_clip
            vo_clip = padded

        preview_path = check_dir / f"match_{sanitise_name(clip_name)}.wav"
        sf.write(str(preview_path), vo_clip[:target_len], SR)

        if confidence >= REVIEW_THRESHOLD:
            status = "ok"
            output[out_start:out_end] = vo_clip[:target_len]
        elif confidence >= FAIL_THRESHOLD:
            status = "review"
            output[out_start:out_end] = vo_clip[:target_len]
        else:
            status = "fail"
            failed = True

        item = {
            "index": i,
            "clip_name": clip_name,
            "ref_name": ref_path.name,
            "timeline_start_tc": start_tc,
            "timeline_end_tc": end_tc,
            "timeline_start_sec": round(out_start / SR, 6),
            "timeline_end_sec": round(out_end / SR, 6),
            "source_match_sec": round(match_start / SR, 6),
            "confidence": round(confidence, 6),
            "status": status,
            "output_preview": str(preview_path.relative_to(base)),
        }
        report.append(item)

        print(
            f"{clip_name} -> {ref_path.name} -> "
            f"{match_start / SR:.6f}s | conf={confidence:.4f} | {status.upper()}"
        )

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if not failed:
        sf.write(str(final_path), output, SR)

    write_summary(summary_path, report, final_path, failed)

    if failed:
        print("STOPPED: one or more clips failed validation")
        print(report_path)
        print(summary_path)
        return

    shutil.copy2(final_path, deliver_final_path)
    shutil.copy2(summary_path, deliver_summary_path)

    print("DONE")
    print(final_path)
    print(report_path)
    print(summary_path)
    print(deliver_final_path)
    print(deliver_summary_path)


if __name__ == "__main__":
    main()