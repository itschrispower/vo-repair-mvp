import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SR = 48000
MIN_CONFIDENCE = 0.60


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

            ref_name, start_tc, end_tc = parts
            clips.append((ref_name, start_tc, end_tc))

    return clips


def resolve_job_root() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).resolve()
    return Path(__file__).resolve().parent.parent


def main():
    base = resolve_job_root()

    vo_path = base / "audio" / "VOBU_48k.wav"
    ref_dir = base / "clips"
    out_dir = base / "rebuild_audio"
    check_dir = base / "match_check"
    positions_path = base / "positions.txt"

    final_path = out_dir / "final.wav"
    report_path = out_dir / "report.json"

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

    max_end = max(tc_to_samples(end_tc) for _, _, end_tc in clips)
    output = np.zeros(max_end, dtype=np.float32)

    report = []
    failed = False

    for i, (ref_name, start_tc, end_tc) in enumerate(clips, start=1):
        ref_path = ref_dir / ref_name
        if not ref_path.exists():
            raise FileNotFoundError(f"Missing reference clip: {ref_path}")

        ref_clip, ref_sr = librosa.load(str(ref_path), sr=SR)
        if ref_sr != SR:
            raise ValueError(f"Reference clip must be {SR} Hz: {ref_name}")

        match_start, confidence = match_clip(vo, ref_clip)

        out_start = tc_to_samples(start_tc)
        out_end = tc_to_samples(end_tc)
        target_len = out_end - out_start

        vo_clip = vo[match_start:match_start + target_len]

        if len(vo_clip) < target_len:
            padded = np.zeros(target_len, dtype=np.float32)
            padded[:len(vo_clip)] = vo_clip
            vo_clip = padded

        preview_path = check_dir / f"match_clip_{i}.wav"
        sf.write(str(preview_path), vo_clip[:target_len], SR)

        status = "ok"
        if confidence < MIN_CONFIDENCE:
            status = "low_confidence"
            failed = True
        else:
            output[out_start:out_end] = vo_clip[:target_len]

        report.append(
            {
                "index": i,
                "ref_name": ref_name,
                "timeline_start_tc": start_tc,
                "timeline_end_tc": end_tc,
                "timeline_start_sec": round(out_start / SR, 6),
                "timeline_end_sec": round(out_end / SR, 6),
                "source_match_sec": round(match_start / SR, 6),
                "confidence": round(confidence, 6),
                "status": status,
                "output_preview": str(preview_path.relative_to(base)),
            }
        )

        print(f"{ref_name} -> {match_start / SR:.6f}s | conf={confidence:.4f} | {status}")

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if failed:
        print("STOPPED: one or more clips were low confidence")
        print(report_path)
        return

    sf.write(str(final_path), output, SR)

    print("DONE")
    print(final_path)
    print(report_path)


if __name__ == "__main__":
    main()