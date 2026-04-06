import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SR = 48000
FAIL_THRESHOLD = 0.75
REVIEW_THRESHOLD = 0.90

job = Path("VO_JOB_01")
vo_path = job / "audio" / "VOBU_48k.wav"
ref_dir = job / "auto_refs"
positions_path = job / "positions.txt"
out_dir = job / "auto_ref_test_output"
check_dir = out_dir / "match_check"

out_dir.mkdir(exist_ok=True)
check_dir.mkdir(exist_ok=True)

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
    confidence = 0.0 if denom == 0 else float(corr[match_start] / denom)
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

vo, vo_sr = librosa.load(str(vo_path), sr=SR)
if vo_sr != SR:
    raise ValueError(f"VO file must be {SR} Hz")

clips = load_positions(positions_path)
max_end = max(tc_to_samples(end_tc) for _, _, end_tc in clips)
output = np.zeros(max_end, dtype=np.float32)

report = []
failed = False

for i, (clip_name, start_tc, end_tc) in enumerate(clips, start=1):
    ref_path = ref_dir / f"{clip_name}_auto_ref.wav"
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

    preview_path = check_dir / f"match_{clip_name}.wav"
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

    report.append(
        {
            "index": i,
            "clip_name": clip_name,
            "timeline_start_tc": start_tc,
            "timeline_end_tc": end_tc,
            "source_match_sec": round(match_start / SR, 6),
            "confidence": round(confidence, 6),
            "status": status,
            "output_preview": str(preview_path),
        }
    )

    print(f"{clip_name} -> {match_start / SR:.6f}s | conf={confidence:.4f} | {status.upper()}")

with open(out_dir / "report.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

if not failed:
    sf.write(str(out_dir / "final.wav"), output, SR)
    print("\nDONE")
    print(out_dir / "final.wav")
else:
    print("\nFAILED")
    