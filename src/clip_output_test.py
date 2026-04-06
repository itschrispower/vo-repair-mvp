import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SR = 48000

job = Path("VO_JOB_01")
vo_path = job / "audio" / "VOBU_48k.wav"
aaf_ref_path = job / "audio" / "aaf_reference.wav"
positions_path = job / "positions.txt"

out_dir = job / "clip_output_test"
clips_dir = out_dir / "clips"
out_dir.mkdir(exist_ok=True)
clips_dir.mkdir(exist_ok=True)


def tc_to_seconds(tc: str) -> float:
    if ":" in tc:
        hh, mm, ss, ff = tc.split(":")
        return int(ss) + (int(ff) / 25.0)
    secs, frames = tc.split(".")
    return int(secs) + (int(frames) / 25.0)


def tc_to_samples(tc: str) -> int:
    return int(tc_to_seconds(tc) * SR)


def norm(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    if x.size == 0:
        return x
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
    matched = full_n[match_start:match_start + len(clip_n)]
    denom = np.linalg.norm(matched) * np.linalg.norm(clip_n)
    confidence = 0.0 if denom == 0 else float(corr[match_start] / denom)
    return match_start, confidence


def load_positions(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, start, end = line.split()
            rows.append((name, start, end))
    return rows


def build_auto_refs(aaf_reference_path: Path, clips, auto_ref_dir: Path):
    audio, sr = sf.read(str(aaf_reference_path))
    if sr != SR:
        raise ValueError(f"aaf_reference.wav must be {SR} Hz, got {sr}")

    auto_ref_dir.mkdir(exist_ok=True)
    refs = {}

    for clip_name, start_tc, end_tc in clips:
        start = tc_to_samples(start_tc)
        end = tc_to_samples(end_tc)
        clip_audio = audio[start:end]
        out_path = auto_ref_dir / f"{clip_name}_auto_ref.wav"
        sf.write(str(out_path), clip_audio, sr)
        refs[clip_name] = out_path

    return refs


vo, vo_sr = librosa.load(str(vo_path), sr=SR)
if vo_sr != SR:
    raise ValueError(f"VO file must be {SR} Hz")

clips = load_positions(positions_path)
auto_refs = build_auto_refs(aaf_ref_path, clips, out_dir / "auto_refs")

manifest = []

for i, (clip_name, start_tc, end_tc) in enumerate(clips, start=1):
    ref_path = auto_refs[clip_name]
    ref_clip, _ = librosa.load(str(ref_path), sr=SR)

    match_start, confidence = match_clip(vo, ref_clip)

    out_start = tc_to_samples(start_tc)
    out_end = tc_to_samples(end_tc)
    target_len = out_end - out_start

    rebuilt_clip = vo[match_start:match_start + target_len]

    if len(rebuilt_clip) < target_len:
        padded = np.zeros(target_len, dtype=np.float32)
        padded[:len(rebuilt_clip)] = rebuilt_clip
        rebuilt_clip = padded

    clip_out = clips_dir / f"{clip_name}_rebuilt.wav"
    sf.write(str(clip_out), rebuilt_clip, SR)

    manifest.append(
        {
            "index": i,
            "clip_name": clip_name,
            "timeline_start_tc": start_tc,
            "timeline_end_tc": end_tc,
            "timeline_start_sec": round(out_start / SR, 6),
            "timeline_end_sec": round(out_end / SR, 6),
            "duration_sec": round(target_len / SR, 6),
            "source_match_sec": round(match_start / SR, 6),
            "confidence": round(confidence, 6),
            "rebuilt_file": str(clip_out),
        }
    )

    print(
        f"{clip_name} -> {clip_out.name} | "
        f"timeline {start_tc}-{end_tc} | "
        f"source {match_start / SR:.6f}s | conf={confidence:.4f}"
    )

with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print("\nDONE")
print(out_dir / "manifest.json")