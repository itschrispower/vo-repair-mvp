import librosa
import numpy as np
import soundfile as sf
from pathlib import Path

SR = 48000

BASE = Path(__file__).resolve().parent.parent

VO_PATH = BASE / "audio" / "VOBU_48k.wav"
REF_DIR = BASE / "clips"
OUT_DIR = BASE / "rebuild_audio"
POSITIONS_PATH = BASE / "positions.txt"

FINAL_PATH = OUT_DIR / "final.wav"


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


def match_clip(full: np.ndarray, clip: np.ndarray) -> int:
    full_n = norm(full)
    clip_n = norm(clip)
    corr = np.correlate(full_n, clip_n, mode="valid")
    return int(np.argmax(corr))


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


def main():
    vo, _ = librosa.load(str(VO_PATH), sr=SR)
    clips = load_positions(POSITIONS_PATH)

    OUT_DIR.mkdir(exist_ok=True)

    max_end = max(tc_to_samples(end_tc) for _, _, end_tc in clips)
    output = np.zeros(max_end, dtype=np.float32)

    for ref_name, start_tc, end_tc in clips:
        ref_path = REF_DIR / ref_name
        ref_clip, _ = librosa.load(str(ref_path), sr=SR)

        match_start = match_clip(vo, ref_clip)

        out_start = tc_to_samples(start_tc)
        out_end = tc_to_samples(end_tc)
        target_len = out_end - out_start

        vo_clip = vo[match_start:match_start + target_len]

        if len(vo_clip) < target_len:
            padded = np.zeros(target_len, dtype=np.float32)
            padded[:len(vo_clip)] = vo_clip
            vo_clip = padded

        output[out_start:out_end] = vo_clip[:target_len]

        print(f"{ref_name} -> {match_start / SR:.6f}s -> placed at {start_tc}-{end_tc}")

    sf.write(str(FINAL_PATH), output, SR)

    print("DONE")
    print(FINAL_PATH)


if __name__ == "__main__":
    main()