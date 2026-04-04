from audio_utils import load_wav
from match_utils import find_best_match
from aaf_utils import extract_aaf_clips_linked

import os
import json
import wave
import numpy as np

AAF_PATH = "3.aaf"
AUDIO_FOLDER = "Audio Files"
VO_BACKUP = "the VO’s continuous SLP backup-01.wav"
HANDLE_SAMPLES = 48000 * 2


def write_wav(path, samples, sample_rate):
    samples_int16 = np.clip(samples * 32767, -32768, 32767).astype("<i2")
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(samples_int16.tobytes())


def main():
    print("ENGINE STARTED")

    vo = load_wav(VO_BACKUP)

    print("Loading AAF...")
    clips = extract_aaf_clips_linked(AAF_PATH, AUDIO_FOLDER)
    print(f"Clips found: {len(clips)}")

    os.makedirs("rebuild_audio", exist_ok=True)

    rebuilt_clips = []
    report = []

    for i, clip in enumerate(clips):
        print(f"Processing clip_{i}")

        source_path = clip["source_file"]

        if not source_path:
            print("No source file, skipping")
            continue

        if not os.path.exists(source_path):
            print(f"Missing source file: {source_path}")
            continue

        ref = load_wav(source_path)

        start = clip["source_start"]
        duration = clip["duration"]

        if duration < 2000:
            print("Skipping tiny fragment")
            continue

        ref_chunk = ref["samples"][start:start + duration]

        if len(ref_chunk) == 0:
            print("Empty clip, skipping")
            continue

        energy = np.mean(ref_chunk ** 2)
        if energy < 1e-6:
            print("Skipping silent clip")
            continue

        result = find_best_match(
            ref_chunk,
            vo["samples"],
            vo["sample_rate"],
            expected_start=None
        )

        if result is None:
            print("No match, skipping")
            continue

        if result["confidence"] < 0.5:
            print("Weak match, skipping")
            continue

        match_start = result["match_sample"]

        extract_start = max(0, match_start - HANDLE_SAMPLES)
        extract_end = min(len(vo["samples"]), match_start + duration + HANDLE_SAMPLES)

        extracted = vo["samples"][extract_start:extract_end]

        out_name = f"rebuild_audio/clip_{i}.wav"
        write_wav(out_name, extracted, vo["sample_rate"])

        rebuilt_clips.append({
            "file": out_name,
            "timeline_start": clip["timeline_start"],
            "duration": duration,
            "handle_offset": match_start - extract_start
        })

        report.append({
            "clip": i,
            "confidence": result["confidence"]
        })

    with open("rebuild_map.json", "w") as f:
        json.dump(rebuilt_clips, f, indent=2)

    with open("report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("DONE → rebuild_audio + rebuild_map.json")


if __name__ == "__main__":
    main()