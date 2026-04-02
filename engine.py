from __future__ import annotations

import json
import os
import wave

import numpy as np

from aaf_utils import extract_aaf_clips_linked, validate_aaf_file
from audio_utils import load_wav
from match_utils import find_best_match


def apply_fades(audio: np.ndarray, fade_in: int, fade_out: int) -> np.ndarray:
    audio = audio.copy()

    if fade_in > 0 and fade_in < len(audio):
        fade = np.linspace(0.0, 1.0, fade_in)
        audio[:fade_in] *= fade

    if fade_out > 0 and fade_out < len(audio):
        fade = np.linspace(1.0, 0.0, fade_out)
        audio[-fade_out:] *= fade

    return audio


def main():
    AAF_PATH = "3.aaf"
    AUDIO_FOLDER = "Audio Files"
    VO_WAV = "the VO’s continuous SLP backup-01.wav"

    ok, msg = validate_aaf_file(AAF_PATH)
    print(msg)
    if not ok:
        return

    vo = load_wav(VO_WAV)
    vo_sr = vo["sample_rate"]

    clips = extract_aaf_clips_linked(AAF_PATH, AUDIO_FOLDER)

    if not clips:
        print("No clips found in AAF")
        return

    print(f"Found {len(clips)} clips")

    end_points = [c["timeline_start"] + c["duration"] for c in clips]
    total_length = max(end_points) + 1000

    output = np.zeros(total_length, dtype=np.float32)
    report = []
    source_cache: dict[str, dict] = {}

    for clip in clips:
        print(f"Processing {clip['id']}")

        source_file = clip["source_file"]

        if not source_file:
            print(f"Skipping {clip['id']} (no resolved source file for mob '{clip['mob_name']}')")
            continue

        if not os.path.exists(source_file):
            print(f"Skipping {clip['id']} (missing source file: {source_file})")
            continue

        if source_file not in source_cache:
            source_cache[source_file] = load_wav(source_file)

        ref_audio = source_cache[source_file]

        if ref_audio["sample_rate"] != vo_sr:
            print(
                f"Skipping {clip['id']} (sample rate mismatch: "
                f"{ref_audio['sample_rate']} vs {vo_sr})"
            )
            continue

        ref_clip = ref_audio["samples"][
            clip["source_start"] : clip["source_start"] + clip["duration"]
        ]

        if len(ref_clip) == 0:
            print(f"Skipping {clip['id']} (empty ref clip)")
            continue

        result = find_best_match(
            ref_clip,
            vo["samples"],
            vo_sr,
        )

        match_start = result["match_sample"]
        vo_clip = vo["samples"][match_start : match_start + clip["duration"]]

        if len(vo_clip) == 0:
            print(f"Skipping {clip['id']} (empty VO clip)")
            continue

        vo_clip = apply_fades(
            vo_clip,
            clip["fade_in"],
            clip["fade_out"],
        )

        start = clip["timeline_start"]
        end = start + len(vo_clip)
        output[start:end] += vo_clip

        report.append(
            {
                "id": clip["id"],
                "mob_name": clip["mob_name"],
                "source_file": source_file,
                "source_start": clip["source_start"],
                "timeline_start": clip["timeline_start"],
                "duration": clip["duration"],
                "match_sample": int(match_start),
                "confidence": float(result["confidence"]),
            }
        )

    output = np.clip(output, -1.0, 1.0)
    output_int16 = np.int16(output * 32767)

    with wave.open("output.wav", "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(vo_sr)
        wf.writeframes(output_int16.tobytes())

    with open("report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("DONE → output.wav + report.json")


if __name__ == "__main__":
    main()