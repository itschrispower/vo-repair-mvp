import librosa
import numpy as np
import soundfile as sf

SR = 48000

CLIPS = [
    ("RECORD_35-04.wav", "2.06"),
    ("RECORD_35-07.wav", "6.03"),
    ("RECORD_35-06.wav", "11.03"),
    ("RECORD_35-02.wav", "18.01"),
]

def tc_to_sec(tc):
    s, f = tc.split(".")
    return int(s) + (int(f) / 25.0)

loaded = []
max_end = 0

for name, comp_tc in CLIPS:
    clip, _ = librosa.load(f"clips/{name}", sr=SR)
    comp_time = tc_to_sec(comp_tc)

    start = int(comp_time * SR)
    end = start + len(clip)

    loaded.append((clip, start, end))
    max_end = max(max_end, end)

output = np.zeros(max_end, dtype=np.float32)

for clip, start, end in loaded:
    output[start:end] = clip

sf.write("rebuild_clean.wav", output, SR)

print("DONE")
