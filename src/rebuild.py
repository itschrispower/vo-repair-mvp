import librosa
import soundfile as sf
import numpy as np

SLP_PATH = "the VO’s continuous SLP backup-01.wav"
CLIPS = [
    ("RECORD_35-04.wav", 3.17075),
    ("RECORD_35-07.wav", 5.472375),
    ("RECORD_35-06.wav", 6.548875),
    ("RECORD_35-02.wav", 6.613125),
]

sr = 48000

slp, _ = librosa.load(SLP_PATH, sr=sr)
output = np.zeros_like(slp)

def trim_audio(y):
    thresh = 0.02
    idx = np.where(np.abs(y) > thresh)[0]
    if len(idx) == 0:
        return y
    return y[idx[0]:idx[-1]]

for name, t in CLIPS:
    y, _ = librosa.load(f"clips/{name}", sr=sr)
    y = trim_audio(y)

    start = int(t * sr)
    end = start + len(y)

    output[start:end] += y

sf.write("rebuild_timed.wav", output, sr)
print("DONE → rebuild_timed.wav")
