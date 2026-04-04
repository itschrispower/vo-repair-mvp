import librosa
import numpy as np
import os

SLP_PATH = "the VO’s continuous SLP backup-01.wav"
CLIPS_DIR = "clips"

slp, sr = librosa.load(SLP_PATH, sr=None)
target_sr = 8000
slp = librosa.resample(slp, orig_sr=sr, target_sr=target_sr)
slp = slp / np.max(np.abs(slp))

def match_clip(path):
    test, sr2 = librosa.load(path, sr=None)
    test = librosa.resample(test, orig_sr=sr2, target_sr=target_sr)
    test = test / np.max(np.abs(test))

    corr = np.correlate(slp, test, mode='valid')
    pos = np.argmax(corr)
    return pos / target_sr

results = []

for f in os.listdir(CLIPS_DIR):
    if f.endswith(".wav"):
        path = os.path.join(CLIPS_DIR, f)
        t = match_clip(path)
        results.append((f, t))

# sort by time
results.sort(key=lambda x: x[1])

for f, t in results:
    print(f"{f} → {t} sec")
