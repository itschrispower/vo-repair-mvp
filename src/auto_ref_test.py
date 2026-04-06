from pathlib import Path
import re
import soundfile as sf

SR = 48000

job = Path("VO_JOB_01")
txt = job / "session_info.txt"
aaf_ref = job / "audio" / "aaf_reference.wav"
out_dir = job / "auto_refs"

out_dir.mkdir(exist_ok=True)

audio, sr = sf.read(str(aaf_ref))
if sr != SR:
    raise ValueError(f"aaf_reference.wav must be {SR} Hz, got {sr}")

in_comp = False
rows = []

def tc_to_samples(tc: str) -> int:
    hh, mm, ss, ff = tc.split(":")
    return int((int(ss) + int(ff) / 25.0) * SR)

with open(txt, "r", encoding="utf-8") as f:
    for raw in f:
        s = raw.strip()

        if s.startswith("TRACK NAME:"):
            in_comp = (s.split("TRACK NAME:", 1)[1].strip() == "COMP")
            continue

        if not in_comp or not s:
            continue

        if s.startswith(("COMMENTS:", "USER DELAY:", "STATE:", "PLUG-INS:")):
            continue

        if "CHANNEL" in s and "EVENT" in s and "CLIP NAME" in s:
            continue

        parts = re.split(r"\s+", s)
        if len(parts) < 6:
            continue

        if not parts[0].isdigit() or not parts[1].isdigit():
            continue

        clip_name = parts[2]
        start_tc = parts[3]
        end_tc = parts[4]

        rows.append((clip_name, start_tc, end_tc))

print("\nWRITING AUTO REFS:\n")

for clip_name, start_tc, end_tc in rows:
    start = tc_to_samples(start_tc)
    end = tc_to_samples(end_tc)

    clip_audio = audio[start:end]

    out_path = out_dir / f"{clip_name}_auto_ref.wav"
    sf.write(str(out_path), clip_audio, sr)

    print(f"{out_path.name}  {start_tc} -> {end_tc}")