from pathlib import Path
import re

job = Path("VO_JOB_01")
txt = job / "session_info.txt"

in_comp = False
rows = []

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
        dur_tc = parts[5]

        rows.append((clip_name, start_tc, end_tc, dur_tc))

print("\nCOMP CLIPS:\n")
for row in rows:
    print(row)