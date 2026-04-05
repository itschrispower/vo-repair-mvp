from pathlib import Path
import sys
import re

def tc_to_simple(tc):
    parts = tc.split(":")
    sec = int(parts[2])
    frame = int(parts[3].split(".")[0])
    return f"{sec:02d}.{frame:02d}"

def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python3 src/pt_to_positions.py <job_folder>")

    job = Path(sys.argv[1]).resolve()
    txt_file = next(job.glob("*.txt"))
    out_file = job / "positions.txt"

    lines_out = []

    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            if "clip_" not in line:
                continue

            parts = re.split(r"\s+", line.strip())
            if len(parts) < 5:
                continue

            clip_name = parts[2]
            start_tc = parts[3]
            end_tc = parts[4]

            if not clip_name.startswith("clip_"):
                continue

            simple_start = tc_to_simple(start_tc)
            simple_end = tc_to_simple(end_tc)

            ref_name = f"{clip_name}.wav"
            lines_out.append(f"{ref_name} {simple_start} {simple_end}")

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out) + "\n")

    print(f"Wrote {out_file}")

if __name__ == "__main__":
    main()