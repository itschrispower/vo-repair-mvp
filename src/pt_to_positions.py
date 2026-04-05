from pathlib import Path
import sys
import re


def tc_to_simple(tc: str) -> str:
    parts = tc.split(":")
    sec = int(parts[2])
    frame = int(parts[3].split(".")[0])
    return f"{sec:02d}.{frame:02d}"


def find_session_info_txt(job: Path) -> Path:
    candidates = sorted(
        p for p in job.glob("*.txt")
        if p.name.lower() != "positions.txt"
    )

    if not candidates:
        raise FileNotFoundError(f"No Pro Tools text export found in {job}")

    if len(candidates) > 1:
        raise ValueError(
            "Expected exactly one Pro Tools text export, found: "
            + ", ".join(p.name for p in candidates)
        )

    return candidates[0]


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python3 src/pt_to_positions.py <job_folder>")

    job = Path(sys.argv[1]).resolve()
    txt_file = find_session_info_txt(job)
    out_file = job / "positions.txt"

    lines_out = []
    in_comp_track = False

    with open(txt_file, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            stripped = line.strip()

            if stripped.startswith("TRACK NAME:"):
                track_name = stripped.split("TRACK NAME:", 1)[1].strip()
                in_comp_track = (track_name == "COMP")
                continue

            if not in_comp_track:
                continue

            if not stripped:
                continue

            if stripped.startswith(("COMMENTS:", "USER DELAY:", "STATE:", "PLUG-INS:")):
                continue

            if "CHANNEL" in stripped and "EVENT" in stripped and "CLIP NAME" in stripped:
                continue

            parts = re.split(r"\s+", stripped)
            if len(parts) < 6:
                continue

            if not parts[0].isdigit() or not parts[1].isdigit():
                continue

            clip_name = parts[2]
            start_tc = parts[3]
            end_tc = parts[4]

            lines_out.append(
                f"{clip_name} {tc_to_simple(start_tc)} {tc_to_simple(end_tc)}"
            )

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out) + ("\n" if lines_out else ""))

    print(f"Using: {txt_file.name}")
    print(f"Wrote {out_file}")
    print(f"Lines: {len(lines_out)}")


if __name__ == "__main__":
    main()