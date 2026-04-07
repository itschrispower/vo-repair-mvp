from pathlib import Path
import sys
import math

import aaf2

from utils import FPS, find_single_aaf

AAF_FPS = FPS


def units_to_seconds(units: int, edit_rate: float) -> float:
    return units / edit_rate


def seconds_to_simple_tc(seconds: float) -> str:
    total_frames = int(math.floor(seconds * AAF_FPS + 1e-9))
    secs = total_frames // 25
    frames = total_frames % 25
    return f"{secs:02d}.{frames:02d}"


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python3 src/pt_to_positions.py <job_folder>")

    job = Path(sys.argv[1]).resolve()
    aaf_path = find_single_aaf(job)
    out_file = job / "positions.txt"

    lines_out = []

    with aaf2.open(str(aaf_path), "r") as f:
        top = None
        for mob in f.content.mobs:
            if getattr(mob, "usage", None) == "Usage_TopLevel":
                top = mob
                break

        if top is None:
            raise ValueError("No Usage_TopLevel mob found in AAF")

        seq = None
        edit_rate = None

        for slot in getattr(top, "slots", []):
            seg = slot.segment
            if seg.__class__.__name__ == "Sequence":
                seq = seg
                try:
                    rate = slot.edit_rate
                    if isinstance(rate, str):
                        edit_rate = float(rate.split("/")[0])
                    elif isinstance(rate, (tuple, list)) and len(rate) >= 2:
                        edit_rate = float(rate[0]) / float(rate[1])
                    else:
                        edit_rate = float(rate)
                except Exception:
                    edit_rate = 48000.0
                break

        if seq is None:
            raise ValueError("No Sequence found on top-level mob")
        if not edit_rate:
            edit_rate = 48000.0

        pos = 0
        clip_index = 0

        for comp in seq.components:
            length = int(getattr(comp, "length", 0))
            comp_type = comp.__class__.__name__

            if comp_type == "OperationGroup" and length > 1:
                clip_index += 1

                start_sec = units_to_seconds(pos, edit_rate)
                end_sec = units_to_seconds(pos + length, edit_rate)

                start_tc = seconds_to_simple_tc(start_sec)
                end_tc = seconds_to_simple_tc(end_sec)

                lines_out.append(f"{clip_index} {start_tc} {end_tc}")

            pos += length

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out) + ("\n" if lines_out else ""))

    print(f"Using: {aaf_path.name}")
    print(f"Wrote {out_file}")
    print(f"Lines: {len(lines_out)}")


if __name__ == "__main__":
    main()