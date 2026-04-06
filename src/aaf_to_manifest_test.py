import json
from pathlib import Path

import aaf2

AAF_EDIT_RATE = 48000.0

job = Path("VO_JOB_01")
aaf_path = next(job.glob("*.aaf"))
out_dir = job / "aaf_to_manifest_test"
out_dir.mkdir(exist_ok=True)

manifest_path = out_dir / "manifest_from_aaf.json"


def units_to_seconds(units: int) -> float:
    return round(units / AAF_EDIT_RATE, 6)


def units_to_tc(units: int) -> str:
    total_seconds = int(units // AAF_EDIT_RATE)
    frames = int(units % AAF_EDIT_RATE)
    return f"{total_seconds:02d}.{frames:02d}"


manifest = []

with aaf2.open(str(aaf_path), "r") as f:
    top = None

    for mob in f.content.mobs:
        if getattr(mob, "usage", None) == "Usage_TopLevel":
            top = mob
            break

    if top is None:
        raise ValueError("No Usage_TopLevel mob found")

    seq = None
    for slot in getattr(top, "slots", []):
        seg = slot.segment
        if seg.__class__.__name__ == "Sequence":
            seq = seg
            break

    if seq is None:
        raise ValueError("No Sequence found on top-level mob")

    pos = 0
    clip_index = 0

    for comp in seq.components:
        length = int(getattr(comp, "length", 0))
        comp_type = comp.__class__.__name__

        if comp_type == "OperationGroup" and length > 1:
            clip_index += 1

            start_units = pos
            end_units = pos + length

            manifest.append(
                {
                    "index": clip_index,
                    "clip_name": str(clip_index),
                    "timeline": {
                        "start_tc": units_to_tc(start_units),
                        "end_tc": units_to_tc(end_units),
                        "start_sec": units_to_seconds(start_units),
                        "end_sec": units_to_seconds(end_units),
                        "duration_sec": units_to_seconds(length),
                    },
                    "aaf": {
                        "component_type": comp_type,
                        "length_units": length,
                    },
                }
            )

        pos += length

manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

print("DONE")
print(manifest_path)
print("")
print(json.dumps(manifest, indent=2))