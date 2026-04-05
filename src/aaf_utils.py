import aaf2
from pathlib import Path


def extract_aaf_clips(aaf_path: str):
    clips = []

    with aaf2.open(aaf_path, "r") as f:
        for mob in f.content.mobs:
            if getattr(mob, "usage", None) != "Usage_TopLevel":
                continue

            for slot in mob.slots:
                segment = slot.segment

                if not hasattr(segment, "components"):
                    continue

                pos = 0

                for comp in segment.components:
                    length = int(getattr(comp, "length", 0))

                    if comp.__class__.__name__ == "SourceClip":
                        clips.append({
                            "start_units": pos,
                            "length_units": length,
                        })

                    pos += length

    return clips