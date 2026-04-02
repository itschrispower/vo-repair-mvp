from __future__ import annotations

from pathlib import Path

import aaf2


def validate_aaf_file(aaf_path: str) -> tuple[bool, str]:
    path = Path(aaf_path)

    if not path.exists():
        return False, "AAF file path does not exist."

    if path.suffix.lower() != ".aaf":
        return False, "File must use the .aaf extension."

    return True, "AAF looks valid."


def _all_audio_files(audio_folder: str) -> list[Path]:
    folder = Path(audio_folder)
    if not folder.exists():
        return []
    return [p for p in folder.iterdir() if p.is_file()]


def _resolve_linked_audio_path(audio_folder: str, mob_name: str | None) -> str | None:
    files = _all_audio_files(audio_folder)
    if not files:
        return None

    if len(files) == 1:
        return str(files[0])

    if not mob_name:
        return None

    exact = Path(audio_folder) / mob_name
    if exact.exists():
        return str(exact)

    exact_wav = Path(audio_folder) / f"{mob_name}.wav"
    if exact_wav.exists():
        return str(exact_wav)

    stem = Path(mob_name).stem.lower()
    for f in files:
        if f.stem.lower() == stem:
            return str(f)

    for f in files:
        if stem in f.stem.lower() or f.stem.lower() in stem:
            return str(f)

    return None


def extract_aaf_clips_linked(aaf_path: str, audio_folder: str) -> list[dict]:
    clips: list[dict] = []

    with aaf2.open(aaf_path, "r") as f:
        for mob in f.content.mobs:
            if getattr(mob, "usage", None) == "Usage_TopLevel":
                for slot in mob.slots:
                    segment = slot.segment

                    if not hasattr(segment, "components"):
                        continue

                    timeline_pos = 0

                    for comp in segment.components:
                        length = int(getattr(comp, "length", 0))

                        if comp.__class__.__name__ == "SourceClip":
                            source_start = int(getattr(comp, "start_time", 0))
                            source_mob = comp.mob
                            mob_name = getattr(source_mob, "name", None)
                            file_path = _resolve_linked_audio_path(audio_folder, mob_name)

                            clips.append(
                                {
                                    "id": f"clip_{len(clips)}",
                                    "source_file": file_path,
                                    "mob_name": mob_name,
                                    "source_start": source_start,
                                    "duration": length,
                                    "timeline_start": timeline_pos,
                                    "fade_in": 0,
                                    "fade_out": 0,
                                }
                            )

                        timeline_pos += length

    return clips