"""
src/pt_to_positions.py

Extract clip positions from a Pro Tools AAF export and write positions.txt.

Output format (one clip per line):
    {clip_name} {start_tc} {end_tc}

Comment lines (starting with #) precede each clip and carry optional metadata:
    # type=regular source_start_sec=5.520
    # type=crossfade
    # type=regular source_start_sec=12.840

These comment lines are safely ignored by load_positions() in engine.py but
can be used for diagnostic purposes or by future pipeline stages.
"""
from pathlib import Path
import sys
import math
from typing import Optional

import aaf2

from utils import FPS, find_single_aaf

AAF_FPS = FPS


def units_to_seconds(units: int, edit_rate: float) -> float:
    return float(units) / float(edit_rate)


def seconds_to_simple_tc(seconds: float) -> str:
    total_frames = int(math.floor(seconds * AAF_FPS + 1e-9))
    fps_int = int(round(AAF_FPS))
    secs = total_frames // fps_int
    frames = total_frames % fps_int
    return f"{secs:02d}.{frames:02d}"


def get_edit_rate(slot) -> float:
    try:
        rate = slot.edit_rate
        if isinstance(rate, str):
            if "/" in rate:
                num, den = rate.split("/")
                return float(num) / float(den)
            return float(rate)
        if isinstance(rate, (tuple, list)) and len(rate) >= 2:
            return float(rate[0]) / float(rate[1])
        return float(rate)
    except Exception:
        return 48000.0


def _find_source_clips(comp) -> list:
    """
    Recursively locate SourceClip objects within any component.
    Handles OperationGroup → InputSegment → SourceClip nesting used by Pro Tools.
    """
    name = comp.__class__.__name__
    if name == "SourceClip":
        return [comp]

    found = []
    # OperationGroup.segments or similar child iterables
    for attr in ("segments", "components", "value"):
        children = getattr(comp, attr, None)
        if children is None:
            continue
        # Some AAF properties return a single object rather than an iterable
        if not hasattr(children, "__iter__"):
            children = [children]
        for child in children:
            try:
                found.extend(_find_source_clips(child))
            except Exception:
                pass
    return found


def is_real_clip_component(comp) -> bool:
    """
    Return True for components that represent actual audio content
    (not silence / filler).
    """
    name = comp.__class__.__name__

    if name == "SourceClip":
        return True

    if name == "OperationGroup":
        # Crossfade / gain group — real if it contains any source clips OR
        # has positive length (fallback for unusual AAF structures).
        if _find_source_clips(comp):
            return True
        return getattr(comp, "length", 0) > 1

    return False


def is_crossfade(comp) -> bool:
    """Return True if this component is an OperationGroup (likely a crossfade)."""
    return comp.__class__.__name__ == "OperationGroup"


def get_clip_name(comp, fallback_index: int) -> str:
    try:
        if hasattr(comp, "name") and comp.name:
            return str(comp.name)
    except Exception:
        pass

    try:
        if comp.__class__.__name__ == "SourceClip":
            mob = comp.resolve_ref()
            if mob and getattr(mob, "name", None):
                return str(mob.name)
    except Exception:
        pass

    # For OperationGroups (crossfades), try to get name from first inner clip
    try:
        inner = _find_source_clips(comp)
        if inner:
            mob = inner[0].resolve_ref()
            if mob and getattr(mob, "name", None):
                return str(mob.name)
    except Exception:
        pass

    return str(fallback_index)


def get_source_start_sec(comp, edit_rate: float) -> Optional[float]:
    """
    Return the source-file start position in seconds for a SourceClip,
    or None if unavailable / not applicable.

    Note: in a typical Pro Tools AAF this references the engineer's recording
    file (not the VOBU), but the value is logged for diagnostic use.
    """
    try:
        if comp.__class__.__name__ == "SourceClip":
            start = getattr(comp, "start", None)
            if start is not None:
                return units_to_seconds(int(start), edit_rate)
    except Exception:
        pass

    # For OperationGroup, try from the first inner SourceClip
    try:
        inner = _find_source_clips(comp)
        if inner:
            start = getattr(inner[0], "start", None)
            if start is not None:
                return units_to_seconds(int(start), edit_rate)
    except Exception:
        pass

    return None


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python3 src/pt_to_positions.py <job_folder>")

    job = Path(sys.argv[1]).resolve()
    aaf_path = find_single_aaf(job)
    out_file = job / "positions.txt"

    lines_out = []
    clip_count = 0
    xfade_count = 0

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
                edit_rate = get_edit_rate(slot)
                break

        if seq is None:
            raise ValueError("No Sequence found on top-level mob")

        pos = 0
        clip_index = 0

        for comp in seq.components:
            length = int(getattr(comp, "length", 0))
            if length <= 0:
                pos += length
                continue

            if is_real_clip_component(comp):
                clip_index += 1

                start_sec = units_to_seconds(pos, edit_rate)
                end_sec = units_to_seconds(pos + length, edit_rate)

                start_tc = seconds_to_simple_tc(start_sec)
                end_tc = seconds_to_simple_tc(end_sec)

                clip_name = get_clip_name(comp, clip_index)
                xfade = is_crossfade(comp)

                # Build metadata comment
                meta_parts = [f"type={'crossfade' if xfade else 'regular'}"]
                src_sec = get_source_start_sec(comp, edit_rate)
                if src_sec is not None:
                    meta_parts.append(f"source_start_sec={src_sec:.6f}")

                lines_out.append(f"# {' '.join(meta_parts)}")
                lines_out.append(f"{clip_name} {start_tc} {end_tc}")

                clip_count += 1
                if xfade:
                    xfade_count += 1

            pos += length

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out) + ("\n" if lines_out else ""))

    print(f"Using: {aaf_path.name}")
    print(f"Wrote {out_file}")
    print(f"Clips: {clip_count} ({xfade_count} crossfade regions)")


if __name__ == "__main__":
    main()
