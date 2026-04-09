"""
src/pt_to_positions.py

Extract clip positions from a Pro Tools AAF export and write positions.txt.

Output format (one clip per line):
    {clip_name} {start_tc} {end_tc}

Comment lines (starting with #) precede each clip and carry optional metadata:
    # type=regular source_start_sec=5.520

These comment lines are safely ignored by load_positions() in engine.py but
can be used for diagnostic purposes or by future pipeline stages.

── OperationGroup handling ───────────────────────────────────────────────────
Pro Tools wraps every placed clip in an OperationGroup (for fade, clip-gain,
or AudioSuite metadata).  When two clips share a crossfade, Pro Tools inserts
a *second* OperationGroup into the sequence whose two InputSegments point to
the outgoing and incoming clips respectively.

Detection: a crossfade OperationGroup has exactly 2 InputSegments.  A
regular clip-wrapping OperationGroup (fade, gain, AudioSuite) has exactly 1.

── Placed-duration reading ───────────────────────────────────────────────────
For a single-InputSegment OperationGroup, OperationGroup.length can diverge
from the true placed duration.  Reading the InputSegment's own .length is
more reliable when it is positive and shorter than comp.length.

── Diagnostic mode ──────────────────────────────────────────────────────────
Pass --diag as a second argument to print a full per-component dump of the
AAF sequence structure.  This is the primary tool for debugging crossfade AAFs:

    python3 src/pt_to_positions.py <job_folder> --diag

The dump shows class, seq_length, input_segment count (and every access
attempt), operation name/AUID, placed_len, and emit/skip decision for every
component in the top-level sequence.
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

    For OperationGroups, uses comp['InputSegments'] (dictionary-style) first
    because comp.input_segments raises AttributeError on this aaf2 version.
    Falls back to .segments, .components, .value for other component types.
    """
    if comp.__class__.__name__ == "SourceClip":
        return [comp]

    found = []

    # OperationGroup: use dictionary-style access — comp.input_segments is
    # not a valid attribute on this aaf2 version (raises AttributeError).
    if comp.__class__.__name__ == "OperationGroup":
        try:
            for seg in comp["InputSegments"]:
                found.extend(_find_source_clips(seg))
            if found:
                return found
        except Exception:
            pass

    # General fallback for Sequence, Selector, and other container types
    for attr in ("segments", "components", "value"):
        children = getattr(comp, attr, None)
        if children is None:
            continue
        if not hasattr(children, "__iter__"):
            children = [children]
        for child in children:
            try:
                found.extend(_find_source_clips(child))
            except Exception:
                pass
        if found:
            return found

    return found


def _probe_input_segments(comp) -> tuple:
    """
    Try every known way to access the InputSegments of an OperationGroup.

    Returns (count, method_used, error_notes) where:
      count       — int number of input segments found (0 if none/error)
      method_used — short string describing how count was obtained
      error_notes — list of strings describing access attempts and exceptions

    This is used by both the live code path (for xfade detection) and the
    diagnostic dump (for understanding AAF structure).
    """
    notes = []

    # ── Method 1: comp.input_segments iterable ────────────────────────────
    try:
        segs = list(comp.input_segments)
        return len(segs), "input_segments", notes
    except AttributeError as e:
        notes.append(f"input_segments AttributeError: {e}")
    except TypeError as e:
        notes.append(f"input_segments TypeError (not iterable?): {e}")
    except Exception as e:
        notes.append(f"input_segments Exception: {type(e).__name__}: {e}")

    # ── Method 2: comp['InputSegments'] dictionary-style access ───────────
    try:
        segs = list(comp["InputSegments"])
        notes.append("input_segments failed; used comp['InputSegments']")
        return len(segs), "comp['InputSegments']", notes
    except Exception as e:
        notes.append(f"comp['InputSegments'] Exception: {type(e).__name__}: {e}")

    # ── Method 3: comp.getvalue('InputSegments') ──────────────────────────
    try:
        segs = list(comp.getvalue("InputSegments"))
        notes.append("used comp.getvalue('InputSegments')")
        return len(segs), "getvalue('InputSegments')", notes
    except Exception as e:
        notes.append(f"getvalue('InputSegments') Exception: {type(e).__name__}: {e}")

    # ── Method 4: operation name / AUID heuristic ─────────────────────────
    try:
        op_name = str(getattr(comp.operation, "name", "") or "").lower()
        if any(x in op_name for x in ("xfade", "crossfade", "audiocrossfade")):
            notes.append(f"heuristic: op_name='{op_name}' → count=2")
            return 2, "op_name_heuristic", notes
        notes.append(f"op_name='{op_name}' (no xfade keyword)")
    except Exception as e:
        notes.append(f"comp.operation access Exception: {type(e).__name__}: {e}")

    notes.append("all methods failed → defaulting to count=1")
    return 1, "fallback_default", notes


def _input_segment_count(comp) -> int:
    """Return number of InputSegments; uses _probe_input_segments internally."""
    if comp.__class__.__name__ != "OperationGroup":
        return 0
    count, _, _ = _probe_input_segments(comp)
    return count


def is_xfade_op(comp) -> bool:
    """Return True if this OperationGroup is a crossfade (2 InputSegments)."""
    return comp.__class__.__name__ == "OperationGroup" and _input_segment_count(comp) == 2


def is_real_clip_component(comp) -> bool:
    """
    Return True for components representing a single independent placed clip.
    Crossfade OperationGroups (2 InputSegments) are excluded.
    """
    name = comp.__class__.__name__

    if name == "SourceClip":
        return True

    if name == "OperationGroup":
        if _input_segment_count(comp) == 2:
            return False
        if _find_source_clips(comp):
            return True
        return getattr(comp, "length", 0) > 1

    return False


def get_placed_length(comp) -> tuple:
    """
    Return (placed_len, source_description) where placed_len is the
    component's placed duration in edit-rate units and source_description
    explains where the value came from.

    For single-InputSegment OperationGroups, prefers InputSegment.length
    when it is positive and shorter than comp.length.
    """
    raw = int(getattr(comp, "length", 0))

    if comp.__class__.__name__ != "OperationGroup":
        return raw, "comp.length(SourceClip)"

    # Use comp['InputSegments'] — comp.input_segments raises AttributeError
    # on this aaf2 version.
    try:
        segs = list(comp["InputSegments"])
        if len(segs) == 1:
            seg_len = int(getattr(segs[0], "length", 0))
            if 0 < seg_len < raw:
                return seg_len, f"InputSegment[0].length={seg_len} (shorter than comp.length={raw})"
            if seg_len > 0:
                return seg_len, f"InputSegment[0].length={seg_len} (== or > comp.length={raw})"
    except Exception as e:
        return raw, f"comp.length={raw} (InputSegment access failed: {e})"

    return raw, f"comp.length={raw} (no InputSegments found)"


def _probe_operation_info(comp) -> dict:
    """
    Extract operation name, AUID, and any other identifiers from an
    OperationGroup for diagnostic display.
    """
    info = {}
    for attr in ("name", "auid", "description"):
        try:
            val = getattr(comp.operation, attr, None)
            if val is not None:
                info[f"op.{attr}"] = str(val)
        except Exception as e:
            info[f"op.{attr}_err"] = str(e)

    # Some aaf2 versions store the operation def differently
    try:
        op_id = str(comp.operation)
        if op_id and op_id not in info.values():
            info["op.str"] = op_id[:80]
    except Exception:
        pass

    return info


def _diag_dump(seq, edit_rate: float, aaf_path: Path) -> None:
    """
    Print a full per-component diagnostic dump of the top-level sequence.
    Activated by passing --diag on the command line.
    """
    print()
    print("=" * 72)
    print("DIAGNOSTIC DUMP — AAF sequence component analysis")
    print(f"  File:      {aaf_path.name}")
    print(f"  Edit rate: {edit_rate}")
    print("=" * 72)

    pos = 0
    for idx, comp in enumerate(seq.components):
        cls   = comp.__class__.__name__
        seq_l = int(getattr(comp, "length", 0))
        sec_s = units_to_seconds(pos, edit_rate)
        sec_e = units_to_seconds(pos + max(0, seq_l), edit_rate)

        print(f"\ncomp[{idx:02d}]  class={cls}  seq_len={seq_l}  "
              f"pos={pos} ({sec_s:.3f}s → {sec_e:.3f}s)")

        if seq_l <= 0:
            print(f"  → SKIP  reason=zero_or_negative_length")
            pos += seq_l
            continue

        if cls == "Filler":
            print(f"  → SKIP  reason=Filler")
            pos += seq_l
            continue

        if cls == "OperationGroup":
            # Show operation identity
            op_info = _probe_operation_info(comp)
            for k, v in op_info.items():
                print(f"  {k} = {v}")

            # Show InputSegment probe results
            count, method, notes = _probe_input_segments(comp)
            print(f"  input_seg_count={count}  (via {method})")
            for note in notes:
                print(f"    note: {note}")

            # Show each InputSegment if accessible
            try:
                segs = list(comp["InputSegments"])
                for si, seg in enumerate(segs):
                    seg_cls = seg.__class__.__name__
                    seg_len = getattr(seg, "length", "?")
                    inner   = _find_source_clips(seg)
                    mob_names = []
                    for sc in inner:
                        try:
                            m = sc.resolve_ref()
                            mob_names.append(getattr(m, "name", "?") or "?")
                        except Exception:
                            mob_names.append("<unresolvable>")
                    print(f"  InputSegment[{si}]  class={seg_cls}  "
                          f"length={seg_len}  "
                          f"inner_mobs={mob_names}")
            except Exception as e:
                print(f"  InputSegment iteration failed: {e}")

            # Decision
            if count == 2:
                print(f"  → SKIP  reason=is_xfade_op (2 InputSegments)")
            elif _find_source_clips(comp) or seq_l > 1:
                placed_len, placed_src = get_placed_length(comp)
                clip_name_val = get_clip_name(comp, idx)
                tc_s = seconds_to_simple_tc(units_to_seconds(pos, edit_rate))
                tc_e = seconds_to_simple_tc(units_to_seconds(pos + placed_len, edit_rate))
                print(f"  placed_len={placed_len}  source='{placed_src}'")
                print(f"  clip_name='{clip_name_val}'  tc={tc_s}→{tc_e}")
                print(f"  → EMIT  clip")
            else:
                print(f"  → SKIP  reason=no_inner_SourceClips")

        elif cls == "SourceClip":
            mob_name_val = "?"
            try:
                m = comp.resolve_ref()
                mob_name_val = getattr(m, "name", "?") or "?"
            except Exception:
                pass
            tc_s = seconds_to_simple_tc(units_to_seconds(pos, edit_rate))
            tc_e = seconds_to_simple_tc(units_to_seconds(pos + seq_l, edit_rate))
            print(f"  mob_name='{mob_name_val}'  tc={tc_s}→{tc_e}")
            print(f"  → EMIT  clip")

        else:
            print(f"  → SKIP  reason=unrecognised_class")

        pos += seq_l

    print()
    print("=" * 72)
    print("END DIAGNOSTIC DUMP")
    print("=" * 72)
    print()


def get_clip_name(comp, fallback_index: int) -> str:
    cls = comp.__class__.__name__

    if cls == "SourceClip":
        # Direct SourceClip: resolve its referenced mob name.
        try:
            mob = comp.resolve_ref()
            if mob and getattr(mob, "name", None):
                return str(mob.name)
        except Exception:
            pass

    elif cls == "OperationGroup":
        # Do NOT use comp.name — on this aaf2 version it returns the string
        # 'OperationGroup' (the class name) for every OperationGroup, which
        # is truthy and would cause all clips to share the same name.
        # Instead, dig into the inner SourceClip and resolve its mob name.
        for sc in _find_source_clips(comp):
            try:
                mob = sc.resolve_ref()
                if mob and getattr(mob, "name", None):
                    return str(mob.name)
            except Exception:
                pass

    else:
        # For any other component type, try comp.name if it exists and is not
        # a bare class-name string.
        try:
            n = comp.name
            if n and str(n) != cls:
                return str(n)
        except Exception:
            pass
        # Then try resolving any inner SourceClip mob name.
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
    try:
        if comp.__class__.__name__ == "SourceClip":
            start = getattr(comp, "start", None)
            if start is not None:
                return units_to_seconds(int(start), edit_rate)
    except Exception:
        pass

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
        raise SystemExit("Usage: python3 src/pt_to_positions.py <job_folder> [--diag]")

    job      = Path(sys.argv[1]).resolve()
    diag     = "--diag" in sys.argv[2:]
    aaf_path = find_single_aaf(job)
    out_file = job / "positions.txt"

    lines_out     = []
    clip_count    = 0
    xfade_skipped = 0

    with aaf2.open(str(aaf_path), "r") as f:
        top = None
        for mob in f.content.mobs:
            if getattr(mob, "usage", None) == "Usage_TopLevel":
                top = mob
                break

        if top is None:
            raise ValueError("No Usage_TopLevel mob found in AAF")

        seq        = None
        edit_rate  = None
        slot_info  = "?"

        for slot in getattr(top, "slots", []):
            seg = slot.segment
            if seg.__class__.__name__ == "Sequence":
                seq       = seg
                edit_rate = get_edit_rate(slot)
                try:
                    slot_info = (f"slot_id={slot.slot_id}  "
                                 f"media_kind={slot.media_kind}  "
                                 f"edit_rate={edit_rate}")
                except Exception:
                    slot_info = f"edit_rate={edit_rate}"
                break

        if seq is None:
            raise ValueError("No Sequence found on top-level mob")

        print(f"Using: {aaf_path.name}")
        print(f"  Slot: {slot_info}")
        print(f"  Sequence components: {sum(1 for _ in seq.components)}")

        if diag:
            _diag_dump(seq, edit_rate, aaf_path)

        pos        = 0
        clip_index = 0

        for comp in seq.components:
            seq_length = int(getattr(comp, "length", 0))
            if seq_length <= 0:
                pos += seq_length
                continue

            if is_xfade_op(comp):
                xfade_skipped += 1
                pos += seq_length
                continue

            if is_real_clip_component(comp):
                clip_index += 1

                placed_len, _placed_src = get_placed_length(comp)
                if placed_len <= 0:
                    pos += seq_length
                    continue

                start_sec = units_to_seconds(pos, edit_rate)
                end_sec   = units_to_seconds(pos + placed_len, edit_rate)

                start_tc = seconds_to_simple_tc(start_sec)
                end_tc   = seconds_to_simple_tc(end_sec)

                clip_name = get_clip_name(comp, clip_index)

                meta_parts = ["type=regular"]
                src_sec = get_source_start_sec(comp, edit_rate)
                if src_sec is not None:
                    meta_parts.append(f"source_start_sec={src_sec:.6f}")

                lines_out.append(f"# {' '.join(meta_parts)}")
                lines_out.append(f"{clip_name} {start_tc} {end_tc}")
                clip_count += 1

            pos += seq_length

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out) + ("\n" if lines_out else ""))

    print(f"Wrote {out_file}")
    print(f"Clips: {clip_count}"
          + (f"  ({xfade_skipped} crossfade region(s) skipped)" if xfade_skipped else ""))
    if diag:
        print("(Diagnostic mode active — see dump above for full structure)")


if __name__ == "__main__":
    main()
