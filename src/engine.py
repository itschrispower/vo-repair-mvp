import json
import logging
import shutil
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import aaf2
import librosa
import numpy as np
import soundfile as sf

from utils import FPS, SR
from matcher import (
    bandpass,
    downsample,
    match_clip_robust,
    DriftTracker,
    COARSE_RATIO,
    SR_COARSE,
    SILENCE_DB,
    rms_db,
)

# ── Logging setup ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

BP_MARGIN = int(0.10 * SR)  # 100 ms margin for bandpass filtering


# ── Timecode utilities ───────────────────────────────────────────────────────

def tc_to_seconds(tc: str) -> float:
    """
    Parse timecode to seconds.

    Supports:
      - HH:MM:SS:FF (SMPTE timecode)
      - SS.FF (simple frame offset)

    Raises ValueError on invalid format.
    """
    tc = str(tc).strip()

    try:
        if ":" in tc:
            parts = tc.split(":")
            if len(parts) == 4:
                hh, mm, ss, ff = parts
                hours = int(hh)
                minutes = int(mm)
                seconds = int(ss)
                frames = int(ff)

                if not (0 <= hours < 100 and 0 <= minutes < 60 and
                        0 <= seconds < 60 and 0 <= frames < int(FPS) + 1):
                    raise ValueError(f"Timecode out of range: {tc}")

                return hours * 3600 + minutes * 60 + seconds + (frames / FPS)
            else:
                raise ValueError(f"Invalid SMPTE format (expected HH:MM:SS:FF): {tc}")
        else:
            parts = tc.split(".")
            if len(parts) == 2:
                secs = int(parts[0])
                frames = int(parts[1])

                if secs < 0 or frames < 0 or frames >= int(FPS) + 1:
                    raise ValueError(f"Invalid frame offset: {tc}")

                return secs + (frames / FPS)
            else:
                raise ValueError(f"Invalid timecode format: {tc}")
    except (ValueError, IndexError) as e:
        raise ValueError(f"Failed to parse timecode '{tc}': {e}")


def tc_to_samples(tc: str) -> int:
    """Convert timecode to sample index at SR."""
    try:
        seconds = tc_to_seconds(tc)
        samples = int(round(seconds * SR))
        if samples < 0:
            raise ValueError(f"Negative sample index from {tc}")
        return samples
    except Exception as e:
        raise ValueError(f"Failed to convert timecode to samples '{tc}': {e}")


# ── Audio utilities ──────────────────────────────────────────────────────────

def ensure_mono(audio: np.ndarray) -> np.ndarray:
    """Convert to mono float32."""
    if audio.ndim == 1:
        return audio.astype(np.float32)
    return audio.mean(axis=1).astype(np.float32)


def load_positions(path: Path) -> List[Tuple[str, str, str]]:
    """
    Load clip positions from positions.txt.

    Format:
      # metadata line (skipped)
      clip_name start_tc end_tc

    Returns list of (name, start_tc, end_tc) tuples.
    Raises ValueError if file is missing or malformed.
    """
    if not path.exists():
        raise FileNotFoundError(f"positions.txt not found: {path}")

    clips = []
    lines = path.read_text(encoding="utf-8").splitlines()

    for line_num, line in enumerate(lines, 1):
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 3:
            raise ValueError(
                f"positions.txt:{line_num}: expected 3+ parts (name start end), "
                f"got {len(parts)}: {line}"
            )

        name, start_tc, end_tc = parts[0], parts[1], parts[2]

        # Validate timecodes early
        try:
            tc_to_samples(start_tc)
            tc_to_samples(end_tc)
        except ValueError as e:
            raise ValueError(f"positions.txt:{line_num}: {e}")

        clips.append((name, start_tc, end_tc))

    if not clips:
        raise ValueError("positions.txt contains no valid clip lines")

    return clips


def prepare_vobu(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load VOBU WAV and precompute coarse-rate bandpass for search.

    Returns (vobu, vobu_coarse) as mono float32.
    """
    if not path.exists():
        raise FileNotFoundError(f"VOBU WAV not found: {path}")

    try:
        vobu, sr = librosa.load(str(path), sr=SR, mono=True)
    except Exception as e:
        raise RuntimeError(f"Failed to load VOBU: {e}")

    if sr != SR:
        raise ValueError(f"VOBU must be {SR} Hz; got {sr} Hz")

    vobu = vobu.astype(np.float32)

    if len(vobu) == 0:
        raise ValueError("VOBU is empty")

    try:
        vobu_bp = bandpass(vobu, sr=SR)
        vobu_coarse = downsample(vobu_bp, from_sr=SR, to_sr=SR_COARSE)
    except Exception as e:
        raise RuntimeError(f"Failed to process VOBU: {e}")

    log.info(f"VOBU loaded: {len(vobu):,} samples ({len(vobu)/SR/60:.1f} min)")
    log.info(f"Coarse: {len(vobu_coarse):,} samples")

    return vobu, vobu_coarse


def load_reference(path: Path) -> np.ndarray:
    """Load aaf_reference.wav as mono float32."""
    if not path.exists():
        raise FileNotFoundError(f"aaf_reference.wav not found: {path}")

    try:
        raw, sr = sf.read(str(path))
    except Exception as e:
        raise RuntimeError(f"Failed to load aaf_reference.wav: {e}")

    if sr != SR:
        raise ValueError(f"aaf_reference.wav must be {SR} Hz; got {sr} Hz")

    audio = ensure_mono(raw)

    if len(audio) == 0:
        raise ValueError("aaf_reference.wav is empty")

    log.info(f"Reference loaded: {len(audio):,} samples ({len(audio)/SR:.1f}s)")

    return audio


# ── Main pipeline ────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python3 src/engine.py <job_folder>")

    try:
        base = Path(sys.argv[1]).resolve()

        # ── Input validation ─────────────────────────────────────────────────

        vo_path = base / "audio" / "VOBU_48k.wav"
        ref_path = base / "audio" / "aaf_reference.wav"
        pos_path = base / "positions.txt"

        for p, label in [(vo_path, "VOBU_48k.wav"),
                         (ref_path, "aaf_reference.wav"),
                         (pos_path, "positions.txt")]:
            if not p.exists():
                raise FileNotFoundError(f"Missing {label}: {p}")

        # ── Output directories ───────────────────────────────────────────────

        out_dir = base / "rebuild_audio"
        check_dir = base / "match_check"
        deliver_dir = base / "deliverables"

        out_dir.mkdir(exist_ok=True, parents=True)
        check_dir.mkdir(exist_ok=True, parents=True)
        deliver_dir.mkdir(exist_ok=True, parents=True)

        # ── Load audio ───────────────────────────────────────────────────────

        log.info("Loading audio files…")
        vobu, vobu_coarse = prepare_vobu(vo_path)
        ref_audio = load_reference(ref_path)

        # ── Load positions ───────────────────────────────────────────────────

        log.info("Loading positions…")
        clips = load_positions(pos_path)
        log.info(f"Found {len(clips)} clips to process")

        # ── Allocate output buffer ───────────────────────────────────────────

        # Find max end time to size output buffer
        max_end_sample = 0
        for name, start_tc, end_tc in clips:
            try:
                end_sample = tc_to_samples(end_tc)
                max_end_sample = max(max_end_sample, end_sample)
            except Exception as e:
                log.warning(f"Skipping clip '{name}' (invalid timecode): {e}")
                continue

        if max_end_sample <= 0:
            raise ValueError("No valid clip timecodes found")

        output = np.zeros(max_end_sample, dtype=np.float32)
        log.info(f"Output buffer: {max_end_sample:,} samples ({max_end_sample/SR:.1f}s)")

        # ── Initialize DriftTracker ──────────────────────────────────────────

        tracker = DriftTracker(vobu_len=len(vobu), sr=SR)

        # ── Match clips ──────────────────────────────────────────────────────

        log.info(f"\nMatching {len(clips)} clips…\n")

        processed = 0
        skipped = 0
        report = []

        for clip_idx, (name, start_tc, end_tc) in enumerate(clips, 1):
            try:
                # Parse timecodes
                try:
                    start_sample = tc_to_samples(start_tc)
                    end_sample = tc_to_samples(end_tc)
                except ValueError as e:
                    log.warning(f"[{clip_idx:3d}] {name} skipped: {e}")
                    skipped += 1
                    continue

                target_len = end_sample - start_sample

                # Validate clip extent
                if target_len <= 0:
                    log.warning(
                        f"[{clip_idx:3d}] {name} skipped "
                        f"(invalid range: {start_tc} → {end_tc})"
                    )
                    skipped += 1
                    continue

                if start_sample >= len(output):
                    log.warning(
                        f"[{clip_idx:3d}] {name} skipped "
                        f"(start {start_sample} beyond output buffer)"
                    )
                    skipped += 1
                    continue

                # Extract reference clip
                if end_sample > len(ref_audio):
                    log.warning(
                        f"[{clip_idx:3d}] {name} skipped "
                        f"(end {end_tc} beyond reference length)"
                    )
                    skipped += 1
                    continue

                ref_clip = ref_audio[start_sample:end_sample].astype(np.float32)

                if len(ref_clip) == 0:
                    log.warning(f"[{clip_idx:3d}] {name} skipped (empty reference)")
                    skipped += 1
                    continue

                ref_bp = bandpass(ref_clip, sr=SR)

                # ── Get search window ────────────────────────────────────────

                search_lo, search_hi = tracker.get_search_window(start_sample, target_len)

                # Validate search window
                search_lo = max(0, search_lo)
                search_hi = min(len(vobu), search_hi)

                if search_lo >= search_hi:
                    log.warning(
                        f"[{clip_idx:3d}] {name} skipped (invalid search window)"
                    )
                    skipped += 1
                    continue

                if (search_hi - search_lo) < len(ref_clip):
                    log.warning(
                        f"[{clip_idx:3d}] {name} skipped "
                        f"(search window {search_hi-search_lo} < clip len {len(ref_clip)})"
                    )
                    skipped += 1
                    continue

                # ── Compute coarse indices ───────────────────────────────────

                lo_c = search_lo // COARSE_RATIO
                hi_c = min(len(vobu_coarse), (search_hi + COARSE_RATIO - 1) // COARSE_RATIO)

                if lo_c >= hi_c:
                    log.warning(f"[{clip_idx:3d}] {name} skipped (invalid coarse window)")
                    skipped += 1
                    continue

                # ── Extract and bandpass search region ───────────────────────

                bp_lo = max(0, search_lo - BP_MARGIN)
                bp_hi = min(len(vobu), search_hi + BP_MARGIN)

                region_bp_raw = bandpass(vobu[bp_lo:bp_hi], sr=SR)
                bp_start = search_lo - bp_lo
                bp_end = bp_start + (search_hi - search_lo)

                if bp_end > len(region_bp_raw):
                    bp_end = len(region_bp_raw)

                region_bp = region_bp_raw[bp_start:bp_end]

                if len(region_bp) != (search_hi - search_lo):
                    log.warning(f"[{clip_idx:3d}] {name} skipped (bandpass mismatch)")
                    skipped += 1
                    continue

                # ── First pass: narrow window match ──────────────────────────

                try:
                    result = match_clip_robust(
                        region=vobu[search_lo:search_hi],
                        region_bp=region_bp,
                        region_coarse=vobu_coarse[lo_c:hi_c],
                        clip=ref_clip,
                        clip_bp=ref_bp,
                    )
                except Exception as e:
                    log.warning(f"[{clip_idx:3d}] {name} skipped (matching failed): {e}")
                    skipped += 1
                    continue

                # ── Two-pass retry: expand window for uncertain results ──────

                should_retry = (
                    result.status == "fail"
                    or result.confidence < 0.7
                    or result.status in ("review", "forced")
                )

                if should_retry and (search_lo > 0 or search_hi < len(vobu)):
                    try:
                        if len(tracker._anchors) < 2:
                            expand = int(20.0 * SR)
                            lo_w = max(0, search_lo - expand)
                            hi_w = min(len(vobu), search_hi + expand)
                        else:
                            expand = (search_hi - search_lo) // 2
                            lo_w = max(0, search_lo - expand)
                            hi_w = min(len(vobu), search_hi + expand)

                        if (hi_w - lo_w) > (search_hi - search_lo) * 1.3:
                            lo_c_w = lo_w // COARSE_RATIO
                            hi_c_w = min(len(vobu_coarse), (hi_w + COARSE_RATIO - 1) // COARSE_RATIO)

                            bp_lo_w = max(0, lo_w - BP_MARGIN)
                            bp_hi_w = min(len(vobu), hi_w + BP_MARGIN)

                            region_bp_w_raw = bandpass(vobu[bp_lo_w:bp_hi_w], sr=SR)
                            bp_start_w = lo_w - bp_lo_w
                            bp_end_w = bp_start_w + (hi_w - lo_w)

                            if bp_end_w > len(region_bp_w_raw):
                                bp_end_w = len(region_bp_w_raw)

                            region_bp_w = region_bp_w_raw[bp_start_w:bp_end_w]

                            if len(region_bp_w) == (hi_w - lo_w):
                                result_w = match_clip_robust(
                                    region=vobu[lo_w:hi_w],
                                    region_bp=region_bp_w,
                                    region_coarse=vobu_coarse[lo_c_w:hi_c_w],
                                    clip=ref_clip,
                                    clip_bp=ref_bp,
                                )

                                if result_w.confidence > result.confidence:
                                    result = result_w
                                    search_lo, search_hi = lo_w, hi_w
                    except Exception as e:
                        log.debug(f"[{clip_idx:3d}] {name} retry failed: {e}")

                # ── Compute VOBU match position ──────────────────────────────

                vobu_match = max(0, search_lo + result.offset - result.trim_samples)

                # ── Extract output audio ─────────────────────────────────────

                if vobu_match + target_len > len(vobu):
                    available = len(vobu) - vobu_match
                    vo_clip = vobu[vobu_match:vobu_match + available]

                    if len(vo_clip) < target_len:
                        padded = np.zeros(target_len, dtype=np.float32)
                        padded[:len(vo_clip)] = vo_clip
                        vo_clip = padded
                else:
                    vo_clip = vobu[vobu_match:vobu_match + target_len].astype(np.float32)

                # Ensure exact length
                if len(vo_clip) > target_len:
                    vo_clip = vo_clip[:target_len]
                elif len(vo_clip) < target_len:
                    padded = np.zeros(target_len, dtype=np.float32)
                    padded[:len(vo_clip)] = vo_clip
                    vo_clip = padded

                # Apply polarity if needed
                if result.polarity == -1:
                    vo_clip = -vo_clip

                # ── Write to output buffer ───────────────────────────────────

                write_end = min(end_sample, len(output))
                write_len = write_end - start_sample

                if write_len > 0:
                    output[start_sample:write_end] = vo_clip[:write_len]

                # ── Record anchor for DriftTracker ──────────────────────────

                tracker.record(start_sample, vobu_match, result.status, confidence=result.confidence)

                # ── Write preview file ───────────────────────────────────────

                try:
                    preview_path = check_dir / f"{clip_idx:03d}_{name}.wav"
                    sf.write(str(preview_path), vo_clip, SR)
                except Exception as e:
                    log.debug(f"[{clip_idx:3d}] {name} preview write failed: {e}")

                # ── Log result ───────────────────────────────────────────────

                status_sym = {"ok": "✓", "review": "~", "fail": "✗", "forced": "⚡"}.get(result.status, "?")
                log.info(
                    f"[{clip_idx:3d}/{len(clips)}] {name:30s} {status_sym} {result.status.upper():7s} "
                    f"conf={result.confidence:.3f} vobu={vobu_match/SR:.2f}s"
                )

                report.append({
                    "index": clip_idx,
                    "name": name,
                    "timeline_start_tc": start_tc,
                    "timeline_end_tc": end_tc,
                    "timeline_start_samples": start_sample,
                    "timeline_end_samples": end_sample,
                    "status": result.status,
                    "confidence": round(result.confidence, 4),
                    "vobu_match_sec": round(vobu_match / SR, 4),
                    "vobu_match_samples": vobu_match,
                    "raw_ncc": round(result.raw_ncc, 4),
                    "stability": round(result.stability, 4),
                })

                processed += 1

            except Exception as e:
                log.error(f"[{clip_idx:3d}] {name} error: {e}", exc_info=False)
                skipped += 1

        # ── Write output ─────────────────────────────────────────────────────

        log.info(f"\nProcessed: {processed} | Skipped: {skipped}")

        final_path = out_dir / "final.wav"

        try:
            sf.write(str(final_path), output, SR)
            log.info(f"Final WAV: {final_path}")
        except Exception as e:
            log.error(f"Failed to write final.wav: {e}")
            raise

        # ── Copy to deliverables ─────────────────────────────────────────────

        try:
            deliver_final = deliver_dir / "final.wav"
            shutil.copy2(final_path, deliver_final)
            log.info(f"Deliverable: {deliver_final}")
        except Exception as e:
            log.error(f"Failed to copy final.wav to deliverables: {e}")
            raise

        # ── Write report ─────────────────────────────────────────────────────

        try:
            report_path = out_dir / "report.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            log.info(f"Report: {report_path}")
        except Exception as e:
            log.warning(f"Failed to write report: {e}")

        # ── Write manifest ──────────────────────────────────────────────────

        try:
            manifest_path = out_dir / "manifest.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            log.info(f"Manifest: {manifest_path}")
        except Exception as e:
            log.warning(f"Failed to write manifest: {e}")

        # ── Generate AAF from manifest ──────────────────────────────────────

        try:
            if report:
                aaf_out_path = out_dir / "rebuilt.aaf"
                with aaf2.open(str(aaf_out_path), "w") as f:
                    comp = f.create.CompositionMob("VO_REPAIR_OUTPUT")
                    comp.usage = "Usage_TopLevel"
                    f.content.mobs.append(comp)

                    slot = comp.create_sound_slot(edit_rate=SR)
                    seq = slot.segment

                    # Import final.wav as essence once
                    master = f.create.MasterMob("rebuilt_audio_master")
                    f.content.mobs.append(master)
                    essence_slot = master.import_audio_essence(str(final_path), SR)

                    current_pos = 0

                    for item in report:
                        start = item["timeline_start_samples"]
                        end = item["timeline_end_samples"]
                        duration = end - start

                        # Add filler if gap
                        if start > current_pos:
                            filler = f.create.Filler("sound")
                            filler.length = start - current_pos
                            seq.components.append(filler)
                            current_pos = start

                        # Create source clip pointing to segment of final.wav
                        src_clip = master.create_source_clip(
                            slot_id=essence_slot.slot_id,
                            start=start,
                            length=duration,
                            media_kind="sound",
                        )
                        src_clip.length = duration
                        seq.components.append(src_clip)
                        current_pos += duration

                    f.save()
                log.info(f"AAF: {aaf_out_path}")
                shutil.copy2(aaf_out_path, deliver_dir / aaf_out_path.name)
        except Exception as e:
            log.warning(f"AAF generation failed: {e}")

        log.info("\n✓ DONE")
        print(final_path)
        print(deliver_final)

    except Exception as e:
        log.error(f"Fatal error: {e}", exc_info=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
