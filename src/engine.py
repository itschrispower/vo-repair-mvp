import json
import math
import re
import shutil
import sys
from pathlib import Path

import aaf2
import librosa
import numpy as np
import soundfile as sf

from utils import FPS, SR
from matcher import (
    bandpass,
    downsample,
    match_clip_robust,
    rms_db,
    DriftTracker,
    COARSE_RATIO,
    SILENCE_DB,
    SR_COARSE,
    DRIFT_JUMP_S,
)

# Legacy thresholds kept for reference — status is now assigned by matcher.py
FAIL_THRESHOLD = 0.42
REVIEW_THRESHOLD = 0.75

# Per-window bandpass margin to avoid filter-transient contamination at window edges
BP_MARGIN: int = int(0.10 * SR)   # 100 ms


# ── Timecode helpers ──────────────────────────────────────────────────────────

def tc_to_seconds(tc: str) -> float:
    if ":" in tc:
        hh, mm, ss, ff = tc.split(":")
        return int(ss) + (int(ff) / FPS)
    secs, frames = tc.split(".")
    return int(secs) + (int(frames) / FPS)


def tc_to_samples(tc: str) -> int:
    return int(round(tc_to_seconds(tc) * SR))


def seconds_to_tc(seconds: float) -> str:
    total_frames = int(math.floor(seconds * FPS + 1e-9))
    secs = total_frames // int(FPS)
    frames = total_frames % int(FPS)
    return f"{secs:02d}.{frames:02d}"


# ── Audio helpers ─────────────────────────────────────────────────────────────

def _ensure_mono(audio: np.ndarray) -> np.ndarray:
    """Mix stereo/multichannel down to mono."""
    if audio.ndim == 1:
        return audio.astype(np.float32)
    return audio.mean(axis=1).astype(np.float32)


def load_positions(path: Path):
    clips = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"Bad line in positions.txt: {line}")
            name, start, end = parts
            clips.append((name, start, end))
    return clips


def sanitise_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return cleaned or "job"


def detect_job_label(base: Path) -> str:
    aaf_files = sorted(base.glob("*.aaf"))
    if aaf_files:
        return sanitise_name(aaf_files[0].stem)
    txt_files = sorted(
        p for p in base.glob("*.txt")
        if p.name.lower() != "positions.txt"
    )
    if txt_files:
        return sanitise_name(txt_files[0].stem)
    return sanitise_name(base.name)


def build_auto_refs(aaf_reference_path: Path, clips, auto_ref_dir: Path):
    """
    Extract each clip segment from aaf_reference.wav, write as mono WAV.
    Always writes mono regardless of the source file channel count.
    """
    raw, sr = sf.read(str(aaf_reference_path))
    if sr != SR:
        raise ValueError(f"aaf_reference.wav must be {SR} Hz, got {sr}")

    audio = _ensure_mono(raw)
    auto_ref_dir.mkdir(exist_ok=True)
    refs = {}

    for clip_name, start_tc, end_tc in clips:
        start = tc_to_samples(start_tc)
        end = tc_to_samples(end_tc)

        if start < 0 or end <= start or end > len(audio):
            raise ValueError(
                f"Clip {clip_name} falls outside aaf_reference.wav: "
                f"{start_tc} → {end_tc} ({start}–{end} samples, "
                f"file length {len(audio)} samples)"
            )

        clip_audio = audio[start:end]
        out_path = auto_ref_dir / f"{sanitise_name(clip_name)}_auto_ref.wav"
        sf.write(str(out_path), clip_audio, SR)
        refs[clip_name] = out_path

    return refs


def prepare_vobu(vo_path: Path):
    """
    Load the VOBU WAV and precompute the coarse-rate variant for fast search.

    Returns (vobu, vobu_coarse) as mono float32 arrays.

    vobu        — full rate (48 kHz), used for output extraction and per-window
                  bandpass computation during matching
    vobu_coarse — downsampled bandpass at SR_COARSE, used for coarse search

    Note: a full-rate bandpass copy (vobu_bp) is NOT stored globally — that
    would use ~700 MB for a 1-hour VOBU. Instead, bandpass is computed on-demand
    per search window with BP_MARGIN padding to avoid transient artefacts.
    """
    print(f"Loading VOBU: {vo_path.name}")
    vobu, vo_sr = librosa.load(str(vo_path), sr=SR, mono=True)
    if vo_sr != SR:
        raise ValueError(f"VOBU must be {SR} Hz (got {vo_sr})")
    vobu = vobu.astype(np.float32)

    print("  Computing coarse-rate bandpass for search…")
    vobu_bp_tmp = bandpass(vobu, sr=SR)
    vobu_coarse = downsample(vobu_bp_tmp, from_sr=SR, to_sr=SR_COARSE)
    del vobu_bp_tmp  # free ~700 MB — bandpass recomputed per window at runtime

    duration_min = len(vobu) / SR / 60.0
    print(
        f"  VOBU: {len(vobu):,} samples ({duration_min:.1f} min) | "
        f"coarse: {len(vobu_coarse):,} samples"
    )
    return vobu, vobu_coarse


# ── Output / reporting helpers ────────────────────────────────────────────────

def write_summary(
    path: Path,
    report: list,
    final_path: Path,
    aaf_path,           # Path or None
    has_failures: bool,
):
    lines = ["VO REPAIR SUMMARY", "=" * 52, ""]

    ok_count     = sum(1 for r in report if r["status"] == "ok")
    review_count = sum(1 for r in report if r["status"] == "review")
    fail_count   = sum(1 for r in report if r["status"] == "fail")
    forced_count = sum(1 for r in report if r["status"] == "forced")
    incons_count = sum(1 for r in report if not r.get("consistency_ok", True))

    lines.append(
        f"Clips: {len(report)} total | "
        f"OK={ok_count}  REVIEW={review_count}  FAIL={fail_count}"
        + (f"  FORCED={forced_count}" if forced_count else "")
    )
    if incons_count:
        lines.append(f"Consistency warnings: {incons_count}")
    if has_failures:
        lines.append(
            f"⚠ {forced_count} clip(s) had low-confidence matches — "
            f"best-guess audio included and flagged FORCED."
        )
    lines.append("")

    for item in report:
        st = item["status"]
        status_sym = {"ok": "✓", "review": "~", "fail": "✗", "forced": "⚡"}.get(st, "?")
        lines.append(f"Clip {item['index']} — {status_sym} {st.upper()}")
        lines.append(f"  Name:    {item['clip_name']}")
        lines.append(f"  Ref:     {item['ref_name']}")
        lines.append(
            f"  Placed:  {item['timeline_start_tc']} → {item['timeline_end_tc']}"
        )
        lines.append(
            f"  VOBU at: {item['source_match_sec']:.4f}s "
            f"(conf={item['confidence']:.4f}, "
            f"raw={item.get('raw_ncc', 0):.4f}, "
            f"stab={item.get('stability', 1.0):.2f}, "
            f"agree={item.get('method_agreement', 1.0):.2f}, "
            f"polarity={'inv' if item.get('polarity', 1) == -1 else 'ok'})"
        )
        if not item.get("consistency_ok", True):
            dev = item.get("consistency_deviation_s", 0.0)
            lines.append(f"  ⚠ Consistency: {dev:.3f}s deviation from trend")
        if item.get("drift_jump_s", 0.0) > DRIFT_JUMP_S:
            lines.append(f"  ⚠ Drift jump: {item['drift_jump_s']:.2f}s — check for false positive")
        for reason in item.get("reasons", []):
            lines.append(f"  ⚑ {reason}")
        lines.append("")

    lines.append(f"Final WAV: {final_path}")
    if aaf_path and Path(aaf_path).exists():
        lines.append(f"AAF:       {aaf_path}")
    else:
        lines.append("AAF:       not generated (see log for details)")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_embedded_aaf(manifest: list, out_aaf: Path):
    with aaf2.open(str(out_aaf), "w") as f:
        comp = f.create.CompositionMob("VOREPAIR_REBUILT")
        comp.usage = "Usage_TopLevel"
        f.content.mobs.append(comp)

        slot = comp.create_sound_slot(edit_rate=SR)
        seq = slot.segment

        current_pos = 0

        for item in manifest:
            timeline_start = int(item["timeline"]["start_samples"])
            duration = int(item["timeline"]["duration_samples"])
            rebuilt_file = Path(item["rebuilt_file"]).resolve()

            if timeline_start > current_pos:
                filler = f.create.Filler("sound")
                filler.length = timeline_start - current_pos
                seq.components.append(filler)
                current_pos = timeline_start

            mob_name = f"clip_{item['index']}_{item['clip_name']}"
            master_mob = f.create.MasterMob(mob_name)
            f.content.mobs.append(master_mob)

            essence_slot = master_mob.import_audio_essence(str(rebuilt_file), SR)

            src_clip = master_mob.create_source_clip(
                slot_id=essence_slot.slot_id,
                start=0,
                length=duration,
                media_kind="sound",
            )
            src_clip.length = duration

            seq.components.append(src_clip)
            current_pos += duration

        f.save()


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python3 src/engine.py <job_folder>")

    base = Path(sys.argv[1]).resolve()
    job_label = detect_job_label(base)

    vo_path       = base / "audio" / "VOBU_48k.wav"
    aaf_ref_path  = base / "audio" / "aaf_reference.wav"
    positions_path = base / "positions.txt"

    out_dir          = base / "rebuild_audio"
    check_dir        = base / "match_check"
    auto_ref_dir     = base / "auto_refs"
    rebuilt_clips_dir = out_dir / "rebuilt_clips"
    deliver_dir      = base / "deliverables"

    final_path    = out_dir / f"{job_label}_final.wav"
    report_path   = out_dir / f"{job_label}_report.json"
    summary_path  = out_dir / f"{job_label}_summary.txt"
    manifest_path = out_dir / f"{job_label}_manifest.json"
    aaf_out_path  = out_dir / f"{job_label}_rebuilt.aaf"

    deliver_final_path   = deliver_dir / f"{job_label}_final.wav"
    deliver_summary_path = deliver_dir / f"{job_label}_summary.txt"
    deliver_aaf_path     = deliver_dir / f"{job_label}_rebuilt.aaf"

    for p, label in [
        (vo_path,        "VOBU WAV"),
        (aaf_ref_path,   "aaf_reference.wav"),
        (positions_path, "positions.txt"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"Missing {label}: {p}")

    out_dir.mkdir(exist_ok=True)
    check_dir.mkdir(exist_ok=True)
    rebuilt_clips_dir.mkdir(exist_ok=True)
    deliver_dir.mkdir(exist_ok=True)

    # ── Load + precompute VOBU variants ──────────────────────────────────────
    vobu, vobu_coarse = prepare_vobu(vo_path)

    clips     = load_positions(positions_path)
    auto_refs = build_auto_refs(aaf_ref_path, clips, auto_ref_dir)

    max_end = max(tc_to_samples(end_tc) for _, _, end_tc in clips)
    output  = np.zeros(max_end, dtype=np.float32)

    tracker = DriftTracker(vobu_len=len(vobu), sr=SR)

    report: list       = []
    manifest: list     = []
    has_failures: bool = False   # True if any clip landed in "forced" status

    print(f"\nMatching {len(clips)} clips against VOBU…")

    for i, (clip_name, start_tc, end_tc) in enumerate(clips, start=1):
        ref_path = auto_refs[clip_name]

        ref_raw, ref_sr = sf.read(str(ref_path))
        if ref_sr != SR:
            raise ValueError(f"Reference clip must be {SR} Hz: {ref_path.name}")
        ref_clip    = _ensure_mono(ref_raw)
        ref_clip_bp = bandpass(ref_clip, sr=SR)

        out_start  = tc_to_samples(start_tc)
        out_end    = tc_to_samples(end_tc)
        target_len = out_end - out_start

        # ── Bounded search window from DriftTracker ───────────────────────
        search_lo, search_hi = tracker.get_search_window(out_start, target_len)

        lo_c = search_lo // COARSE_RATIO
        hi_c = min(len(vobu_coarse), (search_hi + COARSE_RATIO - 1) // COARSE_RATIO)

        # ── Per-window bandpass (memory-efficient: avoids storing full VOBU BP) ──
        bp_lo           = max(0, search_lo - BP_MARGIN)
        bp_hi           = min(len(vobu), search_hi + BP_MARGIN)
        region_bp_padded = bandpass(vobu[bp_lo:bp_hi], sr=SR)
        bp_inner         = search_lo - bp_lo
        region_bp        = region_bp_padded[bp_inner: bp_inner + (search_hi - search_lo)]
        del region_bp_padded

        # ── Match ─────────────────────────────────────────────────────────
        result = match_clip_robust(
            region=vobu[search_lo:search_hi],
            region_bp=region_bp,
            region_coarse=vobu_coarse[lo_c:hi_c],
            clip=ref_clip,
            clip_bp=ref_clip_bp,
        )

        # ── vobu_match: correct for edge-trim offset ──────────────────────
        # result.offset = start of the trimmed clip in region.
        # Subtract trim_samples to get the true start of the original clip.
        vobu_match = max(0, search_lo + result.offset - result.trim_samples)

        # ── Classify and assign effective status ──────────────────────────
        clip_db         = rms_db(ref_clip)
        is_silent_clip  = clip_db < SILENCE_DB
        is_original_fail = (result.status == "fail")
        effective_status = result.status
        effective_reasons = list(result.reasons)
        forced = False

        if is_original_fail and is_silent_clip:
            # Silent clip that also failed: write zeros and mark review.
            # Better to have silence than random VOBU audio at an unknown offset.
            effective_status = "review"
            effective_reasons.append(
                "silence detected — zeros written at this position (review expected)"
            )
        elif is_original_fail:
            # Non-silent clip with a failed match: include best-guess audio
            # and mark FORCED so the engineer knows to check it.
            effective_status = "forced"
            forced = True
            has_failures = True
            effective_reasons.append(
                "FORCED: low-confidence match — best-guess audio written, manual review required"
            )

        # ── Drift jump detection ──────────────────────────────────────────
        drift_jump_s  = 0.0
        is_drift_jump = False
        if not (is_original_fail and is_silent_clip):
            drift_jump_s  = tracker.check_drift_jump(out_start, vobu_match)
            is_drift_jump = drift_jump_s > DRIFT_JUMP_S
            if is_drift_jump:
                effective_reasons.append(
                    f"drift jump: {drift_jump_s:.2f}s from predicted position — "
                    f"possible false positive"
                )

        # Record as drift anchor — skip silence overrides and jumps to avoid
        # poisoning the regression with unreliable positions.
        if not (is_original_fail and is_silent_clip) and not is_drift_jump:
            tracker.record(
                out_start, vobu_match, effective_status,
                confidence=result.confidence,
            )

        # ── Extract VOBU segment ──────────────────────────────────────────
        if is_original_fail and is_silent_clip:
            vo_clip = np.zeros(target_len, dtype=np.float32)
        else:
            vo_clip = vobu[vobu_match: vobu_match + target_len]
            if len(vo_clip) < target_len:
                padded = np.zeros(target_len, dtype=np.float32)
                padded[: len(vo_clip)] = vo_clip
                vo_clip = padded
            if result.polarity == -1:
                vo_clip = -vo_clip

        preview_path = check_dir / f"match_{sanitise_name(clip_name)}.wav"
        sf.write(str(preview_path), vo_clip[:target_len], SR)

        rebuilt_clip_path = rebuilt_clips_dir / f"{sanitise_name(clip_name)}_rebuilt.wav"
        sf.write(str(rebuilt_clip_path), vo_clip[:target_len], SR)

        # ── ALWAYS place clip into output array ───────────────────────────
        # Never abort due to failed/forced clips — always include the best
        # available audio. Forced clips are flagged in the report and summary.
        output[out_start:out_end] = vo_clip[:target_len]

        # ── Build report item ─────────────────────────────────────────────
        item = {
            "index": i,
            "clip_name": clip_name,
            "ref_name": ref_path.name,
            "timeline_start_tc": start_tc,
            "timeline_end_tc": end_tc,
            "timeline_start_sec": round(out_start / SR, 6),
            "timeline_end_sec": round(out_end / SR, 6),
            "timeline_start_samples": int(out_start),
            "timeline_end_samples": int(out_end),
            "source_match_sec": round(vobu_match / SR, 6),
            "vobu_match_samples": int(vobu_match),
            "search_lo_sec": round(search_lo / SR, 4),
            "search_hi_sec": round(search_hi / SR, 4),
            "confidence": round(result.confidence, 6),
            "raw_ncc": round(result.raw_ncc, 6),
            "stability": round(result.stability, 4),
            "method_agreement": round(result.method_agreement, 4),
            "trim_samples": result.trim_samples,
            "near_cands_count": result.near_cands_count,
            "drift_jump_s": round(drift_jump_s, 4),
            "status": effective_status,
            "forced": forced,
            "polarity": result.polarity,
            "reasons": effective_reasons,
            "candidates_count": len(result.candidates),
            "candidates_top": [
                {
                    "offset_sec": round(
                        (search_lo + c.offset - result.trim_samples) / SR, 4
                    ),
                    "ncc": round(c.ncc, 4),
                    "ncc_wb": round(c.ncc_wb, 4),
                    "ncc_bp": round(c.ncc_bp, 4),
                    "stability": round(c.stability, 4),
                    "polarity": c.polarity,
                }
                for c in result.candidates
            ],
            "consistency_ok": True,
            "consistency_deviation_s": 0.0,
            "output_preview": str(preview_path.relative_to(base)),
            "rebuilt_file": str(rebuilt_clip_path.relative_to(base)),
        }
        report.append(item)

        manifest.append(
            {
                "index": i,
                "clip_name": clip_name,
                "timeline": {
                    "start_tc": start_tc,
                    "end_tc": end_tc,
                    "start_sec": round(out_start / SR, 6),
                    "end_sec": round(out_end / SR, 6),
                    "duration_sec": round(target_len / SR, 6),
                    "start_samples": int(out_start),
                    "end_samples": int(out_end),
                    "duration_samples": int(target_len),
                },
                "source": {
                    "file": vo_path.name,
                    "match_sec": round(vobu_match / SR, 6),
                    "match_samples": int(vobu_match),
                    "duration_sec": round(target_len / SR, 6),
                    "duration_samples": int(target_len),
                },
                "rebuilt_file": str(rebuilt_clip_path),
                "confidence": round(result.confidence, 6),
                "status": effective_status,
                "forced": forced,
                "polarity": result.polarity,
            }
        )

        extras = []
        if result.polarity == -1:
            extras.append("INV")
        if is_drift_jump:
            extras.append(f"JUMP={drift_jump_s:.1f}s")
        if forced:
            extras.append("FORCED")
        elif effective_reasons:
            extras.append(effective_reasons[0][:60])
        extra_str = " | " + " | ".join(extras) if extras else ""
        print(
            f"  [{i:3d}/{len(clips)}] {clip_name:30s} | "
            f"VOBU={vobu_match / SR:.3f}s | "
            f"conf={result.confidence:.3f} stab={result.stability:.2f} | "
            f"{effective_status.upper()}{extra_str}"
        )

    # ── Consistency post-pass ─────────────────────────────────────────────────
    DriftTracker.check_consistency(report, sr=SR)

    for item in report:
        if not item.get("consistency_ok", True) and item["status"] == "ok":
            item["status"] = "review"
            dev = item.get("consistency_deviation_s", 0.0)
            item["reasons"].append(
                f"demoted ok→review: position {dev:.3f}s from trend "
                f"(possible wrong candidate)"
            )
            print(
                f"  ⚠ Clip {item['index']} ({item['clip_name']}): "
                f"consistency deviation {dev:.3f}s → demoted to REVIEW"
            )

    # ── Write JSON report and manifest ────────────────────────────────────────
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # ── ALWAYS write final WAV ────────────────────────────────────────────────
    sf.write(str(final_path), output, SR)
    print(f"\nFinal WAV: {final_path.name}")

    # ── ALWAYS attempt AAF generation ─────────────────────────────────────────
    aaf_written = False
    aaf_deliver = None
    try:
        write_embedded_aaf(manifest, aaf_out_path)
        aaf_written = True
        aaf_deliver = aaf_out_path
        print(f"AAF:       {aaf_out_path.name}")
    except Exception as exc:
        print(f"⚠ AAF generation failed: {exc}")

    # ── Write summary ─────────────────────────────────────────────────────────
    write_summary(summary_path, report, final_path, aaf_deliver, has_failures)

    # ── Copy everything to deliverables (always) ──────────────────────────────
    deliver_dir.mkdir(exist_ok=True)
    shutil.copy2(final_path, deliver_final_path)
    shutil.copy2(summary_path, deliver_summary_path)
    if aaf_written and aaf_out_path.exists():
        shutil.copy2(aaf_out_path, deliver_aaf_path)

    ok_n     = sum(1 for r in report if r["status"] == "ok")
    rev_n    = sum(1 for r in report if r["status"] == "review")
    fail_n   = sum(1 for r in report if r["status"] == "fail")
    forced_n = sum(1 for r in report if r["status"] == "forced")
    incons_n = sum(1 for r in report if not r.get("consistency_ok", True))

    print(
        f"\nResult: {ok_n} OK | {rev_n} REVIEW | {fail_n} FAIL"
        + (f" | {forced_n} FORCED" if forced_n else "")
        + (f" | {incons_n} consistency warnings" if incons_n else "")
    )
    if has_failures:
        print(
            f"⚠ {forced_n} clip(s) low-confidence — "
            f"best-guess audio written. Review FORCED clips before delivery."
        )
    print("DONE")
    print(final_path)
    print(report_path)
    print(summary_path)
    print(manifest_path)
    if aaf_written:
        print(aaf_out_path)
        print(deliver_aaf_path)
    print(deliver_final_path)
    print(deliver_summary_path)


if __name__ == "__main__":
    main()
