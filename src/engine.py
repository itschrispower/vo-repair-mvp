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

SR = 48000
FPS = 25.0
FAIL_THRESHOLD = 0.75
REVIEW_THRESHOLD = 0.90


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


def norm(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    if x.size == 0:
        return x
    x -= np.mean(x)
    peak = np.max(np.abs(x))
    if peak > 0:
        x /= peak
    return x


def match_clip(full: np.ndarray, clip: np.ndarray) -> tuple[int, float]:
    full_n = norm(full)
    clip_n = norm(clip)

    corr = np.correlate(full_n, clip_n, mode="valid")
    match_start = int(np.argmax(corr))

    matched = full_n[match_start:match_start + len(clip_n)]
    denom = np.linalg.norm(matched) * np.linalg.norm(clip_n)
    confidence = 0.0 if denom == 0 else float(corr[match_start] / denom)

    return match_start, confidence


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
    audio, sr = sf.read(str(aaf_reference_path))
    if sr != SR:
        raise ValueError(f"aaf_reference.wav must be {SR} Hz, got {sr}")

    auto_ref_dir.mkdir(exist_ok=True)
    refs = {}

    for clip_name, start_tc, end_tc in clips:
        start = tc_to_samples(start_tc)
        end = tc_to_samples(end_tc)

        if start < 0 or end <= start or end > len(audio):
            raise ValueError(
                f"Clip {clip_name} falls outside aaf_reference.wav: {start_tc} -> {end_tc}"
            )

        clip_audio = audio[start:end]
        out_path = auto_ref_dir / f"{sanitise_name(clip_name)}_auto_ref.wav"
        sf.write(str(out_path), clip_audio, sr)
        refs[clip_name] = out_path

    return refs


def write_summary(path: Path, report: list[dict], final_path: Path, aaf_path: Path, failed: bool):
    lines = []
    lines.append("VO REPAIR SUMMARY")
    lines.append("")

    for item in report:
        lines.append(f"Clip {item['index']} — {item['status'].upper()}")
        lines.append(f"Name: {item['clip_name']}")
        lines.append(f"Ref: {item['ref_name']}")
        lines.append(f"Placed: {item['timeline_start_tc']} to {item['timeline_end_tc']}")
        lines.append(f"Matched in VOBU at: {item['source_match_sec']:.6f}s")
        lines.append(f"Confidence: {item['confidence']:.4f}")
        lines.append(f"Rebuilt clip: {item['rebuilt_file']}")
        lines.append("")

    if failed:
        lines.append("Final output was NOT written because one or more clips failed validation.")
        lines.append("AAF output was NOT written.")
    else:
        lines.append(f"Final output written: {final_path}")
        lines.append(f"AAF output written: {aaf_path}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_embedded_aaf(manifest: list[dict], out_aaf: Path):
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


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python3 src/engine.py <job_folder>")

    base = Path(sys.argv[1]).resolve()
    job_label = detect_job_label(base)

    vo_path = base / "audio" / "VOBU_48k.wav"
    aaf_ref_path = base / "audio" / "aaf_reference.wav"
    positions_path = base / "positions.txt"

    out_dir = base / "rebuild_audio"
    check_dir = base / "match_check"
    auto_ref_dir = base / "auto_refs"
    rebuilt_clips_dir = out_dir / "rebuilt_clips"
    deliver_dir = base / "deliverables"

    final_path = out_dir / f"{job_label}_final.wav"
    report_path = out_dir / f"{job_label}_report.json"
    summary_path = out_dir / f"{job_label}_summary.txt"
    manifest_path = out_dir / f"{job_label}_manifest.json"
    aaf_path = out_dir / f"{job_label}_rebuilt.aaf"

    deliver_final_path = deliver_dir / f"{job_label}_final.wav"
    deliver_summary_path = deliver_dir / f"{job_label}_summary.txt"
    deliver_aaf_path = deliver_dir / f"{job_label}_rebuilt.aaf"

    if not vo_path.exists():
        raise FileNotFoundError(f"Missing VO file: {vo_path}")
    if not aaf_ref_path.exists():
        raise FileNotFoundError(f"Missing aaf_reference.wav: {aaf_ref_path}")
    if not positions_path.exists():
        raise FileNotFoundError(f"Missing positions file: {positions_path}")

    out_dir.mkdir(exist_ok=True)
    check_dir.mkdir(exist_ok=True)
    rebuilt_clips_dir.mkdir(exist_ok=True)
    deliver_dir.mkdir(exist_ok=True)

    vo, vo_sr = librosa.load(str(vo_path), sr=SR)
    if vo_sr != SR:
        raise ValueError(f"VO file must be {SR} Hz")

    clips = load_positions(positions_path)
    auto_refs = build_auto_refs(aaf_ref_path, clips, auto_ref_dir)

    max_end = max(tc_to_samples(end_tc) for _, _, end_tc in clips)
    output = np.zeros(max_end, dtype=np.float32)

    report = []
    manifest = []
    failed = False

    for i, (clip_name, start_tc, end_tc) in enumerate(clips, start=1):
        ref_path = auto_refs[clip_name]

        ref_clip, ref_sr = librosa.load(str(ref_path), sr=SR)
        if ref_sr != SR:
            raise ValueError(f"Reference clip must be {SR} Hz: {ref_path.name}")

        match_start, confidence = match_clip(vo, ref_clip)

        out_start = tc_to_samples(start_tc)
        out_end = tc_to_samples(end_tc)
        target_len = out_end - out_start

        vo_clip = vo[match_start:match_start + target_len]

        if len(vo_clip) < target_len:
            padded = np.zeros(target_len, dtype=np.float32)
            padded[:len(vo_clip)] = vo_clip
            vo_clip = padded

        preview_path = check_dir / f"match_{sanitise_name(clip_name)}.wav"
        sf.write(str(preview_path), vo_clip[:target_len], SR)

        rebuilt_clip_path = rebuilt_clips_dir / f"{sanitise_name(clip_name)}_rebuilt.wav"
        sf.write(str(rebuilt_clip_path), vo_clip[:target_len], SR)

        if confidence >= REVIEW_THRESHOLD:
            status = "ok"
            output[out_start:out_end] = vo_clip[:target_len]
        elif confidence >= FAIL_THRESHOLD:
            status = "review"
            output[out_start:out_end] = vo_clip[:target_len]
        else:
            status = "fail"
            failed = True

        item = {
            "index": i,
            "clip_name": clip_name,
            "ref_name": ref_path.name,
            "timeline_start_tc": start_tc,
            "timeline_end_tc": end_tc,
            "timeline_start_sec": round(out_start / SR, 6),
            "timeline_end_sec": round(out_end / SR, 6),
            "source_match_sec": round(match_start / SR, 6),
            "confidence": round(confidence, 6),
            "status": status,
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
                    "match_sec": round(match_start / SR, 6),
                    "match_samples": int(match_start),
                    "duration_sec": round(target_len / SR, 6),
                    "duration_samples": int(target_len),
                },
                "rebuilt_file": str(rebuilt_clip_path),
                "confidence": round(confidence, 6),
                "status": status,
            }
        )

        print(
            f"{clip_name} -> {ref_path.name} -> "
            f"{match_start / SR:.6f}s | conf={confidence:.4f} | {status.upper()}"
        )

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if not failed:
        sf.write(str(final_path), output, SR)
        write_embedded_aaf(manifest, aaf_path)

    write_summary(summary_path, report, final_path, aaf_path, failed)

    if failed:
        print("STOPPED: one or more clips failed validation")
        print(report_path)
        print(summary_path)
        print(manifest_path)
        return

    shutil.copy2(final_path, deliver_final_path)
    shutil.copy2(summary_path, deliver_summary_path)
    shutil.copy2(aaf_path, deliver_aaf_path)

    print("DONE")
    print(final_path)
    print(report_path)
    print(summary_path)
    print(manifest_path)
    print(aaf_path)
    print(deliver_final_path)
    print(deliver_summary_path)
    print(deliver_aaf_path)


if __name__ == "__main__":
    main()