from pathlib import Path
import sys
import re
import aaf2

AAF_RATE = 25


def resolve_job_root() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).resolve()
    return Path(__file__).resolve().parent.parent


def find_aaf(job_root: Path) -> Path:
    aafs = sorted(job_root.glob("*.aaf"))
    if not aafs:
        raise FileNotFoundError("No .aaf file found in job root")
    if len(aafs) > 1:
        raise ValueError(f"Multiple AAF files found: {[p.name for p in aafs]}")
    return aafs[0]


def aaf_units_to_tc(units: int) -> str:
    secs = units // AAF_RATE
    frames = units % AAF_RATE
    return f"{secs:02d}.{frames:02d}"


def extract_aaf_sections(aaf_path: Path):
    sections = []

    with aaf2.open(str(aaf_path), "r") as f:
        for mob in f.content.mobs:
            if getattr(mob, "usage", None) != "Usage_TopLevel":
                continue

            for slot in getattr(mob, "slots", []):
                seg = slot.segment
                if seg.__class__.__name__ != "Sequence":
                    continue

                pos = 0
                for comp in seg.components:
                    length = int(getattr(comp, "length", 0))

                    if comp.__class__.__name__ == "OperationGroup" and length > 1:
                        sections.append((pos, pos + length, length, comp.__class__.__name__))

                    pos += length

    return sections


def extract_pt_sections(txt_path: Path):
    clips = []

    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            if "clip_" not in line:
                continue

            parts = re.split(r"\s+", line.strip())
            if len(parts) < 6:
                continue

            clip_name = parts[2]
            start_tc = parts[3]
            end_tc = parts[4]
            duration_tc = parts[5]

            if not clip_name.startswith("clip_"):
                continue

            clips.append((clip_name, start_tc, end_tc, duration_tc))

    return clips


def main():
    job = resolve_job_root()
    aaf_path = find_aaf(job)

    txts = sorted(job.glob("*.txt"))
    if not txts:
        raise FileNotFoundError("No .txt session info file found in job root")
    if len(txts) > 1:
        print("Warning: multiple .txt files found, using:", txts[0].name)
    txt_path = txts[0]

    aaf_sections = extract_aaf_sections(aaf_path)
    pt_sections = extract_pt_sections(txt_path)

    print("AAF:")
    for i, (start, end, length, kind) in enumerate(aaf_sections, start=1):
        print(
            f"clip_{i}: {aaf_units_to_tc(start)} -> {aaf_units_to_tc(end)} "
            f"(len {aaf_units_to_tc(length)}) [{kind}]"
        )

    print("\nPT TXT:")
    for clip_name, start_tc, end_tc, duration_tc in pt_sections:
        print(f"{clip_name}: {start_tc} -> {end_tc} (len {duration_tc})")


if __name__ == "__main__":
    main()