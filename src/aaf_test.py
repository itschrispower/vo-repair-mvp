from pathlib import Path
import sys
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


def extract_sections(aaf_path: Path):
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
                        sections.append((pos, pos + length, length))

                    pos += length

    return sections


def main():
    job = resolve_job_root()
    aaf_path = find_aaf(job)

    sections = extract_sections(aaf_path)

    print("AAF:")
    for i, (start, end, length) in enumerate(sections, start=1):
        print(
            f"clip_{i}: {aaf_units_to_tc(start)} -> {aaf_units_to_tc(end)} "
            f"(len {aaf_units_to_tc(length)})"
        )


if __name__ == "__main__":
    main()