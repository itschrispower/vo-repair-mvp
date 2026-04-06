from pathlib import Path
import aaf2

job = Path("VO_JOB_01")
aaf_path = next(job.glob("*.aaf"))

print(f"AAF: {aaf_path.name}\n")

with aaf2.open(str(aaf_path), "r") as f:
    print("MOBS:\n")

    for mob in f.content.mobs:
        mob_name = getattr(mob, "name", None)
        mob_usage = getattr(mob, "usage", None)
        mob_type = mob.__class__.__name__

        print(f"MOB: {mob_name} | type={mob_type} | usage={mob_usage}")

        descriptor = getattr(mob, "descriptor", None)
        if descriptor is not None:
            print(f"  descriptor: {descriptor.__class__.__name__}")

        for i, slot in enumerate(getattr(mob, "slots", []), start=1):
            seg = slot.segment
            seg_type = seg.__class__.__name__
            seg_len = getattr(seg, "length", None)

            print(f"  SLOT {i}: {seg_type} length={seg_len}")

            if seg_type == "Sequence":
                for j, comp in enumerate(seg.components, start=1):
                    comp_type = comp.__class__.__name__
                    comp_len = getattr(comp, "length", None)
                    print(f"    COMP {j}: {comp_type} length={comp_len}")

                    if comp_type == "OperationGroup":
                        inner = getattr(comp, "segments", [])
                        for k, inner_seg in enumerate(inner, start=1):
                            inner_type = inner_seg.__class__.__name__
                            inner_len = getattr(inner_seg, "length", None)
                            print(f"      INNER {k}: {inner_type} length={inner_len}")

    print("\nDONE")