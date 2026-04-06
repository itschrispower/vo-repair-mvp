from pathlib import Path
import aaf2

job = Path("VO_JOB_01")
aaf_path = next(job.glob("*.aaf"))
out_dir = job / "extract_aaf_audio_test"
out_dir.mkdir(exist_ok=True)

print(f"AAF: {aaf_path.name}\n")

found = False

with aaf2.open(str(aaf_path), "r") as f:
    for mob in f.content.mobs:
        mob_type = mob.__class__.__name__
        mob_name = getattr(mob, "name", "")

        descriptor = getattr(mob, "descriptor", None)
        if mob_type != "SourceMob":
            continue
        if descriptor is None:
            continue
        if descriptor.__class__.__name__ != "PCMDescriptor":
            continue

        print(f"FOUND PCM SOURCE MOB: {mob_name}")

        essence = getattr(mob, "essence", None)
        if essence is None:
            print("  No essence object")
            continue

        out_path = out_dir / f"{mob_name}_extracted.bin"

        try:
            data = essence.read()
            with open(out_path, "wb") as out_f:
                out_f.write(data)
            print(f"  WROTE: {out_path}")
            found = True
        except Exception as e:
            print(f"  READ FAILED: {e}")

if not found:
    print("NO PCM AUDIO EXTRACTED")

print("\nDONE")