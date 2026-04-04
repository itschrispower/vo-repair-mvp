import aaf2

AAF_PATH = "3.aaf"

with aaf2.open(AAF_PATH, 'r') as f:
    mob = next(f.content.toplevel())

    print("Opened AAF")

    for slot in mob.slots:
        segment = slot.segment

        if hasattr(segment, 'components'):
            print("Found sequence with components")

            for comp in segment.components:
                print("TYPE:", comp.__class__.__name__)

                if comp.__class__.__name__ == "SourceClip":
                    print("FOUND SOURCE CLIP")

                    start = getattr(comp, "start", 0)
                    length = getattr(comp, "length", 0)

                    print("AAF clip:", start, length)
