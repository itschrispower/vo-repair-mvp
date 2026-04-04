from __future__ import annotations

import aaf2

AAF_PATH = "3.aaf"


def main():
    with aaf2.open(AAF_PATH, "r") as f:
        for mob in f.content.mobs:
            mob_name = getattr(mob, "name", None)
            mob_usage = getattr(mob, "usage", None)

            print("=" * 60)
            print(f"MOB NAME: {mob_name}")
            print(f"USAGE: {mob_usage}")
            print(f"CLASS: {mob.__class__.__name__}")

            descriptor = getattr(mob, "descriptor", None)
            if descriptor is not None:
                print(f"DESCRIPTOR CLASS: {descriptor.__class__.__name__}")

                try:
                    for prop in descriptor.properties():
                        print(f"  {prop.name}: {prop.value}")
                except Exception as exc:
                    print(f"  Could not read descriptor properties: {exc}")

            for slot in getattr(mob, "slots", []):
                print(f"  SLOT ID: {getattr(slot, 'slot_id', None)}")
                segment = getattr(slot, "segment", None)
                if segment is None:
                    continue

                print(f"  SEGMENT CLASS: {segment.__class__.__name__}")

                if hasattr(segment, "components"):
                    for comp in segment.components:
                        print(f"    COMPONENT: {comp.__class__.__name__}")
                        print(f"      length: {getattr(comp, 'length', None)}")
                        print(f"      start_time: {getattr(comp, 'start_time', None)}")

                        source = getattr(comp, "source", None)
                        if source is not None:
                            print(f"      source class: {source.__class__.__name__}")
                            print(f"      source start: {getattr(source, 'start', None)}")

                            src_mob = getattr(source, "mob", None)
                            if src_mob is not None:
                                print(f"      source mob name: {getattr(src_mob, 'name', None)}")
                                print(f"      source mob class: {src_mob.__class__.__name__}")

                                src_desc = getattr(src_mob, "descriptor", None)
                                if src_desc is not None:
                                    print(f"      source mob descriptor: {src_desc.__class__.__name__}")
                                    try:
                                        for prop in src_desc.properties():
                                            print(f"        {prop.name}: {prop.value}")
                                    except Exception as exc:
                                        print(f"        Could not read source descriptor properties: {exc}")


if __name__ == "__main__":
    main()