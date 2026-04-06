import json
from pathlib import Path

import aaf2

SR = 48000

job = Path("VO_JOB_01")
manifest_path = job / "clip_output_test" / "manifest.json"
out_dir = job / "aaf_write_test"
out_dir.mkdir(exist_ok=True)

out_aaf = out_dir / "VOREPAIR_rebuilt_test.aaf"

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))


def sec_to_samples(sec: float) -> int:
    return int(round(sec * SR))


with aaf2.open(str(out_aaf), "w") as f:
    # top-level timeline
    comp = f.create.CompositionMob("VOREPAIR_REBUILT")
    comp.usage = "Usage_TopLevel"
    f.content.mobs.append(comp)

    # one mono sound track at 48k edit rate
    slot = comp.create_sound_slot(edit_rate=SR)
    seq = slot.segment  # create_sound_slot already gives an empty Sequence('sound')

    current_pos = 0

    for item in manifest:
        timeline_start = sec_to_samples(item["timeline"]["start_sec"])
        duration = sec_to_samples(item["timeline"]["duration_sec"])
        rebuilt_file = Path(item["rebuilt_file"]).resolve()

        # add filler if there is a gap before this clip
        if timeline_start > current_pos:
            filler = f.create.Filler("sound")
            filler.length = timeline_start - current_pos
            seq.components.append(filler)
            current_pos = timeline_start

        # create a MasterMob for this rebuilt clip and embed the wav
        mob_name = f"clip_{item['index']}_{item['clip_name']}"
        master_mob = f.create.MasterMob(mob_name)
        f.content.mobs.append(master_mob)

        # import embedded PCM essence at 48k
        essence_slot = master_mob.import_audio_essence(str(rebuilt_file), SR)

        # create a source clip that references the imported essence
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

print("DONE")
print(out_aaf)