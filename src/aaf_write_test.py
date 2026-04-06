import json
from pathlib import Path
import shutil

job = Path("VO_JOB_01")
manifest_path = job / "clip_output_test" / "manifest.json"
out_dir = job / "aaf_write_test"
out_dir.mkdir(exist_ok=True)

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

aaf_text = []
aaf_text.append("AAF WRITE TEST")
aaf_text.append("")
aaf_text.append("This is not a real AAF yet.")
aaf_text.append("It is the exact one-track structure we will write into AAF next.")
aaf_text.append("")

for item in manifest:
    aaf_text.append(f"Clip {item['index']}")
    aaf_text.append(f"Name: {item['clip_name']}")
    aaf_text.append(f"Timeline start: {item['timeline']['start_tc']}")
    aaf_text.append(f"Timeline end: {item['timeline']['end_tc']}")
    aaf_text.append(f"Timeline start sec: {item['timeline']['start_sec']}")
    aaf_text.append(f"Duration sec: {item['timeline']['duration_sec']}")
    aaf_text.append(f"Source file: {item['rebuilt_file']}")
    aaf_text.append(f"Confidence: {item['confidence']}")
    aaf_text.append("")

(out_dir / "aaf_structure_preview.txt").write_text(
    "\n".join(aaf_text) + "\n",
    encoding="utf-8"
)

clips_dir = out_dir / "linked_clips"
clips_dir.mkdir(exist_ok=True)

for item in manifest:
    src = Path(item["rebuilt_file"])
    dst = clips_dir / src.name
    shutil.copy2(src, dst)

print("DONE")
print(out_dir / "aaf_structure_preview.txt")
print(clips_dir)