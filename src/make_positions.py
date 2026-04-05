from pathlib import Path
import sys

def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python3 src/make_positions.py <job_folder>")

    job = Path(sys.argv[1]).resolve()
    csv_path = job / "timeline.csv"
    out_path = job / "positions.txt"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing file: {csv_path}")

    lines_out = []

    with open(csv_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                raise ValueError(f"Bad CSV line {i}: {line}")

            ref_name, start_tc, end_tc = parts
            lines_out.append(f"{ref_name} {start_tc} {end_tc}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_out) + "\n")

    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()