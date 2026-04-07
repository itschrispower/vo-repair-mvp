from pathlib import Path

FPS = 25.0
SR = 48000


def find_single_aaf(directory: Path) -> Path:
    aafs = sorted(directory.glob("*.aaf"))
    if not aafs:
        raise FileNotFoundError(f"No AAF found in {directory}")
    if len(aafs) > 1:
        raise ValueError(
            "Expected exactly one AAF, found: "
            + ", ".join(p.name for p in aafs)
        )
    return aafs[0]
