# VO Repair Pipeline

Production-grade audio clip matching and reconstruction for Pro Tools VO sessions.

## Quick Start

### macOS
Double-click `local_run.command` to launch the GUI.

### Linux/Unix
```bash
chmod +x local_run.sh
./local_run.sh
```

### Windows
Double-click `local_run.bat` to launch the GUI.

---

## What It Does

Analyzes a Pro Tools session and reconstructs voice-over clips from a backup recording:

1. **Position Extraction**: Reads AAF file to locate all clips on the timeline
2. **Clip Matching**: Uses sliding Pearson correlation (NCC) to find each clip in a VOBU (VO backup)
3. **Confidence Scoring**: Multi-stage validation with peak sharpness, method agreement, and consistency checks
4. **Audio Reconstruction**: Extracts matched clips, applies polarity correction, and assembles into a final WAV
5. **Report & AAF**: Generates JSON report with per-clip confidence and rebuilds AAF for Pro Tools import

### Key Features

- **Robust matching**: Coarse→fine multi-resolution search (4 kHz → 48 kHz)
- **Bandpass filtering**: Speech-range emphasis (120–7500 Hz) for cleaner correlations
- **Confidence weighting**: 70% NCC score + 20% peak stability + 10% bandpass consistency
- **Always output**: Even low-confidence clips are extracted (marked as FORCED with warnings)
- **Drift tracking**: Learns position drift across clips to tighten search windows
- **Two-pass retry**: Expands search window if first pass is uncertain

---

## Input Files

Place in a folder with this structure:

```
VO_JOB_01/
  INPUT/
    your_session.aaf              (Pro Tools AAF export)
    audio/
      aaf_reference.wav           (full-session bounce from Pro Tools)
      VOBU_48k.wav                (VO backup recording at 48 kHz)
```

**Requirements:**
- All audio files must be **48 kHz**, mono or stereo
- AAF must be a valid Pro Tools export (not Avid Media Composer)

---

## Output Files

Saved to `OUTPUT/` in the same folder:

- **VORepair_final.wav** — Reconstructed audio with all clips
- **VORepair_rebuilt.aaf** — New AAF file for Pro Tools import (segments point to final.wav)
- **VORepair_report.json** — Detailed per-clip results with confidence, NCC, status
- **VORepair_summary.txt** — Human-readable summary

---

## GUI Usage

1. Select the AAF file
2. Select the AAF Full Bounce WAV
3. Select the VO Backup (VOBU) WAV
4. Choose output folder
5. Click **Process**

Results stream live in the log area. Color-coded status:
- **✓ OK** — High confidence (≥0.7), clean match
- **~ REVIEW** — Medium confidence, check report before use
- **⚡ FORCED** — Low confidence, extracted as best-guess (confidence < 0.5 or NCC < 0.35)
- **✗ FAIL** — Could not locate clip (rare, see log for details)

---

## Dependencies

Installs automatically via the launcher script:

```
numpy
scipy
pyaaf2
librosa
soundfile
```

If auto-install fails, install manually:
```bash
pip install -r requirements.txt
python3 run_app.py
```

---

## CLI Usage (Advanced)

```bash
python3 src/run_job.py /path/to/VO_JOB_01
```

---

## Report Format

JSON output includes per-clip:
- index, name, timeline positions (timecode + samples)
- status: ok, review, forced, fail
- confidence: 0.0–1.0
- vobu_match_sec/samples: where clip was found
- raw_ncc, stability: correlation details

---

Built with numpy, scipy, librosa, pyaaf2, soundfile. GUI uses Tkinter.
