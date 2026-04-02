from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st

from aaf_utils import process_aaf_stub, validate_aaf_file
from audio_utils import estimate_drift, estimate_offset, load_wav


st.set_page_config(page_title="VO Repair Analyzer", layout="centered")
st.title("VO Repair Analyzer")
st.caption("Compare source and target WAVs, validate AAF, and estimate offset + drift.")


def _persist_upload(uploaded_file, suffix: str) -> str:
    if uploaded_file is None:
        raise ValueError("No file was provided.")

    temp_dir = Path(tempfile.gettempdir()) / "vo_repair_mvp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(uploaded_file.name).stem.replace(" ", "_")
    out_path = temp_dir / f"{safe_name}_{uploaded_file.size}{suffix}"
    out_path.write_bytes(uploaded_file.getbuffer())
    return str(out_path)


def _show_wav_metrics(label: str, wav_info: dict) -> None:
    col1, col2 = st.columns(2)
    col1.metric(f"{label} Duration", f"{wav_info['duration_seconds']:.3f} s")
    col2.metric(f"{label} Sample Rate", f"{wav_info['sample_rate']} Hz")


def _movement_text(offset_samples: int, offset_ms: float) -> str:
    if offset_samples > 0:
        return f"Move TARGET earlier by {offset_samples} samples ({offset_ms:.3f} ms)"
    if offset_samples < 0:
        return f"Move TARGET later by {abs(offset_samples)} samples ({abs(offset_ms):.3f} ms)"
    return "No start offset correction needed"


def _stretch_text(stretch_percent: float) -> str:
    if abs(stretch_percent) < 0.01:
        return "No stretch correction needed"
    if stretch_percent > 0:
        return f"Stretch TARGET longer by {stretch_percent:.4f}%"
    return f"Compress TARGET shorter by {abs(stretch_percent):.4f}%"


st.subheader("Select Files")
source_wav = st.file_uploader("1) Source / Reference WAV", type=["wav"], key="source")
target_wav = st.file_uploader("2) Target / Repaired WAV", type=["wav"], key="target")
aaf_file = st.file_uploader("3) AAF file", type=["aaf"], key="aaf")

if source_wav and target_wav and aaf_file:
    st.divider()
    st.subheader("Validation & Analysis")

    try:
        source_path = _persist_upload(source_wav, ".wav")
        target_path = _persist_upload(target_wav, ".wav")
        aaf_path = _persist_upload(aaf_file, ".aaf")

        is_valid_aaf, aaf_message = validate_aaf_file(aaf_path)
        if not is_valid_aaf:
            st.error(f"AAF validation failed: {aaf_message}")
            st.stop()

        st.success(f"AAF validation passed: {aaf_message}")

        aaf_stub_result = process_aaf_stub(aaf_path)
        st.info(
            f"AAF stub processed file: {os.path.basename(aaf_stub_result['aaf_path'])} "
            f"({aaf_stub_result['status']})"
        )

        source_info = load_wav(source_path)
        target_info = load_wav(target_path)

        _show_wav_metrics("Source WAV", source_info)
        _show_wav_metrics("Target WAV", target_info)

        alignment = estimate_offset(source_info, target_info)
        drift = estimate_drift(source_info, target_info)

        st.markdown("### Alignment Results")
        col1, col2, col3 = st.columns(3)
        col1.metric("Offset (samples)", f"{alignment['offset_samples']}")
        col2.metric("Offset (milliseconds)", f"{alignment['offset_milliseconds']:.3f} ms")
        col3.metric("Confidence score", f"{alignment['confidence']:.4f}")

        st.markdown("### Drift Results")
        col4, col5 = st.columns(2)
        col4.metric("Drift (samples)", f"{drift['drift_samples']}")
        col5.metric("Stretch (%)", f"{drift['stretch_percent']:.4f}")

        st.markdown("### What to do in Pro Tools")
        st.write(f"- {_movement_text(alignment['offset_samples'], alignment['offset_milliseconds'])}")
        st.write(f"- {_stretch_text(drift['stretch_percent'])}")

        st.markdown("### Detail")
        st.write(
            f"Start-window offset: {drift['start_offset_samples']} samples | "
            f"End-window offset: {drift['end_offset_samples']} samples"
        )

    except ValueError as exc:
        st.error(f"Input error: {exc}")
    except RuntimeError as exc:
        st.error(f"Analysis error: {exc}")
    except Exception as exc:
        st.exception(exc)
else:
    st.info("Please select all three files to run validation and analysis.")