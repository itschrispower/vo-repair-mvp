from __future__ import annotations

import numpy as np


def find_best_match(
    reference_clip: np.ndarray,
    target_audio: np.ndarray,
    sample_rate: int,
    expected_start: int | None = None,
    search_radius_seconds: float = 15.0,
) -> dict:
    """
    Find where reference_clip exists inside target_audio.

    Parameters:
        reference_clip: numpy array (mono)
        target_audio: numpy array (mono)
        sample_rate: int
        expected_start: optional sample index to search around
        search_radius_seconds: window size for search

    Returns:
        dict with:
            match_sample
            match_time_sec
            confidence
    """

    ref = reference_clip.astype(np.float32)
    tgt = target_audio.astype(np.float32)

    if ref.size == 0 or tgt.size == 0:
        raise ValueError("Empty audio provided")

    # Normalize (important for stability)
    ref = ref - np.mean(ref)
    tgt = tgt - np.mean(tgt)

    ref_norm = np.linalg.norm(ref)
    if ref_norm == 0:
        raise ValueError("Reference clip is silent")

    # Define search window
    if expected_start is not None:
        radius = int(search_radius_seconds * sample_rate)

        start = max(0, expected_start - radius)
        end = min(len(tgt), expected_start + radius)

        search_audio = tgt[start:end]
        offset_base = start
    else:
        search_audio = tgt
        offset_base = 0

    if len(search_audio) < len(ref):
        raise ValueError("Search window smaller than reference clip")

    # FFT cross-correlation
    n = len(search_audio) + len(ref)
    n = 1 << (n - 1).bit_length()

    ref_fft = np.fft.rfft(ref, n=n)
    tgt_fft = np.fft.rfft(search_audio, n=n)

    corr = np.fft.irfft(tgt_fft * np.conj(ref_fft), n=n)

    # Valid region
    valid = corr[: len(search_audio) - len(ref)]
    peak_idx = int(np.argmax(np.abs(valid)))
    peak_val = float(np.abs(valid[peak_idx]))

    # Confidence
    tgt_segment = search_audio[peak_idx : peak_idx + len(ref)]
    tgt_norm = np.linalg.norm(tgt_segment)

    if tgt_norm == 0:
        confidence = 0.0
    else:
        confidence = peak_val / (ref_norm * tgt_norm)

    match_sample = peak_idx + offset_base
    match_time = match_sample / sample_rate

    return {
        "match_sample": int(match_sample),
        "match_time_sec": float(match_time),
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
    }