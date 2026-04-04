from __future__ import annotations

import numpy as np


def compute_energy(audio: np.ndarray) -> float:
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def find_best_match(
    reference_clip: np.ndarray,
    target_audio: np.ndarray,
    sample_rate: int,
    expected_start: int | None = None,
    search_window_sec: float = 20.0,
    min_energy: float = 0.01,
    min_length_samples: int = 2000,
    min_confidence: float = 0.5,
) -> dict | None:
    """
    Find where reference_clip exists inside target_audio.

    Returns:
        {
            "match_sample": int,
            "match_time_sec": float,
            "confidence": float,
        }

    Returns None if the clip is too short, too quiet, or the match is too weak.
    """

    if reference_clip is None or target_audio is None:
        return None

    if len(reference_clip) < min_length_samples:
        return None

    if len(target_audio) == 0:
        return None

    ref = np.asarray(reference_clip, dtype=np.float32)
    tgt_full = np.asarray(target_audio, dtype=np.float32)

    if compute_energy(ref) < min_energy:
        return None

    if expected_start is not None:
        window = int(search_window_sec * sample_rate)
        search_start = max(0, expected_start - window)
        search_end = min(len(tgt_full), expected_start + window)
        tgt = tgt_full[search_start:search_end]
        offset_base = search_start
    else:
        tgt = tgt_full
        offset_base = 0

    if len(tgt) < len(ref):
        return None

    ref = ref - np.mean(ref)
    tgt = tgt - np.mean(tgt)

    ref_norm = float(np.linalg.norm(ref))
    if ref_norm == 0.0:
        return None

    corr = np.correlate(tgt, ref, mode="valid")
    if len(corr) == 0:
        return None

    peak_idx = int(np.argmax(np.abs(corr)))
    peak_value = float(np.abs(corr[peak_idx]))

    target_segment = tgt[peak_idx : peak_idx + len(ref)]
    tgt_norm = float(np.linalg.norm(target_segment))
    if tgt_norm == 0.0:
        return None

    confidence = peak_value / (ref_norm * tgt_norm)
    confidence = float(np.clip(confidence, 0.0, 1.0))

    if confidence < min_confidence:
        return None

    match_sample = peak_idx + offset_base
    match_time_sec = match_sample / float(sample_rate)

    return {
        "match_sample": int(match_sample),
        "match_time_sec": float(match_time_sec),
        "confidence": confidence,
    }