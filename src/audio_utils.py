from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


_INT24_MAX = float(2**23)


def _decode_pcm(raw_frames: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        pcm = np.frombuffer(raw_frames, dtype=np.uint8).astype(np.float32)
        return (pcm - 128.0) / 128.0

    if sample_width == 2:
        pcm = np.frombuffer(raw_frames, dtype="<i2").astype(np.float32)
        return pcm / float(2**15)

    if sample_width == 3:
        data = np.frombuffer(raw_frames, dtype=np.uint8).reshape(-1, 3)
        pcm = (
            data[:, 0].astype(np.int32)
            | (data[:, 1].astype(np.int32) << 8)
            | (data[:, 2].astype(np.int32) << 16)
        )
        sign = pcm & 0x800000
        pcm = pcm - (sign << 1)
        return pcm.astype(np.float32) / _INT24_MAX

    if sample_width == 4:
        pcm = np.frombuffer(raw_frames, dtype="<i4").astype(np.float32)
        return pcm / float(2**31)

    raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")


def load_wav(path: str) -> dict:
    wav_path = Path(path)
    if not wav_path.exists():
        raise ValueError(f"WAV file does not exist: {path}")

    try:
        with wave.open(str(wav_path), "rb") as wav:
            channels = wav.getnchannels()
            sample_rate = wav.getframerate()
            sample_width = wav.getsampwidth()
            frame_count = wav.getnframes()
            raw = wav.readframes(frame_count)
    except wave.Error as exc:
        raise ValueError(f"Invalid WAV file '{path}': {exc}") from exc

    if channels < 1:
        raise ValueError(f"WAV file has invalid channel count: {channels}")
    if sample_rate <= 0:
        raise ValueError(f"WAV file has invalid sample rate: {sample_rate}")

    samples = _decode_pcm(raw, sample_width)
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)

    duration_seconds = len(samples) / float(sample_rate)

    return {
        "path": str(wav_path),
        "sample_rate": int(sample_rate),
        "channels": int(channels),
        "sample_width": int(sample_width),
        "samples": samples.astype(np.float32),
        "duration_seconds": float(duration_seconds),
    }


def _fft_cross_correlation(reference: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_ref = len(reference)
    n_tgt = len(target)
    n = 1 << (n_ref + n_tgt - 1).bit_length()

    ref_fft = np.fft.rfft(reference, n=n)
    tgt_fft = np.fft.rfft(target, n=n)
    corr = np.fft.irfft(tgt_fft * np.conj(ref_fft), n=n)

    corr_full = np.concatenate((corr[-(n_tgt - 1):], corr[:n_ref]))
    lags = np.arange(-(n_tgt - 1), n_ref)
    return corr_full, lags


def estimate_offset(reference_wav: dict, target_wav: dict) -> dict:
    ref_sr = int(reference_wav["sample_rate"])
    tgt_sr = int(target_wav["sample_rate"])
    if ref_sr != tgt_sr:
        raise ValueError(
            f"Sample rates must match for alignment (source={ref_sr}, target={tgt_sr})."
        )

    ref = np.asarray(reference_wav["samples"], dtype=np.float32)
    tgt = np.asarray(target_wav["samples"], dtype=np.float32)

    if ref.size == 0 or tgt.size == 0:
        raise ValueError("One or both WAV files contain no audio samples.")

    ref = ref - np.mean(ref)
    tgt = tgt - np.mean(tgt)

    ref_norm = np.linalg.norm(ref)
    tgt_norm = np.linalg.norm(tgt)
    if ref_norm == 0 or tgt_norm == 0:
        raise ValueError("One or both WAV files are effectively silent.")

    corr, lags = _fft_cross_correlation(ref, tgt)

    peak_idx = int(np.argmax(np.abs(corr)))
    offset_samples = int(lags[peak_idx])
    peak_value = float(np.abs(corr[peak_idx]))

    confidence = peak_value / (ref_norm * tgt_norm)
    confidence = float(np.clip(confidence, 0.0, 1.0))

    offset_ms = (offset_samples / ref_sr) * 1000.0

    return {
        "offset_samples": offset_samples,
        "offset_milliseconds": float(offset_ms),
        "confidence": confidence,
    }