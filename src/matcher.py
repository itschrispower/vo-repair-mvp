"""
src/matcher.py

Robust VOBU clip matcher for VO Repair.

Replaces the original np.correlate approach with:
  - Proper sliding Pearson NCC via FFT (O(N log N)) — not biased by window energy
  - Bandpass speech-range comparison fused with wideband NCC
  - Coarse→fine multi-resolution search for efficiency on long VOBUs
  - Edge trimming (25 ms from each end) to remove fade contamination before matching
  - Polarity inversion detection with minimum-delta threshold (6 %) at the fine stage
  - Peak sharpness / stability scoring — flat NCC peaks lower confidence
  - Method agreement cross-validation (wideband NCC vs RMS-envelope NCC)
  - Short / low-energy clip classification with graduated confidence penalties
  - Expanded ambiguity detection: ratio test + near-equal candidate count +
    spatial far-duplicate detection for repeated phrases
  - DriftTracker: confidence-weighted anchors, weighted regression,
    drift-jump detection before recording anchors
  - Consistency post-pass to flag outlier matches against the global trend

Returned offsets are always relative to the `region` array passed in.
The caller (engine.py) must adjust for trim_samples and search_lo.
"""
from __future__ import annotations

import dataclasses
import logging
from math import gcd
from typing import List, Optional

import numpy as np
from scipy import signal as sp

log = logging.getLogger(__name__)

# ── Sample rates ──────────────────────────────────────────────────────────────
SR_FULL: int = 48_000
SR_COARSE: int = 4_000
COARSE_RATIO: int = SR_FULL // SR_COARSE          # 12

# ── Bandpass limits for speech matching ───────────────────────────────────────
SPEECH_LO_HZ: float = 120.0
SPEECH_HI_HZ: float = 7_500.0

# ── Clip classification thresholds ───────────────────────────────────────────
SHORT_SAMPLES: int = int(0.250 * SR_FULL)          # 250 ms
VERY_SHORT_SAMPLES: int = int(0.080 * SR_FULL)     # 80 ms
SILENCE_DB: float = -52.0
WEAK_DB: float = -40.0

# ── Confidence thresholds (after all penalties applied) ───────────────────────
CONF_OK: float = 0.72
CONF_REVIEW: float = 0.36

# ── Ambiguity ratios and counts ───────────────────────────────────────────────
AMBIG_HARD: float = 0.92       # 2nd-best / best → hard fail of confidence
AMBIG_SOFT: float = 0.80       # 2nd-best / best → soft penalty
NEAR_EQUAL_RATIO: float = 0.85  # candidates within this fraction of best → "near equal"
NEAR_CANDS_AMBIG: int = 3       # ≥ this many near-equal candidates → extra penalty
FAR_DUPLICATE_MIN_S: float = 10.0   # spatial far-duplicate threshold (seconds)

# ── Search / refinement parameters ───────────────────────────────────────────
TOP_K: int = 5
FINE_HALF_S: float = 0.50          # ±s fine refinement window around each coarse peak
DEFAULT_WINDOW_S: float = 45.0     # ±s when no prior drift anchor
NEIGHBOUR_WINDOW_S: float = 12.0    # ±s when anchor available

# ── Edge trimming to remove fade contamination ────────────────────────────────
EDGE_TRIM_MS: float = 25.0          # trim this many ms from each end of ref clip
EDGE_TRIM_MIN_CLIP_MS: float = 120.0  # only trim if clip is longer than this

# ── Stability / peak sharpness ────────────────────────────────────────────────
STABILITY_DELTA_MS: float = 15.0   # ±ms window around NCC peak for sharpness check
MIN_NCC_CANDIDATE: float = 0.25    # ignore candidates below this raw NCC

# ── Polarity flip minimum margin ──────────────────────────────────────────────
POLARITY_FLIP_MIN_DELTA: float = 0.06  # inverted must beat normal by at least this much

# ── DriftTracker anchor gating ────────────────────────────────────────────────
ANCHOR_MIN_CONF: float = 0.58       # below this confidence, skip recording as anchor
ANCHOR_WEIGHT_CONF: float = 0.72    # at this confidence, anchor weight reaches 1.0
DRIFT_JUMP_S: float = 3.0           # deviation > this → possible false-positive jump


# ── Data classes ─────────────────────────────────────────────────────────────
@dataclasses.dataclass
class MatchCandidate:
    offset: int        # sample offset relative to region[0], full SR
    ncc: float         # fused NCC score (before penalties)
    polarity: int      # +1 normal, -1 inverted
    ncc_wb: float = 0.0       # wideband NCC at this offset
    ncc_bp: float = 0.0       # bandpass NCC at this offset
    stability: float = 0.5    # peak sharpness score [0, 1]


@dataclasses.dataclass
class ClipMatchResult:
    offset: int        # sample offset relative to region[0], full SR
    confidence: float  # 0–1, all penalties applied
    status: str        # "ok" | "review" | "fail"
    reasons: List[str]
    candidates: List[MatchCandidate]
    polarity: int      # +1 or -1
    raw_ncc: float = 0.0          # fused NCC before any penalties
    stability: float = 0.5        # NCC peak sharpness of best candidate
    method_agreement: float = 1.0 # wideband vs envelope agreement score [0, 1]
    trim_samples: int = 0         # samples trimmed from each end before matching
    near_cands_count: int = 0     # count of near-equal candidates besides best


# ── Audio utilities ───────────────────────────────────────────────────────────

def rms_db(x: np.ndarray) -> float:
    """RMS energy in dBFS."""
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    return 20.0 * np.log10(max(rms, 1e-10))


def bandpass(
    x: np.ndarray,
    sr: int = SR_FULL,
    lo: float = SPEECH_LO_HZ,
    hi: float = SPEECH_HI_HZ,
) -> np.ndarray:
    """4th-order Butterworth bandpass, speech range. Fails gracefully."""
    nyq = sr / 2.0
    lo_n = lo / nyq
    hi_n = min(hi, nyq * 0.98) / nyq
    if len(x) < 48 or lo_n <= 0.0 or hi_n <= lo_n or hi_n >= 1.0:
        return x.astype(np.float32)
    try:
        sos = sp.butter(4, [lo_n, hi_n], btype="bandpass", output="sos")
        return sp.sosfiltfilt(sos, x.astype(np.float64)).astype(np.float32)
    except Exception as exc:
        log.warning("bandpass failed: %s", exc)
        return x.astype(np.float32)


def downsample(
    x: np.ndarray,
    from_sr: int = SR_FULL,
    to_sr: int = SR_COARSE,
) -> np.ndarray:
    """Polyphase downsample via scipy.signal.resample_poly."""
    if from_sr == to_sr or len(x) == 0:
        return x.astype(np.float32)
    g = gcd(from_sr, to_sr)
    up = to_sr // g
    dn = from_sr // g
    return sp.resample_poly(x.astype(np.float64), up, dn).astype(np.float32)


# ── Core: sliding Pearson NCC ─────────────────────────────────────────────────

def sliding_ncc(haystack: np.ndarray, needle: np.ndarray) -> np.ndarray:
    """
    Per-window Pearson correlation coefficient (proper NCC).

    ncc[i] = Pearson-r between haystack[i:i+M] and needle,
    computed via FFT cross-correlation + running window statistics.

    Key property: because the needle is zero-meaned before correlation,
    the raw FFT result equals sum((h_window - h_window_mean) * n_zm),
    which is the cross-covariance without needing to zero-mean each window
    of the haystack explicitly.

    Returns array of shape (max(0, N-M+1),), values clipped to [-1, 1].
    Complexity: O(N log N).
    """
    N = len(haystack)
    M = len(needle)
    L = N - M + 1
    if L <= 0 or M < 2:
        return np.zeros(max(0, L), dtype=np.float32)

    h = haystack.astype(np.float64)
    n = needle.astype(np.float64)
    n -= n.mean()                   # zero-mean needle
    n_std = float(n.std())
    if n_std < 1e-9:                # constant needle → undefined correlation
        return np.zeros(L, dtype=np.float32)

    # FFT cross-correlation
    raw = sp.correlate(h, n, mode="valid", method="fft")

    # Running window mean and variance of haystack via prefix sums
    cs = np.r_[0.0, np.cumsum(h)]
    cs2 = np.r_[0.0, np.cumsum(h * h)]
    w_sum = cs[M:] - cs[:L]
    w_sq = cs2[M:] - cs2[:L]
    w_var = np.maximum(w_sq / M - (w_sum / M) ** 2, 0.0)
    w_std = np.sqrt(w_var)

    # NCC = raw / (M * std_haystack_window * std_needle)
    denom = w_std * (n_std * M)
    denom = np.where(denom < 1e-9, 1e-9, denom)

    return np.clip((raw / denom).astype(np.float32), -1.0, 1.0)


def _top_k_peaks(arr: np.ndarray, k: int, min_dist: int) -> List[tuple]:
    """Top-k local maxima with minimum sample separation between peaks."""
    a = arr.copy().astype(np.float32)
    results: List[tuple] = []
    for _ in range(k * 4):
        if len(a) == 0:
            break
        idx = int(np.argmax(a))
        val = float(a[idx])
        if val < -0.9:
            break
        results.append((idx, val))
        lo = max(0, idx - min_dist)
        hi = min(len(a), idx + min_dist + 1)
        a[lo:hi] = -2.0
        if len(results) >= k:
            break
    return results


# ── Stability helpers ─────────────────────────────────────────────────────────

def _peak_sharpness(arr: np.ndarray, peak_idx: int, delta: int) -> float:
    """
    Measure sharpness of NCC peak — how much peak exceeds surrounding flanks.

    Returns [0, 1]:
      1.0 = very sharp peak (high discrimination, trustworthy alignment)
      0.0 = completely flat (non-discriminative, uncertain alignment)

    A flat peak means many offset positions score similarly, so the exact
    best-matching offset is unreliable even if the NCC value is high.
    """
    n = len(arr)
    if n < 3:
        return 0.5
    lo = max(0, peak_idx - delta)
    hi = min(n, peak_idx + delta + 1)
    left = arr[lo:peak_idx]
    right = arr[peak_idx + 1:hi]
    flank_parts = [p for p in (left, right) if len(p) > 0]
    if not flank_parts:
        return 0.5
    flank = np.concatenate(flank_parts)
    peak_val = float(arr[peak_idx])
    flank_mean = float(np.mean(flank))
    local_min = float(np.min(arr[lo:hi]))
    denom = max(peak_val - local_min, 1e-9)
    return float(np.clip((peak_val - flank_mean) / denom, 0.0, 1.0))


def _envelope_ncc_at(
    haystack: np.ndarray,
    clip: np.ndarray,
    offset: int,
    sr: int = SR_FULL,
    frame_ms: float = 10.0,
) -> float:
    """
    Pearson correlation of RMS envelopes between haystack[offset:offset+M] and clip.

    Provides an independent cross-validation signal: if wideband NCC says
    "great match" but the amplitude envelope shapes differ, the match is
    suspicious (e.g. it may have landed on a different similarly-timbred phrase).

    Returns float in [-1, 1].  Returns 0.0 if insufficient frames for correlation.
    """
    M = len(clip)
    seg = haystack[offset: offset + M]
    if len(seg) < M:
        # Pad if haystack ends early — shouldn't happen in practice
        seg = np.concatenate([seg, np.zeros(M - len(seg), dtype=np.float32)])

    frame_len = max(1, int(frame_ms / 1000.0 * sr))
    n_frames = M // frame_len
    if n_frames < 4:
        return 0.0

    seg_f = seg.astype(np.float64)
    clip_f = clip.astype(np.float64)

    env_h = np.array(
        [float(np.sqrt(np.mean(seg_f[j * frame_len:(j + 1) * frame_len] ** 2)))
         for j in range(n_frames)],
        dtype=np.float64,
    )
    env_c = np.array(
        [float(np.sqrt(np.mean(clip_f[j * frame_len:(j + 1) * frame_len] ** 2)))
         for j in range(n_frames)],
        dtype=np.float64,
    )

    std_h = float(env_h.std())
    std_c = float(env_c.std())
    if std_h < 1e-9 or std_c < 1e-9:
        return 0.0

    return float(np.clip(np.corrcoef(env_h, env_c)[0, 1], -1.0, 1.0))


# ── Main matching function ────────────────────────────────────────────────────

def match_clip_robust(
    region: np.ndarray,
    region_bp: np.ndarray,
    region_coarse: np.ndarray,
    clip: np.ndarray,
    clip_bp: np.ndarray,
    coarse_ratio: int = COARSE_RATIO,
) -> ClipMatchResult:
    """
    Find the best match for `clip` within `region` using a multi-stage pipeline.

    Parameters
    ----------
    region          vobu[search_lo:search_hi] at SR_FULL, mono float32
    region_bp       bandpass-filtered version of region at SR_FULL (on-demand)
    region_coarse   bandpass-filtered + downsampled region at SR_COARSE
    clip            reference clip at SR_FULL, mono float32
    clip_bp         bandpass-filtered version of clip at SR_FULL
    coarse_ratio    SR_FULL / SR_COARSE (default 12)

    Returns
    -------
    ClipMatchResult with .offset relative to region[0] and .trim_samples set.
    The caller computes: vobu_match = max(0, search_lo + result.offset - result.trim_samples)
    """
    reasons: List[str] = []
    M_orig = len(clip)

    # ── Guard conditions ──────────────────────────────────────────────────────
    if M_orig == 0:
        reasons.append("empty reference clip")
        return ClipMatchResult(0, 0.0, "fail", reasons, [], 1)
    if len(region) < M_orig:
        reasons.append("search region smaller than clip length")
        return ClipMatchResult(0, 0.0, "fail", reasons, [], 1)

    # ── 1. Clip classification ────────────────────────────────────────────────
    db = rms_db(clip)
    dur_ms = M_orig / SR_FULL * 1000.0
    is_silent = db < SILENCE_DB
    is_weak = db < WEAK_DB
    is_very_short = M_orig < VERY_SHORT_SAMPLES
    is_short = M_orig < SHORT_SAMPLES

    if is_silent:
        reasons.append(f"near-silence ({db:.1f} dB)")
    elif is_weak:
        reasons.append(f"weak signal ({db:.1f} dB)")
    if is_very_short:
        reasons.append(f"very short clip ({dur_ms:.0f} ms)")
    elif is_short:
        reasons.append(f"short clip ({dur_ms:.0f} ms)")

    # ── 2. Edge trimming (remove fade contamination from clip edges) ──────────
    trim_samples = 0
    clip_trimmed = clip
    clip_bp_trimmed = clip_bp

    _min_trim_samp = int(EDGE_TRIM_MIN_CLIP_MS / 1000.0 * SR_FULL)
    if M_orig >= _min_trim_samp and not is_very_short:
        _ts = int(EDGE_TRIM_MS / 1000.0 * SR_FULL)
        _trimmed = clip[_ts: M_orig - _ts]
        _trimmed_bp = clip_bp[_ts: M_orig - _ts]
        if len(_trimmed) >= 2:
            trim_samples = _ts
            clip_trimmed = _trimmed
            clip_bp_trimmed = _trimmed_bp

    M = len(clip_trimmed)

    # Re-check guard after trimming
    if len(region) < M:
        reasons.append("search region smaller than trimmed clip length")
        return ClipMatchResult(0, 0.0, "fail", reasons, [], 1, trim_samples=trim_samples)

    # ── 3. Coarse search on downsampled bandpass region ───────────────────────
    M_c = max(1, M // coarse_ratio)
    clip_c = downsample(clip_bp_trimmed, SR_FULL, SR_COARSE)

    coarse_cands: List[tuple] = []  # (region_offset_full_SR, coarse_score, polarity)

    if not is_silent and len(region_coarse) >= M_c + 1 and M_c >= 2:
        ncc_c_n = sliding_ncc(region_coarse, clip_c)
        ncc_c_i = sliding_ncc(region_coarse, -clip_c)
        ncc_c_best = np.maximum(ncc_c_n, ncc_c_i)
        min_dist_c = max(1, M_c // 4)
        for idx_c, score_c in _top_k_peaks(ncc_c_best, TOP_K * 2, min_dist_c):
            pol = (
                1
                if (idx_c < len(ncc_c_n) and float(ncc_c_n[idx_c]) >= float(ncc_c_i[idx_c]))
                else -1
            )
            coarse_cands.append((idx_c * coarse_ratio, score_c, pol))
    else:
        # Fallback: single candidate at region center (better than hardcoded offset 0)
        # Fine search will explore ±0.5s around this center point
        fallback_offset = len(region) // 2 if len(region) > M else 0
        coarse_cands = [(fallback_offset, 0.0, 1)]
        if is_silent:
            reasons.append("coarse search skipped: silence")
        elif M_c < 2:
            reasons.append("coarse search skipped: clip too short at coarse resolution")

    # ── 4. Fine search (full-rate NCC) around each coarse candidate ───────────
    fine_half = int(FINE_HALF_S * SR_FULL)
    clip_inv = -clip_trimmed
    clip_bp_inv = -clip_bp_trimmed
    stab_delta = max(1, int(STABILITY_DELTA_MS / 1000.0 * SR_FULL))

    # For weak/very-short clips, lean heavily on bandpass (better SNR in speech band)
    bp_w = 0.80 if (is_weak or is_very_short) else 0.65
    wb_w = 1.0 - bp_w

    fine_cands: List[MatchCandidate] = []

    for coarse_off, _cs, _cp in coarse_cands[: TOP_K * 2]:
        lo = int(max(0, coarse_off - fine_half))
        hi = int(min(len(region) - M, coarse_off + fine_half))
        if lo > hi:
            continue

        sub = region[lo: hi + M]
        sub_bp = region_bp[lo: hi + M]
        if len(sub) < M:
            continue

        # NCC for both normal and polarity-inverted, wideband and bandpass
        ncc_wb_n = sliding_ncc(sub, clip_trimmed)
        ncc_wb_i = sliding_ncc(sub, clip_inv)
        ncc_bp_n = sliding_ncc(sub_bp, clip_bp_trimmed)
        ncc_bp_i = sliding_ncc(sub_bp, clip_bp_inv)

        if len(ncc_wb_n) == 0:
            continue

        fused_n = bp_w * ncc_bp_n + wb_w * ncc_wb_n
        fused_i = bp_w * ncc_bp_i + wb_w * ncc_wb_i

        best_n_idx = int(np.argmax(fused_n))
        best_i_idx = int(np.argmax(fused_i))
        score_n = float(fused_n[best_n_idx])
        score_i = float(fused_i[best_i_idx])

        # REJECT weak peaks early
        if score_n < MIN_NCC_CANDIDATE and score_i < MIN_NCC_CANDIDATE:
            continue

        # Per-candidate method scores for later cross-validation
        sc_n_wb = float(ncc_wb_n[best_n_idx]) if best_n_idx < len(ncc_wb_n) else 0.0
        sc_n_bp = float(ncc_bp_n[best_n_idx]) if best_n_idx < len(ncc_bp_n) else 0.0
        sc_i_wb = float(ncc_wb_i[best_i_idx]) if best_i_idx < len(ncc_wb_i) else 0.0
        sc_i_bp = float(ncc_bp_i[best_i_idx]) if best_i_idx < len(ncc_bp_i) else 0.0

        # Polarity decision: inverted must beat normal by POLARITY_FLIP_MIN_DELTA
        if score_n >= score_i or (score_i - score_n) < POLARITY_FLIP_MIN_DELTA:
            pol = 1
            off = lo + best_n_idx
            score = score_n
            cand_ncc_wb = sc_n_wb
            cand_ncc_bp = sc_n_bp
            stab = _peak_sharpness(fused_n, best_n_idx, stab_delta)
        else:
            pol = -1
            off = lo + best_i_idx
            score = score_i
            cand_ncc_wb = sc_i_wb
            cand_ncc_bp = sc_i_bp
            stab = _peak_sharpness(fused_i, best_i_idx, stab_delta)

        fine_cands.append(MatchCandidate(
            offset=off,
            ncc=score,
            polarity=pol,
            ncc_wb=cand_ncc_wb,
            ncc_bp=cand_ncc_bp,
            stability=stab,
        ))

    if not fine_cands:
        reasons.append("no fine candidates found in search window")
        return ClipMatchResult(0, 0.0, "fail", reasons, [], 1, trim_samples=trim_samples)

    # ── 5. Sort, deduplicate, select best ────────────────────────────────────
    # Sort by composite score: NCC (70%) + stability (20%) + BP consistency (10%)
    # This prefers peaks that are both high-scoring AND sharp AND consistent
    def candidate_score(c: MatchCandidate) -> float:
        bp_consistency = min(c.ncc_bp, c.ncc_wb) / max(c.ncc_bp, c.ncc_wb + 1e-9)
        return 0.70 * c.ncc + 0.20 * c.stability + 0.10 * bp_consistency

    fine_cands.sort(key=candidate_score, reverse=True)

    dedup_dist = max(1, int(0.050 * SR_FULL))      # 50 ms minimum separation
    deduped: List[MatchCandidate] = []
    for c in fine_cands:
        if not any(abs(c.offset - d.offset) < dedup_dist for d in deduped):
            deduped.append(c)

    top = deduped[:TOP_K]
    best = top[0]
    raw_ncc = float(np.clip(best.ncc, 0.0, 1.0))

    # ── 6. Ambiguity detection ────────────────────────────────────────────────
    ambig_hard = ambig_soft = False
    near_cands_count = 0
    far_dup = False

    if len(top) >= 2 and best.ncc > 1e-6:
        ratio = top[1].ncc / best.ncc

        if ratio >= AMBIG_HARD:
            ambig_hard = True
            reasons.append(
                f"ambiguous: 2nd candidate is {ratio:.2f}x of best "
                f"(ncc={top[1].ncc:.3f}) — possible repeated/duplicate content"
            )
        elif ratio >= AMBIG_SOFT:
            ambig_soft = True
            reasons.append(
                f"soft ambiguity: 2nd candidate is {ratio:.2f}x of best "
                f"(ncc={top[1].ncc:.3f})"
            )

        # Count near-equal candidates (within NEAR_EQUAL_RATIO fraction of best NCC)
        near_cands_count = sum(
            1 for c in top[1:]
            if c.ncc >= best.ncc * NEAR_EQUAL_RATIO
        )
        if near_cands_count >= NEAR_CANDS_AMBIG:
            reasons.append(
                f"many near-equal candidates ({near_cands_count}) — possibly repeated phrase"
            )

        # Spatial far-duplicate: near-equal candidate far from best
        far_dup_dist = int(FAR_DUPLICATE_MIN_S * SR_FULL)
        far_dup = any(
            abs(c.offset - best.offset) > far_dup_dist
            for c in top[1:]
            if c.ncc >= best.ncc * NEAR_EQUAL_RATIO
        )
        if far_dup:
            reasons.append(
                "spatially distant near-equal candidate — possible repeated phrase "
                "at a different position in the VOBU"
            )

    # ── 7. Stability check ────────────────────────────────────────────────────
    stability = best.stability
    # Stricter penalty: flat peak → confidence heavily penalized
    if stability < 0.40:
        stability_factor = 0.50 + 0.50 * stability  # ranges 0.50-1.0, not 0.70-1.0
    else:
        stability_factor = 1.0

    # ── 8. Method agreement: wideband NCC vs RMS-envelope NCC ────────────────
    method_agreement = 1.0
    env_ncc_val: Optional[float] = None

    if M >= VERY_SHORT_SAMPLES * 2 and not is_silent:
        try:
            env_ncc_val = _envelope_ncc_at(region, clip_trimmed, best.offset, sr=SR_FULL)
        except Exception as exc:
            log.debug("envelope NCC failed: %s", exc)
            env_ncc_val = None

        if env_ncc_val is not None:
            # Stricter disagreement penalty: more aggressive than before
            if best.ncc_wb > 0.45 and env_ncc_val < best.ncc_wb - 0.20:
                spread = best.ncc_wb - env_ncc_val
                method_agreement = float(np.clip(
                    1.0 - (spread - 0.20) / 0.40 * 0.65,  # was 0.50, now 0.65
                    0.35, 1.0  # was 0.50, now 0.35
                ))

    # ── 9. Confidence scoring with all penalties applied ─────────────────────
    conf = float(np.clip(best.ncc, 0.0, 1.0))

    # Energy penalties
    if is_silent:
        conf *= 0.25
        reasons.append("confidence penalized: silence")
    elif is_weak:
        conf *= 0.80
        reasons.append("confidence penalized: weak signal")

    # Duration penalties
    if is_very_short:
        conf *= 0.72
    elif is_short:
        conf *= 0.85

    # Ambiguity penalties
    if ambig_hard:
        conf *= 0.40
    elif ambig_soft:
        conf *= 0.78

    if near_cands_count >= NEAR_CANDS_AMBIG:
        conf *= 0.75
    if far_dup:
        conf *= 0.78

    # Polarity inversion (unusual; warrants extra review)
    if best.polarity == -1:
        conf *= 0.85
        reasons.append("polarity inversion detected")

    # Peak sharpness / stability (flat peak → very uncertain offset)
    if stability < 0.40:
        conf *= stability_factor
        reasons.append(
            f"flat NCC peak (stability={stability:.2f}) — alignment may be uncertain"
        )

    # Method agreement (wideband vs envelope) — stricter penalty
    if method_agreement < 0.90:
        conf *= method_agreement
        if env_ncc_val is not None:
            reasons.append(
                f"method disagreement (agreement={method_agreement:.2f}): "
                f"wideband NCC={best.ncc_wb:.3f} vs envelope={env_ncc_val:.3f}"
            )

    conf = float(np.clip(conf, 0.0, 1.0))

    # ── 10. Minimum confidence floor ──────────────────────────────────────────
    # If raw NCC is very low, force fail UNLESS peak is sharp and methods agree
    if raw_ncc < 0.35 and conf < 0.50:
        if stability < 0.60 or method_agreement < 0.85:
            conf = 0.0
            reasons.append("minimum confidence floor: raw NCC too low")
        else:
            reasons.append("minimum floor relaxed: sharp peak + strong method agreement")

    # ── 11. Status assignment ─────────────────────────────────────────────────
    if conf >= CONF_OK:
        status = "ok"
    elif conf >= CONF_REVIEW:
        status = "review"
    else:
        status = "fail"

    return ClipMatchResult(
        offset=best.offset,
        confidence=conf,
        status=status,
        reasons=reasons,
        candidates=top,
        polarity=best.polarity,
        raw_ncc=raw_ncc,
        stability=stability,
        method_agreement=method_agreement,
        trim_samples=trim_samples,
        near_cands_count=near_cands_count,
    )


# ── DriftTracker ──────────────────────────────────────────────────────────────

class DriftTracker:
    """
    Maintains a running estimate of the timeline→VOBU position relationship
    using confidence-weighted, recency-weighted linear regression on recent
    high-confidence matched clips.

    Changes vs v1:
      - Anchors store (timeline, vobu, weight) 3-tuples.
      - record() gates by ANCHOR_MIN_CONF and scales weight by confidence.
      - _estimate() uses combined confidence × recency weighting.
      - check_drift_jump() returns deviation in seconds before recording,
        so the caller can skip poisoning the tracker with jump anchors.

    Provides bounded search windows for each new clip, dramatically reducing
    false-positive matches by limiting the search to the expected neighbourhood.
    Falls back to a broad default window when no anchors are available.
    """

    def __init__(
        self,
        vobu_len: int,
        sr: int = SR_FULL,
        default_window_s: float = DEFAULT_WINDOW_S,
        neighbour_window_s: float = NEIGHBOUR_WINDOW_S,
    ) -> None:
        self._vlen = vobu_len
        self._sr = sr
        self._default_half = int(default_window_s * sr)
        self._neigh_half = int(neighbour_window_s * sr)
        self._anchors: List[tuple] = []   # (timeline_sample, vobu_sample, weight)

    def record(
        self,
        timeline_start: int,
        vobu_match: int,
        status: str,
        confidence: float = 1.0,
    ) -> None:
        """
        Register a match as a confidence-weighted anchor for future estimates.

        REVIEW matches with usable evidence are allowed to contribute anchors,
        but at a lower weight than clean OK matches.
        """
        if status not in ("ok", "review"):
            return
        if confidence < ANCHOR_MIN_CONF:
            return

        t = float(np.clip(
            (confidence - ANCHOR_MIN_CONF) / max(1e-6, ANCHOR_WEIGHT_CONF - ANCHOR_MIN_CONF),
            0.0, 1.0,
        ))
        base_weight = 0.25 + 0.75 * t
        if status == "review":
            base_weight *= 0.75
        self._anchors.append((int(timeline_start), int(vobu_match), float(base_weight)))
        if len(self._anchors) > 14:
            self._anchors = self._anchors[-14:]

    def check_drift_jump(self, timeline_start: int, vobu_match: int) -> float:
        """
        Return the deviation in seconds between vobu_match and the drift-predicted
        position for timeline_start.

        Returns 0.0 if fewer than 2 anchors are available (no reliable estimate).

        The caller should compare this against DRIFT_JUMP_S:
          if tracker.check_drift_jump(tl, vo) > DRIFT_JUMP_S:
              # skip recording this anchor — possible false positive or real jump
        """
        if len(self._anchors) < 2:
            return 0.0
        est = self._estimate(timeline_start)
        return float(abs(vobu_match - est) / self._sr)

    def get_search_window(self, timeline_start: int, clip_len: int) -> tuple:
        """
        Return (search_lo, search_hi) in VOBU samples.

        Window width adapts to tracker certainty:
          - 0 anchors  → full search
          - 1 anchor   → broad neighbourhood
          - 2 anchors  → medium neighbourhood
          - 3+ anchors → normal neighbourhood
        """
        n = len(self._anchors)
        if n == 0:
            lo, hi = 0, self._vlen
        else:
            est = int(max(0, self._estimate(timeline_start)))
            if n == 1:
                half = max(self._neigh_half * 3, int(20.0 * self._sr))
            elif n == 2:
                half = max(self._neigh_half * 2, int(14.0 * self._sr))
            else:
                half = self._neigh_half

            lo = max(0, est - half)
            hi = min(self._vlen, est + half)

        if hi - lo < clip_len:
            mid = (lo + hi) // 2
            half = max(clip_len, self._neigh_half)
            lo = max(0, mid - half)
            hi = min(self._vlen, mid + half)

        return lo, hi

    def _estimate(self, timeline_pos: int) -> float:
        """
        Confidence-weighted + recency-weighted linear regression → VOBU estimate.

        Each anchor weight = confidence_weight × exp_recency_decay.
        Slope is clamped to [0.3, 3.0] to prevent wild extrapolation.
        """
        if not self._anchors:
            return float(timeline_pos)
        if len(self._anchors) == 1:
            tl0, vo0, _ = self._anchors[0]
            return float(vo0 + (timeline_pos - tl0))

        tls = np.array([a[0] for a in self._anchors], dtype=np.float64)
        vos = np.array([a[1] for a in self._anchors], dtype=np.float64)
        anchor_w = np.array([a[2] for a in self._anchors], dtype=np.float64)

        # Combine confidence weight with exponential recency decay
        recency = np.exp(np.linspace(-1.5, 0.0, len(tls)))
        w = recency * anchor_w
        ws = float(w.sum())
        if ws < 1e-9:
            w = np.ones(len(tls))
            ws = float(len(tls))

        tl_m = float(np.dot(w, tls) / ws)
        vo_m = float(np.dot(w, vos) / ws)

        num = float(np.dot(w, (tls - tl_m) * (vos - vo_m)))
        den = float(np.dot(w, (tls - tl_m) ** 2))

        # Slope should be ~1.0 for no varispeed; clamp to [0.3, 3.0] for safety
        slope = float(np.clip(num / den if abs(den) > 0.5 else 1.0, 0.3, 3.0))
        intercept = vo_m - slope * tl_m
        return slope * float(timeline_pos) + intercept

    @staticmethod
    def check_consistency(results: list, sr: int = SR_FULL) -> list:
        """
        Post-pass: flag clips whose VOBU positions deviate significantly from
        the linear trend established by all ok/review clips.

        Adds to each dict in-place:
          consistency_ok          bool
          consistency_deviation_s float  (deviation in seconds from trend line)

        Threshold: max(300 ms, 5 × median-absolute-deviation of the fitted residuals).
        """
        for r in results:
            r.setdefault("consistency_ok", True)
            r.setdefault("consistency_deviation_s", 0.0)

        anchors = [
            (r["timeline_start_samples"], r["vobu_match_samples"])
            for r in results
            if r.get("status") in ("ok", "review")
            and r.get("vobu_match_samples", -1) >= 0
        ]
        if len(anchors) < 3:
            return results

        tls = np.array([a[0] for a in anchors], dtype=np.float64)
        vos = np.array([a[1] for a in anchors], dtype=np.float64)

        try:
            slope, intercept = np.polyfit(tls, vos, 1)
        except np.linalg.LinAlgError:
            return results

        residuals = np.abs(vos - (slope * tls + intercept))
        mad = float(np.median(residuals))
        threshold = max(int(0.30 * sr), int(mad * 5))   # 300 ms or 5× MAD

        for r in results:
            if r.get("status") not in ("ok", "review"):
                continue
            tl = r.get("timeline_start_samples", 0)
            vo = r.get("vobu_match_samples", -1)
            if vo < 0:
                continue
            expected = float(slope * tl + intercept)
            dev = abs(vo - expected)
            if dev > threshold:
                r["consistency_ok"] = False
                r["consistency_deviation_s"] = round(dev / sr, 4)

        return results
