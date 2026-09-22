"""Intensity preprocessing shared by training, evaluation, and inference."""

from __future__ import annotations

from typing import Optional

import numpy as np


def normalize_intensity(vol: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    """Percentile-clip to float32 [0, 1], preserving the range of signed images.

    Nonfinite voxels are sanitized before scaling. Images with more than 1%
    negative voxels use both percentile bounds; others scale positive voxels."""
    vol = np.asarray(vol, dtype=np.float32)
    # Clamp nonfinite values to background or the largest finite positive value.
    if not np.isfinite(vol).all():
        finite_pos = vol[np.isfinite(vol) & (vol > 0)]
        cap = float(finite_pos.max()) if finite_pos.size else 0.0
        vol = np.nan_to_num(vol, nan=0.0, posinf=cap, neginf=0.0)
    neg = vol < 0
    if neg.any() and float(neg.mean()) > 0.01:
        lo = float(np.percentile(vol, 100.0 - percentile))
        hi = float(np.percentile(vol, percentile))
        if hi <= lo:  # near-constant after clipping the tails -> fall back to the full min/max
            lo, hi = float(vol.min()), float(vol.max())
        if hi <= lo:
            return np.zeros_like(vol)
        return np.clip((vol - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    pos = vol[vol > 0]
    hi = float(np.percentile(pos, percentile)) if pos.size else float(vol.max())
    if hi <= 0:
        hi = float(vol.max()) if vol.max() > 0 else 1.0
    out = np.clip(vol, 0.0, hi) / (hi + 1e-8)
    return out.astype(np.float32)


def zscore(vol01: np.ndarray, region_mask: Optional[np.ndarray]) -> np.ndarray:
    """Standardize within a mask, falling back to positive voxels or the full array."""
    if region_mask is not None and np.asarray(region_mask).any():
        region = vol01[np.asarray(region_mask).astype(bool)]
    else:
        pos = vol01[vol01 > 0]
        region = pos if pos.size else vol01
    mu = float(region.mean())
    sd = float(region.std())
    return ((vol01 - mu) / (sd + 1e-8)).astype(np.float32)
