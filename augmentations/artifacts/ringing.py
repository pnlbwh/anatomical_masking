"""Cortical-halo ringing and severity-dependent rendering parameters.

The slice kernel accepts a random source explicitly. ``master_params`` supplies
severity settings to the production 3-D ringing renderer.
"""
from __future__ import annotations

from typing import Dict

import cv2
import numpy as np


def _resize_centered_float(arr, scale):
    h, w = arr.shape[:2]
    if abs(scale - 1.0) < 1e-6:
        return arr.copy()
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    small = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((h, w), dtype=arr.dtype)
    sy = (h - new_h) // 2
    sx = (w - new_w) // 2
    out[sy:sy + new_h, sx:sx + new_w] = small
    return out


def _resize_mask_centered(mask, scale):
    h, w = mask.shape[:2]
    if scale >= 1.0:
        return mask.copy()
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    small = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    out = np.zeros((h, w), dtype=mask.dtype)
    sy = (h - new_h) // 2
    sx = (w - new_w) // 2
    out[sy:sy + new_h, sx:sx + new_w] = small
    return out


def cortical_halo(img, mask, *, rnd, n_passes, scale_step, erosion_range,
                  bump_neg_div, bump_pos_div, init_scale=0.98):
    """Nested cortical-ribbon ring stamping (verbatim ``add_cortical_halo_ringing``).

    Parameters
    ----------
    img, mask : 2D float slice + binary-ish brain mask.
    rnd : a stdlib-``random``-like object (the global ``random`` module for
        byte-identical production, or ``random.Random(seed)`` for the UI).
    n_passes : number of ring passes.
    scale_step : ``(lo, hi)`` per-pass slice-shrink increment (drawn per pass).
    erosion_range : ``(lo, hi)`` inner-mask shrink (the cortical annulus).
    bump_neg_div, bump_pos_div : per-pass bump = uniform(-mean/neg, +mean/pos);
        smaller divisor = stronger ring (bias ~positive = bright-dominant).
    init_scale : starting ring size (legacy 0.98).
    """
    out = np.asarray(img, dtype=np.float32).copy()
    mask_arr = np.asarray(mask)
    if int((mask_arr > 0.5).sum()) < 100:
        return np.clip(out, 0.0, 1.0).astype(np.float32)
    mask_bin = (mask_arr > 0.5).astype(np.uint8)
    brain_mean = float(out[mask_bin > 0].mean()) if mask_bin.sum() > 0 else 0.5
    bump_neg_div = float(bump_neg_div)
    bump_pos_div = float(bump_pos_div)
    for i in range(int(n_passes)):
        scale = init_scale - rnd.uniform(*scale_step) * i
        if scale <= 0.20:
            break
        scaled_slice = _resize_centered_float(out.astype(np.float32), scale)
        sx = cv2.Sobel(scaled_slice, cv2.CV_64F, 1, 0, ksize=1)
        sy = cv2.Sobel(scaled_slice, cv2.CV_64F, 0, 1, ksize=1)
        gmag = np.sqrt(sx ** 2 + sy ** 2)
        gmax = float(gmag.max())
        if gmax < 1e-6:
            continue
        edges_norm = gmag / gmax
        edges = (edges_norm >= (50.0 / 255.0)) & (mask_bin > 0)
        erosion_scale = rnd.uniform(*erosion_range)
        small_mask = _resize_mask_centered(mask_bin, erosion_scale)
        edges = edges & (small_mask <= 0)
        if int(edges.sum()) < 20:
            continue
        bump = rnd.uniform(-brain_mean / bump_neg_div, +brain_mean / bump_pos_div)
        out = out.astype(np.float32)
        out[edges] += bump
    return np.clip(out, 0.0, 1.0).astype(np.float32)


DEFAULT_PARAMS = {
    'n_passes': 22,
    'scale_step': 0.005,
    'init_scale': 0.98,
    'erosion_lo': 0.3,
    'erosion_hi': 0.92,
    'ring_bright': 0.083,
    'ring_dark': 0.033,
    'seed': 0,
    'ringing_on': True,
}


def default_params() -> Dict:
    return DEFAULT_PARAMS.copy()


MASTER_MARK = 1.0
MASTER_MAX = 1.3
_MASTER_KEYS = ["n_passes", "scale_step", "erosion_lo", "erosion_hi",
                "ring_bright", "ring_dark", "init_scale"]

_MASTER_CLEAN = dict(n_passes=0, scale_step=0.005, erosion_lo=0.30, erosion_hi=0.92,
                     ring_bright=0.083, ring_dark=0.033, init_scale=0.98)
_MASTER_REAL = dict(n_passes=22, scale_step=0.005, erosion_lo=0.30, erosion_hi=0.92,
                    ring_bright=0.083, ring_dark=0.033, init_scale=0.98)
_MASTER_EXTREME = dict(n_passes=68, scale_step=0.009, erosion_lo=0.20, erosion_hi=0.95,
                       ring_bright=0.150, ring_dark=0.045, init_scale=0.98)


def master_params(t: float) -> Dict:
    t = max(0.0, min(MASTER_MAX, float(t)))
    if t <= MASTER_MARK:
        u = (t / MASTER_MARK) if MASTER_MARK > 0 else 1.0
        lo, hi = _MASTER_CLEAN, _MASTER_REAL
    else:
        span = (MASTER_MAX - MASTER_MARK) or 1.0
        u = (t - MASTER_MARK) / span
        lo, hi = _MASTER_REAL, _MASTER_EXTREME
    p = default_params()
    for k in _MASTER_KEYS:
        val = lo[k] + (hi[k] - lo[k]) * u
        p[k] = int(round(val)) if k == "n_passes" else float(val)
    p["ringing_on"] = True
    return p
