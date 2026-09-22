"""Realistic cross-scanner appearance variation for masker training.

The synth GMM stream (`label_synth.synthesize_from_labels`) decorrelates contrast but looks fake
(flat-class salt-and-pepper). Per the user, training should be MOSTLY realistic variations â€” a real
T1 brain made to look like it came from a DIFFERENT scanner / protocol / intensity â€” with only a
MINORITY of the aggressive synthetic samples.

This module is a faithful 3D port of the REALISTIC `scan_morph` techniques (the engine works on 256^2
2D slices via cv2/SimpleITK; the load-bearing math is pure-numpy and n-D-portable). It KEEPS the real
scan's tissue texture and transfers only TONE, BIAS, RESOLUTION and gentle SHAPE â€” the axes a real
different-scanner scan actually differs on. References are to `scan_morph_core.py` in the repo root.

Gotchas honored (from the scan_morph extraction):
  * FULL HEAD kept â€” never skull-strip (a masker must see what it segments).
  * Tone curve is MONOTONE with knot[0]=0 (air stays black; no contrast inversion / WM<GM<CSF scramble).
  * Grain/resolution use a FRESH rng each call (no constant-grain shortcut).
  * Mask co-transforms with NEAREST (geometry only); tone/bias/resolution do NOT touch the mask.
  * No DCT detail tier (that copies a target â€” not an independent variation).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import ndimage as ndi
from augmentations.config import enabled, settings

# 6-knot tone curve sampled at fixed inputs (scan_morph_core.py:80, TONE_X).
TONE_X = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float32)


# --------------------------------------------------------------------------- intensity / tone
def _monotone_knots(rng: np.random.Generator, strength: float = 0.22) -> np.ndarray:
    """A MONOTONE 6-knot tone curve near identity (scan_morph keeps WM>GM>CSF order + air black).

    Jitter the identity knots, re-SORT to guarantee non-decreasing (a realistic per-scanner transfer
    curve never re-orders tissues), clip, and pin knot[0]=0 so background/air stays black."""
    ys = np.sort(TONE_X + rng.uniform(-strength, strength, len(TONE_X)))
    ys = np.clip(ys, 0.0, 1.15).astype(np.float32)
    ys[0] = 0.0
    return ys


def tone_map(v: np.ndarray, gamma: float, knots: np.ndarray, contrast: float, brightness: float) -> np.ndarray:
    """scan_morph_core.tone_map (:416), n-D: gamma -> 6-knot interp -> contrast about 0.5 + brightness."""
    v = np.clip(v, 0.0, 1.0) ** gamma
    v = np.interp(v, TONE_X, knots).astype(np.float32)
    v = 0.5 + contrast * (v - 0.5) + brightness
    return np.clip(v, 0.0, 1.0).astype(np.float32)


def realistic_tone(scan01: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Realistic global intensity/contrast spread across scanners/protocols (KEEPS texture)."""
    options = settings('tone_mapping')
    return tone_map(scan01, float(rng.uniform(*options["gamma_range"])), _monotone_knots(rng, strength=options["knot_jitter"]),
                    float(rng.uniform(*options["contrast_range"])), float(rng.uniform(*options["brightness_range"])))


def quantile_tone_transfer(scan01: np.ndarray, mask: np.ndarray, donor01: np.ndarray,
                           donor_mask: np.ndarray) -> np.ndarray:
    """Match the scan's intensity DISTRIBUTION to a real DONOR subject's (scan_morph_core
    `_quantile_tone_knots` :579) â€” the most authentic 'different scanner' move: same brain + texture,
    the donor's tissue intensities. Histogram-transfer over the brain region; air pinned to 0."""
    s = scan01[mask > 0.5] if np.any(mask > 0.5) else scan01[scan01 > 0]
    d = donor01[donor_mask > 0.5] if np.any(donor_mask > 0.5) else donor01[donor01 > 0]
    if s.size < 8 or d.size < 8:
        return scan01.astype(np.float32)
    q = np.linspace(0.0, 1.0, 256)
    qs = np.quantile(s, q).astype(np.float32)
    qd = np.quantile(d, q).astype(np.float32)
    qs, idx = np.unique(qs, return_index=True)          # np.interp needs strictly-increasing xp
    out = np.interp(np.clip(scan01, 0, 1), qs, qd[idx]).astype(np.float32)
    out[scan01 <= 1e-4] = 0.0                            # keep air black
    return np.clip(out, 0.0, 1.0)


# --------------------------------------------------------------------------- bias / shading
def realistic_bias(scan01: np.ndarray, rng: np.random.Generator, order: int = 3,
                   max_strength: float = 0.7) -> np.ndarray:
    """Smooth multiplicative coil/RF shading: exp of a low-order 3D cosine field, DC excluded
    (scan_morph_core `_bias_basis`/`bias_field` :431/:450, ported to n-D). Gentle (realistic) strength."""
    shape = scan01.shape
    axes = [np.cos(np.outer(np.arange(order), np.pi * np.linspace(0, 1, s)).reshape(order, *([1] * i), s, *([1] * (len(shape) - 1 - i))))
            for i, s in enumerate(shape)]
    field = np.zeros(shape, dtype=np.float32)
    for i in range(order):
        for j in range(order):
            for k in range(order):
                if i == 0 and j == 0 and k == 0:
                    continue
                field = field + float(rng.uniform(-1, 1)) * (axes[0][i] * axes[1][j] * axes[2][k])
    m = float(np.abs(field).max()) or 1.0
    field = (field / m) * float(rng.uniform(0.0, max_strength))
    return (scan01 * np.exp(np.clip(field, -1.2, 1.2))).astype(np.float32)


# --------------------------------------------------------------------------- resolution (the scan_morph gap)
def realistic_resolution(scan01: np.ndarray, rng: np.random.Generator, p: float = 0.7) -> np.ndarray:
    """Different acquired resolution / slice thickness: gentle anisotropic blur -> downsample ->
    upsample (the cross-scanner axis scan_morph omits). Milder than the aggressive synth version."""
    if rng.random() > p:
        return scan01
    options = settings('resolution')
    factors = np.array([float(rng.uniform(*options["factor_range"])) if rng.random() < options["axis_probability"] else 1.0 for _ in scan01.shape])
    if np.allclose(factors, 1.0):
        return scan01
    sigma = [(f / 3.0 if f > 1.0 else 0.0) for f in factors]
    small = ndi.zoom(ndi.gaussian_filter(scan01, sigma), 1.0 / factors, order=1)
    up = ndi.zoom(small, np.asarray(scan01.shape) / np.asarray(small.shape), order=1)
    if up.shape != scan01.shape:
        out = np.zeros(scan01.shape, np.float32)
        sl = tuple(slice(0, min(a, b)) for a, b in zip(scan01.shape, up.shape))
        out[sl] = up[sl]; up = out
    return up.astype(np.float32)


# --------------------------------------------------------------------------- geometry (gentle, realistic)
def _rot_matrix(dx, dy, dz):
    rx, ry, rz = np.radians([dx, dy, dz])
    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def realistic_geometry(scan01: np.ndarray, mask: np.ndarray, rng: np.random.Generator,
                       rot_deg: float = 10.0, scale_range: Tuple[float, float] = (0.9, 1.12),
                       trans_frac: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """GENTLE realistic head positioning / FOV / voxel-size affine (no 90deg/SI-flip), mask
    co-transformed with NEAREST (order=0) so the label boundary is not cubic-rung."""
    scan01 = np.asarray(scan01, np.float32)
    mask = np.asarray(mask, np.float32)
    if rng.random() < 0.5:                       # an occasional lateral (L/R) flip is realistic
        scan01 = np.ascontiguousarray(np.flip(scan01, 2)); mask = np.ascontiguousarray(np.flip(mask, 2))
    shape = scan01.shape
    center = (np.array(shape) - 1) / 2.0
    M = _rot_matrix(*(rng.uniform(-rot_deg, rot_deg, 3))) @ np.diag(1.0 / rng.uniform(*scale_range, size=3))
    t = rng.uniform(-trans_frac, trans_frac, 3) * np.asarray(shape)
    offset = center - M @ center - M @ t
    scan01 = ndi.affine_transform(scan01, M, offset=offset, order=1, mode="constant", cval=0.0)
    mask = ndi.affine_transform(mask, M, offset=offset, order=0, mode="constant", cval=0.0)
    return scan01.astype(np.float32), (mask > 0.5)


# --------------------------------------------------------------------------- orchestration
def realistic_augment(scan01: np.ndarray, mask: np.ndarray, rng: np.random.Generator,
                      donor: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                      geometry: bool = True, noise_max: float = 0.02,
                      resolution: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """One MOSTLY-REALISTIC variation: gentle geometry (co-transform mask) -> intensity (donor
    histogram transfer OR monotone tone curve) -> bias -> resolution -> mild noise. KEEPS the real
    tissue texture (the point). Returns (scan01 in [0,1], mask bool)."""
    out = np.asarray(scan01, np.float32)
    m = np.asarray(mask, bool)
    if geometry:
        out, m = realistic_geometry(out, m, rng)
    if enabled("donor_histogram_transfer") and donor is not None and rng.random() < settings('donor_histogram_transfer')["probability"]:
        out = quantile_tone_transfer(out, m, donor[0], donor[1])
    elif enabled("tone_mapping"):
        out = realistic_tone(out, rng)
    if enabled("realistic_bias_field"):
        out = realistic_bias(out, rng, **settings('realistic_bias_field'))
    if resolution:
        out = realistic_resolution(out, rng)
    sigma = float(rng.uniform(0.0, noise_max)) if enabled("realistic_noise") else 0.0
    if sigma > 1e-4:
        out = out + rng.normal(0.0, sigma, out.shape).astype(np.float32)
    return np.clip(out, 0.0, 1.0).astype(np.float32), m
