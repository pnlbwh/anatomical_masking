"""Registered metal artifact primitives and 3-D severity settings.

Dipole, signal-void, and pile-up operators expose explicit parameters.
``master_params`` supplies settings to the production 3-D metal renderer.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from augmentations.registry import register


def _grid(shape, cx, cy):
    yy, xx = np.indices(shape).astype(np.float32)
    rx = xx - cx
    ry = yy - cy
    r = np.sqrt(rx ** 2 + ry ** 2) + 1e-3
    return rx, ry, r


def _r_nd(shape, cx, cy, cz=None):
    """Radial distance for a 2D slice OR 3D volume. 3D adds a through-plane axis at ``cz``
    (default = center). 2D path is byte-identical to ``_grid``."""
    if len(shape) == 2:
        return _grid(shape, cx, cy)[2]
    xx, yy, zz = np.indices(shape).astype(np.float32)
    czz = (shape[2] - 1) / 2.0 if cz is None else float(cz)
    return np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - czz) ** 2) + 1e-3


@register("metal_dipole", kind="focal", dims="either",
          label_preserving=True, has_detector=True)
def metal_dipole(arr, *, severity=0.0, rng=None, mask=None,
                 cx=0.0, cy=0.0, cz=None, orientation=0.0, radius=18.0, amp=0.6, **kw):
    """Bilobed susceptibility dipole (VERBATIM from add_metal_dipole; 3D adds a through-plane axis)."""
    img = np.asarray(arr, dtype=np.float32)
    if img.ndim == 2:
        rx, ry, r = _grid(img.shape, cx, cy)
        cos_t = (rx * np.cos(orientation) + ry * np.sin(orientation)) / r
    else:
        xx, yy, zz = np.indices(img.shape).astype(np.float32)
        czz = (img.shape[2] - 1) / 2.0 if cz is None else float(cz)
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - czz) ** 2) + 1e-3
        cos_t = (zz - czz) / r                          # B0 along the through-plane axis
    dipole = (3.0 * cos_t ** 2 - 1.0) * np.exp(-(r ** 2) / (2.0 * float(radius) ** 2))
    return np.clip(img + float(amp) * dipole, 0.0, 1.0).astype(np.float32)


@register("metal_void", kind="focal", dims="either",
          label_preserving=True, has_detector=True)
def metal_void(arr, *, severity=0.0, rng=None, mask=None,
               cx=0.0, cy=0.0, cz=None, radius=12.0, radius_mm=None,
               voxel_sizes=None, depth=0.7, **kw):
    """Gaussian SIGNAL VOID (multiplicative darkening at the implant); 2D slice or 3D volume."""
    img = np.asarray(arr, dtype=np.float32)
    if radius_mm is not None and voxel_sizes is not None:
        spacing = np.asarray(voxel_sizes[:img.ndim], dtype=np.float32)
        grids = np.indices(img.shape, dtype=np.float32)
        centers = ([float(cy), float(cx)] if img.ndim == 2 else
                   [float(cx), float(cy), float((img.shape[2] - 1) / 2.0 if cz is None else cz)])
        r = np.sqrt(sum(((grids[a] - centers[a]) * spacing[a]) ** 2 for a in range(img.ndim)))
        rad = float(radius_mm)
    else:
        r = _r_nd(img.shape, cx, cy, cz)
        rad = float(radius)
    fac = 1.0 - float(depth) * np.exp(-(r ** 2) / (2.0 * rad ** 2))
    return np.clip(img * fac, 0.0, 1.0).astype(np.float32)


@register("metal_pileup", kind="focal", dims="either",
          label_preserving=True, has_detector=True)
def metal_pileup(arr, *, severity=0.0, rng=None, mask=None,
                 cx=0.0, cy=0.0, cz=None, radius=14.0, amp=0.4, width=4.0, **kw):
    """Bright PILE-UP rim: a Gaussian shell of mis-mapped signal at ``radius``; 2D or 3D."""
    img = np.asarray(arr, dtype=np.float32)
    r = _r_nd(img.shape, cx, cy, cz)
    ring = np.exp(-((r - float(radius)) ** 2) / (2.0 * float(width) ** 2))
    return np.clip(img + float(amp) * ring, 0.0, 1.0).astype(np.float32)


DEFAULT_PARAMS = {
    'loc_x': 0.5,
    'loc_y': 0.32,
    'dipole_amp': 0.6,
    'radius': 18.0,
    'orientation': 0.0,
    'void_depth': 0.7,
    'void_radius': 12.0,
    'pileup_amp': 0.4,
    'pileup_width': 4.0,
    'dipole_on': True,
    'void_on': True,
    'pileup_on': True,
}

_FLAGS = ("dipole_on", "void_on", "pileup_on")


def default_params() -> Dict:
    return DEFAULT_PARAMS.copy()


MASTER_MARK = 1.0
MASTER_MAX = 1.3
_MASTER_KEYS = ["dipole_amp", "radius", "void_depth", "void_radius",
                "pileup_amp", "pileup_width"]
_MASTER_CLEAN = dict(dipole_amp=0.0, radius=18.0, void_depth=0.0, void_radius=12.0,
                     pileup_amp=0.0, pileup_width=4.0)
_MASTER_REAL = dict(dipole_amp=0.6, radius=20.0, void_depth=0.7, void_radius=12.0,
                    pileup_amp=0.4, pileup_width=4.0)
_MASTER_EXTREME = dict(dipole_amp=1.3, radius=34.0, void_depth=0.95, void_radius=24.0,
                       pileup_amp=0.8, pileup_width=7.0)


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
        p[k] = float(lo[k] + (hi[k] - lo[k]) * u)
    p["loc_x"] = 0.5
    p["loc_y"] = 0.32
    p["orientation"] = 0.0
    for f in _FLAGS:
        p[f] = True
    return p
