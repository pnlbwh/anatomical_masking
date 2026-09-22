"""Registered T1-to-T2 tissue remapping and 3-D severity settings.

The slice primitive exposes tissue targets explicitly. ``master_params``
supplies the blend strength to the production 3-D T2 renderer.
"""
from __future__ import annotations

from typing import Dict

import cv2
import numpy as np

from augmentations.registry import register


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


@register("t2_remap", kind="intensity", dims="2d",
          label_preserving=True, has_detector=False)
def t2_remap(arr, *, severity=0.0, rng=None, mask=None,
             csf_thresh=0.2, gm_thresh=0.6, csf_target=0.9, gm_target=0.55,
             wm_target=0.4, mod=0.12, skull_factor=0.7, blur_sigma=0.6,
             trans_frac=0.09, detail_sigma=2.0, edge_k=0.08, **kw):
    """Region-aware T1->T2 remap with PRESERVED within-tissue texture.

    The legacy remap split in-brain intensities into 3 HARD percentile bands and
    flattened each to ``target + mod*(rescaled-0.5)`` — which posterized the
    histogram into 3 spikes and crushed within-tissue detail (WM std ~5x too low),
    so it looked flat/cartoonish, not like a real T2. This version fixes both:

    * SOFT tissue memberships (sigmoids at the csf/gm thresholds, width a fraction
      of the in-brain intensity span) blend a smooth per-voxel TARGET field — no
      hard-band posterization edges. The independent CSF/GM/WM targets (incl. the
      GM/WM order-swap the caller may pass) still set the cross-tissue T2 contrast.
    * ADD-BACK high-pass DETAIL (img - blur) restores the within-tissue texture the
      flatten step destroyed. ``detail_gain`` is derived from the legacy ``mod``
      draw (consumes NO extra RNG), calibrated so in-brain texture magnitude matches
      a real T2 acquisition.
    * EDGE-WEIGHTED detail: the detail carries the *T1* sign, but the T2 target is
      tissue-INVERTED, so at a CSF/WM boundary raw add-back would paint a dark rim
      inside bright CSF / a bright fringe inside dark WM (a contrast-reversal halo).
      We suppress detail where the TARGET steps (1/(1+(|grad target|/edge_k)^2)),
      so interiors keep full texture and tissue boundaries stop fighting the target.
    """
    img = np.asarray(arr, dtype=np.float32)
    in_brain = np.asarray(mask) > 0.5
    out = img.copy()
    if in_brain.sum() < 50:
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    bp = img[in_brain]
    span = max(1e-3, float(bp.max() - bp.min()))
    w_t = max(1e-3, float(trans_frac) * span)
    w_csf = _sigmoid((float(csf_thresh) - img) / w_t)
    w_wm = _sigmoid((img - float(gm_thresh)) / w_t)
    w_gm = np.clip(1.0 - w_csf - w_wm, 0.0, None)
    s = w_csf + w_gm + w_wm + 1e-6
    w_csf, w_gm, w_wm = w_csf / s, w_gm / s, w_wm / s
    target = (w_csf * float(csf_target) + w_gm * float(gm_target)
              + w_wm * float(wm_target)).astype(np.float32)

    detail = img - cv2.GaussianBlur(img, (0, 0), sigmaX=float(detail_sigma))
    gx = cv2.Sobel(target, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(target, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.sqrt(gx * gx + gy * gy)
    edge_w = 1.0 / (1.0 + (gmag / max(1e-4, float(edge_k))) ** 2)
    # texture amount from the legacy ``mod`` (0.05..0.20) -> gain ~0.56..0.89
    detail_gain = float(np.clip(0.45 + 2.2 * float(mod), 0.3, 1.0))

    new_in = target + detail_gain * (detail * edge_w)
    out[in_brain] = new_in[in_brain]
    outer = (~in_brain) & (img > 0.1)
    out[outer] = img[outer] * float(skull_factor)
    out = cv2.GaussianBlur(out, (0, 0), sigmaX=float(blur_sigma))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# Tissue targets preserve the measured T2 spread: dark WM and bright CSF.
DEFAULT_PARAMS = {
    'strength': 1.0,
    'csf_brightness': 0.96,
    'gm_level': 0.56,
    'wm_level': 0.26,
    'texture': 0.12,
    'csf_pct': 20.0,
    'gm_pct': 67.0,
    'skull_compress': 0.7,
    'smoothing': 0.6,
}


def default_params() -> Dict:
    return DEFAULT_PARAMS.copy()


MASTER_MARK = 1.0
MASTER_MAX = 1.3
_MASTER_KEYS = ["strength"]
_MASTER_CLEAN = dict(strength=0.0)     # original T1
_MASTER_REAL = dict(strength=1.0)      # full T2 look
_MASTER_EXTREME = dict(strength=1.4)   # exaggerated beyond T2


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
    return p
