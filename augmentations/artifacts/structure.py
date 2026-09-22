"""structure augmentations (through-plane spin-history slice banding).

New simulator from AUGMENTATION_RESEARCH.md (QC #6): interleaved-acquisition
spin-history produces alternating bright/dark slice BANDS (a few percent
intensity modulation along the slice axis). 3D only.

Canonical signature: ``fn(arr, *, severity, rng, mask=None, **kw)``.
"""
from __future__ import annotations

import numpy as np

from augmentations.registry import register


@register(
    "spin_history",
    kind="structure",
    severity_range=(0.0, 1.0),
    dims="3d",
    label_preserving=True,
    has_detector=False,
)
def spin_history(arr, *, severity, rng, mask=None, axis=2, k=None,
                 drift=None, **kw):
    """Through-plane spin-history banding: period-``k`` bright/dark slice bands.

    Interleaved multi-slice 2D acquisition orders slices in ``k`` groups, so the
    incomplete-recovery brightness modulation repeats with PERIOD ``k`` (not just
    the factor-2 venetian blind). ``k`` is drawn ``rng.choice([2,2,2,3,4])`` so
    k=2 (classic odd/even venetian blind) stays the dominant draw while k=3/4
    interleaves broaden the superset; pass ``k`` explicitly to override.

    Each period group gets its OWN signed level; the per-slice magnitude keeps the
    ``U(0.5,1.0)`` jitter and the 3%..33% amplitude superset. An optional smooth
    saturation-drift ramp (``drift`` in [0,1], probabilistic when ``None``) tilts
    the band envelope along ``axis`` (progressive saturation down the slab).
    Multiplied factor stays strictly positive, so the op is intensity-only and
    label-preserving.
    """
    data = np.asarray(arr, dtype=np.float32)
    sev = float(severity)
    if sev <= 0.0:
        return data.copy()
    # resolve the per-ACQUISITION parameters ONCE (before 4D dispatch) so every
    # timepoint of one volume shares the same interleave/drift.
    if k is None:
        k = int(rng.choice([2, 2, 2, 3, 4]))
    k = max(2, int(k))
    if drift is None:
        drift = float(rng.uniform(0.0, 1.0)) if rng.random() < 0.5 else 0.0
    if data.ndim == 4:
        out = np.empty_like(data)
        for t in range(data.shape[-1]):
            out[..., t] = spin_history(data[..., t], severity=sev, rng=rng,
                                       mask=mask, axis=axis, k=k, drift=drift, **kw)
        return out
    if data.ndim != 3:
        raise ValueError(f"spin_history expects 3D or 4D, got {data.ndim}D")
    n = data.shape[axis]
    max_amp = 0.03 + 0.30 * sev  # 3%..33% — over-generate beyond the real 3-10% for coverage
    idx = np.arange(n)
    # one signed level per period group (k=2 with levels [+1,-1] recovers the
    # classic alternating venetian blind); span [-1,1] so groups go bright/dark.
    group_levels = rng.uniform(-1.0, 1.0, size=k).astype(np.float32)
    level = group_levels[idx % k]
    jitter = rng.uniform(0.5, 1.0, size=n).astype(np.float32)
    band = max_amp * level * jitter  # in (-max_amp, max_amp), |band| < 1
    if drift > 0.0 and n > 1:
        # smooth saturation-drift ramp: progressive dimming along the slab axis
        ramp = 1.0 - drift * max_amp * (idx / float(n - 1)).astype(np.float32)
        factor = ((1.0 + band) * ramp).astype(np.float32)
    else:
        factor = (1.0 + band).astype(np.float32)
    shape = [1, 1, 1]
    shape[axis] = n
    return (data * factor.reshape(shape)).astype(np.float32)
