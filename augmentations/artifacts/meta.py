"""meta augmentations (nonfinite injection, constant volume).

These are 'meta' failure modes — not physical MRI artifacts but data-integrity
defects the QC pipeline must catch. Moved verbatim from
``_make_artifact_battery`` (``inject_nonfinite`` / ``make_constant``).

Canonical signature: ``fn(arr, *, severity, rng, mask=None, **kw)``.
"""
from __future__ import annotations

import numpy as np

from augmentations.registry import register


def _head_median(data):
    pos = data[data > 0]
    if pos.size == 0:
        return 1.0
    return float(np.median(pos))


@register(
    "nonfinite",
    kind="meta",
    severity_range=(0.0, 0.05),
    dims="either",
    label_preserving=True,
    has_detector=True,
)
def nonfinite(arr, *, severity, rng, mask=None, **kw):
    """Inject NaNs at a random fraction ``severity`` of voxels. Verbatim from
    ``_make_artifact_battery.inject_nonfinite`` (``frac`` == ``severity``).
    """
    data = np.asarray(arr)
    out = data.copy().astype(np.result_type(data.dtype, np.float32), copy=False)
    idx = rng.random(data.shape) < float(severity)
    out[idx] = np.nan
    return out


@register(
    "constant",
    kind="meta",
    severity_range=(0.0, 1.0),
    dims="either",
    label_preserving=False,
    has_detector=True,
)
def constant(arr, *, severity, rng, mask=None, **kw):
    """Replace the whole volume with a constant = head median. Verbatim from
    ``_make_artifact_battery.make_constant`` (``sev`` unused; kept for symmetry).
    """
    data = np.asarray(arr)
    med = _head_median(data)
    return np.full(data.shape, med, dtype=np.float32)
