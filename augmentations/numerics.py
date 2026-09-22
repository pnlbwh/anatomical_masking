"""Numerical operations shared by MRI appearance and artifact renderers."""

import numpy as np
from scipy import ndimage as ndi


def iter_4d(fn, data, severity, rng, mask, **kw):
    """Dispatch a 3D transform over the trailing axis of a 4D volume."""
    out = np.empty_like(data, dtype=np.result_type(data.dtype, np.float32))
    for t in range(data.shape[-1]):
        out[..., t] = fn(data[..., t], severity=severity, rng=rng, mask=mask, **kw)
    return out


def head_mask(data, mask):
    """Boolean head/tissue region: use ``mask`` if given else an intensity head.

    Mirrors ``focal._head`` (``data > 0.25 * median(pos)``) so mask-free callers
    still get a sane region for anatomy-referencing transforms.
    """
    if mask is not None:
        m = np.asarray(mask).astype(bool)
        if m.any():
            return m
    pos = data[data > 0]
    if pos.size == 0:
        return np.zeros_like(data, bool)
    med = float(np.median(pos))
    return data > 0.25 * med


def smooth_random_field(shape, rng, sigma):
    """Zero-mean unit-ish smooth random scalar field of ``shape``."""
    f = rng.standard_normal(shape).astype(np.float32)
    f = ndi.gaussian_filter(f, sigma=sigma)
    f -= float(f.mean())
    s = float(np.abs(f).max())
    if s > 1e-6:
        f /= s
    return f


def robust_sigma(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return 0.0
    median = float(np.median(values))
    return float(1.4826 * np.median(np.abs(values - median)))
