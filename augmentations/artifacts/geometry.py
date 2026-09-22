"""geometry augmentations (anisotropy).

The QC battery's ``anisotropy`` is HEADER-ONLY (it only rewrites NIfTI voxel
zooms; the voxel array is untouched) and therefore cannot be expressed as an
array->array transform — it stays in ``_make_artifact_battery`` as a header
edit. Here we register the canonical REAL anisotropy ("venetian-blind"):
gaussian-blur + downsample + upsample along ONE axis, which actually degrades
through-plane resolution in the voxel data (AUGMENTATION_RESEARCH.md QC #3).

Canonical signature: ``fn(arr, *, severity, rng, mask=None, **kw)``.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from augmentations.registry import register


@register(
    "anisotropy",
    kind="geometry",
    severity_range=(0.0, 1.0),
    dims="3d",
    label_preserving=True,
    has_detector=True,
)
def anisotropy(arr, *, severity, rng, mask=None, axis=None, ratio=None,
               slice_gap=None, **kw):
    """Real through-plane anisotropy: blur + down/up-sample along the through axis.

    The through-plane (slice) axis is the acquisition's low-resolution direction,
    which depends on the protocol (axial / coronal / sagittal). Previously it was
    HARDCODED to axis-2 (L-R = sagittal-only, atypical), so every case degraded
    the same direction — a dataset-level tell. The through axis is now drawn
    ``rng∈{0,1,2}`` per call (uniform), broadening the superset to all three
    acquisition planes; pass ``axis`` explicitly to override.

    ``ratio`` is the effective slice-thickness factor (>1). If omitted it is
    derived from ``severity`` as ``1 + 8*severity`` (so sev 0.25 -> 3x, sev 1 ->
    9x, matching the battery's header-only ANISO levels of 3x / 9x). The blur
    sigma is ``ratio/4`` (SynthStrip partial-volume model). Down/up sampling is
    along the through axis only so in-plane resolution is preserved.

    ``slice_gap`` (in [0,1], probabilistic when ``None``) optionally zeros a thin
    inter-slice gap on the decimated grid (skip-acquisition / slice gap), the 3D
    analog of the 2D sibling's gap zeroing. ``slice_gap=0`` disables it.
    """
    data = np.asarray(arr, dtype=np.float32)
    sev = float(severity)
    if sev <= 0.0 and ratio is None:
        return data.copy()
    # resolve the per-ACQUISITION axis ONCE (before 4D dispatch) so every
    # timepoint of one volume degrades the same through-plane direction.
    if axis is None:
        axis = int(rng.integers(0, 3))
    if slice_gap is None:
        # Periodic inter-slice DROPOUT (dark bands) is a DISTINCT feature from PSF resolution loss;
        # severity-GATE both its firing probability and its depth so (a) the band depth is monotone
        # in severity instead of a flat uniform(0.15,0.5), and (b) the MILD end stays pure-PSF
        # anisotropy (bands rarely fire) -- decoupling the two features the old code entangled.
        slice_gap = ((0.08 + 0.42 * sev) * float(rng.uniform(0.8, 1.2))
                     if rng.random() < (0.10 + 0.30 * sev) else 0.0)
    if data.ndim == 4:
        out = np.empty_like(data)
        for t in range(data.shape[-1]):
            out[..., t] = anisotropy(data[..., t], severity=sev, rng=rng,
                                     mask=mask, axis=axis, ratio=ratio,
                                     slice_gap=slice_gap, **kw)
        return out
    if data.ndim != 3:
        raise ValueError(f"anisotropy expects 3D or 4D, got {data.ndim}D")
    r = float(ratio) if ratio is not None else (1.0 + 8.0 * sev)
    if r <= 1.0:
        return data.copy()
    n = data.shape[axis]
    sigma = r / 4.0
    # blur along the chosen axis (partial-volume), then decimate + restore size
    blurred = ndi.gaussian_filter1d(data, sigma=sigma, axis=axis)
    small_n = max(1, int(round(n / r)))
    zoom_down = [1.0, 1.0, 1.0]
    zoom_down[axis] = small_n / n
    small = ndi.zoom(blurred, zoom_down, order=1)
    # optional slice gap: zero a thin slab between acquired slices on the coarse
    # grid (skip-acquisition). Keep at least one acquired slice intact.
    if slice_gap > 0.0 and small.shape[axis] >= 3:
        # attenuate every other coarse slice (a periodic inter-slice gap)
        sg = [slice(None)] * 3
        sg[axis] = slice(1, None, 2)
        small[tuple(sg)] *= float(max(0.0, 1.0 - slice_gap))
    zoom_up = [1.0, 1.0, 1.0]
    zoom_up[axis] = n / small.shape[axis]
    up = ndi.zoom(small, zoom_up, order=1)
    # fix any off-by-one from rounding
    if up.shape[axis] != n:
        sl = [slice(None)] * 3
        sl[axis] = slice(0, n)
        up = up[tuple(sl)]
        if up.shape[axis] != n:
            pad = [(0, 0)] * 3
            pad[axis] = (0, n - up.shape[axis])
            up = np.pad(up, pad, mode="edge")
    return up.astype(np.float32)
