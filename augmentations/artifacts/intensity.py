"""intensity (and noise) augmentations.

Canonical implementations for the intensity/noise kinds. Moved verbatim from
the existing systems except where a single canonical (more-correct) variant was
chosen per the Stage 2 hard rules:

- ``bias`` is the SIGNAL-PRESERVING log-bias (de-mean the log field, multiply,
  renormalize by in-mask p99) — NOT the masking clip-to-1.0 applier (which
  manufactures saturation) and NOT ``mri_qc.augmentations.bias_field`` (clips the
  multiplicative field to >=1e-3, manufacturing voids). The old polynomial
  ``mri_qc`` bias stays byte-identical behind its own package wrapper; this is
  the canonical realistic bias used going forward.
- ``noise``/``clipping`` are moved verbatim from ``_make_artifact_battery``
  (``add_noise`` / ``clip_intensity``): WM-level additive Gaussian and a
  percentile intensity cap.

Canonical signature: ``fn(arr, *, severity, rng, mask=None, **kw)``.
"""
from __future__ import annotations

import numpy as np

from augmentations.registry import register


# --------------------------------------------------------------------------- noise
def _gaussian_thermal(data, sev, rng):
    """Additive zero-mean Gaussian (thermal) noise at a WM-ish tissue level.

    Verbatim core of the old battery ``add_noise``: sigma = ``sev * median(bright
    tissue)`` where bright = ``data > 0.4 * p99(pos)`` (a real tissue-noise level,
    not the air-dominated global median).
    """
    pos = data[data > 0]
    if pos.size == 0:
        return data.copy()
    bright = data > 0.4 * float(np.percentile(pos, 99))
    wm = float(np.median(data[bright])) if bright.any() else float(np.median(pos))
    return data + rng.normal(0.0, sev * wm, data.shape).astype(np.float32)


def _grain_smooth_delta(delta, sigma_grain):
    """Impose a spatial-correlation length on a noise *delta* with POWER renorm.

    Real recon noise is grainy/correlated (~1-2 voxel autocorrelation FWHM from
    k-space apodization / zero-fill / partial-Fourier / vendor denoising), not
    spatially white. This Gaussian-smooths the noise delta to give it a grain,
    then renormalizes its variance by the ANALYTIC factor ``1 / sqrt(sum k^2)``
    (k = the smoothing kernel) so noise POWER is preserved.

    Unlike ``mp2rage_grain``'s global-std renorm (which is dominated by the
    air/background variance and so under-preserves IN-MASK power), the analytic
    factor restores the variance of any zero-mean input exactly regardless of
    region. The Rician floor (which lives in dark out-of-mask voxels) is scaled
    by this factor and stays positive (preserved, not destroyed). ``sigma_grain
    <= 0`` returns the delta unchanged (white grain, the old behavior).
    """
    sg = float(sigma_grain)
    if sg <= 0.0:
        return delta
    from scipy.ndimage import gaussian_filter as _gf

    smoothed = _gf(delta, sg)
    # analytic 1/sqrt(sum kernel^2): filter a unit impulse to get the exact
    # kernel the smoother used (separable, truncated, edge-handled identically),
    # then its L2 norm gives the variance-shrink factor of smoothing white noise.
    imp = np.zeros_like(delta)
    imp[tuple(s // 2 for s in imp.shape)] = 1.0
    ker = _gf(imp, sg)
    k2 = float((ker.astype(np.float64) ** 2).sum())
    if k2 > 1e-12:
        smoothed = smoothed * np.float32(1.0 / np.sqrt(k2))
    return smoothed.astype(np.float32)


@register(
    "noise",
    kind="noise",
    severity_range=(0.0, 0.4),
    dims="3d",
    label_preserving=True,
    has_detector=True,
)
def noise(arr, *, severity, rng, mask=None, mode=None, sigma_grain=None, **kw):
    """Canonical MRI noise: a DECOMPOSED + OVER-GENERATED mixture sampler.

    Real MRI noise is not one thing. This canonical ``noise`` DECOMPOSES it into
    three low-level constituents and randomly SAMPLES/MIXES across them so the
    synthetic distribution is a SUPERSET enveloping each:

      * **thermal** — additive zero-mean Gaussian at the WM tissue level (the old
        verbatim battery ``add_noise``; can go negative).
      * **rician** — true magnitude (modulus-of-complex) noise with a positive
        low-signal FLOOR / bias (delegates to physics ``rician_noise``).
      * **structured** — spatially-varying parallel-imaging g-factor amplified
        noise (delegates to physics ``g_factor_noise``).

    Each call picks a random subset (1-3 of them) and a random per-component
    weight, then sums their noise CONTRIBUTIONS (so combinations are reachable,
    over-generating beyond any single model). A pure-Gaussian path is always
    reachable (``mode='thermal'`` forces it; the random sampler keeps a sizeable
    thermal-only probability). ``severity`` is the overall noise fraction (battery
    used 0.05-0.30). Same signature, single ndarray out, rng-driven.

    **Spatial grain (over-generated):** real scanner noise is spatially
    CORRELATED, not white. The summed noise delta is Gaussian-smoothed by a
    per-call ``sigma_grain`` (px) drawn ``U(0, 2.5)`` when not supplied, with the
    noise POWER preserved via the analytic ``1/sqrt(sum k^2)`` renorm (see
    ``_grain_smooth_delta``). White noise stays REACHABLE — the draw includes 0,
    and ``sigma_grain=0`` forces it. This broadens the white-only distribution
    into a superset that also envelops the grainy real regime.
    """
    data = np.asarray(arr, dtype=np.float32)
    sev = float(severity)
    if sev <= 0.0:
        return data.copy()
    if data.ndim == 4:
        out = np.empty_like(data)
        for t in range(data.shape[-1]):
            out[..., t] = noise(data[..., t], severity=sev, rng=rng, mask=mask,
                                mode=mode, sigma_grain=sigma_grain, **kw)
        return out

    # lazy import (physics imports only from .registry; avoids any import cycle)
    from augmentations.artifacts.physics import rician_noise as _rician, g_factor_noise as _gfac

    components = {
        "thermal": lambda w: _gaussian_thermal(data, sev * w, rng) - data,
        "rician": lambda w: _rician(data, severity=sev * w, rng=rng, mask=mask) - data,
        "structured": lambda w: _gfac(data, severity=sev * w, rng=rng, mask=mask) - data,
    }

    if mode in components:
        chosen = [mode]
    else:
        keys = list(components)
        # ~40% pure-thermal, otherwise a random 1-3 subset (thermal kept frequent)
        if rng.random() < 0.4:
            chosen = ["thermal"]
        else:
            k = int(rng.integers(1, len(keys) + 1))
            chosen = list(rng.choice(keys, size=k, replace=False))

    # Grain is imposed PER COMPONENT, and only on the WHITE thermal noise: the
    # rician / structured components delegate to physics.rician_noise /
    # g_factor_noise, which now SELF-grain their noise floor power-preservingly
    # (physics._spatial_grain). Re-graining their already-correlated delta with the
    # white-noise power renorm of _grain_smooth_delta would double-correlate it and
    # mis-scale its power, so they are summed AS-IS. The grain draw INCLUDES 0 so
    # white thermal noise stays reachable (sigma_grain=0 forces it).
    sg = float(rng.uniform(0.0, 2.5)) if sigma_grain is None else float(sigma_grain)
    delta = np.zeros_like(data)
    for key in chosen:
        w = float(rng.uniform(0.5, 1.2))  # over-generate weights past 1.0
        comp = components[key](w).astype(np.float32)
        if key == "thermal":
            comp = _grain_smooth_delta(comp, sg)  # the only spatially-white component
        delta = delta + comp

    # CALIBRATE the total noise AMPLITUDE to severity. The mode MIXTURE + grain above stay the
    # noise-CHARACTER knobs, but the summed delta is rescaled so its in-foreground RMS tracks
    # `sev` with only a controlled tail. The old code summed 1-3 components at unnormalized random
    # weights, so realized RMS spread ~5x within a band and adjacent bands OVERLAPPED (severity was
    # not a calibrated SNR change). Target ~= 0.5*sev matches the prior per-band MEAN, just tightened.
    region = (np.asarray(mask) > 0) if (mask is not None and np.asarray(mask).any()) else (data > 0.05)
    sq = np.square(delta[region]) if region.any() else np.square(delta)
    cur = float(np.sqrt(np.mean(sq))) if sq.size else 0.0
    if cur > 1e-8 and sev > 0.0:
        target = 0.5 * sev * float(rng.uniform(0.8, 1.25))   # centered on severity, controlled tail
        delta = delta * np.float32(target / cur)

    return (data + delta).astype(np.float32)


# --------------------------------------------------------------------------- clipping
@register(
    "clipping",
    kind="intensity",
    severity_range=(0.0, 1.0),
    dims="3d",
    label_preserving=True,
    has_detector=True,
)
def clipping(arr, *, severity, rng, mask=None, percentile=None, **kw):
    """Cap intensities at a high percentile (intensity clipping / saturation).

    Moved verbatim from ``_make_artifact_battery.clip_intensity``. The battery
    parameterizes this by the cap PERCENTILE (``q``: 80 -> mild, 50 -> strong),
    not a 0-1 severity, so pass ``percentile=q`` for byte-identical battery
    output. If ``percentile`` is omitted it is derived from ``severity`` as
    ``100 - 50*severity`` (advisory only).
    """
    data = np.asarray(arr, dtype=np.float32)
    if percentile is not None:                        # explicit battery path — byte-identical (unchanged)
        pos = data[data > 0]
        if pos.size == 0:
            return data.copy()
        return np.minimum(data, float(np.percentile(pos, float(percentile)))).astype(np.float32)
    # SEVERITY path (the masker uses this): cap only the BRIGHT TAIL of the FOREGROUND. Real
    # receiver-gain/window/DICOM saturation clips ~the top 1-15%. The old code took the percentile over
    # ALL positive voxels (~85% are dark non-brain bg/skull/neck), which dragged the cap BELOW the brain
    # body and clamped the whole brain to a flat near-background blob — training the over-inclusion
    # failure this synth effort exists to fix. Fix: q maps sev->[99.5,74.5] (top tail, never below the
    # median) and the cap is over the brain mask (or, mask-free, the >0.1*max foreground), not all >0.
    q = float(99.5 - 25.0 * float(severity))
    if mask is not None and np.asarray(mask).any():
        fg = data[np.asarray(mask).astype(bool)]
    else:
        fg = data[data > 0.1 * float(data.max())]
    if fg.size == 0:
        return data.copy()
    return np.minimum(data, float(np.percentile(fg, q))).astype(np.float32)


# --------------------------------------------------------------------------- quantization (bit-depth contouring)
@register(
    "quantization",
    kind="intensity",
    severity_range=(0.0, 1.0),
    dims="3d",
    label_preserving=True,
    has_detector=False,
)
def quantization(arr, *, severity, rng, mask=None, levels=None,
                 lut_warp=None, dither=None, **kw):
    """Intensity-amplitude QUANTIZATION / bit-depth contouring.

    A real degraded-provenance tell: 8/12-bit DICOM-windowed re-exports, lossy
    archival, and vendor LUT rescaling collapse the intensity onto a small set of
    discrete levels, producing terraced banding (contouring) in smooth regions
    and a comb-like histogram. Spatially, smooth gradients (bias fields, partial
    volume ramps, CSF) break into visible steps.

    DECOMPOSE into low-level features and OVER-GENERATE to a SUPERSET of reality:

      * **N levels** — round the normalized image onto ``N`` amplitude levels.
        Drawn log-uniform across ``~[16, 4096]`` (12-bit re-export ... near-clean)
        with the dominant level count tied to ``severity`` (high severity = FEWER
        levels = heavier contouring), then jittered. ``levels=N`` overrides.
      * **non-uniform / windowed LUT** — real DICOM window/level + vendor LUTs
        space levels non-uniformly (a gamma-like warp), so the step sizes vary
        with intensity. Applied with probability, warp exponent over-generated
        about 1.0 (1.0 recovers uniform steps). ``lut_warp=g`` overrides.
      * **dither** — light pre-quantization noise that scatters the banding
        (error-diffusion-like). rng-driven, re-clipped. ``dither=s`` overrides.

    Quantization is over the data's own ``[vmin, vmax]`` range, so it is robust to
    inputs already outside ``[0, 1]``; output is finite and clipped to ``[0, 1]``.
    ``label_preserving=True`` (pure amplitude remap; anatomy does not move).
    ``severity<=0`` returns an identity copy (uniform steps recover at large N).
    """
    data = np.asarray(arr, dtype=np.float32)
    sev = float(severity)
    if sev <= 0.0:
        return data.copy()
    if data.ndim == 4:
        out = np.empty_like(data)
        for t in range(data.shape[-1]):
            out[..., t] = quantization(
                data[..., t], severity=sev, rng=rng, mask=mask,
                levels=levels, lut_warp=lut_warp, dither=dither, **kw)
        return out

    vmin = float(data.min())
    vmax = float(data.max())
    span = vmax - vmin
    if span <= 1e-8:
        return data.copy()
    norm = (data - vmin) / span  # -> [0, 1]

    # --- N levels: severity drives the dominant count (fewer = heavier), then
    #     OVER-GENERATE log-uniform across [4, 4096] with a sev-coupled center.
    if levels is not None:
        n = int(levels)
    else:
        # dominant count: ~4096 (clean) at sev->0 down to ~8 at sev->1, log-spaced
        lo_log, hi_log = np.log(4.0), np.log(4096.0)
        center = hi_log - sev * (hi_log - lo_log)
        # jitter +/- ~1.5 natural-log units, clamped to the superset bounds
        lv = center + float(rng.uniform(-1.5, 1.5))
        lv = float(np.clip(lv, np.log(4.0), np.log(4096.0)))
        n = int(round(np.exp(lv)))
    n = max(2, int(n))

    work = norm
    # --- optional dither: scatter banding with light pre-quant noise (1/N scale)
    if dither is None:
        do_dither = bool(rng.random() < 0.4)
        d_scale = float(rng.uniform(0.2, 0.8)) / n if do_dither else 0.0
    else:
        d_scale = float(dither)
    if d_scale > 0.0:
        work = work + rng.normal(0.0, d_scale, work.shape).astype(np.float32)
        work = np.clip(work, 0.0, 1.0)

    # --- optional non-uniform (windowed/gamma-like) LUT warp: forward warp ->
    #     uniform quantize -> inverse warp, so step SIZES vary with intensity.
    if lut_warp is None:
        g = float(np.exp(rng.uniform(-0.7, 0.7))) if rng.random() < 0.5 else 1.0
    else:
        g = float(lut_warp)
    if abs(g - 1.0) > 1e-6:
        warped = np.power(np.clip(work, 0.0, 1.0), g)
        q = np.round(warped * (n - 1)) / (n - 1)
        q = np.clip(q, 0.0, 1.0)
        quant = np.power(q, 1.0 / g)
    else:
        quant = np.round(work * (n - 1)) / (n - 1)

    out = quant * span + vmin
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- bias (canonical, signal-preserving)
@register(
    "bias",
    kind="intensity",
    severity_range=(0.0, 0.8),
    dims="3d",
    label_preserving=True,
    has_detector=True,
    extra_parameters=("field_kind",),
)
def bias(arr, *, severity, rng, mask=None, voxel_length_range=(3, 96), **kw):
    """Signal-PRESERVING multiplicative log-bias field (SynthStrip Table-1 style).

    Recipe (signal_preserving_log_bias): sample a low-frequency Gaussian grid
    (voxel length 4-64), upsample, DE-MEAN the log field (so the field does not
    shift overall brightness), exponentiate, multiply, then RENORMALIZE by the
    in-mask p99 instead of hard-clipping. This avoids the two known bugs:

      * masking ``add_bias_field`` clip-to-1.0 (manufactures saturation), and
      * ``mri_qc.augmentations.bias_field`` clip-field-to-1e-3 (manufactures voids).

    ``mask`` is optional: with no mask the p99 renorm uses positive voxels, so
    mask-free callers (the battery / pipeline) still get a sane scale.
    ``severity`` plays the role of the Gaussian-grid sigma.
    """
    from scipy.ndimage import zoom as _zoom

    data = np.asarray(arr, dtype=np.float32)
    sev = float(severity)
    if sev <= 0.0:
        return data.copy()
    if data.ndim == 4:
        out = np.empty_like(data)
        for t in range(data.shape[-1]):
            out[..., t] = bias(data[..., t], severity=sev, rng=rng, mask=mask, **kw)
        return out
    if data.ndim != 3:
        raise ValueError(f"bias expects 3D or 4D, got {data.ndim}D")

    shape = data.shape
    # OVER-GENERATE the field SHAPE: randomly pick a generator so the synthetic
    # bias distribution supersets {low-freq Gaussian grid, multi-lobe surface-coil
    # drop-off, 7T-style elliptical inhomogeneity}, with wide voxel-scale / sigma.
    field_kind = kw.get("field_kind")
    if field_kind is None:
        field_kind = str(rng.choice(["grid", "coil", "elliptical"]))

    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    if field_kind == "coil":
        # multi-lobe surface-coil: sum of a few smooth radial drop-offs from
        # random coil centers near the surface (strong near-coil bright lobes).
        logf = np.zeros(shape, np.float32)
        n_coils = int(rng.integers(2, 6))
        for _ in range(n_coils):
            cz = rng.uniform(0, shape[0]); cy = rng.uniform(0, shape[1]); cx = rng.uniform(0, shape[2])
            # wide spatial scale (voxel-length analogue) and sev-scaled amplitude
            scl = float(rng.uniform(*voxel_length_range)) * float(rng.uniform(1.0, 3.0))
            amp = float(rng.uniform(0.5, 2.0)) * sev * (1 if rng.random() < 0.5 else -1)
            d2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
            logf = logf + (amp * np.exp(-d2 / (2.0 * scl ** 2))).astype(np.float32)
    elif field_kind == "elliptical":
        # 7T-style elliptical (B1+) inhomogeneity: a smooth anisotropic quadratic
        # bowl with random axis ratios + a low-freq grid ripple on top.
        cz, cy, cx = (shape[0] / 2.0, shape[1] / 2.0, shape[2] / 2.0)
        az = float(rng.uniform(0.6, 2.0)); ay = float(rng.uniform(0.6, 2.0)); ax = float(rng.uniform(0.6, 2.0))
        q = (((zz - cz) / (shape[0] * az)) ** 2
             + ((yy - cy) / (shape[1] * ay)) ** 2
             + ((xx - cx) / (shape[2] * ax)) ** 2).astype(np.float32)
        amp = float(rng.uniform(1.0, 3.0)) * sev * (1 if rng.random() < 0.5 else -1)
        logf = amp * q
        # add a coarse grid ripple so it is not a clean bowl
        vl = float(rng.uniform(*voxel_length_range))
        grid = [max(2, int(round(s / vl))) for s in shape]
        ripple = rng.normal(0.0, 0.5 * sev, size=tuple(grid)).astype(np.float32)
        factors = [shape[i] / ripple.shape[i] for i in range(3)]
        ripple = _zoom(ripple, factors, order=3)
        sl = tuple(slice(0, shape[i]) for i in range(3))
        ripple = ripple[sl]
        if ripple.shape != shape:
            ripple = np.pad(ripple, [(0, shape[i] - ripple.shape[i]) for i in range(3)], mode="edge")
        logf = logf + ripple
    else:
        # canonical low-frequency Gaussian grid (over-generated voxel-length range)
        vl = float(rng.uniform(*voxel_length_range))
        grid = [max(2, int(round(s / vl))) for s in shape]
        field = rng.normal(0.0, sev, size=tuple(grid)).astype(np.float32)
        factors = [shape[i] / field.shape[i] for i in range(3)]
        logf = _zoom(field, factors, order=3)
        sl = tuple(slice(0, shape[i]) for i in range(3))
        logf = logf[sl]
        if logf.shape != shape:
            pad = [(0, shape[i] - logf.shape[i]) for i in range(3)]
            logf = np.pad(logf, pad, mode="edge")
    # de-mean the log field so the field does not shift overall brightness
    logf = logf - float(logf.mean())
    bias_mult = np.exp(logf).astype(np.float32)
    out = (data * bias_mult).astype(np.float32)
    # renormalize by in-mask (or positive-voxel) p99 instead of hard-clipping
    use_mask = mask is not None and np.asarray(mask).any()
    ref = out[np.asarray(mask).astype(bool)] if use_mask else out[out > 0]
    if ref.size:
        p99 = float(np.percentile(ref, 99))
        in_ref = data[np.asarray(mask).astype(bool)] if use_mask else data[data > 0]
        p99_in = float(np.percentile(in_ref, 99)) if in_ref.size else 1.0
        if p99 > 1e-6:
            out = out * (p99_in / p99)
    return out.astype(np.float32)
