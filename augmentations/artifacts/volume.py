"""Volume renderers for ringing, metal, T2, and MP2RAGE/UNI artifacts.

These registered renderers share the parameter landmarks of their lower-level
families, with volume geometry and acquisition physics coherent in all views.
Severity is the master strength: 0 is clean, 1 is the reference appearance,
and values up to 1.3 extend beyond that reference.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from augmentations.registry import apply, register
from augmentations.artifacts import ringing as _ringing
from augmentations.artifacts import metal as _metal
from augmentations.protocols import t2 as _t2
from augmentations.protocols import mp2rage as _mp2rage


def _rng(rng):
    return np.random.default_rng(rng)


def _brain_mask(arr, mask):
    if mask is not None and np.asarray(mask).shape == arr.shape:
        return np.asarray(mask) > 0.5
    return _image_head_mask(arr)


def _image_head_components(arr):
    """Return image-derived ``(filled_head, signal_foreground)`` supports.

    Ringing must not use a training target brain mask: doing so places its support edge exactly on the
    answer. Build a conservative foreground/head component from intensities instead. Other sequence
    renderers may still call ``_brain_mask`` with a mask because their tissue statistics need it; the
    ringing wrapper below deliberately calls this image-only helper.
    """
    img = np.nan_to_num(np.asarray(arr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    pos = img[img > 0]
    if not pos.size:
        empty = np.zeros_like(img, dtype=bool)
        return empty, empty.copy()
    scale = float(np.percentile(pos, 99.0))
    thr = max(0.02, 0.08 * scale)
    fg = img > thr
    structure = ndi.generate_binary_structure(img.ndim, 1)
    fg = ndi.binary_closing(fg, structure=structure, iterations=2)
    head = ndi.binary_fill_holes(fg)
    lbl, n = ndi.label(head, structure=structure)
    if n > 1:
        sizes = ndi.sum(np.ones_like(lbl), lbl, index=np.arange(1, n + 1))
        head = lbl == (int(np.argmax(sizes)) + 1)
    if int(head.sum()) < 100:
        fg = img > max(0.01, 0.04 * scale)
        head = ndi.binary_fill_holes(fg)
    head = np.asarray(head, dtype=bool)
    return head, np.asarray(fg & head, dtype=bool)


def _image_head_mask(arr):
    """Image-derived whole-head support for boundary effects."""
    return _image_head_components(arr)[0]


def _soft_membership(mask, width=2.0, *, sampling=None):
    """Feather a supplied annotation so rendering never switches at its exact edge."""
    b = np.asarray(mask) > 0.5
    width = float(width)
    if width <= 0.0 or not b.any() or b.all():
        return b.astype(np.float32)
    if sampling is not None:
        sampling = tuple(float(v) for v in sampling)
        if len(sampling) != b.ndim:
            sampling = None
    signed = _signed_distance(b, sampling=sampling)
    return _soft_membership_from_signed(signed, width)


def _signed_distance(mask, *, sampling=None):
    """Physical signed distance, positive inside a Boolean support."""
    b = np.asarray(mask) > 0.5
    return (ndi.distance_transform_edt(b, sampling=sampling)
            - ndi.distance_transform_edt(~b, sampling=sampling))


def _soft_membership_from_signed(signed, width=2.0, *, dilation=0.0):
    """Smoothstep membership from one reusable signed-distance field."""
    width = max(float(width), 1e-6)
    t = np.clip(
        0.5 + (np.asarray(signed) + float(dilation)) / (2.0 * width),
        0.0, 1.0).astype(np.float32)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _coarse_acquisition_support(mask, coarsen=0.0, *, sampling=None):
    """Low-pass a semantic support so fine target detail cannot gate acquisition.

    The zero endpoint is an exact copy for legacy replay. Positive ``coarsen`` is
    a physical Gaussian FWHM when spacing is supplied, otherwise voxel units.
    """
    support = np.asarray(mask) > 0.5
    coarsen = max(float(coarsen), 0.0)
    if coarsen <= 0.0 or not support.any() or support.all():
        return support.copy()
    if sampling is None:
        spacing = np.ones(support.ndim, dtype=np.float64)
    else:
        spacing = np.asarray(tuple(float(v) for v in sampling), dtype=np.float64)
        if spacing.size != support.ndim or np.any(spacing <= 0.0):
            spacing = np.ones(support.ndim, dtype=np.float64)
    sigma = coarsen / 2.354820045 / spacing
    probability = ndi.gaussian_filter(
        support.astype(np.float32), sigma=np.maximum(sigma, 0.25), mode="nearest")
    coarse = probability >= 0.5
    return coarse if coarse.any() else support.copy()


def _resize_centered_3d(arr, scale, order):
    """Shrink (scale<1) / grow a volume toward its center, padded to original shape."""
    if abs(scale - 1.0) < 1e-6:
        return arr.copy()
    small = ndi.zoom(arr, (scale, scale, scale), order=order)
    out = np.zeros(arr.shape, dtype=arr.dtype)
    sl = tuple(slice(max(0, (s - ns) // 2), max(0, (s - ns) // 2) + min(s, ns))
               for s, ns in zip(arr.shape, small.shape))
    csl = tuple(slice(0, min(s, ns)) for s, ns in zip(arr.shape, small.shape))
    out[sl] = small[csl]
    return out


def _grad_mag_3d(v):
    return np.sqrt(ndi.sobel(v, 0) ** 2 + ndi.sobel(v, 1) ** 2 + ndi.sobel(v, 2) ** 2)


# =========================================================================== #
# 1. ringing — cortical-halo SHELLS (3D port of ringing.cortical_halo)         #
# =========================================================================== #
@register("ringing_3d", kind="kspace", severity_range=(0.0, 1.3), dims="3d",
          label_preserving=True, has_detector=False)
def ringing_3d(arr, *, severity=1.0, rng=None, mask=None, master_t=None, **params):
    img = np.asarray(arr, dtype=np.float32)
    # Deliberately ignore the segmentation target: its exact boundary is an answer cue. Ring support
    # comes from the rendered image/head itself.
    m = _image_head_mask(img)
    pp = _ringing.master_params(float(master_t) if master_t is not None else float(severity))
    if int(pp["n_passes"]) <= 0 or int((m).sum()) < 100:
        return np.clip(img, 0, 1).astype(np.float32)
    g = _rng(rng)
    ss = float(pp["scale_step"])
    elo, ehi = sorted((float(pp["erosion_lo"]), float(pp["erosion_hi"])))
    bump_pos = 1.0 / max(1e-6, float(pp["ring_bright"]))
    bump_neg = (1.0 / float(pp["ring_dark"])) if pp["ring_dark"] > 1e-6 else 1e9
    out = img.copy()
    mb = m.astype(np.uint8)
    bmean = float(out[mb > 0].mean())
    for i in range(int(pp["n_passes"])):
        scale = float(pp["init_scale"]) - ss * i
        if scale <= 0.20:
            break
        grad = _grad_mag_3d(_resize_centered_3d(out, scale, 1))
        gmax = float(grad.max())
        if gmax < 1e-6:
            continue
        edges = (grad / gmax >= 50.0 / 255.0) & (mb > 0)
        small_mask = _resize_centered_3d(mb.astype(np.float32), float(g.uniform(elo, ehi)), 0) > 0.5
        edges &= ~small_mask
        if int(edges.sum()) < 20:
            continue
        out[edges] += float(g.uniform(-bmean / bump_neg, bmean / bump_pos))
    return np.clip(out, 0, 1).astype(np.float32)


# =========================================================================== #
# 2. metal — focal 3D susceptibility (3D port of metal.compose_metal)          #
# =========================================================================== #
def _metal_center(img, mask, loc, g):
    if loc is not None:
        return [float(loc[a]) * img.shape[a] for a in range(3)]
    if mask is not None and np.asarray(mask).sum() > 50:
        idx = np.where(np.asarray(mask) > 0.5)
        c = [(idx[a].min() + idx[a].max()) / 2.0 for a in range(3)]
        ext = [idx[a].max() - idx[a].min() for a in range(3)]
        ax = int(np.argmax(ext))                       # put it at an edge along the longest axis
        c[ax] = idx[ax].min() + 0.20 * ext[ax]
        return c
    return [0.5 * s for s in img.shape]


@register("metal_3d", kind="focal", severity_range=(0.0, 1.3), dims="3d",
          label_preserving=True, has_detector=True, extra_parameters=('size_scale',))
def metal_3d(arr, *, severity=1.0, rng=None, mask=None, master_t=None, loc=None, **params):
    img = np.asarray(arr, dtype=np.float32)
    pp = _metal.master_params(float(master_t) if master_t is not None else float(severity))
    cx, cy, cz = _metal_center(img, mask, loc, _rng(rng))
    xx, yy, zz = np.indices(img.shape).astype(np.float32)
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) + 1e-3
    cos_t = (zz - cz) / r                              # bilobed about the through-slice axis
    sc = float(params.get("size_scale", 1.0))          # tiny clip <-> head-spanning dental/orbital bloom
    radius, vrad, pw = float(pp["radius"]) * sc, float(pp["void_radius"]) * sc, float(pp["pileup_width"]) * sc
    out = img + float(pp["dipole_amp"]) * (3.0 * cos_t ** 2 - 1.0) * np.exp(-r ** 2 / (2 * radius ** 2))
    out = out * (1.0 - float(pp["void_depth"]) * np.exp(-r ** 2 / (2 * vrad ** 2)))
    out = out + float(pp["pileup_amp"]) * np.exp(-((r - vrad) ** 2) / (2 * pw ** 2))
    return np.clip(out, 0, 1).astype(np.float32)


# =========================================================================== #
# 3. t2 — region-aware T1->T2 remap (3D port of t2.compose_t2)                  #
# =========================================================================== #
@register("t2_3d", kind="intensity", severity_range=(0.0, 1.3), dims="3d",
          label_preserving=True, has_detector=False)
def t2_3d(arr, *, severity=1.0, rng=None, mask=None, master_t=None,
          csf_pct=20.0, gm_pct=67.0, csf_target=0.95, gm_target=0.50, wm_target=0.24,
          trans_frac=0.09, detail_sigma=2.0, edge_k=0.08, detail_gain=0.7,
          skull_factor=0.7, blur_sigma=0.6, **params):
    img = np.asarray(arr, dtype=np.float32)
    inb = _brain_mask(img, mask)
    if int(inb.sum()) < 50:
        return np.clip(img, 0, 1).astype(np.float32)
    strength = float(_t2.master_params(float(master_t) if master_t is not None else float(severity))["strength"])
    bp = img[inb]
    csf_t = float(np.percentile(bp, csf_pct))
    gm_t = float(np.percentile(bp, gm_pct))
    span = max(1e-3, float(bp.max() - bp.min()))
    wt = max(1e-3, trans_frac * span)
    w_csf = 1.0 / (1.0 + np.exp(-(csf_t - img) / wt))
    w_wm = 1.0 / (1.0 + np.exp(-(img - gm_t) / wt))
    w_gm = np.clip(1.0 - w_csf - w_wm, 0.0, None)
    s = w_csf + w_gm + w_wm + 1e-6
    target = (w_csf / s * csf_target + w_gm / s * gm_target + w_wm / s * wm_target).astype(np.float32)
    detail = img - ndi.gaussian_filter(img, detail_sigma)
    edge_w = 1.0 / (1.0 + (_grad_mag_3d(target) / max(1e-4, edge_k)) ** 2)
    remap = img.copy()
    remap[inb] = (target + detail_gain * (detail * edge_w))[inb]
    outer = (~inb) & (img > 0.1)
    remap[outer] = img[outer] * skull_factor
    remap = ndi.gaussian_filter(remap, blur_sigma)
    out = (1.0 - strength) * img + strength * remap
    return np.clip(out, 0, 1).astype(np.float32)


# =========================================================================== #
# 4. mp2rage — empirical UNI tissue family + physical complex-ratio air          #
# =========================================================================== #
# Real MP2RAGE/UNI in-brain intensity quantiles (0..100 step 5), measured
# after the deployment pipeline's positive-p99 normalization. The original
# (darker) reference and the bundled UNIT1 (brighter) are both retained: real
# protocol/site variation is coherent, not independent per-quantile jitter.
_UNI_REF_Q = np.array(
    [0.003, 0.055, 0.120, 0.199, 0.260, 0.306, 0.340, 0.367, 0.391, 0.414, 0.438,
     0.465, 0.497, 0.534, 0.576, 0.618, 0.652, 0.678, 0.700, 0.725, 0.95], dtype=np.float32)
_UNI_REF_Q_BRIGHT = np.array(
    [0.007499, 0.117801, 0.218471, 0.294511, 0.341371, 0.371617,
     0.394647, 0.415528, 0.437758, 0.463988, 0.497734, 0.540028,
     0.591438, 0.642564, 0.683276, 0.710039, 0.728520, 0.743501,
     0.758232, 0.776446, 1.0], dtype=np.float32)
_UNI_REF_P = np.linspace(0.0, 100.0, len(_UNI_REF_Q))

# Master-t landmarks for the 3D renderer, mirroring ``mp2rage.MASTER_MARK`` / ``MASTER_MAX``:
# MARK is where the look matches a real MP2RAGE, and MARK..MAX is the deliberately
# OVER-GENERATED zone (the superset strategy — reality is one region of a wider space).
MP2RAGE_3D_MARK = 1.0
MP2RAGE_3D_MAX = 1.3


def _uni_ref_curve(gen, *, ref_spread=0.0, ref_mix=None, fold_prob=0.0,
                   fold_depth=(0.15, 0.40)):
    """One draw from a FAMILY of UNI reference quantile curves around ``_UNI_REF_Q``.

    Two measured anchors span a coherent dark-to-bright protocol/site family:

    * ``ref_mix`` explicitly interpolates dark (0) to bundled bright (1). When it is
      omitted and ``ref_spread>0``, a mix is drawn uniformly.
    * ``ref_spread`` perturbs the strictly-positive log *increments* of that curve,
      then renormalizes them. This varies shape without flat/collapsed rank intervals.
    * ``fold_prob`` pulls 1-2 interior points DOWN, making the curve NON-MONOTONE. This
      is the structural gap: ``realistic_acquisition._remap3d`` interpolates three
      ORDERED tissue anchors, so the GM/WM ordering change that makes MP2RAGE and PSIR
      recognizable is unreachable there at any parameter value. Fold count and depth
      mirror ``mp2rage.sample_params``.

    Returns ``_UNI_REF_Q`` itself and consumes no RNG when every family knob is off,
    preserving deterministic replay for the explicit hard-boundary compatibility path.
    """
    if ref_spread <= 0.0 and fold_prob <= 0.0 and ref_mix is None:
        return _UNI_REF_Q
    mix = (float(np.clip(ref_mix, 0.0, 1.0)) if ref_mix is not None
           else (float(gen.uniform(0.0, 1.0)) if ref_spread > 0.0 else 0.0))
    q = ((1.0 - mix) * _UNI_REF_Q.astype(np.float64)
         + mix * _UNI_REF_Q_BRIGHT.astype(np.float64))
    if ref_spread > 0.0:
        increments = np.diff(q)
        log_sd = min(0.50, 4.0 * float(ref_spread))
        increments *= np.exp(gen.normal(0.0, log_sd, increments.size))
        span = float(q[-1] - q[0])
        increments *= span / max(float(increments.sum()), 1e-12)
        q[1:] = q[0] + np.cumsum(increments)
        q[-1] = ((1.0 - mix) * float(_UNI_REF_Q[-1])
                 + mix * float(_UNI_REF_Q_BRIGHT[-1]))
    if fold_prob > 0.0 and gen.random() < float(fold_prob):
        for _ in range(int(gen.integers(1, 3))):
            j = int(gen.integers(1, q.size - 1))
            q[j] = max(0.0, q[j] - float(gen.uniform(*fold_depth)))
    return q.astype(np.float32)


def _quantile_map(values, src_q, dst_q):
    """Interpolate a quantile map without undefined repeated source knots."""
    src = np.asarray(src_q, dtype=np.float64)
    dst = np.asarray(dst_q, dtype=np.float64)
    unique, inverse = np.unique(src, return_inverse=True)
    if unique.size == 1:
        return np.full(np.asarray(values).shape, float(np.median(dst)), dtype=np.float32)
    counts = np.bincount(inverse)
    target = np.bincount(inverse, weights=dst) / np.maximum(counts, 1)
    return np.interp(values, unique, target).astype(np.float32)


def _smoothstep01(values):
    values = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    return (values * values * (3.0 - 2.0 * values)).astype(np.float32)


def _grain_preserve_moments(field, sigma=0.0, anisotropy=1.0):
    """Correlate an n-D field without silently changing its mean or variance."""
    field = np.asarray(field, dtype=np.float32)
    sigma = max(float(sigma), 0.0)
    if sigma <= 0.0 or field.size < 2:
        return field.copy()
    sigmas = [sigma] * field.ndim
    sigmas[0] *= max(float(anisotropy), 1e-3)
    before_mean, before_sd = float(field.mean()), float(field.std())
    out = ndi.gaussian_filter(field, sigmas)
    after_mean, after_sd = float(out.mean()), float(out.std())
    if before_sd > 1e-8 and after_sd > 1e-8:
        out = (out - after_mean) * (before_sd / after_sd) + before_mean
    else:
        out = out - after_mean + before_mean
    return np.asarray(out, dtype=np.float32)


def _unit_interval(value, name):
    """Validate one explicit mixture fraction without silently clipping it."""
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return value


def _thermal_grain_mixture(fine, correlated_source, *, sigma=0.0,
                           anisotropy=1.0, thermal_fraction=0.0):
    """Mix fine thermal noise with correlated grain at fixed mean and variance.

    ``thermal_fraction`` is the requested pre-correction fraction of variance in
    the unsmoothed realization.  The correlated component is independently drawn
    by the caller, then filtered here.  A final finite-sample correction restores
    the fine realization's exact mean and standard deviation, so changing the
    spectrum cannot silently change overall SNR.  The zero endpoint deliberately
    calls the historical helper unchanged for byte-level replay.
    """
    fraction = _unit_interval(thermal_fraction, "thermal_fraction")
    fine = np.asarray(fine, dtype=np.float32)
    sigma = max(float(sigma), 0.0)
    if fraction <= 0.0:
        return _grain_preserve_moments(
            fine, sigma=sigma, anisotropy=anisotropy)
    if fraction >= 1.0 or sigma <= 0.0 or fine.size < 2:
        return fine.copy()

    correlated = _grain_preserve_moments(
        np.asarray(correlated_source, dtype=np.float32),
        sigma=sigma, anisotropy=anisotropy)
    target_mean = float(fine.mean())
    target_sd = float(fine.std())
    corr_mean = float(correlated.mean())
    corr_sd = float(correlated.std())
    if target_sd <= 1e-12 or corr_sd <= 1e-12:
        return fine.copy()

    # Standardize before the variance-share mix. The two fields come from
    # independent draws; normalize once more afterward to remove their tiny
    # finite-volume covariance while preserving the requested spectrum.
    thermal_z = (fine - target_mean) / target_sd
    correlated_z = (correlated - corr_mean) / corr_sd
    mixed = (np.sqrt(fraction) * thermal_z
             + np.sqrt(1.0 - fraction) * correlated_z)
    mixed_mean = float(mixed.mean())
    mixed_sd = float(mixed.std())
    if mixed_sd <= 1e-12:
        return fine.copy()
    mixed = (mixed - mixed_mean) * (target_sd / mixed_sd) + target_mean
    return np.asarray(mixed, dtype=np.float32)


def _uni_tissue_noise(gen, matched, sigma, *, gfactor=0.12,
                      grain_sigma=0.0, grain_anisotropy=1.0,
                      thermal_fraction=0.0):
    """Cheap first-order approximation to signal-dependent UNI tissue noise.

    For encoded UNI ``u=q+0.5``, the analytic numerator contains
    ``sqrt(1 - 4*q**2)``. A small floor represents reconstruction/physiologic noise,
    while an optional smooth multiplier approximates receive/g-factor variation.
    """
    sigma = max(float(sigma), 0.0)
    if sigma <= 0.0:
        return np.zeros(np.asarray(matched).shape, dtype=np.float32)
    u = np.clip(np.asarray(matched, dtype=np.float32), 0.0, 1.0)
    profile = np.sqrt(np.clip(1.0 - 4.0 * (u - 0.5) ** 2, 0.0, 1.0))
    profile = 0.15 + 0.85 * profile
    if gfactor > 0.0 and u.size >= 64:
        corr = ndi.gaussian_filter(
            gen.standard_normal(u.shape).astype(np.float32),
            max(1.0, 0.12 * float(min(u.shape))))
        corr /= float(corr.std()) or 1.0
        profile *= np.exp(np.clip(float(gfactor) * corr, -0.5, 0.5))
    fraction = _unit_interval(thermal_fraction, "tissue_noise_thermal_fraction")
    white = gen.normal(0.0, sigma, u.shape).astype(np.float32)
    if 0.0 < fraction < 1.0 and float(grain_sigma) > 0.0:
        correlated_source = gen.normal(0.0, sigma, u.shape).astype(np.float32)
    else:
        # The helper ignores this endpoint when the mixture is neutral or pure
        # thermal. Avoiding a second draw preserves the historical RNG contract.
        correlated_source = white
    white = _thermal_grain_mixture(
        white, correlated_source, sigma=grain_sigma,
        anisotropy=grain_anisotropy, thermal_fraction=fraction)
    return white * profile.astype(np.float32)


# Absolute bounds are valid here because every input to this renderer has already
# been robustly normalized to [0, 1].  This is a cap on the *pre-blend noise
# field's* local standard deviation, not a target for any displayed/QA RSD.
_POSTERIOR_NOISE_FIELD_SD_CEILING = 0.090
_POSTERIOR_NOISE_SD_PER_SNR_LOSS = 0.120


def _posterior_noise_increment_scale(
        base_noise, posterior_increment, posterior_gain_delta, membership,
        snr_ratio, *, sd_ceiling=_POSTERIOR_NOISE_FIELD_SD_CEILING,
        sd_per_snr_loss=_POSTERIOR_NOISE_SD_PER_SNR_LOSS):
    """Bound the interaction between site noise and posterior SNR loss.

    The posterior package adds two correlated terms to the ordinary site noise:
    a gain on its existing realization and a separately seeded local realization.
    Applying both terms at full strength makes a grainy site multiplied by a low
    ``snr_ratio`` far noisier than either reference family, while an ordinary
    UNIT1 site can remain too smooth.  Measure their actual covariance in the
    smooth scanner-field core and solve

    ``var(base + alpha * increment) == target**2``

    for one deterministic scalar ``alpha``.  The target starts at the site's
    measured base-noise SD, rises with SNR loss, and is capped absolutely.  Thus a
    smooth site receives the missing posterior texture while a grainy site uses
    most of the available headroom before the posterior package is applied.

    ``posterior_gain_delta`` is the existing smooth scanner response
    (``noise_gain - 1``).  It, together with the soft acquisition membership,
    defines the QA core; the exact supervision-mask boundary is never used.
    This function consumes no RNG, so serialized-spec replay remains exact.
    """
    ratio = float(snr_ratio)
    if not np.isfinite(ratio) or not 0.0 < ratio <= 1.0:
        raise ValueError("posterior noise snr_ratio must be finite and in (0, 1]")
    if ratio >= 1.0 - 1e-12:
        return 0.0
    ceiling = float(sd_ceiling)
    per_loss = float(sd_per_snr_loss)
    if (not np.isfinite(ceiling) or ceiling <= 0.0
            or not np.isfinite(per_loss) or per_loss < 0.0):
        raise ValueError("posterior noise SD bounds must be finite and nonnegative")

    base = np.asarray(base_noise, dtype=np.float32)
    increment = np.asarray(posterior_increment, dtype=np.float32)
    gain_delta = np.asarray(posterior_gain_delta, dtype=np.float32)
    soft_membership = np.asarray(membership, dtype=np.float32)
    if not (base.shape == increment.shape == gain_delta.shape
            == soft_membership.shape):
        raise ValueError("posterior noise fields and membership must have identical shapes")

    multiplier = 1.0 / ratio - 1.0
    if multiplier <= 1e-12:
        return 0.0
    response = np.clip(gain_delta / multiplier, 0.0, 1.0)
    # A field-core threshold is stable across anatomy and excludes the long
    # ellipsoid tail.  Soft acquisition membership prevents external air from
    # setting a tissue-noise target without introducing a hard label contour.
    core = response >= 0.35
    if int(np.count_nonzero(core)) < 64:
        core = response > 0.05
    if int(np.count_nonzero(core)) < 16:
        return 0.0
    weights = (response[core].astype(np.float64)
               * np.clip(soft_membership[core], 0.0, 1.0).astype(np.float64))
    weight_sum = float(weights.sum())
    if weight_sum <= 1e-8:
        return 0.0
    b = base[core].astype(np.float64)
    d = increment[core].astype(np.float64)
    mean_b = float(np.dot(weights, b) / weight_sum)
    mean_d = float(np.dot(weights, d) / weight_sum)
    b -= mean_b
    d -= mean_d
    var_b = max(0.0, float(np.dot(weights, b * b) / weight_sum))
    var_d = max(0.0, float(np.dot(weights, d * d) / weight_sum))
    covariance = float(np.dot(weights, b * d) / weight_sum)
    if var_d <= 1e-14:
        return 0.0

    base_sd = float(np.sqrt(var_b))
    target_sd = min(ceiling, base_sd + per_loss * (1.0 - ratio))
    if target_sd <= base_sd + 1e-12:
        return 0.0
    # Positive solution of var_b + 2*a*cov + a^2*var_d = target^2.
    discriminant = covariance * covariance + var_d * (
        target_sd * target_sd - var_b)
    alpha = (-covariance + np.sqrt(max(discriminant, 0.0))) / var_d
    # The solved variance target is the primary bound; this numerical guard only
    # prevents a nearly-zero increment from receiving an uninformative huge gain.
    return float(np.clip(alpha, 0.0, 8.0))


def _ratio_bg_field_3d(gen, shape, *, amp=0.5, sigma=0.20, inv2_scale=1.0,
                       effective_coils=32.0):
    """Stored UNI air from the signed complex multi-coil MP2RAGE combination.

    Kept under the historical helper name for API compatibility.  ``sigma`` is
    scale-invariant in the normalized ratio; see
    :func:`augmentations.protocols.mp2rage._complex_uni_air_field` for the derivation.
    """
    del sigma
    return _mp2rage._complex_uni_air_field(
        gen, shape, center=amp, effective_coils=effective_coils,
        inv2_scale=inv2_scale)


def _mp2rage_acquisition_support(
        inb, posterior_fossa, voxel_sizes, *, acquisition_guard_coarsen_mm,
        acquisition_guard_dilation_mm, acquisition_guard_feather_mm,
        acquisition_transition_width_mm):
    """Resolve the coarse acquisition support and versioned posterior overrides."""
    posterior_spec = None
    render_inb = inb
    active_guard_dilation = acquisition_guard_dilation_mm
    active_guard_feather = acquisition_guard_feather_mm
    active_transition_width = acquisition_transition_width_mm
    acquisition_guard_enabled = any(value > 0.0 for value in (
        acquisition_guard_coarsen_mm, active_guard_dilation,
        active_guard_feather, active_transition_width))
    if posterior_fossa is not None:
        from augmentations.curricula.mp2rage_posterior import (
            posterior_acquisition_support,
            validate_posterior_fossa_spec,
        )
        posterior_spec = validate_posterior_fossa_spec(posterior_fossa)
        # A versioned posterior spec is authoritative, including the all-zero
        # migrated-v6 endpoint. Top-level policy-12 defaults must not turn a v6
        # replay into the new composition merely because its cfg bears a marker.
        active_guard_dilation = float(
            posterior_spec["acquisition_guard_dilation_mm"])
        active_guard_feather = float(
            posterior_spec["acquisition_guard_feather_mm"])
        active_transition_width = float(
            posterior_spec["acquisition_transition_width_mm"])
        acquisition_guard_enabled = any(value > 0.0 for value in (
            active_guard_dilation, active_guard_feather,
            active_transition_width))
        # The exact target remains the returned label and supplies scalar tissue
        # statistics.  Visible acquisition transitions use a heavily coarsened,
        # smoothly displaced proxy instead: enough intracranial geometry to avoid
        # brainifying scalp, without preserving sulcal/folial answer detail.
        render_inb = posterior_acquisition_support(
            inb, posterior_spec, voxel_sizes=voxel_sizes)
        if int(render_inb.sum()) < 100 and not acquisition_guard_enabled:
            # Frozen v6 and older specs retain their historical fallback. A
            # policy-12 guard must never replace a failed coarse proxy with the
            # exact supervision contour; the small proxy remains preferable.
            render_inb = inb
    elif acquisition_guard_enabled:
        # Ordinary/lower-feature/superset policy-12 draws need the same absence
        # of fine answer-shaped gating. Their mild global proxy is a physical
        # low-pass of the label; posterior specs replace it with their regional,
        # displaced acquisition support above.
        render_inb = _coarse_acquisition_support(
            inb, acquisition_guard_coarsen_mm, sampling=voxel_sizes)
    return (render_inb, posterior_spec, acquisition_guard_enabled, active_guard_dilation,
            active_guard_feather, active_transition_width)


def _mp2rage_acquisition_weights(
        img, inb, render_inb, *, posterior_spec, boundary_width,
        voxel_sizes, acquisition_guard_enabled, active_guard_dilation, active_guard_feather,
        active_transition_width):
    """Feather brain/guard supports and the posterior low-signal floor."""
    transition_width = float(boundary_width)
    acquisition_guard_w = None
    if acquisition_guard_enabled:
        requested_transition = active_transition_width
        if requested_transition > 0.0:
            transition_width = requested_transition
        guard_feather = active_guard_feather
        if guard_feather <= 0.0:
            # A partially specified non-neutral guard remains continuous. The
            # explicit all-zero migrated-v6 endpoint never enters this branch.
            guard_feather = max(transition_width, 0.5)
        # EDT is the dominant cost at full 224^3 resolution. Both weights derive
        # from the same support, so compute its inside/outside distances once.
        signed_render = _signed_distance(render_inb, sampling=voxel_sizes)
        acquisition_guard_w = _soft_membership_from_signed(
            signed_render, guard_feather, dilation=active_guard_dilation)
        brain_w = _soft_membership_from_signed(
            signed_render, transition_width)
        del signed_render
    else:
        brain_w = _soft_membership(
            render_inb, transition_width, sampling=voxel_sizes)
    # A contracted acquisition proxy describes where reliable brain-like UNI signal
    # remains; it must not redefine labelled cerebellum as scalp.  Keep the semantic
    # intracranial exclusion separate, and softly route only the omitted band toward
    # the ratio-noise floor below.  Matching that band to its surroundings removes an
    # answer-shaped edge at the true target contour.
    semantic_inb = render_inb
    failed_intracranial_w = None
    if posterior_spec is not None:
        # Labelled tissue must never be reclassified as bright scalp merely because
        # the coarse acquisition proxy contracted. Apply the signal floor to the
        # omitted internal band and let a submillimetre Gaussian feather cross the
        # true boundary naturally. Existing exterior CSF/skull/scalp rendering is
        # otherwise untouched.
        from augmentations.curricula.mp2rage_posterior import _ellipsoid_weight
        spacing = np.asarray(
            voxel_sizes if voxel_sizes is not None else (1.0,) * img.ndim,
            dtype=np.float64)
        if spacing.size != img.ndim or np.any(spacing <= 0.0):
            spacing = np.ones(img.ndim, dtype=np.float64)
        floor_mix = float(posterior_spec["render_boundary_floor_mix"])
        if acquisition_guard_enabled:
            # The directed low-signal shell is defined only by two acquisition
            # supports: the coarse tissue transition and its expanded guard.
            # It therefore cannot reproduce fine supervision-mask detail.
            omitted_proxy_w = np.clip(
                acquisition_guard_w - brain_w, 0.0, 1.0).astype(np.float32)
            if floor_mix > 1e-8 and np.any(omitted_proxy_w > 1e-6):
                posterior_field = _ellipsoid_weight(
                    img.shape, render_inb,
                    posterior_spec["boundary_center_frac_ras"],
                    posterior_spec["boundary_field_fwhm_mm_ras"], spacing)
                failed_intracranial_w = omitted_proxy_w
                failed_intracranial_w *= floor_mix * np.sqrt(posterior_field)
                np.clip(failed_intracranial_w, 0.0, 1.0,
                        out=failed_intracranial_w)
                del posterior_field
            del omitted_proxy_w
        else:
            # Exact historical v6 behavior. The policy-12 path above never uses
            # the target contour as an acquisition support.
            omitted_label = inb & (~render_inb)
            semantic_inb = inb
        if (not acquisition_guard_enabled and floor_mix > 1e-8
                and omitted_label.any()):
            posterior_field = _ellipsoid_weight(
                img.shape, render_inb,
                posterior_spec["boundary_center_frac_ras"],
                posterior_spec["boundary_field_fwhm_mm_ras"], spacing)
            failed_intracranial_w = ndi.gaussian_filter(
                omitted_label.astype(np.float32),
                sigma=np.maximum(0.85 / spacing, 0.20), mode="nearest")
            failed_intracranial_w *= floor_mix * np.sqrt(posterior_field)
            np.clip(failed_intracranial_w, 0.0, 1.0,
                    out=failed_intracranial_w)
            del posterior_field
        if not acquisition_guard_enabled:
            del omitted_label
    return (brain_w, acquisition_guard_w, semantic_inb, failed_intracranial_w)


def _mp2rage_tissue(
        img, out, inb, *, render_inb, brain_w, curve_g, tissue_g, strength, extra,
        compatibility, brain_on, tissue_noise_on, tissue_noise, noise_seed, noise_gfactor,
        tissue_noise_grain_sg, tissue_noise_aniso, tissue_noise_thermal_fraction, ref_spread,
        ref_mix, fold_prob, ref_logit_gain, lut_xs, lut_ys, lut_per_region, detail_gain,
        detail_sigma, detail_edge_k, posterior_spec, voxel_sizes):
    """Map tissue to UNI, add acquisition noise, and restore bounded detail."""
    tissue_noise_field = None
    posterior_noise_gain = None
    noise_sigma = (float(tissue_noise) * (1.0 + 2.0 * extra)
                   if bool(tissue_noise_on) else 0.0)
    if bool(brain_on) and int(inb.sum()) >= 50:                 # (1) brain -> UNI distribution
        bv = img[inb]
        use_explicit_lut = (lut_xs is not None and lut_ys is not None)
        if use_explicit_lut:
            xs = np.asarray(lut_xs, dtype=np.float64)
            ys = np.asarray(lut_ys, dtype=np.float64)
            use_explicit_lut = bool(
                xs.ndim == ys.ndim == 1 and xs.size == ys.size and xs.size >= 2
                and np.isfinite(xs).all() and np.isfinite(ys).all()
                and np.all(np.diff(xs) > 0.0))
        if use_explicit_lut:
            if bool(lut_per_region):
                lut_region = inb
            else:
                positive = img[img > 0]
                scale = float(np.percentile(positive, 99.0)) if positive.size else 1.0
                lut_region = img > max(0.02, 0.02 * scale)
            if np.allclose(xs, ys, rtol=0.0, atol=1e-12):
                matched_full = img.copy()
            else:
                lo, hi = np.percentile(img[lut_region], [0.5, 99.5])
                norm = np.clip((img - float(lo)) / max(float(hi - lo), 1e-6), 0.0, 1.0)
                matched_full = np.interp(norm, xs, ys).astype(np.float32)
            src_q = ref_q = None
        else:
            ref_q = _uni_ref_curve(
                curve_g, ref_spread=ref_spread, ref_mix=ref_mix,
                fold_prob=fold_prob)
            if abs(ref_logit_gain - 1.0) > 1e-8:
                # A monotone display/site-style contrast temperature: values below
                # the stored-UNI midpoint move darker and values above it move
                # brighter.  Unlike a non-monotone LUT fold this preserves tissue
                # rank and therefore remains a plausible MP2RAGE/UNIT1 family.
                eps = np.float32(1e-5)
                q = np.clip(ref_q.astype(np.float32), eps, 1.0 - eps)
                logits = np.log(q / (1.0 - q)) * ref_logit_gain
                ref_q = (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)
            src_q = np.percentile(bv, _UNI_REF_P)
            matched_full = None
        if compatibility:
            # Explicit compatibility mode: preserve the historical draw count and
            # exact hard-mask arithmetic for old renders/tests.
            matched = (matched_full[inb] if use_explicit_lut
                       else np.interp(bv, src_q, ref_q).astype(np.float32))
            if bool(tissue_noise_on) and noise_seed is None:
                matched += tissue_g.normal(0.0, noise_sigma, matched.shape).astype(np.float32)
            elif bool(tissue_noise_on):
                # An explicit lower-level seed owns a full-FOV noise field before
                # masking/graining. Keep the
                # historical brain-vector draw above when no seed is supplied.
                lower_noise = tissue_g.normal(
                    0.0, noise_sigma, img.shape).astype(np.float32)
                lower_noise = _grain_preserve_moments(
                    lower_noise, sigma=tissue_noise_grain_sg,
                    anisotropy=tissue_noise_aniso)
                matched += lower_noise[inb]
            out[inb] = (1.0 - strength) * bv + strength * matched
        else:
            # The target annotation may supply tissue statistics, but a scanner
            # cannot make its noise/remap stop exactly at that annotation.  Build a
            # full-volume realization and blend it through a signed-distance band.
            matched = (matched_full if use_explicit_lut
                       else _quantile_map(img, src_q, ref_q))
            if posterior_spec is not None:
                from augmentations.curricula.mp2rage_posterior import apply_posterior_fossa_signal
                matched, posterior_noise_gain = apply_posterior_fossa_signal(
                    matched, inb, posterior_spec, voxel_sizes=voxel_sizes)
            tissue_noise_field = _uni_tissue_noise(
                tissue_g, matched, noise_sigma, gfactor=noise_gfactor,
                grain_sigma=tissue_noise_grain_sg,
                grain_anisotropy=tissue_noise_aniso,
                thermal_fraction=tissue_noise_thermal_fraction)
            if posterior_noise_gain is not None:
                from augmentations.curricula.mp2rage_posterior import correlated_posterior_noise
                # Compose the site's ordinary tissue noise and the posterior
                # increment first, then bound their *joint* local variance.  A
                # simple multiplication makes dark-PF x grainy-site draws explode,
                # while lowering snr_ratio alone still leaves ordinary UNIT1 too
                # smooth.  Scaling only the increment preserves the global site
                # phenotype and the independently seeded posterior realization.
                posterior_noise_gain -= 1.0
                posterior_increment = correlated_posterior_noise(
                    img.shape, inb, posterior_spec, noise_sigma,
                    voxel_sizes=voxel_sizes)
                posterior_increment += tissue_noise_field * posterior_noise_gain
                posterior_scale = _posterior_noise_increment_scale(
                    tissue_noise_field, posterior_increment,
                    posterior_noise_gain, brain_w,
                    posterior_spec["snr_ratio"])
                posterior_increment *= posterior_scale
                tissue_noise_field += posterior_increment
                del posterior_increment
            matched += tissue_noise_field
            rendered = (1.0 - strength) * img + strength * matched
            out = out + brain_w * (rendered - out)
        if use_explicit_lut and not bool(lut_per_region):
            outside = lut_region & (~render_inb)
            global_rendered = (1.0 - strength) * img + strength * matched_full
            if compatibility:
                out[outside] = global_rendered[outside]
            else:
                outside_w = outside.astype(np.float32) * (1.0 - brain_w)
                out = out + outside_w * (global_rendered - out)
        if (not compatibility and float(detail_gain) != 0.0):
            # One edge-limited acquisition/detail stage replaces the generic
            # pre-remap blur.  Suppressing its strongest-gradient tail improves
            # fine tissue texture without ringing at the pial boundary.  Bound
            # the delta by local headroom so sharpening cannot manufacture exact
            # zero/one masses inside an otherwise continuously valued brain.
            base = np.clip(out, 0.0, 1.0).astype(np.float32)
            detail = base - ndi.gaussian_filter(base, max(float(detail_sigma), 1e-3))
            grad = np.sqrt(sum(gx * gx for gx in np.gradient(base))).astype(np.float32)
            edge_k = max(float(detail_edge_k), 1e-4)
            edge_w = 1.0 / (1.0 + (grad / edge_k) ** 4)
            gain = float(detail_gain) * strength * (1.0 + extra)
            delta = brain_w * gain * detail * edge_w
            delta = np.clip(delta, -0.90 * base, 0.90 * (1.0 - base))
            out = base + delta
    elif bool(tissue_noise_on) and int(inb.sum()) >= 50:
        # The lower-level noise primitive is independent of the brain/LUT stage.
        # Keep that endpoint reachable even when ``brain_on=False``.
        if compatibility:
            tissue_noise_field = tissue_g.normal(
                0.0, noise_sigma, img.shape).astype(np.float32)
            tissue_noise_field = _grain_preserve_moments(
                tissue_noise_field, sigma=tissue_noise_grain_sg,
                anisotropy=tissue_noise_aniso)
            out[inb] += strength * tissue_noise_field[inb]
        else:
            tissue_noise_field = _uni_tissue_noise(
                tissue_g, img, noise_sigma, gfactor=noise_gfactor,
                grain_sigma=tissue_noise_grain_sg,
                grain_anisotropy=tissue_noise_aniso,
                thermal_fraction=tissue_noise_thermal_fraction)
            out += brain_w * strength * tissue_noise_field
    return (out, tissue_noise_field)


def _mp2rage_air_support(
        source, img, out, *, semantic_inb, support_source, acquisition_guard_w,
        failed_intracranial_w, air_mode, bg_model, bg_thr_frac, extracranial_snr_lo,
        extracranial_snr_hi, bone_retention, voxel_sizes):
    """Classify ratio-noise air and coherent extracranial tissue."""
    head_w = scale = None
    # (2) WHERE is the ratio noise-dominated?  Filled head geometry says only where
    # anatomy can exist; it does not imply useful INV1/INV2 signal.  The default SNR
    # mixture sends no-signal bone/cavities toward midpoint noise while retaining a
    # thin coherent boundary.  ``head`` and ``threshold`` preserve the older hard
    # endpoints for replay and lower-level composition.
    mode = str(air_mode if air_mode is not None
               else ("threshold" if str(bg_model).lower() == "uniform" else "snr")).lower()
    snr_w = None
    snr_lo = snr_hi = None
    if mode == "snr":
        head, signal_fg = _image_head_components(support_source)
        if acquisition_guard_w is None:
            # Frozen v6/legacy spatial composition.
            head = head | semantic_inb
            outer_head = head & (~semantic_inb)
            head_w = None
            outer_w = None
        else:
            # A scanner sees continuous SNR and broad intracranial geometry, not
            # the supervision contour. The image-derived filled head owns the
            # external boundary; the coarse, displaced guard only prevents scalp
            # transfer from brainifying its low-signal intracranial shell.
            head_w = np.maximum(
                head.astype(np.float32), acquisition_guard_w).astype(np.float32)
            outer_w = np.clip(
                head_w * (1.0 - acquisition_guard_w), 0.0, 1.0).astype(np.float32)
            outer_head = outer_w > 1e-6
        positive = support_source[support_source > 0]
        scale = float(np.percentile(positive, 99.0)) if positive.size else 1.0
        if bg_thr_frac is None:
            snr_lo = float(extracranial_snr_lo) * max(scale, 1e-6)
            snr_hi = float(extracranial_snr_hi) * max(scale, 1e-6)
        else:
            snr_lo = max(0.0, float(bg_thr_frac)) * max(scale, 1e-6)
            snr_hi = 2.2 * snr_lo
        if snr_hi <= snr_lo + 1e-6:
            snr_hi = snr_lo + max(0.02 * max(scale, 1e-6), 1e-3)
        snr_w = _smoothstep01((support_source - snr_lo) / (snr_hi - snr_lo))
        coherent_signal = support_source >= snr_hi
        bone_threshold = snr_lo + 0.25 * (snr_hi - snr_lo)
        coherent_bone = (outer_head & (source < bone_threshold)
                         & ndi.binary_dilation(coherent_signal, iterations=1))
        bone_core = coherent_bone.astype(np.float32)
        bone_soft = ndi.gaussian_filter(bone_core, 0.55)
        bone_w = np.maximum(bone_core, 0.25 * bone_soft)
        ratio_inside = ((1.0 - snr_w)
                        * (1.0 - float(np.clip(bone_retention, 0.0, 1.0)) * bone_w))
        if acquisition_guard_w is None:
            air_w = np.where(~head, 1.0,
                             np.where(outer_head, ratio_inside, 0.0)).astype(np.float32)
            scalp_support = outer_head
        else:
            air_w = np.clip(
                (1.0 - head_w) + outer_w * ratio_inside,
                0.0, 1.0).astype(np.float32)
            scalp_support = outer_w
        air = air_w > 1e-6
        scalp = np.asarray(scalp_support) > 1e-6
    elif mode == "threshold":
        # Exact lower-level composition gates after preceding skull/brain/LUT
        # stages, not on the untouched source image.
        gate_image = out
        positive = gate_image[gate_image > 0]
        scale = float(np.percentile(positive, 99.0)) if positive.size else 1.0
        default_threshold = (0.10 * max(scale, 1e-6)
                             if str(bg_model).lower() == "uniform" else 0.22)
        threshold = (default_threshold if bg_thr_frac is None
                     else max(0.0, float(bg_thr_frac)) * max(scale, 1e-6))
        air = (~semantic_inb) & (gate_image < threshold)
        scalp = (~semantic_inb) & (~air)
        scalp_support = scalp
        air_w = air.astype(np.float32)
    else:
        head, signal_fg = _image_head_components(support_source)
        head = head | semantic_inb
        # Deep, no-signal cavities outside the annotated brain (e.g. sinuses) carry
        # complex ratio noise; a thin low-signal rim remains dark cortical bone.
        sampling = None
        if voxel_sizes is not None:
            candidate_sampling = tuple(float(v) for v in voxel_sizes)
            if len(candidate_sampling) == img.ndim:
                sampling = candidate_sampling
        depth = ndi.distance_transform_edt(head, sampling=sampling)
        depth_threshold = (2.0 if sampling is not None
                           else max(2.0, 0.015 * float(min(img.shape))))
        deep = depth > depth_threshold
        brain_guard = ndi.binary_dilation(semantic_inb, iterations=1)
        cavity_air = head & (~signal_fg) & deep & (~brain_guard)
        air = (~head) | cavity_air
        scalp = head & (~semantic_inb) & (~cavity_air)
        scalp_support = head & (~cavity_air)
        air_w = air.astype(np.float32)

    if failed_intracranial_w is not None:
        # Use the identical complex-ratio background realization generated below,
        # but only as a partial signal floor.  The omitted labelled band is neither
        # brightened as scalp nor replaced by a featureless constant.
        np.maximum(air_w, failed_intracranial_w, out=air_w)
        air = air_w > 1e-6
        if acquisition_guard_w is not None:
            # The directed omitted-tissue proxy belongs to the complex-ratio
            # floor, never to bright scalp. Both weights are coarse/continuous.
            scalp_support = (np.asarray(scalp_support, dtype=np.float32)
                             * (1.0 - failed_intracranial_w))
            scalp = scalp_support > 1e-6
    return (mode, air_w, air, scalp, scalp_support, snr_w, snr_lo, scale, head_w)


def _mp2rage_background(
        img, out, inb, *, air, air_w, air_g, padding_g, strength, extra,
        compatibility, mode, background_on, background_weight, bg_mean, bg_std, bg_model,
        bg_ratio_inv2_scale, bg_effective_coils, bg_grain_sg, bg_grain_aniso, bg_thermal_fraction,
        bg_fov_zero_prob, bg_fov_zero_frac, fov_padding_on, params):
    """Blend ratio-noise background and sample the final FOV crop."""
    w_bg = 0.0
    zero_mask = None
    if bool(background_on) and np.any(air_w > 0.0):
        w_bg = float(np.clip(
            min(strength, 1.0) if background_weight is None else background_weight,
            0.0, 1.0))
        resolved_bg_model = str(bg_model).lower()
        if resolved_bg_model in ("complex", "ratio"):
            # Real stored UNI air has values on BOTH sides of its midpoint.  The
            # effective-mode model matches the bundled 32-channel-coil reference.
            ratio_kwargs = dict(
                amp=float(np.clip(
                    float(bg_mean) * (1.0 + 0.8 * extra), 0.0, 1.0)),
                sigma=float(params.get("bg_ratio_sigma", 0.20))
                * (1.0 + 0.5 * extra),
                inv2_scale=float(
                    params.get("bg_inv2_scale", 1.0)
                    if bg_ratio_inv2_scale is None else bg_ratio_inv2_scale),
                effective_coils=float(params.get(
                    "bg_effective_coils", bg_effective_coils)))
            field = _ratio_bg_field_3d(air_g, img.shape, **ratio_kwargs)
            if (0.0 < bg_thermal_fraction < 1.0
                    and float(bg_grain_sg) > 0.0):
                correlated_source = _ratio_bg_field_3d(
                    air_g, img.shape, **ratio_kwargs)
            else:
                correlated_source = field
        elif resolved_bg_model == "uniform":                  # explicit legacy box endpoint
            field = air_g.uniform(0.0, float(bg_mean), img.shape).astype(np.float32)
        else:                                                  # legacy smooth gaussian (back-compat)
            field = np.clip(float(bg_mean) + air_g.normal(
                0.0, float(bg_std) * (0.4 + 0.6 * strength), img.shape), 0.0, 1.0)
        if resolved_bg_model in ("complex", "ratio"):
            field = _thermal_grain_mixture(
                field, correlated_source, sigma=bg_grain_sg,
                anisotropy=bg_grain_aniso,
                thermal_fraction=bg_thermal_fraction)
        else:
            # Legacy/explicit non-ratio endpoints retain their historical law.
            field = _grain_preserve_moments(
                field, sigma=bg_grain_sg, anisotropy=bg_grain_aniso)
        if compatibility and mode == "threshold" and float(bg_grain_sg) <= 0.0:
            # Frozen hard-mask mode remains byte-identical to the pre-superset path.
            out[air] = ((1.0 - w_bg) * out[air] + w_bg * field[air]).astype(np.float32)
        else:
            blend_w = np.clip(w_bg * air_w, 0.0, 1.0).astype(np.float32)
            out = out + blend_w * (field - out)
        zero_mask = (_mp2rage._edge_zero_mask(
            padding_g, img.shape, probability=float(bg_fov_zero_prob),
            # The mask is only a rejection guard here: a whole planar candidate is
            # discarded if it would crop labelled tissue, so no target-shaped
            # intensity boundary is ever rendered.
            fraction=float(bg_fov_zero_frac), protect=inb)
            if bool(fov_padding_on) else None)
    return (out, zero_mask, w_bg)


def _mp2rage_extracranial(
        out, source, scalp, *, scalp_support, brain_w, air_w, head_w, snr_w,
        snr_lo, scale, mode, strength, compatibility, extracranial_on, extracranial_strength,
        extracranial_noise_gain, scalp_compress, scalp_gamma, scalp_floor, tissue_noise_field,
        acquisition_guard_w):
    """Transfer scalp signal and fill any residual acquisition-noise annulus."""
    # (3) extracranial. `scalp_compress` < 1 dims it, but UNI fat is among the BRIGHTEST structures
    # in the image (brighter than WM), so values > 1 are legitimate and GAIN it — the formula below
    # already handles both directions.
    scalp_factor = 1.0 - (1.0 - float(scalp_compress)) * strength
    # A scalar zero is also the final scalp weight when the extracranial stage is
    # disabled. Keeping the weight explicit lets the policy-12 annulus accounting
    # below use one formula without allocating an unnecessary zero volume.
    scalp_w = 0.0
    if compatibility:
        if bool(extracranial_on):
            out[scalp] = out[scalp] * scalp_factor
    elif bool(extracranial_on):
        if mode == "snr" and snr_w is not None:
            # A monotone confidence transfer preserves anatomical texture while
            # moving coherent soft tissue/fat onto the bright UNIT1 range.
            denom = max(float(scale) - float(snr_lo), 1e-6)
            norm = np.clip((source - float(snr_lo)) / denom, 0.0, 1.0)
            gamma = max(float(scalp_gamma), 1e-3)
            floor = float(np.clip(scalp_floor, 0.0, 1.0))
            scalp_target = floor + (1.0 - floor) * norm ** gamma
            uni_w = np.clip(scalp_support.astype(np.float32)
                            * (1.0 - brain_w) * snr_w, 0.0, 1.0)
            xstrength = max(0.0, float(extracranial_strength)) * strength
            out = out + xstrength * uni_w * (scalp_target - out)
        # Select bright soft tissue/fat smoothly; do not suppress or amplify the
        # dark cortical-bone band with a global extracranial multiplier.
        fat_w = _smoothstep01((source - 0.12) / 0.30)
        scalp_w = np.clip(scalp_support.astype(np.float32)
                          * (1.0 - brain_w) * fat_w, 0.0, 1.0)
        out = out * (1.0 + scalp_w * (scalp_factor - 1.0))
        if tissue_noise_field is not None:
            # Soft tissue is part of the same acquisition; do not leave it on the
            # source T1 noise law while changing only annotated brain tissue.
            out += extracranial_noise_gain * strength * scalp_w * tissue_noise_field
    if (acquisition_guard_w is not None and mode == "snr"
            and tissue_noise_field is not None):
        # The continuous guard can leave a narrow mass that belongs to the
        # image-derived head but to none of brain, ratio-background, or bright
        # scalp. Leaving it on the smooth source law creates a low-noise annulus
        # around the coarse support. Fill only that residual with the acquisition's
        # existing tissue-noise realization: no exact target mask, no extra RNG,
        # and no double-noising where air/scalp already owns the voxel.
        annulus_w = np.clip(
            head_w - brain_w - air_w - scalp_w, 0.0, 1.0).astype(np.float32)
        annulus_mass = float(np.sum(annulus_w, dtype=np.float64))
        if annulus_mass > 1e-6:
            annulus_mean = float(np.sum(
                annulus_w * tissue_noise_field, dtype=np.float64) / annulus_mass)
            # Center under the same soft weights so the added field has exactly
            # zero weighted mean and cannot form a deterministic bright/dark halo.
            tissue_noise_field -= annulus_mean
            out += strength * annulus_w * tissue_noise_field
        del annulus_w
    return out


@register("mp2rage_3d", kind="intensity", severity_range=(0.0, MP2RAGE_3D_MAX), dims="3d",
          label_preserving=True, has_detector=False, extra_parameters=('bg_inv2_scale', 'bg_ratio_sigma'))
def mp2rage_3d(arr, *, severity=1.0, rng=None, mask=None, master_t=None,
               bg_mean=0.5, bg_std=0.11, tissue_noise=0.02, scalp_compress=0.85,
               ref_spread=0.0, ref_mix=None, ref_logit_gain=1.0,
               fold_prob=0.0, air_mode=None,
               boundary_width=2.0, voxel_sizes=None, background_weight=None,
               noise_gfactor=0.12, bg_effective_coils=32.0,
               detail_gain=0.50, detail_sigma=0.70, detail_edge_k=0.12,
               bg_fov_zero_prob=0.0, bg_fov_zero_frac=0.25,
               bg_grain_sg=0.0, bg_grain_aniso=1.0,
               bg_thermal_fraction=0.0,
               tissue_noise_grain_sg=0.0, tissue_noise_aniso=1.0,
               tissue_noise_thermal_fraction=0.0,
               acquisition_guard_coarsen_mm=0.0,
               acquisition_guard_dilation_mm=0.0,
               acquisition_guard_feather_mm=0.0,
               acquisition_transition_width_mm=0.0,
               extracranial_strength=1.0, extracranial_snr_lo=0.07,
               extracranial_snr_hi=0.22, bone_retention=0.85,
               scalp_floor=0.46, scalp_gamma=0.75,
               extracranial_noise_gain=0.5,
               brain_on=True, background_on=True, extracranial_on=True,
               tissue_noise_on=True, fov_padding_on=True,
               brain_contrast=None, skull_contrast=None,
               lut_xs=None, lut_ys=None, lut_per_region=True,
               bg_thr_frac=None, bg_model="complex", bg_amp=None,
               bg_ratio_inv2_scale=None, bg_seed=None,
               noise_sigma=None, noise_grain_sg=None, noise_aniso=None,
               noise_thermal_fraction=None, noise_seed=None,
               skull_contrast_on=True, bg_on=None,
               noise_on=None, posterior_fossa=None, **params):
    """Realistic MP2RAGE/UNI look:
      (1) in-brain quantile-matched to a real UNI distribution,
      (2) the symmetric complex multi-coil MID-GRAY FOV background (the defining UNI
          'tell') — see ``_ratio_bg_field_3d``; pass ``bg_model='gaussian'`` for the
          legacy smooth gaussian background,
      (3) extracranial soft tissue kept visible (UNI is not skull-stripped),
    blended by ``severity`` (master t): 0 = original T1, ``MP2RAGE_3D_MARK`` (1.0) =
    full UNI look.

    SUPERSET ZONE. ``t`` is no longer clamped at 1.0 — it now runs to
    ``MP2RAGE_3D_MAX`` (1.3): the in-brain blend weight passes 1.0 so the
    transfer overshoots the matched image, in-tissue grain and background
    amplitude/spread rise, and the scalp compresses further. The BACKGROUND blend
    weight is deliberately capped at 1.0 — extrapolating there would drive air
    negative rather than making it more UNI-like.

    ``ref_mix`` / ``ref_spread`` draw a coherent curve between two measured scans;
    ``fold_prob`` is an explicit unusual-protocol superset knob and should stay zero
    for normal UNIT1. ``background_weight`` can decouple acquisition background from
    the tissue-preview slider; a labelled MP2RAGE acquisition should pass 1.0.

    ``air_mode='snr'`` is the realistic default.  A UNI value is a normalized ratio,
    so genuinely low-SNR bone/cavity voxels approach the same mid-gray complex-noise
    law as external air; a narrow image-derived boundary retains coherent dark bone.
    The older hard ``'head'`` and ``'threshold'`` classifications remain explicit
    superset/back-compat endpoints.  Optional LUT/contrast, uniform-background,
    background-grain, tissue-grain, and per-stage switches expose the corresponding
    lower-level MP2RAGE features without changing the neutral canonical path.

    Set ``boundary_width=0`` only for legacy hard-mask/RNG compatibility. The default
    uses physical EDT sampling when ``voxel_sizes`` is supplied and independent random
    substreams for curve, tissue, air, and reconstructed support.

    ``posterior_fossa`` is an optional versioned acquisition spec sampled by
    :mod:`augmentations.curricula.mp2rage_posterior`. It adds a broad scanner-coordinate
    low-inversion/low-SNR package and an extracerebral tentorium hard negative;
    it is intentionally separate from the lower-level one-family override API.
    """
    source = np.asarray(arr, dtype=np.float32)
    if bg_amp is not None:
        bg_mean = float(bg_amp)
    if noise_sigma is not None:
        tissue_noise = float(noise_sigma)
    if noise_grain_sg is not None:
        tissue_noise_grain_sg = float(noise_grain_sg)
    if noise_aniso is not None:
        tissue_noise_aniso = float(noise_aniso)
    if noise_thermal_fraction is not None:
        tissue_noise_thermal_fraction = float(noise_thermal_fraction)
    if bg_on is not None:
        background_on = bool(bg_on)
    if noise_on is not None:
        tissue_noise_on = bool(noise_on)
    ref_logit_gain = float(ref_logit_gain)
    extracranial_noise_gain = float(extracranial_noise_gain)
    bg_thermal_fraction = _unit_interval(
        bg_thermal_fraction, "bg_thermal_fraction")
    tissue_noise_thermal_fraction = _unit_interval(
        tissue_noise_thermal_fraction, "tissue_noise_thermal_fraction")
    acquisition_guard_coarsen_mm = float(acquisition_guard_coarsen_mm)
    acquisition_guard_dilation_mm = float(acquisition_guard_dilation_mm)
    acquisition_guard_feather_mm = float(acquisition_guard_feather_mm)
    acquisition_transition_width_mm = float(acquisition_transition_width_mm)
    for guard_name, guard_value in (
            ("acquisition_guard_coarsen_mm", acquisition_guard_coarsen_mm),
            ("acquisition_guard_dilation_mm", acquisition_guard_dilation_mm),
            ("acquisition_guard_feather_mm", acquisition_guard_feather_mm),
            ("acquisition_transition_width_mm", acquisition_transition_width_mm)):
        if not np.isfinite(guard_value) or guard_value < 0.0:
            raise ValueError(f"{guard_name} must be finite and >= 0")
    if not np.isfinite(ref_logit_gain) or ref_logit_gain <= 0.0:
        raise ValueError("ref_logit_gain must be finite and > 0")
    if (not np.isfinite(extracranial_noise_gain)
            or not 0.0 <= extracranial_noise_gain <= 2.0):
        raise ValueError("extracranial_noise_gain must be finite and in [0, 2]")
    img = source.copy()
    inb = _brain_mask(img, mask)
    (render_inb, posterior_spec, acquisition_guard_enabled, active_guard_dilation,
        active_guard_feather, active_transition_width) = _mp2rage_acquisition_support(inb,
        posterior_fossa, voxel_sizes, acquisition_guard_coarsen_mm=acquisition_guard_coarsen_mm,
        acquisition_guard_dilation_mm=acquisition_guard_dilation_mm,
        acquisition_guard_feather_mm=acquisition_guard_feather_mm,
        acquisition_transition_width_mm=acquisition_transition_width_mm)
    # Optional lower-level contrast stages.  ``None`` is deliberately neutral;
    # explicit factors reproduce the standalone feature family in n-D.
    if bool(skull_contrast_on) and skull_contrast is not None:
        img = apply("skull_contrast", img.copy(), severity=0.0, mask=render_inb,
                    contrast_factor=float(skull_contrast))
    if bool(brain_on) and brain_contrast is not None:
        img = apply("brain_contrast", img.copy(), severity=0.0, mask=render_inb,
                    contrast_factor=float(brain_contrast))
    g = _rng(rng)
    t = float(master_t) if master_t is not None else float(severity)
    strength = float(np.clip(t, 0.0, MP2RAGE_3D_MAX))
    extra = max(0.0, strength - MP2RAGE_3D_MARK)               # 0 inside the realistic zone
    out = img.copy()

    compatibility = float(boundary_width) <= 0.0
    if posterior_spec is not None and compatibility:
        raise ValueError(
            "posterior_fossa requires boundary_width > 0; the legacy hard-boundary "
            "compatibility path cannot safely compose the extra seeded stages")
    if compatibility and acquisition_guard_enabled:
        raise ValueError(
            "continuous acquisition guards require boundary_width > 0; the explicit "
            "hard-mask compatibility path is frozen")
    if compatibility and (bg_thermal_fraction > 0.0
                          or tissue_noise_thermal_fraction > 0.0):
        raise ValueError(
            "thermal/grain mixtures require boundary_width > 0; the explicit "
            "hard-mask compatibility path is frozen")
    if compatibility:
        curve_g = tissue_g = air_g = padding_g = g
    else:
        seeds = g.integers(0, np.iinfo(np.int64).max, size=4, dtype=np.int64)
        curve_g, tissue_g, air_g, padding_g = [np.random.default_rng(int(s)) for s in seeds]
    if noise_seed is not None:
        tissue_g = np.random.default_rng(int(noise_seed))
    if bg_seed is not None:
        air_g = np.random.default_rng(int(bg_seed))
    (brain_w, acquisition_guard_w, semantic_inb,
     failed_intracranial_w) = _mp2rage_acquisition_weights(
        img, inb, render_inb, posterior_spec=posterior_spec,
        boundary_width=boundary_width, voxel_sizes=voxel_sizes,
        acquisition_guard_enabled=acquisition_guard_enabled,
        active_guard_dilation=active_guard_dilation, active_guard_feather=active_guard_feather,
        active_transition_width=active_transition_width)
    (out, tissue_noise_field) = _mp2rage_tissue(img, out, inb, render_inb=render_inb,
        brain_w=brain_w, curve_g=curve_g, tissue_g=tissue_g, strength=strength, extra=extra,
        compatibility=compatibility, brain_on=brain_on, tissue_noise_on=tissue_noise_on,
        tissue_noise=tissue_noise, noise_seed=noise_seed, noise_gfactor=noise_gfactor,
        tissue_noise_grain_sg=tissue_noise_grain_sg, tissue_noise_aniso=tissue_noise_aniso,
        tissue_noise_thermal_fraction=tissue_noise_thermal_fraction, ref_spread=ref_spread,
        ref_mix=ref_mix, fold_prob=fold_prob, ref_logit_gain=ref_logit_gain, lut_xs=lut_xs,
        lut_ys=lut_ys, lut_per_region=lut_per_region, detail_gain=detail_gain,
        detail_sigma=detail_sigma, detail_edge_k=detail_edge_k, posterior_spec=posterior_spec,
        voxel_sizes=voxel_sizes)
    if posterior_spec is not None:
        # The hard-negative signal must exist before SNR/air classification or the
        # ratio-background stage would immediately paint it over as no-signal air.
        # Its delta receives the same scanner-axis PSF as the tissue signal; the
        # tissue path above has already been filtered, so filter only this late delta.
        from augmentations.curricula.mp2rage_posterior import apply_tentorium_mimic
        mimic = apply_tentorium_mimic(
            out, inb, posterior_spec, voxel_sizes=voxel_sizes)
        mimic_delta = mimic - out
        spacing = np.asarray(
            voxel_sizes if voxel_sizes is not None else (1.0,) * out.ndim,
            dtype=np.float64)
        if spacing.size != out.ndim or np.any(spacing <= 0.0):
            spacing = np.ones(out.ndim, dtype=np.float64)
        mimic_sigma = (np.asarray(posterior_spec["psf_fwhm_mm_ras"], dtype=np.float64)
                       / 2.354820045 / spacing)
        mimic_delta = ndi.gaussian_filter(
            mimic_delta.astype(np.float32), np.maximum(mimic_sigma, 0.0)).astype(np.float32)
        mimic_weight = float(np.clip(strength, 0.0, 1.0))
        out = out + mimic_weight * mimic_delta
        support_source = out
    else:
        support_source = source

    (mode, air_w, air, scalp, scalp_support, snr_w, snr_lo,
     scale, head_w) = _mp2rage_air_support(
        source, img, out, semantic_inb=semantic_inb,
        support_source=support_source, acquisition_guard_w=acquisition_guard_w,
        failed_intracranial_w=failed_intracranial_w, air_mode=air_mode, bg_model=bg_model,
        bg_thr_frac=bg_thr_frac, extracranial_snr_lo=extracranial_snr_lo,
        extracranial_snr_hi=extracranial_snr_hi, bone_retention=bone_retention,
        voxel_sizes=voxel_sizes)
    (out, zero_mask, w_bg) = _mp2rage_background(img, out, inb, air=air, air_w=air_w, air_g=air_g,
        padding_g=padding_g, strength=strength, extra=extra, compatibility=compatibility,
        mode=mode, background_on=background_on, background_weight=background_weight,
        bg_mean=bg_mean, bg_std=bg_std, bg_model=bg_model,
        bg_ratio_inv2_scale=bg_ratio_inv2_scale, bg_effective_coils=bg_effective_coils,
        bg_grain_sg=bg_grain_sg, bg_grain_aniso=bg_grain_aniso,
        bg_thermal_fraction=bg_thermal_fraction, bg_fov_zero_prob=bg_fov_zero_prob,
        bg_fov_zero_frac=bg_fov_zero_frac, fov_padding_on=fov_padding_on, params=params)
    out = _mp2rage_extracranial(out, source, scalp, scalp_support=scalp_support, brain_w=brain_w,
        air_w=air_w, head_w=head_w, snr_w=snr_w, snr_lo=snr_lo, scale=scale, mode=mode,
        strength=strength, compatibility=compatibility, extracranial_on=extracranial_on,
        extracranial_strength=extracranial_strength,
        extracranial_noise_gain=extracranial_noise_gain, scalp_compress=scalp_compress,
        scalp_gamma=scalp_gamma, scalp_floor=scalp_floor, tissue_noise_field=tissue_noise_field,
        acquisition_guard_w=acquisition_guard_w)
    if posterior_spec is not None:
        # Stored-image posterior ambiguity must mix BOTH sides of the interface.
        # Running after brain, scalp and ratio background composition prevents the
        # exact target mask from acting as a rendering gate at the cerebellar edge.
        from augmentations.curricula.mp2rage_posterior import apply_posterior_boundary_ambiguity
        ambiguous = apply_posterior_boundary_ambiguity(
            out, inb, posterior_spec, voxel_sizes=voxel_sizes)
        boundary_weight = float(np.clip(strength, 0.0, 1.0))
        out = out + boundary_weight * (ambiguous - out)
    if zero_mask is not None:
        # Reconstructed support is the final acquisition geometry.  Applying it
        # last prevents later scalp transfer/noise from refilling cropped voxels.
        # Fade with the master background weight so t=0 remains exact identity.
        out[zero_mask] *= (1.0 - w_bg)
    return np.clip(out, 0.0, 1.0).astype(np.float32)
