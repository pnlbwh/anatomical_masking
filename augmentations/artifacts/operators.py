"""MRI augmentation operators used by online training and 3-D evaluation.

The registry exposes volume noise, resolution, tissue contrast, and focal
artifacts. Online operators use the supplied NumPy Generator. Legacy metal
painting retains process-global randomness, which evaluation isolates per trial.
Shared slice helpers remain where the 3-D renderers call them.
"""
from __future__ import annotations

import copy
import random

import cv2
import numpy as np
import scipy.ndimage
from scipy import ndimage as ndi
from scipy.ndimage import zoom

from augmentations.registry import register


# Shared helpers
def _identity_elastic(arr, **kw):
    """Default ``elastic_fn`` when none is supplied (registry / standalone call).

    The wrapper-supplied ``elastic_transform_opencv`` is a 2D-only cv2 routine;
    when a transform is invoked directly via ``augmentations.apply`` no callable
    is passed, so we fall back to a shape-preserving identity (no-op). This keeps
    the focal transforms finite + same-shape without altering the wrapper path,
    which always supplies the real ``elastic_fn``.
    """
    return arr


def _center_fit(arr, out_shape):
    """Center crop and/or zero-pad ``arr`` to exactly ``out_shape`` (n-D)."""
    out = np.zeros(out_shape, dtype=arr.dtype)
    src_slices, dst_slices = [], []
    for o, a in zip(out_shape, arr.shape):
        n = min(o, a)
        s_start = max((a - o) // 2, 0)
        d_start = max((o - a) // 2, 0)
        src_slices.append(slice(s_start, s_start + n))
        dst_slices.append(slice(d_start, d_start + n))
    out[tuple(dst_slices)] = arr[tuple(src_slices)]
    return out


# Background and overlay noise
@register(
    "noise_straight",
    kind="noise",
    severity_range=(0.0, 1.0),
    dims="either",
    label_preserving=True,
    has_detector=False,
)
def noise_straight(image, *, severity=None, rng=None, mask=None, **kw):
    """Background straight-line noise (verbatim from ``add_straight_noise``).

    For each row, fill the leading + trailing background run (pixels <= 0.05,
    up to the first foreground pixel from each side) with a per-row uniform
    value. Operates in place on a copy and returns it. ``severity``/``rng``
    unused on the 2D path (preserves the original global-``random`` draw
    sequence).

    3D: the same leading/trailing background fill is applied along the LAST axis
    for every (i, j, ...) line through the volume, so a thick-slice background is
    filled coherently. The new 3D path draws from ``rng`` (rule 8) rather than
    the global ``random`` module.
    """
    img = image  # mutate in place (caller passes the slice it wants changed)
    if img.ndim == 2:
        # --- verbatim 2D behavior (unchanged) ---
        for row in range(0, len(img)):
            max_pixel = random.uniform(0.5, 1)
            min_pixel = random.uniform(0.5, max_pixel)
            for pixel in range(0, len(img[row])):
                if img[row][pixel] > 0.05:
                    break
                else:
                    img[row][pixel] = random.uniform(min_pixel, max_pixel)
            for pixel in range(len(img[row]) - 1, 0, -1):
                if img[row][pixel] > 0.05:
                    break
                else:
                    img[row][pixel] = random.uniform(min_pixel, max_pixel)
        return img

    # --- 3D path: per-line background fill along the last axis ---
    r = np.random.default_rng(rng)
    last = img.shape[-1]
    for idx in np.ndindex(img.shape[:-1]):
        line = img[idx]
        max_pixel = float(r.uniform(0.5, 1))
        min_pixel = float(r.uniform(0.5, max_pixel))
        for pixel in range(0, last):
            if line[pixel] > 0.05:
                break
            line[pixel] = float(r.uniform(min_pixel, max_pixel))
        for pixel in range(last - 1, 0, -1):
            if line[pixel] > 0.05:
                break
            line[pixel] = float(r.uniform(min_pixel, max_pixel))
    return img


@register(
    "noise_random",
    kind="noise",
    severity_range=(0.0, 1.0),
    dims="either",
    label_preserving=True,
    has_detector=False,
)
def noise_random(image, *, severity=None, rng=None, mask=None, **kw):
    """Background random noise (verbatim from ``add_random_noise``).

    3D: same per-pixel background fill applied along the LAST axis for every line
    through the volume; the new 3D path draws from ``rng`` (rule 8).
    """
    img = image
    if img.ndim == 2:
        # --- verbatim 2D behavior (unchanged) ---
        for row in range(0, len(img)):
            for pixel in range(0, len(img[row])):
                max_pixel = random.uniform(0.5, 1)
                min_pixel = random.uniform(0.5, max_pixel)
                if img[row][pixel] > 0.05:
                    break
                else:
                    img[row][pixel] = random.uniform(min_pixel, max_pixel)
            for pixel in range(len(img[row]) - 1, 0, -1):
                max_pixel = random.uniform(0.5, 1)
                min_pixel = random.uniform(0.5, max_pixel)
                if img[row][pixel] > 0.05:
                    break
                else:
                    img[row][pixel] = random.uniform(min_pixel, max_pixel)
        return img

    # --- 3D path: per-line background fill along the last axis ---
    r = np.random.default_rng(rng)
    last = img.shape[-1]
    for idx in np.ndindex(img.shape[:-1]):
        line = img[idx]
        for pixel in range(0, last):
            max_pixel = float(r.uniform(0.5, 1))
            min_pixel = float(r.uniform(0.5, max_pixel))
            if line[pixel] > 0.05:
                break
            line[pixel] = float(r.uniform(min_pixel, max_pixel))
        for pixel in range(last - 1, 0, -1):
            max_pixel = float(r.uniform(0.5, 1))
            min_pixel = float(r.uniform(0.5, max_pixel))
            if line[pixel] > 0.05:
                break
            line[pixel] = float(r.uniform(min_pixel, max_pixel))
    return img


@register(
    "noise_overlay",
    kind="noise",
    severity_range=(0.0, 1.0),
    dims="either",
    label_preserving=True,
    has_detector=False,
)
def noise_overlay(image, *, severity=None, rng=None, mask=None, orig=None, **kw):
    """Whole-image additive uniform overlay noise (core of ``overlay_random_noise``).

    Adds the SAME random realization to ``image`` and (if given) ``orig`` so the
    caller's artifact-scoring diff is preserved. ``mask`` gates the optional
    brain-only mode. The score-collection bookkeeping stays in the wrapper.
    Returns ``(image, orig)``.

    3D: the same per-voxel additive uniform overlay generalized to n-D (built
    from ``arr.shape``), gated by the (optional) brain mask. The new 3D path
    draws from ``rng`` (rule 8) rather than the global ``random`` module.
    """
    if image.ndim == 2:
        # --- verbatim 2D behavior (unchanged) ---
        brain_only = random.randint(0, 1)
        absolute_max = random.uniform(0.03, 0.2)
        for row in range(0, len(image)):
            for pixel in range(0, len(image[row])):
                if brain_only == 1:
                    if mask[row][pixel] < 1:
                        continue
                max_pixel = random.uniform(0.01, absolute_max)
                min_pixel = random.uniform(0, max_pixel)
                image[row][pixel] += random.uniform(min_pixel, max_pixel)
                if orig is not None:
                    orig[row][pixel] += random.uniform(min_pixel, max_pixel)
        return image, orig

    # --- 3D path: vectorized per-voxel additive uniform overlay ---
    r = np.random.default_rng(rng)
    brain_only = int(r.integers(0, 2))
    absolute_max = float(r.uniform(0.03, 0.2))
    # Per-voxel max/min then a uniform draw in [min, max], generalized to n-D.
    max_pixel = r.uniform(0.01, absolute_max, size=image.shape)
    min_pixel = r.uniform(0.0, max_pixel)
    add_img = r.uniform(min_pixel, max_pixel)
    if brain_only == 1 and mask is not None:
        gate = np.asarray(mask) >= 1
        add_img = np.where(gate, add_img, 0.0)
    image += add_img.astype(image.dtype, copy=False)
    if orig is not None:
        add_orig = r.uniform(min_pixel, max_pixel)
        if brain_only == 1 and mask is not None:
            add_orig = np.where(gate, add_orig, 0.0)
        orig += add_orig.astype(orig.dtype, copy=False)
    return image, orig


# Resolution
@register(
    "randomize_resolution",
    kind="geometry",
    severity_range=(0.35, 0.95),
    dims="either",
    label_preserving=True,
    has_detector=False,
)
def randomize_resolution(image, *, severity=None, rng=None, mask=None, **kw):
    """Partial-volume resolution loss (verbatim from ``randomize_resolution``).

    Gaussian anti-alias prefilter -> INTER_AREA downscale -> varied recon upscale.
    Image-only; the wrapper assigns the result back to ``slice_image`` only.

    3D: ``ndi.zoom`` down by random per-axis factors (order=1) then back up to the
    original shape, mimicking a thick-slice acquisition (the through-plane factor
    is drawn separately so it can be more aggressive / anisotropic). The new 3D
    path draws from ``rng`` (rule 8).
    """
    # severity -> downscale factor (MONOTONE): higher severity = LOWER scale = more resolution
    # loss. severity_range is (0.35,0.95) so band 0 -> scale ~0.95 (mild), band 1 -> scale ~0.35
    # (aggressive); the old random.uniform(0.35,0.95) ignored severity (RMS flat across bands).
    sev = 0.5 if severity is None else float(np.clip(severity, 0.0, 1.0))
    scale_center = float(np.clip(0.97 - 0.65 * sev, 0.20, 0.97))   # band0->0.97 mild, band1->0.32 aggressive
    if image.ndim == 2:
        height, width = image.shape[:2]
        scale_factor = float(np.clip(scale_center * random.uniform(0.9, 1.1), 0.2, 0.97))
        sigma = 0.5 * (1.0 / scale_factor - 1.0)
        prefiltered = (cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
                       if sigma > 1e-2 else image)
        new_width = max(1, int(width * scale_factor))
        new_height = max(1, int(height * scale_factor))
        downscaled_image = cv2.resize(prefiltered, (new_width, new_height),
                                      interpolation=cv2.INTER_AREA)
        up_kernel = random.choice(
            [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4])
        upscaled_image = cv2.resize(downscaled_image, (width, height),
                                    interpolation=up_kernel)
        return upscaled_image

    # --- 3D path: anisotropic zoom down then back up (thick-slice acquisition) ---
    r = np.random.default_rng(rng)
    shape = image.shape
    # in-plane (last two axes) share a factor; through-plane (axis 0) its own (more aggressive).
    # Both CENTERED on the severity-driven scale (+/-10% tail) so degree is monotone in severity,
    # while the in-plane/through split preserves the anisotropy knob.
    in_plane = float(np.clip(scale_center * r.uniform(0.9, 1.1), 0.2, 0.97))
    through = float(np.clip(scale_center * r.uniform(0.72, 1.0), 0.18, 0.97))
    factors = (through, in_plane, in_plane)
    small = ndi.zoom(image, factors, order=1)
    # zoom back to the EXACT original shape (factor = out/in per axis).
    up_factors = tuple(s_out / s_in for s_out, s_in in zip(shape, small.shape))
    up_kernel = int(r.integers(0, 4))  # order in {0,1,2,3} ~ varied recon kernel
    upscaled = ndi.zoom(small, up_factors, order=up_kernel)
    # guard exact shape (rounding can be off by 1) via center crop/pad.
    if upscaled.shape != shape:
        upscaled = _center_fit(upscaled, shape)
    return upscaled.astype(image.dtype, copy=False)


# Mask-conditioned intensity and contrast
@register(
    "brain_intensity",
    kind="intensity",
    severity_range=(0.1, 3.0),
    dims="either",
    label_preserving=True,
    has_detector=False,
)
def brain_intensity(image, *, severity=None, rng=None, mask=None,
                    intensity_change=None, add=None, normalize=None, **kw):
    """Scale in-brain intensity (verbatim from ``change_brain_intensity``).

    ``mask`` is the brain mask. ``normalize`` is the instance's ``normalize``
    callable (passed by the wrapper) so the exact per-instance normalization is
    preserved. The boolean-mask math is already n-D agnostic, so this works on a
    2D slice OR a 3D volume unchanged; only the wrapper-only ``intensity_change``/
    ``add`` defaults are supplied here for the standalone (registry) call.
    """
    if intensity_change is None:
        r = np.random.default_rng(rng)
        intensity_change = float(r.uniform(0.1, 3.0)) if severity is None \
            else float(np.clip(severity, 0.1, 3.0))
    if add is None:
        add = 0
    if mask is None:
        mask = np.ones_like(image)
    m = mask > 0
    image[m & (add == 0)] *= intensity_change
    image[m & (add == 1)] *= intensity_change
    if normalize is not None:
        image = normalize(image)
    np.clip(image, 0, 1, out=image)
    return image


@register(
    "skull_intensity",
    kind="intensity",
    severity_range=(0.1, 3.0),
    dims="either",
    label_preserving=True,
    has_detector=False,
)
def skull_intensity(image, *, severity=None, rng=None, mask=None,
                    intensity_change=None, add=None, normalize=None, **kw):
    """Scale extracranial intensity (verbatim from ``change_skull_intensity``).

    Boolean-mask math is n-D agnostic (works on a 2D slice OR 3D volume); the
    wrapper-only ``intensity_change``/``add`` defaults are supplied for the
    standalone (registry) call.
    """
    if intensity_change is None:
        r = np.random.default_rng(rng)
        intensity_change = float(r.uniform(0.1, 3.0)) if severity is None \
            else float(np.clip(severity, 0.1, 3.0))
    if add is None:
        add = 0
    if mask is None:
        mask = np.zeros_like(image)
    condition_mask = (mask == 0) & (image > 0.1)
    image[condition_mask] *= intensity_change
    if normalize is not None:
        image = normalize(image)
    np.clip(image, 0, 1, out=image)
    return image


def _region_contrast(image, region, contrast_factor, severity, rng):
    """Apply the shared contrast rule in place to the selected tissue region."""
    if contrast_factor is None:
        r = np.random.default_rng(rng)
        contrast_factor = float(r.uniform(0.0, 3.0)) if severity is None \
            else float(np.clip(severity, 0.0, 3.0))
    if np.any(region):
        mean_intensity = np.mean(image[region])
        image[region] = (image[region] - mean_intensity) * contrast_factor + mean_intensity
    np.clip(image, 0, 1, out=image)
    return image


@register(
    "brain_contrast",
    kind="intensity",
    severity_range=(0.0, 3.0),
    dims="either",
    label_preserving=True,
    has_detector=False,
)
def brain_contrast(image, *, severity=None, rng=None, mask=None,
                   contrast_factor=None, **kw):
    """In-brain contrast about the brain mean (verbatim from ``change_brain_contrast``).

    Boolean-mask math is n-D agnostic (works on a 2D slice OR 3D volume); the
    wrapper-only ``contrast_factor`` default is supplied for the standalone
    (registry) call.
    """
    if mask is None:
        mask = np.ones_like(image)
    return _region_contrast(image, mask > 0, contrast_factor, severity, rng)


@register(
    "skull_contrast",
    kind="intensity",
    severity_range=(0.0, 3.0),
    dims="either",
    label_preserving=True,
    has_detector=False,
)
def skull_contrast(image, *, severity=None, rng=None, mask=None,
                   contrast_factor=None, **kw):
    """Extracranial contrast about the skull mean (verbatim from ``change_skull_contrast``).

    Boolean-mask math is n-D agnostic (works on a 2D slice OR 3D volume); the
    wrapper-only ``contrast_factor`` default is supplied for the standalone
    (registry) call.
    """
    if mask is None:
        mask = np.zeros_like(image)
    return _region_contrast(image, (mask == 0) & (image > 0.1), contrast_factor, severity, rng)


# Signal dropout, metal, and ringing
@register(
    "signal_drop_band",
    kind="noise",
    severity_range=(0.0, 1.0),
    dims="either",
    label_preserving=True,
    has_detector=True,
    excluded_parameters=("heavy_tail",),
)
def signal_drop_band(image, *, severity=None, rng=None, mask=None,
                     heavy_tail=None, **kw):
    """Soft-edged contiguous signal-dropout band(s).

    TWO call modes:
    * TRAINING (``heavy_tail`` supplied = the Augmentimage instance's
      ``_heavy_tail`` callable): unchanged per-slice 2D behavior; band width /
      attenuation come from the per-slice tier state. Returns ``(image, min_weight)``.
    * STANDALONE / registry (``heavy_tail`` is None): deterministic from ``rng``,
      3D-COHERENT (one contiguous band-slab across the whole volume, not scattered
      per-slice), and OVER-GENERATED per the superset philosophy -- at high severity
      the band is WIDE (up to ~30% of the axis) and DEEP (drops toward 0). The old
      standalone path produced a 2-8 PIXEL band dimmed to only ~0.55-0.80 (and got
      *weaker* as severity rose), so a "signal drop" was near-invisible; this fixes
      that generation weakness.
    """
    if heavy_tail is not None:
        # --- training path: verbatim 2D per-slice behavior (unchanged) ---
        h, w = image.shape[:2]
        axis = random.randint(0, 1)
        n = h if axis == 0 else w
        weight = np.ones(n, dtype=np.float32)
        for _ in range(random.randint(1, 2)):
            width = int(round(heavy_tail(2.0, 4.0, 8.0)))
            pos = random.randint(0, max(0, n - width))
            atten = heavy_tail(0.10, 0.55, 0.80)
            band = np.ones(n, dtype=np.float32)
            band[pos:pos + width] = atten
            weight = np.minimum(weight, band)
        weight = scipy.ndimage.gaussian_filter1d(
            weight, sigma=random.uniform(0.3, 0.7))
        mult = weight[:, None] if axis == 0 else weight[None, :]
        out = np.maximum(image * mult, 0.0).astype(np.float32)
        return out, float(weight.min())

    # --- standalone / registry path: strong, deterministic, 3D-coherent ---
    s = 0.5 if severity is None else float(np.clip(severity, 0.0, 1.0))
    r = rng if rng is not None else np.random.default_rng()
    arr = np.asarray(image, dtype=np.float32)
    ndim = arr.ndim
    axis = int(r.integers(0, min(2, ndim)))         # band runs across rows or columns
    n = int(arr.shape[axis])
    weight = np.ones(n, dtype=np.float32)
    for _ in range(int(r.integers(1, 3))):          # 1-2 bands
        frac = 0.03 + 0.27 * s                       # WIDTH 3% (mild) -> 30% (severe)
        width = max(3, int(round(n * frac * float(r.uniform(0.7, 1.3)))))
        pos = int(r.integers(0, max(1, n - width + 1)))
        # DEPTH: deeper drop as severity rises (atten -> ~0 at sev 1, ~0.4 at sev 0.3)
        atten = float(np.clip((1.0 - s) * 0.6 * float(r.uniform(0.6, 1.1)), 0.0, 0.85))
        band = np.ones(n, dtype=np.float32)
        band[pos:pos + width] = atten
        weight = np.minimum(weight, band)
    weight = scipy.ndimage.gaussian_filter1d(weight, sigma=max(0.5, 0.015 * n))
    weight = np.maximum(weight, 0.15)               # FLOOR: in-band brain stays >=0.15*signal (not air-like
    #   ~0) so the masker is never taught "near-zero voxel = brain" (the over-inclusion accelerant). Air
    #   (~0) * 0.15 is still ~0, so the band stays clearly visible; width/depth/orientation unchanged.
    shape = [1] * ndim
    shape[axis] = n
    out = np.maximum(arr * weight.reshape(shape), 0.0).astype(np.float32)  # mult in [0,1]
    return out, float(weight.min())


def _metal_paint_2d(image, mask, orig, elastic_fn):
    """Verbatim 2D body of :func:`metal_paint` (extracted so the registry fn can
    dispatch 2D vs 3D). MATH unchanged; the only difference from the original
    method is that ``mask``/``orig``/``elastic_fn`` arrive as explicit args.
    """
    orig_mri_copy = orig
    brain_only = random.randint(0, 1)
    brain_only = 1
    blur_size = (2 * random.randint(10, 30)) + 1
    height, width = image.shape[:2]
    overlay = np.zeros_like(image)
    alpha = random.uniform(0.4, 1)
    intensity = random.uniform(-0.5, 0.5)
    center = (random.randint(0, width), random.randint(0, height))
    color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
    bright_val = random.uniform(0, 1)
    color = (bright_val, bright_val, bright_val)
    dark_upper = np.mean(image[mask > 0]) * 2
    dark_rgb_val = random.uniform(dark_upper / 2, dark_upper)
    dark_color = (dark_rgb_val, dark_rgb_val, dark_rgb_val)
    radius = random.randint(1, min(height, width) // 16)

    cv2.circle(overlay, center, radius, color, thickness=-1)

    alpha_val = random.randint(5, random.randint(15, 100))
    sigma_val = random.randint(1, random.randint(2, 5))
    overlay = elastic_fn(overlay, alpha=alpha_val, sigma=sigma_val)
    overlay = cv2.GaussianBlur(overlay, (blur_size, blur_size), 0)
    if brain_only == 0:
        overlay[mask < 0.5] = 0
    image = cv2.addWeighted(image, 1, overlay, alpha, 0)
    orig_mri_copy = cv2.addWeighted(orig_mri_copy, 1, overlay, alpha, 0)
    overlay = np.zeros_like(image)

    cv2.circle(overlay, center, int(round(radius * 1.2)), (0.001, 0.001, 0.001), thickness=10)
    blur_size = (2 * random.randint(20, random.randint(25, 50))) + 1

    cv2.circle(overlay, center, int(round(radius * 1.2)), dark_color, thickness=1)

    blur_size = (2 * random.randint(4, random.randint(5, 30))) + 1
    overlay = cv2.GaussianBlur(overlay, (blur_size, blur_size), 0)

    overlay = elastic_fn(overlay, alpha=alpha_val, sigma=sigma_val)
    overlay = cv2.GaussianBlur(overlay, (blur_size, blur_size), 0)
    overlay_max = float(np.max(overlay))
    if overlay_max <= 1e-8:
        # Degenerate metal overlay (rare); skip the dark-ring layer.
        return image, orig_mri_copy, False
    ratio = dark_rgb_val / overlay_max

    overlay *= ratio

    overlay[overlay > dark_rgb_val] = dark_rgb_val

    if brain_only == 0:
        overlay[mask < 0.5] = 0
    image = cv2.addWeighted(image, 1, overlay, -1, 0)

    orig_mri_copy = cv2.addWeighted(orig_mri_copy, 1, overlay, -1, 0)
    np.clip(image, 0, 1, out=image)
    darken = random.randint(0, 1)
    image[overlay > 0] -= (overlay[overlay > 0] / random.randint(1, 8))
    image[image < 0] = 0
    image = np.array(image)

    np.clip(orig_mri_copy, 0, 1, out=orig_mri_copy)
    orig_mri_copy[overlay > 0] -= (overlay[overlay > 0] / random.randint(1, 8))
    orig_mri_copy[orig_mri_copy < 0] = 0
    return image, orig_mri_copy, True


@register(
    "metal_paint",
    kind="focal",
    severity_range=(0.0, 1.0),
    dims="either",
    label_preserving=True,
    has_detector=True,
)
def metal_paint(image, *, severity=None, rng=None, mask=None,
                orig=None, elastic_fn=None, **kw):
    """Legacy cv2-paint metal-susceptibility artifact (verbatim from
    ``simulate_metal_artifact``).

    NOTE: distinct algorithm from the Stage-2 3D bright-blob ``metal`` sim; kept
    as its own entry per the no-rewrite hard rule. The SAME random overlay is
    applied to ``image`` and ``orig`` (the score diff is computed by the wrapper).
    ``elastic_fn`` is the instance's ``elastic_transform_opencv`` callable.
    Returns ``(image, orig, applied)`` — ``applied`` is False for the degenerate
    early-return case so the wrapper can skip the score update.

    3D: cv2 cannot draw on a 3D array, so per rule 6 the 2D paint is applied to a
    contiguous slab of central slices (a focal 3D blob) along axis 0; off-slab
    voxels are untouched. ``elastic_fn`` defaults to identity and ``orig`` to a
    copy of ``image`` when called standalone via the registry.
    """
    if mask is None:
        mask = np.ones_like(image)
    if elastic_fn is None:
        elastic_fn = _identity_elastic
    if orig is None:
        orig = image.copy()

    if image.ndim == 2:
        return _metal_paint_2d(image, mask, orig, elastic_fn)

    # --- 3D path: focal blob on a contiguous central slab of slices ---
    # The 2D body reads ``np.mean(image[mask>0])``, which is NaN on a slice with
    # no brain voxels; center the slab on the mask centroid and only paint slices
    # that contain brain so the blob is anchored to anatomy and never NaNs.
    r = np.random.default_rng(rng)
    s = 0.5 if severity is None else float(np.clip(severity, 0.0, 1.0))
    nz = image.shape[0]
    brain = mask > 0
    if not np.any(brain):
        return image, orig, False
    per_slice = brain.reshape(nz, -1).any(axis=1)
    brain_zs = np.flatnonzero(per_slice)
    cz = int(brain_zs[int(r.integers(0, len(brain_zs)))])  # a slice that has brain
    half = max(1, int(round(1 + 3 * s)))         # slab half-thickness grows w/ sev
    z0 = max(0, cz - half)
    z1 = min(nz, cz + half + 1)
    applied_any = False
    # COHERENT 3D blob: _metal_paint_2d draws its center/radius/color/alpha from the global `random`
    # state, so calling it per-slice gave each slice an INDEPENDENT blob -> incoherent flicker in the
    # sagittal/coronal views (adjacent-slice artifact corr ~0). Reset to the SAME state before every
    # slab slice so all slices share one blob -> a coherent metal column anchored to anatomy.
    paint_state = random.getstate()
    for z in range(z0, z1):
        if not per_slice[z]:
            continue                              # skip empty-mask slices (would NaN)
        random.setstate(paint_state)
        sl, ol, ap = _metal_paint_2d(
            image[z], mask[z], orig[z], elastic_fn)
        image[z] = sl
        orig[z] = ol
        applied_any = applied_any or ap
    return image, orig, applied_any


def _create_ring_2d(image, mask, orig, scale, blurr_strength, pos_or_neg_x,
                    pos_or_neg_y, x_threshold, y_threshold, partial, *, rng=None):
    """2D ring helper shared by the 2D/3D paths, using the caller's RNG.

    Ring math and draw distributions are preserved. The inner
    ``resize`` canvas is derived from ``image.shape`` instead of the literal
    ``256`` (identical at the production 256x256 size, but valid for any size so
    the edge map stays mask-shaped — the original hardcoded-256 canvas raised an
    IndexError for non-256 slices).
    """
    r = np.random.default_rng(rng)
    cy, cx = image.shape[:2]

    def resize(img, scale):
        resized_image = zoom(img, (scale, scale), order=1)
        background = np.zeros((cy, cx), dtype=img.dtype)
        start_x = max((cx - resized_image.shape[1]) // 2, 0)
        start_y = max((cy - resized_image.shape[0]) // 2, 0)
        resized_image_clipped = resized_image[:min(cy, resized_image.shape[0]),
                                              :min(cx, resized_image.shape[1])]
        background[start_y:start_y + resized_image_clipped.shape[0],
                   start_x:start_x + resized_image_clipped.shape[1]] = resized_image_clipped
        return background

    resized_mask = resize(image, scale)
    sobel_x = cv2.Sobel(resized_mask, cv2.CV_64F, 1, 0, ksize=1)
    sobel_y = cv2.Sobel(resized_mask, cv2.CV_64F, 0, 1, ksize=1)

    gradient_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    gmag_max = float(gradient_magnitude.max())
    if gmag_max <= 1e-8:
        return image, orig, False
    gradient_magnitude = np.uint8(gradient_magnitude / gmag_max * 255)
    _, edges = cv2.threshold(gradient_magnitude / 1, 50, 255, cv2.THRESH_BINARY)
    edges_max = float(np.max(edges))
    if edges_max <= 1e-8:
        return image, orig, False
    edges = edges / edges_max
    edges[mask == 0] = 0
    small_mask = copy.deepcopy(mask)
    small_mask = resize(small_mask, r.uniform(0.2, 0.9))

    edges[small_mask == 1] = 0

    if partial == 1:
        if pos_or_neg_x == 0:
            edges[:, x_threshold:] = 0
        else:
            edges[:, :x_threshold] = 0
        if pos_or_neg_y == 0:
            edges[y_threshold:, :] = 0
        else:
            edges[:y_threshold, :] = 0

    indices = np.where(edges > 0.5)

    interval = int(r.integers(20, 101))
    brain_mean = np.mean(image[mask > 0])
    brightness_factor = r.uniform(-1 * (brain_mean / 30), brain_mean / 12)
    only_brightness = int(r.integers(0, 1))

    def add_rings(img, blurr_strength):
        for ind in range(0, len(indices[0])):
            if ind % interval == 0:
                blurr_strength = int(r.integers(0, 7))
            x = indices[0][ind]
            y = indices[1][ind]
            if only_brightness == 0:
                img[x][y] += brightness_factor
                if img[x][y] > brain_mean * 2:
                    img[x][y] = brain_mean * 2
                if img[x][y] > 1:
                    img[x][y] = 1
                elif img[x][y] < 0:
                    img[x][y] = 0
            else:
                ind_diff = int(r.integers(1, 6))
                img[x][y] = ((blurr_strength * img[indices[0][ind - ind_diff]]
                             [indices[1][ind - ind_diff]]) + img[x][y]) / (blurr_strength + 1)
        return img

    image = add_rings(image, blurr_strength)
    orig = add_rings(orig, blurr_strength)
    return image, orig, True


@register(
    "create_ring",
    kind="focal",
    severity_range=(0.0, 1.0),
    dims="either",
    label_preserving=True,
    has_detector=True,
    excluded_parameters=("div_mag", "blurr_strength", "orig",),
)
def create_ring(image, *, severity=None, rng=None, mask=None, orig=None,
                scale=None, blurr_strength=None, pos_or_neg_x=None,
                pos_or_neg_y=None, x_threshold=None, y_threshold=None,
                partial=None, div_mag=None, **kw):
    """Edge-following ringing splat (verbatim from ``create_ring``).

    The SAME random brightness ring is added to ``image`` and ``orig`` (the score
    diff is computed by the wrapper). ``mask`` is the brain mask. Returns
    ``(image, orig, applied)`` — ``applied`` False marks the degenerate no-op
    cases (constant slice / no edges) so the wrapper skips scoring.

    Wrapper-only params (``scale``/``blurr_strength``/``partial``/...) default to
    no-ops when called standalone via the registry; ``orig`` defaults to a copy
    of ``image``.

    3D: cv2/Sobel edge logic is 2D, so per rule 6 the ring is painted onto a
    contiguous central slab of slices (a focal 3D blob) along axis 0.
    """
    if mask is None:
        mask = np.ones_like(image)
    if orig is None:
        orig = image.copy()
    if scale is None:
        scale = 1.0
    if partial is None:
        partial = 0  # skip the partial-edge masking block

    r = np.random.default_rng(rng)
    if image.ndim == 2:
        return _create_ring_2d(
            image, mask, orig, scale, blurr_strength, pos_or_neg_x,
            pos_or_neg_y, x_threshold, y_threshold, partial, rng=r)

    # --- 3D path: paint the ring on a contiguous slab of brain-bearing slices ---
    # The 2D body reads ``np.mean(image[mask>0])``; anchor the slab to the mask so
    # each painted slice has brain (empty-mask slices are skipped — they also have
    # no edges to ring after ``edges[mask==0]=0``).
    s = 0.5 if severity is None else float(np.clip(severity, 0.0, 1.0))
    nz = image.shape[0]
    brain = mask > 0
    if not np.any(brain):
        return image, orig, False
    per_slice = brain.reshape(nz, -1).any(axis=1)
    brain_zs = np.flatnonzero(per_slice)
    cz = int(brain_zs[int(r.integers(0, len(brain_zs)))])
    half = max(1, int(round(1 + 3 * s)))
    z0 = max(0, cz - half)
    z1 = min(nz, cz + half + 1)
    applied_any = False
    for z in range(z0, z1):
        if not per_slice[z]:
            continue
        sl, ol, ap = _create_ring_2d(
            image[z], mask[z], orig[z], scale, blurr_strength, pos_or_neg_x,
            pos_or_neg_y, x_threshold, y_threshold, partial, rng=r)
        image[z] = sl
        orig[z] = ol
        applied_any = applied_any or ap
    return image, orig, applied_any


# Focal shading
@register(
    "draw_shape",
    kind="intensity",
    severity_range=(0.0, 1.0),
    dims="either",
    label_preserving=True,
    has_detector=False,
)
def draw_shape(image, *, severity=None, rng=None, mask=None, **kw):
    """Focal B1+/dielectric shading pocket (physics-following replacement for the
    legacy painted random-shape overlay).

    The old transform pasted random coloured rectangles/circles/lines — a
    giveaway-fake that corresponds to no real MRI artifact and can teach the QC
    head a hard-shape shortcut. This version instead imposes a single SMOOTH,
    LOCALIZED multiplicative lobe that brightens or darkens one region — the look
    of a dielectric-resonance / transmit-B1 inhomogeneity pocket (common at 3T/7T,
    especially the temporal lobes). It is distinguished from the global ``bias``
    field by being a SINGLE FOCAL mid-scale lobe (placed inside the brain when a
    mask is given), works in 2D or 3D, and never moves anatomy.
    """
    r = np.random.default_rng(rng)
    img = np.asarray(image, dtype=np.float32)
    s = 0.5 if severity is None else float(np.clip(severity, 0.0, 1.0))
    if s <= 0.0:
        return img.copy()
    shape, nd = img.shape, img.ndim
    grids = np.ogrid[tuple(slice(0, n) for n in shape)]
    if mask is not None and np.any(np.asarray(mask) > 0.5):
        idx = np.where(np.asarray(mask) > 0.5)
        k = int(r.integers(len(idx[0])))
        center = [float(idx[ax][k]) for ax in range(nd)]
    else:
        center = [float(r.uniform(0.3 * n, 0.7 * n)) for n in shape]
    scl = float(r.uniform(0.06, 0.13)) * float(np.mean(shape))     # TIGHT focal pocket (vs global bias)
    d2 = sum((grids[ax] - center[ax]) ** 2 for ax in range(nd)).astype(np.float32)
    lobe = np.exp(-d2 / (2.0 * scl ** 2)).astype(np.float32)
    amp = float(r.uniform(0.3, 0.8)) * s * (1.0 if r.random() < 0.5 else -1.0)
    return np.clip(img * (1.0 + amp * lobe), 0.0, 1.0).astype(np.float32)


@register(
    "draw_shape_opacity",
    kind="intensity",
    severity_range=(0.0, 1.0),
    dims="either",
    label_preserving=True,
    has_detector=False,
)
def draw_shape_opacity(image, *, severity=None, rng=None, mask=None, **kw):
    """Subtle translucent shading — a low-amplitude, broader variant of
    :func:`draw_shape`'s focal B1/dielectric pocket (physics-following replacement
    for the legacy translucent painted overlay).

    NOTE: this uses the SAME focal scale as ``draw_shape`` with roughly HALF the
    amplitude — i.e. it is a genuinely quieter twin, not a separate phenomenon. It
    is a candidate to FOLD INTO ``draw_shape`` rather than keep as a separate QC
    label; surfaced for review rather than removed unilaterally.
    """
    r = np.random.default_rng(rng)
    img = np.asarray(image, dtype=np.float32)
    s = 0.5 if severity is None else float(np.clip(severity, 0.0, 1.0))
    if s <= 0.0:
        return img.copy()
    shape, nd = img.shape, img.ndim
    grids = np.ogrid[tuple(slice(0, n) for n in shape)]
    if mask is not None and np.any(np.asarray(mask) > 0.5):
        idx = np.where(np.asarray(mask) > 0.5)
        k = int(r.integers(len(idx[0])))
        center = [float(idx[ax][k]) for ax in range(nd)]
    else:
        center = [float(r.uniform(0.3 * n, 0.7 * n)) for n in shape]
    scl = float(r.uniform(0.07, 0.14)) * float(np.mean(shape))     # focal, ~draw_shape scale
    d2 = sum((grids[ax] - center[ax]) ** 2 for ax in range(nd)).astype(np.float32)
    lobe = np.exp(-d2 / (2.0 * scl ** 2)).astype(np.float32)
    amp = float(r.uniform(0.10, 0.30)) * s * (1.0 if r.random() < 0.5 else -1.0)   # quieter than draw_shape
    return np.clip(img * (1.0 + amp * lobe), 0.0, 1.0).astype(np.float32)


# Receive-coil dropout
@register(
    "random_erasing",
    kind="intensity",
    severity_range=(0.0, 1.0),
    dims="either",
    label_preserving=True,
    has_detector=False,
)
def random_erasing(image, *, severity=None, rng=None, mask=None,
                   magnitude_noise=False, **kw):
    """Receive-coil element failure / low-sensitivity dropout (physics-following
    replacement for the legacy cutout that pasted hard mean-filled / zero boxes).

    A failed or low-sensitivity surface-coil channel produces, over the region it
    covers, BOTH a smooth signal LOSS *and* a NOISE-FLOOR RISE (image SNR is
    proportional to coil sensitivity, so where the signal drops the relative noise
    grows). The drop is anchored at the head PERIPHERY (surface coils sit on the
    skin), asymmetric (one side), with a smooth spatial falloff. The combined
    darken+grain signature is what distinguishes it from a pure multiplicative
    ``bias`` shading (which never raises the noise floor and may also brighten).
    Works in 2D or 3D and never moves anatomy (label-preserving). The historical
    default adds one signed Gaussian channel and clips byte-identically.
    ``magnitude_noise=True`` instead adds independent real/imaginary local noise
    channels and returns their magnitude (Rician), preventing signed-noise floor
    clipping while retaining the same spatial coil-loss field.
    """
    r = np.random.default_rng(rng)
    img = np.asarray(image, dtype=np.float32)
    s = 0.5 if severity is None else float(np.clip(severity, 0.0, 1.0))
    if s <= 0.0:
        return img.copy()
    shape, nd = img.shape, img.ndim
    grids = np.ogrid[tuple(slice(0, n) for n in shape)]
    atten = np.ones(shape, dtype=np.float32)
    for _ in range(int(r.integers(1, 3))):              # 1-2 failed elements
        center = []
        for ax in range(nd):
            n = shape[ax]
            if r.random() < 0.6:                        # bias the anchor to a FOV face (skin)
                center.append(0.0 if r.random() < 0.5 else float(n - 1))
            else:
                center.append(float(r.uniform(0.25 * n, 0.75 * n)))
        scl = float(r.uniform(0.16, 0.30)) * float(np.mean(shape))   # REGIONAL falloff (one side, not global)
        d2 = sum((grids[ax] - center[ax]) ** 2 for ax in range(nd)).astype(np.float32)
        lobe = np.exp(-d2 / (2.0 * scl ** 2)).astype(np.float32)
        atten = atten * (1.0 - float(r.uniform(0.4, 0.9)) * s * lobe)
    out = img * atten
    # noise-floor rise where sensitivity dropped (SNR collapse): std scales with (1-atten)
    if mask is not None and np.any(np.asarray(mask) > 0.5):
        base_sigma = float(img[np.asarray(mask) > 0.5].std()) or 0.05
    else:
        base_sigma = float(img.std()) or 0.05
    nstd = base_sigma * (0.3 + 1.0 * s) * (1.0 - atten)
    if bool(magnitude_noise):
        channel_std = nstd * np.float32(1.0 / np.sqrt(2.0))
        real_noise = r.normal(0.0, 1.0, shape).astype(np.float32) * channel_std
        imag_noise = r.normal(0.0, 1.0, shape).astype(np.float32) * channel_std
        out = np.hypot(out + real_noise, imag_noise)
    else:
        out = out + r.normal(0.0, 1.0, shape).astype(np.float32) * nstd
    return np.clip(out, 0.0, 1.0).astype(np.float32)
