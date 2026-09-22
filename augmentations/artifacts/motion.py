"""Registered motion primitives and 3-D FSE echo-train blur.

The translation, ghosting, pulsation, and blur primitives take explicit
parameters. Rigid k-space motion lives in ``kspace.py``.
"""
from __future__ import annotations


import cv2
import numpy as np

from augmentations.registry import register


def oriented_gaussian_kernel(sigma_a, sigma_b, theta, size=15):
    """Anisotropic Gaussian kernel oriented at ``theta`` (verbatim core)."""
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    ct, st = np.cos(theta), np.sin(theta)
    x_rot = xx * ct + yy * st
    y_rot = -xx * st + yy * ct
    k = np.exp(-0.5 * ((x_rot / max(sigma_a, 0.1)) ** 2
                       + (y_rot / max(sigma_b, 0.1)) ** 2))
    s = float(k.sum())
    if s > 0:
        k = k / s
    return k.astype(np.float32)


def motion_psf_kernel(length, theta, size=15):
    """Linear motion-blur PSF: a line kernel at ``theta`` (verbatim core).

    LEGACY rasterized variant: ``cv2.line`` draws a 1px BINARY line, so the
    orientation is quantized to ~8-20 distinct kernels over 180deg (sub-degree
    angle changes render identically). Kept verbatim because the byte-identity
    guard (``legacy_aniso_blur``) routes through this exact builder; use
    :func:`motion_psf_kernel_aa` for the smooth continuum.
    """
    k = np.zeros((size, size), dtype=np.float32)
    cx = cy = size // 2
    x1 = int(round(cx - 0.5 * length * np.cos(theta)))
    y1 = int(round(cy - 0.5 * length * np.sin(theta)))
    x2 = int(round(cx + 0.5 * length * np.cos(theta)))
    y2 = int(round(cy + 0.5 * length * np.sin(theta)))
    cv2.line(k, (x1, y1), (x2, y2), 1.0, 1)
    s = float(k.sum())
    if s > 0:
        k = k / s
    return k


def motion_psf_kernel_aa(length, theta, size=15):
    """Anti-aliased linear motion-blur PSF (smooth length/angle continuum).

    A constant-velocity smear is a TOP-HAT line profile (every point along the
    streak weighted equally) -- physically correct, same as the legacy kernel,
    NOT a Gaussian. The legacy ``cv2.line`` rasterizes that line to a 1px binary
    stamp, which quantizes orientation to a handful of kernels per 180deg. Here we
    instead SPLAT the line by finely sampling points along it and depositing each
    sample with BILINEAR weights onto the 4 nearest pixels. Result: fractional
    kernel weights and a kernel that varies continuously (smoothly) with both
    ``length`` and ``theta``. The integrated profile is still uniform (top-hat).
    """
    k = np.zeros((size, size), dtype=np.float32)
    cx = cy = size // 2
    L = float(length)
    ct, st = np.cos(theta), np.sin(theta)
    if L <= 1e-6:
        k[cy, cx] = 1.0
        return k
    # sample densely along the line so each unit of length deposits equal weight
    n = max(2, int(np.ceil(L * 8.0)) + 1)
    ts = np.linspace(-0.5 * L, 0.5 * L, n)
    for t in ts:
        x = cx + t * ct
        y = cy + t * st
        x0 = int(np.floor(x)); y0 = int(np.floor(y))
        fx = x - x0; fy = y - y0
        for dy, wy in ((0, 1.0 - fy), (1, fy)):
            yy = y0 + dy
            if yy < 0 or yy >= size or wy <= 0.0:
                continue
            for dx, wx in ((0, 1.0 - fx), (1, fx)):
                xx = x0 + dx
                if xx < 0 or xx >= size or wx <= 0.0:
                    continue
                k[yy, xx] += np.float32(wx * wy)
    s = float(k.sum())
    if s > 0:
        k = k / s
    return k.astype(np.float32)


@register("motion_pe_replicas", kind="kspace", dims="2d",
          label_preserving=True, has_detector=True)
def motion_pe_replicas(arr, *, severity=0.0, rng=None, mask=None,
                       pe_axis=0, shifts=(), alphas=(), **kw):
    """Bulk translational motion: PE-locked alpha-blended shifted replicas.

    Verbatim core of ``add_pe_motion``: out = (1-Σα)·img + Σ αᵢ·roll(img, shiftᵢ),
    each replica rolled from the ORIGINAL image along ``pe_axis``.
    """
    img = np.asarray(arr, dtype=np.float32)
    total = float(sum(alphas))
    accum = (1.0 - total) * img
    for shift, a in zip(shifts, alphas):
        accum = accum + float(a) * np.roll(img, int(shift), axis=pe_axis)
    return np.clip(accum, 0.0, 1.0).astype(np.float32)


@register("motion_ghost_stack", kind="kspace", dims="2d",
          label_preserving=True, has_detector=True)
def motion_ghost_stack(arr, *, severity=0.0, rng=None, mask=None,
                       pe_axis=0, offsets=(), alphas=(), blur_sigmas=None, **kw):
    """Translational ghosts at fractional-FOV offsets (verbatim ``add_ghost_stack``).

    Each ghost is rolled from the CUMULATIVELY-updated image, optionally blurred
    (``blur_sigmas[i]`` is the σ, or None for no blur), then alpha-blended:
    img = (1-αᵢ)·img + αᵢ·rolled.
    """
    img = np.asarray(arr, dtype=np.float32)
    out = img.copy()
    n = len(offsets)
    sigmas = blur_sigmas if blur_sigmas is not None else [None] * n
    for offset, a, sig in zip(offsets, alphas, sigmas):
        rolled = np.roll(out, int(offset), axis=pe_axis)
        if sig is not None and float(sig) > 0:
            rolled = cv2.GaussianBlur(rolled, (0, 0), sigmaX=float(sig))
        out = (1.0 - float(a)) * out + float(a) * rolled
    return np.clip(out, 0.0, 1.0).astype(np.float32)


@register("motion_pulsation", kind="kspace", dims="2d",
          label_preserving=True, has_detector=True)
def motion_pulsation(arr, *, severity=0.0, rng=None, mask=None,
                     pe_axis=0, offsets=(), alphas=(), bright_pct=95.0, **kw):
    """CSF/vessel pulsation: ghosts of the BRIGHT in-brain compartment.

    Verbatim core of ``add_pulsation``: take the top-``bright_pct`` in-brain
    voxels as a ghost source and add shifted copies along ``pe_axis``. No-op if
    there is no mask / too few bright voxels.
    """
    img = np.asarray(arr, dtype=np.float32)
    if mask is None:
        return img.copy()
    m = np.asarray(mask) > 0
    if m.sum() < 50:
        return img.copy()
    in_brain = img[m]
    if in_brain.size < 50:
        return img.copy()
    thresh = np.percentile(in_brain, bright_pct)
    bright = (img > thresh) & m
    if int(bright.sum()) < 50:
        return img.copy()
    ghost_layer = np.zeros_like(img)
    ghost_layer[bright] = img[bright]
    out = img.copy()
    for offset, a in zip(offsets, alphas):
        out = out + float(a) * np.roll(ghost_layer, int(offset), axis=pe_axis)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


@register("motion_blur", kind="kspace", dims="2d",
          label_preserving=True, has_detector=True)
def motion_blur(arr, *, severity=0.0, rng=None, mask=None,
                preset="linear_motion", sigma_a=1.5, sigma_b=0.5,
                length=6.0, theta=0.0, **kw):
    """Directional motion blur (verbatim ``add_anisotropic_blur`` presets + AA).

    ``preset='linear_motion'`` -> legacy rasterized line PSF (length, theta);
    ``preset='linear_motion_aa'`` -> the anti-aliased line PSF
    (:func:`motion_psf_kernel_aa`) with a smooth length/angle continuum (same
    top-hat profile, just sub-pixel-accurate -- the realistic default for new
    callers); otherwise (``'oriented_gauss'`` / ``'fse_pe'``) -> oriented Gaussian
    (sigma_a, sigma_b, theta). ``fse_pe`` is the same kernel with theta locked to
    the PE axis (the caller passes theta=0 or pi/2).
    """
    img = np.asarray(arr, dtype=np.float32)
    if preset == "linear_motion":
        kernel = motion_psf_kernel(float(length), float(theta))
    elif preset == "linear_motion_aa":
        kernel = motion_psf_kernel_aa(float(length), float(theta))
    else:
        kernel = oriented_gaussian_kernel(float(sigma_a), float(sigma_b), float(theta))
    out = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT).astype(np.float32)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _fse_envelope(n, *, etl, decay, order, asym):
    """Echo-train k-space apodization envelope along the PE axis (length ``n``).

    In a turbo/fast spin-echo readout the ``n`` PE lines are acquired across an
    echo train: each ky line's amplitude is weighted by ``exp(-TE/T2)`` at the
    echo time it was sampled. The *ordering* maps a ky index to an echo index.

    Both modes anchor echo index 0 (envelope weight 1) at the INTEGER DC bin
    ``n//2`` — the bin that ``np.fft.fftshift`` places the zero frequency on, which
    is where the caller multiplies this envelope. So DC always keeps weight 1, the
    MEAN signal is preserved, and the result is a genuine T2-decay BLUR (a low-pass),
    never a DC-destroying high-pass:

    * ``order='centric'`` — the center of k-space is acquired first (shortest TE),
      both edges last -> a SYMMETRIC peaked envelope (clean isotropic-along-PE blur);
    * ``order='linear'``  — DC still carries echo 0, but the decay grows away from DC
      toward the edges with the two halves decaying UNEQUALLY (``asym`` tilts the
      per-half decay RATE), a directional / partial-Fourier-like blur. ``asym`` tilts
      the rate; it does NOT move the TE=0 pivot off DC (doing so made DC the envelope
      minimum -> a high-pass that blacked out the image).

    ``etl`` is the effective echo-train length, ``decay`` the per-echo T2 decay
    rate (echoes/T2). All weights are in (0, 1]. Returns a 1-D float32 array.
    """
    idx = np.arange(n, dtype=np.float64)
    dc = n // 2                          # DC bin after fftshift -> echo 0, weight 1
    half = max(dc, n - 1 - dc, 1)        # farthest DC->edge distance (normalizer)
    dist = np.abs(idx - dc) / half       # 0 at DC, ~1 at the far edge
    if order == "centric":
        echo = dist * float(etl)         # symmetric peaked low-pass
    else:  # linear: DC anchored (echo 0); decay grows away from DC, the two halves
        # decaying at unequal RATES (asym), directional but always DC-preserving.
        a = float(np.clip(asym, -0.9, 0.9))
        side = np.where(idx >= dc, 1.0 + a, 1.0 - a)
        echo = dist * float(etl) * side
    env = np.exp(-float(decay) * echo)
    return env.astype(np.float32)


@register(
    "fse_echo_train",
    kind="kspace",
    severity_range=(0.0, 1.0),
    dims="3d",
    label_preserving=True,
    has_detector=False,
)
def fse_echo_train(arr, *, severity, rng, mask=None, pe_axis=None, etl=None,
                   decay=None, order=None, asym=None, tissue_weight=None, **kw):
    """FSE/TSE echo-train (T2-decay) blur — signal/contrast-dependent PE PSF.

    The dominant clinical T2/PD/FLAIR readout is turbo/fast spin-echo: the
    phase-encode lines are acquired along an echo train at increasing TE, so
    k-space is multiplied by an ``exp(-TE/T2)`` envelope ALONG THE PE AXIS. This
    is an anisotropic, contrast-dependent blur (NOT the isotropic ``blur_2d`` nor
    the through-plane ``anisotropy`` nor the uniform oriented ``fse_pe`` gaussian):

    * **anisotropic** — only the PE axis is apodized, so the PSF smears along PE
      and leaves the readout axis sharp;
    * **tissue-weighted** — long-T2 / bright structures (CSF, edema, WMH) blur
      MORE. A pure global k-space multiply cannot express this, so the apodized
      image is blended back in with a weight that RISES with local brightness
      (``tissue_weight`` in [0,1]; 0 = uniform global blur).

    Over-generates (when a knob is left ``None`` it is sampled from ``rng``,
    scaled by ``severity``): echo-train length ``etl``, per-echo ``decay`` rate,
    ``order`` ('linear' asymmetric vs 'centric' symmetric), ``asym`` train tilt,
    PE axis, and tissue weighting. ``severity<=0`` is identity.
    """
    data = np.asarray(arr, dtype=np.float32)
    sev = float(severity)
    if sev <= 0.0:
        return data.copy()
    if data.ndim == 4:
        out4 = np.empty_like(data, dtype=np.float32)
        for t in range(data.shape[-1]):
            out4[..., t] = fse_echo_train(
                data[..., t], severity=severity, rng=rng, mask=mask,
                pe_axis=pe_axis, etl=etl, decay=decay, order=order, asym=asym,
                tissue_weight=tissue_weight, **kw)
        return out4
    if data.ndim != 3:
        raise ValueError(f"fse_echo_train expects 3D or 4D, got {data.ndim}D")

    ax = int(rng.integers(0, 2)) if pe_axis is None else int(pe_axis)
    etl_v = float(etl) if etl is not None else float(rng.uniform(4.0, 4.0 + 20.0 * sev))
    decay_v = float(decay) if decay is not None else float(rng.uniform(0.05, 0.05 + 0.8 * sev))
    order_v = order if order is not None else ("centric" if rng.random() < 0.45 else "linear")
    asym_v = float(asym) if asym is not None else float(rng.uniform(-0.6, 0.6))
    tw = (float(tissue_weight) if tissue_weight is not None
          else float(rng.uniform(0.0, 1.0)))

    # work in a frame where the PE axis is axis-0 (rows), then move it back
    work = data if ax == 0 else np.swapaxes(data, 0, 1)
    ny, nx, nz = work.shape
    env = _fse_envelope(ny, etl=etl_v, decay=decay_v, order=order_v, asym=asym_v)
    env2 = env[:, None]
    out = np.empty_like(work)
    for z in range(nz):
        sl = work[:, :, z]
        k = np.fft.fftshift(np.fft.fft2(sl))
        k = k * env2  # apodize ALONG the PE axis only -> anisotropic blur
        blurred = np.real(np.fft.ifft2(np.fft.ifftshift(k))).astype(np.float32)
        if tw > 0.0:
            # tissue-weighted: long-T2/bright structures blur more
            lo, hi = float(sl.min()), float(sl.max())
            if hi - lo > 1e-6:
                bright = (sl - lo) / (hi - lo)
            else:
                bright = np.zeros_like(sl)
            w = tw * bright
            out[:, :, z] = (1.0 - w) * sl + w * blurred
        else:
            out[:, :, z] = blurred
    out = out if ax == 0 else np.swapaxes(out, 0, 1)
    out = np.clip(out, 0.0, 1.0)
    return np.ascontiguousarray(out).astype(np.float32)
