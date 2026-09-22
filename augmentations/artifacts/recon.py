"""deep-learning reconstruction artifact augmentations.

DL / compressed-sensing reconstruction artifacts are often called
"unsynthesizable" because there is no closed-form forward model. The strategy
here is DECOMPOSE + OVER-GENERATE: a DL-recon failure is not one thing, it is a
random, spatially-varying COMBINATION of several independently-known degradations.
By enumerating each constituent and over-generating its parameters (and randomly
gating/weighting each component), the synthetic distribution becomes a SUPERSET
that envelops real DL-recon failures plus a margin — i.e. reality is a subset.

Constituents (each randomly weighted + gated):
  1. over-smoothing / detail loss        (low-pass blur)
  2. high-frequency suppression          (k-space high-freq attenuation)
  3. HALLUCINATED structure              (plausible fake edges / blobs / texture)
  4. local texture replacement           (swap patch texture with neighbor stats)
  5. region inconsistency                (per-region gain/contrast jumps)
  6. over-sharpening / blocking          (unsharp mask + block quantization)

``kind="structure"`` (a learned/structural transform). ``has_detector=False``.
Canonical signature: ``fn(arr, *, severity, rng, mask=None, **kw)``.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from augmentations.registry import register
from augmentations.numerics import iter_4d, head_mask, smooth_random_field


# --------------------------------------------------------------------------- constituents
def _over_smooth(data, sev, rng):
    """Over-smoothing / detail loss via Gaussian low-pass (sev-scaled sigma)."""
    sigma = 0.4 + 3.5 * sev * float(rng.uniform(0.7, 1.3))
    return ndi.gaussian_filter(data, sigma=sigma).astype(np.float32)


def _hf_suppress(data, sev, rng):
    """High-frequency suppression: attenuate outer k-space (radial roll-off)."""
    out = np.empty_like(data, np.float32)
    for z in range(data.shape[2]):
        sl = data[:, :, z]
        k = np.fft.fftshift(np.fft.fft2(sl))
        ny, nx = sl.shape
        yy, xx = np.indices((ny, nx)).astype(np.float32)
        r = np.sqrt(((yy - ny / 2) / (ny / 2 + 1e-6)) ** 2
                    + ((xx - nx / 2) / (nx / 2 + 1e-6)) ** 2)
        # roll-off radius shrinks (more suppression) with severity
        rc = 0.65 - 0.5 * sev * float(rng.uniform(0.7, 1.3))
        rc = max(0.08, rc)
        rolloff = 1.0 / (1.0 + (r / rc) ** 6)  # smooth low-pass
        out[:, :, z] = np.real(np.fft.ifft2(np.fft.ifftshift(k * rolloff)))
    return out


def _local_box(c, shape, half):
    """Axis-aligned LOCAL bounding box (slices + per-axis origin) around center ``c``."""
    lo = [max(0, int(c[a]) - half[a]) for a in range(3)]
    hi = [min(shape[a], int(c[a]) + half[a] + 1) for a in range(3)]
    sl = tuple(slice(lo[a], hi[a]) for a in range(3))
    return sl, lo


def _hallucinate_edge(out, c, sev, rng, head, amp, sign):
    """Insert a FAKE oblique/curved false-sulcus EDGE, BOUNDED to a local box and
    FEATHERED with a Gaussian taper.

    Old behavior was a pencil-straight, axis-aligned slab spanning the WHOLE FOV in
    two axes with a hard binary ``+= sign*amp``. Real DL/CS hallucinations are
    smooth, local, plausible-looking fake anatomy: so the edge is confined to a
    small box, its orientation is a thresholded smooth field (oblique/curved, not
    grid-aligned), and the added intensity is feathered (continuous, not binary).
    """
    shape = out.shape
    # local extent: a modest box that GROWS with severity but never spans the FOV
    half = [int(rng.integers(6, 9 + int(round(10 * sev)))) for _ in range(3)]
    sl, _lo = _local_box(c, shape, half)
    sub = out[sl]
    if sub.size == 0:
        return
    # oblique/curved boundary: a smooth random field thresholded near 0 yields a
    # wavy ridge rather than a straight plane.
    field = smooth_random_field(sub.shape, rng, sigma=max(1.5, min(sub.shape) / 3.0))
    thr = float(rng.uniform(-0.2, 0.2))
    band_half = float(rng.uniform(0.12, 0.30))
    # ridge membership feathered by distance of the field from the threshold
    ridge = np.exp(-((field - thr) / max(band_half, 1e-3)) ** 2).astype(np.float32)
    # radial taper so the edge fades out toward the box border (no hard cut)
    zz, yy, xx = np.indices(sub.shape).astype(np.float32)
    cz = (np.array(sub.shape, np.float32) - 1.0) / 2.0
    rad = np.sqrt(((zz - cz[0]) / max(sub.shape[0], 1)) ** 2
                  + ((yy - cz[1]) / max(sub.shape[1], 1)) ** 2
                  + ((xx - cz[2]) / max(sub.shape[2], 1)) ** 2)
    taper = np.exp(-(rad / 0.5) ** 2).astype(np.float32)
    feather = ridge * taper
    hsub = head[sl]
    out[sl] = sub + (sign * amp) * feather * hsub


def _hallucinate_texture(out, c, sev, rng, head, amp):
    """Insert a FAKE BAND-PASS texture patch (locally-correlated grain), bounded +
    feathered.

    Old behavior added i.i.d. WHITE ``standard_normal`` noise — instantly reads as
    fake (no real anatomy or recon grain is white). Real hallucinated texture is
    spatially correlated; here the noise is band-pass filtered (low-pass minus a
    coarser low-pass) so it has structure, then power-renormed and feathered into a
    local box.
    """
    shape = out.shape
    r = int(rng.integers(4, 6 + int(round(8 * sev))))
    half = [r, r, r]
    sl, _lo = _local_box(c, shape, half)
    sub = out[sl]
    if sub.size == 0:
        return
    white = rng.standard_normal(sub.shape).astype(np.float32)
    sig_lo = float(rng.uniform(0.8, 1.8))
    low = ndi.gaussian_filter(white, sigma=sig_lo)
    band = low - ndi.gaussian_filter(low, sigma=sig_lo * 2.5)  # band-pass
    s = float(band.std())
    if s > 1e-6:
        band = band / s  # unit power so amp controls the grain strength
    zz, yy, xx = np.indices(sub.shape).astype(np.float32)
    cz = (np.array(sub.shape, np.float32) - 1.0) / 2.0
    rad = np.sqrt(((zz - cz[0]) / max(sub.shape[0], 1)) ** 2
                  + ((yy - cz[1]) / max(sub.shape[1], 1)) ** 2
                  + ((xx - cz[2]) / max(sub.shape[2], 1)) ** 2)
    taper = np.exp(-(rad / 0.5) ** 2).astype(np.float32)
    hsub = head[sl]
    out[sl] = sub + (0.3 * amp) * band * taper * hsub


def _hallucinate(data, sev, rng, head):
    """Insert plausible FAKE structure: fake edges, blobs, and texture patches.

    The edge branch is a BOUNDED + FEATHERED oblique/curved false-sulcus (not a
    pencil-straight FOV-spanning slab); the texture branch inserts BAND-PASS
    (locally-correlated) grain (not i.i.d. white). The blob branch is feathered.
    """
    out = data.copy()
    coords = np.argwhere(head)
    if coords.size == 0:
        return out
    scale = float(np.percentile(data[head & (data > 0)], 90)) if (head & (data > 0)).any() else float(data.max())
    n = int(rng.integers(2, 4 + int(round(8 * sev))))  # more fakes at high sev
    zz, yy, xx = np.ogrid[:data.shape[0], :data.shape[1], :data.shape[2]]
    for _ in range(n):
        c = coords[rng.integers(len(coords))]
        kind = rng.random()
        amp = (0.2 + 0.8 * sev) * float(rng.uniform(0.5, 1.5)) * scale
        sign = 1.0 if rng.random() < 0.6 else -1.0
        if kind < 0.4:
            # fake blob (feathered Gaussian, not hard binary)
            r = float(rng.integers(2, 3 + int(round(6 * sev))))
            d2 = (zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2
            blob = np.exp(-d2 / (2.0 * max(r, 1.0) ** 2)).astype(np.float32)
            out += (sign * amp) * blob * head
        elif kind < 0.75:
            # fake edge / ridge: bounded, feathered, oblique/curved
            _hallucinate_edge(out, c, sev, rng, head, amp, sign)
        else:
            # fake texture patch: band-pass (correlated) grain, bounded + feathered
            _hallucinate_texture(out, c, sev, rng, head, amp)
    return out.astype(np.float32)


def _texture_replace(data, sev, rng, head):
    """Local texture replacement: overwrite patches with shifted-neighbor content."""
    out = data.copy()
    coords = np.argwhere(head)
    if coords.size == 0:
        return out
    n = int(rng.integers(1, 3 + int(round(6 * sev))))
    for _ in range(n):
        c = coords[rng.integers(len(coords))]
        half = int(rng.integers(3, 5 + int(round(8 * sev))))
        sl = tuple(slice(max(0, int(c[a]) - half), int(c[a]) + half) for a in range(3))
        patch = data[sl]
        if patch.size == 0:
            continue
        # replace with a rolled (neighbor) copy + mild blur = "wrong" texture
        shift = [int(rng.integers(-half, half + 1)) for _ in range(3)]
        repl = ndi.gaussian_filter(np.roll(patch, shift=shift, axis=(0, 1, 2)), sigma=0.8)
        blend = 0.4 + 0.6 * sev
        out[sl] = (1.0 - blend) * patch + blend * repl
    return out.astype(np.float32)


def _region_inconsistency(data, sev, rng, head):
    """Per-region gain/contrast jumps: a smooth random multiplicative field that
    creates visible discontinuities a faithful recon would never produce."""
    field = smooth_random_field(data.shape, rng, sigma=max(2.0, min(data.shape) / 8.0))
    # quantize into a few regions so boundaries are sharp (inconsistent)
    n_lvl = int(rng.integers(2, 5))
    q = np.round((field * 0.5 + 0.5) * (n_lvl - 1)) / max(1, n_lvl - 1)  # 0..1 levels
    amp = 0.15 + 0.45 * sev
    gain = (1.0 - amp) + 2.0 * amp * q  # per-region gain around 1
    out = data * np.where(head, gain, 1.0).astype(np.float32)
    return out.astype(np.float32)


def _over_sharpen_block(data, sev, rng):
    """Over-sharpening (unsharp mask) + blocking (coarse quantization)."""
    blur = ndi.gaussian_filter(data, sigma=1.0 + 1.5 * float(rng.uniform(0.5, 1.5)))
    amount = 0.5 + 2.0 * sev
    sharp = data + amount * (data - blur)
    # blocking: quantize on a coarse grid (downsample-then-upsample, nearest)
    if rng.random() < 0.7:
        block = int(rng.integers(2, 3 + int(round(4 * sev))))
        if block >= 2:
            small = sharp[::block, ::block, ::block]
            sharp = np.repeat(np.repeat(np.repeat(small, block, 0), block, 1), block, 2)
            sl = tuple(slice(0, data.shape[a]) for a in range(3))
            sharp = sharp[sl]
            if sharp.shape != data.shape:
                pad = [(0, data.shape[a] - sharp.shape[a]) for a in range(3)]
                sharp = np.pad(sharp, pad, mode="edge")
    return sharp.astype(np.float32)


# --------------------------------------------------------------------------- dl_recon
@register(
    "dl_recon",
    kind="structure",
    severity_range=(0.0, 1.0),
    dims="3d",
    label_preserving=True,
    has_detector=False,
)
def dl_recon(arr, *, severity, rng, mask=None, **kw):
    """Deep-learning reconstruction artifact = randomly weighted+gated COMBINATION
    of six decomposed constituents (over-smoothing, high-freq suppression,
    hallucinated structure, local texture replacement, region inconsistency,
    over-sharpening/blocking).

    Each constituent is independently GATED (present or not, biased on by
    severity) and WEIGHTED (random blend), then composited. The dominant overall
    deviation scales with ``severity`` via per-constituent magnitudes and the
    number of active constituents, while WHICH constituents fire and HOW is
    randomized — over-generating the failure-mode space so real DL-recon outputs
    fall inside it.
    """
    data = np.asarray(arr, dtype=np.float32)
    sev = float(severity)
    if sev <= 0.0:
        return data.copy()
    if data.ndim == 4:
        return iter_4d(dl_recon, data, severity, rng, mask, **kw)
    if data.ndim != 3:
        raise ValueError(f"dl_recon expects 3D or 4D, got {data.ndim}D")

    head = head_mask(data, mask)
    out = data.copy()

    # gate probability rises with severity (more constituents fire when worse),
    # but each is at least sometimes active so the space stays well-covered.
    p_on = 0.4 + 0.5 * sev
    constituents = [
        ("over_smooth", lambda d: _over_smooth(d, sev, rng)),
        ("hf_suppress", lambda d: _hf_suppress(d, sev, rng)),
        ("hallucinate", lambda d: _hallucinate(d, sev, rng, head)),
        ("texture_replace", lambda d: _texture_replace(d, sev, rng, head)),
        ("region_inconsistency", lambda d: _region_inconsistency(d, sev, rng, head)),
        ("over_sharpen_block", lambda d: _over_sharpen_block(d, sev, rng)),
    ]
    rng.shuffle(constituents)

    n_active = 0
    for _name, fn in constituents:
        if rng.random() < p_on:
            comp = fn(out)
            # random blend weight; dominant blend grows with severity
            w = (0.3 + 0.6 * sev) * float(rng.uniform(0.5, 1.0))
            out = (1.0 - w) * out + w * comp
            n_active += 1

    # guarantee at least one constituent fires (so output is always "changed")
    if n_active == 0:
        out = _over_smooth(out, max(sev, 0.3), rng)

    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    # A DL/CS recon hallucinates, over-smooths and shifts texture, but it does NOT
    # punch SIGNAL VOIDS -- yet a negative-sign hallucinated blob could drive in-head
    # voxels to ~0 and read as a dropout (false signal_void). Floor in-head voxels to
    # a small fraction of tissue so dark hallucinations darken but never void.
    pos = data[np.isfinite(data) & (data > 0)]
    if pos.size:
        # floor wherever the ORIGINAL had real signal (covers brain AND scalp, not
        # just the brain-mask head) so no hallucinated dark blob reads as a dropout.
        bmed = float(np.median(pos[pos > np.median(pos)]))  # bright-tissue scale
        if bmed > 0:
            floor = 0.06 * bmed
            out = np.where((data > floor) & (out < floor), floor, out)
    return np.maximum(out, 0.0).astype(np.float32)
