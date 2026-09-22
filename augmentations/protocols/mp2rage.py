"""MP2RAGE feature primitives and training parameter sampling.

Registered operators expose tissue transfer, background, and noise controls.
The 3-D renderer shares the complex-UNI air field and FOV padding helpers.
``sample_params`` supplies the optional lower-feature training curriculum.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter

from augmentations.registry import register

MAX_LUT_POINTS = 7


@register(
    "mp2rage_lut",
    kind="intensity",
    severity_range=(0.0, 1.0),
    dims="2d",
    label_preserving=True,
    has_detector=False,
)
def mp2rage_lut(arr, *, severity=0.0, rng=None, mask=None,
                xs=None, ys=None, per_region=True, **kw):
    """Per-tissue intensity transfer curve (the *defining* MP2RAGE primitive).

    A smooth monotone-OR-folded remap built by interpolating the control points
    ``(xs, ys)`` continuously in floating point, applied either in-brain only
    (``per_region=True``) or over all non-air pixels. A non-monotone (folded)
    ``ys`` reproduces the GM/WM/CSF separation and occasional ordering change
    (PSIR / INV2-like) that makes MP2RAGE recognizable.

    The historical 256-bin display LUT is intentionally not reproduced: native
    UNIT1 is commonly 12-bit and resampling creates more unique values, so an
    8-bit intermediate would add visible posterization before noise.
    """
    img = np.asarray(arr, dtype=np.float32)
    if xs is None or ys is None:  # no curve supplied (e.g. a generic registry sweep)
        return img.copy()
    if img.size == 0 or not np.isfinite(img).all():
        return img.copy()
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if (xs.ndim != 1 or ys.ndim != 1 or xs.size < 2 or xs.size != ys.size
            or not np.isfinite(xs).all() or not np.isfinite(ys).all()
            or np.any(np.diff(xs) <= 0.0)):
        return img.copy()
    if xs.shape == ys.shape and np.allclose(xs, ys, rtol=0.0, atol=1e-12):
        return img.copy()
    if (per_region and mask is not None and np.asarray(mask).shape == img.shape
            and np.sum(mask) > 20):
        region = np.asarray(mask) > 0.5
    else:
        positive = img[img > 0]
        scale = float(np.percentile(positive, 99)) if positive.size else 1.0
        region = img > max(0.02, 0.02 * scale)  # leave reconstructed air alone
    values = img[region]
    if values.size < 2:
        return img.copy()
    lo, hi = np.percentile(values, [0.5, 99.5])
    if float(hi - lo) <= 1e-6:
        return img.copy()
    # Do not quantize anatomy through the historical 256-bin display LUT. Native
    # UNIT1 is commonly 12-bit (and resampling creates still more unique values), so
    # direct floating-point interpolation avoids a synthetic posterization cue.
    norm = np.clip((img - float(lo)) / float(hi - lo), 0.0, 1.0)
    mapped = np.interp(norm, xs, ys).astype(np.float32)
    out = img.copy()
    out[region] = mapped[region]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


@register(
    "mp2rage_background",
    kind="noise",
    severity_range=(0.0, 1.0),
    dims="2d",
    label_preserving=True,
    has_detector=False,
)
def mp2rage_background(arr, *, severity=0.0, rng=None, mask=None,
                       thr_frac=0.1, amp=0.5, grain_sg=0.0, field=None,
                       bg_model="complex", ratio_sigma=0.18, ratio_dc=0.0,
                       ratio_inv2_scale=1.0, effective_coils=32.0,
                       fov_zero_prob=0.15,
                       fov_zero_frac=0.25, **kw):
    """Background field (the dominant UNI "tell") — complex UNI or legacy box.

    Low-signal pixels (below ``thr_frac * p99``, outside the brain mask) are
    replaced with a background field of amplitude/center ``amp``. ``field`` is the
    raw background texture; when ``field`` is PASSED IN (production / replay) it is
    used VERBATIM and ``bg_model`` is ignored — this is what keeps
    ``qa_mp2rage_equivalence_test.py`` byte-identical. When ``field is None`` it is
    generated per ``bg_model``:

    * ``bg_model='uniform'`` — the legacy ``uniform[0, amp]`` texture (a flat box;
      one mode of the broadened distribution, byte-identical to the old default
      when the caller pre-draws it the legacy way);
    * ``bg_model='complex'`` (the realistic DEFAULT; ``'ratio'`` is an alias) —
      reconstructs the signed complex MP2RAGE combination
      ``Re(conj(INV1)*INV2) / (|INV1|²+|INV2|²)`` and then applies the usual
      stored-UNI offset. With equal inversion noise and ``N`` effective independent
      complex modes, the normalized air values are exactly ``Beta(N,N)``. This gives
      salt *and* pepper around the stored midpoint; multiplying Rician magnitudes
      would incorrectly generate only the dark half. ``amp`` is the cluster center,
      ``effective_coils`` controls its width, and ``ratio_inv2_scale`` models an
      INV1/INV2 noise imbalance. ``ratio_sigma``/``ratio_dc`` remain accepted for
      old call sites but a common air-noise scale cancels from the normalized ratio.
      With ``field is None`` an INDEPENDENT low-probability FOV-zero patch
      (``fov_zero_prob`` / ``fov_zero_frac``) is also stamped in, reproducing the
      separate edge/FOV-zero mass without baking zeros into the noise peak.

    Grain (``grain_sg``) is applied after generation, coupled INVERSELY to
    amplitude by the caller's sampler.

    Math note: the legacy ``Augmentimage.add_background_field`` corresponds to
    ``bg_model='uniform'`` with the field pre-drawn by the caller; the byte-identity
    guard always passes ``field=`` so it is unaffected by the new default.
    """
    img = np.asarray(arr, dtype=np.float32)
    if img.size == 0 or not np.isfinite(img).all():
        return img.copy()
    fov_zero_mask = None
    if field is None:
        gen = np.random.default_rng(rng)
        if str(bg_model) in ("complex", "ratio"):
            field, fov_zero_mask = _ratio_bg_field(
                gen, img.shape, amp=amp, sigma=ratio_sigma, dc=ratio_dc,
                inv2_scale=ratio_inv2_scale, effective_coils=effective_coils,
                fov_zero_prob=fov_zero_prob,
                fov_zero_frac=fov_zero_frac, protect_mask=mask)
        else:
            field = gen.uniform(0.0, amp, img.shape).astype(np.float32)
    field = np.asarray(field, dtype=np.float32).copy()
    hi = float(np.percentile(img, 99)) if img.max() > 0 else 1.0
    hi = hi if hi > 1e-6 else 1.0
    thr = thr_frac * hi
    gate = img < thr
    if mask is not None and np.asarray(mask).shape == img.shape:
        gate = gate & (np.asarray(mask) <= 0.5)  # never overwrite in-brain
    if gate.sum() < 10:
        return img.copy()
    if grain_sg > 0.1:
        # Reconstruction filtering correlates neighbouring noise, but it must not
        # move the stored-UNI midpoint.  The old max-normalization changed both the
        # mean and the width according to image size; preserve the first two moments.
        pre_mean, pre_std = float(field.mean()), float(field.std())
        field = gaussian_filter(field, grain_sg)
        post_mean, post_std = float(field.mean()), float(field.std())
        if post_std > 1e-8 and pre_std > 1e-8:
            field = (field - post_mean) * (pre_std / post_std) + pre_mean
        else:
            field = field - post_mean + pre_mean
        field = np.clip(field, 0.0, 1.0).astype(np.float32)
    out = img.copy()
    out[gate] = field[gate]
    if fov_zero_mask is not None:
        # Reconstructed support is acquisition geometry, not a semantic-air mask.
        # The generator protects the brain, but may validly crop peripheral scalp.
        out[fov_zero_mask] = 0.0
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _ratio_bg_field(gen, shape, *, amp=0.5, sigma=0.18, dc=0.0,
                    inv2_scale=1.0, effective_coils=32.0,
                    fov_zero_prob=0.15, fov_zero_frac=0.25,
                    protect_mask=None):
    """Complex multi-mode UNI air plus optional oblique FOV padding.

    ``sigma`` and ``dc`` are retained for API compatibility.  A common complex
    Gaussian scale cancels from the MP2RAGE ratio; air is correctly centered by
    the stored-UNI offset rather than by a magnitude DC term.
    """
    del sigma, dc
    field = _complex_uni_air_field(
        gen, shape, center=amp, effective_coils=effective_coils,
        inv2_scale=inv2_scale)
    return field, _edge_zero_mask(
        gen, shape, probability=fov_zero_prob, fraction=fov_zero_frac,
        protect=protect_mask)


def _complex_uni_air_field(gen, shape, *, center=0.5, effective_coils=32.0,
                           inv2_scale=1.0):
    """Sample the stored MP2RAGE UNI background without allocating coil volumes.

    For ``N`` independent complex receiver channels with equal INV1/INV2 noise,

    ``0.5 + Re(sum(conj(s1)*s2)) / sum(|s1|²+|s2|²) ~ Beta(N,N)``.

    That identity gives an exact, memory-efficient draw.  For unequal inversion
    scales we draw the two Gamma-distributed vector norms and their random angular
    dot product, which is also exact but needs only three arrays irrespective of N.
    ``center`` accounts for later robust scaling (0.5 in native stored units; about
    0.55 in the bundled reference after its 99th-percentile normalization).
    """
    n_eff = max(float(effective_coils), 1.0)
    inv2_scale = max(float(inv2_scale), 1e-3)
    scale = 2.0 * float(center)
    if abs(inv2_scale - 1.0) <= 1e-6:
        stored = gen.beta(n_eff, n_eff, size=shape)
    else:
        # A complex N-channel vector has 2N real dimensions. Its squared norm is
        # Gamma(N, scale); the cosine between two random directions is obtained
        # from a symmetric Beta((2N-1)/2, (2N-1)/2) variate.
        r1_sq = gen.gamma(n_eff, 1.0, size=shape)
        r2_sq = gen.gamma(n_eff, inv2_scale ** 2, size=shape)
        angle_shape = max(n_eff - 0.5, 0.5)
        cosine = 2.0 * gen.beta(angle_shape, angle_shape, size=shape) - 1.0
        signed = (np.sqrt(r1_sq * r2_sq) * cosine
                  / (r1_sq + r2_sq + 1e-12))
        stored = 0.5 + signed
    return np.clip(np.asarray(stored, dtype=np.float32) * scale,
                   0.0, 1.0).astype(np.float32)


def _edge_zero_mask(gen, shape, *, probability=0.0, fraction=0.25,
                    protect=None, obliquity=0.35, max_attempts=12):
    """Draw an oblique, face-connected reconstructed-FOV padding region.

    Reoriented MP2RAGE volumes often retain a sizeable zero-valued region outside
    the reconstructed FOV.  It is a separate component from complex air noise and
    therefore must not be represented by broadening or skewing the noise law.
    """
    if probability <= 0.0 or gen.random() >= float(probability):
        return None
    shape = tuple(int(s) for s in shape)
    if not shape or any(s <= 0 for s in shape):
        return None
    frac = float(np.clip(fraction, 1.0 / max(shape), 0.75))
    safe = None
    if protect is not None and np.asarray(protect).shape == shape:
        safe = binary_dilation(np.asarray(protect) > 0.5, iterations=2)

    # A reoriented acquisition is bounded by planes, so padding appears as oblique
    # wedges rather than a synthetic axis-aligned rectangle. Build the plane on the
    # (n-1)-D orthogonal grid to avoid allocating n coordinate volumes.
    for _ in range(max(1, int(max_attempts))):
        axis = int(gen.integers(0, len(shape)))
        low_side = bool(gen.integers(0, 2))
        plane_shape = list(shape)
        plane_shape[axis] = 1
        boundary = np.full(plane_shape, frac, dtype=np.float32)
        tilt_cap = min(float(obliquity), 0.9 * frac + 0.05)
        for other, size in enumerate(shape):
            if other == axis:
                continue
            view = [1] * len(shape)
            view[other] = size
            coord = ((np.arange(size, dtype=np.float32) + 0.5) / size - 0.5).reshape(view)
            boundary += float(gen.uniform(-tilt_cap, tilt_cap)) * coord
        boundary = np.clip(boundary, 0.5 / shape[axis], 0.80)
        view = [1] * len(shape)
        view[axis] = shape[axis]
        along = ((np.arange(shape[axis], dtype=np.float32) + 0.5)
                 / shape[axis]).reshape(view)
        candidate = along < boundary if low_side else along > (1.0 - boundary)
        candidate = np.asarray(candidate, dtype=bool)
        if safe is None or not np.any(candidate & safe):
            return candidate
    return None


@register(
    "mp2rage_grain",
    kind="noise",
    severity_range=(0.0, 1.0),
    dims="2d",
    label_preserving=True,
    has_detector=False,
)
def mp2rage_grain(arr, *, severity=0.0, rng=None, mask=None,
                  grain_sg=0.0, aniso=1.0, **kw):
    """Impose a spatial-correlation length ("grain") on a noise field.

    Deterministic counterpart of ``Augmentimage._apply_noise_grain``: Gaussian-
    smooth with per-axis sigma ``(grain_sg*aniso, grain_sg)`` then renormalize
    back to the input std (preserving noise power). ``grain_sg <= 0`` (or the
    smoothing being a no-op) returns the field unchanged.
    """
    field = np.asarray(arr, dtype=np.float32)
    if grain_sg <= 0.0:
        return field.copy()
    sy = float(grain_sg) * float(aniso)
    sx = float(grain_sg)
    pre = float(np.std(field))
    out = gaussian_filter(field, (sy, sx))
    post = float(np.std(out))
    if post > 1e-8 and pre > 1e-8:
        out = out * (pre / post)
    return out.astype(np.float32)


@register(
    "mp2rage_tissue_noise",
    kind="noise",
    severity_range=(0.0, 1.0),
    dims="2d",
    label_preserving=True,
    has_detector=False,
)
def mp2rage_tissue_noise(arr, *, severity=0.0, rng=None, mask=None,
                         noise=None, **kw):
    """Add a (pre-built, grained) noise field to in-brain pixels only.

    Verbatim composite from the noise-floor block of ``convert_to_mp2rage``.
    ``noise`` must be the SAME shape as ``arr``; it is added where ``mask>0.5``.
    """
    img = np.asarray(arr, dtype=np.float32)
    if mask is None or noise is None:
        return img.copy()
    m = np.asarray(mask) > 0.5
    noise = np.asarray(noise, dtype=np.float32)
    out = img.copy()
    out[m] = out[m] + noise[m]
    return out.astype(np.float32)


def _derive(p: Dict) -> Dict:
    """Build the array-valued knobs (lut_xs/lut_ys) from the scalar sliders."""
    k = int(p.get("lut_k", 5))
    k = max(3, min(MAX_LUT_POINTS, k))
    xs = np.linspace(0.0, 1.0, k)
    ys = np.array([float(p.get(f"lut_y{i}", i / (k - 1))) for i in range(k)],
                  dtype=np.float64)
    return {"lut_xs": xs, "lut_ys": ys}


def sample_params(rng=None) -> Dict:
    """Draw one parameter set from the PRODUCTION distribution.

    Draws a physically valid complex-UNI background plus the deliberately broad
    contrast superset used for training. The normal channel-count mass is centred
    on modern multi-channel head arrays; an occasional low-effective-mode draw
    covers correlated/single-combination reconstructions without reverting to the
    nonphysical magnitude-only box model.

    This sampler drives the explicit 2D explorer/superset. Canonical n-D training
    draws are calibrated separately in ``realistic_acquisition`` and
    ``qc_benign_variations`` but reuse the same signed-complex air primitive. The
    opt-in masker ``mp2rage_lower_feature_fraction`` curriculum also draws from
    this schema, forwarding exactly one physical feature family per selected n-D
    sample so all production ranges are reachable without stacking their extremes.
    """
    g = np.random.default_rng(rng)
    p: Dict = {flag: True for flag in
               ("skull_contrast_on", "brain_on", "bg_on", "noise_on")}
    p["skull_contrast"] = float(g.uniform(0.5, 2.0))
    p["brain_contrast"] = float(g.uniform(0.4, 2.2))

    # LUT: k uniform control points, optional 1-2 interior folds (depth 0.15-0.40)
    k = int(g.integers(3, 8))  # [3,7]
    p["lut_k"] = k
    ys = np.sort(g.uniform(0.0, 1.0, k))
    if g.random() < 0.5:
        for _ in range(int(g.integers(1, 3))):
            j = int(g.integers(1, k - 1))
            ys[j] = max(0.0, ys[j] - float(g.uniform(0.15, 0.40)))
    for i in range(MAX_LUT_POINTS):
        p[f"lut_y{i}"] = float(ys[i]) if i < k else float(ys[-1])
    p["lut_per_region"] = True

    # background: stored midpoint / effective coil count / reconstruction support.
    p["bg_thr_frac"] = float(g.uniform(0.02, 0.18))
    amp = float(g.uniform(0.45, 0.65))
    p["bg_amp"] = amp
    p["bg_grain_sg"] = float(g.uniform(0.0, 0.4))
    p["bg_seed"] = int(g.integers(0, 10000))
    p["bg_model"] = "complex"
    p["bg_effective_coils"] = (int(g.integers(1, 9)) if g.random() < 0.10
                                else int(g.integers(16, 49)))
    p["bg_ratio_inv2_scale"] = float(g.uniform(0.80, 1.25))
    p["bg_fov_zero_prob"] = float(g.uniform(0.0, 0.4))
    p["bg_fov_zero_frac"] = float(g.uniform(0.08, 0.30))

    # noise floor
    p["noise_sigma"] = float(g.uniform(0.01, 0.05))
    p["noise_grain_sg"] = float(g.uniform(0.3, 2.5))
    p["noise_aniso"] = float(g.uniform(1.0, 3.0))
    p["noise_seed"] = int(g.integers(0, 10000))

    p.update(_derive(p))
    return p
