"""Anatomical deformation modes with paired image/mask geometry.

scan_morph_core/morph_modes give an interpretable shape basis (width, height, ventricle, atrophy,
taper, bend, twist, midline-shift, asymmetry, temporal bulge + regional RBF pushes) as closed-form
smooth displacement fields in brain-normalized coords. They are 2D (cv2.remap). The math is n-D, so
this ports it to operate on a 3D VOLUME — and crucially it is meant to deform the LABEL MAP
(`label_synth`), not the image: warping labels with nearest interpolation gives large-scale, named,
crisp region-size / lobe / hemisphere changes with NOTHING to smear (the failure mode of warping the
real image). Ranges are deliberately superset-wide (a dial of ~1 is a strong, bounded warp; the
sampler can push past any real scan).

Public API:
    frame(mask) -> (center, R)
    modes_displacement(dials, shape, center, R) -> (ndim, *shape) pull field
    deform_anatomy(labels, rng, strength=1.0) -> warped labels (nearest; mask = union of brain labels)
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy import ndimage as ndi

# global named modes (axes are generic 0..ndim-1; "lateral/AP/SI" are approximate). NOTE: no
# "midline shift" — it pushed a central slab and could tear the brain in two (removed per review).
_GLOBAL = ["scale", "ventricle", "atrophy", "taper", "bend", "twist", "asym", "bulge"]
_RBF_GRID = 3                                        # GxGxG anchors -> regional pushes

# A strictly-positive Jacobian is the actual no-fold condition for the pull map
# ``x -> x + disp(x)`` used by ``ndi.map_coordinates``.  Keep a small margin above
# zero rather than accepting a numerically singular field that can change sign after
# float32 rounding / finite-difference error.
_MIN_JACOBIAN = 0.05
_MAX_JACOBIAN_HALVINGS = 14
_MAX_CONNECTIVITY_HALVINGS = 8

# Posterior-fossa morphology is deliberately much gentler than the broad named
# anatomy basis above.  The field is specified in physical millimetres on the
# canonical RAS grid: a common 0.5--3.5 mm band, a rare safely-bounded extension
# to 5.5 mm, and a long 22--45 mm correlation/envelope scale.  A stricter local
# Jacobian margin keeps the rare endpoint comfortably away from a fold.
_PF_MORPH_IDENTITY_PROBABILITY = 0.10
_PF_MORPH_RARE_PROBABILITY = 0.05
_PF_MORPH_MIN_JACOBIAN = 0.35
_PF_MORPH_COMMON_MM = (0.5, 3.5)
_PF_MORPH_RARE_MM = (3.5, 5.5)
_PF_MORPH_SCALE_MM = (22.0, 45.0)
_PF_MORPH_VOLUME_RATIO = (0.94, 1.06)


def frame(mask: np.ndarray) -> Tuple[np.ndarray, float]:
    """Brain centroid + effective radius R (px) of an n-D mask."""
    b = np.asarray(mask) > 0.5
    nd = b.ndim
    if not b.any():
        return (np.array(b.shape, float) - 1) / 2.0, max(min(b.shape) / 4.0, 1.0)
    idx = np.array(np.nonzero(b), dtype=np.float32)
    center = idx.mean(axis=1)
    unit = {2: np.pi, 3: 4.0 * np.pi / 3.0}.get(nd, np.pi)   # area/volume of unit ball
    R = max(float((b.sum() / unit) ** (1.0 / nd)), 1.0)
    return center, R


def _coords(shape, center, R):
    g = np.indices(shape, dtype=np.float32)
    u = [(g[i] - center[i]) / R for i in range(len(shape))]
    r = np.sqrt(sum(ui ** 2 for ui in u)).astype(np.float32)
    return u, r


def _zero(shape, nd):
    return [np.zeros(shape, np.float32) for _ in range(nd)]


def minimum_jacobian_determinant(disp: np.ndarray, block_size: int = 8) -> float:
    """Return ``min(det(I + grad(disp)))`` for a 2-D/3-D pull field.

    The determinant is evaluated in slabs so a 256^3 field does not materialize
    nine full-volume derivative arrays at once.  A one-voxel halo makes the slab
    finite differences identical to a whole-volume ``np.gradient`` at internal
    slab boundaries.  Non-finite or malformed fields return ``-inf`` and can never
    be accepted by :func:`_positive_jacobian_displacement`.
    """
    d = np.asarray(disp, dtype=np.float32)
    if d.ndim < 3 or d.shape[0] != d.ndim - 1:
        return float("-inf")
    nd = int(d.shape[0])
    shape = tuple(int(s) for s in d.shape[1:])
    if nd not in (2, 3) or any(s < 2 for s in shape) or not np.isfinite(d).all():
        return float("-inf")

    step = max(1, int(block_size))
    min_det = float("inf")
    for start in range(0, shape[0], step):
        end = min(shape[0], start + step)
        lo = max(0, start - 1)
        hi = min(shape[0], end + 1)
        slab = d[(slice(None), slice(lo, hi), *[slice(None)] * (nd - 1))]
        gradients = [np.gradient(slab[i], edge_order=1) for i in range(nd)]
        core = (slice(start - lo, end - lo), *[slice(None)] * (nd - 1))

        if nd == 2:
            j00 = 1.0 + gradients[0][0][core]
            j01 = gradients[0][1][core]
            j10 = gradients[1][0][core]
            j11 = 1.0 + gradients[1][1][core]
            det = j00 * j11 - j01 * j10
        else:
            j00 = 1.0 + gradients[0][0][core]
            j01 = gradients[0][1][core]
            j02 = gradients[0][2][core]
            j10 = gradients[1][0][core]
            j11 = 1.0 + gradients[1][1][core]
            j12 = gradients[1][2][core]
            j20 = gradients[2][0][core]
            j21 = gradients[2][1][core]
            j22 = 1.0 + gradients[2][2][core]
            det = (
                j00 * (j11 * j22 - j12 * j21)
                - j01 * (j10 * j22 - j12 * j20)
                + j02 * (j10 * j21 - j11 * j20)
            )

        if not np.isfinite(det).all():
            return float("-inf")
        min_det = min(min_det, float(np.min(det)))
    return min_det


def _positive_jacobian_displacement(
    disp: np.ndarray,
    min_jacobian: float = _MIN_JACOBIAN,
    max_halvings: int = _MAX_JACOBIAN_HALVINGS,
) -> Tuple[np.ndarray, float, float]:
    """Scale a field until its pull-map Jacobian is safely positive.

    Returns ``(accepted_field, accepted_scale, min_jacobian)``.  Scaling is
    deterministic and consumes no additional RNG state.  The identity field is the
    guaranteed-safe fallback (Jacobian exactly one), so a folding field is never
    passed to an image or label resampler.
    """
    d = np.asarray(disp, dtype=np.float32)
    if d.ndim < 3 or d.shape[0] != d.ndim - 1 or not np.isfinite(d).all():
        if d.ndim >= 3 and d.shape[0] == d.ndim - 1:
            z = np.zeros_like(d, dtype=np.float32)
        else:
            raise ValueError("displacement must have shape (ndim, *spatial_shape)")
        return z, 0.0, 1.0

    scale = 1.0
    for _ in range(max(0, int(max_halvings)) + 1):
        candidate = d if scale == 1.0 else (d * scale).astype(np.float32)
        min_det = minimum_jacobian_determinant(candidate)
        if np.isfinite(min_det) and min_det >= float(min_jacobian):
            return candidate, scale, min_det
        scale *= 0.5

    identity = np.zeros_like(d, dtype=np.float32)
    return identity, 0.0, 1.0


def _component_count(mask: np.ndarray) -> int:
    """Number of full-connectivity components (8-connected in 2-D, 26 in 3-D)."""
    b = np.asarray(mask, dtype=bool)
    if not b.any():
        return 0
    structure = ndi.generate_binary_structure(b.ndim, b.ndim)
    return int(ndi.label(b, structure=structure)[1])


def _fit_connected_displacement(mask: np.ndarray, disp: np.ndarray, *, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    """Shrink an already sampled field until its nearest-warped mask stays connected.

    This is a validity retry over *scale*, not a new random draw, so image and label
    callers remain reproducible.  If even the progressively shrunken fields fail,
    identity is returned.  A disconnected source mask is rejected rather than
    silently deleting anatomy to manufacture a connected target.
    """
    b = np.asarray(mask, dtype=bool)
    if _component_count(b) != 1:
        raise ValueError("anatomical deformation requires one connected, non-empty source mask")

    base, _, _ = _positive_jacobian_displacement(disp)
    grid = np.indices(b.shape, dtype=np.float32)
    for attempt in range(_MAX_CONNECTIVITY_HALVINGS + 1):
        factor = 0.5 ** attempt
        # ``base`` was just accepted above; avoid a duplicate full-volume determinant pass on the
        # overwhelmingly common first attempt. Revalidate every subsequently scaled candidate.
        candidate = base if attempt == 0 else _positive_jacobian_displacement(base * factor)[0]
        coords = grid + candidate
        warped = ndi.map_coordinates(
            b.astype(np.float32), coords, order=0, mode=mode, cval=0.0
        ).reshape(b.shape) > 0.5
        if _component_count(warped) == 1:
            return candidate, warped

    # Identity is both diffeomorphic and topology preserving for the connected source.
    return np.zeros_like(base, dtype=np.float32), b.copy()


def _mode(name, u, r, R, shape, axis=0, axis2=1, anchor=None, sigma=0.45):
    """One named unit displacement field (list of ndim arrays, pixels at dial=1)."""
    nd = len(u)
    g_center = np.exp(-(r / 0.6) ** 2).astype(np.float32)
    g_rim = np.exp(-((r - 0.9) / 0.4) ** 2).astype(np.float32)
    d = _zero(shape, nd)
    if name == "scale":                                    # stretch along one axis (width/height/depth)
        d[axis] = u[axis] * R
    elif name == "ventricle":                              # pull walls inward at center -> ventricle grows
        for i in range(nd):
            d[i] = -1.6 * u[i] * R * g_center
    elif name == "atrophy":                                # rim inward -> CSF/space widens
        for i in range(nd):
            d[i] = -u[i] * R * g_rim
    elif name == "taper":                                  # axis-scale varies along axis2 (cone/taper)
        d[axis] = u[axis] * u[axis2] * R
    elif name == "bend":                                   # parabolic bow of one axis along another
        d[axis] = (u[axis2] ** 2) * R
    elif name == "twist":                                  # annular rotation in the (axis, axis2) plane
        ann = (r * np.exp(-((r - 0.7) / 0.6) ** 2)).astype(np.float32)
        d[axis] = -u[axis2] * R * ann
        d[axis2] = u[axis] * R * ann
    elif name == "asym":                                   # scale one hemisphere (ramp along axis)
        ramp = (0.5 * (1.0 + np.tanh(2.0 * u[axis]))).astype(np.float32)
        d[axis] = u[axis] * R * ramp
    elif name == "bulge":                                  # localized lobe bulge (temporal-like)
        loc = np.exp(-sum((u[i] - (anchor[i] if anchor else 0.5)) ** 2 for i in range(nd)) / 0.18).astype(np.float32)
        for i in range(nd):
            d[i] = np.sign(u[i]) * R * loc * 0.7
    elif name == "rbf":                                    # regional Gaussian push at an anchor along `axis`
        w = np.exp(-sum((u[i] - anchor[i]) ** 2 for i in range(nd)) / (2.0 * sigma ** 2)).astype(np.float32)
        d[axis] = w * R
    return d


def modes_displacement(dials: List[Tuple], shape, center, R) -> np.ndarray:
    """Sum a list of (name, weight, kwargs) modes into an (ndim, *shape) pull field."""
    nd = len(shape)
    u, r = _coords(shape, center, R)
    total = np.zeros((nd, *shape), np.float32)
    for name, w, kw in dials:
        if abs(w) < 1e-6:
            continue
        fld = _mode(name, u, r, R, shape, **kw)
        for i in range(nd):
            total[i] += w * fld[i]
    return total


def _sample_dials(shape, rng, strength) -> List[Tuple]:
    """Sample MANY modes per draw so each is a DISTINCT anatomy: a few global named shapes PLUS a random
    SUBSET of the GxGxG regional-RBF grid (independent lobe-level bulges/indentations in many places at
    once — the combinatorial lever that makes thousands of distinct brains). The brain-confine window +
    connectivity guard in deform_anatomy keep the result valid, so amplitudes are superset-wide."""
    nd = len(shape)
    dials = []
    amp = lambda hi=1.4: float(rng.uniform(-hi, hi)) * strength      # superset-wide (was 0.9)
    if rng.random() < 0.7:                                 # overall axis proportions
        dials.append(("scale", 0.5 * amp(), {"axis": int(rng.integers(nd))}))
    pool = ["ventricle", "atrophy", "taper", "bend", "twist", "asym", "bulge"]
    k = int(rng.integers(2, 6))                            # 2-5 global named modes per sample (was 1-3)
    for name in rng.choice(np.array(pool, dtype=object), size=min(k, len(pool)), replace=False):
        ax = int(rng.integers(nd)); ax2 = int((ax + 1) % nd)
        anchor = [float(rng.uniform(-0.7, 0.7)) for _ in range(nd)]
        dials.append((str(name), amp(), {"axis": ax, "axis2": ax2, "anchor": anchor}))
    # TILED regional RBF grid: activate a random subset of the GxGxG anchors, each an independent local
    # push (was: a single optional rbf). This is the main distinct-anatomy multiplier.
    axes = [np.linspace(-0.85, 0.85, _RBF_GRID) for _ in range(nd)]
    anchors = list(np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, nd))
    n_active = int(rng.integers(4, min(11, len(anchors) + 1)))
    for ai in rng.choice(len(anchors), size=min(n_active, len(anchors)), replace=False):
        dials.append(("rbf", 0.6 * amp(), {"axis": int(rng.integers(nd)),
                                           "anchor": [float(x) for x in anchors[int(ai)]],
                                           "sigma": float(rng.uniform(0.25, 0.5))}))
    return dials


def build_morph_field(shape, mask, rng, strength: float = 1.0, brain_confine: bool = True) -> np.ndarray:
    """The brain-confined anatomical displacement field used by `deform_anatomy`, exposed so the SAME
    warp can be applied to a real-texture IMAGE (+mask), not only a label map. Returns a finite
    ``(ndim, *shape)`` pull field with a safely-positive Jacobian everywhere; sampled fields that
    would fold are deterministically scaled down."""
    b = np.asarray(mask) > 0.5
    center, R = frame(b)
    disp = modes_displacement(_sample_dials(shape, rng, strength), shape, center, R)
    if brain_confine:
        _, r = _coords(shape, center, R)
        disp = disp * (0.5 * (1.0 - np.tanh((r - 1.2) / 0.13))).astype(np.float32)
    disp, _, _ = _positive_jacobian_displacement(disp)
    return disp


def _sample_posterior_fossa_morph_params(rng) -> Dict[str, object]:
    """Sample the small canonical-RAS posterior-fossa co-warp package.

    Sampling is kept separate from field construction so tests and run manifests
    can audit the actual policy without allocating a 3-D displacement.  Magnitude
    is the maximum requested physical displacement; Jacobian/connectivity guards
    may only reduce it.  A genuine zero-magnitude draw keeps an identity endpoint
    in the curriculum.
    """
    g = np.random.default_rng(rng)
    u = float(g.random())
    if u < _PF_MORPH_IDENTITY_PROBABILITY:
        magnitude_mm = 0.0
    elif u < (_PF_MORPH_IDENTITY_PROBABILITY
              + _PF_MORPH_RARE_PROBABILITY):
        magnitude_mm = float(g.uniform(*_PF_MORPH_RARE_MM))
    else:
        magnitude_mm = float(g.uniform(*_PF_MORPH_COMMON_MM))
    return {
        "magnitude_mm": magnitude_mm,
        "correlation_mm": float(g.uniform(*_PF_MORPH_SCALE_MM)),
        "envelope_fwhm_mm_ras": tuple(
            float(g.uniform(*_PF_MORPH_SCALE_MM)) for _ in range(3)),
    }


def _physical_spacing(voxel_sizes, ndim: int) -> np.ndarray:
    if voxel_sizes is None:
        return np.ones(ndim, dtype=np.float64)
    spacing = np.asarray(tuple(voxel_sizes), dtype=np.float64)
    if spacing.shape != (ndim,) or not np.isfinite(spacing).all() \
            or np.any(spacing <= 0.0):
        raise ValueError(
            f"voxel_sizes must contain {ndim} finite positive values")
    return spacing


def build_posterior_fossa_morph_field(
    shape,
    mask,
    rng,
    *,
    voxel_sizes=None,
    magnitude_mm=None,
    correlation_mm=None,
    envelope_fwhm_mm_ras=None,
) -> np.ndarray:
    """Build a smooth, local posterior/inferior pull field in physical units.

    The input must already obey the canonical RAS axis contract used by the
    posterior-fossa acquisition renderer (low axis-1 is posterior, low axis-2 is
    inferior).  A correlated random vector field is evaluated only in a compact
    crop around the posterior fossa, multiplied by a 22--45 mm Gaussian envelope,
    and normalized to the sampled physical magnitude.  The returned full-frame
    field is finite, fold-safe, and never exceeds the requested millimetre bound.
    """
    spatial_shape = tuple(int(v) for v in shape)
    b = np.asarray(mask) > 0.5
    if len(spatial_shape) != 3 or b.ndim != 3 or b.shape != spatial_shape:
        raise ValueError(
            "posterior-fossa morphology requires a 3-D mask matching shape")
    spacing = _physical_spacing(voxel_sizes, 3)
    params = _sample_posterior_fossa_morph_params(rng)
    if magnitude_mm is None:
        magnitude_mm = params["magnitude_mm"]
    if correlation_mm is None:
        correlation_mm = params["correlation_mm"]
    if envelope_fwhm_mm_ras is None:
        envelope_fwhm_mm_ras = params["envelope_fwhm_mm_ras"]

    magnitude_mm = float(magnitude_mm)
    correlation_mm = float(correlation_mm)
    envelope_fwhm_mm_ras = np.asarray(
        tuple(envelope_fwhm_mm_ras), dtype=np.float64)
    if not np.isfinite(magnitude_mm) or not 0.0 <= magnitude_mm <= 5.5:
        raise ValueError(
            "posterior-fossa morph magnitude_mm must be finite and in [0, 5.5]")
    if (not np.isfinite(correlation_mm)
            or not _PF_MORPH_SCALE_MM[0] <= correlation_mm <= _PF_MORPH_SCALE_MM[1]):
        raise ValueError(
            "posterior-fossa morph correlation_mm must be in [22, 45]")
    if (envelope_fwhm_mm_ras.shape != (3,)
            or not np.isfinite(envelope_fwhm_mm_ras).all()
            or np.any(envelope_fwhm_mm_ras < _PF_MORPH_SCALE_MM[0])
            or np.any(envelope_fwhm_mm_ras > _PF_MORPH_SCALE_MM[1])):
        raise ValueError(
            "posterior-fossa morph envelope FWHM values must be in [22, 45] mm")
    field = np.zeros((3, *spatial_shape), dtype=np.float32)
    if magnitude_mm <= 1e-8 or not b.any():
        return field
    if _component_count(b) != 1:
        raise ValueError(
            "posterior-fossa morphology requires one connected, non-empty source mask")

    # Coarse anatomical frame only: this localizer does not copy the detailed
    # target boundary into the displacement.  Both cerebellar hemispheres remain
    # reachable while the long envelope dies away before anterior/superior cortex.
    idx = np.argwhere(b)
    lo = idx.min(axis=0).astype(np.float64)
    hi = idx.max(axis=0).astype(np.float64)
    extent = np.maximum(hi - lo, 1.0)
    center = lo + np.asarray((0.50, 0.23, 0.24), dtype=np.float64) * extent
    envelope_sigma_mm = envelope_fwhm_mm_ras / 2.354820045
    correlation_sigma_vox = correlation_mm / 2.354820045 / spacing

    # Three envelope standard deviations contain >98% of the local response.
    # Filtering this crop rather than the full 224^3 canvas makes the extra warp
    # inexpensive enough for online synthesis while the final field remains full
    # size for a single image/mask resample.
    radius_vox = np.ceil(3.0 * envelope_sigma_mm / spacing).astype(int)
    crop_lo = np.maximum(0, np.floor(center).astype(int) - radius_vox)
    crop_hi = np.minimum(np.asarray(spatial_shape),
                         np.ceil(center).astype(int) + radius_vox + 1)
    slices = tuple(slice(int(a), int(z)) for a, z in zip(crop_lo, crop_hi))
    local_shape = tuple(int(z - a) for a, z in zip(crop_lo, crop_hi))

    axes = []
    for axis in range(3):
        coordinate_mm = ((np.arange(crop_lo[axis], crop_hi[axis], dtype=np.float32)
                          - float(center[axis])) * float(spacing[axis]))
        axes.append(coordinate_mm / float(envelope_sigma_mm[axis]))
    envelope = np.exp(-0.5 * (
        axes[0][:, None, None] ** 2
        + axes[1][None, :, None] ** 2
        + axes[2][None, None, :] ** 2)).astype(np.float32)

    g = np.random.default_rng(rng)
    local_mm = np.empty((3, *local_shape), dtype=np.float32)
    envelope_sum = max(float(envelope.sum()), 1e-8)
    for axis in range(3):
        noise = g.standard_normal(local_shape).astype(np.float32)
        smooth = ndi.gaussian_filter(
            noise, sigma=np.maximum(correlation_sigma_vox, 0.5),
            mode="reflect").astype(np.float32)
        # Remove a pure bulk translation while retaining smooth local bending and
        # lobar shape variation inside the envelope.
        smooth -= float(np.sum(smooth * envelope) / envelope_sum)
        local_mm[axis] = smooth * envelope

    physical_magnitude = np.sqrt(
        np.sum(local_mm.astype(np.float64) ** 2, axis=0))
    active = envelope >= 0.10
    reference = (float(np.percentile(physical_magnitude[active], 95.0))
                 if np.any(active) else float(physical_magnitude.max(initial=0.0)))
    if not np.isfinite(reference) or reference <= 1e-8:
        return field
    local_mm *= magnitude_mm / reference
    physical_magnitude = np.sqrt(
        np.sum(local_mm.astype(np.float64) ** 2, axis=0))
    limiter = np.minimum(
        1.0, magnitude_mm / np.maximum(physical_magnitude, 1e-12))
    local_mm *= limiter.astype(np.float32)[None, ...]
    for axis in range(3):
        field[(axis, *slices)] = local_mm[axis] / float(spacing[axis])

    field, _, _ = _positive_jacobian_displacement(
        field, min_jacobian=_PF_MORPH_MIN_JACOBIAN)
    return field.astype(np.float32, copy=False)


def _compose_pull_displacements(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Compose two pull fields so ``second`` acts after ``first`` in one resample."""
    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    if first.shape != second.shape or first.ndim != 4 or first.shape[0] != 3:
        raise ValueError("3-D displacement fields must have identical shapes")
    coords = np.indices(first.shape[1:], dtype=np.float32) + second
    composed = second.copy()
    for axis in range(3):
        composed[axis] += ndi.map_coordinates(
            first[axis], coords, order=1, mode="constant", cval=0.0
        ).reshape(first.shape[1:])
    return composed.astype(np.float32, copy=False)


def _fit_posterior_fossa_displacement(
    mask: np.ndarray,
    disp: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bound a local field by fold, topology, and induced-volume checks.

    The posterior-fossa perturbation is supposed to vary local shape, not turn a
    training subject into a globally smaller or larger brain.  Starting from the
    single sampled field, deterministically halve its amplitude until its nearest
    co-warped support is connected and remains within six percent of the incoming
    support volume.  No replacement random draw is made, and identity is the safe
    fallback.
    """
    b = np.asarray(mask, dtype=bool)
    if _component_count(b) != 1:
        raise ValueError(
            "posterior-fossa morphology requires one connected, non-empty source mask")
    d = np.asarray(disp, dtype=np.float32)
    if d.shape != (b.ndim, *b.shape):
        raise ValueError("displacement and posterior-fossa mask shapes do not match")

    base, _, _ = _positive_jacobian_displacement(
        d, min_jacobian=_PF_MORPH_MIN_JACOBIAN)
    grid = np.indices(b.shape, dtype=np.float32)
    source_volume = float(np.count_nonzero(b))
    low_ratio, high_ratio = _PF_MORPH_VOLUME_RATIO
    for attempt in range(_MAX_CONNECTIVITY_HALVINGS + 1):
        factor = 0.5 ** attempt
        candidate = base if attempt == 0 else _positive_jacobian_displacement(
            base * factor, min_jacobian=_PF_MORPH_MIN_JACOBIAN)[0]
        warped = ndi.map_coordinates(
            b.astype(np.float32), grid + candidate, order=0,
            mode="constant", cval=0.0).reshape(b.shape) > 0.5
        volume_ratio = float(np.count_nonzero(warped)) / source_volume
        if (_component_count(warped) == 1
                and low_ratio <= volume_ratio <= high_ratio):
            return candidate, warped

    return np.zeros_like(base, dtype=np.float32), b.copy()


def morph_image(
    img,
    mask,
    rng,
    strength: float = 0.5,
    *,
    posterior_fossa: bool = False,
    voxel_sizes=None,
    posterior_fossa_magnitude_mm=None,
    posterior_fossa_correlation_mm=None,
    posterior_fossa_envelope_fwhm_mm_ras=None,
):
    """Apply a MODERATE smooth anatomical warp to a real-texture (image, mask) pair so the NON-synthetic
    tiers each become a DISTINCT anatomy (different lobe proportions / ventricle size / asymmetry) while
    KEEPING the real texture. Kept gentle (small `strength`) so a smooth relocation, not a smear — the
    large/aggressive anatomy changes stay on the label-driven synthetic tier (which repaints). The mask
    co-warps with NEAREST. Returns (img in [0,1], bool mask)."""
    img = np.asarray(img, np.float32)
    m = np.asarray(mask) > 0.5
    if img.shape != m.shape:
        raise ValueError(f"image/mask shape mismatch: {img.shape} vs {m.shape}")
    if not m.any():
        return np.clip(img, 0.0, 1.0), m
    _, R = frame(m)
    disp = build_morph_field(img.shape, m, rng, strength=strength, brain_confine=True)
    # VISUAL magnitude bound for real texture: keep max in-brain displacement <=0.4R so interpolation
    # remains a gentle relocation rather than a smear. This does not prove Jacobian positivity.
    mag = np.sqrt((disp ** 2).sum(0)); mx = float(mag[m].max())
    if mx > 0.4 * R:
        disp = disp * ((0.4 * R) / mx)
    # A magnitude cap alone is not a no-fold proof. The joint Jacobian/connectivity acceptance below
    # rechecks derivatives after the cap, then shrinks further only if nearest sampling disconnects.
    disp, wm = _fit_connected_displacement(m, disp, mode="constant")
    if bool(posterior_fossa):
        # Define the local field in the already globally-morphed anatomical frame,
        # then compose the two pull maps before touching the image.  This has the
        # semantics of a posterior-fossa co-warp after the existing mild global
        # morph without paying a second interpolation blur.
        local = build_posterior_fossa_morph_field(
            img.shape, wm, rng, voxel_sizes=voxel_sizes,
            magnitude_mm=posterior_fossa_magnitude_mm,
            correlation_mm=posterior_fossa_correlation_mm,
            envelope_fwhm_mm_ras=posterior_fossa_envelope_fwhm_mm_ras)
        if np.any(local):
            local, _local_mask = _fit_posterior_fossa_displacement(wm, local)
            composed = _compose_pull_displacements(disp, local)
            composed, _, _ = _positive_jacobian_displacement(
                composed, min_jacobian=_PF_MORPH_MIN_JACOBIAN)
            # The final guard must cover the *composed* global + local field.  A
            # locally volume-safe perturbation can otherwise inherit an unusually
            # expansive/contractive global draw and still pass connectivity alone.
            disp, wm = _fit_posterior_fossa_displacement(m, composed)
    coords = np.indices(img.shape, dtype=np.float32) + disp
    wimg = ndi.map_coordinates(img, coords, order=1, mode="constant", cval=0.0).reshape(img.shape)
    return np.clip(wimg, 0.0, 1.0).astype(np.float32), wm


def deform_anatomy(labels: np.ndarray, rng: np.random.Generator, strength: float = 1.0,
                   mask: np.ndarray = None, brain_labels=None, fragment_label: int = 0) -> np.ndarray:
    """Apply a random sum of named anatomical modes to the LABEL map (nearest -> crisp, big, clean).

    `brain_labels` (e.g. label_synth.BRAIN_LABELS) enables a CONNECTIVITY GUARD: after warping, the
    brain must stay ONE connected component — any detached fragment is relabelled to `fragment_label`
    (default 0/background). So no deformation can split the brain in two. (Passed by the caller to
    avoid a circular import.)"""
    labels = np.asarray(labels)
    csf_label = int(min(brain_labels)) if brain_labels is not None else 0  # CSF == lowest brain id

    if brain_labels is None:                               # legacy: warp the whole frame (no skull guard)
        b = (labels > 0) if mask is None else (np.asarray(mask) > 0.5)
        if not b.any():
            return labels.copy()
        center, R = frame(b)
        disp = modes_displacement(_sample_dials(labels.shape, rng, strength), labels.shape, center, R)
        disp, _, _ = _positive_jacobian_displacement(disp)
        disp, warped_support = _fit_connected_displacement(b, disp, mode="nearest")
        coords = np.indices(labels.shape, dtype=np.float32) + disp
        warped = ndi.map_coordinates(labels, coords, order=0, mode="nearest").reshape(labels.shape)
        if mask is not None:
            warped = warped.copy()
            warped[(warped != 0) & ~warped_support] = fragment_label
        return warped

    # SKULL-PRESERVING model: a fixed skull/scalp/bg container; the BRAIN morphs inside it. We warp
    # ONLY the brain labels (the modes' displacement blows up at the periphery and would tear the
    # skull), keep the original skull/scalp/bg untouched, default the intracranial space to CSF, then
    # stamp the morphed brain back — limited to a small dilation of the original brain so growth never
    # eats the skull, and atrophy/shrink just opens CSF space (as in reality).
    bl = list(brain_labels)
    b = np.isin(labels, bl)
    if not b.any():
        return labels.copy()
    disp = build_morph_field(labels.shape, b, rng, strength=strength, brain_confine=True)
    disp, warped_support = _fit_connected_displacement(b, disp, mode="constant")
    coords = np.indices(labels.shape, dtype=np.float32) + disp
    brain_only = np.where(b, labels, 0).astype(labels.dtype)
    warped_brain = ndi.map_coordinates(brain_only, coords, order=0, mode="constant").reshape(labels.shape)

    out = labels.copy()                                    # skull / scalp / bg stay ORIGINAL (realistic)
    out[b] = csf_label                                     # intracranial space defaults to CSF
    intra = ndi.binary_dilation(b, iterations=4)           # modest growth allowed, never into the skull
    wb = (warped_brain > 0) & warped_support & intra
    out[wb] = warped_brain[wb]                             # stamp the morphed brain inside the container
    brain2 = np.isin(out, bl)                              # connectivity: keep ONE brain blob
    structure = ndi.generate_binary_structure(brain2.ndim, brain2.ndim)
    cc, n = ndi.label(brain2, structure=structure)
    if n > 1:
        sizes = ndi.sum(np.ones_like(cc), cc, index=range(1, n + 1))
        # detached fragments -> `fragment_label` (the caller passes SCALP, which is NOT a mask label), so
        # a fragment actually LEAVES the brain mask. (Using csf_label kept it IN the mask -> guard was a
        # no-op, since CSF is a mask label.)
        out[brain2 & (cc != int(np.argmax(sizes)) + 1)] = fragment_label
    return out
