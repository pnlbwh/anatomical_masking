"""Post-spatial, source-adaptive MP2RAGE noise-spectrum superset.

The ordinary MP2RAGE renderer runs before joint pose and optional resolution
augmentations.  Those interpolations can erase its final reconstructed texture.
This module supplies a small, training-only stage on the final grid.  It uses no
label geometry: the label is copied byte-for-byte and is never read to construct
the proposal.

Exactly one spectrum phenotype is selected per sample:

* ``white_baseline`` keeps a quiet, near-iid endpoint;
* ``measured_unit1`` surrounds the stationary spectrum measured from the bundled
  unlabelled UNIT1 acquisition; and
* ``spectral_tail`` extrapolates that spectrum conservatively.

Amplitude is a variance budget, not another unconditional noise stack.  The
source's final-grid fine residual is measured first and only the missing variance
toward the sampled target is added.  Exact-zero reconstruction padding is restored
after clipping.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from scipy import ndimage as ndi

from augmentations.numerics import robust_sigma


MP2RAGE_NOISE_SUPERSET_VERSION = 1
MP2RAGE_NOISE_PROFILE_NAMES = (
    "white_baseline", "measured_unit1", "spectral_tail")
MP2RAGE_NOISE_PROFILE_PROBABILITIES = (0.25, 0.60, 0.15)

# Aggregate-only calibration from mp2rage_target_style_unit1_v2.json.  These are
# stationary covariance coefficients, not target voxels, Fourier phase, or a
# serialized spatial field.  The finite FIR below is the compact equivalent of
# the calibrated positive stationary PSD around the measured operating point.
# The full-volume asset value is 0.0464315277.  This nearby literal is the exact
# median produced by the bounded, seven-patch estimator below on that same frozen
# UNIT1 volume; source and proposal are always measured by this one operator.
_REFERENCE_FINE_RESIDUAL_SIGMA = 0.0465345669
_RAW_LAG1_COEFFICIENTS = np.asarray(
    (-0.0005733909, 0.0639608898, -0.0114562487), dtype=np.float64)
_RAW_LAG2_COEFFICIENTS = np.asarray(
    (0.0031921758, 0.0013857344, 0.0003405170), dtype=np.float64)

_PROFILE_BOUNDS: Mapping[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "white_baseline": ((0.0, 0.0), (0.65, 0.90)),
    "measured_unit1": ((0.75, 1.25), (0.90, 1.15)),
    "spectral_tail": ((1.25, 1.75), (1.15, 1.50)),
}
_MEASUREMENT_CROP_MAX = 64
_EPS = 1e-8
_MAX_NEW_ENDPOINT_FRACTION = 1e-4


def _require_float32_volume(image: np.ndarray) -> np.ndarray:
    raw = np.asarray(image)
    if raw.ndim != 3:
        raise ValueError(f"image must be 3-D, got shape {raw.shape}")
    if raw.dtype != np.float32:
        raise TypeError("image must have dtype float32")
    if not np.isfinite(raw).all():
        raise ValueError("image contains NaN or infinite values")
    if float(np.min(raw)) < -1e-6 or float(np.max(raw)) > 1.0 + 1e-6:
        raise ValueError("image must be normalized to [0, 1]")
    return raw


def _validate_spacing(voxel_sizes_mm: Sequence[float]) -> tuple[float, float, float]:
    try:
        spacing = tuple(float(value) for value in voxel_sizes_mm)
    except (TypeError, ValueError) as exc:
        raise ValueError("voxel_sizes_mm must be exactly (1, 1, 1)") from exc
    if (spacing != (1.0, 1.0, 1.0)
            or not all(np.isfinite(value) for value in spacing)):
        raise ValueError(
            "MP2RAGE noise superset v1 requires voxel_sizes_mm exactly (1, 1, 1)")
    return spacing


def _validate_permutation(permutation: Sequence[int]) -> tuple[int, int, int]:
    try:
        resolved = tuple(int(value) for value in permutation)
    except (TypeError, ValueError) as exc:
        raise ValueError("axis_permutation must be a permutation of (0, 1, 2)") from exc
    if resolved not in {
            (0, 1, 2), (0, 2, 1), (1, 0, 2),
            (1, 2, 0), (2, 0, 1), (2, 1, 0)}:
        raise ValueError("axis_permutation must be a permutation of (0, 1, 2)")
    return resolved


def _stationary_fir_preserve_moments(
        field: np.ndarray, *, strength: float,
        axis_permutation: Sequence[int] = (0, 1, 2)) -> np.ndarray:
    """Apply the calibrated lag-1/lag-2 FIR and restore exact mean/SD.

    ``strength=0`` is an exact copy endpoint.  Reflect boundaries avoid the
    wraparound seam produced by ``np.roll``.  The coefficients remain far inside
    the positive/stable FIR envelope even at the 1.75 tail endpoint.
    """
    source = np.asarray(field, dtype=np.float32)
    if source.ndim != 3 or not np.isfinite(source).all():
        raise ValueError("field must be a finite 3-D array")
    strength = float(strength)
    if not np.isfinite(strength) or not 0.0 <= strength <= 1.75:
        raise ValueError("strength must be finite and in [0, 1.75]")
    permutation = _validate_permutation(axis_permutation)
    if strength == 0.0 or source.size < 2:
        return source.copy()

    lag1 = _RAW_LAG1_COEFFICIENTS[np.asarray(permutation, dtype=np.int64)]
    lag2 = _RAW_LAG2_COEFFICIENTS[np.asarray(permutation, dtype=np.int64)]
    output = source.copy()
    scratch = np.empty_like(source)
    for axis in range(3):
        ndi.correlate1d(
            source, np.asarray((1.0, 0.0, 1.0), dtype=np.float32),
            axis=axis, output=scratch, mode="reflect")
        output += np.float32(0.5 * strength * lag1[axis]) * scratch
        ndi.correlate1d(
            source, np.asarray((1.0, 0.0, 0.0, 0.0, 1.0), dtype=np.float32),
            axis=axis, output=scratch, mode="reflect")
        output += np.float32(0.5 * strength * lag2[axis]) * scratch

    source_mean, source_sd = float(source.mean()), float(source.std())
    output_mean, output_sd = float(output.mean()), float(output.std())
    if source_sd <= _EPS or output_sd <= _EPS:
        return source.copy()
    output = ((output - output_mean) * (source_sd / output_sd)
              + source_mean)
    return np.asarray(output, dtype=np.float32)


def _sample_noise_profile(rng: np.random.Generator) -> dict:
    draw = float(rng.random())
    if draw < MP2RAGE_NOISE_PROFILE_PROBABILITIES[0]:
        name = "white_baseline"
    elif draw < sum(MP2RAGE_NOISE_PROFILE_PROBABILITIES[:2]):
        name = "measured_unit1"
    else:
        name = "spectral_tail"
    strength_bounds, multiplier_bounds = _PROFILE_BOUNDS[name]
    strength = (strength_bounds[0] if strength_bounds[0] == strength_bounds[1]
                else float(rng.uniform(*strength_bounds)))
    multiplier = float(rng.uniform(*multiplier_bounds))
    permutation = tuple(int(value) for value in rng.permutation(3))
    return {
        "name": name,
        "selection_draw": draw,
        "strength": float(strength),
        "target_sigma_multiplier": multiplier,
        "axis_permutation": permutation,
    }


def _support_bounds(support: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Return tight bounds using three tiny projections, never ``argwhere``."""
    bounds = []
    for axis in range(3):
        other_axes = tuple(index for index in range(3) if index != axis)
        positions = np.flatnonzero(np.any(support, axis=other_axes))
        if positions.size == 0:
            return ()
        bounds.append((int(positions[0]), int(positions[-1]) + 1))
    return tuple(bounds)


def _measurement_patch_slices(
        support: np.ndarray) -> tuple[tuple[slice, slice, slice], ...]:
    """Return bounded corner and centre patches over the supported FOV."""
    bounds = _support_bounds(support)
    if not bounds:
        return ()
    axis_rows = []
    centre = []
    for start, stop in bounds:
        width = int(stop - start)
        size = min(_MEASUREMENT_CROP_MAX, width)
        edge_starts = (int(start), int(stop - size))
        axis_rows.append(tuple(dict.fromkeys(edge_starts)))
        centre.append(int(round(0.5 * (start + stop - size))))
    candidates = []
    for first in axis_rows[0]:
        for second in axis_rows[1]:
            for third in axis_rows[2]:
                candidates.append((first, second, third))
    candidates.append(tuple(centre))
    unique = tuple(dict.fromkeys(candidates))
    sizes = tuple(min(_MEASUREMENT_CROP_MAX, stop - start)
                  for start, stop in bounds)
    return tuple(tuple(slice(origin[axis], origin[axis] + sizes[axis])
                       for axis in range(3)) for origin in unique)


def _support_normalized_gaussian(values, support, sigma):
    weights = np.asarray(support, dtype=np.float32)
    coverage = ndi.gaussian_filter(
        weights, sigma=sigma, mode="constant", cval=0.0)
    smooth = ndi.gaussian_filter(
        np.asarray(values, dtype=np.float32) * weights,
        sigma=sigma, mode="constant", cval=0.0)
    smooth /= np.maximum(coverage, np.float32(1e-5))
    return np.asarray(smooth, dtype=np.float32), coverage


def _patch_context(image, support, slices, lower, upper):
    cropped = np.asarray(image[slices], dtype=np.float32)
    cropped_support = np.asarray(support[slices], dtype=bool)
    if int(np.count_nonzero(cropped_support)) < 1024:
        return None
    span = float(upper - lower)
    normalized = np.zeros(cropped.shape, dtype=np.float32)
    normalized[cropped_support] = np.clip(
        (cropped[cropped_support] - float(lower)) / span, 0.0, 1.0)
    base_smooth, base_coverage = _support_normalized_gaussian(
        normalized, cropped_support, 0.8)
    gradients = np.gradient(base_smooth.astype(np.float32))
    gradient_magnitude = np.sqrt(sum(value * value for value in gradients))
    interior = cropped_support & (base_coverage >= 0.995)
    if int(np.count_nonzero(interior)) < 1024:
        interior = cropped_support & (base_coverage >= 0.90)
    interior_values = normalized[interior]
    if interior_values.size < 1024:
        return None
    q02, q98 = np.quantile(interior_values, (0.02, 0.98))
    gradient_cut = float(np.quantile(gradient_magnitude[interior], 0.45))
    flat = (interior & (normalized >= float(q02)) & (normalized <= float(q98))
            & (gradient_magnitude <= gradient_cut))
    if int(np.count_nonzero(flat)) < 1024:
        flat = interior
    smooth, coverage = _support_normalized_gaussian(
        normalized, cropped_support, 0.6)
    valid = flat & (coverage >= 0.995)
    if int(np.count_nonzero(valid)) < 1024:
        valid = flat
    sigma = robust_sigma((normalized - smooth)[valid])
    return {
        "slices": slices,
        "support": cropped_support,
        "flat": flat,
        "lower": float(lower),
        "upper": float(upper),
        "span": span,
        "source_sigma": float(sigma),
    }


def _fine_residual_sigma_and_context(image: np.ndarray):
    """Measure source sigma and freeze one bounded source-derived context."""
    maximum = float(np.max(np.abs(image)))
    threshold = max(np.finfo(np.float32).eps * maximum, 1e-12)
    support = np.abs(image) > threshold
    if int(np.count_nonzero(support)) < 1024:
        return 0.0, 0.0, None
    steps = tuple(max(1, int(np.ceil(size / 64.0))) for size in image.shape)
    sampled = image[::steps[0], ::steps[1], ::steps[2]]
    sampled_support = support[::steps[0], ::steps[1], ::steps[2]]
    values = np.asarray(sampled[sampled_support], dtype=np.float64)
    if values.size < 1024:
        return 0.0, 0.0, None
    lower, upper = np.quantile(values, (0.005, 0.995))
    span = float(upper - lower)
    if not np.isfinite(span) or span <= _EPS:
        return 0.0, 0.0, None
    contexts = []
    for slices in _measurement_patch_slices(support):
        context = _patch_context(image, support, slices, lower, upper)
        if context is not None:
            contexts.append(context)
    if not contexts:
        return 0.0, 0.0, None
    median_sigma = float(np.median(
        [context["source_sigma"] for context in contexts]))
    selected = min(
        contexts, key=lambda context: abs(context["source_sigma"] - median_sigma))
    return float(selected["source_sigma"]), span, selected


def _residual_sigma_with_context(values, context, *, image_values):
    cropped = np.asarray(values[context["slices"]], dtype=np.float32)
    support = np.asarray(context["support"], dtype=bool)
    if image_values:
        normalized = np.zeros(cropped.shape, dtype=np.float32)
        normalized[support] = np.clip(
            (cropped[support] - context["lower"]) / context["span"],
            0.0, 1.0)
        cropped = normalized
    smooth, coverage = _support_normalized_gaussian(cropped, support, 0.6)
    valid = np.asarray(context["flat"], dtype=bool) & (coverage >= 0.995)
    if int(np.count_nonzero(valid)) < 1024:
        valid = np.asarray(context["flat"], dtype=bool)
    return robust_sigma((cropped - smooth)[valid])


def apply_mp2rage_noise_superset_v1(
        image: np.ndarray, label: np.ndarray, *, seed: int,
        voxel_sizes_mm: Sequence[float] = (1.0, 1.0, 1.0),
        return_record: bool = False):
    """Add one final-grid MP2RAGE residual phenotype without changing ``label``."""
    source = _require_float32_volume(image)
    spacing = _validate_spacing(voxel_sizes_mm)
    label_array = np.asarray(label)
    if label_array.shape != source.shape:
        raise ValueError("label shape must match image shape")
    label_copy = label_array.copy()
    source_zero = source == 0.0

    rng = np.random.default_rng(int(seed))
    profile = _sample_noise_profile(rng)
    target_sigma = (_REFERENCE_FINE_RESIDUAL_SIGMA
                    * float(profile["target_sigma_multiplier"]))
    source_sigma, source_span, measurement_context = (
        _fine_residual_sigma_and_context(source))
    variance_gap = max(target_sigma * target_sigma - source_sigma * source_sigma, 0.0)
    record = {
        "version": MP2RAGE_NOISE_SUPERSET_VERSION,
        "profile": profile,
        "source_normalized_sigma": float(source_sigma),
        "target_normalized_sigma": float(target_sigma),
        "source_normalization_span": float(source_span),
        "voxel_sizes_mm": spacing,
        "variance_gap": float(variance_gap),
        "identity_reason": None,
        "target_voxels_used": False,
        "target_fft_or_phase_used": False,
        "exact_zero_padding_preserved": True,
    }
    if source_span <= _EPS or variance_gap <= 0.0:
        record.update({
            "identity_reason": ("insufficient_supported_dynamic_range"
                                if source_span <= _EPS
                                else "source_at_or_above_target"),
            "added_normalized_sigma": 0.0,
            "requested_added_normalized_sigma": 0.0,
            "added_image_scale": 0.0,
            "effective_image_scale": 0.0,
            "achieved_normalized_sigma": float(source_sigma),
            "variance_gap_closed_fraction": 0.0,
            "integrity_backtrack": 0.0,
            "headroom_limited_fraction": 0.0,
            "new_endpoint_fraction": 0.0,
        })
        result = source.copy()
        return ((result, label_copy, record) if return_record
                else (result, label_copy))

    white = rng.standard_normal(source.shape).astype(np.float32)
    white -= float(white.mean())
    white_sd = float(white.std())
    if white_sd > _EPS:
        white /= white_sd
    residual = _stationary_fir_preserve_moments(
        white, strength=float(profile["strength"]),
        axis_permutation=profile["axis_permutation"])
    unit_sigma = _residual_sigma_with_context(
        residual, measurement_context, image_values=False)
    requested_added_normalized_sigma = float(np.sqrt(variance_gap))
    added_image_scale = (requested_added_normalized_sigma * source_span
                         / max(unit_sigma, _EPS))

    source_endpoints = (source == 0.0) | (source == 1.0)
    supported_count = max(1, int(np.count_nonzero(~source_zero)))
    raw_delta = np.float32(added_image_scale) * residual
    delta = np.maximum(
        raw_delta, np.float32(-0.95) * source)
    delta = np.minimum(
        delta, np.float32(0.95) * (np.float32(1.0) - source))
    limited = (delta != raw_delta) & ~source_zero
    candidate = np.asarray(source + delta, dtype=np.float32)
    candidate[source_zero] = source[source_zero]
    new_endpoints = ((candidate == 0.0) | (candidate == 1.0)) & ~source_endpoints
    new_endpoint_fraction = (float(np.count_nonzero(new_endpoints))
                             / supported_count)
    achieved_sigma = _residual_sigma_with_context(
        candidate, measurement_context, image_values=True)
    achieved_variance_gap = max(
        achieved_sigma * achieved_sigma - source_sigma * source_sigma, 0.0)
    gap_closed = min(1.0, achieved_variance_gap / max(variance_gap, _EPS))
    effective_added_sigma = float(np.sqrt(achieved_variance_gap))
    headroom_limited_fraction = (float(np.count_nonzero(limited))
                                 / supported_count)
    if new_endpoint_fraction > _MAX_NEW_ENDPOINT_FRACTION:
        candidate = source.copy()
        new_endpoint_fraction = 0.0
        achieved_sigma = source_sigma
        effective_added_sigma = 0.0
        gap_closed = 0.0
        record["identity_reason"] = "endpoint_integrity_guard"

    record.update({
        "unit_residual_highpass_sigma": float(unit_sigma),
        "requested_added_normalized_sigma": requested_added_normalized_sigma,
        "added_normalized_sigma": effective_added_sigma,
        "added_image_scale": float(added_image_scale),
        "effective_image_scale": (float(added_image_scale)
                                  if record["identity_reason"] is None else 0.0),
        "achieved_normalized_sigma": float(achieved_sigma),
        "variance_gap_closed_fraction": float(gap_closed),
        "integrity_backtrack": (1.0 if record["identity_reason"] is None else 0.0),
        "headroom_limited_fraction": float(headroom_limited_fraction),
        "new_endpoint_fraction": float(new_endpoint_fraction),
    })
    return ((candidate, label_copy, record) if return_record
            else (candidate, label_copy))


__all__ = [
    "MP2RAGE_NOISE_SUPERSET_VERSION",
    "MP2RAGE_NOISE_PROFILE_NAMES",
    "MP2RAGE_NOISE_PROFILE_PROBABILITIES",
    "apply_mp2rage_noise_superset_v1",
]
