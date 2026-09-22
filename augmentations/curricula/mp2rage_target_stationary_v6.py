"""Strict, training-safe wrapper for the stationary MP2RAGE v6 renderer.

The underlying experimental renderer exposes several appearance stages.  This
wrapper deliberately permits only its source-adaptive stationary residual stage
and binds it to the bundled, aggregate-only UNIT1 calibration asset.  It adds a
second, label-free descriptor gate at native resolution: both residual-amplitude
and six-lag distances must improve, while the supported-intensity CDF may not
regress materially.  A rejected candidate is replaced by the exact input in one
pass; this module never retries another seed or weakens a failed gate.

This is still calibration to one unlabelled volume, not a population prior.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from augmentations.curricula import mp2rage_target_style_v2 as _core
from augmentations import numerics


STATIONARY_V6_WRAPPER_VERSION = 1
STATIONARY_V6_DESCRIPTOR_SCHEMA_VERSION = 1
REQUIRED_CORE_RENDER_VERSION = 6

BUNDLED_ASSET_FILENAME = "mp2rage_target_style_unit1_v2.json"
EXPECTED_BUNDLED_ASSET_ID = "mp2rage-target-style-v2-dfc9d07137fff13f"
EXPECTED_BUNDLED_ASSET_SHA256 = (
    "d43cfd6b33a62209d62e146bb3e85ad403c09c4103d63da27ce758bcce082687"
)

MIN_GRADIENT_ALIGNMENT = 0.995
MIN_HIGHPASS_CORRELATION = 0.98
MIN_RESIDUAL_IMPROVEMENT = 1e-6
MIN_LAG_IMPROVEMENT = 1e-6
MAX_CDF_REGRESSION = 1e-3
ISOTROPIC_SPACING_MM = 1.0
ISOTROPIC_SPACING_ATOL_MM = 1e-6
RESIDUAL_LOG_EPSILON = 1e-5

_DESCRIPTOR_SCHEMA: Dict[str, Any] = {
    "version": STATIONARY_V6_DESCRIPTOR_SCHEMA_VERSION,
    "resolution": "native_1mm_isotropic",
    "support": "exact_near_zero_padding_excluded_without_label",
    "normalization": "supported_clip_p0p5_p99p5_then_unit_interval",
    "residual": {
        "sigmas_vox": [0.6, 1.2, 2.4],
        "distance": "rms_log_ratio_to_asset",
        "formula": "sqrt(mean(log((observed+1e-5)/(target+1e-5))^2))",
        "additive_epsilon": RESIDUAL_LOG_EPSILON,
    },
    "lags": {
        "moments": ["lag1_x", "lag1_y", "lag1_z",
                    "lag2_x", "lag2_y", "lag2_z"],
        "distance": "rms_to_asset",
    },
    "cdf": {
        "quantile_count": 33,
        "distance": "rms_to_asset",
    },
    "acceptance": {
        "residual_distance_improvement_strictly_greater_than":
            MIN_RESIDUAL_IMPROVEMENT,
        "lag_distance_improvement_strictly_greater_than":
            MIN_LAG_IMPROVEMENT,
        "maximum_cdf_distance_regression": MAX_CDF_REGRESSION,
    },
}


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _freeze(value: Any) -> Any:
    """Recursively freeze the cached asset so callers cannot poison the cache."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(child)
                                 for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(child) for child in value)
    return value


@lru_cache(maxsize=1)
def load_bundled_stationary_v6_asset() -> Mapping[str, Any]:
    """Load and hard-validate the embedded aggregate calibration.

    The calibration is stored as original JSON bytes in the core Python module,
    independent of the process working directory. Both the original byte digest
    and semantic asset ID are pinned.  The recursively immutable return value is cached once per
    worker process.
    """
    raw = _core.BUNDLED_ASSET_BYTES
    digest = _sha256_bytes(raw)
    if digest != EXPECTED_BUNDLED_ASSET_SHA256:
        raise RuntimeError(
            "bundled MP2RAGE stationary asset SHA-256 mismatch: "
            f"expected {EXPECTED_BUNDLED_ASSET_SHA256}, got {digest}")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("bundled MP2RAGE stationary asset is not valid JSON") from exc
    checked = _core.validate_target_style_v2_asset(payload)
    asset_id = checked.get("asset_id")
    if asset_id != EXPECTED_BUNDLED_ASSET_ID:
        raise RuntimeError(
            "bundled MP2RAGE stationary asset ID mismatch: "
            f"expected {EXPECTED_BUNDLED_ASSET_ID}, got {asset_id!r}")
    spacing = np.asarray(
        checked.get("provenance", {}).get("voxel_sizes_mm", []),
        dtype=np.float64)
    if (spacing.shape != (3,)
            or not np.allclose(spacing, ISOTROPIC_SPACING_MM, rtol=0.0,
                               atol=ISOTROPIC_SPACING_ATOL_MM)):
        raise RuntimeError("bundled MP2RAGE stationary asset is not 1 mm isotropic")
    return _freeze(checked)


def _validate_spacing(voxel_sizes_mm: Sequence[float]) -> Tuple[float, float, float]:
    spacing = np.asarray(voxel_sizes_mm, dtype=np.float64)
    if (spacing.shape != (3,) or not np.isfinite(spacing).all()
            or np.any(spacing <= 0.0)):
        raise ValueError("voxel_sizes_mm must contain three finite positive values")
    if not np.allclose(
            spacing, ISOTROPIC_SPACING_MM, rtol=0.0,
            atol=ISOTROPIC_SPACING_ATOL_MM):
        raise ValueError(
            "stationary v6 augmentation requires 1 mm isotropic input spacing")
    return tuple(float(value) for value in spacing)


def _validate_training_image(image: np.ndarray) -> np.ndarray:
    raw = np.asarray(image)
    if raw.dtype != np.float32:
        raise TypeError("stationary v6 training input must have dtype float32")
    source = _core._require_real_3d(raw, "image")
    if float(np.min(source)) < -1e-6 or float(np.max(source)) > 1.0 + 1e-6:
        raise ValueError("stationary v6 training input must be normalized to [0, 1]")
    return source


def _source_descriptor_context(source: np.ndarray) -> Dict[str, Any]:
    """Freeze the crop and support from the source, never from a proposal."""
    image = _validate_training_image(source)
    maximum = float(np.max(np.abs(image)))
    if maximum <= 1e-8:
        raise ValueError("cannot describe a constant-zero volume")
    threshold = max(np.finfo(np.float32).eps * maximum, 1e-12)
    full_support = np.abs(image) > threshold
    if int(np.count_nonzero(full_support)) < 1024:
        raise ValueError("image has too few supported reconstruction voxels")
    points = np.argwhere(full_support)
    lower = points.min(axis=0)
    upper = points.max(axis=0) + 1
    slices = tuple(slice(int(start), int(stop))
                   for start, stop in zip(lower, upper))
    support = np.asarray(full_support[slices], dtype=bool)
    return {
        "source_shape": tuple(int(value) for value in image.shape),
        "slices": slices,
        "support": support,
        "threshold": float(threshold),
    }


def _descriptor_with_context(
    image: np.ndarray, context: Mapping[str, Any],
) -> Dict[str, Any]:
    source = _validate_training_image(image)
    if tuple(source.shape) != tuple(context["source_shape"]):
        raise ValueError("descriptor candidate shape differs from source shape")
    cropped = np.asarray(source[context["slices"]], dtype=np.float32)
    support = np.asarray(context["support"], dtype=bool)
    if cropped.shape != support.shape:
        raise RuntimeError("frozen descriptor crop/support shape mismatch")
    normalized, _normalization = _core._normalize_supported(cropped, support)
    residual = _core._stationary_residual_summary(normalized, support)
    scales = [float(record["flat_supported_robust_sigma"])
              for record in residual["residual_scales"]]
    lags = [float(value) for value in (
        list(residual["lag1_xyz"]) + list(residual["lag2_xyz"]))]
    probabilities = np.asarray(
        load_bundled_stationary_v6_asset()["style"]
        ["supported_intensity_cdf"]["probabilities"], dtype=np.float64)
    cdf = np.quantile(normalized[support], probabilities).astype(np.float64)
    if len(scales) != 3 or len(lags) != 6 or cdf.shape != (33,):
        raise RuntimeError("native stationary descriptor has an invalid shape")
    values = np.asarray(scales + lags + cdf.tolist(), dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("native stationary descriptor contains non-finite values")
    return {
        "residual_scales": scales,
        "lags": lags,
        "cdf": [float(value) for value in cdf],
    }


def native_style_descriptor(image: np.ndarray) -> Dict[str, Any]:
    """Measure one label-free descriptor on the full native-resolution array."""
    source = _validate_training_image(image)
    return _descriptor_with_context(source, _source_descriptor_context(source))


def paired_native_style_descriptors(
    source: np.ndarray, candidate: np.ndarray,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Measure both arrays with one source-derived crop and support.

    The candidate cannot alter which voxels enter the descriptor.  The context
    is transient and is never serialized into the replay record.
    """
    source_checked = _validate_training_image(source)
    candidate_checked = _validate_training_image(candidate)
    if source_checked.shape != candidate_checked.shape:
        raise ValueError("source and candidate shapes differ")
    context = _source_descriptor_context(source_checked)
    return (
        _descriptor_with_context(source_checked, context),
        _descriptor_with_context(candidate_checked, context),
    )


def _target_descriptor(asset: Mapping[str, Any]) -> Dict[str, Any]:
    residual = asset["style"]["stationary_residual_spectrum"]
    return {
        "residual_scales": [
            float(record["flat_supported_robust_sigma"])
            for record in residual["residual_scales"]],
        "lags": [float(value) for value in (
            list(residual["lag1_xyz"]) + list(residual["lag2_xyz"]))],
        "cdf": [float(value) for value in
                asset["style"]["supported_intensity_cdf"]["quantiles"]],
    }


def _rms(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(array * array)))


def _descriptor_distances(
    measured: Mapping[str, Any], target: Mapping[str, Any],
) -> Dict[str, float]:
    observed_scales = np.asarray(measured["residual_scales"], dtype=np.float64)
    target_scales = np.asarray(target["residual_scales"], dtype=np.float64)
    observed_lags = np.asarray(measured["lags"], dtype=np.float64)
    target_lags = np.asarray(target["lags"], dtype=np.float64)
    observed_cdf = np.asarray(measured["cdf"], dtype=np.float64)
    target_cdf = np.asarray(target["cdf"], dtype=np.float64)
    if (observed_scales.shape != (3,) or target_scales.shape != (3,)
            or observed_lags.shape != (6,) or target_lags.shape != (6,)
            or observed_cdf.shape != (33,) or target_cdf.shape != (33,)):
        raise ValueError("stationary descriptor vectors have invalid shapes")
    if not all(np.isfinite(values).all() for values in (
            observed_scales, target_scales, observed_lags, target_lags,
            observed_cdf, target_cdf)):
        raise ValueError("stationary descriptor vectors must be finite")
    return {
        "residual_log_rms": _rms(
            np.log((observed_scales + RESIDUAL_LOG_EPSILON)
                   / (target_scales + RESIDUAL_LOG_EPSILON))),
        "six_lag_rms": _rms(observed_lags - target_lags),
        "cdf_33_rms": _rms(observed_cdf - target_cdf),
    }


def evaluate_native_descriptor_gate(
    source_descriptor: Mapping[str, Any],
    candidate_descriptor: Mapping[str, Any],
    target_descriptor: Mapping[str, Any],
) -> Dict[str, Any]:
    """Evaluate the fixed R/L-improvement and CDF-regression contract."""
    source_distance = _descriptor_distances(source_descriptor, target_descriptor)
    candidate_distance = _descriptor_distances(candidate_descriptor, target_descriptor)
    residual_improvement = (
        source_distance["residual_log_rms"]
        - candidate_distance["residual_log_rms"])
    lag_improvement = (
        source_distance["six_lag_rms"]
        - candidate_distance["six_lag_rms"])
    cdf_regression = (
        candidate_distance["cdf_33_rms"] - source_distance["cdf_33_rms"])
    residual_pass = residual_improvement > MIN_RESIDUAL_IMPROVEMENT
    lag_pass = lag_improvement > MIN_LAG_IMPROVEMENT
    cdf_pass = cdf_regression <= MAX_CDF_REGRESSION
    return {
        "evaluated": True,
        "passed": bool(residual_pass and lag_pass and cdf_pass),
        "source_distance": source_distance,
        "candidate_distance": candidate_distance,
        "residual_improvement": float(residual_improvement),
        "lag_improvement": float(lag_improvement),
        "cdf_regression": float(cdf_regression),
        "residual_improvement_passed": bool(residual_pass),
        "lag_improvement_passed": bool(lag_pass),
        "cdf_regression_passed": bool(cdf_pass),
        "thresholds": {
            "minimum_residual_improvement_exclusive": MIN_RESIDUAL_IMPROVEMENT,
            "minimum_lag_improvement_exclusive": MIN_LAG_IMPROVEMENT,
            "maximum_cdf_regression_inclusive": MAX_CDF_REGRESSION,
            "residual_log_additive_epsilon": RESIDUAL_LOG_EPSILON,
        },
    }


def _identity_metadata(asset: Mapping[str, Any]) -> Dict[str, Any]:
    core_file = Path(_core.__file__).resolve()
    wrapper_file = Path(__file__).resolve()
    schema_sha = _sha256_bytes(_canonical_json(_DESCRIPTOR_SCHEMA).encode("utf-8"))
    return {
        "wrapper": {
            "id": "mp2rage-target-stationary-v6-training-safe",
            "version": STATIONARY_V6_WRAPPER_VERSION,
            "sha256": _sha256_file(wrapper_file),
        },
        "core": {
            "id": "mp2rage-target-style-v2-stationary-core",
            "render_version": int(_core.TARGET_STYLE_V2_RENDER_VERSION),
            "sha256": _sha256_file(core_file),
            "numerics_sha256": _sha256_file(Path(numerics.__file__).resolve()),
        },
        "descriptor_schema": {
            "version": STATIONARY_V6_DESCRIPTOR_SCHEMA_VERSION,
            "sha256": schema_sha,
        },
        "asset": {
            "id": str(asset["asset_id"]),
            "sha256": EXPECTED_BUNDLED_ASSET_SHA256,
            "calibrated_voxel_sizes_mm": [1.0, 1.0, 1.0],
        },
    }


def stationary_v6_identity_metadata() -> Dict[str, Any]:
    """Validate the installed core/asset and return a path-free run identity."""
    if int(_core.TARGET_STYLE_V2_RENDER_VERSION) != REQUIRED_CORE_RENDER_VERSION:
        raise RuntimeError(
            "stationary wrapper requires target-style core render version 6")
    asset = load_bundled_stationary_v6_asset()
    # JSON round-trip returns an ordinary detached mapping while also proving
    # that the resume identity contains only stable, serializable values.
    return json.loads(_canonical_json(_identity_metadata(asset)))


def _same_bytes(left: np.ndarray, right: np.ndarray) -> bool:
    a, b = np.asarray(left), np.asarray(right)
    return (a.shape == b.shape and a.dtype == b.dtype
            and a.tobytes(order="C") == b.tobytes(order="C"))


def apply_stationary_v6_training_safe(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    seed: int,
    voxel_sizes_mm: Sequence[float] = (1.0, 1.0, 1.0),
    strength: float = 1.0,
    return_record: bool = False,
):
    """Apply one deterministic v6 proposal and veto it unless all gates pass."""
    if int(_core.TARGET_STYLE_V2_RENDER_VERSION) != REQUIRED_CORE_RENDER_VERSION:
        raise RuntimeError(
            "stationary wrapper requires target-style core render version 6")
    spacing = _validate_spacing(voxel_sizes_mm)
    strength = float(strength)
    if not np.isfinite(strength) or strength != 1.0:
        raise ValueError(
            "stationary v6 training-safe strength is locked to exactly 1.0")
    source = _validate_training_image(image)
    label = np.asarray(mask)
    if label.shape != source.shape:
        raise ValueError(f"mask shape {label.shape} does not match image {source.shape}")
    original_label = label.copy()
    asset = load_bundled_stationary_v6_asset()
    identity = _identity_metadata(asset)

    parameters = _core.sample_target_style_v2_parameters(
        asset, seed=int(seed), strength=strength)
    parameters.update({
        "cdf_mix": 0.0,
        "spectrum_mix": 0.0,
        "bias_mix": 0.0,
    })
    if any(float(parameters[key]) != 0.0
           for key in ("cdf_mix", "spectrum_mix", "bias_mix")):
        raise RuntimeError("stationary wrapper enabled a forbidden appearance stage")

    candidate, copied_label, core_record = _core.apply_target_style_v2(
        source, label, asset, parameters=parameters,
        min_gradient_alignment=MIN_GRADIENT_ALIGNMENT,
        min_highpass_correlation=MIN_HIGHPASS_CORRELATION,
        return_record=True)
    if not _same_bytes(copied_label, original_label):
        raise RuntimeError("stationary v6 core modified the label")

    candidate = np.asarray(candidate)
    source_zero = source == 0.0
    candidate_integrity = {
        "shape_preserved": bool(candidate.shape == source.shape),
        "dtype_preserved": bool(candidate.dtype == source.dtype),
        "finite": bool(candidate.shape == source.shape and np.isfinite(candidate).all()),
        "source_zero_values_preserved": False,
        "zero_support_preserved": False,
        "range_preserved": False,
        "core_gradient_guard_passed": bool(
            float(core_record.get("gradient_alignment", -np.inf))
            >= MIN_GRADIENT_ALIGNMENT),
        "core_highpass_guard_passed": bool(
            float(core_record.get("highpass_correlation", -np.inf))
            >= MIN_HIGHPASS_CORRELATION),
        "core_label_free": bool(core_record.get("label_inspected") is False),
    }
    if candidate.shape == source.shape:
        candidate_integrity["source_zero_values_preserved"] = bool(
            np.array_equal(candidate[source_zero], source[source_zero]))
        candidate_integrity["zero_support_preserved"] = bool(
            np.array_equal(candidate == 0.0, source_zero))
        if candidate_integrity["finite"]:
            candidate_integrity["range_preserved"] = bool(
                float(np.min(candidate)) >= -1e-6
                and float(np.max(candidate)) <= 1.0 + 1e-6)

    record: Dict[str, Any] = {
        "identity": identity,
        "seed": int(seed),
        "strength": strength,
        "voxel_sizes_mm": list(spacing),
        "parameters": dict(parameters),
        "core_record": core_record,
        "core_guard_thresholds": {
            "minimum_gradient_alignment": MIN_GRADIENT_ALIGNMENT,
            "minimum_highpass_correlation": MIN_HIGHPASS_CORRELATION,
        },
        "descriptor_gate": {"evaluated": False, "passed": False},
        "candidate_integrity": candidate_integrity,
        "core_zero_record": {
            "new_exact_zero_count": core_record.get("new_exact_zero_count"),
            "new_exact_zero_fraction": core_record.get("new_exact_zero_fraction"),
            "max_new_exact_zero_fraction": core_record.get(
                "max_new_exact_zero_fraction"),
        },
        "accepted": False,
        "identity_reason": None,
        "attempt_count": 1,
        "retry_count": 0,
        "label_inspected": False,
        "mask_byte_identity": True,
    }

    integrity_passed = all(candidate_integrity.values())
    if not integrity_passed:
        record["identity_reason"] = "candidate_integrity_veto"
        result = source.copy()
    elif _same_bytes(candidate, source):
        core_reason = core_record.get("identity_reason")
        if core_reason is None:
            core_reason = core_record.get("residual_stage", {}).get("identity_reason")
        record["identity_reason"] = f"core_identity:{core_reason or 'unchanged'}"
        result = source.copy()
    else:
        try:
            source_descriptor, candidate_descriptor = (
                paired_native_style_descriptors(source, candidate))
            target_descriptor = _target_descriptor(asset)
            gate = evaluate_native_descriptor_gate(
                source_descriptor, candidate_descriptor, target_descriptor)
            gate["source_descriptor"] = source_descriptor
            gate["candidate_descriptor"] = candidate_descriptor
            record["descriptor_gate"] = gate
        except (TypeError, ValueError, RuntimeError, FloatingPointError):
            record["identity_reason"] = "descriptor_unavailable_veto"
            result = source.copy()
        else:
            if gate["passed"]:
                record["accepted"] = True
                result = np.asarray(candidate, dtype=np.float32).copy()
            else:
                record["identity_reason"] = "native_descriptor_veto"
                result = source.copy()

    if not record["accepted"] and not _same_bytes(result, source):
        raise RuntimeError("stationary v6 veto did not return exact source identity")
    if not np.isfinite(result).all():
        raise RuntimeError("stationary v6 wrapper returned non-finite image values")
    if not np.array_equal(result[source_zero], source[source_zero]):
        raise RuntimeError("stationary v6 wrapper changed source exact-zero values")
    if not _same_bytes(copied_label, original_label):
        raise RuntimeError("stationary v6 wrapper changed the label")
    return ((result, copied_label, record) if return_record
            else (result, copied_label))


__all__ = [
    "BUNDLED_ASSET_FILENAME",
    "EXPECTED_BUNDLED_ASSET_ID",
    "EXPECTED_BUNDLED_ASSET_SHA256",
    "ISOTROPIC_SPACING_MM",
    "MAX_CDF_REGRESSION",
    "MIN_GRADIENT_ALIGNMENT",
    "MIN_HIGHPASS_CORRELATION",
    "MIN_LAG_IMPROVEMENT",
    "MIN_RESIDUAL_IMPROVEMENT",
    "RESIDUAL_LOG_EPSILON",
    "REQUIRED_CORE_RENDER_VERSION",
    "STATIONARY_V6_DESCRIPTOR_SCHEMA_VERSION",
    "STATIONARY_V6_WRAPPER_VERSION",
    "apply_stationary_v6_training_safe",
    "evaluate_native_descriptor_gate",
    "load_bundled_stationary_v6_asset",
    "native_style_descriptor",
    "paired_native_style_descriptors",
    "stationary_v6_identity_metadata",
]
