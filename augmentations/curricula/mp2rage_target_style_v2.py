"""Support-normalized, low-capacity MP2RAGE target-style augmentations.

Version 1 of the experimental target-style asset summarized the whole raw UNIT1
array.  That volume contains about 25 percent exact-zero reconstruction padding,
so its whole-volume CDF and Fourier envelope inadvertently encoded padding.  This
module is a deliberately separate v2 experiment so v1 records remain replayable.

The v2 extractor uses exact/near-zero voxels only to *exclude* reconstruction
padding while statistics are measured.  It serializes no support map, crop, target
voxel, Fourier coefficient, phase, or regional statistic.  The retained style is
limited to:

* a one-dimensional CDF of supported intensities;
* multi-scale local residual amplitudes and six lag correlations (a very
  low-capacity stationary noise-spectrum proxy);
* one capped scalar describing low-frequency modulation.

Application is spatially global and image-only.  Its spectral stage multiplies the
source FFT by a positive real transfer function, so it retains source phase.  Bias
and residual fields are newly sampled from the caller's seed.  Two image-only
guards (gradient orientation and high-pass correlation) backtrack toward the
source.  The supplied label is copied byte-for-byte and is never inspected.

This remains a single-volume, single-site appearance calibration.  It is an
opt-in ablation, not a population prior and not segmentation supervision.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy import fft as sfft
from scipy import ndimage as ndi

from augmentations.numerics import robust_sigma


# Aggregate calibration embedded so run_config.json is the only packaged JSON.
BUNDLED_ASSET_BYTES = (
    b'{\r\n'
    b'  "asset_id": "mp2rage-target-style-v2-dfc9d07137fff13f",\r\n'
    b'  "asset_type": "mp2rage_unlabelled_target_style_support_normalized",\r\n'
    b'  "calibration_scope": "single-volume appearance calibration; not a population prior, not segmentation evidence",\r\n'
    b'  "extractor_version": 2,\r\n'
    b'  "normalization": {\r\n'
    b'    "lower_percentile": 252.0160369873047,\r\n'
    b'    "method": "supported_voxels_clip_p0p5_p99p5_then_unit_interval",\r\n'
    b'    "padding_excluded": true,\r\n'
    b'    "upper_percentile": 3847.025634765625\r\n'
    b'  },\r\n'
    b'  "privacy_contract": {\r\n'
    b'    "aggregate_only": true,\r\n'
    b'    "target_complex_coefficients_stored": false,\r\n'
    b'    "target_fourier_phase_stored": false,\r\n'
    b'    "target_mask_used": false,\r\n'
    b'    "target_regional_statistics_stored": false,\r\n'
    b'    "target_support_geometry_stored": false,\r\n'
    b'    "target_voxels_stored": false\r\n'
    b'  },\r\n'
    b'  "provenance": {\r\n'
    b'    "source_id": "sub-conOC971_ses-02_desc-reoriented_UNIT1.nii.gz",\r\n'
    b'    "source_sha256": "2b87034b1a181756a296b4f7b20dce8e31abb2a4176015cf01faf83590d80847",\r\n'
    b'    "source_shape": [\r\n'
    b'      176,\r\n'
    b'      240,\r\n'
    b'      256\r\n'
    b'    ],\r\n'
    b'    "voxel_sizes_mm": [\r\n'
    b'      1.0,\r\n'
    b'      1.0,\r\n'
    b'      1.0\r\n'
    b'    ]\r\n'
    b'  },\r\n'
    b'  "schema_version": 2,\r\n'
    b'  "style": {\r\n'
    b'    "low_frequency_bias": {\r\n'
    b'      "estimator": "broad_to_medium_support_normalized_log_ratio",\r\n'
    b'      "measured_log_robust_sigma": 0.16651574331521987,\r\n'
    b'      "random_field_scale_fraction_range": [\r\n'
    b'        0.08,\r\n'
    b'        0.18\r\n'
    b'      ],\r\n'
    b'      "renderer_log_sigma_cap": 0.08,\r\n'
    b'      "target_bias_field_stored": false\r\n'
    b'    },\r\n'
    b'    "stationary_residual_spectrum": {\r\n'
    b'      "estimator": "support-normalized local Gaussian residuals on low-gradient supported voxels",\r\n'
    b'      "flat_selection_quantile": 0.45,\r\n'
    b'      "flat_supported_fraction_within_crop": 0.35195199360687673,\r\n'
    b'      "lag1_xyz": [\r\n'
    b'        -0.177573811588377,\r\n'
    b'        -0.10789537306812971,\r\n'
    b'        -0.18874263077229106\r\n'
    b'      ],\r\n'
    b'      "lag2_xyz": [\r\n'
    b'        0.013966237655834381,\r\n'
    b'        0.00024031941468082547,\r\n'
    b'        0.013073427719616173\r\n'
    b'      ],\r\n'
    b'      "residual_scales": [\r\n'
    b'        {\r\n'
    b'          "flat_supported_robust_sigma": 0.046431527683138844,\r\n'
    b'          "sigma_vox": 0.6\r\n'
    b'        },\r\n'
    b'        {\r\n'
    b'          "flat_supported_robust_sigma": 0.06293015590310097,\r\n'
    b'          "sigma_vox": 1.2\r\n'
    b'        },\r\n'
    b'        {\r\n'
    b'          "flat_supported_robust_sigma": 0.06856715793907642,\r\n'
    b'          "sigma_vox": 2.4\r\n'
    b'        }\r\n'
    b'      ],\r\n'
    b'      "spectral_capacity": "three residual amplitudes plus lag1/lag2 per axis",\r\n'
    b'      "target_fft_used": false\r\n'
    b'    },\r\n'
    b'    "supported_intensity_cdf": {\r\n'
    b'      "probabilities": [\r\n'
    b'        0.0,\r\n'
    b'        0.03125,\r\n'
    b'        0.0625,\r\n'
    b'        0.09375,\r\n'
    b'        0.125,\r\n'
    b'        0.15625,\r\n'
    b'        0.1875,\r\n'
    b'        0.21875,\r\n'
    b'        0.25,\r\n'
    b'        0.28125,\r\n'
    b'        0.3125,\r\n'
    b'        0.34375,\r\n'
    b'        0.375,\r\n'
    b'        0.40625,\r\n'
    b'        0.4375,\r\n'
    b'        0.46875,\r\n'
    b'        0.5,\r\n'
    b'        0.53125,\r\n'
    b'        0.5625,\r\n'
    b'        0.59375,\r\n'
    b'        0.625,\r\n'
    b'        0.65625,\r\n'
    b'        0.6875,\r\n'
    b'        0.71875,\r\n'
    b'        0.75,\r\n'
    b'        0.78125,\r\n'
    b'        0.8125,\r\n'
    b'        0.84375,\r\n'
    b'        0.875,\r\n'
    b'        0.90625,\r\n'
    b'        0.9375,\r\n'
    b'        0.96875,\r\n'
    b'        1.0\r\n'
    b'      ],\r\n'
    b'      "quantiles": [\r\n'
    b'        0.0,\r\n'
    b'        0.20166555047035217,\r\n'
    b'        0.30764040350914,\r\n'
    b'        0.3518759310245514,\r\n'
    b'        0.3788500726222992,\r\n'
    b'        0.3983275890350342,\r\n'
    b'        0.4136325418949127,\r\n'
    b'        0.42642706632614136,\r\n'
    b'        0.43726521730422974,\r\n'
    b'        0.44701263308525085,\r\n'
    b'        0.4561886787414551,\r\n'
    b'        0.4645336866378784,\r\n'
    b'        0.4726017117500305,\r\n'
    b'        0.48037537932395935,\r\n'
    b'        0.4878893792629242,\r\n'
    b'        0.49540334939956665,\r\n'
    b'        0.5029173493385315,\r\n'
    b'        0.5104313492774963,\r\n'
    b'        0.5182222723960876,\r\n'
    b'        0.5259959697723389,\r\n'
    b'        0.534341037273407,\r\n'
    b'        0.5429630875587463,\r\n'
    b'        0.552433431148529,\r\n'
    b'        0.5627175569534302,\r\n'
    b'        0.5741269588470459,\r\n'
    b'        0.5877525806427002,\r\n'
    b'        0.6044426560401917,\r\n'
    b'        0.6264132261276245,\r\n'
    b'        0.6575772166252136,\r\n'
    b'        0.6940217018127441,\r\n'
    b'        0.7346214652061462,\r\n'
    b'        0.8442147970199585,\r\n'
    b'        1.0\r\n'
    b'      ]\r\n'
    b'    }\r\n'
    b'  },\r\n'
    b'  "support_exclusion_audit": {\r\n'
    b'    "detection_method": "abs(value) > max(float32_eps*max_abs, 1e-12)",\r\n'
    b'    "observed_exact_zero_fraction_original": 0.24995098691998105,\r\n'
    b'    "observed_supported_fraction_original": 0.7500490130800189,\r\n'
    b'    "outer_padding_excluded_before_statistics": true,\r\n'
    b'    "padding_geometry_used_by_renderer": false,\r\n'
    b'    "support_geometry_serialized": false,\r\n'
    b'    "threshold": 0.0004862546920776367\r\n'
    b'  }\r\n'
    b'}\r\n'
)


TARGET_STYLE_V2_ASSET_SCHEMA_VERSION = 2
TARGET_STYLE_V2_EXTRACTOR_VERSION = 2
TARGET_STYLE_V2_RENDER_VERSION = 6

_CDF_PROBABILITIES = np.linspace(0.0, 1.0, 33, dtype=np.float64)
_RESIDUAL_SIGMAS_VOX = (0.6, 1.2, 2.4)
_STATIONARY_HIGHPASS_SIGMA_VOX = 0.6
_STATIONARY_HIGHPASS_TRUNCATE = 4.0
_EPS = 1e-8
_MAX_NEW_EXACT_ZERO_FRACTION = 1e-4
_FORBIDDEN_KEY_FRAGMENTS = (
    "target_phase", "fourier_phase", "complex_coeff", "fourier_coeff",
    "target_voxel", "image_array", "volume_array", "support_map",
    "padding_map", "mask_array", "mask_data", "posterior", "fossa",
    "cerebell", "regional_field",
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def _asset_payload_for_id(asset: Mapping[str, Any]) -> Dict[str, Any]:
    payload = dict(_jsonable(asset))
    payload.pop("asset_id", None)
    return payload


def _asset_id(asset: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        _canonical_json(_asset_payload_for_id(asset)).encode("utf-8")).hexdigest()
    return f"mp2rage-target-style-v2-{digest[:16]}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_real_3d(image: np.ndarray, name: str = "image") -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"{name} must be a 3-D array, got shape {arr.shape}")
    if not np.issubdtype(arr.dtype, np.number) or np.iscomplexobj(arr):
        raise TypeError(f"{name} must be a real numeric array")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return np.asarray(arr, dtype=np.float32)


def _padding_excluded_crop(raw: np.ndarray):
    """Return a tight array/support pair while retaining no target geometry.

    The threshold is deliberately tied only to the target's numeric dynamic
    range.  The bounding crop removes arbitrary outer zero slabs before scale
    selection, and normalized convolutions below prevent irregular support edges
    from becoming residual texture.  Neither the crop nor support array leaves
    extraction.
    """
    arr = _require_real_3d(raw, "target")
    max_abs = float(np.max(np.abs(arr)))
    if max_abs <= _EPS:
        raise ValueError("cannot extract target style from a constant-zero volume")
    threshold = max(np.finfo(np.float32).eps * max_abs, 1e-12)
    support = np.abs(arr) > threshold
    count = int(np.count_nonzero(support))
    if count < 1024:
        raise ValueError("target has too few supported reconstruction voxels")
    points = np.argwhere(support)
    lo, hi = points.min(axis=0), points.max(axis=0) + 1
    slices = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    cropped = np.asarray(arr[slices], dtype=np.float32)
    cropped_support = np.asarray(support[slices], dtype=bool)
    audit = {
        "detection_method": "abs(value) > max(float32_eps*max_abs, 1e-12)",
        "threshold": float(threshold),
        "observed_supported_fraction_original": float(np.mean(support)),
        "observed_exact_zero_fraction_original": float(np.mean(arr == 0.0)),
        "outer_padding_excluded_before_statistics": True,
        "support_geometry_serialized": False,
        "padding_geometry_used_by_renderer": False,
    }
    return cropped, cropped_support, audit


def _normalize_supported(image: np.ndarray, support: np.ndarray):
    values = np.asarray(image[support], dtype=np.float64)
    lo, hi = np.quantile(values, [0.005, 0.995])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo <= _EPS:
        lo, hi = float(np.min(values)), float(np.max(values))
    if hi - lo <= _EPS:
        raise ValueError("cannot extract target style from constant supported values")
    normalized = np.zeros(image.shape, dtype=np.float32)
    normalized[support] = np.clip(
        (image[support] - float(lo)) / float(hi - lo), 0.0, 1.0)
    return normalized, {
        "method": "supported_voxels_clip_p0p5_p99p5_then_unit_interval",
        "lower_percentile": float(lo),
        "upper_percentile": float(hi),
        "padding_excluded": True,
    }


def _support_normalized_gaussian(
    image: np.ndarray, support: np.ndarray, sigma: float | Sequence[float],
) -> Tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(support, dtype=np.float32)
    numerator = ndi.gaussian_filter(
        np.asarray(image, dtype=np.float32) * weights, sigma=sigma, mode="constant",
        cval=0.0)
    denominator = ndi.gaussian_filter(
        weights, sigma=sigma, mode="constant", cval=0.0)
    smooth = numerator / np.maximum(denominator, np.float32(1e-5))
    return np.asarray(smooth, dtype=np.float32), np.asarray(denominator, dtype=np.float32)


def _pair_correlations(field: np.ndarray, support: np.ndarray, lag: int) -> list:
    correlations = []
    arr = np.asarray(field, dtype=np.float64)
    valid = np.asarray(support, dtype=bool)
    for axis in range(3):
        left = [slice(None)] * 3
        right = [slice(None)] * 3
        left[axis] = slice(0, -int(lag))
        right[axis] = slice(int(lag), None)
        pair = valid[tuple(left)] & valid[tuple(right)]
        a = arr[tuple(left)][pair]
        b = arr[tuple(right)][pair]
        if a.size < 64 or float(np.std(a)) <= _EPS or float(np.std(b)) <= _EPS:
            correlations.append(0.0)
        else:
            correlations.append(float(np.corrcoef(a, b)[0, 1]))
    return correlations


def _stationary_residual_summary(image01: np.ndarray, support: np.ndarray) -> Dict[str, Any]:
    base_smooth, coverage = _support_normalized_gaussian(image01, support, 0.8)
    gradients = np.gradient(base_smooth.astype(np.float32))
    gradient_magnitude = np.sqrt(sum(g * g for g in gradients)).astype(np.float32)
    interior = support & (coverage >= 0.995)
    values = image01[interior]
    if values.size < 1024:
        interior = support & (coverage >= 0.90)
        values = image01[interior]
    q02, q98 = np.quantile(values, [0.02, 0.98])
    gradient_cut = float(np.quantile(gradient_magnitude[interior], 0.45))
    flat = (interior & (image01 >= float(q02)) & (image01 <= float(q98))
            & (gradient_magnitude <= gradient_cut))
    if int(np.count_nonzero(flat)) < 1024:
        flat = interior

    residual_records = []
    base_residual = None
    for sigma in _RESIDUAL_SIGMAS_VOX:
        smooth, local_coverage = _support_normalized_gaussian(image01, support, sigma)
        residual = np.asarray(image01 - smooth, dtype=np.float32)
        scale_valid = flat & (local_coverage >= 0.995)
        if int(np.count_nonzero(scale_valid)) < 1024:
            scale_valid = flat
        if abs(float(sigma) - 0.6) < 1e-6:
            base_residual = residual
        residual_records.append({
            "sigma_vox": float(sigma),
            "flat_supported_robust_sigma": robust_sigma(residual[scale_valid]),
        })
    assert base_residual is not None
    lag_support = flat & support
    return {
        "estimator": (
            "support-normalized local Gaussian residuals on low-gradient supported voxels"),
        "flat_selection_quantile": 0.45,
        "flat_supported_fraction_within_crop": float(np.mean(flat)),
        "residual_scales": residual_records,
        "lag1_xyz": _pair_correlations(base_residual, lag_support, 1),
        "lag2_xyz": _pair_correlations(base_residual, lag_support, 2),
        "target_fft_used": False,
        "spectral_capacity": "three residual amplitudes plus lag1/lag2 per axis",
    }


def _bias_summary(image01: np.ndarray, support: np.ndarray) -> Dict[str, Any]:
    steps = tuple(max(1, int(np.ceil(v / 80.0))) for v in image01.shape)
    small = image01[::steps[0], ::steps[1], ::steps[2]].astype(np.float32)
    small_support = support[::steps[0], ::steps[1], ::steps[2]]
    medium_sigma = tuple(max(0.8, 0.025 * v) for v in small.shape)
    broad_sigma = tuple(max(1.2, 0.10 * v) for v in small.shape)
    medium, medium_cov = _support_normalized_gaussian(
        small, small_support, medium_sigma)
    broad, broad_cov = _support_normalized_gaussian(
        small, small_support, broad_sigma)
    valid = small_support & (medium_cov >= 0.98) & (broad_cov >= 0.98)
    if int(np.count_nonzero(valid)) < 256:
        valid = small_support & (medium_cov >= 0.80) & (broad_cov >= 0.80)
    log_ratio = np.log(np.maximum(broad, 0.04)) - np.log(np.maximum(medium, 0.04))
    measured = robust_sigma(log_ratio[valid])
    # A single target cannot separate coil bias from anatomy.  Retain only a
    # conservative upper-bounded amplitude; never retain its spatial field.
    return {
        "estimator": "broad_to_medium_support_normalized_log_ratio",
        "measured_log_robust_sigma": float(measured),
        "renderer_log_sigma_cap": float(min(0.08, measured)),
        "random_field_scale_fraction_range": [0.08, 0.18],
        "target_bias_field_stored": False,
    }


def extract_target_style_v2_asset(
    target: np.ndarray,
    *,
    voxel_sizes: Optional[Sequence[float]] = None,
    source_id: Optional[str] = None,
    source_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract aggregate style while excluding reconstruction padding.

    A label/mask argument is intentionally absent.  The temporary nonzero support
    is used only inside extraction and is neither returned nor serialized.
    """
    raw = _require_real_3d(target, "target")
    cropped, support, support_audit = _padding_excluded_crop(raw)
    image01, normalization = _normalize_supported(cropped, support)
    probabilities = _CDF_PROBABILITIES.tolist()
    supported_curve = np.quantile(
        image01[support], _CDF_PROBABILITIES).astype(np.float64).tolist()
    spacing = tuple(float(v) for v in (voxel_sizes or (1.0, 1.0, 1.0)))
    if len(spacing) != 3 or not np.isfinite(spacing).all() or min(spacing) <= 0.0:
        raise ValueError("voxel_sizes must contain three finite positive values")

    asset: Dict[str, Any] = {
        "asset_type": "mp2rage_unlabelled_target_style_support_normalized",
        "schema_version": TARGET_STYLE_V2_ASSET_SCHEMA_VERSION,
        "extractor_version": TARGET_STYLE_V2_EXTRACTOR_VERSION,
        "calibration_scope": (
            "single-volume appearance calibration; not a population prior, "
            "not segmentation evidence"),
        "privacy_contract": {
            "aggregate_only": True,
            "target_voxels_stored": False,
            "target_fourier_phase_stored": False,
            "target_complex_coefficients_stored": False,
            "target_mask_used": False,
            "target_support_geometry_stored": False,
            "target_regional_statistics_stored": False,
        },
        "provenance": {
            "source_id": str(source_id) if source_id is not None else None,
            "source_sha256": str(source_sha256) if source_sha256 is not None else None,
            "source_shape": [int(v) for v in raw.shape],
            "voxel_sizes_mm": [float(v) for v in spacing],
        },
        "support_exclusion_audit": support_audit,
        "normalization": normalization,
        "style": {
            "supported_intensity_cdf": {
                "probabilities": probabilities,
                "quantiles": supported_curve,
            },
            "stationary_residual_spectrum": _stationary_residual_summary(
                image01, support),
            "low_frequency_bias": _bias_summary(image01, support),
        },
    }
    asset["asset_id"] = _asset_id(asset)
    return validate_target_style_v2_asset(asset)


def extract_target_style_v2_asset_from_nifti(path: str | Path) -> Dict[str, Any]:
    import nibabel as nib

    source = Path(path).resolve()
    image = nib.load(str(source))
    volume = np.asarray(image.dataobj, dtype=np.float32)
    return extract_target_style_v2_asset(
        volume,
        voxel_sizes=image.header.get_zooms()[:3],
        source_id=source.name,
        source_sha256=_sha256_file(source),
    )


def _walk_asset(value: Any, path: str = "asset"):
    if isinstance(value, Mapping):
        for key, child in value.items():
            lower = str(key).lower()
            if any(fragment in lower for fragment in _FORBIDDEN_KEY_FRAGMENTS):
                # Required negative privacy declarations may name forbidden data.
                allowed_negative = (
                    path.endswith("privacy_contract") and child is False)
                allowed_audit = (
                    path.endswith("support_exclusion_audit")
                    and lower == "padding_geometry_used_by_renderer" and child is False)
                if not (allowed_negative or allowed_audit):
                    raise ValueError(f"forbidden spatial/phase payload key: {path}.{key}")
            yield from _walk_asset(child, f"{path}.{key}")
    elif isinstance(value, (tuple, list)):
        if len(value) > 64:
            raise ValueError(f"aggregate vector too long at {path}: {len(value)}")
        for index, child in enumerate(value):
            yield from _walk_asset(child, f"{path}[{index}]")
    elif value is not None and not isinstance(value, (str, bool)):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError(f"non-finite aggregate value at {path}")
        yield number


def validate_target_style_v2_asset(asset: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(asset, Mapping):
        raise TypeError("target style v2 asset must be a mapping")
    out = _jsonable(asset)
    if out.get("asset_type") != "mp2rage_unlabelled_target_style_support_normalized":
        raise ValueError("not a support-normalized MP2RAGE target-style asset")
    if int(out.get("schema_version", -1)) != TARGET_STYLE_V2_ASSET_SCHEMA_VERSION:
        raise ValueError("unsupported target-style v2 schema")
    expected_privacy = {
        "aggregate_only": True,
        "target_voxels_stored": False,
        "target_fourier_phase_stored": False,
        "target_complex_coefficients_stored": False,
        "target_mask_used": False,
        "target_support_geometry_stored": False,
        "target_regional_statistics_stored": False,
    }
    if out.get("privacy_contract") != expected_privacy:
        raise ValueError("target-style v2 asset violates its aggregate-only contract")
    list(_walk_asset(out))

    audit = out.get("support_exclusion_audit", {})
    if audit.get("outer_padding_excluded_before_statistics") is not True:
        raise ValueError("target-style v2 asset did not exclude reconstruction padding")
    if audit.get("support_geometry_serialized") is not False:
        raise ValueError("target support geometry must not be serialized")
    if audit.get("padding_geometry_used_by_renderer") is not False:
        raise ValueError("renderer must not use target padding geometry")

    cdf = out.get("style", {}).get("supported_intensity_cdf", {})
    probabilities = np.asarray(cdf.get("probabilities", []), dtype=np.float64)
    quantiles = np.asarray(cdf.get("quantiles", []), dtype=np.float64)
    if (probabilities.shape != _CDF_PROBABILITIES.shape
            or quantiles.shape != _CDF_PROBABILITIES.shape
            or np.any(np.diff(probabilities) <= 0.0)
            or np.any(np.diff(quantiles) < -1e-8)):
        raise ValueError("invalid supported intensity CDF")
    if not (0.0 <= float(quantiles[0]) <= float(quantiles[-1]) <= 1.0):
        raise ValueError("supported intensity CDF must lie in [0, 1]")

    residual = out["style"].get("stationary_residual_spectrum", {})
    records = residual.get("residual_scales", [])
    if len(records) != len(_RESIDUAL_SIGMAS_VOX):
        raise ValueError("stationary residual summary has wrong scale count")
    for key in ("lag1_xyz", "lag2_xyz"):
        values = np.asarray(residual.get(key, []), dtype=np.float64)
        if values.shape != (3,) or np.any(np.abs(values) > 1.0):
            raise ValueError(f"invalid stationary residual {key}")
    if residual.get("target_fft_used") is not False:
        raise ValueError("target FFT is forbidden in v2 style extraction")

    expected_id = _asset_id(out)
    if out.get("asset_id") != expected_id:
        raise ValueError(
            f"asset_id mismatch: expected {expected_id}, got {out.get('asset_id')!r}")
    return out


def save_target_style_v2_asset(asset: Mapping[str, Any], path: str | Path) -> Path:
    checked = validate_target_style_v2_asset(asset)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(checked, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    return target


def load_target_style_v2_asset(path: str | Path | None = None) -> Dict[str, Any]:
    """Load an explicit asset, or the bundled aggregate calibration by default."""
    raw = BUNDLED_ASSET_BYTES if path is None else Path(path).read_bytes()
    return validate_target_style_v2_asset(json.loads(raw.decode("utf-8")))


def _noise_filter_sigmas(asset: Mapping[str, Any]) -> list:
    lag1 = np.asarray(
        asset["style"]["stationary_residual_spectrum"]["lag1_xyz"],
        dtype=np.float64)
    sigmas = []
    for correlation in lag1:
        rho = float(np.clip(correlation, 0.0, 0.80))
        if rho <= 0.015:
            sigmas.append(0.0)
        else:
            # Gaussian-filtered white noise has approximately this lag-one
            # correlation in the continuous limit.  The range remains deliberately
            # conservative because one scan cannot identify a scanner PSF exactly.
            sigma = np.sqrt(-1.0 / (4.0 * np.log(rho)))
            sigmas.append(float(np.clip(sigma, 0.0, 0.75)))
    return sigmas


def sample_target_style_v2_parameters(
    asset: Mapping[str, Any], *, seed: int, strength: float = 1.0,
) -> Dict[str, Any]:
    checked = validate_target_style_v2_asset(asset)
    strength = float(strength)
    if not np.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be finite and in [0, 1]")
    rng = np.random.default_rng(int(seed))
    return {
        "render_version": TARGET_STYLE_V2_RENDER_VERSION,
        "asset_id": checked["asset_id"],
        "seed": int(seed),
        "strength": strength,
        "cdf_mix": strength * float(rng.uniform(0.025, 0.10)),
        "spectrum_mix": strength * float(rng.uniform(0.025, 0.10)),
        # Fraction of the missing target residual variance to add. Application
        # measures the source first and adds nothing when it is already noisy.
        "residual_mix": strength * float(rng.uniform(0.35, 0.85)),
        "bias_mix": strength * float(rng.uniform(0.04, 0.16)),
        "spectral_sigma_scale": float(rng.uniform(0.80, 1.20)),
        "bias_scale_fraction": float(rng.uniform(0.08, 0.18)),
        "random_field_seed": int(rng.integers(0, np.iinfo(np.int64).max)),
    }


def _validate_parameters(
    parameters: Mapping[str, Any], asset: Mapping[str, Any],
) -> Dict[str, Any]:
    out = _jsonable(parameters)
    if int(out.get("render_version", -1)) != TARGET_STYLE_V2_RENDER_VERSION:
        raise ValueError("unsupported target-style v2 render record")
    if out.get("asset_id") != asset["asset_id"]:
        raise ValueError("render record belongs to a different target-style v2 asset")
    for key in ("strength", "cdf_mix", "spectrum_mix", "residual_mix", "bias_mix"):
        value = float(out.get(key, -1.0))
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{key} must be finite and in [0, 1]")
        out[key] = value
    sigma_scale = float(out.get("spectral_sigma_scale", 0.0))
    bias_scale = float(out.get("bias_scale_fraction", 0.0))
    if not 0.5 <= sigma_scale <= 1.5:
        raise ValueError("spectral_sigma_scale must be in [0.5, 1.5]")
    if not 0.04 <= bias_scale <= 0.30:
        raise ValueError("bias_scale_fraction must be in [0.04, 0.30]")
    out["spectral_sigma_scale"] = sigma_scale
    out["bias_scale_fraction"] = bias_scale
    out["seed"] = int(out.get("seed", 0))
    out["random_field_seed"] = int(out.get("random_field_seed", out["seed"]))
    return out


def _deduplicated_curve(source: np.ndarray, target: np.ndarray):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    unique, starts, counts = np.unique(source, return_index=True, return_counts=True)
    if unique.size < 2:
        return None, None
    mapped = np.asarray([
        float(np.mean(target[start:start + count]))
        for start, count in zip(starts, counts)
    ], dtype=np.float64)
    return unique, np.maximum.accumulate(mapped)


def _cdf_transfer(image: np.ndarray, target_curve: Sequence[float], mix: float):
    if mix <= 0.0:
        return np.asarray(image, dtype=np.float32).copy()
    source_curve = np.quantile(image, _CDF_PROBABILITIES)
    x, y = _deduplicated_curve(source_curve, np.asarray(target_curve, dtype=np.float64))
    if x is None:
        return np.asarray(image, dtype=np.float32).copy()
    mapped = np.interp(image, x, y).astype(np.float32)
    return np.asarray((1.0 - mix) * image + mix * mapped, dtype=np.float32)


def _source_phase_filter(
    image: np.ndarray, *, sigma_vox: float, mix: float,
) -> np.ndarray:
    if mix <= 0.0 or sigma_vox <= 0.0:
        return np.asarray(image, dtype=np.float32).copy()
    shape = image.shape
    frequencies = [np.fft.fftfreq(shape[0]).astype(np.float32),
                   np.fft.fftfreq(shape[1]).astype(np.float32),
                   np.fft.rfftfreq(shape[2]).astype(np.float32)]
    radius_sq = (frequencies[0][:, None, None] ** 2
                 + frequencies[1][None, :, None] ** 2
                 + frequencies[2][None, None, :] ** 2)
    lowpass = np.exp(
        -2.0 * np.pi ** 2 * float(sigma_vox) ** 2 * radius_sq).astype(np.float32)
    # The gain is strictly positive and real.  No target FFT is available or used.
    gain = np.float32(1.0 - mix) + np.float32(mix) * lowpass
    mean = float(np.mean(image))
    spectrum = sfft.rfftn(np.asarray(image, dtype=np.float32) - mean, workers=1)
    filtered = sfft.irfftn(spectrum * gain, s=shape, workers=1).real + mean
    return np.asarray(filtered, dtype=np.float32)


def _random_bias_field(
    shape: Sequence[int], rng: np.random.Generator, scale_fraction: float,
) -> np.ndarray:
    coarse_shape = tuple(max(5, int(np.ceil(v / 20.0)) + 2) for v in shape)
    field = rng.standard_normal(coarse_shape).astype(np.float32)
    sigma = tuple(max(0.8, float(scale_fraction) * v) for v in coarse_shape)
    field = ndi.gaussian_filter(field, sigma=sigma, mode="reflect")
    zoom = tuple(float(v) / float(c) for v, c in zip(shape, coarse_shape))
    field = ndi.zoom(field, zoom=zoom, order=3, mode="reflect", prefilter=True)
    field = field[tuple(slice(0, int(v)) for v in shape)]
    if field.shape != tuple(shape):
        field = np.pad(
            field,
            [(0, int(v) - int(s)) for v, s in zip(shape, field.shape)],
            mode="edge")
    field -= float(np.mean(field))
    sd = float(np.std(field))
    if sd > _EPS:
        field /= sd
    return np.asarray(field, dtype=np.float32)


def _discrete_gaussian_kernel_1d(
    sigma_vox: float, truncate: float = _STATIONARY_HIGHPASS_TRUNCATE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return SciPy's finite order-0 Gaussian kernel without private APIs."""
    sigma = float(sigma_vox)
    trunc = float(truncate)
    if (not np.isfinite(sigma) or sigma <= 0.0
            or not np.isfinite(trunc) or trunc <= 0.0):
        raise ValueError("sigma_vox and truncate must be finite and positive")
    radius = int(trunc * sigma + 0.5)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    weights /= float(np.sum(weights))
    return offsets, weights


def _discrete_gaussian_response(
    frequency: np.ndarray, sigma_vox: float,
    truncate: float = _STATIONARY_HIGHPASS_TRUNCATE,
) -> np.ndarray:
    offsets, weights = _discrete_gaussian_kernel_1d(sigma_vox, truncate)
    values = np.asarray(frequency, dtype=np.float64)
    return np.sum(
        np.cos(2.0 * np.pi * values[..., None] * offsets) * weights,
        axis=-1)


def _calibrated_raw_stationary_lags(
    post_highpass_lag1_xyz: Sequence[float],
    post_highpass_lag2_xyz: Sequence[float], *,
    sigma_vox: float = _STATIONARY_HIGHPASS_SIGMA_VOX,
    truncate: float = _STATIONARY_HIGHPASS_TRUNCATE,
    grid_size: int = 64,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Solve six raw coefficients for requested post-high-pass lag1/lag2.

    The solve uses only a fixed normalized frequency grid and six aggregate lag
    scalars. It never reads target voxels, target FFT coefficients or phase.
    """
    desired = np.clip(np.concatenate((
        np.asarray(post_highpass_lag1_xyz, dtype=np.float64),
        np.asarray(post_highpass_lag2_xyz, dtype=np.float64))), -0.45, 0.45)
    frequency = np.fft.fftfreq(int(grid_size)).astype(np.float64)
    fx = frequency[:, None, None]
    fy = frequency[None, :, None]
    fz = frequency[None, None, :]
    response_x = _discrete_gaussian_response(fx, sigma_vox, truncate)
    response_y = _discrete_gaussian_response(fy, sigma_vox, truncate)
    response_z = _discrete_gaussian_response(fz, sigma_vox, truncate)
    highpass = 1.0 - response_x * response_y * response_z
    weight = highpass * highpass
    cosines = (
        np.cos(2.0 * np.pi * fx),
        np.cos(2.0 * np.pi * fy),
        np.cos(2.0 * np.pi * fz),
        np.cos(4.0 * np.pi * fx),
        np.cos(4.0 * np.pi * fy),
        np.cos(4.0 * np.pi * fz),
    )
    variance0 = float(np.mean(weight))
    variance_basis = np.asarray([
        float(np.mean(weight * 2.0 * cosine)) for cosine in cosines])
    lag0 = np.asarray([
        float(np.mean(weight * cosine)) for cosine in cosines])
    lag_basis = np.asarray([[
        float(np.mean(weight * 2.0 * cosines[i] * cosines[j]))
        for j in range(6)] for i in range(6)])
    matrix = lag_basis - desired[:, None] * variance_basis[None, :]
    rhs = desired * variance0 - lag0
    try:
        raw = np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        raw = np.zeros(6, dtype=np.float64)
    raw = np.clip(raw, -0.24, 0.24)
    stability_scale = 1.0
    bound = 2.0 * float(np.sum(np.abs(raw)))
    if bound > 0.95:
        stability_scale = 0.95 / bound
        raw *= stability_scale
    denominator = variance0 + float(variance_basis @ raw)
    achieved = (lag0 + lag_basis @ raw) / max(denominator, _EPS)
    return raw, {
        "solver": "analytic_post_discrete_gaussian_highpass_lag1_lag2_v2",
        "solver_grid_size": int(grid_size),
        "highpass_filter_family": "scipy_ndimage_gaussian_filter_order0_separable",
        "highpass_sigma_vox": float(sigma_vox),
        "highpass_truncate": float(truncate),
        "highpass_radius": int(float(truncate) * float(sigma_vox) + 0.5),
        "highpass_kernel_offsets": [
            float(v) for v in _discrete_gaussian_kernel_1d(
                sigma_vox, truncate)[0]],
        "highpass_kernel_1d": [
            float(v) for v in _discrete_gaussian_kernel_1d(
                sigma_vox, truncate)[1]],
        "highpass_response_formula": (
            "1-product_axis(sum_offset(kernel[offset]*cos(2*pi*f*offset)))"),
        "requested_post_highpass_lag1_xyz": [float(v) for v in desired[:3]],
        "requested_post_highpass_lag2_xyz": [float(v) for v in desired[3:]],
        "analytic_achieved_post_highpass_lag1_xyz": [float(v) for v in achieved[:3]],
        "analytic_achieved_post_highpass_lag2_xyz": [float(v) for v in achieved[3:]],
        "resolved_raw_lag1_xyz": [float(v) for v in raw[:3]],
        "resolved_raw_lag2_xyz": [float(v) for v in raw[3:]],
        "stability_scale": float(stability_scale),
        "analytic_minimum_psd": float(1.0 - 2.0 * np.sum(np.abs(raw))),
        "target_fft_used": False,
        "target_phase_used": False,
    }


def _stationary_residual(
    shape: Sequence[int], rng: np.random.Generator,
    lag1_xyz: Sequence[float], lag2_xyz: Sequence[float],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    shape = tuple(int(v) for v in shape)
    rho, calibration = _calibrated_raw_stationary_lags(lag1_xyz, lag2_xyz)

    white = rng.standard_normal(shape).astype(np.float32)
    spectrum = sfft.rfftn(white, workers=1)
    frequencies = (
        np.fft.fftfreq(shape[0]).astype(np.float32),
        np.fft.fftfreq(shape[1]).astype(np.float32),
        np.fft.rfftfreq(shape[2]).astype(np.float32),
    )
    psd = np.ones((shape[0], shape[1], shape[2] // 2 + 1), dtype=np.float32)
    psd += np.float32(2.0 * rho[0]) * np.cos(
        2.0 * np.pi * frequencies[0])[:, None, None]
    psd += np.float32(2.0 * rho[1]) * np.cos(
        2.0 * np.pi * frequencies[1])[None, :, None]
    psd += np.float32(2.0 * rho[2]) * np.cos(
        2.0 * np.pi * frequencies[2])[None, None, :]
    psd += np.float32(2.0 * rho[3]) * np.cos(
        4.0 * np.pi * frequencies[0])[:, None, None]
    psd += np.float32(2.0 * rho[4]) * np.cos(
        4.0 * np.pi * frequencies[1])[None, :, None]
    psd += np.float32(2.0 * rho[5]) * np.cos(
        4.0 * np.pi * frequencies[2])[None, None, :]
    minimum_psd_before_clip = float(np.min(psd))
    psd = np.maximum(psd, np.float32(1e-3))
    # Positive-real amplitude applied to newly seeded white phase. No target
    # FFT, coefficient, phase, support map, or regional field is available.
    residual = sfft.irfftn(
        spectrum * np.sqrt(psd).astype(np.float32), s=shape, workers=1).real
    residual = np.asarray(residual, dtype=np.float32)
    residual -= float(np.mean(residual))
    sd = float(np.std(residual))
    if sd > _EPS:
        residual /= sd
    return np.asarray(residual, dtype=np.float32), {
        "model": (
            "calibrated_post_discrete_gaussian_highpass_lag1_lag2_stationary_psd"),
        "calibration": calibration,
        "minimum_psd_before_clip": minimum_psd_before_clip,
        "target_fft_used": False,
        "target_phase_used": False,
    }


def _source_residual_measurement(image: np.ndarray) -> Tuple[float, float]:
    """Measure normalized fine residual amplitude without reading a label."""
    cropped, support, _audit = _padding_excluded_crop(image)
    normalized, normalization = _normalize_supported(cropped, support)
    summary = _stationary_residual_summary(normalized, support)
    sigma = float(summary["residual_scales"][0][
        "flat_supported_robust_sigma"])
    span = float(normalization["upper_percentile"]
                 - normalization["lower_percentile"])
    return sigma, span


def _unit_highpass_sigma(residual: np.ndarray) -> float:
    residual = np.asarray(residual, dtype=np.float32)
    highpass = residual - ndi.gaussian_filter(
        residual, sigma=_STATIONARY_HIGHPASS_SIGMA_VOX, mode="reflect",
        truncate=_STATIONARY_HIGHPASS_TRUNCATE)
    interior = tuple(slice(4, -4) if size > 8 else slice(None)
                     for size in residual.shape)
    return robust_sigma(highpass[interior])


def gradient_orientation_alignment(source: np.ndarray, candidate: np.ndarray) -> float:
    a = _require_real_3d(source, "source")
    b = _require_real_3d(candidate, "candidate")
    if a.shape != b.shape:
        raise ValueError("source and candidate shapes differ")
    steps = tuple(max(1, int(np.ceil(v / 64.0))) for v in a.shape)
    a = a[::steps[0], ::steps[1], ::steps[2]].astype(np.float64)
    b = b[::steps[0], ::steps[1], ::steps[2]].astype(np.float64)
    ga = np.stack(np.gradient(a), axis=0)
    gb = np.stack(np.gradient(b), axis=0)
    mag_a = np.sqrt(np.sum(ga * ga, axis=0))
    if float(np.max(mag_a)) <= _EPS:
        return 1.0 if float(np.std(b)) <= _EPS else 0.0
    threshold = float(np.quantile(mag_a, 0.60))
    weights = np.clip(mag_a - threshold, 0.0, None)
    if float(np.sum(weights)) <= _EPS:
        weights = mag_a
    mag_b = np.sqrt(np.sum(gb * gb, axis=0))
    cosine = np.sum(ga * gb, axis=0) / np.maximum(mag_a * mag_b, 1e-10)
    return float(np.sum(weights * cosine) / np.maximum(np.sum(weights), _EPS))


def highpass_correlation(
    source: np.ndarray, candidate: np.ndarray, *, sigma_vox: float = 0.8,
) -> float:
    a = _require_real_3d(source, "source")
    b = _require_real_3d(candidate, "candidate")
    if a.shape != b.shape:
        raise ValueError("source and candidate shapes differ")
    steps = tuple(max(1, int(np.ceil(v / 96.0))) for v in a.shape)
    a = a[::steps[0], ::steps[1], ::steps[2]].astype(np.float32)
    b = b[::steps[0], ::steps[1], ::steps[2]].astype(np.float32)
    ah = a - ndi.gaussian_filter(a, sigma=float(sigma_vox), mode="reflect")
    bh = b - ndi.gaussian_filter(b, sigma=float(sigma_vox), mode="reflect")
    av = ah.reshape(-1).astype(np.float64)
    bv = bh.reshape(-1).astype(np.float64)
    av -= float(np.mean(av))
    bv -= float(np.mean(bv))
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= _EPS:
        return 1.0 if float(np.std(bv)) <= _EPS else 0.0
    return float(np.dot(av, bv) / denom)


def apply_target_style_v2(
    image: np.ndarray,
    mask: np.ndarray,
    asset: Mapping[str, Any],
    *,
    seed: int = 0,
    strength: float = 1.0,
    parameters: Optional[Mapping[str, Any]] = None,
    min_gradient_alignment: float = 0.96,
    min_highpass_correlation: float = 0.94,
    return_record: bool = False,
):
    """Apply v2 aggregate style and return an untouched label copy."""
    source = _require_real_3d(image, "image")
    if float(np.min(source)) < -1e-6 or float(np.max(source)) > 1.0 + 1e-6:
        raise ValueError("target-style v2 input must be normalized to [0, 1]")
    source_exact_zero = source == 0.0
    source_exact_zero_count = int(np.count_nonzero(source_exact_zero))
    label = np.asarray(mask)
    if label.shape != source.shape:
        raise ValueError(f"mask shape {label.shape} does not match image {source.shape}")
    label_copy = label.copy()
    checked = validate_target_style_v2_asset(asset)
    min_gradient_alignment = float(min_gradient_alignment)
    min_highpass_correlation = float(min_highpass_correlation)
    if not 0.0 <= min_gradient_alignment <= 1.0:
        raise ValueError("min_gradient_alignment must be in [0, 1]")
    if not 0.0 <= min_highpass_correlation <= 1.0:
        raise ValueError("min_highpass_correlation must be in [0, 1]")
    sampled = (sample_target_style_v2_parameters(
        checked, seed=int(seed), strength=float(strength))
        if parameters is None else parameters)
    params = _validate_parameters(sampled, checked)
    record = dict(params)
    record.update({
        "target_phase_used": False,
        "target_fft_used": False,
        "target_voxels_used": False,
        "target_padding_geometry_imported": False,
        "target_regional_field_used": False,
        "label_inspected": False,
        "spectrum_phase_source_only": True,
        "source_exact_zero_count": source_exact_zero_count,
        "source_exact_zero_preserved": True,
        "new_exact_zero_count": 0,
        "new_exact_zero_fraction": 0.0,
        "exact_zero_fraction_change": 0.0,
        "max_new_exact_zero_fraction": _MAX_NEW_EXACT_ZERO_FRACTION,
        "identity_reason": None,
    })

    if params["strength"] <= 0.0:
        record.update({"identity_reason": "zero_strength", "gradient_alignment": 1.0,
                       "highpass_correlation": 1.0, "integrity_backtrack": 0.0})
        result = source.copy()
        return (result, label_copy, record) if return_record else (result, label_copy)
    if float(np.ptp(source)) <= _EPS:
        record.update({"identity_reason": "constant_input", "gradient_alignment": 1.0,
                       "highpass_correlation": 1.0, "integrity_backtrack": 0.0})
        result = source.copy()
        return (result, label_copy, record) if return_record else (result, label_copy)

    style = checked["style"]
    candidate = _cdf_transfer(
        source, style["supported_intensity_cdf"]["quantiles"], params["cdf_mix"])
    filter_sigmas = _noise_filter_sigmas(checked)
    spectral_sigma = float(np.mean(filter_sigmas)) * params["spectral_sigma_scale"]
    candidate = _source_phase_filter(
        candidate, sigma_vox=spectral_sigma, mix=params["spectrum_mix"])

    rng = np.random.default_rng(params["random_field_seed"])
    bias = style["low_frequency_bias"]
    if params["bias_mix"] > 0.0:
        field = _random_bias_field(
            source.shape, rng, params["bias_scale_fraction"])
        log_sd = float(bias["renderer_log_sigma_cap"])
        candidate *= np.exp(
            np.float32(params["bias_mix"] * log_sd) * field).astype(np.float32)

    record["residual_stage"] = {"mode": "disabled"}
    if params["residual_mix"] > 0.0:
        residual_style = style["stationary_residual_spectrum"]
        target_sigma = min(
            0.08,
            float(residual_style["residual_scales"][0][
                "flat_supported_robust_sigma"]))
        source_sigma, source_span = _source_residual_measurement(candidate)
        variance_gap = max(target_sigma ** 2 - source_sigma ** 2, 0.0)
        stage = {
            "mode": "adaptive_discrete_stationary_psd",
            "source_normalized_sigma": float(source_sigma),
            "target_normalized_sigma": float(target_sigma),
            "source_normalization_span": float(source_span),
            "variance_gap": float(variance_gap),
            "variance_mix": float(params["residual_mix"]),
            "identity_reason": None,
        }
        if variance_gap <= 0.0:
            stage.update({
                "identity_reason": "source_at_or_above_target",
                "added_normalized_sigma": 0.0,
                "added_image_scale": 0.0,
            })
        else:
            residual, psd_record = _stationary_residual(
                source.shape, rng, residual_style["lag1_xyz"],
                residual_style["lag2_xyz"])
            unit_sigma = _unit_highpass_sigma(residual)
            added_normalized_sigma = float(np.sqrt(
                params["residual_mix"] * variance_gap))
            added_image_scale = (added_normalized_sigma * source_span
                                 / max(unit_sigma, _EPS))
            candidate += np.float32(added_image_scale) * residual
            stage.update({
                "added_normalized_sigma": added_normalized_sigma,
                "added_image_scale": float(added_image_scale),
                "unit_residual_highpass_sigma": float(unit_sigma),
                "psd": psd_record,
            })
        record["residual_stage"] = stage

    candidate = np.clip(
        np.nan_to_num(candidate, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0
    ).astype(np.float32)
    candidate[source_exact_zero] = source[source_exact_zero]
    new_zero_fraction = float(np.mean(
        (candidate == 0.0) & ~source_exact_zero))

    alignment = gradient_orientation_alignment(source, candidate)
    highpass = highpass_correlation(source, candidate)
    backtrack = 1.0
    while ((alignment < min_gradient_alignment
            or highpass < min_highpass_correlation
            or new_zero_fraction > _MAX_NEW_EXACT_ZERO_FRACTION)
           and backtrack > 1.0 / 128.0):
        backtrack *= 0.5
        trial = np.clip(
            source + np.float32(backtrack) * (candidate - source), 0.0, 1.0
        ).astype(np.float32)
        trial[source_exact_zero] = source[source_exact_zero]
        alignment = gradient_orientation_alignment(source, trial)
        highpass = highpass_correlation(source, trial)
        new_zero_fraction = float(np.mean(
            (trial == 0.0) & ~source_exact_zero))
    if (alignment < min_gradient_alignment
            or highpass < min_highpass_correlation
            or new_zero_fraction > _MAX_NEW_EXACT_ZERO_FRACTION):
        candidate = source.copy()
        alignment, highpass, backtrack, new_zero_fraction = 1.0, 1.0, 0.0, 0.0
        record["identity_reason"] = "integrity_guard"
    elif backtrack < 1.0:
        candidate = np.clip(
            source + np.float32(backtrack) * (candidate - source), 0.0, 1.0
        ).astype(np.float32)
        candidate[source_exact_zero] = source[source_exact_zero]
    candidate[source_exact_zero] = source[source_exact_zero]
    record["gradient_alignment"] = float(alignment)
    record["highpass_correlation"] = float(highpass)
    record["integrity_backtrack"] = float(backtrack)
    record["new_exact_zero_count"] = int(np.count_nonzero(
        (candidate == 0.0) & ~source_exact_zero))
    record["new_exact_zero_fraction"] = float(record["new_exact_zero_count"]
                                                / candidate.size)
    record["exact_zero_fraction_change"] = float(
        np.mean(candidate == 0.0) - np.mean(source_exact_zero))
    record["resolved_spectral_sigma_vox"] = float(spectral_sigma)
    record["resolved_noise_filter_sigmas_xyz"] = [float(v) for v in filter_sigmas]

    if not np.all(candidate[source_exact_zero] == source[source_exact_zero]):
        raise RuntimeError("target-style v2 changed source exact-zero voxels")
    if record["new_exact_zero_fraction"] > _MAX_NEW_EXACT_ZERO_FRACTION + 1e-12:
        raise RuntimeError("target-style v2 created too many new exact-zero voxels")
    if label_copy.dtype != label.dtype or label_copy.tobytes() != label.tobytes():
        raise RuntimeError("target-style v2 application modified the label")
    return ((candidate, label_copy, record) if return_record
            else (candidate, label_copy))


__all__ = [
    "TARGET_STYLE_V2_ASSET_SCHEMA_VERSION",
    "TARGET_STYLE_V2_EXTRACTOR_VERSION",
    "TARGET_STYLE_V2_RENDER_VERSION",
    "apply_target_style_v2",
    "extract_target_style_v2_asset",
    "extract_target_style_v2_asset_from_nifti",
    "gradient_orientation_alignment",
    "highpass_correlation",
    "load_target_style_v2_asset",
    "sample_target_style_v2_parameters",
    "save_target_style_v2_asset",
    "validate_target_style_v2_asset",
]
