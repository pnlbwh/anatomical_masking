"""Versioned, replayable hard-artifact tail for online masker training.

The ordinary masker artifact overlay samples independent operators from a broad
pool.  This module owns a much smaller set of scanner-coherent, deliberately
strong two-operator profiles. There are no retries: excluded, ineffective, or
signal-destroying proposals return the exact input image and mask. Operator
execution errors and invalid array contracts abort instead of hiding a bug.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from augmentations import REGISTRY, apply as _registry_apply
from augmentations.config import AugmentationError


HARD_ARTIFACT_TAIL_VERSION = 1
HARD_ARTIFACT_PROFILE_SCHEMA_VERSION = 5  # Schema 4 was reviewed and rejected.
MAX_CONDITIONAL_FRACTION = 0.10
MAX_GLOBAL_FRACTION = 0.05
MIN_CHANGED_FRACTION = 1e-3
MIN_WHOLE_MEAN_ABSOLUTE_DELTA = 5e-4
MIN_SUPPORTED_MEAN_ABSOLUTE_DELTA = 1e-3
MIN_IN_MASK_MEAN_ABSOLUTE_DELTA = 1e-3
MIN_IN_MASK_P99_ABSOLUTE_DELTA = 0.02
MIN_SUPPORTED_DYNAMIC_RANGE_RETENTION = 0.35
MAX_SUPPORTED_DYNAMIC_RANGE_RATIO = 5.0
MIN_IN_MASK_RMS_RATIO = 0.25
MAX_IN_MASK_RMS_RATIO = 5.0
MIN_SUPPORTED_ABOVE_FLOOR_FRACTION = 0.65
MAX_IN_MASK_PLATEAU_FRACTION = 0.20
MAX_IN_MASK_EXACT_ONE_FRACTION = 0.10
POSITIVE_MAGNITUDE_FLOOR_SOURCE_THRESHOLD = 1e-6
POSITIVE_MAGNITUDE_FLOOR_MINIMUM = 2e-6
POSITIVE_MAGNITUDE_FLOOR_SOURCE_FRACTION = 0.01
MAX_POSITIVE_MAGNITUDE_FLOOR_FRACTION = 0.02

_OPERATOR_DEPENDENCY_FILES = {
    "augmentations.__init__": "__init__.py",
    "augmentations.artifacts.__init__": "artifacts/__init__.py",
    "augmentations.artifacts.kspace": "artifacts/kspace.py",
    "augmentations.artifacts._realistic_ringing": "artifacts/_realistic_ringing.py",
    "augmentations.artifacts.physics": "artifacts/physics.py",
    "augmentations.artifacts.recon": "artifacts/recon.py",
    "augmentations.artifacts.intensity": "artifacts/intensity.py",
    "augmentations.artifacts.operators": "artifacts/operators.py",
    "augmentations.registry": "registry.py",
    "augmentations.numerics": "numerics.py",
}

_PROFILE_SATURATION_CAPS = {
    "motion_gibbs": {"new_zero": 0.01, "new_one": 0.08},
    "ghost_rician": {"new_zero": 0.01, "new_one": 0.08},
    "undersample_recon": {"new_zero": 0.01, "new_one": 0.08},
    "bias_erasing": {"new_zero": 0.15, "new_one": 0.08},
    "alias_gfactor": {"new_zero": 0.01, "new_one": 0.08},
}

# Each profile combines signatures that plausibly coexist in one acquisition
# or reconstruction.  Anisotropy/resampling is deliberately absent: the online
# dataset already owns exactly one acquisition-resolution stage.
_PROFILES: Tuple[Mapping[str, Any], ...] = (
    {
        "id": "motion_gibbs",
        "operators": (
            {"name": "gibbs", "severity": (0.38, 0.44)},
            {"name": "motion", "severity": (0.10, 0.20),
             "kwargs": {"n_poses": 4, "magnitude": True,
                        "slice_workers": 4}},
        ),
    },
    {
        "id": "ghost_rician",
        "operators": (
            {"name": "ghosting", "severity": (0.70, 0.95)},
            {"name": "rician_noise", "severity": (0.25, 0.45)},
        ),
    },
    {
        "id": "undersample_recon",
        "operators": (
            {"name": "pe_undersample", "severity": (0.70, 1.00),
             "kwargs": {"magnitude": True}},
            {"name": "dl_recon", "severity": (0.40, 0.65)},
        ),
    },
    {
        "id": "bias_erasing",
        "operators": (
            {"name": "bias", "severity": (0.50, 0.75)},
            {"name": "random_erasing", "severity": (0.65, 0.90),
             "kwargs": {"magnitude_noise": True}},
        ),
    },
    {
        "id": "alias_gfactor",
        "operators": (
            {"name": "aliasing", "severity": (0.65, 0.90)},
            {"name": "g_factor_noise", "severity": (0.18, 0.30)},
        ),
    },
)

PROFILE_IDS = tuple(str(profile["id"]) for profile in _PROFILES)


def _canonical_json(value: Any) -> str:
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


def _validate_profiles() -> None:
    seen = set()
    slice_worker_consumers = []
    for profile in _PROFILES:
        profile_id = str(profile["id"])
        if not profile_id or profile_id in seen:
            raise RuntimeError("hard-artifact profile IDs must be unique and non-empty")
        seen.add(profile_id)
        operators = tuple(profile["operators"])
        if len(operators) != 2:
            raise RuntimeError(f"profile {profile_id!r} must contain exactly two operators")
        for operator in operators:
            name = str(operator["name"])
            if name == "anisotropy":
                raise RuntimeError("hard-artifact profiles may not add a second resolution pass")
            spec = REGISTRY.get(name)
            if spec is None:
                raise RuntimeError(f"hard-artifact operator {name!r} is not registered")
            if not spec.label_preserving or spec.dims not in ("3d", "either"):
                raise RuntimeError(
                    f"hard-artifact operator {name!r} must be label-preserving and 3-D")
            lo, hi = (float(value) for value in operator["severity"])
            if not (np.isfinite(lo) and np.isfinite(hi) and 0.0 <= lo <= hi):
                raise RuntimeError(f"invalid severity range for {name!r}")
            kwargs = operator.get("kwargs", {})
            if not isinstance(kwargs, Mapping):
                raise RuntimeError(f"operator kwargs for {name!r} must be a mapping")
            if "mask" in kwargs:
                raise RuntimeError("hard-artifact profiles may not receive a target mask")
            if "slice_workers" in kwargs:
                slice_worker_consumers.append(
                    (profile_id, name, kwargs["slice_workers"]))

    expected_motion_profile = {
        "id": "motion_gibbs",
        "operators": (
            {"name": "gibbs", "severity": (0.38, 0.44)},
            {"name": "motion", "severity": (0.10, 0.20),
             "kwargs": {"n_poses": 4, "magnitude": True,
                        "slice_workers": 4}},
        ),
    }
    motion_profiles = [profile for profile in _PROFILES
                       if profile.get("id") == "motion_gibbs"]
    if motion_profiles != [expected_motion_profile]:
        raise RuntimeError(
            "motion_gibbs must exactly match the frozen schema-5 profile")
    if (len(slice_worker_consumers) != 1
            or slice_worker_consumers[0][:2] != ("motion_gibbs", "motion")
            or type(slice_worker_consumers[0][2]) is not int
            or slice_worker_consumers[0][2] != 4):
        raise RuntimeError(
            "motion_gibbs motion must be the sole exact slice_workers=4 "
            "profile consumer")


def _operator_dependency_sha256() -> Dict[str, str]:
    base = Path(__file__).resolve().parents[1]
    return {
        module_id: _sha256_file(base / filename)
        for module_id, filename in _OPERATOR_DEPENDENCY_FILES.items()
    }


def hard_artifact_tail_v1_identity_metadata() -> Dict[str, Any]:
    """Validate the live registry and return a path-free resume identity."""
    _validate_profiles()
    dependency_sha256 = _operator_dependency_sha256()
    schema = {
        "schema_version": HARD_ARTIFACT_PROFILE_SCHEMA_VERSION,
        "profiles": _PROFILES,
        "dependency_sha256": dependency_sha256,
        "acceptance": {
            "minimum_changed_fraction": MIN_CHANGED_FRACTION,
            "minimum_whole_mean_absolute_delta": MIN_WHOLE_MEAN_ABSOLUTE_DELTA,
            "minimum_supported_mean_absolute_delta":
                MIN_SUPPORTED_MEAN_ABSOLUTE_DELTA,
            "minimum_in_mask_mean_absolute_delta": MIN_IN_MASK_MEAN_ABSOLUTE_DELTA,
            "minimum_in_mask_p99_absolute_delta": MIN_IN_MASK_P99_ABSOLUTE_DELTA,
            "minimum_supported_dynamic_range_retention":
                MIN_SUPPORTED_DYNAMIC_RANGE_RETENTION,
            "maximum_supported_dynamic_range_ratio":
                MAX_SUPPORTED_DYNAMIC_RANGE_RATIO,
            "in_mask_rms_ratio": [MIN_IN_MASK_RMS_RATIO, MAX_IN_MASK_RMS_RATIO],
            "profile_saturation_caps": _PROFILE_SATURATION_CAPS,
            "maximum_in_mask_plateau_fraction": MAX_IN_MASK_PLATEAU_FRACTION,
            "maximum_in_mask_exact_one_fraction":
                MAX_IN_MASK_EXACT_ONE_FRACTION,
            "minimum_supported_above_floor_fraction":
                MIN_SUPPORTED_ABOVE_FLOOR_FRACTION,
            "no_retry": True,
            "veto_returns_exact_identity": True,
            "target_blind_operators": True,
            "target_aware_scalar_acceptance_gates": True,
            "positive_magnitude_floor": {
                "condition": "source>1e-6 and raw_proposal<=1e-6",
                "formula": "max(2e-6,0.01*source) voxelwise",
                "source_threshold_exclusive":
                    POSITIVE_MAGNITUDE_FLOOR_SOURCE_THRESHOLD,
                "minimum": POSITIVE_MAGNITUDE_FLOOR_MINIMUM,
                "source_fraction": POSITIVE_MAGNITUDE_FLOOR_SOURCE_FRACTION,
                "applied_after_all_operators_before_metrics": True,
                "target_blind": True,
                "maximum_originally_positive_fraction":
                    MAX_POSITIVE_MAGNITUDE_FLOOR_FRACTION,
            },
        },
        "excludes": ["anisotropy", "geometry", "mask_changes"],
    }
    return {
        "id": "masker-hard-artifact-tail-v1",
        "version": HARD_ARTIFACT_TAIL_VERSION,
        "profile_schema_version": HARD_ARTIFACT_PROFILE_SCHEMA_VERSION,
        "module_sha256": _sha256_file(Path(__file__).resolve()),
        "schema_sha256": _sha256_bytes(_canonical_json(schema).encode("utf-8")),
        "dependency_sha256": dependency_sha256,
        "profile_ids": list(PROFILE_IDS),
    }


def _operator_kwargs(name: str, rng: np.random.Generator) -> Dict[str, Any]:
    if name == "ghosting":
        return {"axis": int(rng.integers(0, 3))}
    return {}


def _same_bytes(left: np.ndarray, right: np.ndarray) -> bool:
    a, b = np.asarray(left), np.asarray(right)
    return (a.shape == b.shape and a.dtype == b.dtype
            and a.tobytes(order="C") == b.tobytes(order="C"))


def _plateau_fraction(values: np.ndarray) -> float:
    quantized = np.rint(np.clip(values, 0.0, 1.0) * 1024.0).astype(np.int32)
    return float(np.bincount(quantized, minlength=1025).max() / max(1, quantized.size))


def _effect_metrics(source: np.ndarray, candidate: np.ndarray,
                    mask: np.ndarray) -> Dict[str, float]:
    delta = np.abs(candidate.astype(np.float64) - source.astype(np.float64))
    positive = source[source > 0.0]
    support_floor = (max(1e-6, float(np.percentile(positive, 1)) * 0.25)
                     if positive.size else 1e-6)
    # Label-free source support. The proposal and numerical floor are generated
    # without GT; the GT mask is used below only for scalar post-proposal safety
    # vetoes, never to modify, resample, or localize the candidate.
    support = source > support_floor
    if not support.any():
        support = np.ones(source.shape, dtype=bool)
    source_supported = source[support].astype(np.float64)
    candidate_supported = candidate[support].astype(np.float64)
    source_range = float(np.percentile(source_supported, 95)
                         - np.percentile(source_supported, 5))
    candidate_range = float(np.percentile(candidate_supported, 95)
                            - np.percentile(candidate_supported, 5))
    target = np.asarray(mask, dtype=bool)
    source_target = source[target].astype(np.float64)
    candidate_target = candidate[target].astype(np.float64)
    target_delta = delta[target]
    source_rms = float(np.sqrt(np.mean(np.square(source_target))))
    candidate_rms = float(np.sqrt(np.mean(np.square(candidate_target))))
    return {
        "whole_mean_absolute_delta": float(delta.mean()),
        "supported_mean_absolute_delta": float(delta[support].mean()),
        "changed_fraction": float(np.mean(delta > 1e-6)),
        "supported_p99_absolute_delta": float(np.percentile(delta[support], 99)),
        "source_supported_p05_p95_range": source_range,
        "candidate_supported_p05_p95_range": candidate_range,
        "supported_dynamic_range_ratio": float(
            candidate_range / (source_range + 1e-8)),
        "source_support_floor": support_floor,
        "supported_low_saturation_fraction": float(
            np.mean(candidate_supported <= 1e-6)),
        "supported_high_saturation_fraction": float(
            np.mean(candidate_supported >= 1.0 - 1e-6)),
        "supported_above_source_floor_fraction": float(
            np.mean(candidate_supported > support_floor)),
        "in_mask_mean_absolute_delta": float(target_delta.mean()),
        "in_mask_p99_absolute_delta": float(np.percentile(target_delta, 99)),
        "in_mask_rms_ratio": float(candidate_rms / (source_rms + 1e-8)),
        "in_mask_new_zero_fraction": float(np.mean(
            (candidate_target <= 1e-6) & (source_target > 1e-6))),
        "in_mask_new_one_fraction": float(np.mean(
            (candidate_target >= 1.0 - 1e-6) & (source_target < 1.0 - 1e-6))),
        "in_mask_exact_one_fraction": float(np.mean(
            candidate_target >= 1.0 - 1e-6)),
        "in_mask_source_plateau_fraction": _plateau_fraction(source_target),
        "in_mask_candidate_plateau_fraction": _plateau_fraction(candidate_target),
    }


def apply_hard_artifact_tail_v1(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    seed: int,
    voxel_sizes_mm: Sequence[float] = (1.0, 1.0, 1.0),
    exclude: Sequence[str] = (),
    return_record: bool = False,
):
    """Apply one deterministic hard profile, or return exact identity on veto."""
    _validate_profiles()
    source = np.asarray(image)
    if source.ndim != 3 or source.dtype != np.float32:
        raise ValueError("hard-artifact tail requires one float32 3-D image")
    if not np.isfinite(source).all():
        raise ValueError("hard-artifact tail image must be finite")
    if float(source.min()) < -1e-6 or float(source.max()) > 1.0 + 1e-6:
        raise ValueError("hard-artifact tail image must be in [0, 1]")
    label = np.asarray(mask)
    if label.shape != source.shape or not np.asarray(label, dtype=bool).any():
        raise ValueError("hard-artifact tail requires a non-empty shape-matched mask")
    spacing = tuple(float(value) for value in voxel_sizes_mm)
    if (len(spacing) != 3 or not all(np.isfinite(value) and value > 0.0
                                     for value in spacing)):
        raise ValueError("voxel_sizes_mm must contain three finite positive values")

    original = source.copy()
    original_label = label.copy()
    rng = np.random.default_rng(int(seed))
    profile = _PROFILES[int(rng.integers(0, len(_PROFILES)))]
    profile_id = str(profile["id"])
    excluded = {str(name) for name in exclude}
    record: Dict[str, Any] = {
        "identity": hard_artifact_tail_v1_identity_metadata(),
        "seed": int(seed),
        "requested_profile": profile_id,
        "applied_profile": None,
        "accepted": False,
        "veto_reason": None,
        "voxel_sizes_mm": list(spacing),
        "excluded_operators": sorted(excluded),
        "operators": [],
        "effect_metrics": None,
        "mask_byte_identity": True,
        "target_blind_operators": True,
        "target_aware_scalar_acceptance_gates": True,
        "attempt": 1,
        "retry_count": 0,
        "proposal_array_sha256": None,
        "raw_proposal_array_sha256": None,
        "post_floor_proposal_array_sha256": None,
        "positive_magnitude_floor": {
            "condition": "source>1e-6 and raw_proposal<=1e-6",
            "formula": "max(2e-6,0.01*source) voxelwise",
            "count": 0,
            "whole_fraction": 0.0,
            "originally_positive_fraction": 0.0,
            "minimum": POSITIVE_MAGNITUDE_FLOOR_MINIMUM,
            "source_fraction": POSITIVE_MAGNITUDE_FLOOR_SOURCE_FRACTION,
            "target_blind": True,
            "maximum_originally_positive_fraction":
                MAX_POSITIVE_MAGNITUDE_FLOOR_FRACTION,
            "raw_in_mask_new_zero_fraction": None,
        },
    }

    work = original.copy()
    for operator in profile["operators"]:
        name = str(operator["name"])
        lo, hi = (float(value) for value in operator["severity"])
        severity = float(rng.uniform(lo, hi))
        operator_seed = int(rng.integers(1, 2 ** 31))
        operator_rng = np.random.default_rng(operator_seed)
        kwargs = dict(operator.get("kwargs", {}))
        generated_kwargs = _operator_kwargs(name, operator_rng)
        overlap = set(kwargs) & set(generated_kwargs)
        if overlap:
            raise RuntimeError(
                f"duplicate static/dynamic operator kwargs for {name!r}: {sorted(overlap)}")
        kwargs.update(generated_kwargs)
        operator_record = {
            "name": name,
            "severity": severity,
            "seed": operator_seed,
            "kwargs": dict(kwargs),
            "applied": False,
            "mean_absolute_delta": None,
        }
        record["operators"].append(operator_record)
        if name in excluded:
            record["veto_reason"] = f"excluded_operator:{name}"
            break
        before = work
        try:
            proposed = np.asarray(_registry_apply(
                name, before, severity, operator_rng, mask=None,
                **kwargs), dtype=np.float32)
        except Exception as exc:
            raise AugmentationError(
                f"Hard-tail augmentation {name!r} failed: {type(exc).__name__}: {exc}"
            ) from exc
        if proposed.shape != original.shape:
            raise AugmentationError(
                f"Hard-tail augmentation {name!r} returned shape {proposed.shape}; "
                f"expected {original.shape}")
        if not np.isfinite(proposed).all():
            raise AugmentationError(f"Hard-tail augmentation {name!r} returned nonfinite values")
        proposed = np.clip(proposed, 0.0, 1.0).astype(np.float32, copy=False)
        operator_delta = float(np.mean(np.abs(
            proposed.astype(np.float64) - before.astype(np.float64))))
        operator_record["mean_absolute_delta"] = operator_delta
        if operator_delta <= 1e-8:
            record["veto_reason"] = f"operator_noop:{name}"
            break
        operator_record["applied"] = True
        work = proposed

    if record["veto_reason"] is None:
        record["raw_proposal_array_sha256"] = _sha256_bytes(
            np.ascontiguousarray(work).tobytes())
        originally_positive = original > POSITIVE_MAGNITUDE_FLOOR_SOURCE_THRESHOLD
        floor_mask = originally_positive & (work <= 1e-6)
        floor_count = int(floor_mask.sum())
        target = np.asarray(original_label, dtype=bool)
        raw_in_mask_new_zero_fraction = float(np.mean(floor_mask[target]))
        if floor_count:
            work = work.copy()
            work[floor_mask] = np.maximum(
                np.float32(POSITIVE_MAGNITUDE_FLOOR_MINIMUM),
                np.float32(POSITIVE_MAGNITUDE_FLOOR_SOURCE_FRACTION)
                * original[floor_mask])
        floor_record = record["positive_magnitude_floor"]
        floor_record["count"] = floor_count
        floor_record["whole_fraction"] = float(floor_count / original.size)
        floor_record["originally_positive_fraction"] = float(
            floor_count / max(1, int(originally_positive.sum())))
        floor_record["raw_in_mask_new_zero_fraction"] = (
            raw_in_mask_new_zero_fraction)
        post_floor_hash = _sha256_bytes(np.ascontiguousarray(work).tobytes())
        record["post_floor_proposal_array_sha256"] = post_floor_hash
        # Backward-friendly generic proposal key is explicitly the candidate
        # evaluated by all scalar gates, i.e. the post-floor proposal.
        record["proposal_array_sha256"] = post_floor_hash
        metrics = _effect_metrics(original, work, original_label)
        record["effect_metrics"] = metrics
        caps = _PROFILE_SATURATION_CAPS[profile_id]
        if (floor_record["originally_positive_fraction"]
                > MAX_POSITIVE_MAGNITUDE_FLOOR_FRACTION):
            record["veto_reason"] = "positive_magnitude_floor_fraction"
        elif floor_record["raw_in_mask_new_zero_fraction"] > caps["new_zero"]:
            record["veto_reason"] = "raw_new_zero_saturation"
        elif (metrics["changed_fraction"] < MIN_CHANGED_FRACTION
                or metrics["whole_mean_absolute_delta"]
                < MIN_WHOLE_MEAN_ABSOLUTE_DELTA
                or metrics["supported_mean_absolute_delta"]
                < MIN_SUPPORTED_MEAN_ABSOLUTE_DELTA
                or metrics["in_mask_mean_absolute_delta"]
                < MIN_IN_MASK_MEAN_ABSOLUTE_DELTA
                or metrics["in_mask_p99_absolute_delta"]
                < MIN_IN_MASK_P99_ABSOLUTE_DELTA):
            record["veto_reason"] = "insufficient_effect"
        elif not (MIN_SUPPORTED_DYNAMIC_RANGE_RETENTION
                  <= metrics["supported_dynamic_range_ratio"]
                  <= MAX_SUPPORTED_DYNAMIC_RANGE_RATIO):
            record["veto_reason"] = "supported_dynamic_range"
        elif not (MIN_IN_MASK_RMS_RATIO <= metrics["in_mask_rms_ratio"]
                  <= MAX_IN_MASK_RMS_RATIO):
            record["veto_reason"] = "in_mask_rms"
        elif metrics["in_mask_new_zero_fraction"] > caps["new_zero"]:
            record["veto_reason"] = "new_zero_saturation"
        elif metrics["in_mask_new_one_fraction"] > caps["new_one"]:
            record["veto_reason"] = "new_one_saturation"
        elif (metrics["in_mask_exact_one_fraction"]
              > MAX_IN_MASK_EXACT_ONE_FRACTION):
            record["veto_reason"] = "exact_one_saturation"
        elif (metrics["in_mask_candidate_plateau_fraction"]
              > MAX_IN_MASK_PLATEAU_FRACTION):
            record["veto_reason"] = "in_mask_plateau"
        elif (metrics["supported_above_source_floor_fraction"]
              < MIN_SUPPORTED_ABOVE_FLOOR_FRACTION):
            record["veto_reason"] = "support_floor_retention"
        elif not _same_bytes(original_label, label):
            record["veto_reason"] = "mask_integrity"
        else:
            record["accepted"] = True
            record["applied_profile"] = profile_id

    result = work if record["accepted"] else original
    copied_label = original_label.copy()
    record["mask_byte_identity"] = _same_bytes(copied_label, original_label)
    if not record["accepted"] and not _same_bytes(result, original):
        raise RuntimeError("hard-artifact veto failed to return exact image identity")
    if not _same_bytes(copied_label, original_label):
        raise RuntimeError("hard-artifact tail modified the mask")
    return ((result, copied_label, record) if return_record
            else (result, copied_label))


__all__ = [
    "HARD_ARTIFACT_TAIL_VERSION",
    "HARD_ARTIFACT_PROFILE_SCHEMA_VERSION",
    "MAX_CONDITIONAL_FRACTION",
    "MAX_GLOBAL_FRACTION",
    "PROFILE_IDS",
    "hard_artifact_tail_v1_identity_metadata",
    "apply_hard_artifact_tail_v1",
]
