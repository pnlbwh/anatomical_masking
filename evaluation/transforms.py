"""Deterministic, paired 3-D augmentation catalog for held-out masker evaluation.

The clean branch preserves the supplied intensities. Appearance renderers receive
the same robust [0, 1] normalization as online training. All outputs retain the
input voxel grid; geometric operations return a co-transformed binary target.
Synthetic protocols measure robustness to these renderers, not real acquisitions.
"""
from __future__ import annotations

import copy
from configuration.presets import resolve_run_config, run_config_sources
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial.transform import Rotation

from augmentations import REGISTRY, apply
from augmentations.config import SAMPLING, STAGES, validate_run_config
from augmentations.pipeline import (
    _ARTIFACT_SEV_RANGE, _focal_kwargs, _masker_artifact_pool, make_training_sample,
)
from imaging.normalization import normalize_intensity


def _range(value, name, *, minimum=0.0):
    result = tuple(float(x) for x in value)
    if (len(result) != 2 or not np.isfinite(result).all()
            or result[0] < minimum or result[1] < result[0]):
        raise ValueError(f"{name} must be an ordered finite pair >= {minimum}")
    return result


def _composite_config(path, severity):
    config = validate_run_config(resolve_run_config(path))
    config["sampling"] = {
        **SAMPLING, "p_clean": 0.0, "p_standard": 0.0,
        "p_benign": 1.0, "p_artifact": 1.0,
        "legacy_anchor_augmentation": False,
    }
    # Explicitly activate the benchmark policy even if a supplied base config is
    # disabled. Specialized curriculum routes require distinct exposure policies.
    pool = set(_masker_artifact_pool())
    required = {
        "realistic_acquisition", "realistic_appearance", "tone_mapping",
        "realistic_bias_field", "realistic_noise", "morphology",
        "orientation", "resolution", "artifact_overlay",
    }
    for entry in config["augmentations"]:
        name = entry["name"]
        entry["enabled"] = name in pool or name in required
        if name in pool:
            entry["settings"]["severity_range"] = list(_severity_range(name, severity))
        if name == "artifact_overlay":
            entry["settings"]["count_range"] = [3, 3]
        if name == "realistic_acquisition":
            entry["settings"]["probability"] = 1.0
    return validate_run_config(config)


def _severity_range(name, requested):
    """Use absolute registry severity, capped by its documented/train-safe range."""
    low, high = requested
    ceiling = min(float(REGISTRY[name].severity_range[1]),
                  float(_ARTIFACT_SEV_RANGE.get(name, (0.0, float("inf")))[1]))
    high = min(high, ceiling)
    low = min(low, high)
    return float(low), float(high)


def build_catalog(*, artifact_severity=(0.15, 0.45), rotation_degrees=10.0,
                  resize_range=(0.9, 1.1), resolution_range=(1.2, 1.8),
                  augmentation_config_path=None):
    """Return JSON-serializable ``specs``, ``excluded``, and benchmark settings.

    Each isolated registry operation runs regardless of its enabled flag in the
    training JSON. Exclusions include replaced geometry, unsafe label contracts,
    and specialized training stages; their reasons are part of the report.
    """
    severity = _range(artifact_severity, "artifact_severity")
    resize = _range(resize_range, "resize_range", minimum=0.01)
    resolution = _range(resolution_range, "resolution_range", minimum=1.0)
    rotation = float(rotation_degrees)
    if not np.isfinite(rotation) or not 0.0 < rotation <= 30.0:
        raise ValueError("rotation_degrees must be in (0, 30] for the mild tier")
    if not 0.75 <= resize[0] <= resize[1] <= 1.25 or resize == (1.0, 1.0):
        raise ValueError("mild resize_range must lie in [0.75, 1.25] and change scale")
    if resolution[1] <= 1.0 or resolution[1] > 3.0:
        raise ValueError("mild resolution_range must include degradation and be <= 3")
    if severity[0] <= 0 or severity[1] > 1.0:
        raise ValueError("artifact_severity must lie in (0, 1]")
    config_path = Path(augmentation_config_path) if augmentation_config_path else (
        Path(__file__).resolve().parents[1] / "configuration/run_config.augmented.json")
    composite = _composite_config(config_path, severity)
    mild = {"rotation_degrees": rotation, "resize_range": list(resize),
            "resolution_range": list(resolution)}
    specs = [{"name": "clean", "tier": "clean", "kind": "clean", "params": {}}]
    for name in ("rotation", "resize", "resolution", "mild_mix"):
        specs.append({"name": name, "tier": "mild", "kind": name, "params": dict(mild)})
    from augmentations.protocols.acquisition import _SEQ
    for sequence in _SEQ:
        specs.append({
            "name": "protocol_" + sequence.lower(), "tier": "protocols", "kind": "protocol",
            "params": {"field": 3.0, "sequence": sequence, "vendor": "Siemens", "recon": "none"},
        })
    excluded = []
    for name, descriptor in sorted(REGISTRY.items()):
        reason = None
        if name == "metal":
            reason = "Composed metal spatially warps anatomy without returning a co-warped target; metal_3d and focal metal cores are included."
        elif descriptor.kind == "meta":
            reason = "Data-integrity/QC defect (constant or NaN/Inf), outside finite-image segmentation augmentation."
        elif descriptor.dims == "2d":
            reason = "2-D-only primitive; no validated volume renderer in this entry."
        elif not descriptor.label_preserving:
            reason = "Moves or removes anatomy without returning its co-transformed target."
        if reason:
            excluded.append({"name": name, "reason": reason})
            continue
        params = {"severity_range": list(_severity_range(name, severity))}
        tier = "full"
        if name in ("t2_3d", "mp2rage_3d"):
            tier = "protocols"
            params["severity_range"] = [1.0, 1.0]
        specs.append({"name": name, "tier": tier, "kind": "registry", "params": params})
    specs.extend([
        {"name": "morphology", "tier": "full", "kind": "morphology", "params": {"strength_range": [0.20, 0.35]}},
        {"name": "label_synthesis", "tier": "full", "kind": "label_synthesis",
         "params": {"strength": 0.6, "realistic": True, "p_pathology": 0.0, "p_pediatric": 0.0}},
        {"name": "realistic_appearance", "tier": "full", "kind": "appearance", "params": {"noise_max": 0.02}},
        {"name": "full_mix", "tier": "full", "kind": "full_mix",
         "params": {"augmentation_config": composite, "max_attempts": 6}},
    ])
    covered = {
        "standard_protocols": "Covered by explicit protocol isolates.",
        "realistic_acquisition": "Covered by protocol isolates and the randomized acquisition in full_mix.",
        "tone_mapping": "Covered by realistic_appearance and full_mix.",
        "realistic_bias_field": "Covered by realistic_appearance/full_mix; registry bias is also isolated.",
        "realistic_noise": "Covered by realistic_appearance/full_mix; noise registry operators are also isolated.",
        "orientation": "Paired mild physical rotation is isolated; broad pose is included in full_mix.",
        "resolution": "Covered by the mild resolution isolate and full_mix.",
        "artifact_overlay": "Every eligible artifact is isolated; full_mix applies multiple artifacts.",
        "saliency_adversarial": "Training-time gradient attack requires a model/loss context; not an array transform.",
        "donor_histogram_transfer": "Requires a separate explicitly selected donor scan/mask pair.",
    }
    isolated = {spec["name"] for spec in specs}
    for name in STAGES:
        if name not in isolated:
            excluded.append({"name": name, "reason": covered.get(
                name, "Specialized MP2RAGE or hard-tail curriculum, not independently exercised by this benchmark; general MP2RAGE and artifact renderers are included.")})
    return {"specs": specs, "excluded": excluded, "settings": {
        **mild, "artifact_severity": list(severity),
        "augmentation_config_path": str(config_path.resolve()),
        "augmentation_config_sources": [str(path) for path in run_config_sources(config_path)],
        "protocols_are_synthetic": True, "catalog_version": 1,
        "severity_semantics": "Absolute operator severity; capped by registry and existing training-safe upper bounds. Some legacy feature cores use separately logged parameters.",
    }}


@contextmanager
def _legacy_random(seed):
    """Isolate legacy primitives that still consult process-global random state.

    Evaluation is sequential. Do not call this module concurrently from threads.
    """
    py_state, np_state = random.getstate(), np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def _paired_affine(scan, mask, affine, *, degrees=0.0, scale=1.0, rng):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    angle = float(rng.uniform(0.5, 1.0) * degrees * rng.choice((-1, 1)))
    forward = Rotation.from_rotvec(axis * np.deg2rad(angle)).as_matrix() * scale
    grid = affine[:3, :3]
    matrix = np.linalg.solve(grid, np.linalg.solve(forward, grid))
    center = (np.asarray(scan.shape) - 1.0) / 2.0
    offset = center - matrix @ center
    out = ndi.affine_transform(scan, matrix, offset, output_shape=scan.shape,
                               order=1, mode="constant", cval=0.0, prefilter=False)
    target = ndi.affine_transform(mask.astype(np.uint8), matrix, offset,
                                  output_shape=mask.shape, order=0,
                                  mode="constant", cval=0, prefilter=False).astype(bool)
    return out, target, {"angle_degrees": angle, "rotation_axis_world": axis.tolist(),
                         "scale": float(scale), "output_to_input_matrix": matrix.tolist(),
                         "output_to_input_offset": offset.tolist()}


def _resolution(scan, bounds, rng):
    axis = int(rng.integers(3))
    factor = float(rng.uniform(*bounds))
    sigma = np.sqrt(max(factor * factor - 1.0, 0.0)) / 2.0
    blurred = ndi.gaussian_filter1d(scan, sigma=max(sigma, 1e-6), axis=axis)
    coarse = list(scan.shape)
    coarse[axis] = max(2, int(round(scan.shape[axis] / factor)))
    down = (np.asarray(scan.shape) - 1.0) / (np.asarray(coarse) - 1.0)
    small = ndi.affine_transform(blurred, np.diag(down), output_shape=tuple(coarse),
                                 order=1, mode="nearest", prefilter=False)
    out = ndi.affine_transform(small, np.diag(1.0 / down), output_shape=scan.shape,
                               order=1, mode="nearest", prefilter=False)
    return out, {"axis": axis, "resolution_factor": factor, "coarse_shape": coarse,
                 "target_unchanged": True}


def _sample_scale(bounds, rng):
    # Keep a nontrivial change instead of a nearly identity default.
    for _ in range(32):
        value = float(rng.uniform(*bounds))
        if abs(value - 1.0) >= min(0.025, max(abs(bounds[0] - 1), abs(bounds[1] - 1)) / 2):
            return value
    return float(bounds[0] if abs(bounds[0] - 1) > abs(bounds[1] - 1) else bounds[1])


def _registry_params(name, scan, mask, spacing, severity, rng):
    params = _focal_kwargs(name, mask, rng, voxel_sizes=tuple(spacing))
    if name in ("metal_dipole", "metal_pileup"):
        locations = np.argwhere(mask)
        center = locations[int(rng.integers(len(locations)))]
        params.update(dict(zip(("cx", "cy", "cz"), map(float, center))))
        params.update(radius=float(max(2.0, min(scan.shape) * rng.uniform(0.04, 0.08))),
                      amp=float(0.15 + severity * 0.5))
    if name in ("brain_contrast", "skull_contrast"):
        params["contrast_factor"] = 1.0 + severity * float(rng.choice((-1, 1)))
    if name in ("brain_intensity", "skull_intensity"):
        params["intensity_change"] = 1.0 + severity * float(rng.choice((-1, 1)))
    if name == "mp2rage_3d":
        params["voxel_sizes"] = tuple(spacing)
    return params


def uses_normalized_source(spec):
    """Whether this condition renders from the normalized original source scan."""
    return spec["kind"] not in ("clean", "rotation", "resize", "resolution", "mild_mix")


def render_augmentation(scan, mask, affine_mm, spec, rng, *, normalized_scan=None):
    """Render one catalog entry and return image, paired target, and replay metadata.

    ``normalized_scan`` optionally caches ``normalize_intensity(scan)`` for repeated
    appearance conditions. It must come from this same, unmodified original scan,
    never a previous augmented image. The caller retains ownership; read-only
    arrays are supported and this function does not mutate the cache. Clean and
    geometric conditions always use the raw scan and ignore the cache.
    """
    scan = np.asarray(scan, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    affine = np.asarray(affine_mm, dtype=np.float64)
    if scan.ndim != 3 or scan.shape != mask.shape or min(scan.shape) < 2:
        raise ValueError("Expected same-grid 3-D scan and mask with every axis >= 2")
    if not np.isfinite(scan).all() or not mask.any():
        raise ValueError("Benchmark input must be finite and contain a nonempty target")
    if affine.shape != (4, 4) or not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-12:
        raise ValueError("affine_mm must be a finite nonsingular 4x4 voxel-to-mm affine")
    generator = np.random.default_rng(rng)
    seed = int(generator.integers(0, 2 ** 32 - 1))
    generator = np.random.default_rng(seed)
    kind, params = spec["kind"], copy.deepcopy(spec.get("params", {}))
    spacing = np.linalg.norm(affine[:3, :3], axis=0).tolist()
    metadata = {"name": spec["name"], "tier": spec["tier"], "render_seed": seed,
                "voxel_sizes_mm": spacing, "native_grid_preserved": True,
                "normalized_for_renderer": uses_normalized_source(spec)}
    target = mask.copy()
    if metadata["normalized_for_renderer"] and normalized_scan is not None:
        if (not isinstance(normalized_scan, np.ndarray)
                or normalized_scan.dtype != np.float32 or normalized_scan.shape != scan.shape):
            raise ValueError("normalized_scan must be a float32 ndarray matching the original scan shape")
        if (not np.isfinite(normalized_scan).all()
                or normalized_scan.min() < 0.0 or normalized_scan.max() > 1.0):
            raise ValueError("normalized_scan must contain finite values in [0, 1]")
        reference = normalized_scan
    else:
        reference = (normalize_intensity(scan) if metadata["normalized_for_renderer"] else scan)
    out = reference.copy()
    with _legacy_random(seed):
        if kind == "clean":
            pass
        elif kind in ("rotation", "resize", "resolution", "mild_mix"):
            operations = []
            if kind in ("rotation", "resize", "mild_mix"):
                out, target, details = _paired_affine(
                    out, target, affine, rng=generator,
                    degrees=params["rotation_degrees"] if kind != "resize" else 0.0,
                    scale=_sample_scale(params["resize_range"], generator) if kind != "rotation" else 1.0)
                operations.append({"operation": "paired_affine", **details})
            if kind in ("resolution", "mild_mix"):
                out, details = _resolution(out, params["resolution_range"], generator)
                operations.append({"operation": "resolution", **details})
            metadata["operations"] = operations
        elif kind == "protocol":
            from augmentations.protocols.acquisition import realistic_acquisition
            out, target, label = realistic_acquisition(
                out, target, generator, geometry=False, cfg=params, clean=True,
                voxel_sizes=spacing, apply_resolution=False)
            metadata.update(protocol=params, render_label=label, synthetic_protocol=True)
        elif kind == "registry":
            severity = float(generator.uniform(*params["severity_range"]))
            kwargs = _registry_params(spec["name"], out, target, spacing, severity, generator)
            out = apply(spec["name"], out, severity=severity, rng=generator, mask=target.copy(), **kwargs)
            metadata.update(severity=severity, operator_params=kwargs)
        elif kind == "morphology":
            from augmentations.anatomy import morph_image
            strength = float(generator.uniform(*params["strength_range"]))
            out, target = morph_image(out, target, generator, strength=strength)
            metadata["strength"] = strength
        elif kind == "label_synthesis":
            from augmentations.label_synthesis import synthesize_from_labels
            out, target = synthesize_from_labels(out, target, generator, **params)
            metadata["synthesis_params"] = params
        elif kind == "appearance":
            from augmentations.appearance import realistic_augment
            out, target = realistic_augment(out, target, generator, geometry=False,
                                            resolution=False, **params)
            metadata["operations"] = ["tone_mapping", "realistic_bias_field", "realistic_noise"]
        elif kind == "full_mix":
            rejected = []
            for attempt in range(int(params["max_attempts"])):
                draw_seed = int(generator.integers(0, 2 ** 32 - 1))
                out, target, label = make_training_sample(
                    reference.copy(), mask.copy(), np.random.default_rng(draw_seed),
                    return_kind=True, voxel_sizes=spacing,
                    augmentation_config=params["augmentation_config"])
                artifacts = label.split("+")[1:]
                if not label.startswith("clean") and len(artifacts) >= 2:
                    metadata.update(render_label=label, applied_artifacts=artifacts,
                                    draw_seed=draw_seed, attempts=attempt + 1,
                                    rejected_draws=rejected)
                    break
                rejected.append({"draw_seed": draw_seed, "label": label,
                                 "reason": "Clean geometry fallback or fewer than two applied artifacts"})
            else:
                raise RuntimeError(f"full_mix failed to produce a valid multi-artifact pair: {rejected}")
        else:
            raise ValueError(f"Unknown augmentation benchmark kind: {kind!r}")
    out, target = np.asarray(out, dtype=np.float32), np.asarray(target, dtype=bool)
    if out.shape != scan.shape or target.shape != mask.shape:
        raise RuntimeError(f"{spec['name']} changed the native grid")
    if not np.isfinite(out).all() or not target.any():
        raise RuntimeError(f"{spec['name']} produced nonfinite intensities or an empty target")
    changed = int(np.count_nonzero(out != reference))
    metadata.update(changed_voxels=changed, image_changed=bool(changed),
                    target_changed=not np.array_equal(target, mask),
                    original_target_voxels=int(mask.sum()), target_voxels=int(target.sum()))
    return np.ascontiguousarray(out), np.ascontiguousarray(target), metadata
