"""Run configuration and per-sample augmentation controls.

The complete JSON is passed through the dataset to each sample, including in
spawned DataLoader workers. ContextVar scopes controls to that sample; loading
configuration never mutates registry globals or relies on inherited processes.
"""
from __future__ import annotations

import copy
import inspect
import math
import re
from contextvars import ContextVar
from functools import wraps
from pathlib import Path

from augmentations.curricula.saliency_adversarial import DEFAULT_CONFIG as _SALIENCY_DEFAULTS
from augmentations.sampling_policy import SAMPLING_DEFAULTS, resolve_sampling_policy

_ACTIVE = ContextVar("masker_augmentation_config", default=None)

STAGES = {
    "saliency_adversarial": dict(_SALIENCY_DEFAULTS),
    "standard_protocols": {"protocols": [
        {"name": name, "config": {"field": 3.0, "sequence": sequence, "vendor": "Siemens", "recon": "none"}}
        for name, sequence in [("MPRAGE", "MPRAGE"), ("MP2RAGE", "MP2RAGE"), ("T2", "SPACE_T2"), ("FLAIR", "FLAIR")]
    ]},
    "realistic_acquisition": {"probability": 0.55, "sequences": ["MPRAGE", "MP2RAGE", "SPGR", "TSE_T2", "SPACE_T2", "FLAIR", "STIR", "DIR", "PSIR", "GRE_SWI"], "fields": [0.064, 0.55, 1.5, 3.0, 5.0, 7.0, 9.4, 10.5, 11.7], "vendors": ["Siemens", "GE", "Philips", "Canon", "UnitedImaging", "Fujifilm"], "reconstructions": ["none", "GRAPPA", "SENSE", "SMS", "CompressedSensing", "DLRecon"]},
    "realistic_appearance": {"noise_max": 0.02},
    "tone_mapping": {"gamma_range": [0.7, 1.5], "contrast_range": [0.8, 1.3], "brightness_range": [-0.10, 0.10], "knot_jitter": 0.22},
    "donor_histogram_transfer": {"probability": 0.5},
    "realistic_bias_field": {"order": 3, "max_strength": 0.7},
    "realistic_noise": {},
    "label_synthesis": {"strength": 1.0, "realistic": True, "p_pathology": 0.22, "p_pediatric": 0.12},
    "morphology": {"probability": 0.7, "strength_range": [0.2, 0.45]},
    "orientation": {"probability": 1.0},
    "resolution": {"probability": 0.6, "factor_range": [1.0, 2.2], "axis_probability": 0.6},
    "artifact_overlay": {"count_range": [1, 2]},
    "mp2rage_morphology": {"strength_range": [0.08, 0.18]},
    "mp2rage_noise_superset": {},
    "mp2rage_superset": {},
    "mp2rage_lower_feature": {},
    "mp2rage_posterior_fossa": {},
    "mp2rage_target_style": {},
    "hard_artifact_tail": {},
}

SAMPLING = SAMPLING_DEFAULTS


def _entries(config):
    return {entry["name"]: entry for entry in config.get("augmentations", [])}


def enabled(name):
    current = _ACTIVE.get()
    if current is None:
        return True
    entry = current.get(name)
    return bool(entry and entry["enabled"])


def settings(name, defaults=None):
    """Return isolated canonical defaults, dynamic caller defaults, then overrides."""
    result = copy.deepcopy(STAGES.get(name, {}))
    result.update(copy.deepcopy(defaults or {}))
    current = _ACTIVE.get()
    if current is not None and name in current:
        result.update(current[name]["settings"])
    return result


def configured_sample(fn):
    """Accept an explicit serializable configuration on each training draw."""
    signature = inspect.signature(fn)

    @wraps(fn)
    def wrapped(*args, augmentation_config=None, **kwargs):
        bound = signature.bind_partial(*args, **kwargs)
        if augmentation_config is None:
            return _checked_sample_result(fn(*args, **kwargs), bound.arguments["scan01"])
        controls = _entries(augmentation_config)
        sampling = dict(augmentation_config.get("sampling", {}))
        sampling.update({key: value for key, value in bound.arguments.items() if key in SAMPLING})
        token = _ACTIVE.set(controls)
        try:
            if not enabled("artifact_overlay"):
                sampling["p_artifact"] = 0.0
            for name in ("mp2rage_superset", "mp2rage_lower_feature", "mp2rage_posterior_fossa", "mp2rage_target_style", "hard_artifact_tail"):
                if not enabled(name):
                    sampling[name + "_fraction"] = 0.0
            if not enabled("standard_protocols"):
                sampling["p_clean"] = sampling.get("p_clean", SAMPLING["p_clean"]) + sampling.get("p_standard", SAMPLING["p_standard"])
                sampling["p_standard"] = 0.0
            for name, value in sampling.items():
                bound.arguments[name] = value
            # A fully disabled configuration is a true identity augmentation.
            if not any(entry["enabled"] for entry in controls.values()):
                import numpy as np
                image = np.asarray(bound.arguments["scan01"], dtype=np.float32).copy()
                mask = np.asarray(bound.arguments["mask"], dtype=bool).copy()
                result = (image, mask, "clean") if bound.arguments.get("return_kind", False) else (image, mask)
                return _checked_sample_result(result, bound.arguments["scan01"])
            return _checked_sample_result(fn(*bound.args, **bound.kwargs), bound.arguments["scan01"])
        except AugmentationError:
            raise
        except Exception as exc:
            raise AugmentationError(
                f"Configured augmentation pipeline failed: {type(exc).__name__}: {exc}"
            ) from exc
        finally:
            _ACTIVE.reset(token)

    return wrapped


def _number(value, name, low=0.0, high=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} must be between {low} and {high}")


def _range(value, name, high=None, integers=False):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must contain [minimum, maximum]")
    for item in value:
        _number(item, name, high=high)
        if integers and not isinstance(item, int):
            raise ValueError(f"{name} must contain integers")
    if value[0] > value[1]:
        raise ValueError(f"{name} minimum exceeds maximum")


def artifact_parameter_names(spec):
    """Explicit keyword parameters, including documented kwargs read by adapters."""
    parameters = inspect.signature(spec.fn).parameters
    reserved = {"arr", "vol", "volume", "image", "scan01", "rng", "mask", "severity"}
    result = {name for name, item in parameters.items()
              if name not in reserved and item.kind not in (item.VAR_KEYWORD, item.VAR_POSITIONAL)}
    return (result | set(spec.extra_parameters)) - set(spec.excluded_parameters) - reserved


class AugmentationError(RuntimeError):
    """A pipeline/operator failure that must abort a run, not skip a data record."""


def _checked_sample_result(result, scan01):
    """Validate the final compound-stage contract before normalization/training."""
    import numpy as np
    if not isinstance(result, (tuple, list)) or len(result) not in (2, 3):
        raise AugmentationError("Augmentation pipeline must return an image and a mask")
    expected = np.shape(scan01)
    for name, value in zip(("image", "mask"), result[:2]):
        array = np.asarray(value)
        if array.shape != expected:
            raise AugmentationError(
                f"Augmentation pipeline returned {name} shape {array.shape}; expected {expected}")
        if not np.isfinite(array).all():
            raise AugmentationError(f"Augmentation pipeline returned nonfinite {name} values")
    return result


def _finite_parameters(value, name):
    """Reject non-JSON and nonfinite values, including nested numeric arrays."""
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _finite_parameters(item, f"{name}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{name} keys must be strings")
            _finite_parameters(item, f"{name}.{key}")
    else:
        raise ValueError(f"{name} must contain JSON values")


def validate_artifact_parameters(name, params):
    """Check the exposed 3-D parameter contracts without rendering a volume."""
    _finite_parameters(params, f"{name}.params")
    boolean_keys = {"magnitude", "batched_3d", "anterior_low", "round_discs", "magnitude_noise"}
    integer_keys = {"n_poses", "n_ghosts", "n_spikes", "n_beams", "n_discs", "levels", "k", "x_threshold", "y_threshold"}
    positive_keys = {"radius", "radius_mm", "size_scale", "scale", "lut_warp", "ratio", "etl", "sigma_lr", "sigma_si", "decay_frac", "outside_decay_frac", "ripple_frac"}
    nonnegative_keys = {"sigma_grain", "dither", "master_t", "bright_thr", "base_amp", "disc_gain", "floor_amp", "outside_amp", "outside_reach_frac", "reach_frac", "step_frac", "ghost_step_frac", "tail_floor", "decay"}
    unit_keys = {"slice_gap", "drift", "tissue_weight", "depth"}
    from augmentations.registry import REGISTRY
    nullable = {key for key, parameter in inspect.signature(REGISTRY[name].fn).parameters.items()
                if parameter.default is None}
    for key, value in params.items():
        label = f"{name}.params.{key}"
        if key in boolean_keys:
            if not isinstance(value, bool):
                raise ValueError(f"{label} must be boolean")
        elif key in {"axis", "pe_axis", "axial_axis", "si_axis", "lr_axis"}:
            if value is None and (key in nullable or key == "axial_axis"):
                continue
            low, high = ((0, 1) if key == "pe_axis" and name in {"motion", "fse_echo_train"} else (-3, 2))
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{label} must be an integer in [{low}, {high}]")
        elif value is None and key in nullable:
            continue
        elif key in integer_keys:
            low = 0 if key in {"x_threshold", "y_threshold"} else (2 if key in {"levels", "k"} else 1)
            if type(value) is not int or value < low:
                raise ValueError(f"{label} must be an integer >= {low}")
        elif key in positive_keys:
            _number(value, label)
            if value <= 0:
                raise ValueError(f"{label} must be positive")
        elif key in nonnegative_keys:
            _number(value, label)
        elif key in unit_keys:
            _number(value, label, high=1.0)
        elif key == "percentile":
            _number(value, label, high=100.0)
        elif key == "asym":
            _number(value, label, low=-1.0, high=1.0)
        elif key in {"cx", "cy", "cz"}:
            # These are voxel coordinates; only metal_3d.loc uses FOV fractions.
            _number(value, label, low=-math.inf)
        elif key == "voxel_length_range":
            _range(value, label)
            if value[0] <= 0:
                raise ValueError(f"{label} minimum must be positive")
        elif key in {"loc", "voxel_sizes"}:
            if not isinstance(value, list) or len(value) != 3:
                raise ValueError(f"{label} must contain three numbers")
            for item in value:
                _number(item, label, high=1.0 if key == "loc" else None)
                if key == "voxel_sizes" and item <= 0:
                    raise ValueError(f"{label} values must be positive")
        elif key in {"partial", "pos_or_neg_x", "pos_or_neg_y"}:
            if type(value) not in (int, bool) or value not in (0, 1):
                raise ValueError(f"{label} must be 0 or 1")
    enums = {
        ("noise", "mode"): {None, "thermal", "rician", "structured"},
        ("dropout", "mode"): {"slice", "patch"},
        ("bias", "field_kind"): {None, "grid", "coil", "elliptical"},
        ("fse_echo_train", "order"): {None, "linear", "centric"},
        ("eye_ghosting", "beam_mode"): {"beams", "vertical", "vertical_beams", "streak", "discrete", "discs"},
    }
    for (operator, key), choices in enums.items():
        if name == operator and key in params:
            value = params[key]
            if not isinstance(value, (str, type(None))) or value not in choices:
                raise ValueError(f"{name}.params.{key} must be one of {sorted(map(str, choices))}")
    if name == "motion":
        from augmentations.artifacts.kspace import _motion_slice_worker_count
        _motion_slice_worker_count(params.get("slice_workers"), params.get("batched_3d", False))
    if name == "eye_ghosting":
        axes = [params[key] for key in ("axis", "si_axis", "lr_axis") if key in params]
        if len({axis % 3 for axis in axes}) != len(axes):
            raise ValueError("eye_ghosting axis, si_axis, and lr_axis must be distinct")
    if name == "create_ring" and params.get("partial"):
        if any(params.get(key) is None for key in ("pos_or_neg_x", "pos_or_neg_y", "x_threshold", "y_threshold")):
            raise ValueError("create_ring partial=1 requires both directions and both thresholds")


def validate_run_config(config):
    """Validate names, parameter typos and route constraints before model loading."""
    from augmentations import REGISTRY
    from augmentations.pipeline import _MASKER_ARTIFACT_NAMES
    from augmentations.protocols.acquisition import _SEQ, _FIELD, _VENDORS, _RECON, _MP2RAGE_FIELDS
    from augmentations.curricula.saliency_adversarial import normalize_config

    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("run_config schema_version must be 1")
    unknown = set(config) - {"schema_version", "description", "training", "sampling", "augmentations"}
    if unknown:
        raise ValueError(f"Unknown run configuration keys: {sorted(unknown)}")
    training = config.get("training", {})
    if not isinstance(training, dict) or training.get("model_type", "masking") != "masking":
        raise ValueError("This condensed workflow supports model_type='masking'")
    if training.get("synth_online", True) is not True:
        raise ValueError("This workflow requires synth_online=true so JSON augmentations apply during training")
    if "synth_kwargs" in training or "saliency_adversarial" in training:
        raise ValueError("Configure sampling and saliency in their JSON sections, not training.synth_kwargs or training.saliency_adversarial")
    if type(training.get("allow_unsafe_init", False)) is not bool:
        raise ValueError("training.allow_unsafe_init must be a JSON boolean")
    subject_regex = training.get("subject_id_regex")
    if subject_regex is not None:
        if not isinstance(subject_regex, str) or not subject_regex:
            raise ValueError("training.subject_id_regex must be a nonempty regex string or null")
        try:
            re.compile(subject_regex)
        except re.error as exc:
            raise ValueError(f"Invalid training.subject_id_regex: {exc}") from exc
    from training.config import TrainingConfig
    allowed_training = TrainingConfig.field_names() | {"resume_from", "init_from"}
    if set(training) - allowed_training:
        raise ValueError(f"Unknown training settings: {sorted(set(training) - allowed_training)}")
    sampling = config.get("sampling", {})
    if not isinstance(sampling, dict) or set(sampling) - set(SAMPLING):
        raise ValueError("Unknown sampling setting; use keys from the supplied run_config.json")
    merged = {**SAMPLING, **sampling}
    for key, value in merged.items():
        if key == "legacy_anchor_augmentation":
            if not isinstance(value, bool):
                raise ValueError(f"{key} must be true or false")
        elif value is not None:
            _number(value, key, high=1.0)
    if sum(merged[key] for key in ("p_clean", "p_standard", "p_benign")) > 1.0 + 1e-12:
        raise ValueError("p_clean + p_standard + p_benign must be <= 1; the remainder is label synthesis")
    entries = config.get("augmentations")
    if not isinstance(entries, list):
        raise ValueError("augmentations must be a list")
    known = set(STAGES) | set(REGISTRY)
    names = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) - {"name", "enabled", "settings", "kind", "supported", "notes"}:
            raise ValueError("Each augmentation needs name, enabled, and settings")
        name = entry.get("name")
        if name not in known:
            raise ValueError(f"Unknown augmentation {name!r}")
        if not isinstance(entry.get("enabled"), bool) or not isinstance(entry.get("settings"), dict):
            raise ValueError(f"{name}: enabled must be boolean and settings must be an object")
        names.append(name)
        options = entry["settings"]
        allowed = set(STAGES[name]) if name in STAGES else {"severity_range", "params"}
        if set(options) - allowed:
            raise ValueError(f"{name}: unknown settings {sorted(set(options) - allowed)}")
        if name == "donor_histogram_transfer" and entry["enabled"]:
            raise ValueError("donor_histogram_transfer requires an explicit donor pair; the condensed trainer does not supply one, so leave this program disabled")
        if name in REGISTRY:
            spec = REGISTRY[name]
            supported = name in _MASKER_ARTIFACT_NAMES and spec.label_preserving and spec.dims in ("3d", "either")
            if entry["enabled"] and not supported:
                raise ValueError(f"{name} is a low-level program unsupported in the 3D mask artifact overlay; leave disabled")
            _range(options.get("severity_range", [0.3, 0.85]), name + ".severity_range")
            params = options.get("params", {})
            if not isinstance(params, dict):
                raise ValueError(f"{name}.params must be an object")
            accepted = artifact_parameter_names(spec)
            if set(params) - accepted:
                raise ValueError(f"{name}: unsupported parameters {sorted(set(params) - accepted)}; accepted: {sorted(accepted)}")
            validate_artifact_parameters(name, params)
        for key, value in options.items():
            if key in ("probability", "p_pathology", "p_pediatric", "axis_probability"):
                _number(value, name + "." + key, high=1.0)
            elif key in ("noise_max", "strength", "max_strength", "knot_jitter"):
                _number(value, name + "." + key)
            elif key == "order":
                _number(value, name + ".order", low=1.0, high=8)
                if not isinstance(value, int):
                    raise ValueError(f"{name}.order must be an integer")
            elif key == "realistic" and not isinstance(value, bool):
                raise ValueError(f"{name}.realistic must be boolean")
            elif key in ("strength_range", "gamma_range", "contrast_range", "factor_range"):
                _range(value, name + "." + key)
                if key == "factor_range" and value[0] < 1.0:
                    raise ValueError("resolution.factor_range minimum must be >= 1")
            elif key == "brightness_range":
                if not isinstance(value, list) or len(value) != 2 or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in value) or value[0] > value[1]:
                    raise ValueError("tone_mapping.brightness_range must be an ordered finite pair")
            elif key == "count_range":
                _range(value, name + "." + key, integers=True)
        if name == "saliency_adversarial":
            normalize_config(options)
        if name == "standard_protocols":
            protocols = options.get("protocols", STAGES[name]["protocols"])
            if not isinstance(protocols, list) or not protocols:
                raise ValueError("standard_protocols.protocols must be a nonempty list")
            for protocol in protocols:
                if not isinstance(protocol, dict) or set(protocol) != {"name", "config"} or not isinstance(protocol["name"], str):
                    raise ValueError("Each standard protocol needs a name and config")
                acquisition = protocol["config"]
                if not isinstance(acquisition, dict) or set(acquisition) != {"field", "sequence", "vendor", "recon"}:
                    raise ValueError("Protocol config needs field, sequence, vendor, recon")
                if acquisition["field"] not in _FIELD or acquisition["sequence"] not in _SEQ or acquisition["vendor"] not in _VENDORS or acquisition["recon"] not in _RECON:
                    raise ValueError("Unknown standard protocol field/sequence/vendor/recon")
                if acquisition["sequence"] == "MP2RAGE" and acquisition["field"] not in _MP2RAGE_FIELDS:
                    raise ValueError("MP2RAGE protocol requires supported high-field acquisition")
        if name == "realistic_acquisition":
            defaults = STAGES[name]
            for key, available in (("sequences", _SEQ), ("fields", _FIELD), ("vendors", _VENDORS), ("reconstructions", _RECON)):
                values = options.get(key, defaults[key])
                if not isinstance(values, list) or not values or len(values) != len(set(values)) or any(value not in available for value in values):
                    raise ValueError(f"realistic_acquisition.{key} must be a nonempty unique list of supported values")
    if len(names) != len(set(names)):
        raise ValueError("Duplicate augmentation names")
    missing = known - set(names)
    if missing:
        raise ValueError(f"Configuration must list every augmentation; missing {sorted(missing)}")
    controls = _entries(config)
    for name in ("mp2rage_superset", "mp2rage_lower_feature", "mp2rage_posterior_fossa", "mp2rage_target_style", "hard_artifact_tail"):
        if not controls[name]["enabled"]:
            merged[name + "_fraction"] = 0.0
    spacing = training.get("conform_mm")
    policy = resolve_sampling_policy(
        merged, voxel_sizes=(spacing,) * 3 if spacing is not None else None,
        require_geometry=True, reject_legacy_curriculum=True)
    dedicated = policy.dedicated_fraction
    acquisition = {**STAGES["realistic_acquisition"], **controls["realistic_acquisition"]["settings"]}
    if ("MP2RAGE" in acquisition["sequences"] or (dedicated or 0) > 0) and not set(acquisition["fields"]) & set(_MP2RAGE_FIELDS):
        raise ValueError("MP2RAGE sampling needs at least one supported high-field value")
    if dedicated is not None:
        if controls["realistic_acquisition"]["enabled"] and not set(acquisition["sequences"]) - {"MP2RAGE"}:
            raise ValueError("Dedicated curricula require at least one non-MP2RAGE broad acquisition sequence")
        protocols = controls["standard_protocols"]["settings"].get("protocols", STAGES["standard_protocols"]["protocols"])
        if controls["standard_protocols"]["enabled"] and not any(item["config"]["sequence"] != "MP2RAGE" for item in protocols):
            raise ValueError("Dedicated curricula require a non-MP2RAGE standard protocol")
    return config


def load_run_config(path):
    """Return Trainer keyword arguments, resolving paths relative to the JSON."""
    path = Path(path).resolve()
    from configuration.presets import resolve_run_config
    config = validate_run_config(resolve_run_config(path))
    result = copy.deepcopy(config.get("training", {}))
    for key in ("data_dir", "model_out_path", "results_out_path", "test_dir", "init_from", "resume_from"):
        value = result.get(key)
        if isinstance(value, str) and value:
            candidate = Path(value)
            result[key] = str(candidate if candidate.is_absolute() else path.parent / candidate)
    entries = _entries(config)
    saliency = entries["saliency_adversarial"]
    result["saliency_adversarial"] = copy.deepcopy(saliency["settings"]) if saliency["enabled"] else None
    result["model_type"] = "masking"
    if result.get("synth_online", True) is not True:
        raise ValueError("This workflow requires synth_online=true so JSON augmentations apply during training")
    result["synth_online"] = True
    effective_sampling = {key: value for key, value in {**SAMPLING, **config.get("sampling", {})}.items() if value is not None}
    for name in ("mp2rage_superset", "mp2rage_lower_feature", "mp2rage_posterior_fossa", "mp2rage_target_style", "hard_artifact_tail"):
        key = name + "_fraction"
        if not entries[name]["enabled"] or not effective_sampling.get(key):
            effective_sampling.pop(key, None)
    result["synth_kwargs"] = {
        **result.get("synth_kwargs", {}),
        **effective_sampling,
        "augmentation_config": copy.deepcopy({key: config[key] for key in ("schema_version", "sampling", "augmentations") if key in config}),
    }
    return result
