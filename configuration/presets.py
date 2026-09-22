"""Expand a shared augmentation preset into a complete, portable run configuration.

Preset paths are relative to the run JSON. Training paths remain relative to that
same run JSON and are resolved by ``augmentations.config.load_run_config``.
Resolved configs contain the full inventory so checkpoints never depend on a
preset file remaining available or unchanged.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path


def _read_object(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a JSON object: {path}")
    return value


def _merge_settings(base: dict, overrides: dict) -> dict:
    """Merge nested settings dictionaries; scalar values and lists replace defaults."""
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_settings(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def run_config_sources(path) -> tuple[Path, ...]:
    """Return input files that reports and other outputs must not overwrite."""
    path = Path(path).resolve()
    config = _read_object(path)
    preset_name = config.get("augmentation_preset")
    if preset_name is None:
        return (path,)
    if not isinstance(preset_name, str) or not preset_name.strip():
        raise ValueError("augmentation_preset must be a nonempty JSON path")
    preset_path = Path(preset_name)
    if not preset_path.is_absolute():
        preset_path = path.parent / preset_path
    return path, preset_path.resolve()


def resolve_run_config(path) -> dict:
    """Read a compact preset-based run or an existing full-inventory run JSON.

    ``augmentation_overrides`` maps augmentation names to ``enabled`` and/or
    ``settings`` changes. Unknown names and fields fail immediately; the normal
    configuration validator subsequently checks setting names and values.
    """
    path = Path(path).resolve()
    config = _read_object(path)
    preset_name = config.pop("augmentation_preset", None)
    overrides = config.pop("augmentation_overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("augmentation_overrides must be an object keyed by augmentation name")

    if preset_name is not None:
        if not isinstance(preset_name, str) or not preset_name.strip():
            raise ValueError("augmentation_preset must be a nonempty JSON path")
        if "augmentations" in config:
            raise ValueError("Use augmentation_preset or a full augmentations list, not both")
        preset_path = Path(preset_name)
        if not preset_path.is_absolute():
            preset_path = path.parent / preset_path
        preset = _read_object(preset_path)
        if preset.get("schema_version") != 1:
            raise ValueError("Augmentation preset schema_version must be 1")
        unknown = set(preset) - {"schema_version", "description", "sampling", "augmentations"}
        if unknown:
            raise ValueError(f"Unknown augmentation preset keys: {sorted(unknown)}")
        sampling = preset.get("sampling", {})
        run_sampling = config.get("sampling", {})
        if not isinstance(sampling, dict) or not isinstance(run_sampling, dict):
            raise ValueError("sampling must be an object")
        config["sampling"] = {**sampling, **run_sampling}
        config["augmentations"] = copy.deepcopy(preset.get("augmentations"))

    if overrides:
        entries = config.get("augmentations")
        if not isinstance(entries, list):
            raise ValueError("augmentation_overrides requires an augmentation inventory")
        by_name = {}
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
                raise ValueError("Every augmentation entry needs a name")
            if entry["name"] in by_name:
                raise ValueError(f"Duplicate augmentation name: {entry['name']}")
            by_name[entry["name"]] = entry
        for name, changes in overrides.items():
            if name not in by_name:
                raise ValueError(f"Unknown augmentation override: {name}")
            if not isinstance(changes, dict) or set(changes) - {"enabled", "settings"}:
                raise ValueError(f"{name} overrides may contain only enabled and settings")
            entry = by_name[name]
            if "enabled" in changes:
                entry["enabled"] = changes["enabled"]
            if "settings" in changes:
                settings = changes["settings"]
                if not isinstance(settings, dict):
                    raise ValueError(f"{name}.settings overrides must be an object")
                base_settings = entry.get("settings", {})
                if not isinstance(base_settings, dict):
                    raise ValueError(f"{name}.settings in the preset must be an object")
                entry["settings"] = _merge_settings(base_settings, settings)
    return config
