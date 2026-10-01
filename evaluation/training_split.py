"""Recover the saved training run's test cohort without starting training.

The checkpoint stores a fingerprint of all three splits, rather than a test
manifest. Recovery repeats discovery and subject partitioning and requires an
exact fingerprint match before exporting any evaluation inputs.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any


_SPLIT_KEYS = (
    "data_dir", "synth_online", "scan_glob", "mask_suffix", "seed",
    "val_fraction", "test_fraction", "test_dir", "patch_size", "subject_id_regex",
)
_RESTAGE = (
    "Run notebook section 5 (stage the training data) at the original saved "
    "data_dir, including the original masks and manifest if applicable. "
    "No training cell needs to run."
)


def _saved_config(model_path: Path, config_path=None) -> dict[str, Any]:
    from inference.checkpoints import (
        MaskingCheckpoint, _checkpoint_name, _find_sidecar_config,
        _read_embedded_masking_config, _read_masking_config,
    )
    from imaging.metadata import preproc_config_hash

    checkpoint = MaskingCheckpoint(model_path)
    payload = checkpoint.read()  # weights_only=True; never retry with unsafe pickle.
    embedded = _read_embedded_masking_config(model_path, checkpoint=checkpoint)
    if embedded is None and isinstance(payload, dict):
        if "preproc_config" in payload:
            raise ValueError("Checkpoint has invalid masking preprocessing metadata")
        if payload.get("format") == "train_resume_v1":
            cfg = payload.get("config")
            if not isinstance(cfg, dict) or cfg.get("model_type") != "masking":
                raise ValueError("Training checkpoint has invalid masking configuration")
            embedded = dict(cfg, __source__=f"{model_path}::config")

    explicit = None
    if config_path is not None:
        explicit = _read_masking_config(Path(config_path))
        if explicit is None:
            raise ValueError(f"Not a masking training-results JSON: {config_path}")
        recorded = explicit.get("__model_path__")
        if recorded and os.path.normcase(_checkpoint_name(recorded)) != os.path.normcase(model_path.name):
            raise ValueError(f"Training results {config_path} identify a different model: {recorded}")

    if embedded is not None:
        if explicit is not None:
            compared = (*_SPLIT_KEYS, "data_split_fingerprint")
            conflicts = [key for key in compared if explicit.get(key) != embedded.get(key)]
            if preproc_config_hash(explicit) != preproc_config_hash(embedded):
                conflicts.append("preprocessing configuration")
            if conflicts:
                raise ValueError(
                    "Training results conflict with the checkpoint's embedded metadata: "
                    + ", ".join(conflicts))
        return embedded

    sidecar = explicit or _find_sidecar_config(model_path, checkpoint=checkpoint)
    if sidecar is None:
        raise ValueError(
            "This model has no saved training split metadata. Supply its matching training "
            "results JSON via --config; a current run configuration cannot prove the held-out split.")
    recorded = sidecar.get("__model_path__")
    if not recorded or os.path.normcase(_checkpoint_name(recorded)) != os.path.normcase(model_path.name):
        raise ValueError(
            "Legacy training results must identify this checkpoint with a matching model_path; "
            "cannot verify that this results file belongs to the selected model.")
    return sidecar


def _split_settings(config):
    missing = [key for key in (*_SPLIT_KEYS, "data_split_fingerprint") if key not in config]
    if missing:
        raise ValueError(
            "Saved training metadata is missing the settings needed to verify the original "
            "held-out test split: " + ", ".join(missing)
            + ". Use the original run's complete checkpoint/results; do not guess from current notebook settings.")
    fingerprint = config["data_split_fingerprint"]
    if not isinstance(fingerprint, str) or re.fullmatch(r"[0-9a-f]{64}", fingerprint) is None:
        raise ValueError(
            "Saved training metadata has no valid data_split_fingerprint; the original "
            "held-out test split cannot be verified automatically.")
    for key in ("data_dir", "scan_glob", "mask_suffix"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"Saved split setting {key} must be a nonempty string")
    if type(config["synth_online"]) is not bool:
        raise ValueError("Saved synth_online must be a boolean")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("Saved split seed must be a nonnegative integer")
    for key in ("val_fraction", "test_fraction"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Saved {key} must be a finite fraction between 0 and 1")
    test_dir = config["test_dir"]
    if test_dir is not None and (not isinstance(test_dir, str) or not test_dir.strip()):
        raise ValueError("Saved test_dir must be a nonempty string or null")
    pattern = config["subject_id_regex"]
    if pattern is not None and (not isinstance(pattern, str) or not pattern):
        raise ValueError("Saved subject_id_regex must be a nonempty string or null")
    settings = SimpleNamespace(**{key: config[key] for key in _SPLIT_KEYS})
    settings.data_dir = Path(settings.data_dir)
    settings.test_dir = Path(test_dir) if test_dir is not None else None
    settings._subject_pattern = re.compile(pattern) if pattern else None
    return settings


def _recover_splits(settings):
    import numpy as np
    from training.data import _records, _preflight, _prepare_subject_records

    for label, directory in (("training", settings.data_dir), ("test", settings.test_dir)):
        if directory is not None and not directory.is_dir():
            raise FileNotFoundError(f"Saved {label} data directory is missing: {directory}. {_RESTAGE}")
    try:
        if settings.synth_online:
            from augmentations.pipeline import discover_pairs
            records = discover_pairs(settings.data_dir, settings.scan_glob, settings.mask_suffix)
        else:
            records = _records(settings)
        records = _prepare_subject_records(settings, _preflight(settings, records, "train"), "train")
        external_test = _records(settings, settings.test_dir) if settings.test_dir is not None else None
        if external_test is not None:
            external_test = _prepare_subject_records(
                settings, _preflight(settings, external_test, "test"), "test")
            overlap = {r["subject_id"] for r in records} & {r["subject_id"] for r in external_test}
            if overlap:
                raise ValueError(f"External test subjects overlap training/validation subjects: {sorted(overlap)}")
    except (FileNotFoundError, RuntimeError) as exc:
        raise RuntimeError(f"Cannot recover the saved training data: {exc}\n{_RESTAGE}") from exc

    # Keep this arithmetic identical to training.data.prepare_data. The final full
    # fingerprint also catches changes to membership, discovery or subject identity.
    groups = {}
    for i, record in enumerate(records):
        groups.setdefault(record["subject_id"], []).append(i)
    keys = sorted(groups)
    if len(keys) < 2 or len(records) <= 1:
        raise ValueError("At least two distinct subjects are required for the original training/validation split")
    rng = np.random.default_rng(settings.seed)
    perm = [keys[j] for j in rng.permutation(len(keys))]
    n = len(perm)
    carve_test = settings.test_fraction if external_test is None else 0.0
    n_test = min(int(round(carve_test * n)), max(0, n - 2)) if carve_test > 0 else 0
    n_val = max(1, int(round(settings.val_fraction * n)))
    n_val = min(n_val, max(1, n - n_test - 1))
    test_keys = set(perm[:n_test])
    val_keys = set(perm[n_test:n_test + n_val])
    test_idx = [i for k in sorted(test_keys) for i in groups[k]]
    val_idx = [i for k in sorted(val_keys) for i in groups[k]]
    train_idx = [i for k in keys if k not in test_keys and k not in val_keys for i in groups[k]]
    train = [records[i] for i in train_idx]
    validation = [records[i] for i in val_idx]
    test = external_test if external_test is not None else [records[i] for i in test_idx]
    return {"train": train, "validation": validation or train, "test": test}


def export_training_test_manifest(model_path, output_path, *, config_path=None) -> dict[str, Any]:
    """Write a new evaluation manifest only after proving the saved split matches.

    Exact verification requires the original data paths and bytes because both
    are part of the training fingerprint. This never instantiates a trainer,
    synthesizes samples, or changes checkpoint/data files.
    """
    model_path = Path(model_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().absolute()
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing manifest or input: {output_path}")
    if output_path.resolve() == model_path or (
            config_path is not None and output_path.resolve() == Path(config_path).expanduser().resolve()):
        raise ValueError("Manifest output must be different from the model and configuration inputs")
    config = _saved_config(model_path, config_path)
    settings = _split_settings(config)
    splits = _recover_splits(settings)
    from training.data import _split_fingerprint
    actual = _split_fingerprint(**splits)
    expected = config["data_split_fingerprint"]
    if actual != expected:
        raise ValueError(
            "Training split fingerprint mismatch: the original held-out test split could not "
            "be verified. Data contents, paths, manifest records, or subject membership have "
            f"changed (saved={expected}, reconstructed={actual}). {_RESTAGE} "
            "Do not evaluate the full training cohort as a substitute.")
    if not splits["test"]:
        raise ValueError(
            "This training run has no held-out test cases (test_fraction was zero or rounded "
            "to zero, and no external test set was saved). It is not valid to substitute the "
            "training or validation split for a held-out test set.")
    cases = []
    seen_ids, seen_scans = set(), set()
    for record in splits["test"]:
        subject, case_id = record["subject_id"], record["case_id"]
        identity = f"{subject}::{case_id}"
        scan = str(Path(record["scan"]).resolve())
        mask = str(Path(record["mask"]).resolve())
        if identity in seen_ids or scan in seen_scans:
            raise ValueError(f"Duplicate test case identity or scan: {identity}")
        if scan == mask or os.path.samefile(scan, mask):
            raise ValueError(f"Test case uses the same file for scan and mask: {identity}")
        if output_path.resolve() in (Path(scan), Path(mask)):
            raise ValueError("Manifest output must not replace a test input")
        seen_ids.add(identity)
        seen_scans.add(scan)
        cases.append({"id": identity, "scan": scan, "mask": mask,
                      "subject_id": subject, "case_id": case_id})
    report = {
        "cases": cases,
        "provenance": {
            "model_path": str(model_path), "config_source": config["__source__"],
            "data_dir": str(settings.data_dir),
            "test_dir": str(settings.test_dir) if settings.test_dir is not None else None,
            "data_split_fingerprint": expected,
            "reconstructed_data_split_fingerprint": actual,
            "counts": {name: len(records) for name, records in splits.items()},
            "subject_counts": {name: len({r["subject_id"] for r in records})
                               for name, records in splits.items()},
        },
    }
    serialized = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as stream:
        stream.write(serialized)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export the fingerprint-verified test split from a saved training run")
    parser.add_argument("--model", required=True, help="Saved masking checkpoint")
    parser.add_argument("--output", required=True, help="New evaluation manifest JSON path")
    parser.add_argument("--config", help="Matching training-results JSON, for legacy models")
    args = parser.parse_args(argv)
    try:
        report = export_training_test_manifest(args.model, args.output, config_path=args.config)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Training test-split recovery failed: {exc}\n")
    counts = report["provenance"]["counts"]
    print(f"Verified original training split: {counts['train']} train, "
          f"{counts['validation']} validation, {counts['test']} held-out test cases.")
    print(f"Held-out test manifest: {Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
