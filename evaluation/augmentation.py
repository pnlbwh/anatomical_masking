"""Reproducible held-out augmentation benchmark using deployed BrainMasker inference.

The Colab cell calls this module in a fresh process. Inputs must be explicitly
held out; no training-directory or split-seed guess is made. Each augmentation
is measured in isolation, plus named mixed conditions. Tier summaries are
cumulative (mild -> protocols -> full), without adding clean scans to those
averages. References may guide synthetic rendering, never model normalization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np

from evaluation.standard import (_check_grid, _load_volume, binary_metrics, discover_cases,
                      read_manifest)
from inference.masker import (BrainMasker, add_inference_arguments, atomic_write_text,
                           ensure_output_paths, inference_kwargs)
from imaging.geometry import nifti_affine_mm
from evaluation.transforms import build_catalog, render_augmentation


TIERS = ("clean", "mild", "protocols", "full")
TIER_LABELS = {
    "clean": "Clean scans",
    "mild": "Mild geometry and resolution",
    "protocols": "Mild + synthetic MRI protocols",
    "full": "All supported conditions + mixed stress",
}


def trial_seed(seed, case_id, augmentation, repeat):
    """Stable across processes, ordering, case limits, and condition subsets."""
    payload = json.dumps([int(seed), str(case_id), augmentation, int(repeat)])
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little")


def tier_members(specs, tier):
    allowed = {"clean"} if tier == "clean" else set(TIERS[1:TIERS.index(tier) + 1])
    return [spec["name"] for spec in specs if spec["tier"] in allowed]


def _statistics(values):
    a = np.asarray(values, dtype=float)
    return {
        "mean_dice": float(a.mean()) if len(a) else None,
        "std_dice": float(a.std()) if len(a) else None,
        "median_dice": float(np.median(a)) if len(a) else None,
        "min_dice": float(a.min()) if len(a) else None,
        "p10_dice": float(np.percentile(a, 10)) if len(a) else None,
        "fraction_at_least_099": float(np.mean(a >= 0.99)) if len(a) else None,
    }


def summarize(rows, specs, case_ids, repeats):
    """Case-balanced scores, requiring all planned trials for each scored case.

    A failed or pending trial excludes that case from its condition/tier mean.
    Counts expose incomplete coverage; failures never become clean samples or
    disappear from the reports. Repeat means precede condition and case means.
    """
    def group(names):
        selected = [r for r in rows if r["augmentation"] in names]
        expected_per_case = sum(1 if name == "clean" else repeats for name in names)
        values, available_values = [], []
        for case_id in case_ids:
            case_rows = [r for r in selected if r["id"] == case_id]
            available_conditions = []
            for name in names:
                draws = [r["dice"] for r in case_rows if r["augmentation"] == name and r["status"] == "ok"]
                if draws:
                    available_conditions.append(float(np.mean(draws)))
            if available_conditions:
                available_values.append(float(np.mean(available_conditions)))
            if len(case_rows) != expected_per_case or any(r["status"] != "ok" for r in case_rows):
                continue
            condition_means = [np.mean([r["dice"] for r in case_rows
                                        if r["augmentation"] == name]) for name in names]
            values.append(float(np.mean(condition_means)))
        good = [r for r in selected if r["status"] == "ok"]
        expected = len(case_ids) * expected_per_case
        return {"conditions": len(names), "expected_trials": expected,
                "successful_trials": len(good),
                "failed_trials": sum(r["status"] == "failed" for r in selected),
                "unchanged_trials": sum(r["status"] == "unchanged" for r in selected),
                "pending_trials": expected - len(selected),
                "complete_cases": len(values), "total_cases": len(case_ids),
                "available_cases": len(available_values),
                "available_mean_dice": float(np.mean(available_values)) if available_values else None,
                "complete": len(good) == expected and expected > 0,
                "review_flagged_trials": sum(bool(r.get("review_flag")) for r in good),
                **_statistics(values)}

    augmentation = [{"augmentation": spec["name"], "introduced_in": spec["tier"],
                     **group([spec["name"]])} for spec in specs]
    tiers = [{"tier": tier, "label": TIER_LABELS[tier],
              "augmentations": tier_members(specs, tier),
              **group(tier_members(specs, tier))} for tier in TIERS
             if tier_members(specs, tier)]
    return tiers, augmentation


def _csv_text(rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _save_reports(out, report, protected):
    summaries = [{"level": "tier", "name": r["tier"], **r} for r in report["tiers"]]
    summaries += [{"level": "augmentation", "name": r["augmentation"], **r}
                  for r in report["augmentations"]]
    summary_fields = ["level", "name", "conditions", "mean_dice", "available_mean_dice", "available_cases", "std_dice", "median_dice",
                      "min_dice", "p10_dice", "fraction_at_least_099", "complete_cases",
                      "total_cases", "expected_trials", "successful_trials", "failed_trials",
                      "unchanged_trials", "pending_trials", "complete", "review_flagged_trials"]
    case_fields = ["id", "scan", "reference_mask", "augmentation", "introduced_in", "repeat",
                   "seed", "status", "dice", "iou", "precision", "recall", "review_flag", "error"]
    contents = {
        "results.json": json.dumps(report, indent=2, allow_nan=False) + "\n",
        "cases.csv": _csv_text(report["cases"], case_fields),
        "summary.csv": _csv_text(summaries, summary_fields),
    }
    for filename, content in contents.items():
        atomic_write_text(out / filename, content, protected_paths=protected, overwrite=True)


def evaluate_augmentations(model_path, *, manifest=None, data_dir=None,
                          scan_glob="*_T1w.nii.gz", mask_suffix="_brainmask",
                          output_dir="augmentation_evaluation", repeats=1, seed=2026,
                          max_cases=None, tiers=TIERS, augmentation_names=None,
                          artifact_severity=(0.15, 0.45), rotation_degrees=10.0,
                          resize_range=(0.9, 1.1), resolution_range=(1.2, 1.8),
                          augmentation_config_path=None, masker_options=None,
                          overwrite=False):
    """Evaluate one checkpoint, loading its model once and streaming one trial at a time."""
    if (manifest is None) == (data_dir is None):
        raise ValueError("Specify one held-out manifest or data_dir (never the full training dataset)")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if max_cases is not None and (isinstance(max_cases, bool) or not isinstance(max_cases, int) or max_cases < 1):
        raise ValueError("max_cases must be a positive integer or None")
    if not tiers or set(tiers) - set(TIERS):
        raise ValueError(f"tiers must be a nonempty subset of {TIERS}")
    cases = read_manifest(manifest) if manifest else discover_cases(data_dir, scan_glob, mask_suffix)
    if max_cases is not None and len(cases) > max_cases:
        # Deterministic subset, independent of manifest ordering; never select by performance.
        cases = sorted(cases, key=lambda c: trial_seed(seed, c["scan"], "case_selection", 0))[:max_cases]
    ids = [case["id"] for case in cases]
    catalog = build_catalog(artifact_severity=artifact_severity, rotation_degrees=rotation_degrees,
                            resize_range=resize_range, resolution_range=resolution_range,
                            augmentation_config_path=augmentation_config_path)
    selected_names = set().union(*(set(tier_members(catalog["specs"], tier)) for tier in tiers))
    if augmentation_names is not None:
        unknown = set(augmentation_names) - {s["name"] for s in catalog["specs"]}
        if unknown:
            raise ValueError(f"Unknown augmentation names: {sorted(unknown)}")
        selected_names &= set(augmentation_names)
    specs = [spec for spec in catalog["specs"] if spec["name"] in selected_names]
    if not specs:
        raise ValueError("No augmentation conditions selected")
    out = Path(output_dir).resolve()
    model_path = Path(model_path).resolve()
    protected = {Path(case[key]).resolve() for case in cases for key in ("scan", "mask")}
    protected.add(model_path)
    for path in (manifest, augmentation_config_path):
        if path:
            protected.add(Path(path).resolve())
    protected.add(Path(catalog["settings"]["augmentation_config_path"]).resolve())
    protected.update(Path(path).resolve()
                     for path in catalog["settings"].get("augmentation_config_sources", []))
    options = dict(masker_options or {})
    options["overwrite"] = False
    masker = BrainMasker(model_path=model_path, **options)
    protected.update(Path(path).resolve() for path in masker.input_paths())
    destinations = [out / name for name in ("results.json", "cases.csv", "summary.csv")]
    ensure_output_paths(destinations, protected, overwrite=overwrite)
    masker.load_model()
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1, "started_at": datetime.now(timezone.utc).isoformat(),
        "model": str(model_path), "status": "running", "benchmark_complete": False,
        "evaluation_space": "native input grid; paired transformed references for geometric conditions",
        "heldout_status": "Explicitly supplied by caller; training overlap not automatically verified",
        "synthetic_protocols": "Synthetic stress tests from source scans, not real acquired T2/FLAIR/MP2RAGE cohorts",
        "aggregation": "Mean across repeats, then conditions, then cases; only cases with every planned trial successful. Failed, pending, or unchanged augmented trials exclude that case from its condition/tier mean. Clean excluded from augmented tier means. std is population std across case means. available_mean_dice is a separately labeled partial-coverage statistic over successful draws/conditions; its case and condition mix may differ.",
        "seed": int(seed), "seed_identity": "resolved scan path (stable when an ID-less manifest is reordered)", "repeats": repeats, "case_count": len(cases), "max_cases": max_cases,
        "input": {"manifest": str(Path(manifest).resolve()) if manifest else None,
                  "data_dir": str(Path(data_dir).resolve()) if data_dir else None,
                  "scan_glob": scan_glob, "mask_suffix": mask_suffix},
        "masker_options": {key: str(value) if isinstance(value, Path) else value
                           for key, value in options.items()},
        "catalog": {**catalog, "specs": specs}, "requested_tiers": list(tiers),
        "cases": [], "tiers": [], "augmentations": [],
    }

    def publish():
        report["tiers"], report["augmentations"] = summarize(report["cases"], specs, ids, repeats)
        report["tiers"] = [row for row in report["tiers"] if row["tier"] in tiers]
        _save_reports(out, report, protected)

    expected = len(cases) * sum(1 if s["name"] == "clean" else repeats for s in specs)
    print(f"[augmentation evaluation] {len(cases)} cases, {len(specs)} conditions, {expected} predictions", flush=True)
    print(f"[augmentation evaluation] Reports updated after each case/condition: {out}", flush=True)
    publish()
    try:
        for case in cases:
            source_error = None
            try:
                if case.get("discovery_error"):
                    raise ValueError(case["discovery_error"])
                scan_img, source = _load_volume(case["scan"])
                ref_img, reference = _load_volume(case["mask"], binary=True)
                _check_grid(scan_img, source.shape, ref_img, reference.shape, affine_atol=1e-4)
                if not reference.any():
                    raise ValueError("Reference has no brain voxels; refusing trivial empty-mask Dice")
                affine_mm = nifti_affine_mm(scan_img)
            except Exception as exc:
                source_error = f"{type(exc).__name__}: {exc}"
            for spec in specs:
                for repeat in range(1 if spec["name"] == "clean" else repeats):
                    draw_seed = trial_seed(seed, case["scan"], spec["name"], repeat)
                    row = {"id": case["id"], "scan": case["scan"], "reference_mask": case["mask"],
                           "augmentation": spec["name"], "introduced_in": spec["tier"],
                           "repeat": repeat, "seed": draw_seed}
                    try:
                        if source_error:
                            raise ValueError(source_error)
                        if spec["name"] == "clean":
                            img, target, metadata = scan_img, reference, {"kind": "clean"}
                        else:
                            image, target, metadata = render_augmentation(
                                source.astype(np.float32), reference.copy(), affine_mm, spec,
                                np.random.default_rng(draw_seed))
                            if image.shape != source.shape or target.shape != reference.shape:
                                raise ValueError("Augmentation changed the native grid shape")
                            if not np.isfinite(image).all() or not np.logical_or(target == 0, target == 1).all():
                                raise ValueError("Augmentation returned nonfinite image or nonbinary reference")
                            if reference.any() and not target.any():
                                raise ValueError("Augmentation removed all foreground; refusing trivial empty-mask Dice")
                            # Preserve the float32 spatial precision of the former NIfTI-1 file path.
                            img = nib.Nifti1Image(np.asarray(image, dtype=np.float32),
                                                 np.asarray(affine_mm, dtype=np.float32))
                            img.header.set_xyzt_units("mm")
                        result = masker.predict(scan_path=case["scan"], image=img)
                        record = result.record
                        _check_grid(img, target.shape, result.image, result.mask.shape, affine_atol=1e-4)
                        row.update(status="ok", **binary_metrics(result.mask, target),
                                   review_flag=bool(record.get("review_flag")), augmentation_metadata=metadata)
                        if spec["name"] != "clean" and metadata.get("image_changed") is False:
                            row.update(status="unchanged", error="Renderer did not change image; scored diagnostically but excluded from augmented Dice aggregates")
                        row["inference"] = {k: v for k, v in record.items() if k != "mask"}
                    except Exception as exc:
                        row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                    report["cases"].append(row)
                    score = f"Dice={row['dice']:.5f}" if row["status"] == "ok" else row["error"]
                    print(f"[{len(report['cases'])}/{expected}] {case['id']} / {spec['name']} / {repeat + 1}: {score}", flush=True)
                publish()
        report["benchmark_complete"] = True
        report["status"] = "completed_with_failures" if any(r["status"] == "failed" for r in report["cases"]) else (
            "completed_with_unchanged_trials" if any(r["status"] == "unchanged" for r in report["cases"]) else "completed")
    except BaseException:
        report["status"] = "interrupted"
        raise
    finally:
        report["updated_at"] = datetime.now(timezone.utc).isoformat()
        publish()
    return report


def _cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest")
    source.add_argument("--data-dir")
    p.add_argument("--scan-glob", default="*_T1w.nii.gz")
    p.add_argument("--mask-suffix", default="_brainmask")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--max-cases", type=int)
    p.add_argument("--tiers", nargs="+", choices=TIERS, default=list(TIERS))
    p.add_argument("--augmentations", nargs="+", help="Optional exact condition subset for a quick check")
    p.add_argument("--artifact-severity", nargs=2, type=float, default=[0.15, 0.45])
    p.add_argument("--rotation-degrees", type=float, default=10.0)
    p.add_argument("--resize-range", nargs=2, type=float, default=[0.9, 1.1])
    p.add_argument("--resolution-range", nargs=2, type=float, default=[1.2, 1.8])
    p.add_argument("--augmentation-config", help="Augmentation inventory/settings; independent of training switches")
    add_inference_arguments(p)
    args = p.parse_args()
    options = inference_kwargs(args)
    options.pop("overwrite")
    report = evaluate_augmentations(
        args.model, manifest=args.manifest, data_dir=args.data_dir, scan_glob=args.scan_glob,
        mask_suffix=args.mask_suffix, output_dir=args.output_dir, repeats=args.repeats, seed=args.seed,
        max_cases=args.max_cases, tiers=args.tiers, augmentation_names=args.augmentations,
        artifact_severity=args.artifact_severity, rotation_degrees=args.rotation_degrees,
        resize_range=args.resize_range, resolution_range=args.resolution_range,
        augmentation_config_path=args.augmentation_config, masker_options=options, overwrite=args.overwrite)
    print(json.dumps(report["tiers"], indent=2))
    return 0 if report["status"] in ("completed", "completed_with_unchanged_trials") else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
