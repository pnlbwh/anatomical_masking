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
import gc
import hashlib
import io
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import nibabel as nib
import numpy as np
import torch

from evaluation.standard import (REFERENCE_MASK_POLICY, _check_grid, _load_volume, binary_metrics, discover_cases,
                      read_manifest)
from inference.masker import (BrainMasker, add_inference_arguments, atomic_write_text,
                           ensure_output_paths, inference_kwargs)
from imaging.geometry import nifti_affine_mm
from imaging.normalization import normalize_intensity
from evaluation.transforms import build_catalog, render_augmentation, uses_normalized_source


TIERS = ("clean", "mild", "protocols", "full")
TIER_LABELS = {
    "clean": "Clean scans",
    "mild": "Mild geometry and resolution",
    "protocols": "Mild + selected synthetic MRI protocols",
    "full": "All selected augmented conditions",
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
                   "seed", "status", "dice", "iou", "precision", "recall", "review_flag", "error",
                   "attempt_count", "source_attempt_count", "initial_sw_batch_size", "final_sw_batch_size",
                   "render_seconds", "predict_seconds", "total_seconds"]
    contents = {
        "results.json": json.dumps(report, indent=2, allow_nan=False) + "\n",
        "cases.csv": _csv_text(report["cases"], case_fields),
        "summary.csv": _csv_text(summaries, summary_fields),
    }
    for filename, content in contents.items():
        atomic_write_text(out / filename, content, protected_paths=protected, overwrite=True)


def _model_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clear_cuda_oom_cache(device):
    """Release unused allocations after a CUDA OOM, without changing inference."""
    gc.collect()
    try:
        if torch.cuda.is_initialized():
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
    except Exception as exc:
        # A cleanup failure must not replace the recorded inference error.
        print(f"[augmentation evaluation] CUDA cache cleanup failed: {type(exc).__name__}: {exc}",
              flush=True)


def _load_case_arrays(case):
    if case.get("discovery_error"):
        raise ValueError(case["discovery_error"])
    scan_img, source = _load_volume(case["scan"])
    ref_img, reference = _load_volume(case["mask"], reference=True)
    _check_grid(scan_img, source.shape, ref_img, reference.shape, affine_atol=1e-4)
    if not reference.any():
        raise ValueError("Reference has no brain voxels; refusing trivial empty-mask Dice")
    return scan_img, source, reference, nifti_affine_mm(scan_img), source.astype(np.float32)


def evaluate_augmentations(model_path, *, manifest=None, data_dir=None,
                          scan_glob="*_T1w.nii.gz", mask_suffix="_brainmask",
                          output_dir="augmentation_evaluation", repeats=1, seed=2026,
                          max_cases=None, tiers=TIERS, augmentation_names=None,
                          artifact_severity=(0.15, 0.45), rotation_degrees=10.0,
                          resize_range=(0.9, 1.1), resolution_range=(1.2, 1.8),
                          augmentation_config_path=None, masker_options=None,
                          overwrite=False, report_every_seconds=0.0, max_retries=2,
                          trial_keys=None, adaptive_oom_retries=True):
    """Evaluate one checkpoint, retrying exceptions without changing trial draws.

    CUDA prediction OOM retries can reduce patch batch size, retaining the lower
    batch for later trials. This never changes augmentation draws or precision.

    ``trial_keys`` is an internal recovery filter of (resolved scan, condition,
    zero-based repeat). Reports retain the complete plan and mark skipped trials
    pending; recovery callers can merge these rows into the original report.
    """
    started = perf_counter()
    if (isinstance(report_every_seconds, bool)
            or not isinstance(report_every_seconds, (int, float))
            or not math.isfinite(report_every_seconds) or report_every_seconds < 0):
        raise ValueError("report_every_seconds must be a finite nonnegative number")
    if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
        raise ValueError("max_retries must be a nonnegative integer")
    if not isinstance(adaptive_oom_retries, bool):
        raise ValueError("adaptive_oom_retries must be a boolean")
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
    planned_keys = {(str(Path(case["scan"]).resolve()), spec["name"], repeat)
                    for case in cases for spec in specs
                    for repeat in range(1 if spec["name"] == "clean" else repeats)}
    selected_keys = None
    if trial_keys is not None:
        selected_keys = set()
        try:
            for key in trial_keys:
                if (not isinstance(key, (tuple, list)) or len(key) != 3
                        or not isinstance(key[0], str) or not isinstance(key[1], str)
                        or isinstance(key[2], bool) or not isinstance(key[2], int)):
                    raise ValueError("Each trial_keys entry must be (scan string, augmentation string, repeat integer)")
                selected_keys.add((str(Path(key[0]).resolve()), key[1], key[2]))
        except TypeError as exc:
            raise ValueError("trial_keys must be a collection of trial keys") from exc
        unknown_keys = selected_keys - planned_keys
        if unknown_keys:
            raise ValueError(f"trial_keys contains trials outside the benchmark plan: {sorted(unknown_keys)}")
    requested_scans = None if selected_keys is None else {key[0] for key in selected_keys}
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
    model_sha256 = _model_sha256(model_path)
    masker = BrainMasker(model_path=model_path, **options)
    protected.update(Path(path).resolve() for path in masker.input_paths())
    destinations = [out / name for name in ("results.json", "cases.csv", "summary.csv")]
    ensure_output_paths(destinations, protected, overwrite=overwrite)
    model_load_started = perf_counter()
    masker.load_model()
    model_load_seconds = perf_counter() - model_load_started
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1, "started_at": datetime.now(timezone.utc).isoformat(),
        "model": str(model_path), "model_sha256": model_sha256,
        "reference_mask_policy": dict(REFERENCE_MASK_POLICY),
        "status": "running", "benchmark_complete": False,
        "planned_cases": [dict(case) for case in cases],
        "planned_trial_count": len(planned_keys),
        "requested_trial_count": len(planned_keys) if selected_keys is None else len(selected_keys),
        "trial_filter": None if selected_keys is None else sorted(selected_keys),
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
        "execution": {"device": str(masker.device), "mode": masker.mode,
                      "sw_batch_size": masker.sw_batch_size,
                      "current_sw_batch_size": masker.sw_batch_size,
                      "adaptive_oom_retries": adaptive_oom_retries,
                      "retry_adjustments": [],
                      "tta_requested": masker.tta,
                      "tta_effective": masker.tta and not masker.mode.startswith("conform"),
                      "report_every_seconds": float(report_every_seconds),
                      "max_retries": max_retries,
                      "retry_policy": "Exceptions only; same case, condition, repeat, and seed. CUDA prediction OOM may reduce patch batch size when adaptive_oom_retries is enabled; all other inference settings are preserved. Successful and unchanged trials are never retried."},
        "catalog": {**catalog, "specs": specs}, "requested_tiers": list(tiers),
        "cases": [], "tiers": [], "augmentations": [],
    }

    last_published = None
    reporting_seconds = 0.0
    case_load_seconds = 0.0
    publications = 0

    def publish(*, force=False):
        nonlocal last_published, reporting_seconds, publications
        now = perf_counter()
        if (not force and last_published is not None
                and now - last_published < report_every_seconds):
            return
        report["tiers"], report["augmentations"] = summarize(report["cases"], specs, ids, repeats)
        report["tiers"] = [row for row in report["tiers"] if row["tier"] in tiers]
        report["updated_at"] = datetime.now(timezone.utc).isoformat()
        report["execution"]["current_sw_batch_size"] = masker.sw_batch_size
        publications += 1
        report["timing"] = {
            "elapsed_seconds": perf_counter() - started,
            "model_load_seconds": model_load_seconds, "case_load_seconds": case_load_seconds,
            "render_seconds": sum(row["render_seconds"] for row in report["cases"]),
            "predict_seconds": sum(row["predict_seconds"] for row in report["cases"]),
            "trial_seconds": sum(row["total_seconds"] for row in report["cases"]),
            # The current write cannot time itself in the snapshot being written.
            "report_seconds_before_current_save": reporting_seconds,
            "report_publications": publications,
        }
        _save_reports(out, report, protected)
        last_published = perf_counter()
        reporting_seconds += last_published - now

    expected = report["requested_trial_count"]
    print(f"[augmentation evaluation] {len(cases)} cases, {len(specs)} conditions, {expected} predictions", flush=True)
    if selected_keys is not None:
        print(f"[augmentation evaluation] Recovery filter: running {expected} of "
              f"{len(planned_keys)} planned trials across {len(requested_scans)} cases", flush=True)
    execution = report["execution"]
    print(f"[augmentation evaluation] Device={execution['device']}, mode={execution['mode']}, "
          f"sw_batch_size={execution['sw_batch_size']}, effective TTA={execution['tta_effective']}", flush=True)
    cadence = ("after each case/condition" if report_every_seconds == 0
               else f"at most once per {report_every_seconds:g}s after a condition")
    print(f"[augmentation evaluation] Reports updated {cadence}, and on completion/interruption: {out}", flush=True)
    publish(force=True)
    try:
        for case in cases:
            scan_key = str(Path(case["scan"]).resolve())
            if requested_scans is not None and scan_key not in requested_scans:
                continue
            source_error = None
            source_errors = []
            source_attempt_count = 0
            normalized_source = None
            scan_img = source = reference = affine_mm = source32 = None
            case_load_started = perf_counter()
            try:
                for source_attempt_count in range(1, max_retries + 2):
                    try:
                        scan_img, source, reference, affine_mm, source32 = _load_case_arrays(case)
                    except Exception as exc:
                        source_errors.append(f"{type(exc).__name__}: {exc}")
                    else:
                        break
                    if case.get("discovery_error") or source_attempt_count > max_retries:
                        source_error = source_errors[-1]
                        break
                    print(f"[augmentation evaluation] Retrying source {case['id']} "
                          f"after attempt {source_attempt_count}: {source_errors[-1]}", flush=True)
            finally:
                case_load_seconds += perf_counter() - case_load_started

            def run_trial(spec, draw_seed, timing, phase):
                # Attempt-local arrays, metadata, and predictions leave scope before a retry.
                nonlocal normalized_source
                phase["name"] = "render"
                render_started = perf_counter()
                try:
                    if spec["name"] == "clean":
                        img, target, metadata = scan_img, reference, {"kind": "clean"}
                    else:
                        if normalized_source is None and uses_normalized_source(spec):
                            normalized_source = normalize_intensity(source32)
                        image, target, metadata = render_augmentation(
                            source32, reference.copy(), affine_mm, spec,
                            np.random.default_rng(draw_seed), normalized_scan=normalized_source)
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
                finally:
                    timing["render_seconds"] += perf_counter() - render_started
                phase["name"] = "predict"
                predict_started = perf_counter()
                try:
                    result = masker.predict(scan_path=case["scan"], image=img)
                finally:
                    timing["predict_seconds"] += perf_counter() - predict_started
                phase["name"] = "score"
                record = result.record
                _check_grid(img, target.shape, result.image, result.mask.shape, affine_atol=1e-4)
                scored = {"status": "ok", **binary_metrics(result.mask, target),
                          "review_flag": bool(record.get("review_flag")), "augmentation_metadata": metadata}
                if spec["name"] != "clean" and metadata.get("image_changed") is False:
                    scored.update(status="unchanged", error="Renderer did not change image; scored diagnostically but excluded from augmented Dice aggregates")
                scored["inference"] = {k: v for k, v in record.items() if k != "mask"}
                return scored

            for spec in specs:
                condition_ran = False
                for repeat in range(1 if spec["name"] == "clean" else repeats):
                    if selected_keys is not None and (scan_key, spec["name"], repeat) not in selected_keys:
                        continue
                    condition_ran = True
                    draw_seed = trial_seed(seed, case["scan"], spec["name"], repeat)
                    row = {"id": case["id"], "scan": case["scan"], "reference_mask": case["mask"],
                           "augmentation": spec["name"], "introduced_in": spec["tier"],
                           "repeat": repeat, "seed": draw_seed,
                           "attempt_count": 0, "attempt_errors": [], "attempt_settings": [],
                           "initial_sw_batch_size": masker.sw_batch_size, "retry_adjustments": [],
                           "source_attempt_count": source_attempt_count,
                           "source_attempt_errors": list(source_errors),
                           "render_seconds": 0.0, "predict_seconds": 0.0}
                    trial_started = perf_counter()
                    if source_error is not None:
                        row.update(status="failed", error=source_error)
                    else:
                        for attempt in range(1, max_retries + 2):
                            row["attempt_count"] = attempt
                            cuda_oom = False
                            phase = {"name": "render"}
                            attempt_settings = {"attempt": attempt, "device": str(masker.device),
                                                "sw_batch_size": masker.sw_batch_size}
                            row["attempt_settings"].append(attempt_settings)
                            try:
                                row.update(run_trial(spec, draw_seed, row, phase))
                            except Exception as exc:
                                row["attempt_errors"].append(f"{type(exc).__name__}: {exc}")
                                cuda_oom = (isinstance(exc, torch.cuda.OutOfMemoryError)
                                            and masker.device.type == "cuda")
                                attempt_settings.update(status="failed", phase=phase["name"],
                                                        error=row["attempt_errors"][-1], cuda_oom=cuda_oom)
                            else:
                                attempt_settings["status"] = row["status"]
                                break  # Low Dice and unchanged renders are completed trials.
                            # The exception/traceback no longer holds attempt-local arrays here.
                            if cuda_oom:
                                unchanged_reason = None
                                if attempt > max_retries:
                                    unchanged_reason = "no retry attempts remain"
                                elif not adaptive_oom_retries:
                                    unchanged_reason = "adaptive OOM retries are disabled"
                                elif phase["name"] != "predict":
                                    unchanged_reason = f"OOM occurred during {phase['name']}, not prediction"
                                elif masker.mode not in ("patch", "conform_patch"):
                                    unchanged_reason = f"{masker.mode} mode does not use patch batches"
                                elif masker.sw_batch_size <= 1:
                                    unchanged_reason = "patch batch size is already 1"
                                if unchanged_reason is None:
                                    previous_batch = masker.sw_batch_size
                                    masker.sw_batch_size = max(1, previous_batch // 2)
                                    adjustment = {"attempt": attempt + 1, "setting": "sw_batch_size",
                                                  "from": previous_batch, "to": masker.sw_batch_size,
                                                  "reason": "cuda_out_of_memory"}
                                    row["retry_adjustments"].append(adjustment)
                                    report["execution"]["retry_adjustments"].append({
                                        **{key: row[key] for key in ("id", "scan", "augmentation", "repeat")},
                                        **adjustment, "phase": "predict", "device": str(masker.device),
                                        "error": row["attempt_errors"][-1]})
                                    print(f"[augmentation evaluation] CUDA OOM: reducing patch batch size "
                                          f"{previous_batch} -> {masker.sw_batch_size} for retry "
                                          f"{attempt + 1} and remaining trials", flush=True)
                                else:
                                    print(f"[augmentation evaluation] CUDA OOM: patch batch size unchanged "
                                          f"({unchanged_reason})", flush=True)
                            if attempt > max_retries:
                                row.update(status="failed", error=row["attempt_errors"][-1])
                                break
                            if cuda_oom:
                                _clear_cuda_oom_cache(masker.device)
                            print(f"[augmentation evaluation] Retrying {case['id']} / {spec['name']} / "
                                  f"{repeat + 1} with seed {draw_seed} after attempt {attempt}: "
                                  f"{row['attempt_errors'][-1]}", flush=True)
                    row["final_sw_batch_size"] = masker.sw_batch_size
                    row["batch_backoff_history"] = [dict(change) for change in report["execution"]["retry_adjustments"]]
                    row["total_seconds"] = perf_counter() - trial_started
                    report["cases"].append(row)
                    score = f"Dice={row['dice']:.5f}" if row["status"] == "ok" else row["error"]
                    print(f"[{len(report['cases'])}/{expected}] {case['id']} / {spec['name']} / {repeat + 1}: "
                          f"{score} (attempts {row['attempt_count']}, render {row['render_seconds']:.2f}s, "
                          f"predict {row['predict_seconds']:.2f}s, total {row['total_seconds']:.2f}s)", flush=True)
                if condition_ran:
                    publish()
        report["benchmark_complete"] = len(report["cases"]) == len(planned_keys)
        report["status"] = "completed_with_failures" if any(r["status"] == "failed" for r in report["cases"]) else (
            "completed_with_unchanged_trials" if any(r["status"] == "unchanged" for r in report["cases"]) else "completed")
    except BaseException:
        report["status"] = "interrupted"
        raise
    finally:
        report["execution"]["final_sw_batch_size"] = masker.sw_batch_size
        publish(force=True)
        print(f"[augmentation evaluation] Elapsed {perf_counter() - started:.1f}s; "
              f"render {report['timing']['render_seconds']:.1f}s; "
              f"predict {report['timing']['predict_seconds']:.1f}s; "
              f"report updates {reporting_seconds:.1f}s ({publications} saves)", flush=True)
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
    p.add_argument("--max-retries", type=int, default=2,
                   help="Retries after an exception, preserving each trial draw (0 disables)")
    p.add_argument("--no-adaptive-oom-retries", dest="adaptive_oom_retries", action="store_false", default=True,
                   help="Disable lowering CUDA patch batch size after prediction out-of-memory errors")
    p.add_argument("--report-every-seconds", type=float, default=0.0,
                   help="Minimum interval between report rewrites (0: every condition); final/interrupt saves always run")
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
        augmentation_config_path=args.augmentation_config, masker_options=options, overwrite=args.overwrite,
        report_every_seconds=args.report_every_seconds, max_retries=args.max_retries,
        adaptive_oom_retries=args.adaptive_oom_retries)
    print(json.dumps(report["tiers"], indent=2))
    return 0 if report["status"] in ("completed", "completed_with_unchanged_trials") else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
