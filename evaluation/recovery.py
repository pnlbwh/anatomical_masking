"""Retry failed augmentation trials into a new report, retaining valid results.

Recovery preserves the original trial plan and random seeds. It does not retry
unchanged renderings or low-scoring successful predictions. Input paths and
settings are checked; scan/mask byte identity cannot be proved from reports
that did not record input content hashes.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from evaluation.augmentation import (
    TIERS, _save_reports, evaluate_augmentations, summarize, tier_members, trial_seed,
)
from evaluation.standard import REFERENCE_MASK_POLICY, discover_cases, read_manifest
from evaluation.transforms import build_catalog
from inference.masker import ensure_output_paths


def _load_report(path):
    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON value in evaluation report: {value}")
    with Path(path).open(encoding="utf-8-sig") as stream:
        report = json.load(stream, parse_constant=reject_constant)
    if (not isinstance(report, dict) or isinstance(report.get("schema_version"), bool)
            or report.get("schema_version") != 1):
        raise ValueError("Recovery requires an augmentation evaluation report with schema_version 1")
    # Also catches overflow such as 1e999, which does not use parse_constant.
    json.dumps(report, allow_nan=False)
    return report


def _integer(value, name, minimum=0):
    if (isinstance(value, bool) or not isinstance(value, int)
            or (minimum is not None and value < minimum)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _key(row):
    if not isinstance(row, dict) or not isinstance(row.get("scan"), str):
        raise ValueError("Each saved trial needs a scan path")
    name = row.get("augmentation")
    if not isinstance(name, str) or not name:
        raise ValueError("Each saved trial needs an augmentation name")
    return str(Path(row["scan"]).resolve()), name, _integer(row.get("repeat"), "trial repeat")


def _case_identity(case):
    if (not isinstance(case, dict) or not isinstance(case.get("id"), str)
            or not isinstance(case.get("scan"), str) or not isinstance(case.get("mask"), str)):
        raise ValueError("Planned cases must contain string id, scan, and mask fields")
    return case["id"], str(Path(case["scan"]).resolve()), str(Path(case["mask"]).resolve())


def _validate_rows(rows, plan, seed, *, allowed=None):
    if not isinstance(rows, list):
        raise ValueError("Saved report cases must be a list of trial rows")
    indexed = {}
    for row in rows:
        key = _key(row)
        if key in indexed:
            raise ValueError(f"Duplicate saved trial key: {key}")
        if key not in plan or (allowed is not None and key not in allowed):
            raise ValueError(f"Saved trial is outside the original trial plan: {key}")
        case, spec = plan[key]
        if (row.get("id") != case["id"] or not isinstance(row.get("reference_mask"), str)
                or str(Path(row["reference_mask"]).resolve()) != str(Path(case["mask"]).resolve())):
            raise ValueError(f"Saved trial case identity/reference differs from current input: {key}")
        if row.get("introduced_in") != spec["tier"]:
            raise ValueError(f"Saved trial tier differs from its catalog entry: {key}")
        if (_integer(row.get("seed"), "trial seed")
                != trial_seed(seed, case["scan"], key[1], key[2])):
            raise ValueError(f"Saved trial seed differs from the original settings: {key}")
        status = row.get("status")
        if status not in ("ok", "unchanged", "failed"):
            raise ValueError(f"Unsupported saved trial status: {status!r}")
        for metric in ("dice", "iou", "precision", "recall"):
            value = row.get(metric)
            if status in ("ok", "unchanged") and metric in ("dice", "iou") and value is None:
                raise ValueError(f"Successful saved trial is missing {metric}: {key}")
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"Saved trial {metric} must be finite and in [0, 1]: {key}")
        if status in ("failed", "unchanged") and not isinstance(row.get("error"), str):
            raise ValueError(f"Saved {status} trial is missing its error: {key}")
        indexed[key] = row
    return indexed


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_INFERENCE_FIELDS = (
    "mode", "target_shape", "patch_size", "conform_mm", "device", "sw_overlap",
    "threshold", "dilate_iters", "dilate_mm", "cc_keep_ratio", "component_policy",
    "canonical_orientation", "canonical_is_guess", "unverified_preprocessing", "tta",
)


def _recorded_inference_settings(report):
    """Collect settings that are constant across trials, excluding data-dependent outputs."""
    settings = {}
    records = [row["inference"] for row in report.get("cases", [])
               if isinstance(row.get("inference"), dict)]
    execution = report.get("execution")
    if isinstance(execution, dict):
        records.append({("tta" if key == "tta_effective" else key): value
                        for key, value in execution.items()
                        if key in ("mode", "device", "tta_effective")})
    for record in records:
        for key in _INFERENCE_FIELDS:
            if key not in record:
                continue
            if key in settings and settings[key] != record[key]:
                raise ValueError(f"Report contains inconsistent effective inference setting: {key}")
            settings[key] = record[key]
    return settings


_TRANSITION_FIELDS = ("attempt", "setting", "from", "to", "reason")


def _configured_batch(report, fallback=4):
    options, execution = report.get("masker_options", {}), report.get("execution", {})
    configured = _integer(options.get("sw_batch_size", execution.get("sw_batch_size", fallback)),
                          "configured sw_batch_size", 1)
    if "sw_batch_size" in execution and execution["sw_batch_size"] != configured:
        raise ValueError("Configured patch batch differs between masker_options and execution")
    return configured


def _validate_backoff_history(history, configured, settings, *, planned_keys=None):
    if not isinstance(history, list):
        raise ValueError("Patch batch backoff history must be a list")
    current = configured
    for entry in history:
        if not isinstance(entry, dict):
            raise ValueError("Invalid patch batch backoff entry")
        key = _key(entry)
        if planned_keys is not None and key not in planned_keys:
            raise ValueError("Patch batch backoff refers to an unplanned trial")
        _integer(entry.get("attempt"), "backoff attempt", 2)
        before = _integer(entry.get("from"), "backoff from", 1)
        after = _integer(entry.get("to"), "backoff to", 1)
        if (entry.get("setting") != "sw_batch_size" or entry.get("reason") != "cuda_out_of_memory"
                or entry.get("phase") != "predict" or settings.get("mode") not in ("patch", "conform_patch")
                or not str(settings.get("device", "")).startswith("cuda")
                or entry.get("device") != settings.get("device")
                or not isinstance(entry.get("error"), str)
                or "outofmemoryerror" not in entry["error"].lower()
                or before != current or before <= 1 or after != max(1, before // 2)):
            raise ValueError("Unsupported or undocumented CUDA patch batch backoff")
        current = after
    return current


def _validate_patch_batches(report, *, configured=None, planned_keys=None):
    """Only documented CUDA prediction OOM halvings may differ from the configured batch.

    Each row carries its own cumulative history, so retained and recovered rows
    from separate runs can coexist without mistaking a fresh run's configured
    starting batch for an increase within an adaptive retry sequence.
    """
    configured = _configured_batch(report, 4 if configured is None else configured)
    settings = _recorded_inference_settings(report)
    execution = report.get("execution", {})
    global_history = execution.get("retry_adjustments", [])
    current = _validate_backoff_history(global_history, configured, settings, planned_keys=planned_keys)
    if global_history and execution.get("adaptive_oom_retries") is not True:
        raise ValueError("Batch backoff recorded while adaptive OOM retries were disabled")
    for field in ("current_sw_batch_size", "final_sw_batch_size"):
        if field in execution and _integer(execution[field], field, 1) != current:
            raise ValueError("Execution patch batch does not match its recorded backoff history")
    for row in report.get("cases", []):
        history = row.get("batch_backoff_history", [])
        final = _validate_backoff_history(history, configured, settings, planned_keys=planned_keys)
        own = [entry for entry in history if _key(entry) == _key(row)]
        if own and history[-len(own):] != own:
            raise ValueError("Trial's patch batch changes must finish its cumulative backoff history")
        initial = own[0]["from"] if own else final
        for field, expected in (("initial_sw_batch_size", initial), ("final_sw_batch_size", final)):
            if field in row and _integer(row[field], field, 1) != expected:
                raise ValueError("Trial patch batch does not match its recorded backoff history")
        inference = row.get("inference", {})
        if isinstance(inference, dict) and "sw_batch_size" in inference:
            if _integer(inference["sw_batch_size"], "inference sw_batch_size", 1) != final:
                raise ValueError("Inference patch batch changed without matching backoff history")
        adjustments = row.get("retry_adjustments", [])
        expected_adjustments = [{key: entry[key] for key in _TRANSITION_FIELDS} for entry in own]
        if adjustments != expected_adjustments:
            raise ValueError("Trial patch batch adjustments disagree with its backoff history")
        attempts = row.get("attempt_settings")
        if attempts is None:
            if history:
                raise ValueError("Adaptive patch batch row is missing attempt settings")
            continue  # Legacy baseline-batch rows have no attempt metadata.
        if not isinstance(attempts, list) or len(attempts) != row.get("attempt_count"):
            raise ValueError("Trial attempt settings/count disagree")
        by_attempt = {entry["attempt"]: entry for entry in own}
        if len(by_attempt) != len(own) or any(number > len(attempts) for number in by_attempt):
            raise ValueError("Patch batch adjustment has no corresponding retry attempt")
        batch = initial
        for number, attempt in enumerate(attempts, 1):
            if number in by_attempt:
                change, previous = by_attempt[number], attempts[number - 2]
                if (change["from"] != batch or previous.get("status") != "failed"
                        or previous.get("phase") != "predict" or previous.get("cuda_oom") is not True
                        or previous.get("error") != change["error"]):
                    raise ValueError("Patch batch change lacks a preceding CUDA prediction OOM")
                batch = change["to"]
            if (not isinstance(attempt, dict) or attempt.get("attempt") != number
                    or attempt.get("sw_batch_size") != batch
                    or attempt.get("device") != settings.get("device")):
                raise ValueError("Retry attempt changed its patch batch/device without matching provenance")
        if batch != final or (not attempts and row.get("status") != "failed"):
            raise ValueError("Final patch batch differs from the attempted batch")
    return configured


def _validate_reference_policy(report):
    # Reports predating this field accepted exact 0/1 references only. Their
    # scored labels are identical under the current training-compatible rule.
    if "reference_mask_policy" in report and report["reference_mask_policy"] != REFERENCE_MASK_POLICY:
        raise ValueError("Unsupported reference mask policy; refusing mixed reference labels")


def _validate_effective_inference(original, child):
    _validate_reference_policy(original)
    _validate_reference_policy(child)
    if "reference_mask_policy" in original and "reference_mask_policy" not in child:
        raise ValueError("Retry is missing the original reference mask policy")
    expected = _recorded_inference_settings(original)
    actual = _recorded_inference_settings(child)
    configured = _validate_patch_batches(original)
    if _configured_batch(child, configured) != configured:
        raise ValueError("Retry changed the originally configured patch batch")
    if "masker_options" in child and child["masker_options"] != original["masker_options"]:
        raise ValueError("Retry changed the original masker_options")
    _validate_patch_batches(child, configured=configured)
    batch_recorded = ("sw_batch_size" in original.get("execution", {}) or any(
        "sw_batch_size" in row.get("inference", {}) for row in original.get("cases", [])))
    for key in expected.keys() & actual.keys():
        if expected[key] != actual[key]:
            raise ValueError(f"Retry effective inference setting differs from the original report: {key}")
    for row in child.get("cases", []):
        if row.get("status") in ("ok", "unchanged"):
            record = row.get("inference", {})
            if (not isinstance(record, dict)
                    or (batch_recorded and "sw_batch_size" not in record)
                    or any(record.get(key) != value or key not in record for key, value in expected.items())):
                raise ValueError("Retry is missing original effective inference settings or uses different preprocessing")


def _validate_original(report):
    _validate_reference_policy(report)
    seed = _integer(report.get("seed"), "seed", minimum=None)
    repeats = _integer(report.get("repeats"), "repeats", 1)
    max_cases = report.get("max_cases")
    if max_cases is not None:
        _integer(max_cases, "max_cases", 1)
    tiers = report.get("requested_tiers")
    if not isinstance(tiers, list) or not tiers or any(t not in TIERS for t in tiers):
        raise ValueError("Report requested_tiers must be a nonempty list of benchmark tiers")
    source = report.get("input")
    if not isinstance(source, dict) or bool(source.get("manifest")) == bool(source.get("data_dir")):
        raise ValueError("Report must record exactly one original manifest or data_dir")
    scan_glob, mask_suffix = source.get("scan_glob"), source.get("mask_suffix")
    if not isinstance(scan_glob, str) or not isinstance(mask_suffix, str):
        raise ValueError("Report must record its scan_glob and mask_suffix")
    cases = (read_manifest(source["manifest"]) if source.get("manifest") else
             discover_cases(source["data_dir"], scan_glob, mask_suffix))
    if max_cases is not None and len(cases) > max_cases:
        cases = sorted(cases, key=lambda c: trial_seed(seed, c["scan"], "case_selection", 0))[:max_cases]
    identities = {_case_identity(case) for case in cases}
    if len(identities) != len(cases) or _integer(report.get("case_count"), "case_count", 1) != len(cases):
        raise ValueError("Current selected case count differs from the saved report")
    if "planned_cases" in report:
        saved_cases = report["planned_cases"]
        if (not isinstance(saved_cases, list) or len(saved_cases) != len(cases)
                or {_case_identity(case) for case in saved_cases} != identities):
            raise ValueError("Current case identities differ from saved planned_cases")
    catalog = report.get("catalog")
    if not isinstance(catalog, dict) or not isinstance(catalog.get("settings"), dict):
        raise ValueError("Report is missing its augmentation catalog/settings")
    settings = catalog["settings"]
    required = ("artifact_severity", "rotation_degrees", "resize_range", "resolution_range",
                "augmentation_config_path")
    if any(key not in settings for key in required):
        raise ValueError("Report is missing original augmentation settings")
    saved_specs = catalog.get("specs")
    if not isinstance(saved_specs, list) or not saved_specs:
        raise ValueError("Report catalog must contain selected conditions")
    names = [spec.get("name") for spec in saved_specs if isinstance(spec, dict)]
    if (len(names) != len(saved_specs) or any(not isinstance(name, str) for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("Report catalog contains invalid or duplicate selected conditions")
    rebuilt = build_catalog(**{key: settings[key] for key in required})
    allowed_names = set().union(*(set(tier_members(rebuilt["specs"], tier)) for tier in tiers))
    selected = [spec for spec in rebuilt["specs"] if spec["name"] in names and spec["name"] in allowed_names]
    if selected != saved_specs or rebuilt["settings"] != settings:
        raise ValueError("Current augmentation catalog/settings differ from the saved report; refusing mixed results")
    plan = {(str(Path(case["scan"]).resolve()), spec["name"], repeat): (case, spec)
            for case in cases for spec in selected
            for repeat in range(1 if spec["name"] == "clean" else repeats)}
    rows = _validate_rows(report.get("cases"), plan, seed)
    if "planned_cases" not in report:
        represented = {_case_identity({"id": row["id"], "scan": row["scan"],
                                       "mask": row["reference_mask"]}) for row in rows.values()}
        if represented != identities:
            raise ValueError("Legacy report does not identify every original case; case identity cannot be checked")
    if report.get("benchmark_complete") is True and len(rows) != len(plan):
        raise ValueError("Report claims completion but has missing planned trials")
    if "planned_trial_count" in report and report["planned_trial_count"] != len(plan):
        raise ValueError("Saved planned_trial_count differs from reconstructed trial plan")
    if not isinstance(report.get("model"), str) or not Path(report["model"]).is_file():
        raise ValueError("Original model file is missing; restore the same checkpoint before retrying")
    options = report.get("masker_options")
    if not isinstance(options, dict):
        raise ValueError("Report must record its original masker_options")
    _recorded_inference_settings(report)
    _validate_patch_batches(report, planned_keys=set(plan))
    model_verified = False
    if "model_sha256" in report:
        recorded = report["model_sha256"]
        if (not isinstance(recorded, str) or len(recorded) != 64
                or recorded.lower() != _sha256(report["model"])):
            raise ValueError("Original model content does not match model_sha256; refusing mixed results")
        model_verified = True
    return cases, selected, plan, rows, model_verified


def retry_failed_evaluation(report_path, output_dir, *, max_retries=2, report_every_seconds=60,
                            adaptive_oom_retries=True):
    """Retry only explicit failures, merge into a fresh output directory, and return its report.

    ``max_retries`` permits that many additional attempts after each retry's first
    attempt. Successful/unchanged prior rows retain their exact saved contents.
    Legacy reports can check paths/settings but cannot prove file byte identity.
    """
    _integer(max_retries, "max_retries")
    if not isinstance(adaptive_oom_retries, bool):
        raise ValueError("adaptive_oom_retries must be a boolean")
    if (isinstance(report_every_seconds, bool) or not isinstance(report_every_seconds, (int, float))
            or not math.isfinite(report_every_seconds) or report_every_seconds < 0):
        raise ValueError("report_every_seconds must be a finite nonnegative number")
    original_path, out = Path(report_path).resolve(), Path(output_dir).resolve()
    if out.exists():
        raise FileExistsError(f"Recovery requires a new output directory: {out}")
    original = _load_report(original_path)
    cases, specs, plan, old_rows, model_verified = _validate_original(original)
    failed = {key for key, row in old_rows.items() if row["status"] == "failed"}
    source, settings = original["input"], original["catalog"]["settings"]
    protected = {original_path, Path(original["model"]).resolve()}
    protected.update(original_path.parent / name for name in ("results.json", "cases.csv", "summary.csv"))
    protected.update(Path(case[key]).resolve() for case in cases for key in ("scan", "mask"))
    protected.update(Path(path).resolve() for path in settings.get("augmentation_config_sources", []))
    protected.add(Path(settings["augmentation_config_path"]).resolve())
    for path in (source.get("manifest"), original["masker_options"].get("config_path")):
        if isinstance(path, str):
            protected.add(Path(path).resolve())
    ensure_output_paths([out / name for name in ("results.json", "cases.csv", "summary.csv")],
                        protected, overwrite=False)
    content_note = ("Checkpoint SHA256 verified. " if model_verified else
                    "Legacy report has no checkpoint hash; checkpoint content identity is NOT verified. ")
    content_note += ("Scan/mask content identity is NOT verified (no saved input hashes); "
                     "the same original input and preprocessing files are required.")
    print("[evaluation recovery] " + content_note, flush=True)
    print(f"[evaluation recovery] Retaining {len(old_rows) - len(failed)} valid/unchanged trials; "
          f"retrying {len(failed)} failed trials with the original seeds.", flush=True)
    out.mkdir(parents=True, exist_ok=False)
    retry_dir = out / "retry_trials"
    recovery_started = datetime.now(timezone.utc).isoformat()

    def publish(child=None, *, interrupted=False, child_report_error=None):
        if child is not None:
            _validate_effective_inference(original, child)
        child_rows = (_validate_rows(child.get("cases"), plan, original["seed"], allowed=failed)
                      if child is not None else {})
        merged = copy.deepcopy(original)
        if child is not None and "reference_mask_policy" in child:
            merged["reference_mask_policy"] = copy.deepcopy(child["reference_mask_policy"])
        merged["cases"] = [copy.deepcopy(child_rows.get(_key(row), row)) for row in original["cases"]]
        complete = len(merged["cases"]) == len(plan) and not interrupted
        merged["benchmark_complete"] = complete
        merged["status"] = ("interrupted" if interrupted else "incomplete" if not complete else
                            "completed_with_failures" if any(row["status"] == "failed" for row in merged["cases"]) else
                            "completed_with_unchanged_trials" if any(row["status"] == "unchanged" for row in merged["cases"]) else
                            "completed")
        merged["updated_at"] = datetime.now(timezone.utc).isoformat()
        merged["planned_cases"] = copy.deepcopy(cases)
        merged["planned_trial_count"] = len(plan)
        merged["recovery"] = {
            "source_report": str(original_path), "retry_report": str(retry_dir / "results.json") if failed else None,
            "started_at": recovery_started, "max_retries": max_retries,
            "adaptive_oom_retries": adaptive_oom_retries,
            "retry_execution": copy.deepcopy(child.get("execution")) if child else None,
            "report_every_seconds": float(report_every_seconds),
            "model_content_verified": model_verified, "input_content_verified": False,
            "content_identity_note": content_note,
            "reference_mask_compatibility": (
                "Legacy successful references passed exact 0/1 validation; their labels are "
                "unchanged under float32 > 0.5. Failed trials may use normalized fractional "
                "references with the training threshold. Original input files must be unchanged."
                if "reference_mask_policy" not in original and child is not None
                   and "reference_mask_policy" in child else
                "Reference mask policy preserved."),
            "requested_failed_trials": len(failed), "retried_trials": len(child_rows),
            "recovered_trials": sum(row["status"] == "ok" for row in child_rows.values()),
            "failed_to_unchanged_trials": sum(row["status"] == "unchanged" for row in child_rows.values()),
            "retained_ok_trials": sum(row["status"] == "ok" for row in old_rows.values()),
            "retained_unchanged_trials": sum(row["status"] == "unchanged" for row in old_rows.values()),
            "prior_failures": [{key: row[key] for key in (
                "id", "scan", "augmentation", "repeat", "seed", "error", "attempt_count",
                "attempt_errors", "attempt_settings", "retry_adjustments", "batch_backoff_history",
                "source_attempt_count", "source_attempt_errors") if key in row}
                for row in original["cases"] if row["status"] == "failed"],
        }
        if original.get("recovery"):
            merged["recovery"]["previous_recovery"] = copy.deepcopy(original["recovery"])
        if child_report_error:
            merged["recovery"]["child_report_error"] = child_report_error
        merged["timing"] = {"original_run": copy.deepcopy(original.get("timing")),
                            "retry_run": copy.deepcopy(child.get("timing")) if child else None,
                            "merged_trial_seconds": sum(row.get("total_seconds", 0.0) for row in merged["cases"])}
        merged["tiers"], merged["augmentations"] = summarize(
            merged["cases"], specs, [case["id"] for case in cases], original["repeats"])
        merged["tiers"] = [row for row in merged["tiers"] if row["tier"] in original["requested_tiers"]]
        _save_reports(out, merged, protected)
        return merged

    if not failed:
        return publish()
    try:
        child = evaluate_augmentations(
            original["model"], manifest=source.get("manifest"), data_dir=source.get("data_dir"),
            scan_glob=source["scan_glob"], mask_suffix=source["mask_suffix"], output_dir=retry_dir,
            repeats=original["repeats"], seed=original["seed"], max_cases=original.get("max_cases"),
            tiers=original["requested_tiers"], augmentation_names=[spec["name"] for spec in specs],
            artifact_severity=settings["artifact_severity"], rotation_degrees=settings["rotation_degrees"],
            resize_range=settings["resize_range"], resolution_range=settings["resolution_range"],
            augmentation_config_path=settings["augmentation_config_path"],
            masker_options=copy.deepcopy(original["masker_options"]), overwrite=False,
            max_retries=max_retries, report_every_seconds=report_every_seconds, trial_keys=failed,
            adaptive_oom_retries=adaptive_oom_retries)
        _validate_effective_inference(original, child)
        child_rows = _validate_rows(child.get("cases"), plan, original["seed"], allowed=failed)
        if set(child_rows) != failed:
            raise ValueError("Retry run did not return every requested failed trial")
    except BaseException:
        child, child_error = None, None
        child_path = retry_dir / "results.json"
        if child_path.is_file():
            try:
                child = _load_report(child_path)
                _validate_rows(child.get("cases"), plan, original["seed"], allowed=failed)
                _validate_effective_inference(original, child)
            except Exception as exc:
                child, child_error = None, f"{type(exc).__name__}: {exc}"
        publish(child, interrupted=True, child_report_error=child_error)
        raise
    return publish(child)


def _cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, help="Original augmentation results.json")
    parser.add_argument("--output-dir", required=True, help="New merged-report directory (must not exist)")
    parser.add_argument("--max-retries", type=int, default=2,
                        help="Additional attempts after the first retry attempt (default: 2)")
    parser.add_argument("--report-every-seconds", type=float, default=60)
    parser.add_argument("--no-adaptive-oom-retries", dest="adaptive_oom_retries", action="store_false", default=True,
                        help="Keep the original patch batch size on CUDA out-of-memory retries")
    args = parser.parse_args()
    report = retry_failed_evaluation(args.report, args.output_dir, max_retries=args.max_retries,
                                     report_every_seconds=args.report_every_seconds,
                                     adaptive_oom_retries=args.adaptive_oom_retries)
    print(json.dumps({"status": report["status"], **report["recovery"]}, indent=2))
    return 0 if report["status"] in ("completed", "completed_with_unchanged_trials") else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
