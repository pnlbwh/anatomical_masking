"""Evaluate a trained brain masker against held-out, native-space reference masks.

Examples:
    python evaluate.py --model runs/model.pt --data-dir heldout --output-dir reports/evaluation
    python evaluate.py --model runs/model.pt --manifest heldout.jsonl --save-masks

Directory discovery uses the training convention: ``*_T1w.nii.gz`` scans paired
with ``<scan stem>_brainmask.nii.gz`` (or ``.nii``). Pairing is exact;
an unrelated sibling mask is never guessed. A JSON list, JSONL, or CSV manifest
can instead specify ``scan`` and ``mask`` paths and an optional ``id`` for each
case. Relative manifest paths resolve against the manifest's own directory.

Inference is the same BrainMasker pipeline used by generate_mask.py, including
native-space restoration and postprocessing. One model is reused across cases.
Normalized reference masks use the training convention: float32 values > 0.5
are foreground. Predicted masks and metric inputs must remain strictly binary.
Reports include every case; failures have no metric values and do not enter
aggregates. Empty/empty Dice and IoU are 1; precision or recall with a zero
denominator is null. Supply a held-out set: this program does not partition data
or prove that a scan was absent from the model's training set.
"""

from __future__ import annotations

import argparse
import csv
import json
import io
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import nibabel as nib
import numpy as np

_HERE = str(Path(__file__).resolve().parents[1])
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from inference.masker import (BrainMasker, add_inference_arguments, inference_kwargs,
                           ensure_output_paths, atomic_write_text, paths_alias)
from imaging.geometry import nifti_affine_mm


METRICS = ("dice", "iou", "precision", "recall")
REFERENCE_MASK_POLICY = {
    "name": "training_float32_gt_0_5",
    "threshold": 0.5,
    "comparison": ">",
    "conversion_dtype": "float32",
    "allowed_range": [0.0, 1.0],
    "input_files_unchanged": True,
}


def _nifti_stem(path: Path) -> str:
    for ext in (".nii.gz", ".nii"):
        if path.name.lower().endswith(ext):
            return path.name[:-len(ext)]
    raise ValueError(f"Expected a .nii or .nii.gz NIfTI path: {path}")


def discover_cases(data_dir, scan_glob="*_T1w.nii.gz",
                   mask_suffix="_brainmask") -> List[Dict[str, str]]:
    """Discover deterministic exact pairs and retain missing/ambiguous cases."""
    root = Path(data_dir).resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)
    if not mask_suffix or "/" in mask_suffix or "\\" in mask_suffix:
        raise ValueError("mask_suffix must be a nonempty filename suffix, without an extension")
    records = []
    for scan in sorted(root.rglob(scan_glob)):
        if not scan.is_file():
            continue
        stem = _nifti_stem(scan)
        if "brainmask" in scan.name or "_brain." in scan.name or stem.endswith(mask_suffix):
            continue
        candidates = [scan.with_name(stem + mask_suffix + ext) for ext in (".nii.gz", ".nii")]
        found = [candidate for candidate in candidates if candidate.is_file()]
        record = {"id": str(scan.relative_to(root)), "scan": str(scan),
                  "mask": str(found[0] if found else candidates[0])}
        if len(found) > 1:
            record["discovery_error"] = f"Ambiguous reference masks: {[str(p) for p in found]}"
        elif not found:
            record["discovery_error"] = f"Reference mask missing; expected one of {[str(p) for p in candidates]}"
        records.append(record)
    if not records:
        raise FileNotFoundError(f"No input scans matched {scan_glob!r} under {root}")
    return records


def read_manifest(manifest) -> List[Dict[str, str]]:
    """Read explicit scan/mask pairs; accept the deployment manifest field names."""
    path = Path(manifest).resolve()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        if path.suffix.lower() == ".csv":
            rows = list(csv.DictReader(stream))
        elif path.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in stream if line.strip()]
        else:
            rows = json.load(stream)
            if isinstance(rows, dict):
                rows = rows.get("cases", rows.get("records"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest must contain a nonempty list of scan/mask records")
    cases, seen_scans, seen_ids = [], set(), set()
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict) or not row.get("scan") or not row.get("mask"):
            raise ValueError(f"Manifest row {index} needs nonempty 'scan' and 'mask' fields")
        case = {}
        for key in ("scan", "mask"):
            if not isinstance(row[key], str):
                raise ValueError(f"Manifest row {index}: {key} must be a path string")
            value = Path(row[key]).expanduser()
            case[key] = str((value if value.is_absolute() else path.parent / value).resolve())
        case["id"] = str(row.get("id") or index)
        if (case["scan"] in seen_scans or case["id"] in seen_ids
                or any(paths_alias(case["scan"], previous) for previous in seen_scans)):
            raise ValueError(f"Duplicate scan or case id in manifest row {index}")
        if paths_alias(case["scan"], case["mask"]):
            raise ValueError(f"Manifest row {index} uses the same file for scan and mask")
        seen_scans.add(case["scan"])
        seen_ids.add(case["id"])
        cases.append(case)
    return cases


def _reference_mask(data, *, path):
    """Apply the training reference protocol without modifying the source data/file."""
    values = np.asarray(data)
    if not np.isfinite(values).all():
        raise ValueError(f"Reference mask must contain finite values in [0, 1]; found NaN or infinity: {path}")
    if ((values < 0) | (values > 1)).any():
        raise ValueError(f"Reference mask must contain normalized values in [0, 1]; "
                         f"observed range [{values.min():g}, {values.max():g}]: {path}")
    # Match training.data's float32 > 0.5 conversion, including values close to 0.5.
    return np.asarray(values, dtype=np.float32) > 0.5


def _load_volume(path, *, binary=False, reference=False):
    """Load a finite 3D volume; reference conversion and strict binary checks are separate."""
    if binary and reference:
        raise ValueError("binary and reference loading modes are mutually exclusive")
    img = nib.load(str(path))
    nifti_affine_mm(img)  # validates the spatial dimensions, units, and affine
    data = img.get_fdata(dtype=np.float64).reshape(img.shape[:3])
    if reference:
        return img, _reference_mask(data, path=path)
    if not np.isfinite(data).all():
        raise ValueError(f"Volume contains NaN or infinity: {path}")
    if binary and not np.logical_or(data == 0, data == 1).all():
        raise ValueError(f"Binary mask must contain only binary 0 and 1 values: {path}")
    return img, data.astype(bool) if binary else data


def _check_grid(scan_img, scan_shape, mask_img, mask_shape, *, affine_atol):
    if tuple(scan_shape) != tuple(mask_shape):
        raise ValueError(f"Scan/mask grid shapes differ: {tuple(scan_shape)} vs {tuple(mask_shape)}")
    if not np.allclose(nifti_affine_mm(scan_img), nifti_affine_mm(mask_img), atol=affine_atol, rtol=0):
        raise ValueError("Scan/mask physical affines differ; align the reference to the scan before evaluation")


def binary_metrics(prediction: np.ndarray, reference: np.ndarray) -> Dict[str, Any]:
    """Confusion counts and overlap scores, with explicit undefined denominators."""
    if prediction.shape != reference.shape:
        raise ValueError("Prediction and reference shapes differ")
    for label, values in (("prediction", prediction), ("reference", reference)):
        if not np.logical_or(values == 0, values == 1).all():
            raise ValueError(f"{label} is not binary")
    pred, ref = prediction.astype(bool), reference.astype(bool)
    tp = int(np.count_nonzero(pred & ref))
    fp = int(np.count_nonzero(pred & ~ref))
    fn = int(np.count_nonzero(~pred & ref))
    tn = int(pred.size - tp - fp - fn)
    return _scores_from_counts(tp, fp, fn, tn)


def _scores_from_counts(tp, fp, fn, tn) -> Dict[str, Any]:
    total = 2 * tp + fp + fn
    union = tp + fp + fn
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn, "true_negative": tn,
            "dice": 2 * tp / total if total else 1.0,
            "iou": tp / union if union else 1.0,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None}


def evaluate_model(model_path, *, data_dir=None, manifest=None,
                   scan_glob="*_T1w.nii.gz", mask_suffix="_brainmask",
                   output_dir="reports/evaluation", save_masks=False, affine_atol=1e-4,
                   overwrite=False, masker_options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run deployed-mask evaluation and save cases.csv plus results.json.

    ``masker_options`` accepts BrainMasker keyword arguments. Failures are recorded
    individually and excluded from aggregate scores; configuration/model-loading
    errors are raised before processing cases.
    """
    if (data_dir is None) == (manifest is None):
        raise ValueError("Specify exactly one of data_dir or manifest")
    if not np.isfinite(affine_atol) or affine_atol < 0:
        raise ValueError("affine_atol must be finite and nonnegative")
    cases = read_manifest(manifest) if manifest is not None else discover_cases(data_dir, scan_glob, mask_suffix)
    model_path = Path(model_path).resolve()
    out = Path(output_dir).resolve()
    report_paths = [out / "cases.csv", out / "results.json"]
    predictions = []
    for index, case in enumerate(cases, 1):
        try:
            stem = _nifti_stem(Path(case["scan"]))
        except ValueError:
            # Invalid scan names belong in per-case failures, not cohort setup.
            stem = "invalid_scan"
        predictions.append(out / "masks" / f"{index:05d}_{stem}_mask.nii.gz")
    protected = {Path(case[key]).resolve() for case in cases for key in ("scan", "mask")}
    protected.add(model_path)
    if manifest is not None:
        protected.add(Path(manifest).resolve())
    options = dict(masker_options or {})
    options["overwrite"] = overwrite
    # Resolve metadata once so its explicit or automatic sidecar is also protected.
    masker = BrainMasker(model_path=model_path, **options)
    protected.update(Path(path).resolve() for path in masker.input_paths())
    all_outputs = report_paths + (predictions if save_masks else [])
    ensure_output_paths(all_outputs, protected, overwrite=overwrite)
    masker.load_model()
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for index, case in enumerate(cases):
        row = {"id": case["id"], "scan": case["scan"], "reference_mask": case["mask"]}
        prediction_path = predictions[index]
        try:
            if case.get("discovery_error"):
                raise ValueError(case["discovery_error"])
            _nifti_stem(Path(case["scan"]))
            scan_img, scan_data = _load_volume(case["scan"])
            ref_img, reference = _load_volume(case["mask"], reference=True)
            _check_grid(scan_img, scan_data.shape, ref_img, reference.shape, affine_atol=affine_atol)
            del scan_data
            result = masker.predict(scan_path=case["scan"], image=scan_img)
            _check_grid(ref_img, reference.shape, result.image, result.mask.shape, affine_atol=affine_atol)
            record = (masker.save_prediction(result, output_path=prediction_path, protected_paths=protected)
                      if save_masks else result.record)
            row.update(status="ok", **binary_metrics(result.mask, reference))
            row["prediction_mask"] = str(prediction_path) if save_masks else None
            row["inference"] = {key: value for key, value in record.items() if key != "mask"}
            row["review_flag"] = record["review_flag"]
        except Exception as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            print(f"[evaluation] {case['id']}: {row['error']}", file=sys.stderr, flush=True)
        results.append(row)
        print(f"[evaluation] {index + 1}/{len(cases)} {case['id']}: {row['status']}", flush=True)
    good = [row for row in results if row["status"] == "ok"]
    macro = {}
    for metric in METRICS:
        values = [row[metric] for row in good if row[metric] is not None]
        macro[metric] = {"mean": float(np.mean(values)) if values else None,
                         "std": float(np.std(values)) if values else None,
                         "defined_cases": len(values)}
    count_keys = ("true_positive", "false_positive", "false_negative", "true_negative")
    micro = _scores_from_counts(*(sum(row[key] for row in good) for key in count_keys)) if good else None
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": str(model_path), "evaluation_space": "native scan grid, deployed binary masks",
        "input": {"manifest": str(Path(manifest).resolve()) if manifest is not None else None,
                  "data_dir": str(Path(data_dir).resolve()) if data_dir is not None else None,
                  "scan_glob": scan_glob, "mask_suffix": mask_suffix},
        "case_count": len(results), "successful_cases": len(good), "failed_cases": len(results) - len(good),
        "review_flagged_cases": sum(bool(row.get("review_flag")) for row in good),
        "affine_tolerance": affine_atol,
        "reference_mask_policy": dict(REFERENCE_MASK_POLICY),
        "metric_policy": {"empty_empty_dice_iou": 1.0, "zero_denominator_precision_recall": None,
                          "failed_cases": "excluded from aggregates; retained in cases",
                          "std": "population standard deviation",
                          "heldout_status": "supplied by caller, not automatically verified"},
        "macro": macro, "micro": micro, "cases": results,
    }
    fields = ["id", "scan", "reference_mask", "status", *METRICS, *count_keys,
              "prediction_mask", "review_flag", "error"]
    # Serialize both reports before changing either destination. A malformed value
    # cannot leave a new CSV paired with an absent/stale JSON report.
    json_text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    csv_stream = io.StringIO(newline="")
    writer = csv.DictWriter(csv_stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(results)
    ensure_output_paths(report_paths, protected, overwrite=overwrite)
    atomic_write_text(report_paths[0], csv_stream.getvalue(), protected_paths=protected, overwrite=overwrite)
    atomic_write_text(report_paths[1], json_text, protected_paths=protected, overwrite=overwrite)
    return report


def _cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="trained masking checkpoint (.pt)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-dir", help="held-out raw scans and reference masks")
    source.add_argument("--manifest", help="JSON list, JSONL, or CSV containing scan/mask paths")
    parser.add_argument("--scan-glob", default="*_T1w.nii.gz")
    parser.add_argument("--mask-suffix", default="_brainmask")
    parser.add_argument("--output-dir", default="reports/evaluation", help="directory for results.json and cases.csv")
    parser.add_argument("--save-masks", action="store_true", help="also keep generated native-space masks")
    parser.add_argument("--affine-atol", type=float, default=1e-4, help="absolute physical affine comparison tolerance in mm")
    add_inference_arguments(parser)
    args = parser.parse_args()
    options = inference_kwargs(args)
    options.pop("overwrite")
    report = evaluate_model(
        args.model, data_dir=args.data_dir, manifest=args.manifest,
        scan_glob=args.scan_glob, mask_suffix=args.mask_suffix, output_dir=args.output_dir,
        save_masks=args.save_masks, affine_atol=args.affine_atol,
        overwrite=args.overwrite, masker_options=options,
    )
    print(json.dumps({key: report[key] for key in ("successful_cases", "failed_cases", "macro", "micro")}, indent=2))
    print(f"Reports: {Path(args.output_dir).resolve()}")
    return 1 if report["failed_cases"] else 0


if __name__ == "__main__":
    raise SystemExit(_cli())
