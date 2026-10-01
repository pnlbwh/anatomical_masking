"""Filter a directory of NIfTI scans and reuse the condensed brain masker."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import random
import sys
import uuid

from anatomical_masking.inference.masker import (
    BrainMasker, add_inference_arguments, atomic_write_text, inference_kwargs,
)


def _strings(value, name):
    values = [value] if isinstance(value, str) else list(value or [])
    if any(not isinstance(item, str) or not item for item in values):
        raise ValueError(f"{name} must contain nonempty strings")
    return values


def _nifti_parts(path):
    name = Path(path).name
    for ext in (".nii.gz", ".nii"):
        if name.lower().endswith(ext):
            return name[:-len(ext)], name[-len(ext):]
    raise ValueError(f"Expected a .nii or .nii.gz file: {path}")


def _path_keys(path):
    """Recognize resolved aliases and existing hard links without quadratic comparisons."""
    path = Path(path)
    keys = {("path", os.path.normcase(str(path.resolve())))}
    try:
        stat = path.stat()
    except FileNotFoundError:
        return keys
    if stat.st_ino:
        keys.add(("inode", stat.st_dev, stat.st_ino))
    return keys


def _process_one(masker, scan, output, overwrite):
    scan, output = Path(scan), Path(output)
    record = {"scan": str(scan), "mask": str(output)}
    lock = output.with_name(output.name + ".masking.lock")
    claimed = False
    result = None
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with lock.open("x", encoding="utf-8") as stream:
                claimed = True
                json.dump({"pid": os.getpid(), "scan": str(scan)}, stream)
        except FileExistsError:
            return {**record, "status": "skipped", "reason": "output_locked"}
        # Another batch may have finished after discovery but before this claim.
        if output.exists() and not overwrite:
            result = {**record, "status": "skipped", "reason": "output_exists"}
        else:
            prediction = masker.run(scan_path=scan, output_path=output)
            result = {**prediction, **record, "status": "succeeded"}
    except Exception as exc:
        result = {**record, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        if claimed:
            try:
                lock.unlink(missing_ok=True)
            except OSError as exc:
                message = f"Could not remove {lock}: {exc}"
                if result is not None:
                    result["lock_cleanup_error"] = message
                print(message, file=sys.stderr, flush=True)
    return result


_worker_masker = None
_worker_overwrite = False


def _init_worker(model_path, options):
    """Spawn-safe initializer: load exactly one independent model per worker."""
    global _worker_masker, _worker_overwrite
    _worker_masker = BrainMasker(model_path=model_path, **options)
    _worker_masker.load_model()
    _worker_overwrite = options["overwrite"]


def _worker_process(scan, output):
    return _process_one(_worker_masker, scan, output, _worker_overwrite)


class MaskDirectory:
    """Generate masks beside matching scans, or under a mirrored output directory.

    All keywords must match the filename; any excluded keyword rejects it. Matching
    is case-insensitive. Only .nii/.nii.gz inputs are supported. Existing outputs
    are skipped unless overwrite=True. One worker reuses one model (the default,
    suitable for a single GPU); each additional worker loads its own model copy.
    Remaining keyword arguments are forwarded to BrainMasker.

    run() returns per-file records and writes logs. run(dry_run=True) only lists
    planned outputs: it does not load the model or create files. Public succeeded,
    failed, and skipped attributes contain scan paths from the latest run.
    """

    def __init__(self, input_dir, model_path, keywords=None, output_suffix="_brainmask",
                 recursive=True, file_extensions=None, overwrite=False,
                 excluded_keywords=None, parent_folder=None, shuffle=False,
                 num_workers=1, output_dir=None, seed=None, remove_from_stem=None,
                 **inference_options):
        self.input_dir = Path(input_dir).expanduser().resolve()
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {self.input_dir}")
        if not self.input_dir.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {self.input_dir}")
        if (not isinstance(output_suffix, str) or not output_suffix.strip()
                or any(c in output_suffix for c in '/\\\x00:*?"<>|')):
            raise ValueError("output_suffix must be a nonempty filename suffix without path separators")
        if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 1:
            raise ValueError("num_workers must be a positive integer")
        self.model_path = Path(model_path).expanduser().resolve()
        self.output_suffix = output_suffix
        self.recursive = bool(recursive)
        self.overwrite = bool(overwrite)
        self.keywords = [s.casefold() for s in _strings(keywords, "keywords")]
        self.excluded_keywords = [s.casefold() for s in _strings(excluded_keywords, "excluded_keywords")]
        extensions = _strings(file_extensions, "file_extensions") if file_extensions is not None else [".nii", ".nii.gz"]
        self.file_extensions = [s.lower() if s.startswith(".") else "." + s.lower() for s in extensions]
        if not self.file_extensions or any(s not in (".nii", ".nii.gz") for s in self.file_extensions):
            raise ValueError("file_extensions must select .nii and/or .nii.gz")
        self.parent_folder = parent_folder.casefold() if parent_folder is not None else None
        self.shuffle = bool(shuffle)
        self.seed = seed
        self.num_workers = num_workers
        self.output_dir = Path(output_dir).expanduser().resolve() if output_dir is not None else None
        if self.output_dir is not None and self.output_dir.exists() and not self.output_dir.is_dir():
            raise NotADirectoryError(f"Output path is not a directory: {self.output_dir}")
        self.remove_from_stem = _strings(remove_from_stem, "remove_from_stem")
        self.inference_options = dict(inference_options, overwrite=self.overwrite)
        self.logs_root = (self.output_dir or self.input_dir) / "mask_logs"
        self.logs_dir = None
        self.succeeded, self.failed, self.skipped = [], [], []
        self._skipped_records = []

    def _make_output_path(self, file_path):
        file_path = Path(file_path)
        stem, ext = _nifti_parts(file_path)
        for text in self.remove_from_stem:
            stem = stem.replace(text, "")
        if not stem:
            raise ValueError(f"remove_from_stem removes the entire filename: {file_path}")
        parent = (self.output_dir / file_path.parent.relative_to(self.input_dir)
                  if self.output_dir is not None else file_path.parent)
        return parent / f"{stem}{self.output_suffix}{ext}"

    def _collect_files(self):
        self.skipped, self._skipped_records = [], []
        files = self.input_dir.rglob("*") if self.recursive else self.input_dir.glob("*")
        pairs = []
        for path in files:
            if not path.is_file():
                continue
            resolved = path.resolve()
            if resolved.is_relative_to(self.logs_root.resolve()):
                continue
            if (self.output_dir is not None and self.output_dir != self.input_dir
                    and self.output_dir.is_relative_to(self.input_dir)
                    and resolved.is_relative_to(self.output_dir)):
                continue
            name = path.name.casefold()
            if not any(name.endswith(ext) for ext in self.file_extensions):
                continue
            stem, _ = _nifti_parts(path)
            if stem.casefold().endswith(self.output_suffix.casefold()):
                continue
            if self.parent_folder is not None and path.parent.name.casefold() != self.parent_folder:
                continue
            if not all(word in name for word in self.keywords):
                continue
            if any(word in name for word in self.excluded_keywords):
                continue
            pairs.append((path, self._make_output_path(path)))
        pairs.sort(key=lambda pair: str(pair[0]).casefold())

        # Validate the entire plan before skipping outputs or writing any masks.
        protected = set()
        sources = [p for p, _ in pairs] + [self.model_path]
        config_path = self.inference_options.get("config_path")
        if config_path:
            sources.append(Path(config_path))
        for path in sources:
            protected.update(_path_keys(path))
        destinations = set()
        pending = []
        for scan, output in pairs:
            keys = _path_keys(output)
            if keys & protected:
                raise ValueError(f"Output would replace an input or model/config file: {output}")
            if keys & destinations:
                raise ValueError(f"Multiple scans map to the same output: {output}")
            destinations.update(keys)
            if output.exists() and not output.is_file():
                raise ValueError(f"Output is not a file: {output}")
            if output.exists() and not self.overwrite:
                self.skipped.append(str(scan))
                self._skipped_records.append({"scan": str(scan), "mask": str(output),
                                              "status": "skipped", "reason": "output_exists"})
            else:
                pending.append((scan, output))
        return pending

    def run(self, dry_run=False):
        self.succeeded, self.failed, self.logs_dir = [], [], None
        pairs = self._collect_files()
        if self.shuffle:
            random.Random(self.seed).shuffle(pairs)
        planned = [{"scan": str(scan), "mask": str(output)} for scan, output in pairs]
        report = {"succeeded": [], "failed": [], "skipped": list(self._skipped_records),
                  "review": [], "planned": planned, "logs_dir": None}
        print(f"Found {len(pairs)} scans to mask; {len(self.skipped)} existing outputs skipped.", flush=True)
        if dry_run:
            for scan, output in pairs:
                print(f"{scan} -> {output}", flush=True)
            return report
        if pairs and not self.model_path.is_file():
            raise FileNotFoundError(f"Model checkpoint does not exist: {self.model_path}")

        masker = None
        if pairs and self.num_workers == 1:
            masker = BrainMasker(model_path=self.model_path, **self.inference_options)
            masker.load_model()  # Fail once on an invalid checkpoint, before touching outputs.
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
        self.logs_dir = self.logs_root / run_id
        self.logs_dir.mkdir(parents=True, exist_ok=False)
        report["logs_dir"] = str(self.logs_dir)
        records_path = self.logs_dir / "results.jsonl"
        completed = 0
        with records_path.open("x", encoding="utf-8") as journal:
            for rec in self._skipped_records:
                journal.write(json.dumps(rec, allow_nan=False) + "\n")
            journal.flush()

            def record(rec):
                nonlocal completed
                completed += 1
                status = rec["status"]
                report[status].append(rec)
                getattr(self, status).append(rec["scan"])
                if status == "succeeded" and rec.get("review_flag"):
                    report["review"].append(rec)
                journal.write(json.dumps(rec, allow_nan=False) + "\n")
                journal.flush()
                detail = rec.get("error") or rec.get("reason") or ("needs review" if rec.get("review_flag") else "")
                print(f"[{completed}/{len(pairs)}] {status}: {rec['scan']}" + (f" ({detail})" if detail else ""), flush=True)

            try:
                if self.num_workers == 1:
                    for scan, output in pairs:
                        record(_process_one(masker, scan, output, self.overwrite))
                elif pairs:
                    with ProcessPoolExecutor(max_workers=self.num_workers,
                                             mp_context=multiprocessing.get_context("spawn"),
                                             initializer=_init_worker,
                                             initargs=(str(self.model_path), self.inference_options)) as executor:
                        futures = {executor.submit(_worker_process, str(scan), str(output)): (scan, output)
                                   for scan, output in pairs}
                        for future in as_completed(futures):
                            scan, output = futures[future]
                            try:
                                rec = future.result()
                            except Exception as exc:
                                rec = {"scan": str(scan), "mask": str(output), "status": "failed",
                                       "error": f"{type(exc).__name__}: {exc}"}
                            record(rec)
            finally:
                for name in ("succeeded", "failed", "skipped", "review"):
                    lines = "".join(rec["scan"] + "\n" for rec in report[name])
                    atomic_write_text(self.logs_dir / f"{name}.txt", lines)
        print(f"Done. {len(self.succeeded)} succeeded, {len(self.failed)} failed, "
              f"{len(self.skipped)} skipped; {len(report['review'])} need review.", flush=True)
        print(f"Logs: {self.logs_dir}", flush=True)
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path, help="trained masking checkpoint (.pt)")
    parser.add_argument("--keywords", nargs="+", default=None, help="all must occur in the filename (case-insensitive)")
    parser.add_argument("--excluded-keywords", nargs="+", default=None, help="skip filenames containing any of these")
    parser.add_argument("--parent-folder", help="only scan files whose immediate parent has this name")
    parser.add_argument("--output-suffix", default="_brainmask")
    parser.add_argument("--output-dir", type=Path, help="mirror input subfolders here; default: beside each scan")
    parser.add_argument("--file-extensions", nargs="+", default=None, help=".nii and/or .nii.gz (default: both)")
    parser.add_argument("--no-recursive", dest="recursive", action="store_false")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, help="reproducible shuffle order")
    parser.add_argument("--num-workers", type=int, default=1, help="one model copy per worker; use 1 for a single GPU")
    parser.add_argument("--remove-from-stem", action="append", help="optional literal text to remove before adding the suffix")
    parser.add_argument("--dry-run", action="store_true", help="list selected scans/outputs without loading a model or writing files")
    add_inference_arguments(parser)
    args = parser.parse_args(argv)
    try:
        runner = MaskDirectory(
            args.input_dir, args.model, keywords=args.keywords, output_suffix=args.output_suffix,
            recursive=args.recursive, file_extensions=args.file_extensions,
            excluded_keywords=args.excluded_keywords, parent_folder=args.parent_folder,
            shuffle=args.shuffle, seed=args.seed, num_workers=args.num_workers,
            output_dir=args.output_dir, remove_from_stem=args.remove_from_stem,
            **inference_kwargs(args),
        )
        report = runner.run(dry_run=args.dry_run)
    except (ValueError, TypeError, OSError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 1 if report["failed"] else 0
