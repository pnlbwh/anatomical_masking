"""Generate a native-space binary brain mask using a trained masking checkpoint.

    python generate_mask.py --scan scan.nii.gz --out mask.nii.gz --model runs/model.pt

Model definitions and postprocessing are shared via models/architectures.py.
Checkpoint metadata determines architecture and resize/conform/patch preprocessing.
The original deployment inference behavior is retained: robust normalization,
foreground z-scoring (optional two-pass refinement), orientation restoration,
optional flip TTA, and configurable threshold/component/dilation postprocessing.
Training may normalize within the reference mask; inference uses the available
foreground proxy. Parameters must be assessed for the model and data being used.

BrainMasker(scan_path, output_path, model_path).run() saves the mask and returns
its inference settings and review flags. Reuse an instance with
run(scan_path=..., output_path=...) to reuse its cached model across scans.
For in-memory work, BrainMasker(model_path=...).predict(scan_path=...) returns a
MaskPrediction containing the native-grid mask, source image, and metadata.
"""

from __future__ import annotations

import json
import os
import sys
import math
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import nibabel as nib
from nibabel.orientations import (
    apply_orientation,
    axcodes2ornt,
    io_orientation,
    ornt_transform,
)
import torch
import torch.nn.functional as F

# Allow direct execution as well as imports through the bundle entry point.
_HERE = str(Path(__file__).resolve().parents[1])
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from models.architectures import postprocess_mask
from imaging.geometry import nifti_affine_mm
from inference.checkpoints import (MaskingCheckpoint, load_masking_model,
                                   _checkpoint_name, _find_sidecar_config, _read_masking_config)
from imaging.normalization import normalize_intensity, zscore


def paths_alias(left, right) -> bool:
    """Compare resolved names and existing filesystem identities (including hardlinks)."""
    left, right = Path(left), Path(right)
    if left.resolve() == right.resolve():
        return True
    return left.exists() and right.exists() and left.samefile(right)


def ensure_output_paths(paths, protected_paths=(), *, overwrite=False) -> None:
    """Reject outputs that identify an input, even when replacing outputs is allowed."""
    outputs = [Path(path) for path in paths]
    protected = [Path(path) for path in protected_paths if path is not None]
    for index, output in enumerate(outputs):
        for source in protected:
            if paths_alias(output, source):
                raise ValueError(f"Output would replace an input file: {output} (input: {source})")
        for previous in outputs[:index]:
            if paths_alias(output, previous):
                raise ValueError(f"Output paths identify the same file: {previous} and {output}")
    if not overwrite:
        existing = [str(path) for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite existing file(s) (pass overwrite=True to allow): "
                + ", ".join(existing)
            )


def atomic_write_text(path, text, *, protected_paths=(), overwrite=False) -> None:
    """Stage a complete text file beside its destination, then publish by replacement."""
    path = Path(path)
    ensure_output_paths([path], protected_paths, overwrite=overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                         prefix=f".{path.name}.", suffix=".tmp",
                                         dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        ensure_output_paths([path], protected_paths, overwrite=overwrite)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _number(value, name, *, minimum=None, maximum=None, integer=False,
            min_inclusive=True, max_inclusive=True):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if isinstance(value, (bool, np.bool_)) or not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and (result < minimum or (not min_inclusive and result == minimum)):
        raise ValueError(f"{name} must be {'>=' if min_inclusive else '>'} {minimum}")
    if maximum is not None and (result > maximum or (not max_inclusive and result == maximum)):
        raise ValueError(f"{name} must be {'<=' if max_inclusive else '<'} {maximum}")
    if integer and not result.is_integer():
        raise ValueError(f"{name} must be an integer")
    return int(result) if integer else result


def _spatial_shape(value, name):
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must contain three positive integers")
    try:
        dimensions = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain three positive integers") from exc
    if len(dimensions) != 3:
        raise ValueError(f"{name} must contain three positive integers")
    return tuple(_number(item, name, minimum=1, integer=True) for item in dimensions)


def _validate_tensor(value, expected_shape, name, *, probability=False):
    if not torch.is_tensor(value) or tuple(value.shape) != tuple(expected_shape):
        raise ValueError(f"{name} must be a tensor of shape {tuple(expected_shape)}")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite or non-floating-point values")
    if probability and (bool((value < 0).any()) or bool((value > 1).any())):
        raise ValueError(f"{name} must contain probabilities in [0, 1]")
    return value


def _infer_logits(model, volume):
    # Validate before sigmoid/clamping, which otherwise hides infinite model outputs.
    return _validate_tensor(model.infer(volume), (volume.shape[0], 1, *volume.shape[2:]),
                            "Model logits")


# ---------------------------------------------------------------------------- spatial canonicalization
_IDENTITY_ORNT = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])  # axcodes unchanged (no permute/flip)


def _parse_axcodes(spec) -> Optional[Tuple[str, str, str]]:
    """Parse a canonical-orientation spec into an (a, b, c) axcodes tuple, or None to disable.

    Accepts a 3-char string ("PIR") or a 3-sequence (("P","I","R")). None / False /
    "none" / "native" -> None (reorientation disabled). Validates via ``axcodes2ornt``
    (raises on a non-orthonormal code) so a typo fails LOUD instead of silently mis-orienting.
    """
    if spec is None or spec is False:
        return None
    if isinstance(spec, str):
        s = spec.strip().lower()
        if s in ("", "none", "native", "off", "false"):
            return None
        codes = tuple(spec.strip().upper())
    else:
        codes = tuple(str(c).upper() for c in spec)
    if len(codes) != 3:
        raise ValueError(f"canonical_orientation must be 3 axis codes (e.g. 'PIR'); got {spec!r}")
    ornt = axcodes2ornt(codes)  # raises ValueError on an unknown axis code (e.g. 'X')
    # axcodes2ornt does NOT reject DUPLICATE axes ('PPP' -> degenerate), so check the axis column
    # is a true permutation of {0,1,2} (one of each L/R, A/P, S/I) ourselves.
    if sorted(int(a) for a in ornt[:, 0]) != [0, 1, 2]:
        raise ValueError(
            f"canonical_orientation {''.join(codes)!r} is not a valid 3D orientation "
            "(need one code per axis from L/R, A/P, S/I, e.g. 'PIR' or 'RAS')."
        )
    return codes  # type: ignore[return-value]


def _orientation_transforms(affine: np.ndarray, canonical_axcodes: Tuple[str, str, str]):
    """LOSSLESS ornt transforms native<->canonical (axis permute + flip only, exactly invertible).

    Returns ``(fwd, bwd, needs)``:
      * ``fwd`` maps a NATIVE-frame voxel array to the ``canonical_axcodes`` frame,
      * ``bwd`` maps a CANONICAL-frame array back to native (the exact inverse of ``fwd``),
      * ``needs`` is False when the scan is ALREADY in the canonical frame (``fwd`` is the
        identity ornt) -> the caller skips the reorientation and stays byte-identical.

    A permute+flip is exactly invertible and resamples NOTHING, so the mask round-trip
    (reorient scan -> infer -> reorient mask back) is bit-exact and never introduces an
    interpolation seam. Spacing/anisotropy is a SEPARATE axis (handled by conform), not here.
    """
    native = io_orientation(affine)
    canon = axcodes2ornt(tuple(canonical_axcodes))
    fwd = ornt_transform(native, canon)
    bwd = ornt_transform(canon, native)
    needs = not np.array_equal(fwd, _IDENTITY_ORNT)
    return fwd, bwd, needs


@dataclass
class MaskPrediction:
    """A binary uint8 mask on the source image grid, plus inference provenance.

    ``image`` retains the original spatial header for saving; ``record['mask']``
    is None until a copy of the record is returned by ``save_prediction``.
    """

    mask: np.ndarray
    image: Any
    record: Dict[str, Any]


class BrainMasker:
    """Produce and save a brain mask for one scan with a trained masking model.

    Parameters
    ----------
    scan_path : str | Path
        Input MRI scan (NIfTI: ``.nii`` / ``.nii.gz``).
    output_path : str | Path
        Where to write the brain mask (NIfTI, uint8, native space).
    model_path : str | Path
        Trained `MaskingUNet` weights — a ``.pt`` state-dict from `train.py`.

    Keyword-only options (sensible defaults; you only need the three paths above)
    ---------------------------------------------------------------------------
    device : str | None
        Torch device (e.g. ``"cuda:0"``). Default: cuda if available, else cpu.
    threshold, dilate_iters, max_fraction : float, int, float
        `postprocess_mask` knobs (defaults: 0.60 / 0 / 0.75).
    target_shape : tuple(int,int,int)
        Whole-volume resize size when there is no sidecar config (default 128³, the training default).
        Overridden by a sidecar `config.target_shape` when one is found.
    refine_normalization : bool
        Opt into the two-pass z-score refinement (default False; see module docstring).
    config_path : str | Path | None
        Explicit training results-JSON to read preprocessing config from. None -> auto-detect next to
        the model; embedded checkpoint metadata takes precedence over detected sidecars. Pass ``False`` to disable
        auto-detection and force the `target_shape` resize default.
    model_kwargs : dict | None
        Extra kwargs for `build_model("masking", ...)`. None -> read `config.model_kwargs` if present,
        else MaskingUNet defaults (which match a default-trained checkpoint).
    sw_overlap, sw_batch_size : float, int
        Patch-mode sliding-window inference overlap / windows-per-forward (defaults 0.5 / 4). A
        sidecar-recorded `sw_overlap` (the training run's val-selected grid) overrides the `sw_overlap`
        arg. Patch stitching uses Gaussian (edge-down-weighted) blending.
    tta : bool
        Mirror-flip test-time augmentation (average the prob over identity + 3 axis-flips; a label-safe
        variance reducer). Default True; automatically skipped in conform mode. Pass False to infer from
        the single identity view only.
    canonical_orientation : str | None
        Training-frame orientation to reorient the scan to (lossless) before resize/patch inference,
        mask mapped back to native. ``"auto"`` (default) reads the sidecar `canonical_orientation`
        if recorded, else falls back to ``"PIR"`` (the NFBS training frame). Pass an explicit axcodes
        string (``"PIR"`` / ``"RAS"``) to override, or ``None`` / ``"native"`` to DISABLE (legacy
        behavior: infer in the scan's native orientation). Ignored in conform mode (which
        self-canonicalizes via the affine).
    overwrite : bool
        Allow overwriting an existing output mask (default False — shared-server safety).
    """

    def __init__(
        self,
        scan_path=None,
        output_path=None,
        model_path=None,
        *,
        device: Optional[str] = None,
        threshold: float = 0.60,
        dilate_iters: int = 0,
        cc_keep_ratio: float = 1.0,
        dilate_mm: Optional[float] = None,
        max_fraction: float = 0.75,
        target_shape: Tuple[int, int, int] = (128, 128, 128),
        refine_normalization: bool = False,
        config_path=None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        sw_overlap: float = 0.5,
        sw_batch_size: int = 4,
        norm_percentile: float = 99.0,
        canonical_orientation="auto",
        tta: bool = True,
        overwrite: bool = False,
    ):
        if model_path is None:
            raise ValueError("model_path is required")
        self.scan_path = Path(scan_path) if scan_path is not None else None
        self.output_path = Path(output_path) if output_path is not None else None
        self.model_path = Path(model_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.threshold = _number(threshold, "threshold", minimum=0, maximum=1)
        self.dilate_iters = _number(dilate_iters, "dilate_iters", minimum=0, integer=True)
        # Component policy, forwarded to postprocess_mask (which owns the default 0.05). 1.0 keeps
        # ONLY the largest component; 0.05 keeps every component >= 5% of it; 0.0 keeps everything.
        self.cc_keep_ratio = _number(cc_keep_ratio, "cc_keep_ratio", minimum=0, maximum=1)
        if self.cc_keep_ratio >= 1.0:
            # Loud, because this is the one setting that can DELETE real tissue rather than add it.
            print(
                "[warn] component policy 'largest': only the single biggest connected component is "
                "kept. Disconnected regions are removed; use --component-policy keep-large "
                "or keep-all when that behavior is inappropriate for the data.",
                file=sys.stderr, flush=True,
            )
        self.dilate_mm = (_number(dilate_mm, "dilate_mm", minimum=0)
                          if dilate_mm is not None else None)
        self.max_fraction = _number(max_fraction, "max_fraction", minimum=0, maximum=1)
        self.refine_normalization = bool(refine_normalization)
        self.sw_overlap = _number(sw_overlap, "sw_overlap", minimum=0, maximum=1, max_inclusive=False)
        self.sw_batch_size = _number(sw_batch_size, "sw_batch_size", minimum=1, integer=True)
        self.norm_percentile = _number(norm_percentile, "norm_percentile", minimum=0,
                                       maximum=100, min_inclusive=False)
        target_shape = _spatial_shape(target_shape, "target_shape")
        self.tta = bool(tta)
        self.overwrite = bool(overwrite)
        self._model = None  # loaded lazily; reuse this instance for a sequence of scans
        self._checkpoint = MaskingCheckpoint(self.model_path)

        # ---- resolve the spatial-preprocessing mode from the sidecar config (or fail-loud defaults).
        cfg: Optional[Dict[str, Any]] = None
        if config_path is False:
            cfg = None  # auto-detection explicitly disabled
        elif config_path is not None:
            cfg = _read_masking_config(Path(config_path))
            if cfg is None:
                raise FileNotFoundError(
                    f"config_path {config_path!r} is not a readable masking results JSON "
                    "(expected {'model_type': 'masking', 'config': {...}})."
                )
        else:
            cfg = _find_sidecar_config(self.model_path, checkpoint=self._checkpoint)
        self.config = cfg
        # Preprocessing is UNVERIFIED when auto-detection found no sidecar: we then assume the
        # documented 128^3 resize default, which is WRONG for a patch/conform-trained checkpoint. Warn
        # loudly and record it (a clinician/operator must confirm the model is default-resize-trained).
        self.preproc_unverified = (config_path is None) and (cfg is None)
        if self.preproc_unverified:
            print(
                f"[warn] no training results-JSON found next to {self.model_path.name}: assuming the "
                f"default {tuple(target_shape)} whole-volume RESIZE preprocessing. If this checkpoint "
                "was trained with --patch-size or --conform-mm, the mask will be WRONG — pass --config "
                "<results.json>. (record: unverified_preprocessing=true)",
                file=sys.stderr, flush=True,
            )

        # patch+conform -> sliding windows on a canonical physical grid; patch only -> native sliding;
        # conform only -> whole-volume canonical grid; else -> legacy resize.
        patch_size = (cfg or {}).get("patch_size")
        conform_mm = (cfg or {}).get("conform_mm")
        cfg_shape = (cfg or {}).get("target_shape")
        self.target_shape = _spatial_shape(cfg_shape if cfg_shape is not None else target_shape, "target_shape")
        self.patch_size = _spatial_shape(patch_size, "patch_size") if patch_size is not None else None
        self.conform_mm = (_number(conform_mm, "conform_mm", minimum=0, min_inclusive=False)
                           if conform_mm is not None else None)
        if self.patch_size is not None and self.conform_mm is not None:
            self.mode = "conform_patch"
        elif self.patch_size is not None:
            self.mode = "patch"
        elif self.conform_mm is not None:
            self.mode = "conform"
        else:
            self.mode = "resize"

        # Patch-mode sliding-window overlap: prefer the value the training run RECORDED in the sidecar
        # (train.py writes `sw_overlap` into its results config), so inference reproduces the exact
        # val-selected grid rather than this module's default. Fall back to the `sw_overlap` arg when the
        # sidecar doesn't carry it (older results JSONs / no sidecar).
        cfg_overlap = (cfg or {}).get("sw_overlap")
        if cfg_overlap is not None:
            self.sw_overlap = _number(cfg_overlap, "sw_overlap", minimum=0, maximum=1,
                                      max_inclusive=False)

        # model_kwargs: explicit arg wins, else sidecar config, else defaults.
        if model_kwargs is not None:
            self.model_kwargs = dict(model_kwargs)
        else:
            self.model_kwargs = dict((cfg or {}).get("model_kwargs") or {})

        # Canonical orientation: "auto" -> sidecar `canonical_orientation` if recorded, else the
        # documented PIR fallback (the NFBS training frame); an explicit value overrides; None/"native"
        # disables. Parsed to an axcodes tuple (or None). Only APPLIED in resize/patch mode (conform
        # self-canonicalizes via the affine) — that gating happens in run().
        if isinstance(canonical_orientation, str) and canonical_orientation.strip().lower() == "auto":
            recorded = (cfg or {}).get("canonical_orientation")
            canon_spec = recorded or "PIR"
            # match-or-fail-loud: PIR here is a GUESS (no frame recorded next to the model). We still
            # default to it (correct for every NFBS-trained checkpoint) but WARN at reorientation time
            # so a non-PIR-trained model can't silently produce a plausible-but-wrong mask.
            self._canon_is_default_guess = recorded is None
        else:
            canon_spec = canonical_orientation
            self._canon_is_default_guess = False
        self.canonical_axcodes = _parse_axcodes(canon_spec)

    # -- public API ---------------------------------------------------------------

    def run(self, *, scan_path=None, output_path=None) -> Dict[str, Any]:
        """Predict and save a mask, retaining the historical path-based API."""
        next_scan = Path(scan_path) if scan_path is not None else self.scan_path
        next_output = self._output_path(output_path)
        ensure_output_paths([next_output], self.input_paths(scan_path=next_scan),
                            overwrite=self.overwrite)
        result = self.predict(scan_path=next_scan)
        return self.save_prediction(result, output_path=next_output)

    def predict(self, *, scan_path=None, image=None) -> MaskPrediction:
        """Return a native-grid mask and metadata without writing any files.

        Supply ``image`` to reuse a loaded NIfTI or predict a synthetic image in
        memory. ``scan_path`` identifies its source in the returned metadata.
        """
        if scan_path is not None:
            next_scan = Path(scan_path)
        elif image is not None:
            filename = image.get_filename()
            next_scan = Path(filename) if filename is not None else None
        else:
            next_scan = self.scan_path
        if image is None:
            if next_scan is None:
                raise ValueError("predict requires a scan_path or an image")
            image = nib.load(str(next_scan))
        img = image
        # Spatial work is always in mm; saving retains native units and forms.
        affine = nifti_affine_mm(img)
        vol = np.asarray(img.get_fdata(), dtype=np.float64).reshape(img.shape[:3])
        native_shape = tuple(int(size) for size in vol.shape)
        self.scan_path = next_scan
        scan_name = next_scan.name if next_scan is not None else "in-memory image"

        # Fail LOUD on a blank/corrupt/constant scan rather than silently emitting an empty mask. The
        # normalizer sanitizes stray NaN/Inf (so a few bad voxels still produce a valid mask), but a
        # wholly non-finite or single-valued volume is a broken file, not a scan — surface it. (`or`
        # short-circuits, so np.nanmax is never reached on an all-NaN volume.)
        if not np.isfinite(vol).any() or float(np.nanmax(vol)) <= float(np.nanmin(vol)):
            raise ValueError(
                f"{self.scan_path} is empty or non-finite (blank, constant, or corrupt download) — "
                "no brain to mask.")

        model = self.load_model()

        # --- spatial canonicalization (resize/patch modes): reorient the scan to the model's training
        # frame with a LOSSLESS axis permute+flip, run the whole pipeline there, then map the binary
        # mask back to native with the exact inverse. This is the cross-dataset generalization fix —
        # a foreign orientation otherwise reaches the net mirrored/permuted (see class docstring).
        # Conform mode self-canonicalizes via the affine, so it stays in native space here.
        reoriented = False
        bwd = None
        work_vol, work_shape = vol, native_shape
        if self.canonical_axcodes is not None and self.mode in ("resize", "patch"):
            fwd, bwd, needs = _orientation_transforms(affine, self.canonical_axcodes)
            if needs:  # already-canonical scans skip this entirely -> byte-identical (e.g. NFBS PIR)
                work_vol = apply_orientation(vol, fwd)
                work_shape = tuple(int(s) for s in work_vol.shape)
                reoriented = True
                if self._canon_is_default_guess:
                    print(
                        f"[warn] reoriented {''.join(nib.aff2axcodes(affine))} -> "
                        f"{''.join(self.canonical_axcodes)} using the DEFAULT training frame (no "
                        "`canonical_orientation` recorded next to the model). Correct for NFBS-trained "
                        "checkpoints; if this model was trained on a non-PIR corpus, pass "
                        "canonical_orientation=<frame> (or retrain so it is recorded).",
                        file=sys.stderr, flush=True,
                    )
            elif self._canon_is_default_guess:
                # Silent-no-reorient case: the scan is ALREADY stored in the guessed frame, so no
                # permute happens — but the frame is still a GUESS. Warn so a non-PIR-trained model
                # can't pass through unnoticed (this branch produced NO warning before).
                print(
                    f"[warn] {scan_name} is already in the assumed training frame "
                    f"{''.join(self.canonical_axcodes)} (a GUESS — none recorded next to the model); no "
                    "reorientation applied. If this checkpoint was trained on a non-PIR corpus the mask "
                    "may be wrong — pass canonical_orientation=<frame>.",
                    file=sys.stderr, flush=True,
                )

        # B2: in resize mode the volume is anisotropically squashed to target_shape IGNORING voxel
        # spacing, so a non-cubic FOV (e.g. a foreign [208,300,320] MPRAGE) reaches the net with warped
        # geometry vs the ~cubic training corpus -> a plausible-but-wrong mask. Flag the aspect risk.
        aspect_ratio = round(float(max(work_shape)) / float(max(1, min(work_shape))), 3)
        if self.mode == "resize" and aspect_ratio > 1.5:
            print(
                f"[warn] {scan_name}: anisotropic FOV (work shape {tuple(work_shape)}, "
                f"aspect {aspect_ratio}) is resized to {self.target_shape} ignoring voxel spacing — the "
                "brain geometry is distorted vs the ~cubic training corpus and the mask may be wrong "
                "(the known foreign-FOV failure). Inspect the mask; a conform-trained model is the fix.",
                file=sys.stderr, flush=True,
            )

        vol01 = normalize_intensity(work_vol, self.norm_percentile)

        # Physical-grid models conform BEFORE z-scoring, matching online training's new
        # conform -> augment -> normalize order. Legacy resize/native-patch models keep their exact
        # historical preprocessing.
        conform_mode = self.mode in ("conform", "conform_patch")
        if conform_mode:
            prob = self._predict_conform(model, vol01, work_shape, affine, region_mask=None,
                                         patch_mode=(self.mode == "conform_patch"))
        else:
            prob = self._predict_prob_tta(model, zscore(vol01, None), work_shape, affine)

        prob = _validate_tensor(prob, (1, 1, *work_shape), "Native probability map", probability=True)

        # Optional pass 2: re-z-score INSIDE a rough mask, then re-predict (opt-in; see docstring).
        refined = False
        if self.refine_normalization:
            rough = (prob[0, 0].cpu().numpy() >= 0.5)
            if rough.any():
                if conform_mode:
                    prob = self._predict_conform(model, vol01, work_shape, affine, region_mask=rough,
                                                 patch_mode=(self.mode == "conform_patch"))
                else:
                    prob = self._predict_prob_tta(model, zscore(vol01, rough), work_shape, affine)
                refined = True

        prob = _validate_tensor(prob, (1, 1, *work_shape), "Native probability map", probability=True)

        # WORK-FRAME voxel sizes for an optional mm-based (spacing-aware) border. `prob` (and the mask
        # dilated from it) is on the work grid at native work resolution — resize/patch return prob at
        # work_shape, conform returns it at native — so derive zooms from the WORK affine (NOT native: a
        # reoriented anisotropic scan would otherwise get transposed per-axis sampling). fwd is defined
        # exactly when reoriented is True.
        work_zooms = None
        if self.dilate_mm is not None:
            from nibabel.affines import voxel_sizes as _voxel_sizes
            if reoriented:
                from nibabel.orientations import inv_ornt_aff
                work_affine = affine @ inv_ornt_aff(fwd, native_shape)
            else:
                work_affine = affine
            work_zooms = tuple(float(z) for z in _voxel_sizes(work_affine))
            aniso = max(work_zooms) / max(1e-6, min(work_zooms))
            if aniso > 1.3:
                print(
                    f"[warn] {scan_name}: anisotropic voxels {tuple(round(z, 2) for z in work_zooms)} "
                    f"mm (ratio {aniso:.2f}) with --dilate-mm {self.dilate_mm}: the physical border grows "
                    "fewer voxels along the thick axis (less over-inclusive there) than a voxel dilation "
                    "would. Verify the through-plane margin on this scan.",
                    file=sys.stderr, flush=True,
                )
        effective_dilate_mm = self.dilate_mm
        if effective_dilate_mm is not None and effective_dilate_mm > 0:
            # NIfTI-1 stores spatial fields as float32: 0.001 meters may round to
            # 1.000000047 mm. Allow header-precision error at the closed EDT border
            # so physically equivalent unit encodings include the same neighbors.
            precision = np.finfo(img.header["srow_x"].dtype).eps
            tolerant_radius = effective_dilate_mm * (1.0 + 8.0 * precision)
            if math.isfinite(tolerant_radius):
                effective_dilate_mm = tolerant_radius
        mask_t, flags_list = postprocess_mask(
            prob, threshold=self.threshold,
            dilate_iters=(0 if self.dilate_mm is not None else self.dilate_iters),
            max_fraction=self.max_fraction, return_flags=True,
            voxel_sizes=work_zooms, dilate_mm=effective_dilate_mm,
            cc_keep_ratio=self.cc_keep_ratio,
        )
        mask = mask_t[0, 0].cpu().numpy().astype(np.uint8)
        qflags = flags_list[0]  # per-volume review flags (oversize/undersize/empty/dropped_component)

        # Map the mask from the WORK (canonical) frame back to the scan's NATIVE frame (exact inverse
        # permute+flip) so it aligns with the source affine we save it under. No-op if not reoriented.
        if reoriented:
            mask = np.ascontiguousarray(apply_orientation(mask, bwd)).astype(np.uint8)

        voxels = int(mask.sum())
        record = {
            "scan": str(next_scan) if next_scan is not None else None,
            "mask": None,
            "model": str(self.model_path),
            "native_shape": list(native_shape),
            "mode": self.mode,
            "target_shape": list(self.target_shape),
            "patch_size": list(self.patch_size) if self.patch_size else None,
            "conform_mm": self.conform_mm,
            "config_source": (self.config or {}).get("__source__"),
            "device": str(self.device),
            "sw_batch_size": self.sw_batch_size,
            "sw_overlap": self.sw_overlap,
            # The full post-processing operating point, recorded so a mask can always be traced back
            # to the settings that produced it (these are tunable and cohort-specific).
            "threshold": self.threshold,
            "dilate_iters": self.dilate_iters,
            "dilate_mm": self.dilate_mm,
            "cc_keep_ratio": self.cc_keep_ratio,
            "component_policy": ("largest" if self.cc_keep_ratio >= 1.0
                                 else "keep-all" if self.cc_keep_ratio <= 0.0 else "keep-large"),
            "canonical_orientation": ("".join(self.canonical_axcodes) if self.canonical_axcodes else None),
            "canonical_is_guess": bool(self._canon_is_default_guess),  # PIR frame was assumed, not recorded
            "reoriented": reoriented,  # True == scan was reoriented to the training frame for inference
            "aspect_ratio": aspect_ratio,  # work-shape max/min axis; >1.5 in resize mode = FOV-distortion risk
            "unverified_preprocessing": bool(self.preproc_unverified),  # no sidecar -> assumed resize default
            "tta": self.tta and not self.mode.startswith("conform"),
            "refined_normalization": refined,
            "mask_voxels": voxels,
            "mask_fraction": round(voxels / float(np.prod(native_shape)), 6),
            # Symmetric review flags: under-segmentation (cutting brain) is surfaced too, not just
            # oversize. Any True here means a human must review/reject this mask before downstream use.
            "oversize_flag": bool(qflags["oversize"]),      # mask exceeds max_fraction -> too large
            "undersize_flag": bool(qflags["undersize"]),    # implausibly small -> brain likely cut
            "empty_flag": bool(qflags["empty"]),            # no mask at all -> maximal false negative
            "dropped_component_flag": bool(qflags["dropped_component"]),  # a real region was discarded
            "review_flag": bool(qflags["review"]),          # any of the above -> needs human review
        }
        if qflags["review"]:
            reasons = [k for k in ("oversize", "undersize", "empty", "dropped_component") if qflags[k]]
            print(
                f"[warn] {scan_name}: brain mask needs REVIEW ({', '.join(reasons)}; "
                f"mask_fraction={record['mask_fraction']}). Do not use downstream without inspection.",
                file=sys.stderr, flush=True,
            )
        return MaskPrediction(mask=mask, image=img, record=record)

    def save_prediction(self, prediction: MaskPrediction, *, output_path=None,
                        protected_paths=()) -> Dict[str, Any]:
        """Atomically save a prediction while protecting its source and model files."""
        destination = self._output_path(output_path)
        protected = [*self.input_paths(scan_path=prediction.record.get("scan")),
                     *protected_paths]
        self._save_mask(prediction.mask, prediction.image, destination,
                        protected_paths=protected, overwrite=self.overwrite)
        self.output_path = destination
        return {**prediction.record, "mask": str(destination)}

    __call__ = run

    def input_paths(self, *, scan_path=None):
        """Files that output publication must never replace."""
        paths = [Path(scan_path) if scan_path is not None else self.scan_path, self.model_path]
        source = (self.config or {}).get("__source__")
        if source and "::preproc_config" not in source:
            paths.append(Path(source))
        return [path for path in paths if path is not None]

    def load_model(self) -> torch.nn.Module:
        """Load and cache the configured model; useful for batch setup validation."""
        if self._model is None:
            self._model = load_masking_model(
                self.model_path, device=self.device, model_kwargs=self.model_kwargs,
                checkpoint=self._checkpoint,
            )
            self._checkpoint = None  # release the extra checkpoint tensor storage
        return self._model

    # -- internals ----------------------------------------------------------------

    def _output_path(self, output_path):
        path = Path(output_path) if output_path is not None else self.output_path
        if path is None:
            raise ValueError("Saving a mask requires an output_path")
        return path

    def _predict_prob_tta(
        self, model: torch.nn.Module, vol_norm: np.ndarray,
        work_shape: Tuple[int, int, int], affine: np.ndarray,
    ) -> torch.Tensor:
        """Test-time augmentation: average the sigmoid prob over the identity + three axis-flips.

        The one genuine net-side OOD lever (the literature attributes contrast-agnosticism to the
        training synthesis, not the net — see MASKER_SYNTHESIS_PLAN.md). Each flipped view is predicted
        then un-flipped so all views align before averaging; brain masking is flip-equivariant, so this
        is label-safe and reduces boundary/pose variance. HONEST: every view passes the SAME net, so on
        a true-OOD input the views can agree on the same wrong blob — TTA helps most AFTER the synthesis
        retrain widens coverage; it is a variance reducer, not a domain-gap fix. Skipped in conform mode
        (flipping the array desyncs the affine the conform path resamples through).
        """
        if not self.tta or self.mode.startswith("conform"):
            return self._predict_prob(model, vol_norm, work_shape, affine)
        probs = [_validate_tensor(self._predict_prob(model, vol_norm, work_shape, affine),
                                  (1, 1, *work_shape), "TTA probability map", probability=True)]
        for ax in range(3):
            flipped = np.ascontiguousarray(np.flip(vol_norm, ax))
            p = _validate_tensor(self._predict_prob(model, flipped, work_shape, affine),
                                 (1, 1, *work_shape), "TTA probability map", probability=True)
            probs.append(torch.flip(p, dims=[ax + 2]))  # un-flip back to the canonical frame (B,C,D,H,W)
        return torch.stack(probs, dim=0).mean(dim=0)

    @torch.no_grad()
    def _predict_prob(
        self, model: torch.nn.Module, vol_norm: np.ndarray,
        native_shape: Tuple[int, int, int], affine: np.ndarray,
    ) -> torch.Tensor:
        """Run the model on a native-resolution z-scored volume and return a probability map
        ``[1, 1, *native_shape]`` (CPU), using the spatial mode the model was trained with."""
        if self.mode == "patch":
            return self._predict_patch(model, vol_norm)
        if self.mode == "conform":
            return self._predict_conform(model, vol_norm, native_shape, affine)
        if self.mode == "conform_patch":
            return self._predict_conform(model, vol_norm, native_shape, affine, patch_mode=True)
        return self._predict_resize(model, vol_norm, native_shape)

    def _predict_resize(self, model, vol_norm, native_shape) -> torch.Tensor:
        """Whole-volume path: resize to `target_shape`, infer, resize the prob back to native.

        Uses trilinear / align_corners=False to mirror the trainer's `_load_resized` exactly (which
        is anisotropic by design — it ignores voxel spacing — so we reproduce that, not 'fix' it)."""
        t = torch.from_numpy(np.ascontiguousarray(vol_norm))[None, None].to(self.device).float()
        t = F.interpolate(t, size=self.target_shape, mode="trilinear", align_corners=False)
        prob = torch.sigmoid(_infer_logits(model, t))
        prob = F.interpolate(prob, size=tuple(native_shape), mode="trilinear", align_corners=False)
        return _validate_tensor(prob, (1, 1, *native_shape), "Resized probability map",
                                probability=True).cpu()

    def _predict_patch(self, model, vol_norm) -> torch.Tensor:
        """Native-resolution patch path: MONAI sliding-window inference (matches patch-mode training)."""
        from monai.inferers import sliding_window_inference

        t = torch.from_numpy(np.ascontiguousarray(vol_norm))[None, None].to(self.device).float()
        # mode="gaussian" (sigma_scale=0.125) down-weights patch-edge voxels when stitching windows, so
        # a boundary that lands mid-patch is smoothly blended instead of hard-seamed — removes the patch-
        # seam FP/FN that "constant" (uniform) blending leaves. overlap default is 0.5 (see sw_overlap).
        logits = sliding_window_inference(
            t, self.patch_size, self.sw_batch_size, lambda window: _infer_logits(model, window),
            overlap=self.sw_overlap, mode="gaussian", sigma_scale=0.125,
        )
        _validate_tensor(logits, (1, 1, *vol_norm.shape), "Stitched model logits")
        return torch.sigmoid(logits).cpu()

    def _predict_conform(self, model, vol01, native_shape, affine, *, region_mask=None,
                         patch_mode: bool = False) -> torch.Tensor:
        """Conform the robust-[0,1] scan first, z-score on that physical grid, infer, and map back.

        ``patch_mode`` runs sliding-window inference on the conformed full volume, matching training
        with both ``--conform-mm`` and ``--patch-size``. ``region_mask`` is an optional native-grid
        rough mask for the refinement pass and is conformed with nearest-neighbour interpolation.
        """
        import nibabel.processing as nibp

        src_img = nib.Nifti1Image(vol01.astype(np.float32), affine)
        src_img.header.set_xyzt_units("mm")
        conf = nibp.conform(
            src_img, out_shape=tuple(self.target_shape),
            voxel_size=(self.conform_mm,) * 3, order=3,
        )
        conf_arr = np.asarray(conf.get_fdata(), dtype=np.float32)
        conf_region = None
        if region_mask is not None and np.asarray(region_mask).any():
            rimg = nib.Nifti1Image(np.asarray(region_mask, dtype=np.float32), affine)
            rimg.header.set_xyzt_units("mm")
            rconf = nibp.resample_from_to(rimg, (conf.shape, conf.affine), order=0)
            conf_region = np.asarray(rconf.get_fdata(), dtype=np.float32) > 0.5
        conf_norm = zscore(conf_arr, conf_region)
        if patch_mode:
            prob_conf = self._predict_patch(model, conf_norm)[0, 0].numpy().astype(np.float32)
        else:
            t = torch.from_numpy(conf_norm)[None, None].to(self.device)
            prob_conf = torch.sigmoid(_infer_logits(model, t))[0, 0].cpu().numpy().astype(np.float32)
        # Resample probability from the conformed grid back to the original voxel grid.
        prob_img = nib.Nifti1Image(prob_conf, conf.affine)
        prob_img.header.set_xyzt_units("mm")
        prob_native = nibp.resample_from_to(prob_img, (tuple(native_shape), affine), order=1)
        arr = np.asarray(prob_native.get_fdata(), dtype=np.float32)
        if not np.isfinite(arr).all():
            raise ValueError("Resampled probability map contains non-finite values")
        return torch.from_numpy(np.clip(arr, 0.0, 1.0))[None, None]

    @staticmethod
    def _save_mask(mask: np.ndarray, ref_img, path: Path, *, protected_paths=(), overwrite=False) -> None:
        """Atomically save a binary mask with the source's exact spatial header fields.

        Copy active and disabled form codes verbatim. In particular, do not invent an
        active qform for a sheared sform: a quaternion cannot represent that shear.
        All non-spatial image metadata starts from a pristine header.
        """
        path = Path(path)
        protected = list(protected_paths)
        if ref_img.get_filename() is not None:
            protected.append(ref_img.get_filename())
        ensure_output_paths([path], protected, overwrite=overwrite)
        suffix = ".nii.gz" if path.name.lower().endswith(".nii.gz") else ".nii"
        if not path.name.lower().endswith((".nii", ".nii.gz")):
            raise ValueError(f"Output mask must end in .nii or .nii.gz: {path}")
        data = np.asarray(mask)
        if data.shape != tuple(ref_img.shape[:3]) or not np.logical_or(data == 0, data == 1).all():
            raise ValueError("Output mask must be binary and match the source's 3D grid")
        image_class = nib.Nifti2Image if isinstance(ref_img, nib.Nifti2Image) else nib.Nifti1Image
        hdr = image_class.header_class()
        hdr.set_data_dtype(np.uint8)
        hdr.set_slope_inter(1, 0)
        for field in ("qform_code", "sform_code", "quatern_b", "quatern_c", "quatern_d",
                      "qoffset_x", "qoffset_y", "qoffset_z", "srow_x", "srow_y", "srow_z",
                      "xyzt_units"):
            hdr[field] = ref_img.header[field]
        hdr["pixdim"][:4] = ref_img.header["pixdim"][:4]
        out = image_class(data.astype(np.uint8), None, hdr)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=suffix,
                                             dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
            nib.save(out, str(temporary))
            ensure_output_paths([path], protected, overwrite=overwrite)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def generate_mask(scan_path, output_path, model_path, **kwargs) -> Dict[str, Any]:
    """One-call convenience wrapper around `BrainMasker(...).run()`."""
    return BrainMasker(scan_path, output_path, model_path, **kwargs).run()


def add_inference_arguments(p):
    """Add shared preprocessing/postprocessing CLI options to a parser."""
    p.add_argument("--device", default=None, help="torch device, e.g. cuda:0 (default: cuda if available else cpu)")
    p.add_argument("--sw-batch-size", type=int, default=4,
                   help="sliding-window patches per forward pass (default 4); larger values use more GPU memory")
    p.add_argument("--threshold", type=float, default=0.60,
                   help="probability threshold for the binary mask (default: 0.60)")
    p.add_argument("--dilate-iters", type=int, default=0,
                   help="number of voxel dilation iterations after thresholding (default: 0)")
    p.add_argument("--component-policy", choices=("largest", "keep-large", "keep-all"),
                   default="largest",
                   help="keep the largest connected component (default), components >=5%% of "
                        "the largest, or all components")
    p.add_argument("--dilate-mm", type=float, default=None,
                   help="grow the border by this physical distance in mm, accounting for voxel "
                        "spacing; overrides --dilate-iters")
    p.add_argument("--max-fraction", type=float, default=0.75, help="flag (never zero) a mask larger than this fraction")
    p.add_argument("--target-shape", type=int, nargs=3, default=[128, 128, 128],
                   help="whole-volume resize size when there is no sidecar config (default 128 128 128)")
    p.add_argument("--refine-normalization", action="store_true",
                   help="opt into two-pass z-score refinement (rough mask -> re-z-score inside -> final)")
    p.add_argument("--config", default=None,
                   help="explicit training results-JSON for preprocessing config (default: auto-detect next to --model)")
    p.add_argument("--no-config", action="store_true",
                   help="disable sidecar auto-detection; force the --target-shape resize default")
    p.add_argument("--canonical-orientation", default="auto",
                   help="reorient the scan to this training-frame orientation (lossless) before "
                        "resize/patch inference; mask mapped back to native. 'auto' (default) reads "
                        "the sidecar or falls back to 'PIR' (the NFBS training frame). Pass an axcodes "
                        "string ('PIR'/'RAS') to override, or 'native' to DISABLE. Ignored in conform mode.")
    p.add_argument("--no-tta", dest="tta", action="store_false", default=True,
                   help="DISABLE test-time augmentation. TTA (averaging the prob over identity + 3 "
                        "axis-flips) is ON by default: a variance reducer that is skipped in conform "
                        "mode. Pass --no-tta to infer from the single identity view only.")
    p.add_argument("--overwrite", action="store_true", help="allow replacing existing outputs")


def inference_kwargs(args) -> Dict[str, Any]:
    """Translate shared CLI flags into BrainMasker keyword arguments."""
    return dict(
        device=args.device,
        sw_batch_size=args.sw_batch_size,
        threshold=args.threshold,
        dilate_iters=args.dilate_iters,
        # Same vocabulary as masker_threshold_sweep.py's COMPONENT_POLICIES, so the winning row of
        # a sweep can be transcribed straight onto this CLI.
        cc_keep_ratio={"largest": 1.0, "keep-large": 0.05, "keep-all": 0.0}[args.component_policy],
        dilate_mm=args.dilate_mm,
        max_fraction=args.max_fraction,
        target_shape=tuple(args.target_shape),
        refine_normalization=args.refine_normalization,
        config_path=(False if args.no_config else args.config),
        canonical_orientation=args.canonical_orientation,
        tta=args.tta,
        overwrite=args.overwrite,
    )


def _cli():
    import argparse

    p = argparse.ArgumentParser(description="Generate a brain mask for one scan with a trained masking model.")
    p.add_argument("--scan", required=True, help="input MRI scan (NIfTI)")
    p.add_argument("--out", required=True, help="output mask path (NIfTI, uint8)")
    p.add_argument("--model", required=True, help="trained masking checkpoint (.pt)")
    add_inference_arguments(p)
    args = p.parse_args()
    rec = BrainMasker(args.scan, args.out, args.model, **inference_kwargs(args)).run()
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    _cli()
