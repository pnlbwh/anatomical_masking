"""NIfTI datasets, physical crops, subject-safe splits, and DataLoader construction."""
from __future__ import annotations

# Import first so spawn workers establish the same numerical thread boundary.
import training.runtime as training_runtime
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from imaging.geometry import nifti_affine_mm
from models.losses import signed_distance_transform_numpy

def _case_identity(record: Dict[str, Any]) -> str:
    """Acquisition identity independent of an input folder or split-list position."""
    explicit = record.get("case_id")
    if explicit is not None:
        if not isinstance(explicit, str) or not explicit.strip():
            raise ValueError("case_id must be a nonempty string")
        return explicit.strip()
    source = Path(str(record.get("source_scan") or record["scan"])).name.casefold()
    scan = Path(str(record["scan"])).name.casefold()
    acquisition = source if source == scan else f"{source}/{scan}"
    return f"{record.get('subject_id', '')}::{acquisition}"


def _case_rng(seed: int, record: Dict[str, Any], variant: int = 0):
    identity = f"{record.get('subject_id', '')}::{_case_identity(record)}"
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    words = [int.from_bytes(digest[i:i + 4], "little") for i in range(0, 16, 4)]
    return np.random.default_rng([int(seed), int(variant), *words])


def _load_resized(path: str, shape: Tuple[int, int, int], mode: str) -> torch.Tensor:
    """Load and resize a scan (trilinear) or reference mask (nearest)."""
    img = nib.load(str(path))
    nifti_affine_mm(img)
    vol = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32).reshape(img.shape[:3])
    return _resize_arr(vol, shape, mode)


def _load_conformed(path: str, out_shape: Tuple[int, int, int], voxel_mm: float, order: int) -> torch.Tensor:
    """Load onto the shared isotropic grid: cubic scans (order=3), nearest masks (0)."""
    img = nib.load(str(path))
    affine = nifti_affine_mm(img)
    arr = np.asarray(img.get_fdata(dtype=np.float32)).reshape(img.shape[:3])
    return _conform_arrays(arr, affine, out_shape, voxel_mm, order)


def _conform_arrays(arr, affine, out_shape: Tuple[int, int, int], voxel_mm: float, order: int) -> torch.Tensor:
    """Conform an array to an isotropic grid using its physical affine.

    Unlike resizing, conforming preserves physical proportions. Paired image/mask
    affines and target grids must agree; interpolation order keeps masks binary.
    """
    import nibabel.processing as nibp
    img = nib.Nifti1Image(np.asarray(arr, dtype=np.float32), affine)
    img.header.set_xyzt_units("mm")
    conf = nibp.conform(img, out_shape=tuple(out_shape), voxel_size=(float(voxel_mm),) * 3, order=order)
    return torch.from_numpy(np.asarray(conf.get_fdata(), dtype=np.float32))[None]  # [1, D, H, W]


def _drop_resampling_specks(mask: np.ndarray, max_fraction: float = 0.01) -> np.ndarray:
    """Remove tiny detached islands that NEAREST conform of a thin mask edge leaves behind.

    A connected native mask can pick up an isolated voxel on the conformed grid (NFBS A00062351 at
    2 mm: one 177259-voxel brain + a single 1-voxel speck). Morph3d's connectivity guard correctly
    refuses a disconnected source, so the speck crashed label synthesis. Only islands smaller than
    ``max_fraction`` of the largest component are dropped; a genuinely split mask stays split and
    still fails loudly downstream."""
    from scipy import ndimage as ndi
    b = np.asarray(mask, dtype=bool)
    cc, n = ndi.label(b, structure=ndi.generate_binary_structure(b.ndim, b.ndim))
    if n <= 1:
        return b
    sizes = np.bincount(cc.ravel())[1:]
    speck_ids = np.flatnonzero(sizes < max_fraction * sizes.max()) + 1
    if speck_ids.size:
        b = b & ~np.isin(cc, speck_ids)
    return b


def _load_full(path: str) -> torch.Tensor:
    """Load a NIfTI volume at NATIVE resolution (no resize). Returns [1, D, H, W].

    Patch mode reads full-res so focal artifacts (metal voids, thin motion/Gibbs bands) survive —
    the 128^3 whole-volume resize is exactly what blurs them away.
    """
    img = nib.load(str(path))
    nifti_affine_mm(img)
    vol = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32).reshape(img.shape[:3])
    return torch.from_numpy(vol)[None]  # [1, D, H, W]


def _resize_arr(arr: np.ndarray, shape: Tuple[int, int, int], mode: str) -> torch.Tensor:
    """Resize an in-memory [D,H,W] array to `shape` -> [1, *shape] (used by online synthesis)."""
    t = torch.from_numpy(np.ascontiguousarray(arr))[None, None].float()
    kw = {"mode": mode}
    if mode != "nearest":
        kw["align_corners"] = False
    return F.interpolate(t, size=shape, **kw)[0]


def _pad_chw(t: torch.Tensor, size: Tuple[int, int, int]) -> torch.Tensor:
    """Pad a [C, D, H, W] tensor with zeros so each spatial dim is >= the patch size.

    Scans are z-scored on disk, so 0 ~ background — a benign pad value. F.pad takes the LAST spatial
    dim first, hence the (W, H, D) ordering of the pad tuple.
    """
    _, d, h, w = t.shape
    pd, ph, pw = max(0, size[0] - d), max(0, size[1] - h), max(0, size[2] - w)
    if pd or ph or pw:
        t = F.pad(t, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2, pd // 2, pd - pd // 2))
    return t


def _worker_init_fn(worker_id: int) -> None:
    """Re-seed each DataLoader worker's dataset ``rng`` so patch crops de-correlate across workers.

    ``num_workers>0`` FORKS the dataset — every worker inherits the SAME ``np.random.default_rng(seed)``
    state, so without reseeding they would all draw the identical crop offset for a given index (the
    classic fork-duplication trap). Reseed from ``torch.initial_seed()`` (NOT the trainer ``--seed``):
    torch already derives a distinct base seed per worker AND per epoch, so this de-correlates crops
    across workers/epochs while staying independent of the reproducibility-sensitive ``--seed`` (seeding
    from ``--seed`` would REintroduce the duplication trap). Datasets without an ``rng`` are left untouched.
    """
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    # SciPy/OpenCV/BLAS may otherwise start their own thread pools inside EVERY loader worker. Eight
    # workers x eight native threads oversubscribes the CPU badly and can make the GPU wait longer
    # than a serial loader. One native thread per worker gives the DataLoader explicit control of
    # preprocessing parallelism. This runs only in worker subprocesses, never in the trainer process.
    torch.set_num_threads(1)
    try:  # OpenCV is optional in some masker-only environments.
        import cv2
        cv2.setNumThreads(0)
    except (ImportError, AttributeError):
        pass
    ds = info.dataset
    if getattr(ds, "rng", None) is not None:
        ds.rng = np.random.default_rng(torch.initial_seed() % (2 ** 32))


def _collate_masker_crops(batch):
    """Flatten K crops from each loaded volume into an ordinary patch batch.

    ``OnlineSynthDataset(crops_per_volume=K>1)`` returns one ``[K,C,D,H,W]`` tensor for the scan
    and mask. Concatenating across loader items produces ``[B*K,C,D,H,W]``, so the model, loss,
    deep-supervision path, AMP, and metrics all retain their established 5D contracts. Keeping this
    as a top-level function also makes it picklable for spawn-started DataLoader workers.
    """
    scans, targets = zip(*batch)
    if isinstance(targets[0], dict):
        target_batch = {k: torch.cat([t[k] for t in targets], dim=0) for k in targets[0]}
    else:
        target_batch = torch.cat(targets, dim=0)
    return torch.cat(scans, dim=0), target_batch


def _boundary_distance_maps(padded_mask_chw, voxel_sizes=None):
    """Compute the two physical EDT maps shared by repeated boundary-crop draws."""
    from scipy import ndimage as ndi

    m = padded_mask_chw[0].detach().cpu().numpy() > 0.5
    if not m.any():
        return m, None, None
    sampling = tuple(float(x) for x in (voxel_sizes or (1.0, 1.0, 1.0)))
    return (m, ndi.distance_transform_edt(m, sampling=sampling),
            ndi.distance_transform_edt(~m, sampling=sampling))


def _boundary_offset(padded_mask_chw, size, dims, rng, voxel_sizes=None, distance_maps=None):
    """Masker BOUNDARY-focused crop offset (ablation): pick the patch's anchor voxel by REGION TYPE via
    the brain mask's distance transform, instead of any-random-brain-voxel — the review's recipe:
      50%  within a 5–15 mm BOUNDARY band (inside OR outside the surface) — the hard decision region;
      20%  deep INTERIOR (>15 mm in) — easy positives / pathology context;
      20%  extracranial HARD NEGATIVES (2–20 mm out) — the dura/skull/orbit/sinus shell that causes leaks;
      10%  uniform / FOV-edge (returns None -> caller's uniform fallback).
    EDT sampling uses ``voxel_sizes``, so these remain physical bands on anisotropic/native grids.
    Costs two EDTs per sample -> opt-in.
    Returns [d0,h0,w0] or None (uniform)."""
    if distance_maps is None:
        m, d_in, d_out = _boundary_distance_maps(padded_mask_chw, voxel_sizes=voxel_sizes)
    else:
        m, d_in, d_out = distance_maps
    if not m.any():
        return None
    r = float(rng.random())
    if r < 0.10:
        return None
    if r < 0.60:
        pool = ((d_in >= 5) & (d_in <= 15)) | ((d_out >= 5) & (d_out <= 15))
    elif r < 0.80:
        pool = d_in > 15
        if not pool.any():
            pool = d_in > 0
    else:
        pool = (d_out >= 2) & (d_out <= 20)
    idx = np.nonzero(pool)
    if len(idx[0]) == 0:
        return None
    j = int(rng.integers(0, len(idx[0])))
    v = [int(idx[0][j]), int(idx[1][j]), int(idx[2][j])]
    off = []
    for ax, p, dim in zip(v, size, dims):
        lo, hi = max(0, ax - p + 1), min(ax, dim - p)
        off.append(int(rng.integers(lo, hi + 1)) if hi > lo else lo)
    return off


def _random_crop(tensors, size: Tuple[int, int, int], rng, fg_mask=None, fg_frac: float = 0.0,
                 boundary_crop: bool = False, voxel_sizes=None, boundary_distance_maps=None) -> list:
    """Crop the SAME random patch-size region from every [C, D, H, W] tensor (shared spatial dims).

    Pads up to the patch size first (small volumes), then picks one random offset applied to all
    tensors — so a scan and its mask stay aligned.

    Foreground oversampling (masker only): a uniform-random offset over a ~256^3 head lands many
    patches on air/skull, whose empty (mask==0) label gives ~0 Tversky gradient and undersamples the
    brain boundary. When ``fg_mask`` (the co-cropped brain mask, [C, D, H, W]) is given and ``fg_frac``
    of the draw succeeds, constrain the offset so the patch is GUARANTEED to contain a random mask>0
    voxel (falling back to uniform on an empty mask). The remaining ``1-fg_frac`` stays uniform so the
    model still sees pure-background context. ``fg_frac<=0`` (the default) reproduces the pure-uniform
    crop exactly.
    """
    tensors = [_pad_chw(t, size) for t in tensors]
    _, d, h, w = tensors[0].shape
    off = None
    if boundary_crop and fg_mask is not None:
        off = _boundary_offset(_pad_chw(fg_mask, size), size, (d, h, w), rng,
                               voxel_sizes=voxel_sizes, distance_maps=boundary_distance_maps)
    elif fg_mask is not None and fg_frac > 0.0 and float(rng.random()) < fg_frac:
        m = _pad_chw(fg_mask, size)[0]  # pad identically so foreground coords match the padded tensors
        fg = torch.nonzero(m > 0.5, as_tuple=False)
        if fg.numel():
            v = fg[int(rng.integers(0, fg.shape[0]))].tolist()  # a random foreground (D, H, W) voxel
            off = []
            for ax, p, dim in zip(v, size, (d, h, w)):
                # offset range that keeps voxel `ax` inside [off, off+p) AND the patch inside [0, dim).
                lo, hi = max(0, ax - p + 1), min(ax, dim - p)
                off.append(int(rng.integers(lo, hi + 1)) if hi > lo else lo)
    if off is None:
        off = [int(rng.integers(0, d - size[0] + 1)),
               int(rng.integers(0, h - size[1] + 1)),
               int(rng.integers(0, w - size[2] + 1))]
    d0, h0, w0 = off
    return [t[:, d0:d0 + size[0], h0:h0 + size[1], w0:w0 + size[2]] for t in tensors]


def _getitem_skip_bad(records, bad: set, idx: int, load_fn, label: str):
    """Return ``load_fn(j)`` for the first LOADABLE record at/after ``idx`` (wrapping around),
    skipping only I/O failures (e.g. a truncated NIfTI or an unreadable mask). Logic and
    augmentation failures propagate. Evaluation datasets never call this replacement helper.

    A scan/mask that fails once is remembered in ``bad`` so it is never retried — training proceeds
    over the readable remainder instead of crashing on the offending file. Each failure is printed to
    stderr (NOT ``warnings.warn``: warnings are at the mercy of global filter state — ``-W`` /
    ``PYTHONWARNINGS`` / a library calling ``simplefilter`` — whereas the inherited stderr fd is
    always visible, including from spawn-started DataLoader workers). With ``num_workers>0`` each
    worker holds its own forked ``bad`` set, so the file is re-discovered (and re-skipped) at most
    once per worker — still no crash. Raises only if EVERY record is unreadable.
    """
    n = len(records)
    last_exc: Optional[Exception] = None
    for attempt in range(n):
        j = (idx + attempt) % n
        if j in bad:
            continue
        try:
            return load_fn(j)
        except (OSError, EOFError, nib.filebasedimages.ImageFileError) as e:
            bad.add(j)
            last_exc = e
            r = records[j]
            print(
                f"[{label}] skipping unreadable sample {j} "
                f"(scan={r.get('scan')!r}, mask={r.get('mask')!r}): {type(e).__name__}: {e}",
                file=sys.stderr, flush=True,
            )
    # EVERY record failed. That is almost never genuine per-file corruption across the whole set —
    # it usually means a systematic fault (a code bug in the load path, a wrong manifest, a missing
    # dependency). Re-raise CHAINED from the last failure so the real root-cause traceback survives
    # instead of being swallowed by the skip loop.
    raise RuntimeError(
        f"[{label}] every sample failed to load ({n} records, all unreadable) — "
        f"this is a systematic fault, not per-file corruption; see the chained exception below"
    ) from last_exc


class NiftiManifestDataset(Dataset):
    """Load scan/mask pairs from an offline manifest.

    Whole-volume mode resizes or conforms each pair to ``target_shape``. Patch
    mode crops during training and returns full volumes for sliding-window evaluation.
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        target_shape: Tuple[int, int, int],
        patch_size: Optional[Tuple[int, int, int]] = None,
        phase: str = "train",
        seed: int = 0,
        conform_mm: Optional[float] = None,
        fg_crop_frac: float = 0.0,
        sdt_supervision: bool = False,
        sdt_band_mm: float = 5.0,
        sdt_far_weight: float = 0.1,
    ):
        self.records = records
        self.shape = target_shape
        self.patch_size = tuple(patch_size) if patch_size else None
        self.phase = phase
        self.fg_crop_frac = float(fg_crop_frac)
        self.conform_mm = float(conform_mm) if conform_mm is not None else None
        self.sdt_supervision = bool(sdt_supervision)
        self.sdt_band_mm = float(sdt_band_mm)
        self.sdt_far_weight = float(sdt_far_weight)
        # Workers reseed this advancing stream so crops differ across workers and epochs.
        self.rng = np.random.default_rng(seed)
        self._bad: set = set()

    def __len__(self) -> int:
        return len(self.records)

    def _load_wv(self, path: str, is_mask: bool) -> torch.Tensor:
        """Conform or resize; nearest-neighbor interpolation keeps masks binary."""
        if self.conform_mm is not None:
            return _load_conformed(path, self.shape, self.conform_mm, order=0 if is_mask else 3)
        return _load_resized(path, self.shape, "nearest" if is_mask else "trilinear")

    def __getitem__(self, idx: int):
        if self.phase != "train":
            return self._load(idx)  # held-out membership must never change on a read failure
        return _getitem_skip_bad(self.records, self._bad, idx, self._load, "NiftiManifestDataset")

    def _mask_target(self, mask: torch.Tensor, voxel_sizes):
        """Return the mask tensor, or a full-volume physical SDT supervision bundle."""
        if not self.sdt_supervision:
            return mask
        raw = signed_distance_transform_numpy(
            mask[0].numpy(), voxel_sizes=voxel_sizes, inside_positive=True)
        sdt = torch.from_numpy(np.clip(raw, -self.sdt_band_mm, self.sdt_band_mm))[None].float()
        weight = torch.from_numpy(np.where(
            np.abs(raw) >= self.sdt_band_mm, self.sdt_far_weight, 1.0).astype(np.float32))[None]
        return {"mask": mask, "sdt": sdt, "sdt_weight": weight}

    @staticmethod
    def _native_voxel_sizes(path: str) -> Tuple[float, float, float]:
        return tuple(float(x) for x in nib.affines.voxel_sizes(nifti_affine_mm(nib.load(str(path)))))

    def _load(self, idx: int):
        r = self.records[idx]
        if self.patch_size is None:
            scan = self._load_wv(r["scan"], is_mask=False)  # already z-scored on disk
            mask = (self._load_wv(r["mask"], is_mask=True) > 0.5).float()
            if self.conform_mm is not None:
                spacing = (self.conform_mm,) * 3
            else:
                img = nib.load(str(r["mask"]))
                zooms = np.asarray(nib.affines.voxel_sizes(nifti_affine_mm(img)), dtype=np.float64)
                spacing = tuple((zooms * np.asarray(img.shape[:3], dtype=np.float64)
                                 / np.asarray(self.shape, dtype=np.float64)).tolist())
            return scan, self._mask_target(mask, spacing)

        # Conform the full scan before cropping; without conform_mm retain its native grid.
        scan = (self._load_wv(r["scan"], is_mask=False)
                if self.conform_mm is not None else _load_full(r["scan"]))
        mask = ((self._load_wv(r["mask"], is_mask=True)
                 if self.conform_mm is not None else _load_full(r["mask"])) > 0.5).float()
        spacing = ((self.conform_mm,) * 3 if self.conform_mm is not None
                   else self._native_voxel_sizes(r["mask"]))
        packed = self._mask_target(mask, spacing)
        if self.phase == "train":
            tensors = ([scan, mask, packed["sdt"], packed["sdt_weight"]]
                       if isinstance(packed, dict) else [scan, mask])
            cropped = _random_crop(tensors, self.patch_size, self.rng,
                                   fg_mask=mask, fg_frac=self.fg_crop_frac)
            if isinstance(packed, dict):
                return cropped[0], {"mask": cropped[1], "sdt": cropped[2],
                                    "sdt_weight": cropped[3]}
            return cropped[0], cropped[1]
        return scan, packed  # eval: full volume -> sliding-window inference (bs=1)


class OnlineSynthDataset(Dataset):
    """MASKING dataset that SYNTHESIZES on the fly from RAW (scan, mask) pairs — no pre-generated
    dataset on disk.

    Each `__getitem__` loads a raw full-head scan + brain mask, robust-normalizes, optionally conforms
    both to a canonical physical grid, and then applies `make_training_sample` with a fresh random draw,
    so every
    epoch sees a brand-new random contrast/pose — effectively infinite variety (the SynthSeg/SynthStrip
    recipe), and far more sample-efficient than a frozen on-disk set. The EVAL phase synthesizes
    NOTHING (uses the real scan) so val/test is a stable, real-contrast measurement.

    Output matches `NiftiManifestDataset` (masking): whole-volume -> resized/conformed tensors;
    patch mode -> a random co-cropped patch (train) or the full native/canonical volume (eval, for sliding
    window). With `num_workers>0` the per-sample synthesis cost is overlapped across workers and
    hidden behind GPU compute — so use workers when training this online.
    """

    def __init__(self, records, target_shape, patch_size=None, phase="train", seed=0,
                 synth_on=True, synth_kwargs=None, fg_crop_frac=0.0, masker_norm_aug=False,
                 boundary_crop=False, conform_mm=None, crops_per_volume=1, volume_repeats=1,
                 sdt_supervision=False, sdt_band_mm=5.0, sdt_far_weight=0.1,
                 eval_category=None, eval_variants=1):
        self.records = records
        self.shape = tuple(target_shape)
        self.patch_size = tuple(patch_size) if patch_size else None
        # Isotropic physical grid used BEFORE online synthesis. In patch mode the full canonical volume
        # is synthesized first and then cropped, so a patch has the same millimetre extent on each scan.
        self.conform_mm = float(conform_mm) if conform_mm is not None else None
        self.phase = phase
        # Masker patch train: fraction of crops forced to contain brain (mask>0); 0 == pure-uniform.
        self.fg_crop_frac = float(fg_crop_frac)
        # Ablation flag: sample the z-score NORM region per-sample (foreground/rough-mask) to match the
        # deployed passes instead of always the GT mask. OFF (default) == GT-mask-only (current behavior).
        self.masker_norm_aug = bool(masker_norm_aug) and phase == "train"
        # Ablation flag: region-typed (boundary/interior/hard-neg/edge) crop sampling. OFF == fg_frac crop.
        self.boundary_crop = bool(boundary_crop) and phase == "train"
        self.synth_on = bool(synth_on) and phase == "train"
        self.synth_kwargs = dict(synth_kwargs or {})
        self.crops_per_volume = int(crops_per_volume) if phase == "train" else 1
        self.volume_repeats = int(volume_repeats) if phase == "train" else 1
        self.sdt_supervision = bool(sdt_supervision)
        self.sdt_band_mm = float(sdt_band_mm)
        self.sdt_far_weight = float(sdt_far_weight)
        if self.crops_per_volume < 1:
            raise ValueError("crops_per_volume must be >= 1")
        if self.volume_repeats < 1:
            raise ValueError("volume_repeats must be >= 1")
        if not np.isfinite(self.sdt_band_mm) or self.sdt_band_mm <= 0.0:
            raise ValueError("sdt_band_mm must be finite and > 0")
        if not 0.0 <= self.sdt_far_weight <= 1.0:
            raise ValueError("sdt_far_weight must be in [0, 1]")
        # deterministic RNG only for the (eval-irrelevant) patch crop; the synth contrast uses fresh
        # OS entropy per call so it is NOT frozen per epoch (the known worker-fork reproducibility trap).
        self.rng = np.random.default_rng(seed)
        self.seed = int(seed)
        # Indices whose scan/mask failed to load — skipped on future access (see `_getitem_skip_bad`).
        self._bad: set = set()
        # VALIDATION-CATEGORY mode (the per-augmentation Dice breakdown). Eval normally synthesizes
        # nothing; with a category set it instead expands each val subject into `eval_variants` FROZEN
        # samples of that one forced category. Frozen matters: the seed derives only from
        # (seed, stable acquisition identity, variant), so the set is bit-identical across epochs, sessions and
        # resumes — a val set that moved would report resampling noise as model change.
        self.eval_category = str(eval_category) if eval_category else None
        self.eval_variants = max(1, int(eval_variants))
        if self.eval_category is not None:
            from augmentations.pipeline import MASKER_VAL_CATEGORIES
            if phase == "train":
                raise ValueError("eval_category is a VALIDATION-only setting (phase must not be 'train')")
            if self.eval_category not in MASKER_VAL_CATEGORIES:
                raise ValueError(f"unknown eval_category {self.eval_category!r}; expected one of "
                                 f"{MASKER_VAL_CATEGORIES}")
            self._eval_key = [(ri, vi) for ri in range(len(self.records))
                              for vi in range(self.eval_variants)]
            # Mirror the underlying record per expanded index so `_getitem_skip_bad`'s report works.
            self._items = [self.records[ri] for ri, _ in self._eval_key]
            # How many draws did not honour the requested category (guard fallback / no artifact
            # applied) even after redrawing — reported once by the trainer so an unfaithful group is
            # visible rather than quietly averaged into its Dice.
            self.unfaithful = 0
            self._unfaithful_warned = False
        else:
            self._eval_key = None
            self._items = self.records

    def __len__(self) -> int:
        if self._eval_key is not None:
            return len(self._eval_key)
        return len(self.records) * self.volume_repeats

    def __getitem__(self, idx: int):
        # Skip records whose scan/mask cannot be opened rather than crashing the run.
        if self.phase != "train":
            return self._load(idx)  # fail evaluation rather than duplicate another held-out case
        return _getitem_skip_bad(self._items, self._bad, idx, self._load, "OnlineSynthDataset")

    def _load(self, idx: int):
        from imaging.normalization import normalize_intensity, zscore
        from augmentations.pipeline import make_training_sample, sample_norm_region

        rec_idx, variant = self._eval_key[idx] if self._eval_key is not None else (idx, 0)
        r = self.records[rec_idx]
        _img = nib.load(str(r["scan"]))
        # Request float32 directly from nibabel instead of first materializing a float64 volume and
        # immediately downcasting later. Online synthesis is float32 throughout, so float64 only
        # doubled worker RAM and memory bandwidth without adding useful precision.
        vol = np.asarray(_img.get_fdata(dtype=np.float32), dtype=np.float32).reshape(_img.shape[:3])
        affine = nifti_affine_mm(_img)                       # source grid -> conform resamples through it
        native_voxel_sizes = tuple(float(x) for x in nib.affines.voxel_sizes(affine))
        _mask_img = nib.load(str(r["mask"]))
        mask_affine = nifti_affine_mm(_mask_img)
        from imaging.geometry import assert_scan_mask_grid
        assert_scan_mask_grid(affine, vol.shape, mask_affine, _mask_img.shape[:3],
                              context="online sample", require_full_affine=True)
        mask = np.asarray(_mask_img.get_fdata(dtype=np.float32)).reshape(_mask_img.shape[:3]) > 0.5
        scan01 = normalize_intensity(vol)

        # Canonicalize BEFORE morphology/artifacts/resolution. Previously conform happened after
        # synthesis, so augmentation parameters expressed in voxels represented different physical
        # sizes on every scanner. Once conformed, one voxel is exactly conform_mm throughout synthesis.
        if self.conform_mm is not None:
            scan01 = np.clip(
                _conform_arrays(scan01, affine, self.shape, self.conform_mm, order=3).numpy()[0],
                0.0, 1.0,
            ).astype(np.float32)
            mask = _drop_resampling_specks(
                _conform_arrays(mask.astype(np.float32), affine, self.shape,
                                self.conform_mm, order=0).numpy()[0] > 0.5)
        effective_voxel_sizes = ((self.conform_mm,) * 3 if self.conform_mm is not None
                                 else native_voxel_sizes)
        if self.eval_category is not None:
            # FROZEN per-category validation sample. Deterministic seed ONLY (no OS entropy, and not
            # `self.rng`, which advances per __getitem__ and would differ between a fresh run and a
            # resume). `self.synth_kwargs` is deliberately NOT forwarded: the category pins the tier
            # probabilities itself, and a run-level `benign_only_mp2rage_fraction` would override it.
            from augmentations.pipeline import make_eval_sample, _category_is_faithful
            # The category is deliberately NOT part of the seed. `benign` and `nonbenign` then consume
            # an IDENTICAL rng stream up to the artifact gate (same tier draw, same acquisition, same
            # morph/pose/resolution — the gate itself consumes one draw either way and only fires for
            # `nonbenign`), so the two categories are the SAME scan with and without artifacts and
            # their Dice gap is the cost of the artifacts alone. `mp2rage` takes a different branch
            # and diverges regardless; it was never meant to be paired.
            vrng = _case_rng(self.seed, r, variant)
            scan01, mask, kind = make_eval_sample(
                scan01, mask, vrng, self.eval_category, voxel_sizes=effective_voxel_sizes)
            if not _category_is_faithful(self.eval_category, kind):
                self.unfaithful += 1
                if not self._unfaithful_warned:
                    # WARN, don't hide: a sample that fell back to the pristine source (or a
                    # `nonbenign` slot that got no artifact) pulls its category's Dice toward the
                    # clean number, which reads as good news. Once per worker process (each holds a
                    # forked copy of this flag), matching the `_getitem_skip_bad` idiom.
                    self._unfaithful_warned = True
                    print(f"[online-masker val-aug] category {self.eval_category!r}: a draw did not "
                          f"honour the category after redraws (kind={kind!r}) — that sample is NOT "
                          f"representative of the group; dice_{self.eval_category} is diluted toward "
                          f"the clean value. Further occurrences in this worker are not printed.",
                          file=sys.stderr, flush=True)
        elif self.synth_on:
            # MIX label-driven coverage synthesis + realistic-texture transfer (fresh per call/epoch).
            synth_kwargs = dict(self.synth_kwargs)
            synth_kwargs.setdefault("voxel_sizes", effective_voxel_sizes)
            scan01, mask = make_training_sample(scan01, mask, np.random.default_rng(), **synth_kwargs)
        sdt = None
        sdt_weight = None
        if self.sdt_supervision:
            # Compute the exact full-volume physical SDT in a loader worker BEFORE cropping. Computing
            # EDT on each patch would invent a boundary wherever the crop cuts through the brain.
            # Shared model/loss helpers are defined above in this file.
            raw_sdt = signed_distance_transform_numpy(
                mask, voxel_sizes=effective_voxel_sizes, inside_positive=True)
            sdt_weight = np.where(
                np.abs(raw_sdt) >= self.sdt_band_mm, self.sdt_far_weight, 1.0).astype(np.float32)
            sdt = np.clip(raw_sdt, -self.sdt_band_mm, self.sdt_band_mm).astype(np.float32)
        # Norm region: GT mask (default), or per-sample foreground/rough-mask to match the deployed passes.
        norm_region = (sample_norm_region(mask, self.rng, voxel_sizes=effective_voxel_sizes)
                       if self.masker_norm_aug else mask)
        z = zscore(scan01, norm_region)

        if self.patch_size is None:  # whole-volume (matches NiftiManifestDataset masking)
            if self.conform_mm is not None:
                # Already on target_shape @ conform_mm: avoid a second interpolation.
                scan_t = torch.from_numpy(np.ascontiguousarray(z))[None].float()
                mask_t = torch.from_numpy(np.ascontiguousarray(mask.astype(np.float32)))[None]
                sdt_t = (torch.from_numpy(np.ascontiguousarray(sdt))[None]
                         if sdt is not None else None)
                sdt_weight_t = (torch.from_numpy(np.ascontiguousarray(sdt_weight))[None]
                                if sdt_weight is not None else None)
            else:
                scan_t = _resize_arr(z, self.shape, "trilinear")
                mask_t = _resize_arr(mask.astype(np.float32), self.shape, "nearest")
                sdt_t = (_resize_arr(sdt, self.shape, "trilinear")
                         if sdt is not None else None)
                sdt_weight_t = (_resize_arr(sdt_weight, self.shape, "nearest")
                                if sdt_weight is not None else None)
            mask_t = (mask_t > 0.5).float()
            return ((scan_t, {"mask": mask_t, "sdt": sdt_t, "sdt_weight": sdt_weight_t})
                    if sdt_t is not None else (scan_t, mask_t))

        scan_t = torch.from_numpy(np.ascontiguousarray(z))[None]                 # [1,D,H,W]
        mask_t = (torch.from_numpy(np.ascontiguousarray(mask.astype(np.float32)))[None] > 0.5).float()
        sdt_t = (torch.from_numpy(np.ascontiguousarray(sdt))[None]
                 if sdt is not None else None)
        sdt_weight_t = (torch.from_numpy(np.ascontiguousarray(sdt_weight))[None]
                        if sdt_weight is not None else None)
        if self.phase == "train":
            # Load/conform/synthesize/z-score ONCE, then take K independently located co-crops. This
            # amortizes the expensive full-volume CPU pipeline and presents B*K patches in one GPU
            # forward. K=1 follows the historical return path and shape exactly.
            crop_tensors = ([scan_t, mask_t, sdt_t, sdt_weight_t]
                            if sdt_t is not None else [scan_t, mask_t])
            boundary_maps = None
            if self.boundary_crop and self.crops_per_volume > 1:
                # EDT is the expensive part of boundary sampling. Reuse one pair of full-volume
                # distance maps for K independent region draws instead of recomputing 2*K EDTs.
                boundary_maps = _boundary_distance_maps(
                    _pad_chw(mask_t, self.patch_size), voxel_sizes=effective_voxel_sizes)
            crops = [
                _random_crop(
                    crop_tensors, self.patch_size, self.rng,
                    fg_mask=mask_t, fg_frac=self.fg_crop_frac,
                    boundary_crop=self.boundary_crop, voxel_sizes=effective_voxel_sizes,
                    boundary_distance_maps=boundary_maps,
                )
                for _ in range(self.crops_per_volume)
            ]
            if self.crops_per_volume == 1:
                target = ({"mask": crops[0][1], "sdt": crops[0][2],
                           "sdt_weight": crops[0][3]}
                          if sdt_t is not None else crops[0][1])
                return crops[0][0], target
            target = ({"mask": torch.stack([c[1] for c in crops], dim=0),
                       "sdt": torch.stack([c[2] for c in crops], dim=0),
                       "sdt_weight": torch.stack([c[3] for c in crops], dim=0)}
                      if sdt_t is not None else torch.stack([c[1] for c in crops], dim=0))
            return torch.stack([c[0] for c in crops], dim=0), target
        target = ({"mask": mask_t, "sdt": sdt_t, "sdt_weight": sdt_weight_t}
                  if sdt_t is not None else mask_t)
        return scan_t, target  # eval: full volume -> sliding-window inference (bs=1)


def _preflight(settings, records: List[Dict[str, Any]], label: str) -> List[Dict[str, Any]]:
    """Validate every record's files BEFORE the split/training and return the readable subset.

    Reads only NIfTI HEADERS (``nib.load`` is lazy — ``.shape`` does not touch the voxel data), so
    this is cheap even over a large manifest. It catches the two faults that otherwise present as a
    confusing mid-epoch skip or an all-records crash:
      * a missing / unreadable / header-corrupt scan or mask, and
      * a scan/mask NATIVE-shape mismatch — but only when the loaders use the two TOGETHER before
        resampling (online synthesis or patch mode); whole-volume resize tolerates a mismatch
        because it resizes each to ``target_shape`` independently.

    Doing this up front (vs the per-sample ``__getitem__`` skip) turns a silent, training-biasing
    skip into ONE clear report, keeps the eval/test denominator honest (bad records are dropped
    before the split, so Dice is computed over the intended held-out cases), and
    ABORTS on a systematic fault instead of training on a lopsided remainder. The per-sample skip
    stays as belt-and-suspenders for truncated-gzip errors a header read can't detect.
    """
    # Empty input has no per-record error to report; guard explicitly so the "all unreadable"
    # branch below (which reads bad[0]) never trips an opaque IndexError, and the fail-loud intent
    # is preserved with a clear message.
    if not records:
        raise RuntimeError(
            f"[preflight:{label}] no records to train on (0 discovered) — check --data-dir / "
            f"manifest.jsonl (offline) or --scan-glob / --mask-suffix (--synth-online).")
    needs_shape_match = settings.synth_online or settings.patch_size is not None
    from imaging.geometry import assert_scan_mask_grid
    good: List[Dict[str, Any]] = []
    bad: List[Tuple[Dict[str, Any], Exception]] = []
    for r in records:
        try:
            simg = nib.load(str(r["scan"]))
            saffine = nifti_affine_mm(simg)
            sshape = tuple(simg.shape[:3])
            mpath = r.get("mask")
            if mpath:
                mimg = nib.load(str(mpath))
                # Shape (full, not [:3], so a 4D scan vs 3D mask is caught) + orientation + spacing;
                # and the full affine under conform (which resamples THROUGH it). The pipeline indexes
                # scan+mask together / resizes them independently before any resampling, so an affine
                # mismatch silently misaligns. NFBS pairs are bit-identical -> a no-op on the supported set.
                assert_scan_mask_grid(saffine, sshape, nifti_affine_mm(mimg), tuple(mimg.shape[:3]),
                                      context=f"preflight:{label}",
                                      check_shape=needs_shape_match,
                                      require_full_affine=True)
            good.append(r)
        except Exception as e:  # noqa: BLE001 — collect, report, and drop; don't crash here
            bad.append((r, e))

    if bad:
        print(f"[preflight:{label}] {len(bad)} of {len(records)} records unreadable/mismatched "
              f"— dropping them:", file=sys.stderr, flush=True)
        for r, e in bad[:20]:
            print(f"    scan={r.get('scan')!r} mask={r.get('mask')!r}: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
        if len(bad) > 20:
            print(f"    ... and {len(bad) - 20} more", file=sys.stderr, flush=True)

    if not good:
        first = bad[0][1]
        raise RuntimeError(
            f"[preflight:{label}] ALL {len(records)} records are unreadable/mismatched — aborting "
            f"(this is a systematic fault, not per-file corruption). First error: "
            f"{type(first).__name__}: {first}")
    if len(bad) > len(records) * 0.5:
        first = bad[0][1]
        raise RuntimeError(
            f"[preflight:{label}] {len(bad)}/{len(records)} (>50%) records unreadable/mismatched — "
            f"aborting rather than training on a biased remainder. This usually means a wrong "
            f"--data-dir / --mask-suffix or a missing dependency. First error: "
            f"{type(first).__name__}: {first}")
    return good


def _smoke_probe(settings, dataset) -> None:
    """Run ONE real full load (the unwrapped ``_load``) to surface a SYSTEMATIC fault the
    header-only ``_preflight`` cannot see — most importantly a missing SYNTHESIS dependency
    (e.g. ``cv2``/OpenCV, imported transitively via the ``augmentation`` package the online
    synth path pulls in), but also any data-level error in the real load+synthesis. Such a
    fault fails for EVERY record identically, so catching it on one sample here — before the
    model build and epoch loop — turns an all-records skip storm into one clear, early abort.
    Chained from the real exception so the root-cause traceback (e.g. ``ModuleNotFoundError:
    cv2``) is preserved.
    """
    if len(dataset) == 0:
        return
    try:
        dataset._load(0)
    except Exception as e:  # noqa: BLE001 — convert to a loud, labelled, fail-fast abort
        raise RuntimeError(
            "smoke-load of the first training sample failed — this is a SYSTEMATIC fault, not "
            "per-file corruption, so it would fail for every record. Usual causes: a missing "
            "dependency (e.g. cv2/OpenCV in the augmentation import chain); a STALE / out-of-sync "
            "module on this machine (an `ImportError: cannot import name ...` means the deployed "
            "copy of a sibling file is incompatible with training/data.py — re-sync the whole deployment_condensed/ "
            "folder); or a data error. Fix this before training. Underlying error: "
            f"{type(e).__name__}: {e}"
        ) from e


def _records(settings, data_dir=None) -> List[Dict[str, Any]]:
    manifest = Path(data_dir or settings.data_dir) / "manifest.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(f"{manifest} not found — run augment_directory.py first.")
    recs = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    recs = [r for r in recs if r.get("mask")]
    return recs


def _subject_id(settings, record: Dict[str, Any]) -> str:
    explicit = record.get("subject_id")
    source = str(record.get("source_scan") or record["scan"])
    name = Path(source).name
    if explicit is not None:
        if not isinstance(explicit, str) or not explicit.strip():
            raise ValueError("subject_id must be a nonempty string")
        subject = explicit.strip()
    elif settings._subject_pattern is not None:
        match = settings._subject_pattern.search(source.replace("\\", "/"))
        if match is None:
            raise ValueError(f"subject_id_regex does not match {source!r}")
        subject = (match.group("subject_id") if "subject_id" in match.groupdict()
                   else match.group(1) if match.lastindex else match.group(0))
        if not subject:
            raise ValueError(f"subject_id_regex extracted an empty ID for {source!r}")
    else:
        # Prefer the filename's BIDS entity; directory names can identify a session or copy.
        match = re.search(r"(?:^|_)sub-([A-Za-z0-9]+)(?=[_.]|$)", name, re.IGNORECASE)
        if match is None:
            match = re.match(r"(A[0-9]{8})(?=[_.-]|$)", name, re.IGNORECASE)
        if match is None:
            raise ValueError(
                f"Cannot determine subject identity for {source!r}. Use BIDS sub-ID/NFBS "
                "filenames, set subject_id_regex, or provide subject_id in each manifest record.")
        subject = match.group(1)
    subject = str(subject).strip().casefold()
    if subject.startswith("sub-"):
        subject = subject[4:]
    if not subject:
        raise ValueError(f"Empty subject identity for {source!r}")
    return subject


def _prepare_subject_records(settings, records, label: str):
    """Assign stable identities and collapse only verified byte-identical acquisition copies."""
    unique = {}
    digests = {}

    def digest(path):
        resolved = str(Path(path).resolve())
        if resolved not in digests:
            h = hashlib.sha256()
            with Path(path).open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    h.update(chunk)
            digests[resolved] = h.digest()
        return digests[resolved]

    for record in records:
        r = dict(record)
        r["subject_id"] = _subject_id(settings, r)
        r["case_id"] = _case_identity(r)
        key = (r["subject_id"], r["case_id"])
        if key in unique:
            previous = unique[key]
            same = all(bool(r.get(k)) == bool(previous.get(k)) and (
                not r.get(k) or digest(r[k]) == digest(previous[k])) for k in ("scan", "mask"))
            if not same:
                raise ValueError(
                    f"Ambiguous {label} acquisition {key!r}: copied names have different scan/mask "
                    "contents. Give distinct acquisitions unique names or explicit case_id values.")
            # Choose deterministically between identical physical copies, regardless of discovery order.
            if str(r["scan"]) < str(previous["scan"]):
                unique[key] = r
            continue
        unique[key] = r
    return [unique[k] for k in sorted(unique)]


def _split_fingerprint(**splits) -> str:
    """Digest split membership and file contents so changed data cannot silently resume."""
    payload = {}
    file_cache = {}
    for split_name, records in splits.items():
        entries = []
        for record in records:
            item = {"record": record, "files": {}}
            for key in ("scan", "mask", "source_scan"):
                value = record.get(key) if isinstance(record, dict) else None
                if not value:
                    continue
                p = Path(str(value))
                try:
                    st = p.stat() if p.is_file() else None
                except OSError:
                    st = None
                resolved = str(p.resolve())
                content_hash = None
                if st is not None:
                    if resolved not in file_cache:
                        digest = hashlib.sha256()
                        with p.open("rb") as stream:
                            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                                digest.update(chunk)
                        after = p.stat()
                        if (st.st_size, st.st_mtime_ns, getattr(st, "st_ctime_ns", None)) != (
                                after.st_size, after.st_mtime_ns,
                                getattr(after, "st_ctime_ns", None)):
                            raise RuntimeError(
                                f"training input changed while it was fingerprinted: {p}"
                            )
                        file_cache[resolved] = digest.hexdigest()
                    content_hash = file_cache[resolved]
                item["files"][key] = {
                    "path": resolved,
                    "size": st.st_size if st else None,
                    "sha256": content_hash,
                }
            entries.append(item)
        # Normalize record order for a content-and-membership fingerprint.
        payload[split_name] = sorted(
            entries, key=lambda x: json.dumps(x, sort_keys=True, default=str, separators=(",", ":")))
    raw = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def prepare_data(settings, selection_metric: str) -> PreparedData:
    epoch_accounting = None
    if settings.synth_online:
        from augmentations.pipeline import discover_pairs
        recs = discover_pairs(settings.data_dir, settings.scan_glob, settings.mask_suffix)
        if not recs:
            raise FileNotFoundError(
                f"--synth-online: no (scan, mask) pairs under {settings.data_dir} matching "
                f"{settings.scan_glob!r} with mask suffix {settings.mask_suffix!r}.")
    else:
        recs = _records(settings)
    # Drop unreadable / shape-mismatched records ONCE, up front (clear report + honest eval N +
    # abort-on-systematic-fault) instead of discovering them as silent per-sample skips mid-epoch.
    recs = _prepare_subject_records(settings, _preflight(settings, recs, "train"), "train")
    rng = np.random.default_rng(settings.seed)

    # An external held-out test set (e.g. real, un-augmented scans via --test-dir) takes
    # precedence; in that case we DON'T also carve a test split from the main manifest. The
    # external test is ALWAYS a real manifest (never synthesized) so it measures real performance.
    external_test = _records(settings, settings.test_dir) if settings.test_dir is not None else None
    if external_test is not None:
        external_test = _prepare_subject_records(settings, _preflight(settings, external_test, "test"), "test")
        overlap = {r["subject_id"] for r in recs} & {r["subject_id"] for r in external_test}
        if overlap:
            raise ValueError(f"External test subjects overlap training/validation subjects: {sorted(overlap)}")
    carve_test = settings.test_fraction if external_test is None else 0.0

    # Every acquisition/variant of one subject stays on one side of the split.
    groups: Dict[str, List[int]] = {}
    for i, r in enumerate(recs):
        key = r["subject_id"]
        groups.setdefault(key, []).append(i)
    keys = sorted(groups)

    test_idx: List[int] = []
    if len(keys) >= 2 and len(recs) > 1:
        perm = [keys[j] for j in rng.permutation(len(keys))]
        n = len(perm)
        n_test = min(int(round(carve_test * n)), max(0, n - 2)) if carve_test > 0 else 0
        n_val = max(1, int(round(settings.val_fraction * n)))
        n_val = min(n_val, max(1, n - n_test - 1))  # always leave >=1 train subject
        test_keys = set(perm[:n_test])
        val_keys = set(perm[n_test:n_test + n_val])
        test_idx = [i for k in sorted(test_keys) for i in groups[k]]
        val_idx = [i for k in sorted(val_keys) for i in groups[k]]
        train_idx = [i for k in keys if k not in test_keys and k not in val_keys for i in groups[k]]
    else:
        raise ValueError("At least two distinct subjects are required for disjoint training/validation splits")

    train_recs = [recs[i] for i in train_idx]
    val_recs = [recs[i] for i in val_idx]
    test_recs = external_test if external_test is not None else [recs[i] for i in test_idx]
    fingerprint = _split_fingerprint(
        train=train_recs, validation=(val_recs or train_recs), test=test_recs)

    patch = settings.patch_size is not None
    # Full-res volumes have varying native shapes -> can't be collated; eval runs one volume at a
    # time through sliding-window / patch-grid inference, so the eval loaders use batch_size=1.
    eval_bs = 1 if patch else settings.batch_size
    train_bs = settings.batch_size

    def make(records, shuffle, phase, bs, online, eval_category=None):
        if online:
            # Masking online: the PRIMARY eval phase synthesizes NOTHING (real scan) -> stable,
            # real-contrast metric. `eval_category` builds the diagnostic loaders instead, which
            # DO synthesize at eval — a frozen set of one forced category (benign/mp2rage/
            # nonbenign) so each group's Dice has a guaranteed, repeatable N.
            ds = OnlineSynthDataset(
                records, settings.target_shape, patch_size=settings.patch_size, phase=phase,
                seed=settings.seed, synth_on=(phase == "train"), synth_kwargs=settings.synth_kwargs,
                fg_crop_frac=settings.fg_crop_frac, masker_norm_aug=settings.masker_norm_aug,
                boundary_crop=settings.boundary_crop, conform_mm=settings.conform_mm,
                crops_per_volume=settings.masker_crops_per_volume,
                volume_repeats=settings.masker_volume_repeats,
                sdt_supervision=settings.sdt_supervision,
                sdt_band_mm=float(settings.loss_kwargs.get("sdt_band_mm", 5.0)),
                sdt_far_weight=float(settings.loss_kwargs.get("sdt_far_weight", 0.1)),
                eval_category=eval_category,
                eval_variants=settings.masker_val_aug_variants,
            )
        else:
            ds = NiftiManifestDataset(
                records, settings.target_shape,
                patch_size=settings.patch_size, phase=phase, seed=settings.seed, conform_mm=settings.conform_mm,
                fg_crop_frac=settings.fg_crop_frac,
                sdt_supervision=settings.sdt_supervision,
                sdt_band_mm=float(settings.loss_kwargs.get("sdt_band_mm", 5.0)),
                sdt_far_weight=float(settings.loss_kwargs.get("sdt_far_weight", 0.1)),
            )
        # With workers, keep them ALIVE across epochs (no per-epoch respawn + module re-import) and
        # prefetch deeper so the (CPU-bound) online synthesis stays ahead of the GPU — this pipeline
        # is data-bound, so a starved GPU (low util / low power) is fixed by feeding it faster, not a
        # bigger card. No effect when num_workers=0 (persistent_workers requires workers>0).
        dl_kw = {}
        if settings.num_workers > 0:
            # The per-category diagnostic loaders are consumed once every `masker_val_aug_every`
            # epochs, so KEEPING their workers alive is pure waste: three extra persistent pools
            # (33 processes at num_workers=11) would idle for the whole run holding gigabytes of
            # prefetched 224^3 volumes in shared memory. Non-persistent workers spawn for the pass
            # and exit, and a shallower queue bounds the transient footprint.
            dl_kw["persistent_workers"] = eval_category is None
            # A multicrop item is K times larger in shared memory. Scale queue depth down so K=4
            # with several workers does not approach ~1 GiB of prefetched scan/mask tensors.
            k_prefetch = (settings.masker_crops_per_volume
                          if (online and phase == "train") else 1)
            dl_kw["prefetch_factor"] = (2 if eval_category is not None
                                        else max(1, 4 // k_prefetch))
        if (online and phase == "train"
                and settings.masker_crops_per_volume > 1):
            dl_kw["collate_fn"] = _collate_masker_crops
        return DataLoader(
            ds, batch_size=bs, shuffle=shuffle,
            num_workers=settings.num_workers, pin_memory=(settings.device.type == "cuda"), drop_last=False,
            worker_init_fn=_worker_init_fn, **dl_kw,
        )

    # --test-dir test set is a real manifest (online=False); a carved test from raw pairs follows
    # the train source (online iff synth_online), evaluated with synthesis OFF (real contrast).
    test_online = settings.synth_online and external_test is None
    test_dl = make(test_recs, False, "eval", eval_bs, test_online) if test_recs else None
    train_dl = make(train_recs, True, "train", train_bs, settings.synth_online)
    if settings.synth_online:
        syntheses = len(train_recs) * settings.masker_volume_repeats
        patches = syntheses * (settings.masker_crops_per_volume if patch else 1)
        batches = len(train_dl)
        steps = (batches + settings.accum_steps - 1) // settings.accum_steps
        epoch_accounting = {
            "source_subjects": len({r["subject_id"] for r in train_recs}),
            "source_volumes": len(train_recs),
            "syntheses": syntheses,
            "patches_or_volumes": patches,
            "loader_batches": batches,
            "optimizer_steps": steps,
        }
        print(
            "[online-masker epoch] "
            f"{len({r['subject_id'] for r in train_recs})} subjects; "
            f"{len(train_recs)} source volumes x repeats {settings.masker_volume_repeats} = "
            f"{syntheses} syntheses; x crops {settings.masker_crops_per_volume if patch else 1} = "
            f"{patches} {'patches' if patch else 'volumes'}; {batches} loader batches; "
            f"~{steps} optimizer steps.", flush=True)
    # Fail-fast: one real full load exercises the synthesis + augmentation import chain (e.g. cv2),
    # which the header-only preflight can't reach — so a systematic load fault aborts here with a
    # clear message, before the model build + epoch loop, instead of as a per-batch skip storm.
    _smoke_probe(settings, train_dl.dataset)
    val_dl = make(val_recs or train_recs, False, "eval", eval_bs, settings.synth_online)
    # Diagnostic per-category loaders over the SAME val subjects, so `dice_benign` /
    # `dice_mp2rage` / `dice_nonbenign` and the primary real-scan `dice` are directly comparable.
    val_aug_loaders = {}
    if settings.masker_val_aug:
        from augmentations.pipeline import MASKER_VAL_CATEGORIES
        n_val = len(val_recs or train_recs)
        active_categories = list(MASKER_VAL_CATEGORIES)
        if float(settings.synth_kwargs.get(
                "mp2rage_posterior_fossa_fraction", 0.0) or 0.0) <= 0.0:
            active_categories.remove("mp2rage_posterior_fossa")
        for cat in active_categories:
            val_aug_loaders[cat] = make(
                val_recs or train_recs, False, "eval", eval_bs, True, eval_category=cat)
        print(
            f"[masker val-aug] per-category val Dice ON: {', '.join(active_categories)} — "
            f"{n_val} source volume(s) x {settings.masker_val_aug_variants} variant(s) = "
            f"{n_val * settings.masker_val_aug_variants} extra val volume(s) per category, every "
            f"{settings.masker_val_aug_every} epoch(s). Selection metric: "
            f"{selection_metric}.", flush=True)
    return PreparedData(train_dl, val_dl, test_dl, val_aug_loaders, fingerprint, epoch_accounting)


def _data_orientation(settings) -> Optional[str]:
    """Axcodes of the on-disk training scans — the voxel frame the net learns in resize/patch
    mode (the trainer loads native + resizes without reorienting). Recorded in the results JSON
    as `canonical_orientation` so `predict_mask` reorients FOREIGN scans to THIS frame instead of
    guessing (avoids a silent plausible-but-wrong mask on a non-PIR-trained model). Cached; None
    on any failure (predict_mask then falls back to its PIR default with a warning)."""
    if getattr(settings, "_data_orient_cache", "unset") != "unset":
        return settings._data_orient_cache
    settings._data_orient_cache = None
    try:
        if settings.synth_online:
            from augmentations.pipeline import discover_pairs
            recs = discover_pairs(settings.data_dir, settings.scan_glob, settings.mask_suffix)
        else:
            recs = _records(settings)
        # Use the first READABLE scan's frame — not a hard-coded recs[0], whose corruption would
        # otherwise null the orientation for the whole (otherwise-fine) dataset.
        for r in recs:
            try:
                settings._data_orient_cache = "".join(nib.aff2axcodes(nib.load(str(r["scan"])).affine))
                break
            except Exception:
                continue
    except Exception:
        settings._data_orient_cache = None
    return settings._data_orient_cache


@dataclass
class PreparedData:
    train: DataLoader
    validation: DataLoader
    test: Optional[DataLoader]
    validation_categories: Dict[str, DataLoader]
    fingerprint: str
    epoch_accounting: Optional[Dict[str, int]]
