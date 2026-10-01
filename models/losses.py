"""Loss functions for the brain-masking models.
"""

from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================ MASKER
def tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    w_fp: float = 0.3,
    w_fn: float = 0.7,
    focal_exponent: float = 0.75,
    eps: float = 1e-6,
) -> torch.Tensor:
    """FN-weighted (Focal) Tversky loss on a single logit head.

    TI = TP / (TP + w_fp*FP + w_fn*FN);  loss = (1 - TI)^focal_exponent, averaged over the batch.
    `w_fn > w_fp` is the over-inclusion lever. `focal_exponent=0.75` matches Abraham & Khan's
    Focal-Tversky gamma=4/3 (loss = (1-TI)^(1/gamma)); use 1.0 for plain Tversky.
    """
    prob = torch.sigmoid(logits)
    dims = tuple(range(1, prob.dim()))  # reduce over channels + spatial, keep batch
    tp = (prob * target).sum(dims)
    fp = (prob * (1.0 - target)).sum(dims)
    fn = ((1.0 - prob) * target).sum(dims)
    ti = (tp + eps) / (tp + w_fp * fp + w_fn * fn + eps)
    return torch.pow(torch.clamp(1.0 - ti, min=eps), focal_exponent).mean()


def signed_distance_transform_numpy(
    mask: np.ndarray,
    voxel_sizes: Optional[Union[float, Sequence[float]]] = None,
    *,
    inside_positive: bool = False,
) -> np.ndarray:
    """Euclidean signed distance for one 3D mask, optionally in physical units.

    The historical/default convention is positive outside and negative inside. Set
    ``inside_positive=True`` for the SynthStrip regression convention. Degenerate masks return
    zeros because no boundary exists. Kept NumPy-level so DataLoader workers can compute a full-
    volume target once and co-crop it with the image, avoiding patch-edge geometry errors and a
    GPU-to-CPU synchronization inside the loss.
    """
    from scipy.ndimage import distance_transform_edt

    g = np.asarray(mask) > 0.5
    if not g.any() or g.all():
        return np.zeros(g.shape, dtype=np.float32)
    if voxel_sizes is None:
        sampling = None
    elif np.isscalar(voxel_sizes):
        sampling = (float(voxel_sizes),) * g.ndim
    else:
        sampling = tuple(float(x) for x in voxel_sizes)
        if len(sampling) != g.ndim:
            raise ValueError(f"voxel_sizes must have {g.ndim} values; got {sampling!r}")
    outside_positive = (distance_transform_edt(~g, sampling=sampling)
                        - distance_transform_edt(g, sampling=sampling))
    out = -outside_positive if inside_positive else outside_positive
    return np.asarray(out, dtype=np.float32)


@torch.no_grad()
def signed_distance_transform(
    target: torch.Tensor,
    voxel_sizes: Optional[Union[float, Sequence[float]]] = None,
    *,
    inside_positive: bool = False,
) -> torch.Tensor:
    """Per-sample signed distance map (SDM) of a binary GT mask (Kervadec et al. boundary loss).

    For each ``[B, C]`` volume in ``target`` (thresholded at 0.5), returns the Euclidean signed
    distance in voxel units by default, or physical units when ``voxel_sizes`` is supplied: NEGATIVE
    inside the GT foreground, POSITIVE outside, ~0 on the boundary. ``inside_positive=True`` reverses
    this sign for direct SynthStrip-style regression.
    Computed with ``scipy.ndimage.distance_transform_edt`` as ``edt(1 - g) - edt(g)`` — ``edt(x)``
    is the distance from each voxel to the nearest ZERO of ``x``, so ``edt(1-g)`` is the outward
    distance-to-foreground (positive outside, 0 inside) and ``edt(g)`` the inward distance-to-
    background (positive inside, 0 outside).

    The map is a CONSTANT derived from the target (no grad; hence ``@torch.no_grad``). It is returned
    on the target's device/dtype. Distances are raw, not normalized; when ``voxel_sizes`` is omitted,
    downstream surface term's magnitude scales with volume size — this is why the ablation weights
    (``w_surface`` = 0.1 / 0.2) are small relative to the O(1) Tversky/BCE terms.

    Degenerate samples (a volume with no foreground OR no background — no boundary exists) get an
    all-zero map: scipy would otherwise return finite-but-arbitrary edge fills, and zeroing cleanly
    disables surface supervision where it is undefined.
    """
    npt = (target.detach().cpu().numpy() > 0.5)  # [B, C, *spatial] boolean GT
    out = torch.zeros_like(target, dtype=torch.float32)
    flat_g = npt.reshape(-1, *npt.shape[2:])          # [(B*C), *spatial]
    flat_o = out.view(-1, *out.shape[2:])
    for i in range(flat_g.shape[0]):
        sdm = signed_distance_transform_numpy(
            flat_g[i], voxel_sizes=voxel_sizes, inside_positive=inside_positive)
        flat_o[i] = torch.from_numpy(sdm).to(dtype=torch.float32)
    return out.to(device=target.device, dtype=target.dtype)


def boundary_loss(logits: torch.Tensor, sdt: torch.Tensor) -> torch.Tensor:
    """Kervadec-style boundary (surface) loss: ``mean( sigmoid(logits) * signed_distance(target) )``.

    ``sdt`` is the signed distance map from :func:`signed_distance_transform` (negative inside the GT,
    positive outside). Probability mass placed OUTSIDE the boundary lands on positive distances and
    RAISES the loss (the further out, the worse); mass INSIDE lands on negative distances and lowers
    it. This directly penalizes over-spill far from the true surface, complementing the region-based
    Tversky/BCE. Reduction is a plain mean over all voxels (batch + channel + spatial), so it slots
    into the same per-head weighting the existing terms use.
    """
    return (torch.sigmoid(logits) * sdt).mean()


def binary_boundary_band_3d(target: torch.Tensor, radius_vox: int = 2) -> torch.Tensor:
    """Return a binary inner+outer boundary ribbon without inventing crop-edge boundaries.

    ``target`` must be ``[B,C,D,H,W]``. Replicate padding is deliberate: zero padding would label
    the edge of an all-foreground patch as an anatomical boundary, teaching the model a crop-box cue.
    The result is detached target geometry and is built entirely on the target device.
    """
    if target.dim() != 5:
        raise ValueError(
            f"binary_boundary_band_3d expects [B,C,D,H,W], got {tuple(target.shape)}")
    radius = int(radius_vox)
    if radius < 1 or radius != radius_vox:
        raise ValueError(f"radius_vox must be a positive integer, got {radius_vox!r}")
    binary = (target.detach() > 0.5).to(dtype=target.dtype)
    pad = (radius, radius, radius, radius, radius, radius)
    kernel = 2 * radius + 1
    padded_fg = F.pad(binary, pad, mode="replicate")
    padded_bg = F.pad(1.0 - binary, pad, mode="replicate")
    dilated = F.max_pool3d(padded_fg, kernel_size=kernel, stride=1)
    eroded = 1.0 - F.max_pool3d(padded_bg, kernel_size=kernel, stride=1)
    return (dilated - eroded).clamp_(0.0, 1.0)


def boundary_focal_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    boundary_band: torch.Tensor,
    *,
    pos_weight: float = 2.5,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Hard-example focal BCE restricted to a target-derived 3D boundary ribbon.

    Each subject is normalized by its own boundary-voxel count so large heads do not dominate a
    batch. Samples with no real boundary contribute differentiable zero rather than NaN. Positive
    boundary misses retain ``pos_weight`` asymmetry, while the outer half of the same ribbon keeps
    pressure on false-positive skull/dura spill.
    """
    if tuple(logits.shape) != tuple(target.shape) or tuple(target.shape) != tuple(boundary_band.shape):
        raise ValueError(
            "boundary focal inputs must have the same shape: "
            f"logits={tuple(logits.shape)}, target={tuple(target.shape)}, "
            f"band={tuple(boundary_band.shape)}")
    if not np.isfinite(gamma) or gamma < 0.0:
        raise ValueError("boundary focal gamma must be finite and >= 0")
    pw = torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device)
    raw = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw, reduction="none")
    prob = torch.sigmoid(logits)
    p_t = target * prob + (1.0 - target) * (1.0 - prob)
    weighted = boundary_band * torch.pow((1.0 - p_t).clamp_min(0.0), float(gamma)) * raw
    dims = tuple(range(1, logits.dim()))
    counts = boundary_band.sum(dims)
    per_sample = weighted.sum(dims) / counts.clamp_min(1.0)
    valid = (counts > 0.0).to(dtype=per_sample.dtype)
    return (per_sample * valid).sum() / valid.sum().clamp_min(1.0)


MASKING_LOSS_PROFILES = {
    # Moderate recall increase plus a hard-boundary objective. Do not push the asymmetry further
    # before measuring precision on real MP2RAGE: excessive FN weighting can simply grow skull/dura.
    "recall-boundary": {
        "w_fp": 0.25,
        "w_fn": 0.75,
        "pos_weight": 2.5,
        "w_boundary_focal": 0.25,
        "boundary_band_vox": 2,
        "boundary_focal_gamma": 2.0,
    },
}


class MaskingLoss(nn.Module):
    """FN-weighted Tversky + BCE (+ optional boundary objectives/SDT), DS aware.

    `pred` may be a single logit map ``[B, C, ...]`` or DynUNet's stacked deep-supervision heads
    ``[B, K, C, ...]`` — in the latter case the loss is a decreasing-weighted sum over the K heads
    (1, 0.5, 0.25, ... normalized), the standard deep-supervision scheme.

    Optional SURFACE term (`w_surface > 0`, DEFAULT OFF): a Kervadec-style boundary loss
    ``mean(prob * signed_distance(target))`` that penalizes probability placed far OUTSIDE the GT
    surface (SynthStrip-style mm/voxel signed-distance supervision). It is an ABLATION lever:
    at ``w_surface == 0.0`` the surface term is skipped ENTIRELY (no SDT computed) and the total is
    byte-identical to the Tversky+BCE loss. The signed distance map is computed ONCE on the full-
    resolution ``target`` and SHARED across every deep-supervision head — consistent with the fact
    that MONAI DynUNet interpolates all DS heads to the main resolution before stacking, which is
    exactly why the existing Tversky/BCE terms already reuse one full-res ``target`` for all heads.
    See :func:`boundary_loss` / :func:`signed_distance_transform` for sign convention and units.
    """

    def __init__(
        self,
        w_fp: float = 0.3,
        w_fn: float = 0.7,
        focal_exponent: float = 0.75,
        ce_weight: float = 1.0,
        pos_weight: float = 2.0,
        w_surface: float = 0.0,
        w_sdt_aux: float = 0.0,
        sdt_band_mm: float = 5.0,
        sdt_far_weight: float = 0.1,
        sdt_voxel_mm: float = 1.0,
        w_boundary_focal: float = 0.0,
        boundary_band_vox: int = 2,
        boundary_focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.w_fp, self.w_fn = float(w_fp), float(w_fn)
        self.focal_exponent = float(focal_exponent)
        self.ce_weight = float(ce_weight)
        # BCE alone is symmetric and DILUTES the Tversky over-inclusion bias. pos_weight > 1 makes
        # the BCE penalize false negatives (missed brain, the positive class) more than false
        # positives, so the CE term reinforces — rather than washes out — the FN-weighted Tversky.
        self.pos_weight = float(pos_weight)
        # Optional boundary/surface term (ablation lever). 0.0 == disabled == byte-identical default.
        self.w_surface = float(w_surface)
        # GPU-native hard-boundary term used by the recall-boundary profile. Unlike w_surface it
        # needs no CPU EDT and emphasizes both inner misses and outer spill in the same local ribbon.
        self.w_boundary_focal = float(w_boundary_focal)
        self.boundary_band_vox = int(boundary_band_vox)
        self.boundary_focal_gamma = float(boundary_focal_gamma)
        # Optional SynthStrip-style direct signed-distance regression. The auxiliary model channel is
        # linear; targets are clipped to +/-5 mm and voxels outside that ribbon receive 0.1x MSE,
        # matching the paper's h=5 mm / b=0.1 recipe. It supplements rather than replaces the stable
        # region loss here, hence the small default experiment weight supplied by the CLI.
        self.w_sdt_aux = float(w_sdt_aux)
        self.sdt_band_mm = float(sdt_band_mm)
        self.sdt_far_weight = float(sdt_far_weight)
        self.sdt_voxel_mm = float(sdt_voxel_mm)
        if not np.isfinite(self.w_boundary_focal) or self.w_boundary_focal < 0.0:
            raise ValueError("w_boundary_focal must be finite and >= 0")
        if self.boundary_band_vox < 1 or self.boundary_band_vox != boundary_band_vox:
            raise ValueError("boundary_band_vox must be a positive integer")
        if not np.isfinite(self.boundary_focal_gamma) or self.boundary_focal_gamma < 0.0:
            raise ValueError("boundary_focal_gamma must be finite and >= 0")
        if self.w_surface > 0.0 and self.w_boundary_focal > 0.0:
            raise ValueError(
                "w_surface and w_boundary_focal are alternative boundary objectives; enable one")
        if self.w_sdt_aux < 0.0:
            raise ValueError("w_sdt_aux must be >= 0")
        if not np.isfinite(self.sdt_band_mm) or self.sdt_band_mm <= 0.0:
            raise ValueError("sdt_band_mm must be finite and > 0")
        if not 0.0 <= self.sdt_far_weight <= 1.0:
            raise ValueError("sdt_far_weight must be in [0, 1]")
        if not np.isfinite(self.sdt_voxel_mm) or self.sdt_voxel_mm <= 0.0:
            raise ValueError("sdt_voxel_mm must be finite and > 0")

    def _single(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        surface_sdt: Optional[torch.Tensor] = None,
        boundary_band: Optional[torch.Tensor] = None,
        aux_sdt: Optional[torch.Tensor] = None,
        aux_sdt_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        expected_channels = 2 if self.w_sdt_aux > 0.0 else 1
        if logits.shape[1] != expected_channels:
            raise ValueError(
                f"MaskingLoss expected {expected_channels} output channel(s) "
                f"(mask{' + SDT' if expected_channels == 2 else ''}), got {logits.shape[1]}")
        mask_logits = logits[:, :1]
        if tuple(mask_logits.shape) != tuple(target.shape):
            raise ValueError(
                f"mask logit/target shape mismatch: {tuple(mask_logits.shape)} vs {tuple(target.shape)}")
        tv = tversky_loss(mask_logits, target, self.w_fp, self.w_fn, self.focal_exponent)
        pw = torch.as_tensor(self.pos_weight, dtype=mask_logits.dtype, device=mask_logits.device)
        ce = F.binary_cross_entropy_with_logits(mask_logits, target, pos_weight=pw)
        loss = tv + self.ce_weight * ce
        if self.w_surface > 0.0 and surface_sdt is not None:
            loss = loss + self.w_surface * boundary_loss(mask_logits, surface_sdt)
        if self.w_boundary_focal > 0.0:
            if boundary_band is None:
                raise RuntimeError(
                    "boundary-focal supervision was enabled but no target band was built")
            loss = loss + self.w_boundary_focal * boundary_focal_bce_loss(
                mask_logits, target, boundary_band, pos_weight=self.pos_weight,
                gamma=self.boundary_focal_gamma)
        if self.w_sdt_aux > 0.0:
            if aux_sdt is None:
                raise RuntimeError("SDT auxiliary supervision was enabled but no SDT target was built")
            pred_sdt = logits[:, 1:2]
            target_sdt = aux_sdt.to(device=pred_sdt.device, dtype=pred_sdt.dtype)
            if tuple(target_sdt.shape) != tuple(pred_sdt.shape):
                raise ValueError(
                    f"SDT prediction/target shape mismatch: {tuple(pred_sdt.shape)} vs "
                    f"{tuple(target_sdt.shape)}")
            # A target at the clipped +/-band value represents a far-away voxel. Exact equality at
            # 5 mm is harmlessly included in the down-weighted set.
            if aux_sdt_weight is None:
                # Fallback targets were clipped locally; values strictly inside the band are known
                # near voxels, while clipped values use the paper's far weight.
                near = target_sdt.abs() < (self.sdt_band_mm - 1e-6)
                weights = torch.where(
                    near, torch.ones_like(target_sdt),
                    torch.full_like(target_sdt, self.sdt_far_weight))
            else:
                weights = aux_sdt_weight.to(device=pred_sdt.device, dtype=pred_sdt.dtype)
                if tuple(weights.shape) != tuple(pred_sdt.shape):
                    raise ValueError(
                        f"SDT weight shape mismatch: {tuple(weights.shape)} vs {tuple(pred_sdt.shape)}")
            sdt_mse = (weights * (pred_sdt - target_sdt).square()).mean()
            loss = loss + self.w_sdt_aux * sdt_mse
        return loss

    def forward(self, pred: torch.Tensor, target) -> torch.Tensor:
        # Online synthesis can precompute a geometrically correct full-volume SDT in loader workers
        # and co-crop it with the mask. Tensor-only targets remain supported (offline datasets and
        # legacy callers); their SDT is computed once here and shared across all supervision heads.
        if isinstance(target, dict):
            mask_target = target["mask"]
            aux_sdt = target.get("sdt")
            aux_sdt_weight = target.get("sdt_weight")
        else:
            mask_target = target
            aux_sdt = None
            aux_sdt_weight = None

        need_raw = self.w_surface > 0.0 or (self.w_sdt_aux > 0.0 and aux_sdt is None)
        raw_sdt = (signed_distance_transform(mask_target, voxel_sizes=self.sdt_voxel_mm)
                   if need_raw else None)  # historical sign: -inside, +outside
        surface_sdt = raw_sdt if self.w_surface > 0.0 else None
        boundary_band = (binary_boundary_band_3d(mask_target, self.boundary_band_vox)
                         if self.w_boundary_focal > 0.0 else None)
        if self.w_sdt_aux > 0.0:
            if aux_sdt is None:
                inside_positive = -raw_sdt
                aux_sdt_weight = torch.where(
                    inside_positive.abs() >= self.sdt_band_mm,
                    torch.full_like(inside_positive, self.sdt_far_weight),
                    torch.ones_like(inside_positive))
                aux_sdt = torch.clamp(inside_positive, -self.sdt_band_mm, self.sdt_band_mm)
            else:
                aux_sdt = torch.clamp(aux_sdt, -self.sdt_band_mm, self.sdt_band_mm)

        if pred.dim() == mask_target.dim() + 1:  # stacked deep-supervision heads on dim=1
            heads = pred.unbind(dim=1)
            weights = [0.5 ** i for i in range(len(heads))]
            total = sum(weights)
            return sum(
                w * self._single(
                    h, mask_target, surface_sdt, boundary_band, aux_sdt, aux_sdt_weight)
                for w, h in zip(weights, heads)
            ) / total
        return self._single(
            pred, mask_target, surface_sdt, boundary_band, aux_sdt, aux_sdt_weight)


def build_loss(model_type: str, profile: Optional[str] = None, **kwargs) -> nn.Module:
    """Factory with an opt-in masking-loss profile and exact legacy defaults.

    Explicit keyword arguments override profile values. ``profile=None`` / ``'legacy'`` performs
    no merge and therefore preserves the historical MaskingLoss construction exactly.
    """
    mt = model_type.lower()
    if mt in ("masking", "mask", "masker"):
        name = "legacy" if profile is None else str(profile).lower()
        if name == "legacy":
            resolved = dict(kwargs)
        elif name in MASKING_LOSS_PROFILES:
            resolved = {**MASKING_LOSS_PROFILES[name], **kwargs}
        else:
            raise ValueError(
                f"Unknown masking loss profile {profile!r}; expected 'legacy' or one of "
                f"{sorted(MASKING_LOSS_PROFILES)}")
        return MaskingLoss(**resolved)
    raise ValueError(f"Unknown model_type {model_type!r} (expected 'masking').")
