"""Brain-masking U-Nets and shared inference helpers.
"""

from typing import List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================ MASKER
def _to_spatial_dropout3d(module: nn.Module) -> int:
    """In-place swap every ``nn.Dropout`` for an equivalent ``nn.Dropout3d`` (channel-wise).

    MONAI's ``DynUNet(dropout=p)`` only ever instantiates ELEMENT-WISE ``nn.Dropout`` (its
    ``Convolution`` blocks default ``dropout_dim=1`` and never override it). On correlated 3D conv
    feature maps element-wise dropout is a weak regularizer; SPATIAL dropout (drop whole channels,
    Tompson et al. 2015) is materially stronger. This walks the built net and upgrades each dropout
    layer in place — preserving its rate/position — and returns how many it swapped. ``nn.Dropout3d``
    is a sibling of ``nn.Dropout`` (not a subclass), so the swap is idempotent.
    """
    swapped = 0
    for parent in module.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Dropout):
                setattr(parent, name, nn.Dropout3d(p=child.p, inplace=child.inplace))
                swapped += 1
    return swapped


def _disable_output_head_dropout(net: nn.Module) -> int:
    """Force every dropout inside the DynUNet OUTPUT heads to ``nn.Identity``; return how many.

    MONAI's ``DynUNet`` appends the constructor ``dropout`` to EVERY conv block, the final
    ``output_block`` and each ``deep_supervision_heads`` entry included. Those heads have
    ``out_channels == 1``, so a channel-wise ``nn.Dropout3d`` there drops the SINGLE output channel
    with probability ``p`` — zeroing the entire logit map ~``p`` of the forward passes (a silent,
    catastrophic training-quality regression: the head gets zero gradient and an all-0.5 loss). This
    is exactly what ``_to_spatial_dropout3d`` would do to the heads if left unattended. Element-wise
    dropout on a 1-channel logit map is also pointless, so we neutralize head dropout outright rather
    than merely skipping the swap. Guards for attribute absence (deep supervision off, MONAI churn).
    """
    replaced = 0
    heads: List[nn.Module] = []
    output_block = getattr(net, "output_block", None)
    if output_block is not None:
        heads.append(output_block)
    ds_heads = getattr(net, "deep_supervision_heads", None)
    if ds_heads is not None:
        heads.extend(list(ds_heads))
    for head in heads:
        for parent in head.modules():
            for name, child in list(parent.named_children()):
                if isinstance(child, (nn.Dropout, nn.Dropout3d)):
                    setattr(parent, name, nn.Identity())
                    replaced += 1
    return replaced


def require_divisible_spatial_shape(shape, stride_product: int = 32, min_size: int = 64) -> None:
    """Validate a 3D spatial shape (patch / target size) against the masker's downsampling factor.

    The masker U-Net has 5 stride-2 downsamplings, so every spatial axis is divided by
    ``stride_product == 2**5 == 32`` on the way to the bottleneck; a size that is not a multiple of
    32 (or below ``min_size``) otherwise triggers a cryptic mid-forward InstanceNorm / shape error at
    the bottleneck. Callers (``train_model`` / ``predict``) should call this on their configured
    patch/target shape to fail loud at config time instead. NOT called inside the models — they never
    see the raw input size. Raises ``ValueError`` naming the offending axis; returns ``None`` on OK.
    """
    dims = tuple(int(s) for s in shape)
    if len(dims) != 3:
        raise ValueError(
            f"require_divisible_spatial_shape expects a 3D (D, H, W) shape; got {dims!r} "
            f"({len(dims)} axes)."
        )
    for axis, size in enumerate(dims):
        if size < min_size:
            raise ValueError(
                f"spatial shape {dims!r} axis {axis} == {size} is below the minimum {min_size} "
                f"(the masker's 5 downsamplings need >= {min_size} per axis)."
            )
        if size % stride_product != 0:
            raise ValueError(
                f"spatial shape {dims!r} axis {axis} == {size} is not a multiple of {stride_product} "
                f"(= 2**5, the masker's 5-downsample stride product); pad/crop to a multiple of "
                f"{stride_product}."
            )


class MaskingUNet(nn.Module):
    """3D brain-mask U-Net (MONAI DynUNet / nnU-Net recipe).

    Output contract
    ---------------
    * eval / `deep_supervision=False`: ``[B, C, D, H, W]`` where C=1 normally, or C=2 when
      ``sdt_auxiliary=True`` (channel 0 mask logit, channel 1 linear signed distance in mm).
    * train with deep supervision: DynUNet stacks the main + auxiliary heads along dim=1, shape
      ``[B, 1 + deep_supr_num, out_channels, D, H, W]``. `MaskingLoss` consumes that directly
      (it weights the heads). Use `.infer(x)` for a clean single-head, mask-only logit map at any time.

    Over-inclusion (prefer a slightly-too-large mask over cutting brain) is handled by the LOSS
    (FN-weighted Tversky) + POST-PROCESSING (`postprocess_mask`), not the architecture.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        sdt_auxiliary: bool = False,
        deep_supervision: bool = True,
        deep_supr_num: int = 2,
        filters: Optional[List[int]] = None,
        dropout: float = 0.1,
        spatial_dropout: bool = True,
    ):
        super().__init__()
        try:
            from monai.networks.nets import DynUNet
        except Exception as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "MaskingUNet requires MONAI (`pip install monai`). It wraps monai.networks.nets.DynUNet."
            ) from exc

        self.sdt_auxiliary = bool(sdt_auxiliary)
        if self.sdt_auxiliary:
            if int(out_channels) not in (1, 2):
                raise ValueError(
                    "sdt_auxiliary=True reserves exactly two output channels: mask logit + SDT")
            out_channels = 2
        elif int(out_channels) != 1:
            raise ValueError(
                "MaskingUNet supports one mask channel, or two channels via sdt_auxiliary=True")

        # 6-level 3D U-Net (5 downsamplings); 128^3 -> 4^3 bottleneck (strides take 128 -> 4).
        strides = [1, 2, 2, 2, 2, 2]
        kernels = [3, 3, 3, 3, 3, 3]
        self.deep_supervision = deep_supervision
        self.deep_supr_num = deep_supr_num if deep_supervision else 0

        self.net = DynUNet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernels,
            strides=strides,
            upsample_kernel_size=strides[1:],
            filters=filters or [32, 64, 128, 256, 320, 320],
            norm_name="instance",          # small-3D-batch safe; nnU-Net/DynUNet default
            act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            deep_supervision=deep_supervision,
            deep_supr_num=self.deep_supr_num,
            res_block=True,                 # the architectural choice that actually helps
            dropout=(dropout or None),      # DEFAULT-ON regularization (None == off; DynUNet's default)
        )

        # DynUNet's `dropout` is ELEMENT-WISE; upgrade to channel-wise SPATIAL dropout by default,
        # which regularizes correlated 3D conv features far better (see _to_spatial_dropout3d).
        self.dropout = float(dropout)
        self.spatial_dropout = bool(spatial_dropout)
        if spatial_dropout and dropout and dropout > 0:
            _to_spatial_dropout3d(self.net)

        # CRITICAL: DynUNet also appends `dropout` to its out_channels==1 output & deep-supervision
        # heads. Channel-wise Dropout3d on a single logit channel drops the WHOLE map ~p of the time
        # (zeroed logits + dead gradient); even element-wise dropout on final logits is pointless.
        # Neutralize head dropout outright, independent of the encoder/decoder spatial swap above.
        _disable_output_head_dropout(self.net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def infer(self, x: torch.Tensor) -> torch.Tensor:
        """Single-head, single-channel MASK logit map for every deployment configuration.

        If the optional signed-distance auxiliary head was trained, channel 1 is deliberately
        discarded here. Serving, diagnosis, and threshold sweeps therefore keep their historical
        sigmoid-mask contract and cannot accidentally threshold the regression channel.
        """
        was_training = self.training
        self.eval()
        try:
            out = self.net(x)
            if out.dim() == x.dim() + 1:  # stacked deep-supervision heads -> take the main head
                out = out[:, 0]
            expected = 2 if self.sdt_auxiliary else 1
            if out.dim() != x.dim() or out.shape[1] != expected:
                raise RuntimeError(
                    f"MaskingUNet inference output contract violated: expected [B,{expected},D,H,W], "
                    f"got {tuple(out.shape)}")
            return out[:, :1]          # channel 0 is always the segmentation logit
        finally:
            if was_training:
                self.train()


class _MaskerResidualBlock3d(nn.Module):
    """ResNet-D-style 3D residual block used by :class:`ResidualEncoderMaskingUNet`.

    The downsampling shortcut averages before its 1x1 projection instead of learning a stride-2
    point sample.  This retains more of a thin brain boundary while the main branch learns the
    strided representation.  InstanceNorm is affine and batch-size independent, which is essential
    for 128^3 patches where the practical batch size is one.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        dropout: float = 0.0,
        spatial_dropout: bool = True,
    ):
        super().__init__()
        if stride not in (1, 2):
            raise ValueError(f"masker residual-block stride must be 1 or 2, got {stride}")
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels, affine=True, eps=1e-5)
        self.act1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        if dropout and dropout > 0.0:
            drop_cls = nn.Dropout3d if spatial_dropout else nn.Dropout
            self.drop = drop_cls(p=float(dropout))
        else:
            self.drop = nn.Identity()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels, affine=True, eps=1e-5)
        self.act2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            shortcut: List[nn.Module] = []
            if stride == 2:
                # Match the padded stride-2 3x3 main convolution's ceil(n/2) size on odd direct
                # inputs. Public train/serve shapes are /32, but this keeps the module self-consistent.
                shortcut.append(nn.AvgPool3d(kernel_size=2, stride=2, ceil_mode=True))
            shortcut.extend([
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True, eps=1e-5),
            ])
            self.shortcut = nn.Sequential(*shortcut)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.act1(self.norm1(out))
        out = self.drop(out)
        out = self.norm2(self.conv2(out))
        return self.act2(out + identity)


class _MaskerDecoderStage3d(nn.Module):
    """Lightweight one-convolution decoder stage from the nnU-Net residual-encoder recipe."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        dropout: float = 0.0,
        spatial_dropout: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm = nn.InstanceNorm3d(out_channels, affine=True, eps=1e-5)
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        if dropout and dropout > 0.0:
            drop_cls = nn.Dropout3d if spatial_dropout else nn.Dropout
            self.drop = drop_cls(p=float(dropout))
        else:
            self.drop = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.norm(self.conv(x))))


class ResidualEncoderMaskingUNet(nn.Module):
    """Deep nnU-Net-style residual-encoder masker for difficult real acquisition domains.

    This is an opt-in alternative to :class:`MaskingUNet`, modeled on the residual-encoder presets
    that improved the controlled nnU-Net benchmarks.  Extra capacity is concentrated after spatial
    downsampling, while a lightweight decoder keeps full-resolution memory practical.  It has five
    stride-2 reductions, so the same factor-32 patch/volume shape contract applies.

    Output contracts intentionally match ``MaskingUNet`` exactly:

    * train + deep supervision: ``[B, 1 + deep_supr_num, C, D, H, W]``;
    * eval: ``[B, C, D, H, W]``;
    * :meth:`infer`: mask channel 0 only, ``[B, 1, D, H, W]``.

    ``C`` is one normally and two with ``sdt_auxiliary=True``.  Auxiliary decoder predictions are
    interpolated to full resolution before stacking so the existing loss can share one target.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        sdt_auxiliary: bool = False,
        deep_supervision: bool = True,
        deep_supr_num: int = 2,
        filters: Optional[Sequence[int]] = None,
        blocks_per_stage: Optional[Sequence[int]] = None,
        dropout: float = 0.0,
        spatial_dropout: bool = True,
    ):
        super().__init__()
        self.sdt_auxiliary = bool(sdt_auxiliary)
        if self.sdt_auxiliary:
            if int(out_channels) not in (1, 2):
                raise ValueError(
                    "sdt_auxiliary=True reserves exactly two output channels: mask logit + SDT")
            out_channels = 2
        elif int(out_channels) != 1:
            raise ValueError(
                "ResidualEncoderMaskingUNet supports one mask channel, or two channels via "
                "sdt_auxiliary=True")

        channels = tuple(int(v) for v in (filters or (32, 64, 128, 256, 320, 320)))
        block_counts = tuple(int(v) for v in (blocks_per_stage or (1, 2, 3, 4, 4, 4)))
        if len(channels) != 6 or any(v <= 0 for v in channels):
            raise ValueError(f"resenc filters must contain six positive values, got {channels!r}")
        if len(block_counts) != 6 or any(v < 1 for v in block_counts):
            raise ValueError(
                f"resenc blocks_per_stage must contain six values >= 1, got {block_counts!r}")
        if int(deep_supr_num) < 0 or int(deep_supr_num) > 4:
            raise ValueError("resenc deep_supr_num must be in [0, 4]")

        self.filters = channels
        self.blocks_per_stage = block_counts
        self.deep_supervision = bool(deep_supervision)
        self.deep_supr_num = int(deep_supr_num) if self.deep_supervision else 0
        self.dropout = float(dropout or 0.0)
        self.spatial_dropout = bool(spatial_dropout)

        self.stem = nn.Sequential(
            nn.Conv3d(int(in_channels), channels[0], kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(channels[0], affine=True, eps=1e-5),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

        encoder_stages: List[nn.Module] = []
        for level, (width, count) in enumerate(zip(channels, block_counts)):
            stage: List[nn.Module] = []
            stage_in = channels[0] if level == 0 else channels[level - 1]
            stage.append(_MaskerResidualBlock3d(
                stage_in, width, stride=(1 if level == 0 else 2),
                dropout=self.dropout, spatial_dropout=self.spatial_dropout))
            stage.extend(
                _MaskerResidualBlock3d(
                    width, width, dropout=self.dropout,
                    spatial_dropout=self.spatial_dropout)
                for _ in range(count - 1)
            )
            encoder_stages.append(nn.Sequential(*stage))
        self.encoder_stages = nn.ModuleList(encoder_stages)

        self.upsamplers = nn.ModuleList()
        self.decoder_stages = nn.ModuleList()
        for level in range(5, 0, -1):
            skip_width = channels[level - 1]
            self.upsamplers.append(nn.ConvTranspose3d(
                channels[level], skip_width, kernel_size=2, stride=2, bias=False))
            self.decoder_stages.append(_MaskerDecoderStage3d(
                2 * skip_width, skip_width, dropout=self.dropout,
                spatial_dropout=self.spatial_dropout))

        # Deliberate names: Trainer's 1-channel -> mask+SDT partial warm-start recognizes these.
        self.output_block = nn.Conv3d(channels[0], out_channels, kernel_size=1)
        self.deep_supervision_heads = nn.ModuleList([
            nn.Conv3d(channels[level], out_channels, kernel_size=1)
            for level in range(1, 1 + self.deep_supr_num)
        ])

        self.apply(self._init_module)
        # Start residual branches as identities/projections. This stabilizes the much deeper encoder
        # without weakening gradient flow through its shortcuts.
        for module in self.modules():
            if isinstance(module, _MaskerResidualBlock3d):
                nn.init.zeros_(module.norm2.weight)

    @staticmethod
    def _init_module(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(
                module.weight, a=0.01, mode="fan_in", nonlinearity="leaky_relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.InstanceNorm3d) and module.affine:
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _decode(self, x: torch.Tensor):
        skips: List[torch.Tensor] = []
        h = self.stem(x)
        for stage in self.encoder_stages:
            h = stage(h)
            skips.append(h)

        decoded: List[torch.Tensor] = []
        h = skips[-1]
        for index, (up, stage) in enumerate(zip(self.upsamplers, self.decoder_stages)):
            skip = skips[-2 - index]
            h = up(h)
            if h.shape[2:] != skip.shape[2:]:
                # The public config rejects non-factor-32 shapes, but keep the module robust for
                # direct callers and sliding-window edge padding.
                h = F.interpolate(h, size=skip.shape[2:], mode="trilinear", align_corners=False)
            h = stage(torch.cat([h, skip], dim=1))
            decoded.append(h)
        return decoded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        decoded = self._decode(x)
        main = self.output_block(decoded[-1])
        if self.training and self.deep_supervision and self.deep_supr_num:
            outputs = [main]
            for index, head in enumerate(self.deep_supervision_heads):
                aux = head(decoded[-2 - index])
                aux = F.interpolate(
                    aux, size=main.shape[2:], mode="trilinear", align_corners=False)
                outputs.append(aux)
            return torch.stack(outputs, dim=1)
        return main

    @torch.no_grad()
    def infer(self, x: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        self.eval()
        try:
            out = self.forward(x)
            expected = 2 if self.sdt_auxiliary else 1
            if out.dim() != x.dim() or out.shape[1] != expected:
                raise RuntimeError(
                    "ResidualEncoderMaskingUNet inference output contract violated: "
                    f"expected [B,{expected},D,H,W], got {tuple(out.shape)}")
            return out[:, :1]
        finally:
            if was_training:
                self.train()


@torch.no_grad()
def postprocess_mask(
    prob: torch.Tensor,
    threshold: float = 0.4,
    dilate_iters: int = 1,
    max_fraction: float = 0.75,
    min_fraction: float = 0.02,
    cc_keep_ratio: float = 0.05,
    return_flags: bool = False,
    voxel_sizes=None,
    dilate_mm: float = None,
):
    """Turn a sigmoid probability map into a binary, over-inclusive brain mask.

    The hard over-inclusion layer (SynthStrip's `-b` border analog):
      threshold (< 0.5) -> keep brain component(s) -> fill holes -> dilation -> SYMMETRIC review flags.

    Connected-component handling (clinical safety): label with 26-connectivity (so diagonally-
    touching brain is not pre-split) and keep EVERY component whose size is >= ``cc_keep_ratio`` of
    the largest. Keeping only the single largest component — the old behavior — silently erases a
    real disconnected region: a cerebellum/cerebrum split across a thin low-SNR/motion brainstem
    bridge, a hemispherectomy/resection, or bilateral disconnected pathology. Keeping more is the
    over-inclusion-safe direction (a clinician reviews extra tissue; they cannot review tissue that
    was deleted). Truly detached small false-positive islands (< the ratio) are still dropped.

    Border grows by ``dilate_iters`` VOXELS by default. Pass ``dilate_mm`` (with ``voxel_sizes`` = the
    per-axis mm spacing) to instead grow a fixed PHYSICAL border via a spacing-aware distance transform —
    the honest cross-scanner border (1 voxel is 0.7 mm or 5 mm depending on acquisition). ``dilate_mm``
    overrides ``dilate_iters`` when set; ``dilate_mm=None`` (default) keeps the legacy voxel dilation.

    Returns a uint8 mask ``[B, 1, D, H, W]``. When ``return_flags=True`` returns ``(mask, flags)``
    where ``flags`` is a list (len B) of per-volume dicts with keys: ``oversize`` (fraction >
    max_fraction), ``undersize`` (non-empty but fraction < min_fraction), ``empty`` (no voxels above
    threshold — the maximal false negative), ``dropped_component`` (a non-trivial component, >
    min_fraction of the volume, was discarded by the ratio gate), ``mask_fraction``, and ``review``
    (any of the above -> a human must look). The mask is NEVER zeroed to "fix" an oversize result
    (keeping all brain is preferable); under-segmentation is FLAGGED, not silently shipped. `prob` is
    expected in [0, 1] (apply sigmoid before calling). scipy on CPU (per-volume; fine at inference).
    """
    import numpy as np
    from scipy import ndimage as ndi

    arr = prob.detach().cpu().numpy()
    out = np.zeros_like(arr, dtype=np.uint8)
    struct = np.ones((3, 3, 3), dtype=bool)  # 26-connectivity
    flags = []
    for b in range(arr.shape[0]):
        m = arr[b, 0] >= threshold
        empty = not m.any()
        dropped = False
        if empty:
            frac = 0.0
        else:
            lbl, n = ndi.label(m, structure=struct)
            if n > 1:  # keep brain component(s); drop only small detached FP islands
                sizes = ndi.sum(np.ones_like(lbl), lbl, index=np.arange(1, n + 1))
                largest = float(sizes.max())
                keep_floor = float(cc_keep_ratio) * largest  # keep >= 5% of largest (e.g. cerebellum)
                keep_ids = np.nonzero(sizes >= keep_floor)[0] + 1
                m = np.isin(lbl, keep_ids)
                # flag a DROPPED component that was non-trivial relative to the brain (>= 2% of the
                # largest component) — a speck below that is a true FP island and dropped silently.
                dropped = bool((sizes[sizes < keep_floor] >= 0.02 * largest).any())
            m = ndi.binary_fill_holes(m)
            if dilate_mm is not None and voxel_sizes is not None and float(dilate_mm) > 0:
                # Fixed PHYSICAL border (opt-in): grow outward by `dilate_mm` mm using per-axis voxel
                # spacing, so 1 mm is 1 mm on a 1mm-iso scan AND on a 5mm-slice scan (a fixed VOXEL count
                # is not). `<=` is load-bearing: a face neighbour sits at exactly one voxel spacing and must
                # be kept -> on 1mm-iso, dilate_mm=1.0 is byte-identical to binary_dilation(iterations=1).
                m = ndi.distance_transform_edt(~m, sampling=voxel_sizes) <= float(dilate_mm)
            elif dilate_iters > 0:
                m = ndi.binary_dilation(m, iterations=dilate_iters)
            out[b, 0] = m.astype(np.uint8)
            frac = float(m.mean())
        oversize = frac > max_fraction
        undersize = (not empty) and frac < min_fraction
        flags.append({
            "oversize": oversize,
            "undersize": undersize,
            "empty": empty,
            "dropped_component": dropped,
            "mask_fraction": frac,
            "review": bool(oversize or undersize or empty or dropped),
        })
    mask = torch.from_numpy(out)
    return (mask, flags) if return_flags else mask


def build_model(model_type: str, **kwargs) -> nn.Module:
    """Factory for the brain-masking models.

    Masking defaults to the historical DynUNet.  ``architecture='resenc'`` selects the deeper
    residual-encoder model; keeping the selector in ``model_kwargs`` makes deploy checkpoints and
    every inference entry point self-describing.
    """
    mt = model_type.lower()
    if mt in ("masking", "mask", "masker"):
        architecture = str(kwargs.pop("architecture", "dynunet")).lower().replace("_", "-")
        if architecture in ("dynunet", "dyn-unet", "legacy"):
            return MaskingUNet(**kwargs)
        if architecture in ("resenc", "res-enc", "residual-encoder"):
            return ResidualEncoderMaskingUNet(**kwargs)
        raise ValueError(
            f"Unknown masking architecture {architecture!r} (expected 'dynunet' or 'resenc').")
    raise ValueError(f"Unknown model_type {model_type!r} (expected 'masking').")
