"""Training settings: defaults, normalization, cross-field checks and serialization.

Importing this schema is stdlib-only; constructing settings activates runtime checks.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass
class TrainingConfig:
    """Validated settings accepted by ``Trainer(config)`` and the JSON configuration loader.

    Paths and numeric values are normalized once. Derived runtime identities are
    resolved only when their augmentation route is enabled.
    """
    model_type: str
    data_dir: str | Path
    model_out_path: str | Path
    results_out_path: str | Path
    target_shape: Tuple[int, int, int] = (128, 128, 128)
    batch_size: int = 2
    epochs: int = 50
    lr: float = 0.001
    weight_decay: float = 0.01
    val_fraction: float = 0.2
    test_fraction: float = 0.15
    test_dir: str | Path | None = None
    num_workers: int = 0
    seed: int = 0
    device: Optional[str] = None
    accum_steps: int = 8
    patch_size: Optional[Tuple[int, int, int]] = None
    conform_mm: Optional[float] = None
    sw_overlap: float = 0.25
    sw_batch_size: int = 4
    fg_crop_frac: float = 0.33
    masker_crops_per_volume: int = 1
    masker_volume_repeats: int = 1
    masker_norm_aug: bool = False
    saliency_adversarial: Optional[Dict[str, Any]] = None
    boundary_crop: bool = False
    select_on_deployed_dice: bool = False
    masker_val_aug: bool = False
    masker_val_aug_variants: int = 1
    masker_val_aug_every: int = 5
    masker_select_category: Optional[str] = None
    ema: bool = False
    ema_decay: float = 0.999
    amp: bool = False
    warmup_epochs: int = 0
    synth_online: bool = False
    scan_glob: str = '*_T1w.nii.gz'
    mask_suffix: str = '_brainmask_refined_crf'
    synth_kwargs: Optional[Dict[str, Any]] = None
    hard_artifact_tail_identity: Optional[Dict[str, Any]] = None
    hardtail_thread_limit_identity: Optional[Dict[str, Any]] = None
    loss_kwargs: Optional[Dict[str, Any]] = None
    model_kwargs: Optional[Dict[str, Any]] = None
    overwrite: bool = False
    resume_from: str | Path | None = None
    init_from: str | Path | None = None
    init_partial: bool = False
    subject_id_regex: Optional[str] = None
    allow_unsafe_init: bool = False

    @classmethod
    def field_names(cls):
        """Explicit accepted constructor keys; no source parsing or Trainer import."""
        return {item.name for item in fields(cls) if item.init}

    def __post_init__(self):
        self._validate_core()
        self._configure_sampling()
        self._normalize()
        self._configure_saliency()
        self._validate_selection()
        self._configure_loss()
        self._configure_outputs()

    def _validate_core(self):
        self.model_type = self.model_type.lower()
        if self.model_type != 'masking':
            raise ValueError("This condensed workflow supports model_type='masking'")
        if type(self.allow_unsafe_init) is not bool:
            raise ValueError('allow_unsafe_init must be a JSON boolean')
        if self.subject_id_regex is not None and (not isinstance(self.subject_id_regex, str) or not self.subject_id_regex):
            raise ValueError('subject_id_regex must be a nonempty regex string or null')
        try:
            self._subject_pattern = re.compile(self.subject_id_regex) if self.subject_id_regex else None
        except re.error as exc:
            raise ValueError(f'Invalid subject_id_regex: {exc}') from exc
        self.synth_online = bool(self.synth_online)
        self.synth_kwargs = dict(self.synth_kwargs or {})

    def _normalize(self):
        from training.runtime import torch
        self.data_dir = Path(self.data_dir)
        self.model_out_path = Path(self.model_out_path)
        self.results_out_path = Path(self.results_out_path)
        self.target_shape = tuple(self.target_shape)
        self.batch_size = int(self.batch_size)
        self.epochs = int(self.epochs)
        self.lr = float(self.lr)
        self.weight_decay = float(self.weight_decay)
        self.val_fraction = float(self.val_fraction)
        self.test_fraction = float(self.test_fraction)
        self.test_dir = Path(self.test_dir) if self.test_dir else None
        self.num_workers = int(self.num_workers)
        self.seed = int(self.seed)
        self.device = torch.device(self.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.accum_steps = max(1, int(self.accum_steps))
        self.patch_size = tuple((int(x) for x in self.patch_size)) if self.patch_size else None
        self.conform_mm = float(self.conform_mm) if self.conform_mm is not None else None
        self.sw_overlap = float(self.sw_overlap)
        self.sw_batch_size = max(1, int(self.sw_batch_size))
        self.fg_crop_frac = float(self.fg_crop_frac)
        self.masker_crops_per_volume = int(self.masker_crops_per_volume)
        self.masker_volume_repeats = int(self.masker_volume_repeats)
        if self.masker_crops_per_volume < 1:
            raise ValueError('masker_crops_per_volume must be >= 1')
        if self.masker_volume_repeats < 1:
            raise ValueError('masker_volume_repeats must be >= 1')
        if self.masker_crops_per_volume > 1 and (not (self.synth_online and self.patch_size is not None)):
            raise ValueError('masker_crops_per_volume > 1 requires masking + synth_online + patch_size')
        if self.masker_volume_repeats > 1 and (not self.synth_online):
            raise ValueError('masker_volume_repeats > 1 requires masking + synth_online')
        self.masker_norm_aug = bool(self.masker_norm_aug)
        self.select_on_deployed_dice = bool(self.select_on_deployed_dice)
        self.boundary_crop = bool(self.boundary_crop)
        self.ema = bool(self.ema)
        self.ema_decay = float(self.ema_decay)
        self.amp = bool(self.amp)
        self.warmup_epochs = max(0, int(self.warmup_epochs))

    def _configure_saliency(self):
        self.saliency_adversarial_identity = None
        if self.saliency_adversarial is not None:
            from augmentations.curricula.saliency_adversarial import IDENTITY, normalize_config
            self.saliency_adversarial = normalize_config(self.saliency_adversarial)
            self.saliency_adversarial_identity = IDENTITY

    def _validate_selection(self):
        self.masker_val_aug = bool(self.masker_val_aug)
        self.masker_val_aug_variants = int(self.masker_val_aug_variants)
        self.masker_val_aug_every = int(self.masker_val_aug_every)
        if self.masker_val_aug:
            if not self.synth_online:
                raise ValueError('masker_val_aug requires masking + synth_online (the categories are forced tiers of the online synthesis pipeline)')
            if self.masker_val_aug_variants < 1:
                raise ValueError('masker_val_aug_variants must be >= 1')
            if self.masker_val_aug_every < 1:
                raise ValueError('masker_val_aug_every must be >= 1')
        self.masker_select_category = str(self.masker_select_category).lower() if self.masker_select_category else None
        if self.masker_select_category is not None:
            from augmentations.pipeline import MASKER_VAL_CATEGORIES
            if not self.masker_val_aug:
                raise ValueError('masker_select_category requires masker_val_aug=True (the per-category Dice it selects on is produced by the val-aug loaders)')
            if self.masker_select_category not in MASKER_VAL_CATEGORIES:
                raise ValueError(f'masker_select_category must be one of {sorted(MASKER_VAL_CATEGORIES)}, got {self.masker_select_category!r}')
            if self.masker_select_category == 'mp2rage_posterior_fossa' and float(self.synth_kwargs.get('mp2rage_posterior_fossa_fraction', 0.0) or 0.0) <= 0.0:
                raise ValueError("masker_select_category='mp2rage_posterior_fossa' requires a positive synth_kwargs['mp2rage_posterior_fossa_fraction']")
            if self.masker_select_category == 'mp2rage_posterior_fossa':
                print('[warn] posterior-fossa checkpoint selection is synthetic-only and shares the training renderer; prefer masker_select_category=None when no independent real MP2RAGE validation set is available', flush=True)
            if self.select_on_deployed_dice:
                raise ValueError('masker_select_category and select_on_deployed_dice both define checkpoint selection; pass only one (the per-category loaders report RAW Dice)')

    def _configure_loss(self):
        from math import isfinite
        self.loss_kwargs = dict(self.loss_kwargs or {})
        self.model_kwargs = dict(self.model_kwargs or {})
        _raw_loss_profile = self.loss_kwargs.get('profile')
        self.masking_loss_profile = 'legacy' if _raw_loss_profile is None else str(_raw_loss_profile).lower()
        _w_sdt_aux = float(self.loss_kwargs.get('w_sdt_aux', 0.0))
        _w_surface = float(self.loss_kwargs.get('w_surface', 0.0))
        if not isfinite(_w_sdt_aux) or _w_sdt_aux < 0.0:
            raise ValueError('w_sdt_aux must be finite and >= 0')
        if not isfinite(_w_surface) or _w_surface < 0.0:
            raise ValueError('w_surface must be finite and >= 0')
        if self.masking_loss_profile == 'recall-boundary' and _w_surface > 0.0:
            raise ValueError('recall-boundary already supplies a boundary objective; do not combine it with w_surface')
        self.sdt_supervision = _w_sdt_aux > 0.0
        self.mask_dice_selection = self.sdt_supervision or self.masking_loss_profile != 'legacy'
        if self.sdt_supervision:
            if _w_surface > 0.0:
                raise ValueError('Ablate w_sdt_aux and w_surface separately; enabling both is intentionally refused')
            requested_aux = self.model_kwargs.get('sdt_auxiliary')
            if requested_aux is False:
                raise ValueError('w_sdt_aux > 0 requires model_kwargs sdt_auxiliary=True')
            self.model_kwargs['sdt_auxiliary'] = True
            self.loss_kwargs.setdefault('sdt_band_mm', 5.0)
            self.loss_kwargs.setdefault('sdt_far_weight', 0.1)
            self.loss_kwargs.setdefault('sdt_voxel_mm', self.conform_mm or 1.0)
        elif self.model_kwargs.get('sdt_auxiliary'):
            raise ValueError('sdt_auxiliary=True requires loss_kwargs w_sdt_aux > 0')

    def _configure_outputs(self):
        self.overwrite = bool(self.overwrite)
        self.resume_from = Path(self.resume_from) if self.resume_from else None
        self.init_from = Path(self.init_from) if self.init_from else None
        self.init_partial = bool(self.init_partial)
        if self.resume_from is not None and self.init_from is not None:
            raise ValueError('Pass at most one of resume_from (full state) / init_from (weights-only warm-start).')
        self.train_state_path = self.model_out_path.with_name(self.model_out_path.stem + '.train_state.pt')

    def to_dict(self):
        """JSON-ready, independent copy of all inputs for a resolved run."""
        result = {item.name: _json_value(getattr(self, item.name)) for item in fields(self)}
        result["device"] = str(self.device)
        return result

    @classmethod
    def resume_settings(cls, settings):
        """Serialize scientific settings, preserving the historical optional-key contract.

        Paths for output, initialization and reporting cadence do not define updates.
        Device may change on resume, but is retained as useful execution metadata.
        """
        excluded = {
            "model_out_path", "results_out_path", "overwrite", "resume_from",
            "init_from", "init_partial", "allow_unsafe_init", "masker_val_aug_every",
            "masker_val_aug", "masker_val_aug_variants", "masker_select_category",
            "saliency_adversarial", "hard_artifact_tail_identity", "hardtail_thread_limit_identity",
        }
        cfg = {item.name: _json_value(getattr(settings, item.name))
               for item in fields(cls) if item.name not in excluded}
        cfg["device"] = str(settings.device)
        crops = settings.masker_crops_per_volume if settings.synth_online and settings.patch_size else 1
        cfg["effective_batch"] = settings.batch_size * crops * settings.accum_steps
        cfg["augmentation_policy_version"] = (
            12 if any(float(settings.synth_kwargs.get(key, 0.0) or 0.0) > 0.0 for key in (
                "benign_only_mp2rage_fraction", "mixed_mp2rage_fraction", "mp2rage_superset_fraction",
                "mp2rage_lower_feature_fraction", "mp2rage_posterior_fossa_fraction",
                "mp2rage_target_style_fraction")) else 2)
        if settings.masker_val_aug:
            cfg.update(masker_val_aug=True, masker_val_aug_variants=settings.masker_val_aug_variants)
        if settings.masker_select_category is not None:
            cfg["masker_select_category"] = settings.masker_select_category
        if settings.saliency_adversarial is not None:
            cfg["saliency_adversarial"] = copy.deepcopy(settings.saliency_adversarial)
            cfg["saliency_adversarial_identity"] = settings.saliency_adversarial_identity
        if "mixed_mp2rage_fraction" in settings.synth_kwargs:
            cfg.update(domain_mix_policy_version=1, domain_mix_identity="mixed-mp2rage-v1")
        for name in ("mp2rage_target_style_identity", "hard_artifact_tail_identity",
                     "hardtail_thread_limit_identity"):
            if getattr(settings, name) is not None:
                cfg[name] = copy.deepcopy(getattr(settings, name))
        return cfg

    def _configure_sampling(self):
        # Config may be constructed before the engine; bootstrap before the registry
        # imports NumPy, SciPy and Torch through its augmentation implementations.
        from training.runtime import (
            _validate_hardtail_thread_limit_activation,
            _hardtail_identity_contains_path_key,
        )
        from augmentations.sampling_policy import resolve_sampling_policy

        expected_tail = self.hard_artifact_tail_identity
        expected_threads = self.hardtail_thread_limit_identity
        self.mp2rage_target_style_identity = None
        self.hard_artifact_tail_identity = None
        self.hardtail_thread_limit_identity = None
        # Explicit zero is identical to omission, including lazy imports and hashes.
        for key in ('mp2rage_target_style_fraction', 'hard_artifact_tail_fraction'):
            value = self.synth_kwargs.get(key)
            if value is None or float(value) == 0.0:
                self.synth_kwargs.pop(key, None)
        voxel_sizes = (float(self.conform_mm),) * 3 if self.conform_mm is not None else None
        policy = resolve_sampling_policy(
            self.synth_kwargs, voxel_sizes=voxel_sizes, require_geometry=True,
            reject_legacy_curriculum=False,
        )
        keys = ('benign_only_mp2rage_fraction', 'mixed_mp2rage_fraction',
                'mp2rage_superset_fraction', 'mp2rage_lower_feature_fraction',
                'mp2rage_posterior_fossa_fraction', 'mp2rage_target_style_fraction',
                'hard_artifact_tail_fraction')
        enabled_keys = [key for key in keys if key in self.synth_kwargs]
        # These two legacy trainer options require a dedicated route even when zero.
        # The sample-level policy accepts omitted/zero conditionals for ordinary samples.
        for key in ("mp2rage_superset_fraction", "mp2rage_lower_feature_fraction"):
            if key in self.synth_kwargs and (policy.dedicated_fraction or 0.0) <= 0.0:
                raise ValueError(f"{key} needs the dedicated MP2RAGE fraction > 0")
        if enabled_keys and not self.synth_online:
            raise ValueError(f'{enabled_keys[0]} is valid only for online masking training')
        for key in enabled_keys:
            self.synth_kwargs[key] = float(self.synth_kwargs[key])
        if (policy.dedicated_fraction or 0.0) > 0 and self.conform_mm is None:
            raise ValueError('policy-12 dedicated MP2RAGE curriculum requires conform_mm: '
                             'its independent posterior-local anatomy mode needs a canonical RAS physical grid')
        if policy.target_style_fraction > 0:
            from augmentations.curricula import mp2rage_target_stationary_v6 as stationary_v6
            self.mp2rage_target_style_identity = copy.deepcopy(
                stationary_v6.stationary_v6_identity_metadata())
            if _hardtail_identity_contains_path_key(self.mp2rage_target_style_identity):
                raise RuntimeError('MP2RAGE stationary-v6 resume identity must not contain paths')
        if policy.hard_tail_fraction > 0:
            identity, _evidence = _validate_hardtail_thread_limit_activation(self.num_workers)
            if expected_threads is not None and expected_threads != identity:
                raise ValueError('hardtail_thread_limit_identity does not match the live frozen thread-limit identity')
            self.hardtail_thread_limit_identity = copy.deepcopy(identity)
            from augmentations.curricula import hard_artifact_tail as hard_tail_v1
            identity = hard_tail_v1.hard_artifact_tail_v1_identity_metadata()
            if (not isinstance(identity, dict) or identity.get('id') != 'masker-hard-artifact-tail-v1'
                    or identity.get('profile_schema_version') != 5):
                raise RuntimeError('positive hard_artifact_tail_fraction requires the live schema-5 hard-artifact-tail identity')
            if expected_tail is not None and expected_tail != identity:
                raise ValueError('hard_artifact_tail_identity does not match the live schema-5 hard-tail identity')
            self.hard_artifact_tail_identity = copy.deepcopy(identity)
            for identity in (self.hard_artifact_tail_identity, self.hardtail_thread_limit_identity):
                if _hardtail_identity_contains_path_key(identity):
                    raise RuntimeError('hard-artifact-tail resume identity must not contain paths')
        elif expected_tail is not None or expected_threads is not None:
            name = 'hard_artifact_tail_identity' if expected_tail is not None else 'hardtail_thread_limit_identity'
            raise ValueError(f'{name} is valid only when hard_artifact_tail_fraction is positive')
        if (policy.dedicated_fraction or 0.0) > 0:
            print(f'[online-masker] MP2RAGE policy 12: {100 * policy.dedicated_fraction:.1f}% dedicated; '
                  f'target stationary-v6={100 * policy.target_style_fraction:.1f}% of MP2RAGE; '
                  f'hard-artifact tail={100 * policy.hard_tail_fraction:.1f}% of eligible draws.', flush=True)


def _json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return copy.deepcopy(value)
