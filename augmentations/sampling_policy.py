"""Shared, deterministic validation for the online masking curricula.

Resolving a policy never consumes random numbers. An omitted dedicated fraction
and an explicit zero are different: zero still selects that curriculum's base.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping


SAMPLING_DEFAULTS = {
    "p_clean": 0.18, "p_standard": 0.22, "p_benign": 0.30,
    "p_artifact": 0.30, "legacy_anchor_augmentation": False,
    "benign_only_mp2rage_fraction": None, "mixed_mp2rage_fraction": None,
    "mp2rage_superset_fraction": 0.0, "mp2rage_lower_feature_fraction": 0.0,
    "mp2rage_posterior_fossa_fraction": 0.0, "mp2rage_target_style_fraction": 0.0,
    "hard_artifact_tail_fraction": 0.0,
}

CONDITIONAL_MP2RAGE_KEYS = (
    "mp2rage_superset_fraction", "mp2rage_lower_feature_fraction",
    "mp2rage_posterior_fossa_fraction", "mp2rage_target_style_fraction",
)
MIXED_NON_MP2_BASE_SHARES = {
    "clean": 0.15, "other_standard": 0.15,
    "broad_benign": 0.35, "label_synth": 0.35,
}


@dataclass(frozen=True)
class SamplingPolicy:
    benign_only: bool
    mixed_domain: bool
    dedicated_fraction: float | None
    superset_fraction: float
    lower_feature_fraction: float
    posterior_fraction: float
    target_style_fraction: float
    hard_tail_fraction: float
    p_clean: float
    p_standard: float
    p_benign: float
    p_artifact: float
    legacy_anchors: bool
    target_style_voxel_sizes: tuple | None = None
    hard_tail_voxel_sizes: tuple | None = None


def _fraction(value, name, maximum=1.0):
    value = float(value or 0.0)
    if not isfinite(value) or not 0.0 <= value <= maximum:
        raise ValueError(f"{name} must be finite and in [0, {maximum:g}]")
    return value


def resolve_sampling_policy(
    sampling: Mapping, *, voxel_sizes=None, require_geometry=False,
    reject_legacy_curriculum=False,
) -> SamplingPolicy:
    """Validate route shares, exposure caps and optional physical-grid checks.

    JSON and Trainer callers reject legacy anchors with either dedicated
    curriculum. Direct legacy sampling keeps its historical benign-only behavior.
    Grid checks are requested at the boundary that owns physical spacing.
    """
    values = {**SAMPLING_DEFAULTS, **sampling}
    benign = values["benign_only_mp2rage_fraction"] is not None
    mixed = values["mixed_mp2rage_fraction"] is not None
    if benign and mixed:
        raise ValueError(
            "mixed_mp2rage_fraction and benign_only_mp2rage_fraction are mutually exclusive")
    dedicated_key = "mixed_mp2rage_fraction" if mixed else "benign_only_mp2rage_fraction"
    dedicated = _fraction(values[dedicated_key], dedicated_key) if benign or mixed else None
    legacy = bool(values["legacy_anchor_augmentation"])
    if legacy and (mixed or (benign and reject_legacy_curriculum)):
        raise ValueError("legacy_anchor_augmentation conflicts with dedicated MP2RAGE curricula")
    conditional = [
        _fraction(values[key], key, 0.10 if "target_style" in key else 1.0)
        for key in CONDITIONAL_MP2RAGE_KEYS
    ]
    if sum(conditional) > 1.0 + 1e-12:
        raise ValueError(" + ".join(CONDITIONAL_MP2RAGE_KEYS) + " must be <= 1")
    if any(conditional) and (dedicated is None or dedicated <= 0.0):
        raise ValueError("MP2RAGE conditional fractions require a positive dedicated curriculum fraction")
    target = conditional[-1]
    if target > 0.0 and dedicated * target > 0.035 + 1e-12:
        raise ValueError("MP2RAGE target-style global exposure must be <= 0.035")
    hard = _fraction(values["hard_artifact_tail_fraction"], "hard_artifact_tail_fraction", 0.10)
    if hard > 0.0:
        if not mixed or not 0.0 < dedicated < 1.0:
            raise ValueError("hard_artifact_tail_fraction > 0 requires mixed_mp2rage_fraction in (0, 1)")
        eligible = (1.0 - dedicated) * (
            MIXED_NON_MP2_BASE_SHARES["broad_benign"] + MIXED_NON_MP2_BASE_SHARES["label_synth"])
        if eligible * hard > 0.05 + 1e-12:
            raise ValueError("hard-artifact-tail global exposure must be <= 0.05")
    spacing = None
    if require_geometry and (target > 0.0 or hard > 0.0):
        try:
            spacing = tuple(float(value) for value in voxel_sizes)
        except (TypeError, ValueError):
            pass
        if target > 0.0 and spacing != (1.0, 1.0, 1.0):
            raise ValueError("mp2rage_target_style_fraction > 0 requires voxel_sizes exactly (1, 1, 1)")
        if hard > 0.0 and (spacing is None or len(spacing) != 3
                          or not all(isfinite(value) and value > 0.0 for value in spacing)):
            raise ValueError("hard_artifact_tail_fraction > 0 requires three finite positive voxel_sizes")
    return SamplingPolicy(
        benign, mixed, dedicated, *conditional, hard,
        *(float(values[key]) for key in ("p_clean", "p_standard", "p_benign", "p_artifact")),
        legacy, spacing if target > 0.0 else None, spacing if hard > 0.0 else None,
    )
