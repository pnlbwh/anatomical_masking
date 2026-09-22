"""Posterior-fossa stress package for synthetic MP2RAGE/UNIT1 training.

This module deliberately separates the supervision contour from the apparent
inferior/posterior acquisition boundary.  The mask is low-pass filtered into a
coarse intracranial proxy and that proxy is displaced only through a broad,
jittered scanner-space envelope.  This avoids rendering sulcal/folial answer detail
while still targeting the part of MP2RAGE that is most vulnerable to B1+ and
inversion-efficiency variation.

The public curriculum exposes only a conditional fraction.  Parameter ranges remain
versioned here until they can be calibrated on a target protocol.  A sampled spec is
plain JSON data and therefore can be logged/replayed by QA tools.

Axis contract
-------------
Inputs are three-dimensional arrays on the canonical RAS grid produced by
``nibabel.processing.conform``: axis 0 increases Right, axis 1 increases Anterior,
and axis 2 increases Superior.  Callers without that contract must not advertise an
anatomical "posterior fossa" augmentations.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy import ndimage as ndi


POSTERIOR_FOSSA_SPEC_VERSION = 7

# Policy-12 posterior-fossa mixture.  The phenotype name is intentionally not
# serialized: every rendering-relevant value is already explicit in the scalar
# spec below, which is sufficient for exact replay without expanding the schema.
POSTERIOR_FOSSA_PHENOTYPE_WEIGHTS = (
    ("sharp_balanced", 0.20),
    ("sharp_grainy_exterior", 0.15),
    ("dark_nonstationary_texture", 0.25),
    ("vague_low_snr", 0.30),
    ("directed_fn_adversary", 0.10),
)

# Only one fifth of vague draws use the deliberately severe PSF endpoint.  Because
# vague itself is 30% of posterior-fossa draws, this freezes the severe tail at 6%
# of posterior-fossa samples without adding a redundant serialized subtype key.
_VAGUE_SEVERE_FRACTION = 0.20

_SPEC_KEYS_V2 = frozenset({
    "version", "seed", "field_strength",
    "field_center_frac_ras", "field_fwhm_mm_ras",
    "contrast_scale", "offset_dynamic_range",
    "bridge_center_frac_ras", "bridge_fwhm_mm_ras", "bridge_signal_scale",
    "snr_ratio", "noise_correlation_mm",
    "psf_fwhm_mm_ras",
    "boundary_center_frac_ras", "boundary_field_fwhm_mm_ras",
    "boundary_blur_fwhm_mm_ras", "boundary_mix",
    "boundary_texture_retention", "boundary_contrast_scale",
    "tentorium_on", "tentorium_level", "tentorium_blend",
    "tentorium_thickness_mm", "tentorium_base_s_frac",
    "tentorium_slope_r", "tentorium_slope_a", "tentorium_curvature",
    "pose_rotation_deg_ras", "pose_scale_ras", "pose_shift_mm_ras",
    "inferior_margin_mm",
})

_SPEC_KEYS_V3 = _SPEC_KEYS_V2 | frozenset({
    # The visible MP2RAGE acquisition-law transition must not remain exactly
    # coincident with the training annotation.  These values describe a smooth,
    # bounded posterior displacement of that transition; they do not move the
    # anatomy or the returned label.
    "render_boundary_bias_mm", "render_boundary_jitter_mm",
    "render_boundary_correlation_mm",
    # Preserve/enhance the fine cerebellar folial residual after the local PSF.
    "folial_detail_gain",
})

_SPEC_KEYS_V4 = _SPEC_KEYS_V3 | frozenset({
    # Low-pass the label before it becomes an acquisition-law proxy.  This removes
    # sulcal/folial answer detail while preserving a plausible intracranial support.
    "render_boundary_coarsen_mm",
})

_SPEC_KEYS_V5 = _SPEC_KEYS_V4 | frozenset({
    # Fraction of the contracted-but-labelled band moved toward the same complex
    # ratio-noise law as its surroundings.  This is separate from scalp rendering.
    "render_boundary_floor_mix",
})

# V6 changes how the floor is spatially applied (omitted-label feather rather than
# an explicit exterior shell) without changing the serialized scalar schema. V7
# explicitly versions the continuous coarse-support guard and ambiguity-confidence
# semantics introduced by policy 12.
_SPEC_KEYS_V6 = _SPEC_KEYS_V5
_SPEC_KEYS = _SPEC_KEYS_V6 | frozenset({
    # Physical outward dilation of the coarse/displaced acquisition support before
    # it excludes exterior air/scalp. It never changes the returned supervision.
    "acquisition_guard_dilation_mm",
    # Physical half-width of the continuous air/scalp transition around that
    # guard (the full transition spans twice this value).
    "acquisition_guard_feather_mm",
    # Physical half-width override for the coarse tissue-rendering transition.
    # Zero delegates to the renderer's legacy boundary half-width.
    "acquisition_transition_width_mm",
    # Minimum image-derived coherence confidence inside the broad scanner-space
    # ambiguity field. Zero exactly preserves the v6 confidence response.
    "boundary_confidence_floor",
})

_V2_UPGRADE_DEFAULTS = {
    "render_boundary_bias_mm": 0.0,
    "render_boundary_jitter_mm": 0.0,
    "render_boundary_correlation_mm": 24.0,
    "folial_detail_gain": 0.0,
}

_V3_UPGRADE_DEFAULTS = {
    "render_boundary_coarsen_mm": 0.0,
}

_V4_UPGRADE_DEFAULTS = {
    "render_boundary_floor_mix": 0.0,
}

_V6_UPGRADE_DEFAULTS = {
    "acquisition_guard_dilation_mm": 0.0,
    "acquisition_guard_feather_mm": 0.0,
    "acquisition_transition_width_mm": 0.0,
    "boundary_confidence_floor": 0.0,
}


def _generator(rng):
    return np.random.default_rng(rng)


def _triple(values, name, *, finite=True):
    values = tuple(float(v) for v in values)
    if len(values) != 3:
        raise ValueError(f"posterior-fossa {name} must contain three RAS values")
    if finite and not np.isfinite(values).all():
        raise ValueError(f"posterior-fossa {name} must be finite")
    return values


def _spacing(voxel_sizes):
    if voxel_sizes is None:
        return (1.0, 1.0, 1.0)
    spacing = _triple(voxel_sizes, "voxel_sizes")
    if any(v <= 0.0 for v in spacing):
        raise ValueError("posterior-fossa voxel_sizes must be > 0")
    return spacing


def _posterior_fossa_phenotype_from_u(value):
    """Map a unit-uniform draw onto the frozen policy-12 mixture."""
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value < 1.0:
        raise ValueError("posterior-fossa phenotype draw must be in [0, 1)")
    cumulative = 0.0
    for name, weight in POSTERIOR_FOSSA_PHENOTYPE_WEIGHTS:
        cumulative += weight
        # Decimal policy boundaries (not binary float addition artefacts) define
        # the categorical intervals, e.g. exactly 0.90 starts the 10% tail.
        if value < round(cumulative, 12):
            return name
    # Protect against floating-point summation without allowing a silent gap.
    return POSTERIOR_FOSSA_PHENOTYPE_WEIGHTS[-1][0]


def sample_posterior_fossa_spec(
        rng, *, field_strength, voxel_sizes=None, phenotype=None):
    """Sample one policy-12 posterior-fossa acquisition phenotype.

    The five-profile mixture spans the independent real-reference appearances
    instead of centring the curriculum on one difficult volume.  ``phenotype`` is
    an optional deterministic QA hook; ordinary training leaves it as ``None`` and
    obtains the frozen 20/15/25/30/10 percent mixture.  The profile name is not
    serialized because all rendering-relevant scalars and the realization seed are
    explicit in the returned JSON-compatible spec.

    Only ``directed_fn_adversary`` contracts the label-derived acquisition support
    or applies its signal floor.  Tentorium is a separate 25-percent Bernoulli draw
    made before any profile-specific parameters, giving it the same independent
    probability for every phenotype.
    """
    del voxel_sizes  # all physical ranges below are in millimetres
    g = _generator(rng)
    field_strength = float(field_strength)
    if not np.isfinite(field_strength) or field_strength <= 0.0:
        raise ValueError("posterior-fossa field_strength must be finite and > 0")

    names = {name for name, _weight in POSTERIOR_FOSSA_PHENOTYPE_WEIGHTS}
    # Consume the selector even for a forced profile.  This keeps the subsequent
    # tentorium coin and scalar stream directly comparable in deterministic QA.
    sampled_phenotype = _posterior_fossa_phenotype_from_u(float(g.random()))
    if phenotype is None:
        phenotype = sampled_phenotype
    elif not isinstance(phenotype, str) or phenotype not in names:
        raise ValueError(
            "posterior-fossa phenotype must be one of "
            + ", ".join(name for name, _weight in POSTERIOR_FOSSA_PHENOTYPE_WEIGHTS))
    tentorium_on = bool(g.random() < 0.25)

    # Profile-specific ranges are deliberately separated along interpretable
    # dimensions.  The two sharp anchors together remain 35% of posterior draws;
    # the dark and vague modes retain texture rather than becoming homogeneous
    # posterior blackouts.
    if phenotype == "sharp_balanced":
        contrast_scale = float(g.uniform(0.90, 1.00))
        offset_dynamic_range = float(g.uniform(-0.05, 0.08))
        snr_ratio = float(g.uniform(0.78, 1.00))
        noise_correlation_mm = float(g.uniform(0.70, 1.60))
        psf_base_range, psf_partition_range = (0.75, 1.05), (0.90, 1.35)
        boundary_base_range = (0.80, 1.45)
        boundary_anisotropy_range = (1.00, 1.20)
        boundary_mix = float(g.uniform(0.12, 0.32))
        boundary_texture_retention = float(g.uniform(0.88, 0.99))
        boundary_contrast_scale = float(g.uniform(0.86, 0.99))
        folial_detail_gain = float(g.uniform(0.35, 0.65))
        bridge_signal_scale = float(g.uniform(0.78, 0.98))
        field_fwhm_ranges = ((55.0, 100.0), (55.0, 100.0), (50.0, 95.0))
        field_center_ranges = ((0.42, 0.58), (0.16, 0.34), (0.10, 0.26))
        render_boundary_jitter_mm = float(g.uniform(0.25, 0.80))
        render_boundary_correlation_mm = float(g.uniform(18.0, 34.0))
        render_boundary_coarsen_mm = float(g.uniform(3.0, 6.0))
        acquisition_guard_dilation_mm = float(g.uniform(1.0, 2.0))
        acquisition_guard_feather_mm = float(g.uniform(1.5, 2.5))
        acquisition_transition_width_mm = float(g.uniform(2.0, 3.0))
        boundary_confidence_floor = float(g.uniform(0.05, 0.15))
    elif phenotype == "sharp_grainy_exterior":
        # The correlated grain and scalp/exterior law are completed by the site
        # renderer.  Here we keep the cerebellar signal sharp and reduce local SNR.
        contrast_scale = float(g.uniform(0.84, 0.98))
        offset_dynamic_range = float(g.uniform(-0.06, 0.08))
        snr_ratio = float(g.uniform(0.55, 0.78))
        noise_correlation_mm = float(g.uniform(0.25, 0.65))
        psf_base_range, psf_partition_range = (0.70, 1.00), (0.85, 1.30)
        boundary_base_range = (0.75, 1.35)
        boundary_anisotropy_range = (1.00, 1.18)
        boundary_mix = float(g.uniform(0.12, 0.34))
        boundary_texture_retention = float(g.uniform(0.90, 1.00))
        boundary_contrast_scale = float(g.uniform(0.82, 0.98))
        folial_detail_gain = float(g.uniform(0.65, 0.95))
        bridge_signal_scale = float(g.uniform(0.72, 0.95))
        field_fwhm_ranges = ((50.0, 95.0), (50.0, 95.0), (45.0, 90.0))
        field_center_ranges = ((0.40, 0.60), (0.15, 0.34), (0.09, 0.26))
        render_boundary_jitter_mm = float(g.uniform(0.30, 0.90))
        render_boundary_correlation_mm = float(g.uniform(14.0, 26.0))
        render_boundary_coarsen_mm = float(g.uniform(3.0, 5.0))
        acquisition_guard_dilation_mm = float(g.uniform(1.5, 2.5))
        acquisition_guard_feather_mm = float(g.uniform(2.0, 3.0))
        acquisition_transition_width_mm = float(g.uniform(2.5, 3.5))
        boundary_confidence_floor = float(g.uniform(0.10, 0.25))
    elif phenotype == "dark_nonstationary_texture":
        # The independent sagittal references contain an equal/darker PF mode
        # that the bright raw-UNIT1 anchor cannot represent by itself.  Use a
        # broad field and preserve fine stochastic/folial texture instead of
        # producing that mode through a narrow blackout or uniform blur.
        contrast_scale = float(g.uniform(0.60, 0.84))
        offset_dynamic_range = -float(g.uniform(0.14, 0.30))
        # The equal/darker references retain substantially more fine texture
        # than a conventional smooth UNIT1 render.  Put the added energy below
        # one millimetre instead of compensating with deterministic folial
        # sharpening, which otherwise creates etched synthetic fissures.
        snr_ratio = float(g.uniform(0.36, 0.62))
        # Keep this component near-iid at 1-mm resolution.  The raw UNIT1's
        # sub-0.6-mm residual is much stronger and less autocorrelated than the
        # former .35--.75 mm field.  Shortening correlation shifts the same
        # bounded noise variance into the missing fine band; it does not increase
        # total variance when this phenotype independently meets a grainy site.
        noise_correlation_mm = float(g.uniform(0.15, 0.45))
        psf_base_range, psf_partition_range = (0.80, 1.20), (1.00, 1.55)
        boundary_base_range = (1.30, 2.30)
        boundary_anisotropy_range = (1.00, 1.20)
        boundary_mix = float(g.uniform(0.35, 0.60))
        boundary_texture_retention = float(g.uniform(0.80, 0.98))
        boundary_contrast_scale = float(g.uniform(0.62, 0.85))
        folial_detail_gain = float(g.uniform(0.45, 0.80))
        bridge_signal_scale = float(g.uniform(0.65, 0.90))
        field_fwhm_ranges = ((45.0, 90.0), (60.0, 100.0), (55.0, 95.0))
        field_center_ranges = ((0.36, 0.64), (0.13, 0.32), (0.07, 0.25))
        render_boundary_jitter_mm = float(g.uniform(0.80, 1.50))
        render_boundary_correlation_mm = float(g.uniform(18.0, 32.0))
        render_boundary_coarsen_mm = float(g.uniform(6.0, 9.0))
        acquisition_guard_dilation_mm = float(g.uniform(2.0, 3.5))
        acquisition_guard_feather_mm = float(g.uniform(2.5, 4.0))
        acquisition_transition_width_mm = float(g.uniform(3.0, 4.5))
        boundary_confidence_floor = float(g.uniform(0.20, 0.40))
    elif phenotype == "vague_low_snr":
        # Policy 11's seed-202-like endpoint remained easier to classify than the
        # raw UNIT1 interface. Policy 12 uses guard-led ambiguity with a
        # detail-preserving majority plus a rare severe-PSF tail, symmetric
        # coarse-support jitter and an image-derived confidence floor, without
        # contracting the target label.
        severe_vague = bool(g.random() < _VAGUE_SEVERE_FRACTION)
        contrast_scale = float(g.uniform(0.55, 0.80))
        offset_dynamic_range = -float(g.uniform(0.08, 0.22))
        snr_ratio = float(g.uniform(0.40, 0.68))
        noise_correlation_mm = float(g.uniform(0.45, 1.50))
        if severe_vague:
            # Rare endpoint covering the strongest observed loss of folial detail.
            psf_base_range, psf_partition_range = (1.70, 2.80), (2.40, 4.00)
            boundary_base_range = (2.00, 3.40)
            boundary_anisotropy_range = (1.00, 1.18)
            boundary_mix = float(g.uniform(0.65, 0.95))
            boundary_contrast_range = (0.55, 0.80)
            folial_detail_gain = float(g.uniform(0.05, 0.30))
        else:
            # Default vague mode: ambiguous interface with faint folia and
            # mid-scale anatomy still visible.  The ranges contain the calibrated
            # (1.1, 1.6, 1.1)-mm PSF / 1.8--2.0-mm boundary candidate.
            psf_base_range, psf_partition_range = (0.90, 1.60), (1.20, 2.40)
            boundary_base_range = (1.50, 2.30)
            boundary_anisotropy_range = (1.00, 1.13)
            boundary_mix = float(g.uniform(0.60, 0.90))
            boundary_contrast_range = (0.58, 0.80)
            folial_detail_gain = float(g.uniform(0.30, 0.60))
        boundary_texture_retention = float(g.uniform(0.70, 0.95))
        boundary_contrast_scale = float(g.uniform(*boundary_contrast_range))
        bridge_signal_scale = float(g.uniform(0.58, 0.86))
        field_fwhm_ranges = ((30.0, 95.0), (50.0, 95.0), (45.0, 95.0))
        field_center_ranges = ((0.36, 0.64), (0.12, 0.33), (0.06, 0.30))
        render_boundary_jitter_mm = float(g.uniform(1.0, 2.0))
        render_boundary_correlation_mm = float(g.uniform(16.0, 30.0))
        render_boundary_coarsen_mm = float(g.uniform(8.0, 12.0))
        acquisition_guard_dilation_mm = float(g.uniform(3.0, 5.0))
        acquisition_guard_feather_mm = float(g.uniform(3.0, 5.0))
        acquisition_transition_width_mm = float(g.uniform(3.5, 5.5))
        boundary_confidence_floor = float(g.uniform(0.35, 0.65))
    else:  # directed_fn_adversary
        # This is the only label-derived false-negative endpoint.  It is kept
        # sharp and rare so the network learns recall rather than a blur shortcut.
        contrast_scale = float(g.uniform(0.84, 0.98))
        offset_dynamic_range = -float(g.uniform(0.06, 0.16))
        snr_ratio = float(g.uniform(0.62, 0.88))
        noise_correlation_mm = float(g.uniform(0.45, 1.20))
        psf_base_range, psf_partition_range = (0.70, 1.00), (0.85, 1.35)
        boundary_base_range = (0.80, 1.40)
        boundary_anisotropy_range = (1.00, 1.15)
        boundary_mix = float(g.uniform(0.00, 0.12))
        boundary_texture_retention = float(g.uniform(0.90, 1.00))
        boundary_contrast_scale = float(g.uniform(0.88, 0.99))
        folial_detail_gain = float(g.uniform(0.55, 0.95))
        bridge_signal_scale = float(g.uniform(0.65, 0.90))
        field_fwhm_ranges = ((65.0, 120.0), (60.0, 100.0), (55.0, 95.0))
        field_center_ranges = ((0.40, 0.60), (0.14, 0.34), (0.08, 0.26))
        render_boundary_jitter_mm = float(g.uniform(0.40, 1.30))
        render_boundary_correlation_mm = float(g.uniform(16.0, 30.0))
        render_boundary_coarsen_mm = float(g.uniform(8.0, 14.0))
        # Cover the maximum 5.5-mm directed contraction without consulting the
        # exact target contour in the exterior-law compositor.
        acquisition_guard_dilation_mm = float(g.uniform(5.5, 7.5))
        acquisition_guard_feather_mm = float(g.uniform(3.0, 5.0))
        acquisition_transition_width_mm = float(g.uniform(3.0, 5.0))
        boundary_confidence_floor = float(g.uniform(0.05, 0.20))

    partition_axis = int(g.choice(3, p=(0.20, 0.25, 0.55)))
    psf = [float(g.uniform(*psf_base_range)) for _ in range(3)]
    psf[partition_axis] = float(g.uniform(*psf_partition_range))
    boundary_partition = int(g.choice(3, p=(0.20, 0.25, 0.55)))
    boundary_base = float(g.uniform(*boundary_base_range))
    boundary_blur = [boundary_base] * 3
    boundary_blur[boundary_partition] *= float(g.uniform(
        *boundary_anisotropy_range))

    directed = phenotype == "directed_fn_adversary"
    render_boundary_bias_mm = (
        -float(g.uniform(2.5, 5.5)) if directed else 0.0)
    render_boundary_floor_mix = (
        float(g.uniform(0.15, 0.35)) if directed else 0.0)

    def draw_triple(ranges):
        return tuple(float(g.uniform(*bounds)) for bounds in ranges)

    seed = int(g.integers(0, np.iinfo(np.int32).max))
    spec = {
        "version": POSTERIOR_FOSSA_SPEC_VERSION,
        "seed": seed,
        "field_strength": field_strength,
        "field_center_frac_ras": draw_triple(field_center_ranges),
        "field_fwhm_mm_ras": draw_triple(field_fwhm_ranges),
        "contrast_scale": contrast_scale,
        "offset_dynamic_range": offset_dynamic_range,
        "bridge_center_frac_ras": (
            float(g.uniform(0.44, 0.56)), float(g.uniform(0.28, 0.52)),
            float(g.uniform(0.18, 0.36))),
        "bridge_fwhm_mm_ras": (
            float(g.uniform(30.0, 65.0)), float(g.uniform(28.0, 58.0)),
            float(g.uniform(28.0, 60.0))),
        "bridge_signal_scale": bridge_signal_scale,
        "snr_ratio": snr_ratio,
        "noise_correlation_mm": noise_correlation_mm,
        "psf_fwhm_mm_ras": tuple(psf),
        "boundary_center_frac_ras": (
            float(g.uniform(0.44, 0.56)), float(g.uniform(0.14, 0.30)),
            float(g.uniform(0.10, 0.26))),
        "boundary_field_fwhm_mm_ras": (
            float(g.uniform(75.0, 120.0)), float(g.uniform(35.0, 65.0)),
            float(g.uniform(30.0, 60.0))),
        "boundary_blur_fwhm_mm_ras": tuple(boundary_blur),
        "boundary_mix": boundary_mix,
        "boundary_texture_retention": boundary_texture_retention,
        "boundary_contrast_scale": boundary_contrast_scale,
        "render_boundary_bias_mm": render_boundary_bias_mm,
        "render_boundary_jitter_mm": render_boundary_jitter_mm,
        "render_boundary_correlation_mm": render_boundary_correlation_mm,
        "render_boundary_coarsen_mm": render_boundary_coarsen_mm,
        "render_boundary_floor_mix": render_boundary_floor_mix,
        "folial_detail_gain": folial_detail_gain,
        "acquisition_guard_dilation_mm": acquisition_guard_dilation_mm,
        "acquisition_guard_feather_mm": acquisition_guard_feather_mm,
        "acquisition_transition_width_mm": acquisition_transition_width_mm,
        "boundary_confidence_floor": boundary_confidence_floor,
        "tentorium_on": tentorium_on,
        "tentorium_level": float(g.uniform(0.30, 0.70)),
        "tentorium_blend": float(g.uniform(0.30, 0.75)),
        "tentorium_thickness_mm": float(g.uniform(0.5, 1.5)),
        "tentorium_base_s_frac": float(g.uniform(0.32, 0.44)),
        "tentorium_slope_r": float(g.uniform(-0.12, 0.12)),
        "tentorium_slope_a": float(g.uniform(-0.08, 0.08)),
        "tentorium_curvature": float(g.uniform(-0.10, 0.10)),
        "pose_rotation_deg_ras": tuple(
            float(g.uniform(-10.0, 10.0)) for _ in range(3)),
        "pose_scale_ras": tuple(float(g.uniform(0.96, 1.04)) for _ in range(3)),
        "pose_shift_mm_ras": (
            float(g.uniform(-6.0, 6.0)), float(g.uniform(-6.0, 6.0)), 0.0),
        "inferior_margin_mm": float(g.uniform(4.0, 24.0)),
    }
    return validate_posterior_fossa_spec(spec)


def validate_posterior_fossa_spec(spec):
    """Validate and normalize a sampled/replayed posterior-fossa spec."""
    if not isinstance(spec, Mapping):
        raise TypeError("posterior_fossa must be a mapping")
    try:
        source_version = int(spec.get("version", -1))
    except (TypeError, ValueError):
        source_version = -1
    if source_version == 2:
        expected_keys = _SPEC_KEYS_V2
    elif source_version == 3:
        expected_keys = _SPEC_KEYS_V3
    elif source_version == 4:
        expected_keys = _SPEC_KEYS_V4
    elif source_version == 5:
        expected_keys = _SPEC_KEYS_V5
    elif source_version == 6:
        expected_keys = _SPEC_KEYS_V6
    elif source_version == POSTERIOR_FOSSA_SPEC_VERSION:
        expected_keys = _SPEC_KEYS
    else:
        raise ValueError(
            f"unsupported posterior-fossa spec version {spec.get('version')!r}; "
            f"expected 2, 3, 4, 5, 6, or {POSTERIOR_FOSSA_SPEC_VERSION}")
    unknown = set(spec) - expected_keys
    missing = expected_keys - set(spec)
    if unknown or missing:
        detail = []
        if unknown:
            detail.append("unknown=" + ",".join(sorted(map(str, unknown))))
        if missing:
            detail.append("missing=" + ",".join(sorted(map(str, missing))))
        raise ValueError("invalid posterior-fossa spec (" + "; ".join(detail) + ")")
    if source_version == 5:
        # V6 intentionally changed how a non-zero floor is spatially applied and
        # how omitted labelled tissue is protected from scalp transfer.  Relabeling
        # a v5 mapping as v6 would therefore claim a pixel replay that is impossible
        # under this implementation.  Saved v5 NIfTIs remain the exact artifact;
        # rerendering requires the original v5 renderer.
        raise ValueError(
            "posterior-fossa spec version 5 cannot be replayed by the v6 renderer; "
            "use the saved NIfTI or the original v5 code")
    out = dict(spec)
    if source_version == 2:
        # Old QA manifests remain loadable and deterministic under their neutral
        # support defaults; byte-identical historical pixels remain available in
        # the original saved NIfTI artifacts.
        out.update(_V2_UPGRADE_DEFAULTS)
        out.update(_V3_UPGRADE_DEFAULTS)
        out.update(_V4_UPGRADE_DEFAULTS)
        out.update(_V6_UPGRADE_DEFAULTS)
    elif source_version == 3:
        out.update(_V3_UPGRADE_DEFAULTS)
        out.update(_V4_UPGRADE_DEFAULTS)
        out.update(_V6_UPGRADE_DEFAULTS)
    elif source_version == 4:
        out.update(_V4_UPGRADE_DEFAULTS)
        out.update(_V6_UPGRADE_DEFAULTS)
    elif source_version == 6:
        # V7 composes a continuous guard from the coarse acquisition support.
        # All-zero fields dispatch the renderer's exact v6 path so a v6 JSON spec
        # remains pixel-replayable rather than merely schema-compatible.
        out.update(_V6_UPGRADE_DEFAULTS)
    out["version"] = POSTERIOR_FOSSA_SPEC_VERSION
    out["seed"] = int(out["seed"])
    if out["seed"] < 0 or out["seed"] > np.iinfo(np.int64).max:
        raise ValueError("posterior-fossa seed must be in [0, int64_max]")
    out["field_strength"] = float(out["field_strength"])
    if (not np.isfinite(out["field_strength"])
            or not 0.0 < out["field_strength"] <= 20.0):
        raise ValueError("posterior-fossa field_strength must be finite and in (0, 20]")
    for key in (
            "field_center_frac_ras", "field_fwhm_mm_ras", "bridge_center_frac_ras",
            "bridge_fwhm_mm_ras", "psf_fwhm_mm_ras", "boundary_center_frac_ras",
            "boundary_field_fwhm_mm_ras", "boundary_blur_fwhm_mm_ras",
            "pose_rotation_deg_ras", "pose_scale_ras", "pose_shift_mm_ras"):
        out[key] = _triple(out[key], key)
    for key in (
            "contrast_scale", "offset_dynamic_range", "bridge_signal_scale", "snr_ratio",
            "noise_correlation_mm", "tentorium_level", "tentorium_blend",
            "tentorium_thickness_mm", "tentorium_base_s_frac", "tentorium_slope_r",
            "tentorium_slope_a", "tentorium_curvature",
            "inferior_margin_mm", "boundary_mix", "boundary_texture_retention",
            "boundary_contrast_scale", "render_boundary_bias_mm",
            "render_boundary_jitter_mm", "render_boundary_correlation_mm",
            "render_boundary_coarsen_mm", "render_boundary_floor_mix",
            "folial_detail_gain", "acquisition_guard_dilation_mm",
            "acquisition_guard_feather_mm", "acquisition_transition_width_mm",
            "boundary_confidence_floor"):
        out[key] = float(out[key])
        if not np.isfinite(out[key]):
            raise ValueError(f"posterior-fossa {key} must be finite")
    if not isinstance(out["tentorium_on"], (bool, np.bool_, int, np.integer)) \
            or int(out["tentorium_on"]) not in (0, 1):
        raise ValueError("posterior-fossa tentorium_on must be boolean")
    out["tentorium_on"] = bool(out["tentorium_on"])
    for key in ("field_center_frac_ras", "bridge_center_frac_ras",
                "boundary_center_frac_ras"):
        if any(not 0.0 <= v <= 1.0 for v in out[key]):
            raise ValueError(f"posterior-fossa {key} values must be in [0, 1]")
    for key in ("field_fwhm_mm_ras", "bridge_fwhm_mm_ras",
                "boundary_field_fwhm_mm_ras"):
        if any(not 5.0 <= v <= 250.0 for v in out[key]):
            raise ValueError(f"posterior-fossa {key} values must be in [5, 250] mm")
    if any(not 0.1 <= v <= 10.0 for v in out["psf_fwhm_mm_ras"]):
        raise ValueError("posterior-fossa psf_fwhm_mm_ras values must be in [0.1, 10] mm")
    if any(not 0.1 <= v <= 12.0 for v in out["boundary_blur_fwhm_mm_ras"]):
        raise ValueError(
            "posterior-fossa boundary_blur_fwhm_mm_ras values must be in [0.1, 12] mm")
    if any(abs(v) > 30.0 for v in out["pose_rotation_deg_ras"]):
        raise ValueError("posterior-fossa pose rotations must be within +/-30 degrees")
    if any(not 0.75 <= v <= 1.25 for v in out["pose_scale_ras"]):
        raise ValueError("posterior-fossa pose scales must be in [0.75, 1.25]")
    if any(abs(v) > 40.0 for v in out["pose_shift_mm_ras"]):
        raise ValueError("posterior-fossa pose shifts must be within +/-40 mm")
    if not 0.0 < out["contrast_scale"] <= 1.25:
        raise ValueError("posterior-fossa contrast_scale must be in (0, 1.25]")
    if not 0.0 < out["bridge_signal_scale"] <= 1.0:
        raise ValueError("posterior-fossa bridge_signal_scale must be in (0, 1]")
    if not 0.0 < out["snr_ratio"] <= 1.0:
        raise ValueError("posterior-fossa snr_ratio must be in (0, 1]")
    if not 0.0 <= out["boundary_mix"] <= 1.0:
        raise ValueError("posterior-fossa boundary_mix must be in [0, 1]")
    if not 0.0 <= out["boundary_texture_retention"] <= 1.0:
        raise ValueError("posterior-fossa boundary_texture_retention must be in [0, 1]")
    if not 0.0 < out["boundary_contrast_scale"] <= 1.0:
        raise ValueError("posterior-fossa boundary_contrast_scale must be in (0, 1]")
    if not -8.0 <= out["render_boundary_bias_mm"] <= 4.0:
        raise ValueError("posterior-fossa render_boundary_bias_mm must be in [-8, 4]")
    if not 0.0 <= out["render_boundary_jitter_mm"] <= 4.5:
        raise ValueError("posterior-fossa render_boundary_jitter_mm must be in [0, 4.5]")
    if not 5.0 <= out["render_boundary_correlation_mm"] <= 60.0:
        raise ValueError(
            "posterior-fossa render_boundary_correlation_mm must be in [5, 60]")
    if not 0.0 <= out["render_boundary_coarsen_mm"] <= 20.0:
        raise ValueError(
            "posterior-fossa render_boundary_coarsen_mm must be in [0, 20]")
    if not 0.0 <= out["render_boundary_floor_mix"] <= 1.0:
        raise ValueError(
            "posterior-fossa render_boundary_floor_mix must be in [0, 1]")
    if not 0.0 <= out["folial_detail_gain"] <= 2.0:
        raise ValueError("posterior-fossa folial_detail_gain must be in [0, 2]")
    if not 0.0 <= out["acquisition_guard_dilation_mm"] <= 12.0:
        raise ValueError(
            "posterior-fossa acquisition_guard_dilation_mm must be in [0, 12]")
    if not 0.0 <= out["acquisition_guard_feather_mm"] <= 10.0:
        raise ValueError(
            "posterior-fossa acquisition_guard_feather_mm must be in [0, 10]")
    if not 0.0 <= out["acquisition_transition_width_mm"] <= 10.0:
        raise ValueError(
            "posterior-fossa acquisition_transition_width_mm must be in [0, 10]")
    if not 0.0 <= out["boundary_confidence_floor"] <= 1.0:
        raise ValueError(
            "posterior-fossa boundary_confidence_floor must be in [0, 1]")
    if not -0.5 <= out["offset_dynamic_range"] <= 0.5:
        raise ValueError("posterior-fossa offset_dynamic_range must be in [-0.5, 0.5]")
    if not 0.0 < out["noise_correlation_mm"] <= 10.0:
        raise ValueError("posterior-fossa noise_correlation_mm must be in (0, 10]")
    if not 0.0 <= out["tentorium_level"] <= 1.0:
        raise ValueError("posterior-fossa tentorium_level must be in [0, 1]")
    if not 0.0 <= out["tentorium_blend"] <= 1.0:
        raise ValueError("posterior-fossa tentorium_blend must be in [0, 1]")
    if not 0.1 <= out["tentorium_thickness_mm"] <= 5.0:
        raise ValueError("posterior-fossa tentorium_thickness_mm must be in [0.1, 5]")
    if not 0.0 <= out["tentorium_base_s_frac"] <= 1.0:
        raise ValueError("posterior-fossa tentorium_base_s_frac must be in [0, 1]")
    if any(abs(out[k]) > 0.5 for k in (
            "tentorium_slope_r", "tentorium_slope_a", "tentorium_curvature")):
        raise ValueError("posterior-fossa tentorium slopes/curvature must be within +/-0.5")
    if not 0.0 <= out["inferior_margin_mm"] <= 50.0:
        raise ValueError("posterior-fossa inferior_margin_mm must be in [0, 50]")
    return out


def _bbox(mask):
    idx = np.argwhere(np.asarray(mask) > 0.5)
    if not idx.size:
        return None
    lo = idx.min(axis=0).astype(np.float64)
    hi = idx.max(axis=0).astype(np.float64)
    return lo, hi, np.maximum(hi - lo, 1.0)


def _coarse_image_head(canvas):
    """Image-derived filled-head support; deliberately independent of the target mask."""
    image = np.nan_to_num(
        np.asarray(canvas, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    positive = image[image > 0.0]
    if not positive.size:
        return np.zeros_like(image, dtype=bool)
    scale = float(np.percentile(positive, 99.0))
    foreground = image > max(0.02, 0.08 * scale)
    structure = ndi.generate_binary_structure(image.ndim, 1)
    foreground = ndi.binary_closing(foreground, structure=structure, iterations=2)
    head = ndi.binary_fill_holes(foreground)
    labels, count = ndi.label(head, structure=structure)
    if count > 1:
        sizes = ndi.sum(np.ones_like(labels), labels, index=np.arange(1, count + 1))
        head = labels == (int(np.argmax(sizes)) + 1)
    return np.asarray(head, dtype=bool)


def _rotation_matrix(degrees_ras):
    dr, da, ds = np.radians(np.asarray(degrees_ras, dtype=np.float64))
    rr = np.array([[1, 0, 0], [0, np.cos(dr), -np.sin(dr)],
                   [0, np.sin(dr), np.cos(dr)]], dtype=np.float64)
    ra = np.array([[np.cos(da), 0, np.sin(da)], [0, 1, 0],
                   [-np.sin(da), 0, np.cos(da)]], dtype=np.float64)
    rs = np.array([[np.cos(ds), -np.sin(ds), 0], [np.sin(ds), np.cos(ds), 0],
                   [0, 0, 1]], dtype=np.float64)
    return rs @ ra @ rr


def apply_posterior_fossa_pose(image, mask, spec, *, voxel_sizes=None):
    """Apply a gentle RAS scanner pose and place inferior brain near the FOV edge.

    The image and label are co-transformed.  A candidate touching any FOV face is
    rejected, so this training package never asks the model to hallucinate tissue
    that was removed by cropping.
    """
    spec = validate_posterior_fossa_spec(spec)
    spacing = np.asarray(_spacing(voxel_sizes), dtype=np.float64)
    image = np.asarray(image, dtype=np.float32)
    mask = np.asarray(mask) > 0.5
    if image.ndim != 3 or image.shape != mask.shape or not mask.any():
        return image.copy(), mask.copy()
    shape = np.asarray(image.shape, dtype=np.float64)
    center = (shape - 1.0) / 2.0
    rotation = _rotation_matrix(spec["pose_rotation_deg_ras"])
    scale = np.asarray(spec["pose_scale_ras"], dtype=np.float64)
    matrix = rotation @ np.diag(1.0 / scale)
    translation = np.asarray(spec["pose_shift_mm_ras"], dtype=np.float64) / spacing
    offset = center - matrix @ center - matrix @ translation
    moved_image = ndi.affine_transform(
        image, matrix, offset=offset, order=1, mode="constant", cval=0.0)
    moved_mask = ndi.affine_transform(
        mask.astype(np.float32), matrix, offset=offset, order=0,
        mode="constant", cval=0.0) > 0.5
    bounds = _bbox(moved_mask)
    if bounds is None:
        return image.copy(), mask.copy()
    lo, _hi, _extent = bounds
    desired = float(spec["inferior_margin_mm"]) / spacing[2]
    shift = np.zeros(3, dtype=np.float64)
    shift[2] = desired - lo[2]
    moved_image = ndi.shift(moved_image, shift, order=1, mode="constant", cval=0.0)
    moved_mask = ndi.shift(
        moved_mask.astype(np.float32), shift, order=0, mode="constant", cval=0.0) > 0.5
    bounds = _bbox(moved_mask)
    if bounds is None:
        return image.copy(), mask.copy()
    lo, hi, _extent = bounds
    # Keep at least one full voxel between acquired brain and every FOV face.
    if np.any(lo < 1.0) or np.any(hi > shape - 2.0):
        return image.copy(), mask.copy()
    return np.clip(moved_image, 0.0, 1.0).astype(np.float32), moved_mask


def _ellipsoid_weight(shape, mask, center_frac, fwhm_mm, spacing):
    bounds = _bbox(mask)
    if bounds is None:
        lo = np.zeros(3, dtype=np.float64)
        extent = np.maximum(np.asarray(shape, dtype=np.float64) - 1.0, 1.0)
    else:
        lo, _hi, extent = bounds
    center = lo + np.asarray(center_frac, dtype=np.float64) * extent
    sigma_vox = (np.asarray(fwhm_mm, dtype=np.float64)
                 / 2.354820045 / np.asarray(spacing, dtype=np.float64))
    sigma_vox = np.maximum(sigma_vox, 0.5)
    axes = []
    for axis, size in enumerate(shape):
        coord = (np.arange(size, dtype=np.float32) - float(center[axis])) / float(sigma_vox[axis])
        axes.append(np.exp(-0.5 * coord * coord).astype(np.float32))
    return (axes[0][:, None, None] * axes[1][None, :, None]
            * axes[2][None, None, :]).astype(np.float32)


def posterior_acquisition_support(mask, spec, *, voxel_sizes=None):
    """Build a coarse, displaced intracranial acquisition-law proxy.

    The returned support is deliberately *not* the supervision label. The
    supervision mask is first low-pass filtered at a physical scale that removes
    sulcal and folial detail, then its posterior transition is displaced smoothly.
    It therefore preserves an intracranial-versus-scalp distinction without placing
    the renderer's tissue/background switch on the exact answer contour.

    Version-2 replay specs upgrade with neutral coarsening/displacement. Version 3
    retains its historical displacement with neutral coarsening, and version 4
    retains its proxy with a neutral signal-floor mix in the parent renderer.
    """
    spec = validate_posterior_fossa_spec(spec)
    spacing = np.asarray(_spacing(voxel_sizes), dtype=np.float64)
    original = np.asarray(mask) > 0.5
    if original.ndim != 3 or not original.any() or original.all():
        return original.copy()
    field = _ellipsoid_weight(
        original.shape, original, spec["boundary_center_frac_ras"],
        spec["boundary_field_fwhm_mm_ras"], spacing)
    signed_original = (ndi.distance_transform_edt(original, sampling=spacing)
                       - ndi.distance_transform_edt(~original, sampling=spacing))
    signed_mm = signed_original
    coarsen = float(spec["render_boundary_coarsen_mm"])
    if coarsen > 1e-8:
        sigma = coarsen / 2.354820045 / spacing
        coarse_probability = ndi.gaussian_filter(
            original.astype(np.float32), sigma=np.maximum(sigma, 0.25), mode="nearest")
        coarse = coarse_probability >= 0.5
        # Do not allow smoothing to erase the entire support on small phantoms.
        if coarse.any():
            signed_coarse = (ndi.distance_transform_edt(coarse, sampling=spacing)
                             - ndi.distance_transform_edt(~coarse, sampling=spacing))
            # Coarsening belongs only to the broad posterior field. Elsewhere the
            # ordinary renderer remains unchanged.
            signed_mm = signed_original + field * (signed_coarse - signed_original)
        del coarse_probability, coarse

    bias = float(spec["render_boundary_bias_mm"])
    jitter = float(spec["render_boundary_jitter_mm"])
    if abs(bias) <= 1e-8 and jitter <= 1e-8:
        return np.asarray(signed_mm > 0.0, dtype=bool)

    g = np.random.default_rng(np.random.SeedSequence([int(spec["seed"]), 43]))
    displacement = g.standard_normal(original.shape).astype(np.float32)
    corr_sigma = (float(spec["render_boundary_correlation_mm"])
                  / 2.354820045 / spacing)
    displacement = ndi.gaussian_filter(
        displacement, sigma=np.maximum(corr_sigma, 0.25), mode="reflect")

    # Normalize inside the useful part of the field so the scientific RMS scalar
    # does not silently depend on FOV size or on Gaussian-kernel truncation.
    sample = displacement[field >= 0.20]
    if sample.size < 32:
        sample = displacement.reshape(-1)
    displacement -= float(sample.mean())
    sd = float(sample.std())
    if sd > 1e-8:
        displacement *= jitter / sd
    else:
        displacement.fill(0.0)
    displacement += bias
    np.clip(displacement, -8.0, 5.0, out=displacement)
    # The Gaussian field itself provides a smooth falloff without spreading the
    # displacement into superior/anterior cortex.  The transition becomes the
    # original support continuously outside the posterior field.
    displacement *= field
    return np.asarray(signed_mm + displacement > 0.0, dtype=bool)


def apply_posterior_fossa_signal(signal, mask, spec, *, voxel_sizes=None):
    """Apply regional UNIT1 contrast response and scanner-axis PSF before noise.

    Returns ``(signal, local_noise_gain)``.  The gain is consumed by the parent
    renderer after it generates the normal signal-dependent UNIT1 noise field.
    """
    spec = validate_posterior_fossa_spec(spec)
    spacing = _spacing(voxel_sizes)
    signal = np.asarray(signal, dtype=np.float32)
    # Extract coherent negative ridges before the low-frequency response and PSF.
    # The small prefilter rejects voxel noise; the minimum filter widens anatomical
    # dark fissures by roughly one voxel.  Unlike symmetric unsharp masking, this
    # cannot manufacture bright white-matter spokes or a bright posterior rim.
    coherent = ndi.gaussian_filter(
        signal, sigma=np.maximum(0.35 / np.asarray(spacing), 0.15), mode="nearest")
    source_detail = coherent - ndi.gaussian_filter(
        coherent, sigma=np.maximum(0.90 / np.asarray(spacing), 0.25),
        mode="nearest")
    np.minimum(source_detail, 0.0, out=source_detail)
    minimum_size = tuple(
        2 * max(1, int(round(0.75 / float(v)))) + 1 for v in spacing)
    widened_dark = ndi.minimum_filter(source_detail, size=minimum_size, mode="nearest")
    source_detail = 0.35 * source_detail + 0.65 * widened_dark
    del coherent, widened_dark
    mask = np.asarray(mask) > 0.5
    field = _ellipsoid_weight(
        signal.shape, mask, spec["field_center_frac_ras"],
        spec["field_fwhm_mm_ras"], spacing)
    bridge = _ellipsoid_weight(
        signal.shape, mask, spec["bridge_center_frac_ras"],
        spec["bridge_fwhm_mm_ras"], spacing)
    # A broad maximum keeps the stress field smooth and avoids a mask-shaped seam.
    np.maximum(field, 0.80 * bridge, out=field)
    values = signal[mask] if mask.any() else signal[np.isfinite(signal)]
    if values.size:
        p5, p95 = np.percentile(values, [5.0, 95.0])
        dynamic = max(float(p95 - p5), 1e-3)
    else:
        dynamic = 1.0
    contrast = 1.0 - field * (1.0 - float(spec["contrast_scale"]))
    out = 0.5 + contrast * (signal - 0.5)
    out += field * float(spec["offset_dynamic_range"]) * dynamic
    # The narrower bridge field specifically weakens the cerebellum/brainstem
    # connection without zeroing or imposing a rectangular dropout.
    bridge_contrast = 1.0 - bridge * (1.0 - float(spec["bridge_signal_scale"]))
    out = 0.5 + bridge_contrast * (out - 0.5)

    fwhm = np.asarray(spec["psf_fwhm_mm_ras"], dtype=np.float64)
    sigma_vox = fwhm / 2.354820045 / np.asarray(spacing, dtype=np.float64)
    if np.any(sigma_vox > 0.05):
        out = ndi.gaussian_filter(out.astype(np.float32), sigma_vox).astype(np.float32)

    # Real MP2RAGE cerebellar failures can remain sharply foliated even when their
    # low-frequency response is shifted.  Re-introduce only the dark, widened source
    # residual through the same smooth scanner-coordinate field; do not sharpen a
    # target-mask edge or boost bright structures.
    out += (field * float(spec["folial_detail_gain"]) * source_detail)

    noise_multiplier = 1.0 / max(float(spec["snr_ratio"]), 1e-3)
    noise_gain = 1.0 + field * (noise_multiplier - 1.0)
    # Make the bridge at least as difficult as the broader posterior region.
    np.maximum(noise_gain, 1.0 + bridge * 0.5 * (noise_multiplier - 1.0), out=noise_gain)
    return np.clip(out, 0.0, 1.0).astype(np.float32), noise_gain.astype(np.float32)


def apply_posterior_boundary_ambiguity(canvas, mask, spec, *, voxel_sizes=None):
    """Mix cerebellum and surrounding signal through a broad scanner-space PSF.

    This stage runs on the *completed* MP2RAGE canvas, after brain, extracranial
    tissue and ratio background have all been rendered. Consequently the local
    low-pass response crosses their interfaces naturally. The target mask is used
    only to establish a coarse canonical-RAS head frame for the ellipsoid center;
    there is no distance-to-mask or inside/outside gate and therefore no synthetic
    edge at the supervision boundary.
    """
    spec = validate_posterior_fossa_spec(spec)
    spacing = np.asarray(_spacing(voxel_sizes), dtype=np.float64)
    canvas = np.asarray(canvas, dtype=np.float32)
    mask = np.asarray(mask) > 0.5
    if canvas.ndim != 3 or canvas.shape != mask.shape or not mask.any():
        return canvas.copy()

    field = _ellipsoid_weight(
        canvas.shape, mask, spec["boundary_center_frac_ras"],
        spec["boundary_field_fwhm_mm_ras"], spacing)
    sigma_vox = (np.asarray(spec["boundary_blur_fwhm_mm_ras"], dtype=np.float64)
                 / 2.354820045 / spacing)
    # Separate sub-millimetre texture/noise from the anatomical signal.  The late
    # stage sees the final complex-ratio air, so a broad unconditional low-pass
    # would create an easily recognized smooth ellipsoid.  Instead derive a
    # target-independent *coherent edge* response from the already blurred image:
    # flat/noise-dominated regions retain their original canvas, while anatomical
    # interfaces receive the partial-volume candidate below.
    split_sigma = np.maximum((1.0 / 2.354820045) / spacing, 0.05)
    low = ndi.gaussian_filter(canvas, split_sigma, mode="nearest").astype(np.float32)
    fine = canvas - low
    blurred = ndi.gaussian_filter(
        low, np.maximum(sigma_vox, 0.05), mode="nearest").astype(np.float32)
    local_mean = ndi.gaussian_filter(
        blurred, np.maximum(1.5 * sigma_vox, 0.25), mode="nearest").astype(np.float32)

    # Reuse ``low`` as the coherent-gradient map. Absolute gradient thresholds are
    # unstable across stored-UNI protocols and let weak random-air gradients compete
    # with anatomy. Normalize instead by the local RMS of the original fine residual:
    # a tissue interface remains coherent after the broad PSF, whereas stochastic
    # ratio-air texture has large fine RMS but little coherent low-frequency slope.
    ndi.gaussian_gradient_magnitude(
        blurred, sigma=np.maximum(0.5 / spacing, 0.25),
        output=low, mode="nearest")

    confidence_floor = float(spec["boundary_confidence_floor"])
    structure_lo = structure_hi = 0.0
    if confidence_floor > 0.0:
        # Calibrate a target-independent structural gate from the completed image.
        # The subsampled filled-head proxy is used only to make the percentiles
        # insensitive to empty FOV padding; it never reads the supervision mask.
        # Percentiles of the already low-pass gradient distinguish coherent tissue,
        # scalp and cavity interfaces from stationary complex-ratio air.
        sample = (slice(None, None, 2),) * canvas.ndim
        sampled_gradient = low[sample]
        sampled_head = _coarse_image_head(canvas[sample])
        structural_values = sampled_gradient[
            sampled_head & np.isfinite(sampled_gradient)]
        if structural_values.size < 32:
            structural_values = sampled_gradient[np.isfinite(sampled_gradient)]
        if structural_values.size:
            structure_lo, structure_hi = (
                float(v) for v in np.percentile(structural_values, [70.0, 97.0]))

    # Build the contrast-compressed low-frequency candidate in ``blurred``.
    # The other work arrays become scratch buffers immediately afterward so the
    # helper does not retain extra 224^3 confidence/denominator arrays.
    contrast = float(spec["boundary_contrast_scale"])
    np.subtract(blurred, local_mean, out=blurred)
    blurred *= contrast
    blurred += local_mean
    np.multiply(fine, float(spec["boundary_texture_retention"]), out=local_mean)
    blurred += local_mean

    if confidence_floor > 0.0:
        # Reuse the no-longer-needed local-mean buffer for a smooth cosine gate.
        # A degenerate gradient distribution (e.g. a constant canvas) has no
        # image evidence of an anatomical interface and therefore gets zero gate.
        gradient_span = structure_hi - structure_lo
        if gradient_span > max(1e-8, 1e-5 * abs(structure_hi)):
            np.subtract(low, structure_lo, out=local_mean)
            local_mean /= gradient_span
            np.clip(local_mean, 0.0, 1.0, out=local_mean)
            local_mean *= np.pi
            np.cos(local_mean, out=local_mean)
            local_mean *= -0.5
            local_mean += 0.5
            ndi.gaussian_filter(
                local_mean, sigma=np.maximum(0.5 / spacing, 0.25),
                output=local_mean, mode="nearest")
            np.clip(local_mean, 0.0, 1.0, out=local_mean)
        else:
            local_mean.fill(0.0)

    # Dimensionless coherence ratio, k=1:
    #   x = coherent_gradient / local_fine_rms
    #   confidence = x^2 / (1 + x^2)
    # This is image-only and scale-invariant; it suppresses coherent boundary detail
    # while preserving almost all high-frequency texture in flat complex-ratio air.
    np.square(fine, out=fine)
    ndi.gaussian_filter(
        fine, sigma=np.maximum(1.5 / spacing, 0.25),
        output=fine, mode="nearest")
    fine += 1e-8
    np.sqrt(fine, out=fine)
    np.divide(low, fine, out=low)
    np.square(low, out=low)
    if confidence_floor > 0.0:
        # Grain raises the fine-RMS denominator and previously disabled ambiguity
        # exactly for the hard grainy reference family.  Raise confidence only where
        # the completed image independently contains a coherent low-frequency
        # structure; flat complex-ratio air remains on the base confidence path.
        # ``fine`` is dead after the ratio and becomes denominator scratch, leaving
        # ``local_mean`` intact as the structural gate.
        np.add(low, 1.0, out=fine)
        np.divide(low, fine, out=low)
        low *= 1.0 - confidence_floor
        local_mean *= confidence_floor
        low += local_mean
    else:
        # Keep the v6/floor-zero operation order and scratch destination exact.
        np.add(low, 1.0, out=local_mean)
        np.divide(low, local_mean, out=low)
    del fine, local_mean

    # The square-root falloff keeps both cerebellar hemispheres inside the broad
    # response without turning the ellipsoid into a hard-edged region. Multiplying
    # by the image-derived edge confidence prevents a scanner-space noise-texture
    # cue in otherwise flat air; no target-boundary gate is introduced.
    np.sqrt(field, out=field)
    field *= float(spec["boundary_mix"])
    field *= low
    np.clip(field, 0.0, 1.0, out=field)
    del low
    np.subtract(blurred, canvas, out=blurred)
    blurred *= field
    out = canvas.copy()
    out += blurred
    np.clip(out, 0.0, 1.0, out=out)
    return out.astype(np.float32, copy=False)


def correlated_posterior_noise(shape, mask, spec, sigma, *, voxel_sizes=None):
    """Additional correlated UNIT1 noise for the locally reduced-SNR package."""
    spec = validate_posterior_fossa_spec(spec)
    sigma = max(float(sigma), 0.0)
    if sigma <= 0.0:
        return np.zeros(shape, dtype=np.float32)
    spacing = _spacing(voxel_sizes)
    field = _ellipsoid_weight(
        shape, mask, spec["field_center_frac_ras"], spec["field_fwhm_mm_ras"], spacing)
    g = np.random.default_rng(np.random.SeedSequence([int(spec["seed"]), 17]))
    noise = g.standard_normal(shape).astype(np.float32)
    corr = np.asarray([float(spec["noise_correlation_mm"]) / v for v in spacing])
    noise = ndi.gaussian_filter(noise, np.maximum(corr, 0.0)).astype(np.float32)
    sd = float(noise.std())
    if sd > 1e-6:
        noise /= sd
    multiplier = max(0.0, 1.0 / float(spec["snr_ratio"]) - 1.0)
    return (field * noise * sigma * multiplier).astype(np.float32)


def apply_tentorium_mimic(canvas, mask, spec, *, voxel_sizes=None):
    """Add a curved, discontinuous dura/tentorium hard negative.

    The sheet is defined by a coarse RAS plane and restricted to low-signal pockets
    inside an image-derived filled head. The target mask supplies only the coarse
    RAS bounding box; its binary edge never gates the hard negative.
    """
    spec = validate_posterior_fossa_spec(spec)
    canvas = np.asarray(canvas, dtype=np.float32)
    if not spec["tentorium_on"] or canvas.ndim != 3:
        return canvas.copy()
    spacing = _spacing(voxel_sizes)
    bounds = _bbox(mask)
    if bounds is None:
        return canvas.copy()
    lo, _hi, extent = bounds
    r = ((np.arange(canvas.shape[0], dtype=np.float32) - float(lo[0]))
         / float(extent[0]))
    a = ((np.arange(canvas.shape[1], dtype=np.float32) - float(lo[1]))
         / float(extent[1]))
    s = ((np.arange(canvas.shape[2], dtype=np.float32) - float(lo[2]))
         / float(extent[2]))
    rr = r[:, None]
    aa = a[None, :]
    plane = (float(spec["tentorium_base_s_frac"])
             + float(spec["tentorium_slope_r"]) * (rr - 0.5)
             + float(spec["tentorium_slope_a"]) * (aa - 0.25)
             + float(spec["tentorium_curvature"]) * (rr - 0.5) ** 2)
    thickness_frac = (float(spec["tentorium_thickness_mm"])
                      / max(float(extent[2]) * float(spacing[2]), 1e-3))
    thickness_frac = max(thickness_frac, 0.003)
    band = np.exp(-0.5 * ((s[None, None, :] - plane[:, :, None])
                          / thickness_frac) ** 2).astype(np.float32)
    posterior = np.clip((0.62 - a) / 0.24, 0.0, 1.0).astype(np.float32)
    lateral = np.exp(-0.5 * ((r - 0.5) / 0.55) ** 2).astype(np.float32)
    band *= lateral[:, None, None] * posterior[None, :, None]

    head = _coarse_image_head(canvas)
    values = canvas[head & np.isfinite(canvas)]
    if not values.size:
        return canvas.copy()
    p10, p90 = np.percentile(values, [10.0, 90.0])
    dynamic = max(float(p90 - p10), 1e-3)
    spacing_array = np.asarray(spacing, dtype=np.float64)
    local = ndi.gaussian_filter(
        canvas, sigma=np.maximum(1.0 / spacing_array, 0.25)).astype(np.float32)
    cutoff = float(p10 + 0.38 * dynamic)
    softness = max(0.16 * dynamic, 1e-3)
    low_signal = np.clip((cutoff + softness - local) / (2.0 * softness), 0.0, 1.0)
    low_signal = (low_signal * low_signal * (3.0 - 2.0 * low_signal)).astype(np.float32)

    # Topology is also image-derived: a tentorium-like hard negative belongs to a
    # low-signal pocket/cavity that communicates with exterior/background signal,
    # not to an enclosed dark island in deep tissue.  Six-connectivity is strict on
    # purpose; the continuous gate below and the parent PSF provide the feather.
    # Use the already calibrated soft low-signal response rather than ``<= cutoff``.
    # The strict core avoids classifying an otherwise homogeneous head as one giant
    # low component when p10 and p90 nearly coincide.
    low_candidate = low_signal >= 0.90
    exterior_seed = np.zeros_like(low_candidate, dtype=bool)
    exterior_seed[[0, -1], :, :] = low_candidate[[0, -1], :, :]
    exterior_seed[:, [0, -1], :] = low_candidate[:, [0, -1], :]
    exterior_seed[:, :, [0, -1]] = low_candidate[:, :, [0, -1]]
    exterior_connected = ndi.binary_propagation(
        exterior_seed, structure=ndi.generate_binary_structure(3, 1),
        mask=low_candidate)
    del exterior_seed, low_candidate

    # A low-signal gate alone can place the synthetic sheet through dark GM/CSF-like
    # voxels deep in the brain. Require a coherent interface derived solely from
    # the image as well. This keeps the dura-like response on visible tissue/cavity
    # boundaries without restoring an exact target-mask shell.
    coherent_edge = ndi.gaussian_gradient_magnitude(
        local, sigma=np.maximum(1.0 / spacing_array, 0.35),
        mode="nearest").astype(np.float32)
    sampled_edge = coherent_edge[::2, ::2, ::2]
    sampled_head = head[::2, ::2, ::2]
    edge_values = sampled_edge[sampled_head]
    if edge_values.size < 32:
        return canvas.copy()
    edge_lo, edge_hi = np.percentile(edge_values, [70.0, 95.0])
    if float(edge_hi - edge_lo) <= 1e-6:
        return canvas.copy()
    np.subtract(coherent_edge, float(edge_lo), out=coherent_edge)
    coherent_edge /= float(edge_hi - edge_lo)
    np.clip(coherent_edge, 0.0, 1.0, out=coherent_edge)
    # Smoothstep in place, using the no-longer-needed local mean as scratch.
    np.multiply(coherent_edge, -2.0, out=local)
    local += 3.0
    np.square(coherent_edge, out=coherent_edge)
    coherent_edge *= local
    del local
    band *= head
    band *= low_signal
    band *= coherent_edge
    band *= exterior_connected
    del low_signal, coherent_edge, exterior_connected

    # Smooth 2-D gaps stop the sheet becoming a perfectly continuous synthetic plane.
    g = np.random.default_rng(np.random.SeedSequence([int(spec["seed"]), 29]))
    gaps = g.standard_normal(canvas.shape[:2]).astype(np.float32)
    gaps = ndi.gaussian_filter(gaps, sigma=max(1.0, 0.035 * min(canvas.shape[:2])))
    gaps = (gaps - float(gaps.mean())) / (float(gaps.std()) or 1.0)
    continuity = np.clip(0.65 + 0.30 * gaps, 0.0, 1.0).astype(np.float32)
    band *= continuity[:, :, None]

    target = float(p10 + float(spec["tentorium_level"]) * (p90 - p10))
    weight = np.clip(float(spec["tentorium_blend"]) * band, 0.0, 1.0)
    out = canvas + weight * (target - canvas)
    return np.clip(out, 0.0, 1.0).astype(np.float32)
