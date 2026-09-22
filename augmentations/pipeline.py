"""Online image synthesis for brain masking.

``make_training_sample`` mixes clean scans, protocol-specific appearances,
real-texture augmentations, and label-driven synthesis. Label-preserving
artifacts train the masker to segment through acquisition defects.

``discover_pairs`` finds raw scan/mask pairs for online training; samples are
generated in memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from augmentations.sampling_policy import MIXED_NON_MP2_BASE_SHARES, SamplingPolicy, resolve_sampling_policy

import numpy as np
from scipy import ndimage as ndi

from augmentations.config import (
    STAGES,
    AugmentationError,
    _checked_sample_result,
    enabled,
    settings,
    configured_sample,
)


# Overlays change intensities while preserving the reference mask. The pool excludes:
# - composed `metal`: it warps anatomy without co-warping the mask; use metal_3d/metal_void.
# - nonfinite: NaN/Inf violate the finite-output contract.
# - sequence/contrast remaps and redundant noise/resolution: other synthesis stages own these.
# - corner-centred or severity-blind primitives that have little effect with default parameters.
_MASKER_ARTIFACT_NAMES = (
    # k-space / motion (classic MRI artifacts)
    "motion", "motion_continuous", "ghosting", "nyquist_ghost", "eye_ghosting", "gibbs", "ringing_3d",
    "spike", "herringbone", "zipper", "aliasing", "slab_wrap", "pe_undersample", "fse_echo_train",
    # noise
    "rician_noise", "noise", "g_factor_noise", "signal_drop_band",
    # intensity-domain artifacts
    "bias", "chemical_shift", "clipping", "quantization", "incomplete_fat_sat",
    "bounce_point_null", "draw_shape",
    # focal hardware / physics (metal_3d/metal_void only â€” composed `metal` warps anatomy, see above)
    "metal_3d", "metal_void", "void", "blob", "dropout", "droppatch", "create_ring",
    "random_erasing",
    # reconstruction
    "dl_recon", "spin_history",
    # acquisition: through-plane thick-slice + inter-slice gap (distinct from the noise-floor augs)
    "anisotropy",
)


# Per-artifact severity RANGE (lo, hi). Default is moderate-to-strong (0.3, 0.85). Pure-noise transforms
# fully obscure the brain well before that (calibrated: plain `noise` washes the brain out by ~0.3,
# `g_factor` by ~0.35; `rician` â€” the realistic Rician model â€” stays readable much higher), which is
# unrealistic (a real brain scan's SNR is never that bad), so they get gentler, lower ranges that keep
# the brain a VALID mask target while still being challenging.
_ARTIFACT_SEV_RANGE = {
    "noise": (0.10, 0.34), "noise_random": (0.10, 0.28), "noise_straight": (0.10, 0.28),
    "g_factor_noise": (0.12, 0.35), "rician_noise": (0.10, 0.60),
    "dropout": (0.20, 0.50),          # cap: slice-dropout near-blacks-out the brain at the top of the range
    "chemical_shift": (0.12, 0.80),   # reach the common subtle 1-2px shift
    "motion": (0.30, 1.10),           # add a severe-reject motion tail (label-preserving at any severity)
    # These caps retain brain signal on NFBS scans while visibly corrupting the image.
    "bounce_point_null": (0.15, 0.45),  # IR tissue null: brain <0.05 frac is HEAVY-TAILED (mean ~1.4% but up to
    #                                     ~17.6% on unlucky null-center draws at 0.55) -> ceiling 0.45 caps the tail
    #                                     to ~5.5% max while keeping a VISIBLE dark band (adversarial-verified)
    "droppatch": (0.20, 0.60),        # enclosed in-brain signal-drop patches; retention excellent, keep moderate
    "anisotropy": (0.20, 0.75),       # thick-slice PSF + inter-slice gap; heavy blur but still segmentable
    # eye_ghosting / random_erasing / draw_shape: no entry -> the (0.3, 0.85) default; measured retention
    # is excellent across the whole band (bright bands / peripheral smooth falloff / focal B1 lobe â€” no obliteration).
}


def _masker_artifact_pool() -> List[str]:
    """The subset of `_MASKER_ARTIFACT_NAMES` that exists, is label-preserving, and runs in 3D."""
    from augmentations import REGISTRY
    return [n for n in _MASKER_ARTIFACT_NAMES
            if n in REGISTRY and REGISTRY[n].label_preserving and REGISTRY[n].dims in ("3d", "either")]


def _focal_kwargs(name: str, mask, rng: np.random.Generator, voxel_sizes=None) -> Dict[str, Any]:
    """Randomize the PLACEMENT/size of focal artifacts whose registry cores default to a FIXED location
    (the metal cores are deterministic-by-design for the slider explorer, so at a fixed severity they
    draw the SAME implant every time). Passing a random location/size varies those low-level features â€”
    the artifact_explorer philosophy â€” so metal lands somewhere new each draw. Non-focal artifacts get
    no extra kwargs (they already randomize their structure internally)."""
    has_mask = mask is not None and np.asarray(mask).any()
    if name in ("ghosting", "nyquist_ghost", "spin_history"):     # DE-PIN the artifact axis (was fixed -> 1 decoy
        return {"axis": int(rng.integers(0, 3))}                  #   orientation; real artifacts present on any axis)
    if name == "eye_ghosting":
        # eye_ghosting bands live on THREE axes: the A-P `axis` PLUS `si_axis`/`lr_axis` for the eye-column
        # localization. Passing only `axis` (leaving the hardcoded si_axis=1/lr_axis=2) makes axis 1 or 2 COLLIDE
        # with an eye band -> `src_mask` empties -> a silent bit-exact no-op (verified: axis 1/2 changed 0/8 draws).
        # Pass a CONSISTENT permutation instead so the effect fires on every axis choice (spatial_augment already
        # rotated the brain, so the brain-relative beam orientation still varies).
        ap = int(rng.integers(0, 3))
        rest = [a for a in range(3) if a != ap]
        return {"axis": ap, "si_axis": rest[0], "lr_axis": rest[1]}
    if name == "metal_3d":                                        # loc FRACTIONAL; size tiny clip -> head-spanning bloom
        return {"loc": [float(rng.uniform(0.12, 0.88)) for _ in range(3)],
                "size_scale": float(rng.uniform(0.5, 3.0))}
    if name == "metal_void" and has_mask:                         # cx/cy/cz VOXEL; ~half from the BRAIN BOUNDARY
        msk = np.asarray(mask) > 0.5
        if rng.random() < 0.5:                                    # boundary-origin void (hardest masking decision)
            shell = msk & ~ndi.binary_erosion(msk, iterations=5)
            pool = np.argwhere(shell) if shell.any() else np.argwhere(msk)
        else:
            pool = np.argwhere(msk)
        c = pool[int(rng.integers(len(pool)))]
        return {"cx": float(c[0]), "cy": float(c[1]), "cz": float(c[2]),
                "radius_mm": float(rng.uniform(6.0, 22.0)),
                "voxel_sizes": tuple(float(x) for x in (voxel_sizes or (1.0, 1.0, 1.0))),
                "depth": float(rng.uniform(0.4, 0.9))}
    return {}


def _configured_artifact_kwargs(name: str, mask, rng: np.random.Generator, voxel_sizes=None):
    """Merge explicit controls with compatible sampled placement defaults."""
    generated = _focal_kwargs(name, mask, rng, voxel_sizes=voxel_sizes)
    params = settings(name, {"params": {}})["params"]
    if name == "eye_ghosting":
        axes = ("axis", "si_axis", "lr_axis")
        used = {params[key] % 3 for key in axes if key in params}
        for key in axes:
            if key not in params:
                preferred = generated[key]
                generated[key] = preferred if preferred not in used else next(
                    axis for axis in range(3) if axis not in used)
                used.add(generated[key])
    if name == "metal_void" and "radius" in params and "radius_mm" not in params:
        # The core prioritizes a physical radius. A user-specified voxel radius
        # must therefore replace the generated physical-radius default.
        generated.pop("radius_mm", None)
    return {**generated, **params}


# Artifacts that DARKEN the brain toward the air floor â€” either directly (dropout / signal_drop_band /
# void / metal_void zero a region) or as an AMPLIFIER (bounce_point_null nulls a whole tissue band that
# stays just above the floor ALONE but is pushed below it by any second dimmer). Stacking two of these
# compounds SUPER-additively: measured on real NFBS, bounce_point_null + dropout blacks out ~31% of the
# brain and bounce_point_null + signal_drop_band up to ~50% â€” far past the single-artifact envelope
# (dropout@cap ~14%), which would train "near-black region = brain" (the project's #1 over-inclusion
# failure). So `apply_artifacts` allows at most ONE of these per sample (see `_drop_extra_dimmers`).
_STRONG_DIMMERS = frozenset({"dropout", "signal_drop_band", "bounce_point_null", "void", "metal_void"})


def _drop_extra_dimmers(names, rng: np.random.Generator):
    """Keep at most ONE strong signal-dimmer (see `_STRONG_DIMMERS`) in the drawn artifact list; drop any
    extras (a randomly-chosen dimmer survives). Non-dimmers are untouched and order is preserved. This is
    the COMPOUNDING GUARD: two stacked dimmers black out too much brain under a full-brain label."""
    dimmers = [n for n in names if n in _STRONG_DIMMERS]
    if len(dimmers) <= 1:
        return names
    keep = str(dimmers[int(rng.integers(len(dimmers)))])
    return [n for n in names if n not in _STRONG_DIMMERS or str(n) == keep]


def apply_artifacts(scan01: np.ndarray, rng: np.random.Generator, mask=None, n_range=(1, 2),
                    voxel_sizes=None, exclude=()):
    """Overlay 1-2 realistic QC-impacting artifacts (motion / eye-ghosting / noise / bias / ringing /
    metal / signal-void / IR-null / coil-dropout / recon â€¦) on a [0,1] scan, keeping it in [0,1].
    Label-preserving, so the brain mask stays valid. Draws from `_masker_artifact_pool()` (the live-
    registry subset of `_MASKER_ARTIFACT_NAMES`), then applies the compounding guard (`_drop_extra_dimmers`:
    at most one strong signal-dimmer per sample). Returns (artifacted_scan01, [applied names]). A
    transform that errors, changes shape, or returns non-finite aborts with AugmentationError."""
    from augmentations import apply as _aug_apply
    excluded = set(str(x) for x in exclude)
    pool = [name for name in _masker_artifact_pool() if name not in excluded and enabled(name)]
    out = np.clip(np.asarray(scan01, np.float32), 0.0, 1.0)
    if not pool:
        return out, []
    n_range = settings("artifact_overlay", {"count_range": list(n_range)})["count_range"]
    k = int(rng.integers(int(n_range[0]), int(n_range[1]) + 1))
    names = list(rng.choice(np.array(pool, dtype=object), size=min(k, len(pool)), replace=False))
    names = _drop_extra_dimmers(names, rng)                       # compounding guard: <=1 strong dimmer/sample
    applied: List[str] = []
    for name in names:
        lo, hi = settings(name, {"severity_range": list(_ARTIFACT_SEV_RANGE.get(name, (0.3, 0.85)))})["severity_range"]       # moderate-strong; gentler for noise
        sev = float(rng.uniform(lo, hi))
        try:
            res = np.asarray(_aug_apply(name, out, sev, int(rng.integers(1, 2 ** 31)), mask=mask,
                                        **_configured_artifact_kwargs(name, mask, rng, voxel_sizes=voxel_sizes)),
                             dtype=np.float32)
        except Exception as exc:
            raise AugmentationError(
                f"Augmentation {name!r} failed: {type(exc).__name__}: {exc}"
            ) from exc
        if res.shape != out.shape:
            raise AugmentationError(
                f"Augmentation {name!r} returned shape {res.shape}; expected {out.shape}")
        if not np.isfinite(res).all():
            raise AugmentationError(f"Augmentation {name!r} returned nonfinite values")
        out = np.clip(res, 0.0, 1.0)
        applied.append(str(name))
    return out, applied


def _valid_final_geometry(reference: np.ndarray, candidate: np.ndarray,
                          min_volume_ratio: float = 0.30,
                          max_volume_ratio: float = 3.0) -> bool:
    """Reject degenerate final masks after pose/scale/FOV augmentation.

    Broad size and one-face partial-FOV variation remain allowed. Empty/nearly-ejected masks,
    explosive growth, new disconnected fragments, or clipping through both faces on multiple axes
    fall back to the pre-geometry sample instead of becoming silent wrong supervision.
    """
    ref = np.asarray(reference, dtype=bool)
    cur = np.asarray(candidate, dtype=bool)
    n0, n1 = int(ref.sum()), int(cur.sum())
    if n0 == 0 or n1 == 0:
        return False
    ratio = n1 / float(n0)
    if ratio < float(min_volume_ratio) or ratio > float(max_volume_ratio):
        return False
    structure = ndi.generate_binary_structure(cur.ndim, cur.ndim)
    ref_components = int(ndi.label(ref, structure=structure)[1])
    cur_components = int(ndi.label(cur, structure=structure)[1])
    if cur_components > max(1, ref_components):
        return False
    opposing_faces = 0
    for axis in range(cur.ndim):
        lo = np.take(cur, 0, axis=axis).any()
        hi = np.take(cur, cur.shape[axis] - 1, axis=axis).any()
        opposing_faces += int(lo and hi)
    return opposing_faces < 2


def spatial_augment(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator,
                    p_geom: float = 1.0, p_res: float = 0.6,
                    mp2rage_background: bool = False):
    """BENIGN spatial augmentation for broad-coverage tiers (and optional legacy anchors),
    making the masker fully ORIENTATION-INVARIANT:
    * a UNIFORMLY RANDOM 3D rotation (full 360Â° per axis) + lateral flip â€” the brain can land in any
      orientation, about the BRAIN CENTROID. The SCALE spans both regimes: most draws keep the brain
      comfortably inside the FOV, but a fraction FILL or slightly over-fill it (co-clipping the mask) so
      the masker sees brains that FILL the FOV / are pushed against an edge â€” the tight-FOV, large-head,
      off-center regime real foreign scans live in (the shrink-only version never showed it). Mask
      co-transformed (label stays correct; a co-clipped brain is a valid partial-FOV target).
    * acquired RESOLUTION / slice-thickness change â€” image only (resolution does not move anatomy).
    Returns (img in [0,1], bool mask)."""
    from augmentations.appearance import realistic_resolution, _rot_matrix
    out = np.clip(np.asarray(img, np.float32), 0.0, 1.0)
    m = np.asarray(mask, bool)
    pre_geom_out, pre_geom_mask = out.copy(), m.copy()
    if rng.random() < float(p_geom) and m.any():
        if rng.random() < 0.5:                                    # lateral flip (mirror -> chirality)
            out = np.ascontiguousarray(out[:, :, ::-1]); m = np.ascontiguousarray(m[:, :, ::-1])
        shape = np.asarray(out.shape, np.float64)
        R = _rot_matrix(*rng.uniform(-180.0, 180.0, 3))           # FULL random 3D orientation
        # SCALE that spans inside-FOV AND fill/over-fill. s_fill ~= the scale at which the brain just
        # fills the FOV (bounding-sphere based); >s_fill over-fills and co-clips (a valid partial FOV).
        idx = np.argwhere(m); ctr = idx.mean(0)
        r_b = float(np.sqrt(((idx - ctr) ** 2).sum(1)).max()) + 1e-3
        s_fill = float(shape.min()) * 0.5 / r_b
        # ORDER-SAFE bounds: a small brain (e.g. after the upstream morph) gives a LARGE s_fill, and a
        # large one gives a small s_fill â€” either could invert `low > high` in the raw uniform() and
        # raise "high - low < 0" (a rare crash the fresh-entropy online path hits ~2-3% on tiny/edge
        # heads; skipped in training but FATAL in the first-sample smoke probe). Cap both ends together.
        if rng.random() < 0.45:                                   # FILL / slightly over-fill the FOV
            lo, hi = min(0.85 * s_fill, 1.4), min(1.25 * s_fill, 1.4)   # both capped at 1.4 -> lo<=hi always
            s = float(rng.uniform(lo, hi))
        else:                                                     # comfortably inside the FOV
            hi = min(1.0, s_fill); lo = min(0.72, hi)             # if the brain barely fits (s_fill<0.72) shrink to fit
            s = float(rng.uniform(lo, hi))
        M = R / s
        vc = (shape - 1.0) / 2.0
        # POSITION: usually ~centered, but 30% strongly OFF-CENTER so the brain can sit against / over a
        # FOV face (the only augs that did this were neutralized by the old unconditional recenter).
        jit = 0.22 if rng.random() < 0.3 else 0.06
        bc = np.asarray(ndi.center_of_mass(m)) + rng.uniform(-jit, jit, 3) * shape
        offset = bc - M @ vc                                      # brain centroid -> (jittered) FOV position
        out = ndi.affine_transform(out, M, offset=offset, order=1, mode="constant", cval=0.0)
        source_support = None
        if bool(mp2rage_background):
            # A stored MP2RAGE UNI has complex-ratio air across its rectangular
            # reconstruction grid. Rotating that already-rendered grid with cval=0
            # otherwise exposes a black corner outside a straight gray diamond -- a
            # repeated synthetic shortcut in every broadly posed MP2RAGE route.
            source_support = ndi.affine_transform(
                np.ones(out.shape, dtype=np.uint8), M, offset=offset, order=0,
                mode="constant", cval=0) > 0
        m = ndi.affine_transform(m.astype(np.float32), M, offset=offset, order=0, mode="constant", cval=0.0) > 0.5
        if not _valid_final_geometry(pre_geom_mask, m):
            out, m = pre_geom_out, pre_geom_mask
        elif source_support is not None and np.any(~source_support):
            # Reuse the transformed acquisition's own face-adjacent air law. A
            # guarded inner shell excludes anatomy/scalp, and robust trimming keeps
            # an occasional explicit reconstructed-FOV zero slab out of the fill.
            inner = ndi.binary_erosion(source_support, iterations=3, border_value=0)
            edge_air = source_support & (~inner)
            anatomy_guard = ndi.binary_dilation(m, iterations=6)
            pool = out[edge_air & (~anatomy_guard) & np.isfinite(out)]
            pool = pool[pool > np.float32(1e-6)]
            if pool.size < 64:
                pool = out[source_support & (~anatomy_guard) & np.isfinite(out)]
                pool = pool[pool > np.float32(1e-6)]
            if pool.size:
                lo, hi = np.quantile(pool, (0.02, 0.98))
                pool = pool[(pool >= lo) & (pool <= hi)]
            if pool.size:
                missing = ~source_support
                out[missing] = pool[rng.integers(
                    0, int(pool.size), size=int(np.count_nonzero(missing)))]
    if rng.random() < float(p_res):
        out = realistic_resolution(out, rng, p=1.0)
    return np.clip(out, 0.0, 1.0).astype(np.float32), m


# Canonical "textbook" examples of the most common protocols â€” one is rendered CLEAN per standard-tier
# sample so the masker always sees recognizable MPRAGE / MP2RAGE / T2 / FLAIR, not only random combos.
_STANDARD_PROTOCOLS = [
    (dict(item["config"]), item["name"])
    for item in STAGES["standard_protocols"]["protocols"]
]

# The inherited MP2RAGE implementation is organized into four physical feature
# families.  A lower-feature curriculum draw varies exactly ONE family while the
# calibrated n-D renderer supplies the others.  Across draws this is a superset
# of every production key emitted by ``augmentation.mp2rage.sample_params``;
# within a draw it avoids stacking unrelated endpoint extremes into an image no
# scanner would plausibly produce.
_MP2RAGE_LOWER_FEATURE_GROUPS = {
    "extracranial_contrast": (
        "skull_contrast_on", "skull_contrast",
    ),
    "brain_transfer": (
        "brain_on", "brain_contrast", "lut_xs", "lut_ys", "lut_per_region",
    ),
    "background": (
        "bg_on", "bg_model", "bg_thr_frac", "bg_amp", "bg_grain_sg", "bg_seed",
        "bg_effective_coils", "bg_ratio_inv2_scale", "bg_fov_zero_prob",
        "bg_fov_zero_frac",
    ),
    "tissue_noise": (
        "noise_on", "noise_sigma", "noise_grain_sg", "noise_aniso", "noise_seed",
    ),
}

# Policy 12 protects a small canonical MP2RAGE reference without changing the
# meaning of the three user-facing conditional fractions.  Those fractions are
# still exact shares of *all* dedicated MP2RAGE draws; the anchor is carved out
# only of whatever ordinary share remains.  Consequently a valid conditional
# sum in (0.90, 1.0] stays valid instead of failing a surprising hidden
# ``+ 0.10`` constraint.
_MP2RAGE_CLEAN_ANCHOR_FRACTION = 0.10
_MP2RAGE_TARGET_STYLE_MAX_GLOBAL_FRACTION = 0.035
_MP2RAGE_TARGET_STYLE_GLOBAL_TOLERANCE = 1e-12
_MP2RAGE_MORPH_MODES = ("identity", "global", "posterior_local")
# Dedicated policy-12 MP2RAGE keeps global anatomy deliberately gentler than the
# broad benign tier.  The earlier U(.20, .45) setting could expand a 224^3 brain by
# ~27%, which is a learnable synthetic size prior rather than realistic shape
# diversity.  A strict support-volume gate is local to this curriculum; legacy
# broad-tier morphology retains its existing superset range.
_POLICY12_MP2RAGE_GLOBAL_MORPH_STRENGTH = (0.08, 0.18)
_POLICY12_MP2RAGE_GLOBAL_VOLUME_RATIO = (0.85, 1.15)

# Opt-in domain-mixture identity.  This is deliberately separate from the
# policy-12 renderer/spec identity: it changes which top-level training domains
# are sampled, not the MP2RAGE appearance renderer itself.
MIXED_MP2RAGE_DOMAIN_MIX_VERSION = 1
_MIXED_NON_MP2_BASE_SHARES = MIXED_NON_MP2_BASE_SHARES

# Version-1 hard-artifact tail is a true tail of only the mixed curriculum's
# broad-benign and label-synthetic base.  The conditional cap keeps any one
# eligible tier from being dominated; the global cap keeps it a minority even
# when the dedicated MP2RAGE share is small.
_HARD_ARTIFACT_TAIL_MAX_CONDITIONAL_FRACTION = 0.10
_HARD_ARTIFACT_TAIL_MAX_GLOBAL_FRACTION = 0.05
_HARD_ARTIFACT_TAIL_GLOBAL_TOLERANCE = 1e-12


def _hard_artifact_tail_gate(draw, hard_fraction, regular_fraction):
    """Map one eligible-tier draw to disjoint hard / regular / none outcomes.

    The probabilities are exactly H, (1-H)*P, and (1-H)*(1-P).  This helper is
    used only when H is positive; omission/zero remains on the historical
    artifact-gate code path and therefore consumes the exact legacy RNG stream.
    """
    value = float(draw)
    hard = float(hard_fraction)
    regular = float(regular_fraction)
    if not np.isfinite(value) or not 0.0 <= value < 1.0:
        raise ValueError("draw must be finite and in [0, 1)")
    if not np.isfinite(hard) or not 0.0 <= hard < 1.0:
        raise ValueError("hard_fraction must be finite and in [0, 1)")
    if not np.isfinite(regular) or not 0.0 <= regular <= 1.0:
        raise ValueError("regular_fraction must be finite and in [0, 1]")
    if value < hard:
        return "hard"
    conditional = (value - hard) / (1.0 - hard)
    return "regular" if conditional < regular else "none"


def _mixed_mp2rage_top_level_route(draw, mixed_mp2rage_fraction):
    """Map one literal top-level uniform draw to MP2RAGE or a non-MP base tier.

    The non-MP route is obtained by conditioning the same draw on ``draw >= M``;
    no second tier coin is hidden here.  Branch-internal samplers remain free to
    draw their own acquisition/morph/artifact parameters.
    """
    value = float(draw)
    fraction = float(mixed_mp2rage_fraction)
    if not np.isfinite(value) or not 0.0 <= value < 1.0:
        raise ValueError("draw must be finite and in [0, 1)")
    if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("mixed_mp2rage_fraction must be finite and in [0, 1]")
    if value < fraction:
        return "mp2rage"
    if fraction >= 1.0:  # unreachable for a valid draw; keeps the division explicit/safe.
        return "mp2rage"
    conditional = (value - fraction) / (1.0 - fraction)
    cumulative = 0.0
    for name, share in _MIXED_NON_MP2_BASE_SHARES.items():
        cumulative += float(share)
        if conditional < cumulative or name == "label_synth":
            return name
    raise RuntimeError("unreachable mixed-domain routing state")


def _mp2rage_route_shares(superset_fraction, lower_feature_fraction,
                          posterior_fossa_fraction, target_style_fraction=0.0):
    """Exact policy-12 conditional shares, including the protected clean anchor."""
    conditional = {
        "superset": float(superset_fraction),
        "lower_feature": float(lower_feature_fraction),
        "posterior_fossa": float(posterior_fossa_fraction),
    }
    # Keep the zero-valued legacy helper result byte/API identical. A positive
    # stationary-style route is inserted after posterior-fossa and is carved
    # exclusively from the former ordinary remainder.
    if float(target_style_fraction) > 0.0:
        conditional["target_style_v6"] = float(target_style_fraction)
    routed = float(sum(conditional.values()))
    remaining = max(0.0, 1.0 - routed)
    anchor = min(_MP2RAGE_CLEAN_ANCHOR_FRACTION, remaining)
    return {
        **conditional,
        "clean_anchor": anchor,
        "ordinary": max(0.0, remaining - anchor),
    }


def _sample_mp2rage_route(rng, superset_fraction, lower_feature_fraction,
                          posterior_fossa_fraction, target_style_fraction=0.0):
    """Draw one mutually-exclusive MP2RAGE phenotype from exact policy shares."""
    shares = _mp2rage_route_shares(
        superset_fraction, lower_feature_fraction, posterior_fossa_fraction,
        target_style_fraction)
    draw = float(rng.random())
    cumulative = 0.0
    names = (["superset", "lower_feature", "posterior_fossa"]
             + (["target_style_v6"] if float(target_style_fraction) > 0.0 else [])
             + ["clean_anchor", "ordinary"])
    for name in names:
        cumulative += shares[name]
        if draw < cumulative or name == "ordinary":
            return name
    raise RuntimeError("unreachable MP2RAGE routing state")


def _sample_mp2rage_morph_mode(rng):
    """One independent equal-probability anatomy coin for a non-anchor MP2RAGE draw."""
    draw = float(rng.random())
    return _MP2RAGE_MORPH_MODES[min(int(draw * len(_MP2RAGE_MORPH_MODES)),
                                    len(_MP2RAGE_MORPH_MODES) - 1)]


def _policy12_global_morph(image, mask, rng):
    """One guarded policy-12 global co-warp, with deterministic safe fallback.

    Two attempts start from the same source pair: the sampled gentle strength,
    then half that strength if the first candidate leaves the 0.85--1.15 support
    volume band.  Each attempt remains a joint image/mask resampling owned by
    ``morph_image``; rejected candidates are discarded rather than compounded.
    The RNG sequence is deterministic, and identity is the final fallback.
    """
    from augmentations.anatomy import morph_image

    source_image = np.asarray(image, dtype=np.float32)
    source_mask = np.asarray(mask, dtype=bool)
    strength = float(rng.uniform(*settings("mp2rage_morphology")["strength_range"]))
    # Reinstantiate the same child stream for a retry so every dial/mode/anchor is
    # identical and only its amplitude changes. Reusing the advanced parent RNG
    # would draw a different anatomy and would not be a deterministic shrink.
    child_seed = int(rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64))
    low, high = _POLICY12_MP2RAGE_GLOBAL_VOLUME_RATIO
    for factor in (1.0, 0.5):
        candidate_image, candidate_mask = morph_image(
            source_image, source_mask, np.random.default_rng(child_seed),
            strength=strength * factor)
        if _valid_final_geometry(
                source_mask, candidate_mask,
                min_volume_ratio=low, max_volume_ratio=high):
            return (np.asarray(candidate_image, dtype=np.float32),
                    np.asarray(candidate_mask, dtype=bool))
    return source_image.copy(), source_mask.copy()


def _sample_mp2rage_lower_feature_probe(rng):
    """Sample one inherited production feature family for the n-D renderer."""
    from augmentations.protocols.mp2rage import sample_params

    sampled = sample_params(rng)
    assigned = {key for keys in _MP2RAGE_LOWER_FEATURE_GROUPS.values() for key in keys}
    metadata = {
        key for key in sampled
        if key == "lut_k" or (key.startswith("lut_y") and key[5:].isdigit())
    }
    unassigned = set(sampled) - assigned - metadata
    missing = assigned - set(sampled)
    if unassigned or missing:
        raise RuntimeError(
            "MP2RAGE lower-feature schema drift: "
            f"unassigned={sorted(unassigned)}, missing={sorted(missing)}")
    names = tuple(_MP2RAGE_LOWER_FEATURE_GROUPS)
    name = names[int(rng.integers(0, len(names)))]
    return name, {key: sampled[key] for key in _MP2RAGE_LOWER_FEATURE_GROUPS[name]}


@dataclass
class _RenderedSample:
    image: np.ndarray
    mask: np.ndarray
    kind: str
    morph_ok: bool = False
    spatial_ok: bool = False
    artifact_ok: bool = False
    mp2rage_background: bool = False
    mp2rage_noise_superset: bool = False
    mp2rage_noise_voxel_sizes: tuple | None = None
    target_style_seed: int | None = None
    resolution_applied: bool = False


def _select_training_route(draw, policy: SamplingPolicy):
    """Use the single top-level draw; dedicated zero still selects its base."""
    if policy.mixed_domain:
        return _mixed_mp2rage_top_level_route(draw, policy.dedicated_fraction)
    if policy.benign_only:
        mp2_cut = policy.dedicated_fraction
        clean_cut = mp2_cut + (1.0 - mp2_cut) * 0.25
        standard_cut = clean_cut + (1.0 - mp2_cut) * 0.25
        benign_cut = 1.0
    else:
        mp2_cut = 0.0
        clean_cut = policy.p_clean
        standard_cut = clean_cut + policy.p_standard
        benign_cut = standard_cut + policy.p_benign
    if policy.benign_only and draw < mp2_cut:
        return "mp2rage"
    if draw < clean_cut:
        return "clean"
    if draw < standard_cut:
        return "other_standard"
    if policy.benign_only or draw < benign_cut:
        return "broad_benign"
    return "label_synth"


def _render_mp2rage_sample(source_img, source_mask, rng, policy, voxel_sizes):
    """Render one disjoint policy-12 appearance and its paired anatomy."""
    from augmentations.protocols.acquisition import realistic_acquisition, sample_config
    superset_fraction = policy.superset_fraction
    lower_feature_fraction = policy.lower_feature_fraction
    posterior_fraction = policy.posterior_fraction
    target_style_fraction = policy.target_style_fraction
    morph_ok = spatial_ok = artifact_ok = False
    target_style_seed = None
    mp2rage_background = mp2rage_noise_superset = False
    mp2rage_noise_voxel_sizes = None
    if target_style_fraction > 0.0:
        route = _sample_mp2rage_route(
            rng, superset_fraction, lower_feature_fraction, posterior_fraction,
            target_style_fraction)
    else:
        # Deliberately retain the historical four-argument call when the
        # new route is omitted/zero; this also protects monkeypatched and
        # external route probes as part of the no-RNG-change contract.
        route = _sample_mp2rage_route(
            rng, superset_fraction, lower_feature_fraction, posterior_fraction)
    if route == "clean_anchor":
        # This is an acquisition anchor, not a copy of the input T1.  A fixed 3 T Siemens config,
        # clean renderer, and identity mask path make it a stable recognizable MP2RAGE reference.
        # It consumes only the ordinary share left after the explicit conditional routes.
        cfg = dict(_STANDARD_PROTOCOLS[1][0])
        img, m, _ = realistic_acquisition(
            source_img, source_mask, rng, cfg=cfg, clean=True, geometry=False,
            voxel_sizes=voxel_sizes, apply_resolution=False)
        # Preserve the public protocol label used by validation/gallery callers.  The exact
        # phenotype remains testable through the route helper and the renderer's clean flag.
        kind = "MP2RAGE"
        morph_ok = False
        spatial_ok = False
    else:
        # Fix the sequence BEFORE the field draw. Mutating an unconstrained config after sampling
        # used to create mislabeled low-field MP2RAGE examples; this sampler restricts the sequence
        # to its supported 3 T+ fields.
        cfg = sample_config(rng, sequence="MP2RAGE")
        # Explicit activation prevents generic/legacy MP2RAGE callers from silently inheriting
        # policy-12 thermal spectra or the mild continuous global acquisition-law guard.  This
        # marker is shared by ordinary/superset/lower/PF draws; a PF spec adds its regional
        # hard-tail settings. The canonical clean anchor intentionally remains neutral.
        cfg["mp2rage_policy_version"] = 12
        # Appearance route and anatomy mode are separate draws.  This prevents the network from
        # learning that every posterior-fossa intensity phenotype has a locally changed outline.
        morph_mode = _sample_mp2rage_morph_mode(rng)
        pre_img, pre_mask = source_img.copy(), source_mask.copy()
        if morph_mode != "identity" and enabled("mp2rage_morphology"):
            from augmentations.anatomy import morph_image
            if morph_mode == "global":
                pre_img, pre_mask = _policy12_global_morph(
                    pre_img, pre_mask, rng)
            else:
                # strength=0 makes this a genuinely local-only field; morph_image still composes
                # and resamples image/mask together under its topology and volume guards.
                pre_img, pre_mask = morph_image(
                    pre_img, pre_mask, rng, strength=0.0,
                    posterior_fossa=True, voxel_sizes=voxel_sizes)

        # OPT-IN over-generation. `realistic_acquisition` renders a realistic MP2RAGE by contract,
        # so its default band is narrow and never folds. Conditional routes remain disjoint.
        # `kind` deliberately stays "MP2RAGE" for clean/ordinary/superset/lower-feature bands:
        # they feed `_category_is_faithful` and the gallery renderer's expected-protocol check.
        lower_params = None
        posterior_spec = None
        # Ordinary, over-generated, and inherited lower-feature MP2 draws all
        # receive one final-grid noise phenotype after pose/resolution.  The
        # clean anchor is exact, posterior-fossa already owns a bounded native-
        # grid noise package, and target-style v6 owns its own post-spatial
        # stationary residual, so none of those routes may double-add here.
        try:
            mp2rage_noise_voxel_sizes = tuple(
                float(value) for value in voxel_sizes)
        except (TypeError, ValueError):
            # Direct array callers historically omit spacing; their canonical
            # training grid is 1 mm. Explicit non-1-mm inputs are left unchanged
            # because this v1 spectrum is calibrated only at native 1 mm.
            mp2rage_noise_voxel_sizes = ((1.0, 1.0, 1.0)
                                         if voxel_sizes is None else None)
        mp2rage_noise_superset = (
            route in ("ordinary", "superset", "lower_feature")
            and mp2rage_noise_voxel_sizes == (1.0, 1.0, 1.0))
        if route == "target_style_v6":
            # The parent stream owns only one seed draw. The stationary
            # proposal/gates use their deterministic child stream later,
            # after all joint spatial augmentation has completed.
            target_style_seed = int(rng.integers(
                0, np.iinfo(np.int64).max, dtype=np.int64))
        if route == "superset":
            cfg["mp2rage_superset"] = True
        elif route == "lower_feature":
            _group, lower_params = _sample_mp2rage_lower_feature_probe(rng)
        elif route == "posterior_fossa":
            from augmentations.curricula.mp2rage_posterior import sample_posterior_fossa_spec
            posterior_spec = sample_posterior_fossa_spec(
                rng, field_strength=float(cfg["field"]), voxel_sizes=voxel_sizes)

        if posterior_spec is not None:
            # Pose remains joint image/mask and precedes the coil-fixed regional field and PSF.
            # The broad full-360 spatial path is bypassed so posterior/inferior stays meaningful.
            from augmentations.curricula.mp2rage_posterior import apply_posterior_fossa_pose
            pre_img, pre_mask = apply_posterior_fossa_pose(
                pre_img, pre_mask, posterior_spec, voxel_sizes=voxel_sizes)
            img, m, _ = realistic_acquisition(
                pre_img, pre_mask, rng, cfg=cfg, clean=False, geometry=False,
                voxel_sizes=voxel_sizes, apply_resolution=False,
                mp2rage_posterior_fossa=posterior_spec)
            kind = "MP2RAGE:posterior_fossa"
            spatial_ok = False
        else:
            img, m, _ = realistic_acquisition(
                pre_img, pre_mask, rng, cfg=cfg, clean=False, geometry=False,
                voxel_sizes=voxel_sizes, apply_resolution=False,
                mp2rage_params=lower_params)
            kind = ("MP2RAGE:target_style_v6"
                    if route == "target_style_v6" else "MP2RAGE")
            spatial_ok = True
            mp2rage_background = True
        # Anatomy has already taken its independent policy-12 mode.  Do not run the shared 70%
        # morph gate again. Artifact injection remains disabled for the benign-only curriculum.
        morph_ok = False
    return _RenderedSample(
        img, m, kind, morph_ok, spatial_ok, artifact_ok,
        mp2rage_background, mp2rage_noise_superset, mp2rage_noise_voxel_sizes,
        target_style_seed)


def _render_training_route(route_name, scan01, mask, source_img, source_mask,
                           rng, policy, donor, voxel_sizes):
    """Create the tier image/mask and declare which later stages it permits."""
    from augmentations.appearance import realistic_augment
    from augmentations.label_synthesis import synthesize_from_labels
    from augmentations.protocols.acquisition import realistic_acquisition, sample_config

    legacy_anchors = policy.legacy_anchors
    benign_only, mixed_domain = policy.benign_only, policy.mixed_domain
    morph_ok = spatial_ok = artifact_ok = False
    if route_name == "mp2rage":
        return _render_mp2rage_sample(source_img, source_mask, rng, policy, voxel_sizes)
    elif route_name == "clean":
        img = source_img.copy(); m = source_mask.copy()
        kind = "clean"
        # Default: a genuine identity anchor. Legacy mode restores the former morph + full-spatial
        # policy; the explicit benign-only curriculum keeps its promised identity anchors.
        morph_ok = legacy_anchors and not (benign_only or mixed_domain)
        spatial_ok = legacy_anchors and not (benign_only or mixed_domain)
    elif route_name == "other_standard":
        configured_protocols = [
            (item["config"], item["name"])
            for item in settings("standard_protocols")["protocols"]]
        protocols = ([item for item in configured_protocols if item[0]["sequence"] != "MP2RAGE"]
                     if (benign_only or mixed_domain) else configured_protocols)
        if enabled("standard_protocols"):
            cfg, name = protocols[int(rng.integers(len(protocols)))]
            img, m, _ = realistic_acquisition(scan01, mask, rng, cfg=dict(cfg), clean=True, geometry=False,
                                              voxel_sizes=voxel_sizes, apply_resolution=False)
            kind = name
        else:
            img, m, kind = source_img.copy(), source_mask.copy(), "clean"
        # `clean=True` disables the acquisition renderer's own resolution downsampling. Bypass the
        # downstream morph/full-orientation/FOV/resolution/artifact stages so this stays a crisp protocol
        # anchor. Legacy mode restores the former shared-stage behavior.
        morph_ok = legacy_anchors and not (benign_only or mixed_domain)
        spatial_ok = legacy_anchors and not (benign_only or mixed_domain)
        artifact_ok = legacy_anchors and not (benign_only or mixed_domain)
    elif route_name == "broad_benign":
        if enabled("realistic_acquisition") and rng.random() < settings("realistic_acquisition")["probability"]:
            if benign_only or mixed_domain:
                # ``F`` is documented as the GLOBAL MP2RAGE fraction. The broad-benign acquisition
                # branch used to draw MP2RAGE again from the generic sequence pool, silently making
                # the true share larger than F. Re-draw the config here so only the dedicated top-
                # level MP2RAGE branch contributes that sequence; all remaining benign acquisition
                # coverage stays broad across field strength/vendor/reconstruction and 9 protocols.
                cfg = sample_config(rng)
                while cfg["sequence"] == "MP2RAGE":
                    cfg = sample_config(rng)
                img, m, _ = realistic_acquisition(
                    scan01, mask, rng, cfg=cfg, geometry=False,
                    voxel_sizes=voxel_sizes, apply_resolution=False)
            else:
                # Preserve the original sampler/RNG sequence byte-for-byte when benign-only mode is
                # not enabled (realistic_acquisition owns its internal config draw).
                img, m, _ = realistic_acquisition(
                    scan01, mask, rng, geometry=False,
                    voxel_sizes=voxel_sizes, apply_resolution=False)
        else:
            img, m = (realistic_augment(scan01, mask, rng, donor=donor, geometry=False,
                                       resolution=False, **settings("realistic_appearance"))
                      if enabled("realistic_appearance") else (source_img.copy(), source_mask.copy()))
        kind = "benign"
        morph_ok = True
        spatial_ok = True
        artifact_ok = not benign_only
    else:
        img, m = (synthesize_from_labels(scan01, mask, rng, **settings("label_synthesis"))
                  if enabled("label_synthesis") else (source_img.copy(), source_mask.copy()))
        kind = "synthetic"
        # Label-driven synthesis already gave it a distinct anatomy; retain broad pose/resolution/artifacts.
        spatial_ok = True
        artifact_ok = True

    return _RenderedSample(img, m, kind, morph_ok, spatial_ok, artifact_ok)


def _postprocess_training_sample(sample, rng, policy):
    """Apply joint anatomy/pose, then final-grid MP2RAGE appearance stages."""
    # DISTINCT ANATOMY on the real-texture tiers: a moderate smooth warp (different lobe proportions /
    # ventricle size / asymmetry) keeping the real texture â€” so every sample is a new brain, not one of
    # ~125 NFBS outlines reposed. (The synthetic tier already morphs its label map; large anatomy changes
    # live there. Kept gentle here so it relocates, not smears.) Mask co-warps.
    if sample.morph_ok and enabled("morphology") and rng.random() < settings("morphology")["probability"]:
        from augmentations.anatomy import morph_image
        strength = float(rng.uniform(*settings("morphology")["strength_range"]))
        sample.image, sample.mask = morph_image(sample.image, sample.mask, rng, strength=strength)

    # Full pose/FOV/resolution augmentation belongs to the broad-coverage tiers. Clean/standard anchors
    # bypass it by default; legacy_anchor_augmentation opts them back into the former policy. Geometry is
    # centralized here (the tier calls pass geometry=False) and precedes acquisition artifacts.
    sample.resolution_applied = False
    if sample.spatial_ok:
        # One acquisition-resolution stage per sample. The tier renderers defer their own resolution
        # corruption here; if this stage fires, exclude the separate anisotropy artifact to prevent a
        # second downsample/upsample stack.
        sample.resolution_applied = enabled("resolution") and bool(
            rng.random() < settings("resolution")["probability"])
        sample.image, sample.mask = spatial_augment(
            sample.image, sample.mask, rng,
            p_geom=settings("orientation")["probability"] if enabled("orientation") else 0.0,
            p_res=1.0 if sample.resolution_applied else 0.0,
            mp2rage_background=sample.mp2rage_background)

    if sample.mp2rage_noise_superset and enabled("mp2rage_noise_superset"):
        # Draw only one child seed after all parent-owned rendering and spatial
        # decisions.  The child owns profile, spectrum, amplitude, and random
        # field draws, so the stage is replayable without coupling noise to route
        # selection or anatomy.  The wrapper never reads or changes the label.
        noise_seed = int(rng.integers(
            0, np.iinfo(np.int64).max, dtype=np.int64))
        from augmentations.curricula.mp2rage_noise_superset_v1 import apply_mp2rage_noise_superset_v1
        sample.image, sample.mask = apply_mp2rage_noise_superset_v1(
            np.asarray(sample.image, dtype=np.float32), sample.mask, seed=noise_seed,
            voxel_sizes_mm=sample.mp2rage_noise_voxel_sizes)

    if sample.target_style_seed is not None:
        # Strictly post-spatial: style is evaluated in the native final grid,
        # while the mask is returned byte-identically by the training-safe
        # wrapper. Dedicated MP2RAGE routes never enable artifact_ok, so this
        # phenotype cannot acquire a second synthetic quality signature.
        from augmentations.curricula.mp2rage_target_stationary_v6 import apply_stationary_v6_training_safe
        sample.image, sample.mask = apply_stationary_v6_training_safe(
            sample.image, sample.mask, seed=sample.target_style_seed,
            voxel_sizes_mm=policy.target_style_voxel_sizes)


def _apply_sample_artifacts(sample, rng, policy, route_name, voxel_sizes, return_kind):
    """Apply one disjoint hard/regular artifact outcome to an eligible tier."""
    hard_tail_eligible = bool(
        policy.hard_tail_fraction > 0.0 and policy.mixed_domain
        and route_name in ("broad_benign", "label_synth"))
    if hard_tail_eligible:
        # One categorical draw owns all three mutually exclusive outcomes.
        # This keeps the ordinary overlay at (1-H)*p rather than silently
        # stacking it on, or adding it on top of, the hard route.
        artifact_route = _hard_artifact_tail_gate(
            rng.random(), policy.hard_tail_fraction, float(policy.p_artifact))
        if artifact_route == "hard":
            hard_seed = int(rng.integers(
                0, np.iinfo(np.int64).max, dtype=np.int64))
            from augmentations.curricula.hard_artifact_tail import apply_hard_artifact_tail_v1
            sample.image, sample.mask, hard_record = apply_hard_artifact_tail_v1(
                sample.image, sample.mask, seed=hard_seed, voxel_sizes_mm=policy.hard_tail_voxel_sizes,
                exclude=tuple(name for name in _MASKER_ARTIFACT_NAMES if not enabled(name)) + (("anisotropy", "randomize_resolution")
                         if sample.resolution_applied else ()),
                return_record=True)
            if return_kind:
                profile = str(hard_record["requested_profile"])
                status = profile if hard_record["accepted"] else f"{profile}_veto"
                sample.kind = f"{sample.kind}+hard_tail_v1:{status}"
        elif artifact_route == "regular":
            sample.image, applied = apply_artifacts(
                sample.image, rng, mask=sample.mask, voxel_sizes=voxel_sizes,
                exclude=(("anisotropy",) if sample.resolution_applied else ()),
            )
            if applied and return_kind:
                sample.kind = f"{sample.kind}+{'+'.join(applied)}"
    elif sample.artifact_ok and rng.random() < float(policy.p_artifact):       # exact historical/off path
        sample.image, applied = apply_artifacts(
            sample.image, rng, mask=sample.mask, voxel_sizes=voxel_sizes,
            exclude=(("anisotropy",) if sample.resolution_applied else ()),
        )
        if applied and return_kind:
            sample.kind = f"{sample.kind}+{'+'.join(applied)}"


@configured_sample
def make_training_sample(scan01: np.ndarray, mask: np.ndarray, rng: np.random.Generator,
                         donor=None, p_clean: float = 0.18, p_standard: float = 0.22,
                         p_benign: float = 0.30, p_artifact: float = 0.30, return_kind: bool = False,
                         legacy_anchor_augmentation: bool = False, voxel_sizes=None,
                         benign_only_mp2rage_fraction=None, mp2rage_superset_fraction=0.0,
                         mp2rage_lower_feature_fraction=0.0,
                         mp2rage_posterior_fossa_fraction=0.0,
                         mixed_mp2rage_fraction=None,
                         mp2rage_target_style_fraction=0.0,
                         hard_artifact_tail_fraction=0.0):
    """Draw one image/mask pair, preserving the parent RNG sequence.

    The default mix contains clean identity, canonical protocols, benign real
    texture, and label synthesis. Dedicated benign-only and mixed MP2RAGE
    curricula select policy-12 appearances with disjoint optional stress routes.
    Explicit zero selects a dedicated curriculum's base; None keeps the default.

    Order is route/render -> joint anatomy/pose -> final-grid appearance ->
    artifacts -> geometry guard. Clean and standard anchors bypass later stages
    unless legacy_anchor_augmentation is enabled. Invalid final geometry falls
    back to the original image and mask. Returns (image, mask[, kind]).
    """
    source_img = np.clip(np.asarray(scan01, np.float32), 0.0, 1.0)
    source_mask = np.asarray(mask) > 0.5
    draw = rng.random()
    policy = resolve_sampling_policy({
        "p_clean": p_clean, "p_standard": p_standard, "p_benign": p_benign,
        "p_artifact": p_artifact, "legacy_anchor_augmentation": legacy_anchor_augmentation,
        "benign_only_mp2rage_fraction": benign_only_mp2rage_fraction,
        "mixed_mp2rage_fraction": mixed_mp2rage_fraction,
        "mp2rage_superset_fraction": mp2rage_superset_fraction,
        "mp2rage_lower_feature_fraction": mp2rage_lower_feature_fraction,
        "mp2rage_posterior_fossa_fraction": mp2rage_posterior_fossa_fraction,
        "mp2rage_target_style_fraction": mp2rage_target_style_fraction,
        "hard_artifact_tail_fraction": hard_artifact_tail_fraction,
    }, voxel_sizes=voxel_sizes, require_geometry=True)
    route = _select_training_route(draw, policy)
    sample = _render_training_route(
        route, scan01, mask, source_img, source_mask, rng, policy, donor, voxel_sizes)
    _postprocess_training_sample(sample, rng, policy)
    _apply_sample_artifacts(sample, rng, policy, route, voxel_sizes, return_kind)
    _checked_sample_result((sample.image, sample.mask), source_img)
    if not _valid_final_geometry(source_mask, sample.mask):
        sample.image, sample.mask = source_img.copy(), source_mask.copy()
        sample.kind = "clean_geometry_fallback"
    out = (sample.image, sample.mask, sample.kind)
    return out if return_kind else out[:2]


# =================================================================================================
# Per-category VALIDATION samples (the masker's `--masker-val-aug` Dice breakdown)
# =================================================================================================

# The masker's aggregate val Dice is measured on the REAL, un-synthesized scan, which says nothing
# about which part of the training distribution the model is weak on. These categories split that
# question into forced tiers of `make_training_sample` â€” same generators, same ranges, only the tier
# draw is pinned â€” so a per-category Dice is directly comparable to the training mix rather than to a
# separate, hand-rolled eval pipeline. The posterior-fossa tier is activated only with its curriculum.
MASKER_VAL_CATEGORIES = (
    "benign", "mp2rage", "mp2rage_posterior_fossa", "nonbenign")

_VAL_CATEGORY_KWARGS: Dict[str, Dict[str, Any]] = {
    # Healthy real-texture acquisition/anatomy variation (the BENIGN tier), artifact overlay OFF.
    # `realistic_acquisition` still draws freely here, so a minority of these ARE MP2RAGE â€” this is
    # the benign distribution as trained, and `mp2rage` below is the dedicated zoom-in on one slice
    # of it, not a disjoint partition.
    "benign": {"p_clean": 0.0, "p_standard": 0.0, "p_benign": 1.0, "p_artifact": 0.0},
    # The dedicated MP2RAGE branch of the benign-only curriculum: F=1.0 sends EVERY draw down it.
    # Artifacts are disabled by that curriculum (`artifact_ok = not benign_only`), which is what we
    # want â€” this group isolates the SEQUENCE, not sequence-plus-degradation.
    "mp2rage": {"benign_only_mp2rage_fraction": 1.0},
    # The difficult posterior-fossa acquisition route, pinned independently from
    # canonical MP2RAGE so checkpoint selection can see the stress suite.
    "mp2rage_posterior_fossa": {
        "benign_only_mp2rage_fraction": 1.0,
        "mp2rage_posterior_fossa_fraction": 1.0,
    },
    # The same benign texture tier with the QC-impacting artifact overlay ALWAYS firing. Every stage
    # before the artifact gate consumes the identical rng draws as `benign` (the gate itself takes one
    # draw either way), so given the SAME seed the two are literally one scan with and without
    # artifacts and their Dice gap is attributable to the artifacts alone. Callers that want that
    # pairing must therefore NOT fold the category name into the seed.
    "nonbenign": {"p_clean": 0.0, "p_standard": 0.0, "p_benign": 1.0, "p_artifact": 1.0},
}


def _category_is_faithful(category: str, kind: str) -> bool:
    """Did a draw actually produce the category it was ASKED for?

    Two paths inside `make_training_sample` can hand back something other than the requested tier,
    and both would silently poison the per-category metric:
      * `clean_geometry_fallback` â€” the final supervision guard replaced the sample with the pristine
        source, so a `benign`/`nonbenign` slot would hold an untouched clean scan and pull that
        category's Dice up toward the clean number (it reads as good news, which is the danger).
      * no artifact applied â€” `p_artifact=1.0` fires the GATE, but `apply_artifacts` can still record
        nothing (the finite-guard drops a draw; `anisotropy` is excluded whenever the resolution
        stage already ran). `kind` carries a `+name` suffix per APPLIED artifact, so its absence is
        the direct signal that a `nonbenign` sample is actually clean.
    """
    if kind == "clean_geometry_fallback":
        return False
    if category == "mp2rage_posterior_fossa":
        return kind == "MP2RAGE:posterior_fossa"
    if category == "nonbenign":
        return "+" in kind
    return True


def make_eval_sample(scan01, mask, rng: np.random.Generator, category: str, *, voxel_sizes=None,
                     max_draws: int = 3):
    """One VALIDATION sample of a FORCED `category` (see `MASKER_VAL_CATEGORIES`).

    Delegates to `make_training_sample` with the tier probabilities PINNED, so validation samples come
    out of the exact generators training uses â€” no parallel eval-only synthesis path to drift.

    Redraws (up to `max_draws`) when the draw did not honour the category, then accepts the last one
    rather than looping forever. Returns `(img, mask, kind)`; `kind` names the tier and any applied
    artifacts, and callers should count `not _category_is_faithful(...)` returns so an unfaithful
    group is visible instead of silently mixed in.

    `rng` should be seeded deterministically by the caller â€” a validation set that moves between
    epochs reports resampling noise as model change.
    """
    try:
        pinned = _VAL_CATEGORY_KWARGS[category]
    except KeyError:
        raise ValueError(
            f"unknown masker validation category {category!r}; expected one of "
            f"{MASKER_VAL_CATEGORIES}") from None
    img = m = None
    kind = ""
    for _ in range(max(1, int(max_draws))):
        img, m, kind = make_training_sample(scan01, mask, rng, return_kind=True,
                                            voxel_sizes=voxel_sizes, **pinned)
        if _category_is_faithful(category, kind):
            break
    return img, m, kind


def sample_norm_region(mask, rng, voxel_sizes=None):
    """Per-sample z-score NORMALIZATION region for masker training â€” covers the region the DEPLOYED passes
    actually use, instead of always the PERFECT GT mask.

    predict_mask normalizes pass-1 over the FOREGROUND (no mask exists yet) and optional pass-2 over an
    IMPERFECT rough mask. Training that always z-scores in the exact GT mask therefore feeds the net brain
    statistics the maskless first pass never reproduces (the same ~1 z-unit train/serve mismatch just fixed
    for QC). This returns a per-sample region so the training z-score DISTRIBUTION covers both passes:
    30% foreground (None -> pass-1), 20% exact GT (the well-refined case), 50% eroded/dilated/shifted GT
    (an imperfect rough mask -> pass-2). Opt-in (masker `--masker-norm-aug`); OFF reproduces GT-only exactly.
    Returns None (foreground) or a boolean mask; feed straight to `zscore`."""
    from scipy import ndimage as ndi
    m = np.asarray(mask) > 0.5
    if not m.any():
        return None
    r = float(rng.random())
    if r < 0.30:
        return None                                    # foreground -> inference pass 1
    if r < 0.50:
        return m                                        # exact GT -> the clean / well-refined case
    op = float(rng.random())                            # imperfect rough mask -> inference pass 2
    spacing = tuple(float(x) for x in (voxel_sizes or (1.0,) * m.ndim))
    delta_mm = float(rng.uniform(1.0, 4.0))
    if op < 0.4:
        p = ndi.distance_transform_edt(m, sampling=spacing) > delta_mm
    elif op < 0.8:
        p = m | (ndi.distance_transform_edt(~m, sampling=spacing) <= delta_mm)
    else:
        # Zero-filled physical translation. np.roll wrapped a shifted brain onto the opposite FOV face.
        sh = tuple(int(round(float(rng.uniform(-4.0, 4.0)) / max(s, 1e-6))) for s in spacing)
        p = np.zeros_like(m)
        src = tuple(slice(max(0, -d), min(n, n - d)) for d, n in zip(sh, m.shape))
        dst = tuple(slice(max(0, d), min(n, n + d)) for d, n in zip(sh, m.shape))
        p[dst] = m[src]
    return p if p.any() else m


def discover_pairs(input_dir, scan_glob: str = "*_T1w.nii.gz",
                   mask_suffix: str = "_brainmask_refined_crf") -> List[Dict[str, str]]:
    """Discover (full-head scan, brain mask) pairs from a RAW directory â€” the input for ON-THE-FLY
    online synthesis (train_model.py --synth-online), so nothing is pre-generated to disk.

    Returns ``[{"scan", "mask", "source_scan"}]`` (source_scan == scan, so the trainer's
    subject-grouped split treats each raw scan as one subject)."""
    input_dir = Path(input_dir)
    found = sorted(input_dir.rglob(scan_glob))
    scans = [p for p in found if "brainmask" not in p.name and "_brain." not in p.name]
    pairs: List[Dict[str, str]] = []
    for sc in scans:
        stem = sc.name
        for ext in (".nii.gz", ".nii"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        cand = sc.with_name(f"{stem}{mask_suffix}.nii.gz")
        if not cand.is_file():
            raise FileNotFoundError(f"Missing requested mask for {sc}: expected {cand}; set mask_suffix to match your labels")
        pairs.append({"scan": str(sc), "mask": str(cand), "source_scan": str(sc)})
    return pairs
