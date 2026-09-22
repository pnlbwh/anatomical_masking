"""Simulation of MRI acquisitions across sequences, scanners, and field strengths.

The label-synth stream maximizes feature COVERAGE (looks synthetic by design). This complements it
with a MINORITY of recognizably-REAL acquisitions: a real T1 rendered as a specific, plausible
{field strength Ã— sequence Ã— vendor Ã— settings Ã— acceleration/recon} configuration, KEEPING the real
tissue texture (soft-membership remap + add-back detail). So the training set isn't all synthetic â€”
it also contains realistic 1.5T MPRAGE, 3T FLAIR, 7T MP2RAGE, 0.55T T2, etc.

Axes simulated (the real acquisition space):
  * Field strength (0.064â€“11.7T) -> SNR/noise (down with field), B1 bias (up with field), contrast.
  * Sequence (MPRAGE/MP2RAGE/SPGR/TSE/SPACE/FLAIR/STIR/DIR/PSIR/GRE-SWI) -> per-tissue contrast.
  * Vendor (Siemens/GE/Philips/Canon/UnitedImaging/Fujifilm) -> small default/flip-angle jitter.
  * Settings (TR/TE/TI/flip ~ the sequence targets; voxel/slice/FOV -> resolution).
  * Acceleration/recon (GRAPPA/SENSE/SMS/CS/DL-recon) -> noise texture / sharpness / mild ringing.

n-D: runs on a 2D review slice and a 3D training volume (geometry is 3D-only -> `geometry=False` in 2D).
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Tuple

import numpy as np
from scipy import ndimage as ndi
from augmentations.config import settings

# field strength T -> (in-brain noise sigma [SNR], B1 bias strength, contrast scale)
_FIELD = {
    0.064: (0.10, 0.12, 0.85), 0.55: (0.060, 0.20, 0.92), 1.5: (0.030, 0.30, 1.00),
    3.0: (0.018, 0.45, 1.00), 5.0: (0.015, 0.65, 1.02), 7.0: (0.013, 0.85, 1.05),
    9.4: (0.012, 1.00, 1.07), 10.5: (0.011, 1.10, 1.08), 11.7: (0.011, 1.20, 1.10),
}
# sequence -> (csf, gm, wm target, skull/scalp compression, tag)
_SEQ = {
    "MPRAGE":    (0.12, 0.45, 0.75, 0.85, ""),
    "MP2RAGE":   (0.12, 0.42, 0.70, 0.55, "mp2bg"),
    "SPGR":      (0.28, 0.46, 0.62, 0.85, ""),
    "TSE_T2":    (0.90, 0.44, 0.24, 0.80, ""),   # T2: WM genuinely DARK, GM medium, CSF bright
    "SPACE_T2":  (0.88, 0.44, 0.24, 0.78, ""),   # (darker parenchyma; was washed-out)
    "FLAIR":     (0.07, 0.62, 0.48, 0.55, ""),
    "STIR":      (0.90, 0.60, 0.42, 0.05, ""),
    "DIR":       (0.10, 0.74, 0.14, 0.40, ""),
    "PSIR":      (0.15, 0.32, 0.85, 0.60, ""),
    "GRE_SWI":   (0.30, 0.52, 0.58, 0.60, "swi"),
}
_VENDORS = ["Siemens", "GE", "Philips", "Canon", "UnitedImaging", "Fujifilm"]
_RECON = ["none", "GRAPPA", "SENSE", "SMS", "CompressedSensing", "DLRecon"]

# --- MP2RAGE / UNI n-D renderer knobs (consumed by the `mp2bg` branch below) -------------------
# `master_t` sweep for the UNI look. 1.0 (= artifacts3d.MP2RAGE_3D_MARK) matches a real
# MP2RAGE. Keep this *realistic acquisition* path near that mark; the explicit
# `mp2rage_3d` augmentation still exposes the deliberately over-generated tail to 1.3.
# `clean=True` is the textbook standard-protocol tier, so it stays pinned near the mark and
# never folds â€” a canonical example must stay canonical and recognizable.
_MP2RAGE_T_RANGE = (0.92, 1.04)
_MP2RAGE_T_CLEAN = (1.0, 1.0)
# Mild log-slope jitter around a coherent mix of two measured real-UNI curves.
_MP2RAGE_REF_SPREAD = 0.020
# A UNI image remains T1-like over normal brain-tissue T1 values. A PSIR/INV2-like
# non-monotone fold is available as an explicit superset augmentation, but is not
# sampled by a function whose contract is a realistic MP2RAGE acquisition.
_MP2RAGE_FOLD_PROB = 0.0
# The OVER-GENERATED tail, reached only when a caller sets `cfg["mp2rage_superset"]` for that sample
# (see `augmentations.pipeline.make_training_sample(mp2rage_superset_fraction=...)`). Runs to
# `artifacts3d.MP2RAGE_3D_MAX`, widens the reference-histogram family, and allows the PSIR/INV2-like
# non-monotone fold that the realistic band deliberately never emits.
_MP2RAGE_T_SUPERSET = (1.00, 1.30)
_MP2RAGE_REF_SPREAD_SUPERSET = 0.060
_MP2RAGE_FOLD_PROB_SUPERSET = 0.35
# Fraction of the magnitude-image B1 bias that survives the UNI ratio. Not 0: B1+ TRANSMIT
# non-uniformity (strong at 7T, where MP2RAGE is most used) is not cancelled by the combination.
_MP2RAGE_BIAS_RESIDUAL = 0.10
# MP2RAGE is overwhelmingly a 3 T / 7 T acquisition.  Keep 5 T and the
# research-only ultra-high-field systems reachable, but do not give an 11.7 T
# experiment the same prior probability as a clinical 3 T scan.
_MP2RAGE_FIELDS = (3.0, 5.0, 7.0, 9.4, 10.5, 11.7)
_MP2RAGE_FIELD_PROBABILITIES = (0.50, 0.05, 0.42, 0.015, 0.010, 0.005)

# Reference-balanced stored-UNI appearances.  The renderer still uses only the
# two measured quantile anchors in ``augmentations.artifacts.volume``.  Display
# screenshots merely bound noise/contrast/context ranges; they are never loaded
# by the training process and do not define a third histogram target.
_MP2RAGE_SITE_STYLES = ("canonical", "unit1", "high_contrast", "grainy")
_MP2RAGE_SITE_PROBABILITIES = (0.20, 0.35, 0.20, 0.25)
_MP2RAGE_SITE_ALIASES = {
    "clean": "canonical",
    "normal": "unit1",
    "normal_unit1": "unit1",
    "high-contrast": "high_contrast",
    "screenshot": "grainy",
    "screenshot_like": "grainy",
}
_RECON_NOISE_FACTOR = {
    "none": 1.00, "GRAPPA": 1.25, "SENSE": 1.35, "SMS": 1.20,
    "CompressedSensing": 0.70, "DLRecon": 0.40,
}
_RECON_MODE_FACTOR = {
    "none": 1.00, "GRAPPA": 0.85, "SENSE": 0.80, "SMS": 0.90,
    "CompressedSensing": 0.90, "DLRecon": 1.00,
}

# Only the inherited, physically meaningful lower-level controls may enter the
# calibrated n-D renderer.  In particular, orchestration controls such as
# ``master_t``, ``air_mode`` and ``boundary_width`` remain owned here so a
# lower-feature probe cannot silently replace the realistic MP2RAGE pipeline.
_MP2RAGE_LOWER_PARAM_KEYS = frozenset({
    "skull_contrast_on", "skull_contrast",
    "brain_on", "brain_contrast", "lut_xs", "lut_ys", "lut_per_region",
    "bg_on", "bg_model", "bg_thr_frac", "bg_amp", "bg_grain_sg", "bg_seed",
    "bg_thermal_fraction",
    "bg_effective_coils", "bg_ratio_inv2_scale", "bg_fov_zero_prob",
    "bg_fov_zero_frac",
    "noise_on", "noise_sigma", "noise_grain_sg", "noise_aniso",
    "noise_thermal_fraction", "noise_seed",
})


def _validated_mp2rage_lower_params(params):
    """Return an allowlisted copy of one inherited MP2RAGE feature-group probe."""
    if params is None:
        return {}
    if not isinstance(params, Mapping):
        raise TypeError("mp2rage_params must be a mapping or None")
    unknown = set(params) - _MP2RAGE_LOWER_PARAM_KEYS
    if unknown:
        names = ", ".join(sorted(str(k) for k in unknown))
        raise ValueError(f"unknown mp2rage_params key(s): {names}")
    return {key: params[key] for key in params}


def _resolve_mp2rage_site_style(rng, cfg, *, clean=False):
    """Resolve one coherent MP2RAGE site style and its grain strength.

    A named ``mp2rage_site_style`` is the new replay control.  The historical
    ``mp2rage_grainy_site_style`` / ``mp2rage_grainy_site_strength`` pair remains
    authoritative so frozen diagnostic configurations continue to work.  A
    canonical clean draw never samples a style.
    """
    if clean:
        return "canonical", 0.0

    if "mp2rage_site_style" in cfg:
        raw = str(cfg["mp2rage_site_style"]).strip().lower().replace(" ", "_")
        style = _MP2RAGE_SITE_ALIASES.get(raw, raw)
        if style not in _MP2RAGE_SITE_STYLES:
            allowed = ", ".join(_MP2RAGE_SITE_STYLES)
            raise ValueError(
                f"unknown mp2rage_site_style {cfg['mp2rage_site_style']!r}; "
                f"expected one of: {allowed}")
    elif "mp2rage_grainy_site_style" in cfg:
        # ``False`` used to mean the ordinary calibrated UNIT1 family.  Map it
        # to that family rather than silently drawing another new style.
        style = "grainy" if bool(cfg["mp2rage_grainy_site_style"]) else "unit1"
    else:
        style = str(rng.choice(
            _MP2RAGE_SITE_STYLES, p=_MP2RAGE_SITE_PROBABILITIES))

    if style == "grainy":
        if "mp2rage_site_strength" in cfg:
            strength = float(cfg["mp2rage_site_strength"])
        elif "mp2rage_grainy_site_strength" in cfg:
            strength = float(cfg["mp2rage_grainy_site_strength"])
        else:
            # A selected phenotype should be observably grainy.  The previous
            # U(0, 1) draw spent a quarter of this component near the ordinary
            # UNIT1 mode and therefore did not add useful support.
            strength = float(rng.uniform(0.25, 1.0))
        strength = float(np.clip(strength, 0.0, 1.0))
    else:
        strength = 0.0
    return style, strength


def _policy12_site_thermal_fractions(style, grainy_strength):
    """Return dominant fine-thermal variance shares for policy-12 sites.

    Thermal noise is intrinsically fine at reconstructed voxel scale.  The older
    site model smoothed the entire realization to add grain, which preserved its
    variance but moved too much energy into a correlated band.  Policy 12 retains
    a dominant fine component and treats grain as a second receive/reconstruction
    component.  Values are deterministic properties of the coherent site style;
    explicit cfg controls remain authoritative.
    """
    style = str(style)
    if style == "canonical":
        return 0.96, 0.95
    if style == "unit1":
        return 0.98, 0.96
    if style == "high_contrast":
        return 0.94, 0.94
    if style == "grainy":
        strength = float(np.clip(grainy_strength, 0.0, 1.0))
        return 0.86 - 0.14 * strength, 0.90 - 0.12 * strength
    raise ValueError(f"unknown MP2RAGE site style {style!r}")


def _policy12_site_acquisition_guard(style, grainy_strength):
    """Mild global coarse/continuous guard for every non-anchor policy-12 MP2 draw.

    Returns physical ``(coarsen, dilation, feather, transition)`` millimetres.
    Posterior-fossa v7 specs remain authoritative and replace these global values
    with their regional hard-tail guard. The site dependence is intentionally
    modest: sharp modes retain narrow transitions, while a grainy reconstruction
    receives a slightly broader low-frequency acquisition proxy.
    """
    style = str(style)
    if style == "canonical":
        return 3.0, 1.0, 1.5, 2.0
    if style == "unit1":
        return 3.5, 1.5, 2.0, 2.5
    if style == "high_contrast":
        return 2.5, 0.75, 1.5, 2.0
    if style == "grainy":
        strength = float(np.clip(grainy_strength, 0.0, 1.0))
        return (3.5 + 0.75 * strength,
                1.5 + 0.75 * strength,
                2.0 + 0.75 * strength,
                2.5 + 0.75 * strength)
    raise ValueError(f"unknown MP2RAGE site style {style!r}")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _feather_mask(mask, width_voxels=3.0, *, voxel_sizes=None, width_mm=None):
    """Return a smooth brain-membership field instead of an exact target-mask step.

    ``width_voxels`` is the half-width of the transition band in voxel units.  The acquisition
    renderer has no affine, so callers with physical spacing should convert their desired millimetre
    band to voxels before calling.  A smoothstep over the signed distance avoids encoding the ground-
    truth extraction boundary as a sharp intensity/noise discontinuity.
    """
    b = np.asarray(mask) > 0.5
    if not b.any() or b.all():
        return b.astype(np.float32)
    sampling = tuple(float(x) for x in voxel_sizes) if voxel_sizes is not None else None
    width = max(float(width_mm if width_mm is not None else width_voxels), 1e-3)
    signed = (ndi.distance_transform_edt(b, sampling=sampling)
              - ndi.distance_transform_edt(~b, sampling=sampling))
    t = np.clip(0.5 + signed / (2.0 * width), 0.0, 1.0).astype(np.float32)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)  # smoothstep


def _remap3d(v, mask, csf_t, gm_t, wm_t, skull, boundary_band_voxels=3.0, *,
             voxel_sizes=None, boundary_band_mm=None):
    """Soft-membership tissue remap with a feathered brain/extracranial transition.

    Tissue statistics still come from the supplied annotation, but the rendered image never switches
    algorithms at its exact binary boundary.  This removes a segmentation-target cue while retaining
    the real scan's within-tissue texture.
    """
    b = mask > 0.5
    if b.sum() < 50:
        return v
    brain_w = _feather_mask(b, boundary_band_voxels, voxel_sizes=voxel_sizes,
                            width_mm=boundary_band_mm)
    bp = v[b]; span = max(1e-3, float(bp.max() - bp.min()))
    csf_th, gm_th = float(np.percentile(bp, 25)), float(np.percentile(bp, 65))
    w = max(1e-3, 0.09 * span)
    w_csf = _sigmoid((csf_th - v) / w); w_wm = _sigmoid((v - gm_th) / w)
    w_gm = np.clip(1.0 - w_csf - w_wm, 0, None)
    s = w_csf + w_gm + w_wm + 1e-6
    target = (w_csf / s * csf_t + w_gm / s * gm_t + w_wm / s * wm_t).astype(np.float32)
    detail = v - ndi.gaussian_filter(v, 2.0)
    gmag = np.sqrt(sum(g ** 2 for g in np.gradient(target))).astype(np.float32)
    edge_w = 1.0 / (1.0 + (gmag / 0.08) ** 2)
    brain_new = np.clip(target + 0.6 * detail * edge_w, 0, 1)
    # Preserve air and compress only image-derived extracranial foreground.  Blend this with the
    # brain remap over the signed-distance band rather than branching on the exact binary target.
    outer_new = np.where(v > 0.1, v * float(skull), v).astype(np.float32)
    out = brain_w * brain_new + (1.0 - brain_w) * outer_new
    return np.clip(ndi.gaussian_filter(out, 0.5), 0, 1).astype(np.float32)


def _add_magnitude_noise(v, rng, sigma, floor_fraction=0.25):
    """Add spatially varying magnitude noise across the *whole* FOV.

    The previous renderer multiplied Gaussian noise by the exact brain mask, making noise texture end
    precisely at the segmentation target.  Here the noise scale is derived only from the rendered
    image, with a non-zero FOV floor, and two quadrature components produce a Rician-like magnitude.
    """
    img = np.clip(np.asarray(v, dtype=np.float32), 0.0, 1.0)
    sigma = max(float(sigma), 0.0)
    if sigma <= 1e-8:
        return img.copy()
    positive = img[img > 0]
    scale = float(np.percentile(positive, 99.0)) if positive.size else 1.0
    signal_w = np.clip(img / max(scale, 1e-3), 0.0, 1.0)
    signal_w = ndi.gaussian_filter(signal_w.astype(np.float32), 1.5)
    floor = float(np.clip(floor_fraction, 0.0, 1.0))
    sigma_map = sigma * (floor + (1.0 - floor) * np.sqrt(np.clip(signal_w, 0.0, 1.0)))
    n_real = rng.normal(0.0, 1.0, img.shape).astype(np.float32) * sigma_map
    n_imag = rng.normal(0.0, 1.0, img.shape).astype(np.float32) * sigma_map
    return np.clip(np.sqrt((img + n_real) ** 2 + n_imag ** 2), 0.0, 1.0).astype(np.float32)


def _bias3d(v, rng, strength):
    if abs(float(strength)) <= 1e-8:
        return np.asarray(v, dtype=np.float32).copy()
    sig = float(rng.uniform(0.12, 0.3)) * float(min(v.shape))
    f = ndi.gaussian_filter(rng.standard_normal(v.shape).astype(np.float32), sig)
    f = f / (f.std() or 1.0) * strength
    out = v * np.exp(np.clip(f, -1.3, 1.3))
    # RENORMALIZE by a foreground percentile instead of a HARD CLIP: at high field the strong B1 gain
    # multiplied bright tissue past 1.0, and a hard clip left a flat pure-white plateau (a synthetic
    # tell â€” a model can memorize it). Dividing by the 99.5th foreground pctile keeps the shading but
    # preserves bright-tissue detail (no white-out).
    p = float(np.percentile(out[out > 0.05], 99.5)) if np.any(out > 0.05) else 1.0
    return np.clip(out / max(p, 1e-3), 0.0, 1.0).astype(np.float32)


def _resolution(v, rng):
    factors = np.array([float(rng.uniform(1.4, 2.6)) if rng.random() < 0.5 else 1.0 for _ in v.shape])
    if np.allclose(factors, 1.0):
        return v
    small = ndi.zoom(ndi.gaussian_filter(v, [f / 3.0 if f > 1 else 0 for f in factors]), 1.0 / factors, order=1)
    up = ndi.zoom(small, np.asarray(v.shape) / np.asarray(small.shape), order=1)
    out = np.zeros(v.shape, np.float32); sl = tuple(slice(0, min(a, b)) for a, b in zip(v.shape, up.shape))
    out[sl] = up[sl]
    return out


def sample_config(rng, *, sequence=None) -> dict:
    """Sample a coherent acquisition configuration, optionally for a fixed sequence."""
    options = settings('realistic_acquisition')
    seq = str(rng.choice(options["sequences"])) if sequence is None else str(sequence)
    if seq not in _SEQ:
        raise ValueError(f"unknown MRI sequence {seq!r}")
    # MP2RAGE is a high-field acquisition in practice; do not emit impossible
    # 64 mT/0.55 T MP2RAGE labels from the independent Cartesian product.
    if seq == "MP2RAGE":
        fields = [field for field in _MP2RAGE_FIELDS if field in options["fields"]]
        weights = np.array([_MP2RAGE_FIELD_PROBABILITIES[_MP2RAGE_FIELDS.index(field)] for field in fields])
        fld = float(rng.choice(fields, p=weights / weights.sum()))
    else:
        fld = float(rng.choice(options["fields"]))
    return {"field": fld, "sequence": seq, "vendor": str(rng.choice(options["vendors"])),
            "recon": str(rng.choice(options["reconstructions"]))}


def realistic_acquisition(scan01, mask, rng, geometry: bool = True,
                          cfg: dict = None, clean: bool = False,
                          boundary_band_voxels: float = 3.0,
                          noise_floor_fraction: float = 0.25, *, voxel_sizes=None,
                          boundary_band_mm: float = 3.0,
                          apply_resolution: bool = True,
                          mp2rage_params=None,
                          mp2rage_posterior_fossa=None) -> Tuple[np.ndarray, np.ndarray, str]:
    """Render the real scan as one realistic acquisition config. Returns (scan01, mask, label).

    `clean=True` -> a CANONICAL textbook example of the config (no resolution downsampling, halved
    bias/noise) so the standard-protocol tier shows recognizable, crisp MPRAGE/MP2RAGE/T2/FLAIR.

    ``mp2rage_params`` is an optional allowlisted override for exactly one inherited lower-level
    MP2RAGE feature group.  It is composed into the same single n-D render; it is never a second pass.
    Canonical clean anchors deliberately reject these superset probes.

    ``mp2rage_posterior_fossa`` is a versioned, replayable acquisition package.
    It is mutually exclusive with ``mp2rage_params`` and the master-slider
    superset route; the curriculum enforces that one-family-at-a-time policy."""
    cfg = cfg or sample_config(rng)
    noise, bias, cscale = _FIELD[cfg["field"]]
    if clean:
        bias *= 0.5; noise *= 0.5
    csf, gm, wm, skull, tag = _SEQ[cfg["sequence"]]
    lower_mp2rage_params = _validated_mp2rage_lower_params(mp2rage_params)
    posterior_spec = None
    if mp2rage_posterior_fossa is not None:
        from augmentations.curricula.mp2rage_posterior import validate_posterior_fossa_spec
        posterior_spec = validate_posterior_fossa_spec(mp2rage_posterior_fossa)
    if lower_mp2rage_params and tag != "mp2bg":
        raise ValueError("mp2rage_params require an MP2RAGE acquisition config")
    if lower_mp2rage_params and clean:
        raise ValueError("mp2rage_params cannot be used with a canonical clean MP2RAGE anchor")
    if posterior_spec is not None and tag != "mp2bg":
        raise ValueError("mp2rage_posterior_fossa requires an MP2RAGE acquisition config")
    if posterior_spec is not None and clean:
        raise ValueError("mp2rage_posterior_fossa cannot modify a canonical clean anchor")
    if posterior_spec is not None and lower_mp2rage_params:
        raise ValueError("mp2rage_posterior_fossa and mp2rage_params cannot be stacked")
    if posterior_spec is not None and bool(cfg.get("mp2rage_superset", False)):
        raise ValueError("mp2rage_posterior_fossa and the MP2RAGE superset cannot be stacked")
    if tag == "mp2bg":
        # The shared n-D UNI renderer owns extracranial appearance, so do NOT pre-compress here. Measured on
        # NFBS: skull=0.55 drops extracranial mean 0.522 -> 0.284 and pushes 32% of skull/scalp below
        # the renderer's air threshold, where it gets painted over with the mid-gray background â€” the
        # head rim vanishes. Real UNI scalp fat is the BRIGHTEST structure in the image.
        skull = 1.0
        # UNI is a RATIO of the two inversion images, so receive-field inhomogeneity CANCELS by
        # construction â€” that is the sequence's headline property and why a real UNI has near-uniform
        # WM across the whole slice while the MPRAGE beside it shows shading. Applying the full
        # magnitude-image bias here measurably crushed the render (brain percentiles
        # [0.04 0.29 0.44 0.65 0.74] -> [0.04 0.20 0.32 0.44 0.66], i.e. the "muddy" look). Keep only
        # a small residual for the B1+ TRANSMIT non-uniformity, which does survive the ratio.
        bias *= _MP2RAGE_BIAS_RESIDUAL
    j = lambda x: float(np.clip(x + rng.uniform(-0.05, 0.05), 0, 1))            # protocol/setting jitter
    # Field-strength contrast is centred about mid-gray so it changes separation,
    # rather than acting as a global scale that later normalization would erase.
    field_contrast = lambda x: float(np.clip(0.5 + (float(x) - 0.5) * cscale, 0, 1))
    source = np.asarray(scan01, np.float32)
    jittered_targets = (j(field_contrast(csf)), j(field_contrast(gm)),
                        j(field_contrast(wm)))
    if tag == "mp2bg":
        # The empirical UNIT1 renderer is already a complete tissue transfer.
        # Running the generic three-anchor sequence remap first double-remapped the
        # anatomy and, critically, applied an extra sigma=0.5 blur.  A final global
        # histogram match could hide that loss numerically but not visually.
        out = source.copy()
    else:
        out = _remap3d(source, np.asarray(mask, np.float32),
                       *jittered_targets, skull, boundary_band_voxels,
                       voxel_sizes=voxel_sizes,
                       boundary_band_mm=(boundary_band_mm if voxel_sizes is not None else None))
    brain_w = _feather_mask(mask, boundary_band_voxels, voxel_sizes=voxel_sizes,
                            width_mm=(boundary_band_mm if voxel_sizes is not None else None))
    # MP2RAGE largely removes receive-field bias. Apply a smaller residual field to
    # the anatomical signal *before* synthesizing the complex-ratio air background;
    # multiplying the stored midpoint background by a generic bias field is not a
    # valid MP2RAGE operation.
    bias_done = False
    if tag == "mp2bg":
        out = _bias3d(out, rng, bias)
        bias_done = True

    if tag == "mp2bg":                                       # MP2RAGE / UNI appearance
        # One n-D renderer is used for review slices and training volumes. The
        # canonical air law is calibrated to the bundled UNIT1: native N_effâ‰ˆ33,
        # with pipeline-normalized centreâ‰ˆ0.549 and almost iid spatial texture.
        from augmentations.artifacts.volume import mp2rage_3d
        # OPT-IN superset draw. This function's contract is a REALISTIC acquisition, so it stays in
        # the narrow band around the mark by default and a non-monotone fold is never sampled here.
        # `cfg["mp2rage_superset"]` (set per sample by the curriculum, never by `sample_config`) opts
        # an individual draw into the over-generated tail instead â€” same renderer, wider endpoints.
        superset = bool(cfg.get("mp2rage_superset", False)) and not clean
        if superset:
            lo, hi = _MP2RAGE_T_SUPERSET
            ref_spread, fold_prob = _MP2RAGE_REF_SPREAD_SUPERSET, _MP2RAGE_FOLD_PROB_SUPERSET
        else:
            lo, hi = _MP2RAGE_T_CLEAN if clean else _MP2RAGE_T_RANGE
            ref_spread, fold_prob = _MP2RAGE_REF_SPREAD, _MP2RAGE_FOLD_PROB
        requested_boundary = (float(boundary_band_mm) if voxel_sizes is not None
                              else float(boundary_band_voxels))
        boundary_width = float(cfg.get(
            "mp2rage_boundary_width",
            min(requested_boundary, 1.5 if clean else 2.0)))
        recon = str(cfg["recon"])

        # Draw one coherent site phenotype rather than independently combining
        # every marginal extreme.  The four components span the deduplicated
        # references: a clean/balanced anchor, the measured raw UNIT1 family, a
        # sharp high-contrast display family, and a grainy screenshot-like family.
        # Only the two measured quantile anchors are used for tissue transfer.
        # Every scalar remains explicitly overrideable through ``cfg``.
        site_style, grainy_strength = _resolve_mp2rage_site_style(
            rng, cfg, clean=clean)
        grainy_site = site_style == "grainy"
        # Identity constraint: generic callers and frozen policy-11 configs retain
        # the historical fully correlated grain endpoint. Only the explicit
        # curriculum marker opts site defaults into the policy-12 mixture. A user
        # may still set either fraction directly without the marker.
        policy12_site = (not clean
                         and cfg.get("mp2rage_policy_version", None) == 12)
        if policy12_site:
            bg_thermal_default, tissue_thermal_default = (
                _policy12_site_thermal_fractions(site_style, grainy_strength))
            (guard_coarsen_default, guard_dilation_default,
             guard_feather_default, guard_transition_default) = (
                _policy12_site_acquisition_guard(site_style, grainy_strength))
        else:
            bg_thermal_default = tissue_thermal_default = 0.0
            (guard_coarsen_default, guard_dilation_default,
             guard_feather_default, guard_transition_default) = (0.0,) * 4

        sampled_tissue_sigma = float(np.clip(
            noise * _RECON_NOISE_FACTOR.get(recon, 1.0)
            * rng.uniform(0.75, 1.15), 0.002, 0.060))
        if grainy_site:
            # Screenshot-like bounds: strong but short-correlated tissue noise,
            # without treating its display histogram as a quantitative anchor.
            sampled_tissue_sigma = 0.028 + 0.024 * grainy_strength
        elif site_style == "unit1":
            # The measured raw UNIT1 has substantially more whole-brain fine
            # residual than the field-strength-only estimate (~.012 at 7 T).
            # Give the measured-anchor component its own narrow global range so
            # posterior stress does not have to manufacture all missing texture
            # locally.  This also removes a learnable noisy-PF / smooth-cerebrum
            # split.  Recon-dependent g-factor texture is still applied below.
            sampled_tissue_sigma = float(rng.uniform(0.018, 0.026))
        elif site_style == "canonical":
            sampled_tissue_sigma = float(np.clip(
                sampled_tissue_sigma * rng.uniform(0.70, 0.95), 0.004, 0.024))
        elif site_style == "high_contrast":
            sampled_tissue_sigma = float(np.clip(
                sampled_tissue_sigma * rng.uniform(0.70, 1.00), 0.004, 0.026))
        tissue_sigma = float(cfg.get(
            "mp2rage_tissue_sigma", 0.012 if clean else sampled_tissue_sigma))

        if clean:
            n_eff = 33.0
            bg_mean = 0.549
            bg_grain_default = 0.0
            tissue_grain_default = 0.0
            tissue_aniso_default = 1.0
            ref_mix_default = 1.0
            ref_gain_default = 1.0
            extracranial_noise_default = 0.5
            scalp_floor_default = 0.46
            extracranial_strength_default = 1.0
            detail_default = 0.50
            fat_sat_probability = 0.0
        elif grainy_site:
            # Coarse ratio-background grain does not imply a salt-and-pepper air
            # law: retain 14--18.5 effective modes across the selected range.
            n_eff = 20.0 - 6.0 * grainy_strength
            bg_mean = 0.43 - 0.045 * grainy_strength
            bg_grain_default = 0.35 + 0.50 * grainy_strength
            tissue_grain_default = 0.25 + 0.30 * grainy_strength
            tissue_aniso_default = 1.0 + 0.35 * grainy_strength
            ref_mix_default = float(rng.uniform(0.0, 1.0))
            ref_gain_default = 1.00 + 0.15 * grainy_strength
            extracranial_noise_default = 0.90 + 0.70 * grainy_strength
            scalp_floor_default = 0.48 + 0.06 * grainy_strength
            extracranial_strength_default = 0.95 + 0.05 * grainy_strength
            detail_default = 0.34 - 0.08 * grainy_strength
            fat_sat_probability = 0.15 if float(cfg["field"]) >= 7.0 else 0.08
        elif site_style == "high_contrast":
            n_eff = float(np.clip(
                rng.uniform(32.0, 52.0) * _RECON_MODE_FACTOR.get(recon, 1.0),
                18.0, 56.0))
            bg_mean = float(rng.uniform(0.50, 0.57))
            bg_grain_default = float(rng.uniform(0.0, 0.12))
            tissue_grain_default = float(rng.uniform(0.0, 0.15))
            tissue_aniso_default = float(rng.uniform(1.0, 1.10))
            ref_mix_default = float(rng.uniform(0.0, 1.0))
            # Contrast temperature acts on an interpolation of the two measured
            # anchors; it does not add a screenshot-derived quantile curve.
            ref_gain_default = float(rng.uniform(1.12, 1.30))
            extracranial_noise_default = float(rng.uniform(0.35, 0.80))
            scalp_floor_default = float(rng.uniform(0.43, 0.49))
            extracranial_strength_default = float(rng.uniform(0.90, 1.05))
            detail_default = float(rng.uniform(0.55, 0.70))
            fat_sat_probability = 0.30 if float(cfg["field"]) >= 7.0 else 0.12
        elif site_style == "canonical":
            n_eff = float(np.clip(
                rng.uniform(30.0, 44.0) * _RECON_MODE_FACTOR.get(recon, 1.0),
                18.0, 56.0))
            bg_mean = float(rng.uniform(0.535, 0.565))
            bg_grain_default = float(rng.uniform(0.0, 0.12))
            tissue_grain_default = float(rng.uniform(0.0, 0.12))
            tissue_aniso_default = float(rng.uniform(1.0, 1.08))
            ref_mix_default = float(rng.uniform(0.25, 0.75))
            ref_gain_default = float(rng.uniform(0.98, 1.04))
            extracranial_noise_default = float(rng.uniform(0.35, 0.65))
            scalp_floor_default = float(rng.uniform(0.44, 0.48))
            extracranial_strength_default = float(rng.uniform(0.92, 1.02))
            detail_default = float(rng.uniform(0.45, 0.55))
            fat_sat_probability = 0.20 if float(cfg["field"]) >= 7.0 else 0.08
        else:  # normal measured-UNIT1 family
            n_eff = float(np.clip(
                rng.uniform(26.0, 42.0) * _RECON_MODE_FACTOR.get(recon, 1.0),
                14.0, 56.0))
            # The raw UNIT1 anchor has a centre near 0.549 and nearly iid air.
            # Keep this component close to that measurement; coarse correlation
            # belongs to the separate grainy phenotype.
            bg_mean = float(rng.uniform(0.53, 0.57))
            bg_grain_default = float(rng.uniform(0.0, 0.10))
            tissue_grain_default = float(rng.uniform(0.0, 0.12))
            tissue_aniso_default = float(rng.uniform(1.0, 1.10))
            ref_mix_default = float(rng.uniform(0.60, 1.0))
            ref_gain_default = float(rng.uniform(0.98, 1.08))
            extracranial_noise_default = float(rng.uniform(0.45, 0.80))
            scalp_floor_default = float(rng.uniform(0.45, 0.49))
            extracranial_strength_default = float(rng.uniform(0.95, 1.03))
            detail_default = float(rng.uniform(0.42, 0.56))
            fat_sat_probability = 0.45 if float(cfg["field"]) >= 7.0 else 0.15

        n_eff = float(cfg.get("mp2rage_bg_effective_coils", n_eff))
        bg_mean = float(cfg.get("mp2rage_bg_mean", bg_mean))
        bg_grain_sg = float(cfg.get("mp2rage_bg_grain_sg", bg_grain_default))
        tissue_grain_sg = float(cfg.get(
            "mp2rage_tissue_noise_grain_sg", tissue_grain_default))
        tissue_noise_aniso = float(cfg.get(
            "mp2rage_tissue_noise_aniso", tissue_aniso_default))
        bg_thermal_fraction = float(cfg.get(
            "mp2rage_bg_thermal_fraction", bg_thermal_default))
        tissue_thermal_fraction = float(cfg.get(
            "mp2rage_tissue_noise_thermal_fraction", tissue_thermal_default))
        guard_coarsen = float(cfg.get(
            "mp2rage_acquisition_guard_coarsen_mm", guard_coarsen_default))
        guard_dilation = float(cfg.get(
            "mp2rage_acquisition_guard_dilation_mm", guard_dilation_default))
        guard_feather = float(cfg.get(
            "mp2rage_acquisition_guard_feather_mm", guard_feather_default))
        guard_transition = float(cfg.get(
            "mp2rage_acquisition_transition_width_mm", guard_transition_default))
        ref_logit_gain = float(cfg.get(
            "mp2rage_ref_logit_gain", ref_gain_default))
        extracranial_noise_gain = float(cfg.get(
            "mp2rage_extracranial_noise_gain", extracranial_noise_default))
        if "fat_sat" in cfg:
            fat_sat = bool(cfg["fat_sat"])
        else:
            fat_sat = bool(rng.random() < fat_sat_probability)
        sampled_scalp_gain = (float(rng.uniform(0.35, 0.70)) if fat_sat
                              else ((1.02 + 0.08 * grainy_strength) if grainy_site
                                    else (float(rng.uniform(0.95, 1.25)) if clean else (
                                        float(rng.uniform(0.95, 1.12))
                                        if site_style == "canonical" else (
                                            float(rng.uniform(1.00, 1.30))
                                            if site_style == "high_contrast" else
                                            float(rng.uniform(0.95, 1.25)))))))
        scalp_gain = float(cfg.get(
            "mp2rage_scalp_gain", 1.0 if clean and not fat_sat else sampled_scalp_gain))
        scalp_floor = float(cfg.get("mp2rage_scalp_floor", scalp_floor_default))
        extracranial_strength = float(cfg.get(
            "mp2rage_extracranial_strength", extracranial_strength_default))
        render_kwargs = dict(
            rng=rng, mask=(np.asarray(mask) > 0.5),
            master_t=float(rng.uniform(lo, hi)),
            # A labelled MP2RAGE has the complete stored-ratio background even
            # when its empirical tissue remap is sampled slightly below t=1.
            background_weight=1.0,
            bg_mean=bg_mean,
            bg_grain_sg=bg_grain_sg,
            bg_thermal_fraction=bg_thermal_fraction,
            tissue_noise_grain_sg=tissue_grain_sg,
            tissue_noise_aniso=tissue_noise_aniso,
            tissue_noise_thermal_fraction=tissue_thermal_fraction,
            acquisition_guard_coarsen_mm=guard_coarsen,
            acquisition_guard_dilation_mm=guard_dilation,
            acquisition_guard_feather_mm=guard_feather,
            acquisition_transition_width_mm=guard_transition,
            tissue_noise=tissue_sigma,
            scalp_compress=scalp_gain,
            ref_spread=float(cfg.get(
                "mp2rage_ref_spread", 0.0 if clean else ref_spread)),
            ref_mix=float(cfg.get("mp2rage_ref_mix", ref_mix_default)),
            ref_logit_gain=ref_logit_gain,
            fold_prob=(0.0 if clean else fold_prob),
            boundary_width=boundary_width,
            voxel_sizes=voxel_sizes,
            # Fine stochastic tissue texture should carry the grainy-site style;
            # a large deterministic unsharp term makes cerebellar folia look etched.
            detail_gain=float(cfg.get("mp2rage_detail_gain", detail_default)),
            air_mode=str(cfg.get("mp2rage_air_mode", "snr")),
            extracranial_strength=extracranial_strength,
            extracranial_snr_lo=float(cfg.get("mp2rage_snr_lo", 0.07)),
            extracranial_snr_hi=float(cfg.get("mp2rage_snr_hi", 0.22)),
            bone_retention=float(cfg.get("mp2rage_bone_retention", 0.85)),
            scalp_floor=scalp_floor,
            scalp_gamma=float(cfg.get("mp2rage_scalp_gamma", 0.75)),
            extracranial_noise_gain=extracranial_noise_gain,
            noise_gfactor=0.08 * _RECON_NOISE_FACTOR.get(recon, 1.0),
            bg_effective_coils=n_eff,
            bg_inv2_scale=(1.0 if clean else float(rng.uniform(0.90, 1.10))),
            bg_fov_zero_prob=float(cfg.get(
                "mp2rage_fov_zero_prob", 0.0 if clean else 0.15)),
            bg_fov_zero_frac=float(cfg.get(
                "mp2rage_fov_zero_frac", rng.uniform(0.08, 0.28))),
            posterior_fossa=posterior_spec,
        )
        # A selected inherited group replaces only its corresponding calibrated
        # controls.  Dictionary merging avoids duplicate-key failures while the
        # renderer's aliases (bg_amp/noise_sigma/bg_on/noise_on) preserve exact
        # compatibility with ``augmentations.protocols.mp2rage.sample_params``.
        render_kwargs.update(lower_mp2rage_params)
        out = mp2rage_3d(out, **render_kwargs)
    elif tag == "swi":                                       # SWI dark susceptibility speckle in brain
        speckle = (rng.random(out.shape) < 0.02).astype(np.float32)
        out = out * (1.0 - 0.8 * brain_w * speckle)
    if not bias_done:
        out = _bias3d(out, rng, bias)                        # B1 inhomogeneity (up with field)
    if cfg["recon"] == "DLRecon":                            # DL recon: sharper, less noise, mild over-smooth
        out = np.clip(out + 0.3 * (out - ndi.gaussian_filter(out, 1.0)), 0, 1)
        noise *= 0.4
    if cfg["recon"] in ("GRAPPA", "SENSE", "SMS"):           # parallel imaging: structured g-factor noise
        noise *= 1.4
    if apply_resolution and (not clean) and rng.random() < 0.6:
        out = _resolution(out, rng)                          # voxel/slice/FOV settings
    # Generic Rician magnitude noise is wrong for the signed/offset UNI ratio and
    # would shift its mid-gray air distribution. `mp2rage_3d` already synthesizes
    # both complex-ratio air and in-tissue noise.
    if tag != "mp2bg":
        out = _add_magnitude_noise(out, rng, noise, floor_fraction=noise_floor_fraction)
    if geometry and out.ndim == 3:                           # gentle realistic head positioning
        from augmentations.appearance import realistic_geometry
        out, mask = realistic_geometry(out, np.asarray(mask), rng)
    suffix = " fat-sat" if tag == "mp2bg" and 'fat_sat' in locals() and fat_sat else ""
    if posterior_spec is not None:
        suffix += " posterior-fossa-stress"
    label = f"{cfg['field']}T {cfg['sequence']} {cfg['vendor']}/{cfg['recon']}{suffix}"
    return out, (np.asarray(mask) > 0.5), label
