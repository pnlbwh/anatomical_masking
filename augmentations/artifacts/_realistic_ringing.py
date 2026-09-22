"""Realistic Gibbs / ringing engine — user-validated 2026-06-26, geometry refit 2026-06-27.

The ring oscillation is ADDED onto the SHARP volume (detail preserved), NOT produced
by k-space truncation (which only blurs — explicitly rejected). Grounded on the user's
real clinical reference (`real_gibbs.jpg`): the rings are fine fringes that run PARALLEL
TO THE INNER SKULL, propagate inward and decay — i.e. concentric OVALS that track the
head's per-slice outline (its basic dimensions + orientation), NOT a single 3-D shell.

Geometry: a 2-D ELLIPSE is fit PER AXIAL SLICE to that slice's brain mask (2nd-moment
covariance -> centre, semi-axes, orientation), smoothed across slices for coherence.
Rings = iso-depth contours of that per-slice ellipse, so every slice shows a proper oval
filling its own brain. This REPLACES the single 3-D ellipsoid (2026-06-27 AM), which made
off-centre axial slices cut a tiny cross-section -> a centred concentric BULLSEYE the user
rejected ("they just look like circles in the center of the brain"). A 3-D EDT is also
rejected (distorts axial rings near the vertex/base; follows the bumpy outline) — the
covariance ellipse is the "smooth oval, skull dimensions, not warped to the skull" the
user asked for. The axial axis is found from the brain's bilateral-symmetry (L-R) axis.

Plus: UNEVEN ring spacing (depth-varying wavelength, NO warp); random per-call curvature
(semi-axis jitter); PARTIAL arcs (angular envelope); and an OCCASIONAL motion/aliasing
fold-over at higher severity — gated on the rng so moderate/severe rings only SOMETIMES
carry the ghost mess (heavy ringing is often, but not always, motion-caused).

QC: classed ``graded_steep`` (magnitude-gated but SENSITIVE) in augment_qc_dataset — only
the faintest ringing stays near-clean; visible mid-band oval rings score ``moderate`` and
an extreme instance drives QC down (band 0.44 -> ~QC 63, band 1.0 -> ~QC 10). Steeper than
plain ``graded`` (gamma 1.1 vs 2.5), which scored clearly-visible ringing as ~QC 88.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def _otsu_foreground(vol01: np.ndarray) -> np.ndarray:
    """Numpy Otsu head/foreground mask (fallback when no brain mask is supplied)."""
    hist, edges = np.histogram(vol01, bins=256, range=(0.0, 1.0))
    centers = (edges[:-1] + edges[1:]) / 2.0
    total = hist.sum()
    if total == 0:
        return np.zeros_like(vol01, dtype=bool)
    p = hist.astype(np.float64) / total
    omega = np.cumsum(p)
    mu = np.cumsum(p * centers)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (mu[-1] * omega - mu) ** 2 / (omega * (1.0 - omega))
    thr = max(float(centers[int(np.nanargmax(np.nan_to_num(sigma_b)))]), 0.02)
    return vol01 > thr


def _axial_axis(brain: np.ndarray) -> int:
    """Index of the AXIAL through-plane (S-I) axis, from the array alone.

    The L-R axis is the brain's bilateral-symmetry axis (flip-overlap Dice is far higher
    than for A-P / S-I); of the remaining two, A-P is the longest, so S-I is the shorter.
    Verified to pick the true axial axis on every NFBS subject (where extent alone ties
    S-I against L-R and fails)."""
    sym, ext = [], []
    for ax in range(3):
        proj = brain.any(axis=tuple(i for i in range(3) if i != ax))
        idx = np.where(proj)[0]
        ext.append(int(idx[-1] - idx[0]) if idx.size else 0)
        if idx.size < 2:
            sym.append(0.0)
            continue
        sl = [slice(None)] * 3
        sl[ax] = slice(idx[0], idx[-1] + 1)
        sub = brain[tuple(sl)]
        flip = np.flip(sub, axis=ax)
        denom = int(sub.sum() + flip.sum())
        sym.append(2.0 * int(np.logical_and(sub, flip).sum()) / denom if denom else 0.0)
    lr = int(np.argmax(sym))
    other = [a for a in range(3) if a != lr]
    return other[0] if ext[other[0]] <= ext[other[1]] else other[1]


def _depth_phase_profile(lam: float, spacing_var: float, dmax: int, rng) -> np.ndarray:
    """Accumulated ring phase vs DEPTH with a smoothly-varying wavelength -> uneven
    ring spacing while each ring stays a clean iso-depth contour (no warp)."""
    prof = ndi.gaussian_filter1d(rng.standard_normal(dmax), sigma=max(6.0, 2.0 * lam))
    prof = (prof - prof.mean()) / (prof.std() + 1e-6)
    lam_prof = np.clip(lam * (1.0 + spacing_var * prof), 0.5 * lam, 2.0 * lam)
    return 2.0 * np.pi * np.cumsum(1.0 / lam_prof)


def realistic_ring(vol, *, severity: float, rng, brain=None, axial_axis=None) -> np.ndarray:
    """Add realistic oval ringing to a 3D ``vol`` in [0, 1]. ``brain`` is an optional
    boolean brain mask (rings are confined to it; falls back to an Otsu head mask).
    ``axial_axis`` overrides the auto-detected axial (S-I) slicing axis."""
    vol = np.asarray(vol, dtype=np.float32)
    s = float(np.clip(severity, 0.0, 1.0))
    if s <= 0.0 or vol.ndim != 3:
        return vol.copy()
    brain = _otsu_foreground(vol) if brain is None else np.asarray(brain, dtype=bool)
    if brain.shape != vol.shape or int(brain.sum()) < 500:
        return vol.copy()
    ax = _axial_axis(brain) if axial_axis is None else int(axial_axis)

    # severity -> continuous params
    # 2026-07-10 REALISM CALIBRATION ("Realistic B", user-approved vs the real clinical reference
    # real_gibbs.jpg): real Gibbs is a FINE, SUBTLE, REGULAR ripple in a BAND near the skull that fades
    # inward — NOT bold rings filling the brain. The intermediate ~2x boost was too bold/uniform/deep
    # (edge_floor up + deep tau made it fill the interior — the OPPOSITE of real). The key to LEARNABILITY
    # is CONSISTENCY (the arc-floor below, rings always present + regular), not raw amplitude: the ORIGINAL
    # faint synthesis failed because arc-gaps zeroed the rings on many samples, not because it was subtle.
    # DIVERSITY (2026-07-10, user "vary in coverage/location/count"), kept RATING-CONSISTENT: band `s`
    # drives amplitude + penetration + angular EXTENT (severe = reliably widespread, mild = localized) so
    # the band->severity label stays consistent; ring COUNT (lam), penetration jitter, and WHICH sector
    # (center, below) randomize per sample WITHOUT changing perceived severity -> variety w/o mislabeling.
    A = 0.34 + 0.32 * s ** 0.70                  # amplitude: user-PICKED band-0.30 -> ~0.475 (clearly-visible
    # MODERATE rings for the qc~53 label). Uniformly visible across bands (~0.42 mild .. 0.66 severe) because
    # the a_edge/env/coverage factors eat ~half the nominal A; the mild->severe SEVERITY gradation is carried
    # mostly by the angular COVERAGE window (extent) below, not by amplitude. Fixes "still not visible enough".
    tau_frac = (0.08 + 0.16 * s) * float(rng.uniform(0.75, 1.35))   # penetration depth ~ band + per-sample jitter
    lam = float(rng.uniform(2.8, 7.0))           # RANDOM fringe spacing per sample -> ring-COUNT varies
    edge_floor, spacing_var = 0.58, 0.30
    ju, jw = rng.uniform(0.88, 1.15, size=2)     # per-call curvature (aspect) jitter

    volm = np.moveaxis(vol, ax, 0).astype(np.float32, copy=True)   # (Z, H, W), axial = axis 0
    brainm = np.moveaxis(brain, ax, 0)
    Z, H, W = volm.shape

    # ---- per-slice 2-D ellipse fit (2nd-moment covariance) ----
    cy = np.zeros(Z); cx = np.zeros(Z)
    vyy = np.ones(Z); vxx = np.ones(Z); vyx = np.zeros(Z)
    valid = np.zeros(Z, dtype=bool)
    for z in range(Z):
        ys, xs = np.nonzero(brainm[z])
        if ys.size < 60:
            continue
        valid[z] = True
        my = ys.mean(); mx = xs.mean()
        dy = ys - my; dx = xs - mx
        n = float(ys.size)
        cy[z] = my; cx[z] = mx
        vyy[z] = float(dy @ dy) / n + 1e-3
        vxx[z] = float(dx @ dx) / n + 1e-3
        vyx[z] = float(dy @ dx) / n
    if not valid.any():
        return vol.copy()

    # fill invalid slices by interpolation, then smooth params across slices (coherence)
    zc = np.arange(Z); zv = np.where(valid)[0]
    params = [cy, cx, vyy, vxx, vyx]
    for p in params:
        p[:] = np.interp(zc, zv, p[zv])
        p[:] = ndi.gaussian_filter1d(p, 2.0)

    # ---- build per-slice depth (iso-ellipse) and angle fields ----
    Y, X = np.mgrid[0:H, 0:W].astype(np.float32)
    depth = np.zeros((Z, H, W), dtype=np.float32)
    theta = np.zeros((Z, H, W), dtype=np.float32)
    rmean = np.ones(Z, dtype=np.float32)          # per-slice mean radius (sets the decay length)
    for z in range(Z):
        evals, evecs = np.linalg.eigh(np.array([[vyy[z], vyx[z]], [vyx[z], vxx[z]]]))
        evals = np.maximum(evals, 1e-3)
        a_min = 2.0 * np.sqrt(evals[0]) * jw      # minor semi-axis
        a_maj = 2.0 * np.sqrt(evals[1]) * ju      # major semi-axis
        nvec, mvec = evecs[:, 0], evecs[:, 1]     # minor, major directions (y, x)
        rm = 0.5 * (a_maj + a_min)
        rmean[z] = max(rm, 1.0)
        dy = Y - cy[z]; dx = X - cx[z]
        u = dy * mvec[0] + dx * mvec[1]           # along major
        w = dy * nvec[0] + dx * nvec[1]           # along minor
        rho = np.sqrt((u / a_maj) ** 2 + (w / a_min) ** 2)
        depth[z] = np.clip((1.0 - rho) * rm, 0.0, None)
        theta[z] = np.arctan2(w, u)

    dmax = int(np.ceil(float(depth.max()))) + 2
    phase_prof = _depth_phase_profile(lam, spacing_var, dmax, rng)
    phase_d = np.interp(depth, np.arange(dmax), phase_prof).astype(np.float32)

    # PARTIAL arcs: random angular envelope, drifting with depth
    ak, phik, omk = rng.uniform(-1, 1, 3), rng.uniform(0, 2 * np.pi, 3), rng.uniform(-0.05, 0.05, 3)
    raw = np.zeros_like(theta); norm = 1e-6
    for k in range(3):
        amp = ak[k] / (k + 1)
        raw = raw + amp * np.cos((k + 1) * theta + phik[k] + omk[k] * depth)
        norm += abs(amp)
    raw /= norm
    # RATING-CONSISTENT COVERAGE: a soft angular WINDOW whose SIZE grows with band `s` (mild -> a localized
    # sector; severe -> nearly the whole rim) at a RANDOM center -> WHICH sector + HOW MUCH vary per sample,
    # yet severity stays consistent (severe is reliably widespread, mild is a faint local patch). The
    # harmonic `raw` roughens the window so it isn't a clean arc; a band-scaled floor keeps severe from ever
    # collapsing to a single dot (learnable) while letting mild be faint/localized.
    center = float(rng.uniform(0, 2 * np.pi))
    cov = float(np.clip(0.32 + 0.58 * s + rng.uniform(-0.12, 0.12), 0.14, 1.0))   # angular coverage ~ band
    soft = 0.25 + 0.40 * (1.0 - cov)
    dth = np.abs(np.mod(theta - center + np.pi, 2.0 * np.pi) - np.pi)             # angular distance to center
    window = 1.0 / (1.0 + np.exp((dth - np.pi * cov) / max(soft, 0.1)))
    window = window * (0.78 + 0.22 * (1.0 / (1.0 + np.exp(-2.0 * raw))))          # roughen with harmonic texture
    floor = 0.10 + 0.22 * s                                                        # baseline coverage grows with band
    ang = (floor + (1.0 - floor) * window).astype(np.float32)
    del theta, raw

    tau_vol = np.maximum(tau_frac * rmean, 1e-3)[:, None, None]   # per-slice decay length
    env = (np.exp(-depth / tau_vol) * ang).astype(np.float32)
    g = ndi.gaussian_gradient_magnitude(volm, 0.8)
    hi = np.percentile(g, 96.0) or (float(g.max()) + 1e-6)
    a_edge = (edge_floor + (1.0 - edge_floor) * np.clip(g / hi, 0.0, 1.0)).astype(np.float32)
    del g
    bgate = ndi.gaussian_filter(brainm.astype(np.float32), 1.0)       # rings ONLY on brain (feathered)
    out = volm + (A * a_edge * np.cos(phase_d) * env * bgate).astype(np.float32)
    del a_edge, env, phase_d, depth

    # OCCASIONAL motion/aliasing fold-over (probabilistic, ramps with severity, NOT always-on)
    if s >= 0.45 and rng.random() < float(np.clip(0.25 + (0.6 - 0.25) * (s - 0.45) / 0.55, 0.0, 0.6)):
        pe = 1 if H >= W else 2                                       # PE = longer in-plane axis
        wrap = int(round(8 + 30 * s))
        folded = 0.5 * (np.roll(volm, wrap, axis=pe) + np.roll(volm, -wrap, axis=pe))
        alias_w = 0.05 + 0.50 * (s - 0.45)
        out = (1.0 - alias_w) * out + alias_w * folded

    return np.clip(np.moveaxis(out, 0, ax), 0.0, 1.0).astype(np.float32)
