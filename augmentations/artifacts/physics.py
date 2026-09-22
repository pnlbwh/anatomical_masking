"""physics-driven MRI artifact augmentations.

New simulators for physical acquisition artifacts not previously covered by the
registry. Each artifact is DECOMPOSED into its low-level constituent features and
OVER-GENERATED with wide / beyond-realistic parameter ranges so the synthetic
distribution is a SUPERSET that envelops the real artifact plus a margin (rather
than a single realistic instance).

The *dominant magnitude* of every transform scales deterministically with
``severity`` (so severity 0.3 vs 0.9 is monotonic-ish in effect size), while the
*structure* — lobe shapes, foci, directions, band centers, gating, combinations —
is randomized widely to over-cover the feature space.

Module filename is ``physics`` for organization, but ``kind`` must be one of the
registry's frozen vocabulary (``geometry/intensity/noise/kspace/focal/...``); the
mapping is annotated per transform. All transforms degrade gracefully with
``mask=None`` (intensity-threshold head fallback) and never raise.

Canonical signature: ``fn(arr, *, severity, rng, mask=None, **kw)``.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from augmentations.registry import register
from augmentations.numerics import iter_4d, head_mask, smooth_random_field


# --------------------------------------------------------------------------- helpers


def _ref_scale(data, head):
    """A robust bright-tissue intensity scale (p90 of in-head positives)."""
    vals = data[head & (data > 0)]
    if vals.size == 0:
        pos = data[data > 0]
        if pos.size == 0:
            return 1.0
        return float(np.percentile(pos, 90))
    return float(np.percentile(vals, 90))


def _peripheral_shell(head, frac):
    """Boolean ring near the head boundary (within ``frac`` of the EDT max)."""
    edt = ndi.distance_transform_edt(head)
    mx = float(edt.max())
    if mx <= 0:
        return head.copy()
    return head & (edt <= frac * mx) & (edt > 0)


def _spatial_grain(noise, rng, max_sigma=2.5):
    """Give a WHITE noise field a real-recon GRAIN (1-2 voxel spatial correlation)
    while PRESERVING its power. Real MR magnitude noise is not white — apodization /
    zero-fill / partial-Fourier / vendor denoising correlate it over ~1-2 voxels — so
    a purely i.i.d. noise floor does not even envelope reality. Smoothing white noise
    by a Gaussian shrinks its std by ~sqrt(sum kernel^2); we rescale back to the input
    std so in-region noise POWER is preserved (cf. the global-std renorm bug noted in
    mp2rage_grain — applied here to UNIT noise BEFORE any spatial sigma scaling, so a
    spatially-varying amplitude profile is untouched). ``sigma`` is drawn per call;
    sigma~0 leaves the noise white (grain=0 stays reachable for the superset)."""
    sigma = float(rng.uniform(0.0, max_sigma))
    if sigma <= 1e-3:
        return noise
    s0 = float(noise.std())
    if s0 <= 1e-9:
        return noise
    sm = ndi.gaussian_filter(noise, sigma=sigma)
    s1 = float(sm.std())
    if s1 <= 1e-9:
        return noise
    return (sm * (s0 / s1)).astype(noise.dtype)


# --------------------------------------------------------------------------- g_factor_noise (kind=noise)
@register(
    "g_factor_noise",
    kind="noise",
    severity_range=(0.0, 1.0),
    dims="3d",
    label_preserving=True,
    has_detector=False,
)
def g_factor_noise(arr, *, severity, rng, mask=None, **kw):
    """Spatially-varying parallel-imaging g-factor noise amplification.

    DECOMPOSE: (1) a smooth random g-map with a central lobe + peripheral lobes
    (g-factor is worst centrally for SENSE-like recon, but we over-generate by
    randomizing lobe count/placement so the map can peak anywhere); (2) Rician
    character (the amplified noise rides on a magnitude image, so the floor is
    non-Gaussian). OVER-GENERATE: lobe count, lobe widths, central-vs-peripheral
    weighting, and a wide base sigma.

    The dominant noise magnitude scales with ``severity``; the g-map *shape* is
    randomized so reality is a subset of the covered distribution.
    """
    data = np.asarray(arr, dtype=np.float32)
    sev = float(severity)
    if sev <= 0.0:
        return data.copy()
    if data.ndim == 4:
        return iter_4d(g_factor_noise, data, severity, rng, mask, **kw)
    if data.ndim != 3:
        raise ValueError(f"g_factor_noise expects 3D or 4D, got {data.ndim}D")

    shape = data.shape
    head = head_mask(data, mask)
    scale = _ref_scale(data, head)

    # --- smooth random g-map: central lobe + several peripheral lobes -------
    zz, yy, xx = np.indices(shape).astype(np.float32)
    cz, cy, cx = (np.array(shape) - 1) / 2.0
    # normalized radius (0 center -> ~1 corner)
    rad = np.sqrt(((zz - cz) / (cz + 1e-6)) ** 2
                  + ((yy - cy) / (cy + 1e-6)) ** 2
                  + ((xx - cx) / (cx + 1e-6)) ** 2)
    gmap = np.zeros(shape, np.float32)
    # central lobe (classic g-factor peak), randomized strength
    central_w = float(rng.uniform(0.3, 1.6))
    gmap += central_w * np.exp(-(rad ** 2) / (2.0 * float(rng.uniform(0.15, 0.6)) ** 2))
    # peripheral lobes: over-generate count + placement + width
    n_lobes = int(rng.integers(2, 7))
    for _ in range(n_lobes):
        lc = np.array([rng.uniform(0, s - 1) for s in shape], np.float32)
        lw = float(rng.uniform(0.08, 0.35)) * float(min(shape))
        amp = float(rng.uniform(0.3, 1.5))
        d2 = (zz - lc[0]) ** 2 + (yy - lc[1]) ** 2 + (xx - lc[2]) ** 2
        gmap += amp * np.exp(-d2 / (2.0 * lw ** 2))
    # blend in a low-frequency random component so the map is not pure lobes
    gmap += 0.5 * np.abs(smooth_random_field(shape, rng, sigma=max(shape) / 6.0))
    gmap = ndi.gaussian_filter(gmap, sigma=max(1.0, min(shape) / 24.0))
    gmax = float(gmap.max())
    if gmax > 1e-6:
        gmap = gmap / gmax  # 0..1 spatial amplification map

    # --- Rician character: magnitude of (signal + n1) + i*n2 ----------------
    base_sigma = (0.04 + 0.5 * sev) * scale      # wide base sigma, sev-scaled
    local_sigma = base_sigma * (0.3 + 1.7 * gmap)  # spatially varying
    # GRAIN the UNIT noise (1-2 voxel correlation, power-preserving) BEFORE applying
    # the spatial g-map amplitude, so the grain rides the noise without distorting the
    # g-factor profile. grain sigma ~0 stays reachable (white).
    u1 = _spatial_grain(rng.standard_normal(shape).astype(np.float32), rng)
    u2 = _spatial_grain(rng.standard_normal(shape).astype(np.float32), rng)
    n1 = u1 * local_sigma
    n2 = u2 * local_sigma
    out = np.sqrt(np.maximum((data + n1) ** 2 + n2 ** 2, 0.0))
    return out.astype(np.float32)


# --------------------------------------------------------------------------- rician_noise (kind=noise)
@register(
    "rician_noise",
    kind="noise",
    severity_range=(0.0, 0.6),
    dims="3d",
    label_preserving=True,
    has_detector=False,
)
def rician_noise(arr, *, severity, rng, mask=None, **kw):
    """True Rician magnitude noise: ``sqrt((s + n1)^2 + n2^2)``.

    DECOMPOSE: an MRI magnitude image is the modulus of a complex signal with
    independent Gaussian noise on the real and imaginary channels. In dark /
    low-signal regions this is NON-Gaussian and produces a positive noise FLOOR
    (the Rician bias) — unlike additive Gaussian ``noise`` which can go negative.
    OVER-GENERATE: wide sigma (well beyond realistic SNR) so the low-signal floor
    is strongly expressed. Dominant sigma scales with ``severity``.
    """
    data = np.asarray(arr, dtype=np.float32)
    sev = float(severity)
    if sev <= 0.0:
        return data.copy()
    if data.ndim == 4:
        return iter_4d(rician_noise, data, severity, rng, mask, **kw)
    if data.ndim != 3:
        raise ValueError(f"rician_noise expects 3D or 4D, got {data.ndim}D")

    head = head_mask(data, mask)
    scale = _ref_scale(data, head)
    sigma = (0.03 + 0.45 * sev) * scale  # wide, sev-scaled
    # GRAIN the unit noise (real recon noise is spatially correlated, not white),
    # power-preserving; grain sigma~0 stays reachable. Preserves the Rician floor
    # (still magnitude of signal + correlated complex noise).
    u1 = _spatial_grain(rng.standard_normal(data.shape).astype(np.float32), rng)
    u2 = _spatial_grain(rng.standard_normal(data.shape).astype(np.float32), rng)
    n1 = u1 * sigma
    n2 = u2 * sigma
    out = np.sqrt(np.maximum((data + n1) ** 2 + n2 ** 2, 0.0))
    return out.astype(np.float32)


# --------------------------------------------------------------------------- chemical_shift (kind=intensity)
@register(
    "chemical_shift",
    kind="intensity",
    severity_range=(0.0, 1.0),
    dims="3d",
    label_preserving=True,
    has_detector=False,
)
def chemical_shift(arr, *, severity, rng, mask=None, axis=None, **kw):
    """Chemical-shift misregistration of bright fat-like edges.

    DECOMPOSE: fat resonates at a different frequency than water, so along the
    FREQUENCY-ENCODE axis fat signal is displaced by ``k`` pixels. At a fat/water
    boundary (scalp, orbit) this produces a BRIGHT mis-registered band on one
    side and a DARK void on the other (the classic bright+dark misreg pair).
    DECOMPOSE further: (a) extract a bright fat-like component, (b) shift it +k,
    (c) carve a dark void where it left. OVER-GENERATE: shift magnitude ``k``,
    direction (axis + sign), and MULTI-BAND (a few independent shifted copies).
    Dominant shift scales with ``severity``.
    """
    data = np.asarray(arr, dtype=np.float32)
    sev = float(severity)
    if sev <= 0.0:
        return data.copy()
    if data.ndim == 4:
        return iter_4d(chemical_shift, data, severity, rng, mask, axis=axis, **kw)
    if data.ndim != 3:
        raise ValueError(f"chemical_shift expects 3D or 4D, got {data.ndim}D")

    # The full intensity HEAD (incl scalp/orbit), NOT the brain mask: chemical shift
    # lives at FAT/WATER interfaces (subcutaneous scalp, orbit, diploic marrow) which
    # are EXTRACRANIAL. Passing the brain mask made head_mask return the brain, so the
    # shift slid in-brain cortex (physically wrong). Restrict the bright fat shell to
    # voxels OUTSIDE a dilated brain (mirrors incomplete_fat_sat's extracranial fat).
    head = head_mask(data, None)
    scale = _ref_scale(data, head)
    brain = np.asarray(mask).astype(bool) if mask is not None else None
    shell = _peripheral_shell(head, frac=float(rng.uniform(0.2, 0.45)))
    if brain is not None and brain.any():
        shell = shell & ~ndi.binary_dilation(brain, iterations=2)
    hp = data[head & (data > 0)]
    fat_thresh = float(np.percentile(hp, 75)) if hp.size else scale
    fat = np.where(shell & (data >= fat_thresh), data, 0.0).astype(np.float32)
    if not np.any(fat):  # fallback: any bright peripheral voxel
        fat = np.where(_peripheral_shell(head, 0.3) & (data >= fat_thresh),
                       data, 0.0).astype(np.float32)

    # ONE frequency-encode axis + ONE sign per call: a single readout bandwidth shifts
    # ALL fat in the SAME direction (per-band opposite shifts were non-physical).
    ax = int(rng.integers(0, 3)) if axis is None else int(axis)
    sign = 1 if rng.random() < 0.5 else -1
    k_max = max(1, int(round((1 + 9 * sev))))  # 1..10 px (sev-scaled dominant)
    n_bands = int(rng.integers(1, 4))           # over-generate: a few fat sources
    out = data.copy()
    for _ in range(n_bands):
        k = int(rng.integers(1, k_max + 1)) * sign
        amp = float(rng.uniform(0.5, 1.2))
        shift_vec = [0.0, 0.0, 0.0]
        shift_vec[ax] = float(k)
        # NON-wrapping shift (np.roll wrapped the fat band to the far FOV edge)
        shifted = ndi.shift(fat, shift=shift_vec, order=1, mode="constant", cval=0.0) * amp
        out = out + shifted                      # bright misregistered copy
        void_amp = float(rng.uniform(0.4, 1.0))
        out = out - void_amp * fat               # dark void where the fat left
    out = np.maximum(out, 0.0)
    return out.astype(np.float32)


# --------------------------------------------------------------------------- susceptibility_distortion (kind=geometry)
@register(
    "susceptibility_distortion",
    kind="geometry",
    severity_range=(0.0, 1.0),
    dims="3d",
    label_preserving=False,
    has_detector=False,
    extra_parameters=("geometry_only",),
)
def susceptibility_distortion(arr, *, severity, rng, mask=None, inf_axis=None, inf_low=None, **kw):
    """Susceptibility-induced geometric distortion + signal void/dropout.

    DECOMPOSE: near air-tissue interfaces (sinuses, ear canals, head periphery)
    B0 inhomogeneity causes (1) a local smooth geometric WARP (a displacement
    field peaked at foci), (2) a SIGNAL VOID (dephasing dropout) at the foci, and
    (3) broader smooth dephasing attenuation. OVER-GENERATE: focus count and
    placement, warp magnitude, and void depth. Dominant displacement + void depth
    scale with ``severity``. NOT label-preserving (anatomy moves).
    """
    data = np.asarray(arr, dtype=np.float32)
    sev = float(severity)
    if sev <= 0.0:
        return data.copy()
    if data.ndim == 4:
        return iter_4d(susceptibility_distortion, data, severity, rng, mask, **kw)
    if data.ndim != 3:
        raise ValueError(f"susceptibility_distortion expects 3D or 4D, got {data.ndim}D")

    shape = data.shape
    head = head_mask(data, mask)
    # foci on the peripheral shell (air-tissue interface proxy)
    shell = _peripheral_shell(head, frac=0.35)
    cand = np.argwhere(shell)
    if cand.size == 0:
        cand = np.argwhere(head)
    if cand.size == 0:
        cand = np.argwhere(np.ones(shape, bool))
    # Real susceptibility foci cluster at INFERIOR air-tissue interfaces (paranasal/
    # sphenoid sinuses, nasal cavity) and LATERAL-INFERIOR petrous/mastoid air cells,
    # not uniformly over the whole head shell. Scan orientation is unknown here, so
    # OVER-GENERATE which axis/side is "inferior" and bias ~70% of foci to that
    # peripheral-extreme band; keep ~30% anywhere on the shell (superset margin).
    inf_axis = int(rng.integers(0, 3)) if inf_axis is None else int(inf_axis)
    inf_low = (rng.random() < 0.5) if inf_low is None else bool(inf_low)
    coord_ax = cand[:, inf_axis]
    thr = float(np.percentile(coord_ax, 35 if inf_low else 65))
    interface = cand[coord_ax <= thr] if inf_low else cand[coord_ax >= thr]
    if interface.size == 0:
        interface = cand

    zz, yy, xx = np.indices(shape).astype(np.float32)
    disp = [np.zeros(shape, np.float32) for _ in range(3)]
    void_field = np.zeros(shape, np.float32)
    n_foci = int(rng.integers(2, 8))            # over-generate focus count
    max_disp = (1.0 + 11.0 * sev)               # px, sev-scaled dominant
    for _ in range(n_foci):
        pool = interface if rng.random() < 0.70 else cand
        c = pool[rng.integers(len(pool))].astype(np.float32)
        w = float(rng.uniform(0.06, 0.22)) * float(min(shape))
        d2 = (zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2
        lobe = np.exp(-d2 / (2.0 * w ** 2)).astype(np.float32)
        # random displacement direction per focus
        dir_vec = rng.standard_normal(3)
        dir_vec /= (np.linalg.norm(dir_vec) + 1e-9)
        amp = max_disp * float(rng.uniform(0.4, 1.0))
        for a in range(3):
            disp[a] += amp * dir_vec[a] * lobe
        void_field = np.maximum(void_field, lobe * float(rng.uniform(0.5, 1.0)))

    # smooth the displacement field (local but smooth)
    for a in range(3):
        disp[a] = ndi.gaussian_filter(disp[a], sigma=max(1.0, min(shape) / 30.0))
    coords = np.array([zz + disp[0], yy + disp[1], xx + disp[2]])
    warped = ndi.map_coordinates(data, coords, order=1, mode="nearest").astype(np.float32)

    # MASK-SAFE geometry-only: when co-transforming a label/mask (augment_scan passes
    # geometry_only=True) apply ONLY the geometric warp, NOT the signal void -- the void
    # would dim interior mask voxels below the binarize threshold and erode the label.
    # The image path (geometry_only unset) still gets the full void below.
    if kw.get("geometry_only"):
        return np.nan_to_num(warped, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # signal void / dephasing dropout at foci (void depth sev-scaled)
    void_depth = 0.4 + 0.6 * sev
    out = warped * (1.0 - void_depth * void_field)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32)


# --------------------------------------------------------------------------- gradient_nonlinearity (kind=geometry)
@register(
    "gradient_nonlinearity",
    kind="geometry",
    severity_range=(0.0, 1.0),
    dims="3d",
    label_preserving=False,
    has_detector=False,
    extra_parameters=("geometry_only",),
)
def gradient_nonlinearity(arr, *, severity, rng, mask=None, **kw):
    """Gradient-nonlinearity geometric warp (barrel / pincushion).

    DECOMPOSE: imperfect gradient coils produce a smooth low-order polynomial
    spatial warp that grows toward the FOV PERIPHERY. The radial term can be
    NEGATIVE (barrel/compression) or POSITIVE (pincushion/expansion). OVER-
    GENERATE: randomize the sign (barrel AND pincushion), the cubic/quintic
    coefficients, anisotropic per-axis scaling, and overall magnitude. Dominant
    warp magnitude scales with ``severity``. NOT label-preserving.
    """
    data = np.asarray(arr, dtype=np.float32)
    sev = float(severity)
    if sev <= 0.0:
        return data.copy()
    if data.ndim == 4:
        return iter_4d(gradient_nonlinearity, data, severity, rng, mask, **kw)
    if data.ndim != 3:
        raise ValueError(f"gradient_nonlinearity expects 3D or 4D, got {data.ndim}D")

    shape = data.shape
    zz, yy, xx = np.indices(shape).astype(np.float32)
    cz, cy, cx = (np.array(shape) - 1) / 2.0
    # normalized centered coords (-1..1)
    nz = (zz - cz) / (cz + 1e-6)
    ny = (yy - cy) / (cy + 1e-6)
    nx = (xx - cx) / (cx + 1e-6)
    r2 = nz ** 2 + ny ** 2 + nx ** 2

    # radial polynomial distortion: r' = r * (1 + k1 r^2 + k2 r^4).
    # ``base`` (the dominant peak deviation at the FOV edge, r2~1) scales
    # tightly with severity; the coefficient split between cubic/quintic is
    # randomized for over-generation but normalized so the EDGE deviation stays
    # ~= base regardless of the split (keeps total warp magnitude monotonic in
    # severity instead of letting a wide quintic draw dominate low-sev cases).
    sign = 1.0 if rng.random() < 0.5 else -1.0   # pincushion vs barrel
    base = (0.05 + 0.45 * sev)                    # dominant magnitude, sev-scaled
    w2 = float(rng.uniform(0.0, 0.6))             # fraction routed to quintic
    k1 = sign * base * (1.0 - w2) * float(rng.uniform(0.85, 1.15))
    k2 = sign * base * w2 * float(rng.uniform(0.85, 1.15))
    factor = (1.0 + k1 * r2 + k2 * r2 ** 2).astype(np.float32)
    # anisotropic per-axis gain (over-generate non-radial component). The
    # deviation from unity is sev-scaled so the TOTAL warp magnitude grows with
    # severity (a sev-independent gain would dominate low-sev draws and break
    # the monotonic-ish requirement).
    aniso = 0.08 * sev
    gz = 1.0 + float(rng.uniform(-aniso, aniso))
    gy = 1.0 + float(rng.uniform(-aniso, aniso))
    gx = 1.0 + float(rng.uniform(-aniso, aniso))

    # displacement = (factor*g - 1) * centered coords, in voxels. Cap the per-
    # voxel displacement to the FOV half-extent so an extreme draw cannot fold
    # signal out of frame (which non-monotonically saturates the diff metric).
    dz = (factor * gz - 1.0) * (zz - cz)
    dy = (factor * gy - 1.0) * (yy - cy)
    dx = (factor * gx - 1.0) * (xx - cx)
    cap_z, cap_y, cap_x = cz, cy, cx
    dz = np.clip(dz, -cap_z, cap_z)
    dy = np.clip(dy, -cap_y, cap_y)
    dx = np.clip(dx, -cap_x, cap_x)
    coords = np.array([zz + dz, yy + dy, xx + dx])
    out = ndi.map_coordinates(data, coords, order=1, mode="nearest").astype(np.float32)
    # MASK-SAFE geometry-only: when co-transforming a label/mask (geometry_only=True) skip
    # the Jacobian luminance shading below -- brightness changes would push interior mask
    # voxels under the binarize threshold and erode the label. The image path keeps the shading.
    if kw.get("geometry_only"):
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    # JACOBIAN LUMINANCE: an UNCORRECTED gradient-nonlinearity warp redistributes
    # signal, so brightness changes by the local volume Jacobian (compression/barrel
    # brightens, expansion/pincushion dims) -- a peripheral shading the geometry-only
    # warp was missing. Local isotropic volume scaling ~ factor**3; brightness ~ J**-c.
    # OVER-GENERATE the correction completeness c (0 = fully gradunwarp-corrected = no
    # shading; larger = raw/under-corrected), sev-scaled so shading grows with severity.
    c_corr = float(rng.uniform(0.0, 1.0)) * sev
    if c_corr > 1e-3:
        jac = np.clip(np.abs(factor) ** 3, 0.2, 5.0).astype(np.float32)
        lum = np.clip(jac ** (-c_corr), 0.4, 2.5).astype(np.float32)
        out = out * lum
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32)


# --------------------------------------------------------------------------- defacing (kind=focal)
@register(
    "defacing",
    kind="focal",
    severity_range=(0.0, 1.0),
    dims="3d",
    # NOT label-preserving: an aggressive (high-severity) cut NICKS the brain (zeroes up to
    # ~22% of in-brain voxels at sev 1.0). With label_preserving=True the mask kept those zeroed
    # voxels -> a benign-labelled scan whose brain mask covers signal-less holes. Marking it
    # non-label-preserving makes augment_scan re-run the SAME-seed cut on the mask (geometry_only
    # is ignored here, so the cut carves), so the label follows the nick. Below the brain-reach
    # severity (sev<0.45) the in-function guard keeps the cut in the face -> mask unchanged.
    label_preserving=False,
    has_detector=False,
)
def defacing(arr, *, severity, rng, mask=None, **kw):
    """Anonymization defacing: zero/clip an anterior face wedge.

    DECOMPOSE: a defacing tool removes the face by zeroing voxels on the anterior
    side of a (roughly coronal) cutting PLANE. Failure modes we over-generate:
    (a) plane orientation/tilt (random normal near the anterior axis), (b) cut
    DEPTH (sometimes shallow, sometimes deep enough to NICK the brain edge), and
    (c) RESIDUE (a faint smeared remnant left behind instead of clean zeros).
    OVER-GENERATE: plane normal, depth, residue level. Cut depth scales with
    ``severity`` (deeper / more brain-nicking at high severity).
    """
    data = np.asarray(arr, dtype=np.float32)
    sev = float(severity)
    if sev <= 0.0:
        return data.copy()
    if data.ndim == 4:
        return iter_4d(defacing, data, severity, rng, mask, **kw)
    if data.ndim != 3:
        raise ValueError(f"defacing expects 3D or 4D, got {data.ndim}D")

    shape = data.shape
    # Use the INTENSITY head (full head incl the FACE), NOT the brain mask: a defacer
    # zeroes the anterior FACE (an extra-cranial wedge); passing the brain mask made
    # head_mask return the brain, so the wedge carved the brain edge instead of the
    # face. With the intensity head, a shallow (low-sev) cut removes only face/scalp
    # and a deep (high-sev) cut grows inward to nick the brain.
    head = head_mask(data, None)
    coords = np.argwhere(head)
    if coords.size == 0:
        coords = np.argwhere(np.ones(shape, bool))
    lo = coords.min(0)
    hi = coords.max(0)
    extent = (hi - lo).astype(np.float32) + 1.0

    # REPRODUCIBLE cut geometry (so the augment_scan mask co-transform carves EXACTLY the brain
    # the scan zeroed): derive the face axis, centroid, and the brain-penetrating plane from the
    # BRAIN MASK, which is passed identically whether ``arr`` is the scan or the re-run mask.
    # (Deriving face_axis from ``head_mask(data)`` was the bug: on the mask call the "head" IS the
    # brain, giving a different longest axis -> the carve missed/over-removed by ~3x.)
    brain = np.asarray(mask).astype(bool) if mask is not None else None
    have_brain = brain is not None and brain.any()
    if have_brain:
        bcoords = np.argwhere(brain)
        b_lo_all = bcoords.min(0); b_hi_all = bcoords.max(0)
        axis_extent = (b_hi_all - b_lo_all).astype(np.float32) + 1.0
        ref_centroid = bcoords.mean(0).astype(np.float32)
    else:
        axis_extent = extent
        ref_centroid = coords.mean(0).astype(np.float32)

    # anterior axis = longest brain (else head) axis; ~10% off-axis over-generation tail.
    if rng.random() < 0.9:
        face_axis = int(np.argmax(axis_extent))
    else:
        face_axis = int(rng.integers(0, 3))
    front_low = rng.random() < 0.5

    zz, yy, xx = np.indices(shape).astype(np.float32)
    grid = [zz, yy, xx]
    centroid = ref_centroid

    # Cut plane. A real defacer removes the FACE; only an aggressive (high-severity) cut reaches
    # the brain. Decompose into two regimes so the BRAIN-touching part is brain-relative (exactly
    # reproducible on the mask call) and the FACE-only part (which the mask never sees) can stay
    # head-relative:
    #   * NICK (sev>0.5): plane = brain_front + nick, an EDGE clip up to ~12% of the brain at sev 1
    #     (a modest tail -- defacing is benign-class, so we don't teach a masker to drop a quarter
    #     of the brain; the mask now tracks whatever IS clipped, so the label stays correct).
    #   * FACE-only (sev<0.5): plane recedes ANTERIOR of the brain into the face by ``recede``.
    if have_brain:
        b_lo, b_hi = int(b_lo_all[face_axis]), int(b_hi_all[face_axis])
        b_ext = float(b_hi - b_lo + 1)
        nick = max(0.0, (sev - 0.5) / 0.5) * 0.12 * b_ext * float(rng.uniform(0.85, 1.0))
        face_gap = max(1.0, float((b_lo - lo[face_axis]) if front_low else (hi[face_axis] - b_hi)))
        recede = (1.0 - min(1.0, sev / 0.5)) * face_gap * float(rng.uniform(0.85, 1.0))
        plane_edge0 = (b_lo + nick - recede) if front_low else (b_hi - nick + recede)
    else:
        depth_frac = (0.18 + 0.42 * sev) * float(rng.uniform(0.9, 1.1))
        cut_len = depth_frac * float(extent[face_axis])
        plane_edge0 = (lo[face_axis] + cut_len) if front_low else (hi[face_axis] - cut_len)

    # The cut SURFACE is a depth-proportional plane perpendicular to the face
    # axis, TILTED by a small amount along the other two axes (over-generation:
    # a real defacer's plane is not perfectly axis-aligned). The tilt perturbs
    # WHERE the cut edge falls per-voxel but does NOT carve away half the slab,
    # so the removed volume stays dominated by ``cut_len`` (monotonic in sev).
    other = [a for a in range(3) if a != face_axis]
    tilt = {a: float(rng.uniform(-0.15, 0.15)) for a in other}  # gentle: keep low-sev cut in the face
    # per-voxel face-axis position of the tilted cut edge
    edge = np.zeros(shape, np.float32)
    for a in other:
        edge = edge + tilt[a] * (grid[a] - centroid[a])
    if front_low:
        wedge = (grid[face_axis] <= plane_edge0 + edge) & head
    else:
        wedge = (grid[face_axis] >= plane_edge0 + edge) & head
    if not wedge.any():  # degenerate tilt -> fall back to the pure axial slab
        wedge = ((grid[face_axis] <= plane_edge0) if front_low
                 else (grid[face_axis] >= plane_edge0)) & head
    # HARD guarantee: a mild defacer removes only the FACE -- below the brain-reach
    # severity the cut must not touch the brain (the tilt could otherwise nick it).
    if brain is not None and brain.any() and sev < 0.45:
        wedge = wedge & ~brain

    out = data.copy()
    # Residue is an over-generation feature (a failed defacer leaves a faint
    # smeared remnant) but its magnitude/probability must SHRINK with severity:
    # otherwise a bright-residue draw at high sev produces a SMALLER diff than a
    # clean-zero draw at low sev, inverting the monotonic-ish ordering. So the
    # dominant signal is the cut volume (grows with sev) while residue fades.
    residue = float(rng.uniform(0.0, 0.25)) * (1.0 - sev)
    p_residue = 0.5 * (1.0 - sev)
    if rng.random() < p_residue and residue > 0:
        # smeared low-amplitude remnant
        smear = ndi.gaussian_filter(data * wedge.astype(np.float32), sigma=2.0)
        out[wedge] = residue * smear[wedge]
    else:
        out[wedge] = 0.0
    return out.astype(np.float32)


# --------------------------------------------------------------------------- bounce_point_null (kind=intensity)
@register(
    "bounce_point_null",
    kind="intensity",
    severity_range=(0.0, 1.0),
    dims="3d",
    label_preserving=True,
    has_detector=False,
)
def bounce_point_null(arr, *, severity, rng, mask=None, **kw):
    """Inversion-recovery bounce-point null: drive a tissue band to the null.

    DECOMPOSE: in IR sequences a tissue whose T1 matches the inversion time TI
    passes through the signal NULL (zero magnitude). Around that null the
    magnitude image shows (a) a DARK band where that intensity is suppressed and
    (b) a local CONTRAST INVERSION (tissue just below/above the null swap relative
    brightness, because magnitude folds the negative lobe). OVER-GENERATE: null
    CENTER (which intensity band nulls), null WIDTH, and inversion strength.
    Dominant suppression depth scales with ``severity``.
    """
    data = np.asarray(arr, dtype=np.float32)
    sev = float(severity)
    if sev <= 0.0:
        return data.copy()
    if data.ndim == 4:
        return iter_4d(bounce_point_null, data, severity, rng, mask, **kw)
    if data.ndim != 3:
        raise ValueError(f"bounce_point_null expects 3D or 4D, got {data.ndim}D")

    head = head_mask(data, mask)
    vals = data[head & (data > 0)]
    if vals.size == 0:
        return data.copy()
    lo, hi = float(np.percentile(vals, 5)), float(np.percentile(vals, 95))
    span = max(hi - lo, 1e-6)
    # null center: random intensity within the tissue range (over-generate)
    center = lo + float(rng.uniform(0.2, 0.8)) * span
    width = (0.08 + 0.3 * float(rng.uniform(0.5, 1.5))) * span  # over-generate width
    depth = 0.6 + 0.4 * sev  # suppression depth -> near-complete null at high severity

    out = data.astype(np.float32, copy=True)
    # distance of each voxel intensity from the null center
    diff = data - center
    # Gaussian suppression well centered at the null -> dark band
    suppress = depth * np.exp(-(diff ** 2) / (2.0 * width ** 2)).astype(np.float32)
    # The magnitude-FOLD refill is V-SHAPED: inv_strength*|diff| -> ZERO at the null and
    # grows away from it (the local contrast inversion). The old code added a constant
    # ``center`` baseline here, which FLOORED the null at ~0.5*center (only ~0.47x dim)
    # instead of near-black; dropping it lets a matched tissue reach the true IR null.
    inv_strength = 0.4 * sev
    folded = inv_strength * np.abs(diff).astype(np.float32)
    region = head & (np.abs(diff) < 2.0 * width)
    out = np.where(region, data * (1.0 - suppress) + folded * suppress, data)
    out = np.maximum(out, 0.0)
    return out.astype(np.float32)


# --------------------------------------------------------------------------- incomplete_fat_sat (kind=intensity)
@register(
    "incomplete_fat_sat",
    kind="intensity",
    severity_range=(0.0, 1.0),
    dims="3d",
    label_preserving=True,
    has_detector=False,
)
def incomplete_fat_sat(arr, *, severity, rng, mask=None, **kw):
    """Incomplete fat saturation: heterogeneous bright residual in scalp/marrow.

    DECOMPOSE: failed/partial fat-suppression leaves bright fat signal in the
    subcutaneous scalp ring and diploic (marrow) layer, and it is spatially
    HETEROGENEOUS (some regions suppress, others don't) because of B0/B1
    inhomogeneity. DECOMPOSE: (a) a peripheral fat ring, (b) a smooth random
    heterogeneity field gating where the failure occurs, (c) a brightness boost.
    OVER-GENERATE: brightness, heterogeneity spatial frequency/contrast, ring
    thickness. Dominant brightness boost scales with ``severity``.
    """
    data = np.asarray(arr, dtype=np.float32)
    sev = float(severity)
    if sev <= 0.0:
        return data.copy()
    if data.ndim == 4:
        return iter_4d(incomplete_fat_sat, data, severity, rng, mask, **kw)
    if data.ndim != 3:
        raise ValueError(f"incomplete_fat_sat expects 3D or 4D, got {data.ndim}D")

    # Use the INTENSITY head (full head incl scalp), NOT the brain mask: passing the
    # brain mask to head_mask returns the brain, which made the "scalp ring" the
    # OUTER CORTEX rim -> fat-sat residual brightened the brain instead of the scalp.
    head = head_mask(data, None)
    scale = _ref_scale(data, head)
    ring = _peripheral_shell(head, frac=float(rng.uniform(0.10, 0.30)))  # thin scalp/marrow ring
    brain = np.asarray(mask).astype(bool) if mask is not None else None
    if brain is not None and brain.any():
        ring = ring & ~ndi.binary_dilation(brain, iterations=2)  # EXTRA-cranial only
    if not ring.any():
        ring = _peripheral_shell(head, frac=0.15)

    # heterogeneity field: smooth random gate in [0,1], over-generate frequency
    sigma = max(1.0, float(min(data.shape)) / float(rng.uniform(4.0, 16.0)))
    het = smooth_random_field(data.shape, rng, sigma=sigma)
    het = 0.5 * (het + 1.0)  # 0..1
    contrast = float(rng.uniform(1.0, 3.0))      # heterogeneity contrast
    het = np.clip(het ** contrast, 0.0, 1.0)
    # Floor the failure magnitude: where fat-sat fails it leaves CLEARLY bright fat,
    # so the residual must over-generate ABOVE the (already bright) healthy scalp
    # rather than fade into it (the ^contrast skew otherwise left it sub-healthy).
    het = 0.35 + 0.65 * het

    boost = (0.6 + 2.6 * sev) * scale            # dominant brightness, sev-scaled
    residual = boost * het * ring.astype(np.float32)
    out = data + residual
    return out.astype(np.float32)
