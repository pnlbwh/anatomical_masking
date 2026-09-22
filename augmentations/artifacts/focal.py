"""focal augmentations (metal/blob, void, dropout slice+patch, droppatch).

Canonical picks:
- ``dropout`` -> moved VERBATIM from ``mri_qc.augmentations.signal_dropout``
  (slice attenuation OR cuboid zeroing), parameterized by ``mode``.
- ``metal`` / ``blob`` -> bright focal hyperintensity (battery ``metal_bright`` +
  ``_feature_bank.blob``).
- ``void`` -> dark focal dropout (``make_panel_showcase_png.void``).
- ``droppatch`` -> fully-enclosed in-brain cuboid voids
  (``_make_artifact_battery.droppatch_inbrain``).

Canonical signature: ``fn(arr, *, severity, rng, mask=None, **kw)``.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from augmentations.registry import register


def _head(data):
    """Crude head mask + median (from _make_artifact_battery._head)."""
    pos = data[data > 0]
    if pos.size == 0:
        return np.zeros_like(data, bool), 1.0
    med = float(np.median(pos))
    return data > 0.25 * med, med


# --------------------------------------------------------------------------- dropout (verbatim from mri_qc)
def _slice_dropout(arr, severity, rng):
    work = arr.astype(np.float32, copy=True)
    n_slices = work.shape[2]
    n_drop = max(1, int(round(float(severity) * n_slices)))
    n_drop = min(n_drop, n_slices)
    picks = rng.choice(n_slices, size=n_drop, replace=False)
    for z in picks:
        atten = 1.0 - float(rng.uniform(0.5, 1.0))
        work[:, :, z] = work[:, :, z] * atten
    return work


def _patch_dropout(arr, severity, rng):
    work = arr.astype(np.float32, copy=True)
    n_patches = 1 + int(float(severity) * 5)
    nx, ny, nz = work.shape
    max_side = max(2, int(round(float(severity) * min(nx, ny, nz) * 0.5)))
    for _ in range(n_patches):
        sx = int(rng.integers(2, max(3, max_side + 1)))
        sy = int(rng.integers(2, max(3, max_side + 1)))
        sz = int(rng.integers(2, max(3, max_side + 1)))
        x0 = int(rng.integers(0, max(1, nx - sx + 1)))
        y0 = int(rng.integers(0, max(1, ny - sy + 1)))
        z0 = int(rng.integers(0, max(1, nz - sz + 1)))
        work[x0:x0 + sx, y0:y0 + sy, z0:z0 + sz] = 0.0
    return work


@register(
    "dropout",
    kind="focal",
    severity_range=(0.0, 1.0),
    dims="3d",
    label_preserving=True,
    has_detector=True,
)
def dropout(arr, *, severity, rng, mask=None, mode="slice", **kw):
    """Signal dropout: attenuate random axial slices (``mode='slice'``) or zero
    random cuboid patches (``mode='patch'``). Moved verbatim from
    ``mri_qc.augmentations.signal_dropout``.
    """
    data = np.asarray(arr)
    if float(severity) <= 0.0:
        return data.copy()
    if mode not in ("slice", "patch"):
        raise ValueError(f"mode must be 'slice' or 'patch', got {mode!r}")
    if data.ndim == 4:
        out = np.empty_like(data, dtype=np.result_type(data.dtype, np.float32))
        for t in range(data.shape[-1]):
            out[..., t] = dropout(data[..., t], severity=severity, rng=rng, mask=mask, mode=mode, **kw)
        return out
    if data.ndim != 3:
        raise ValueError(f"dropout expects 3D or 4D, got {data.ndim}D")
    dtype = data.dtype
    if mode == "slice":
        out = _slice_dropout(data, severity, rng)
    else:
        out = _patch_dropout(data, severity, rng)
    return out.astype(dtype, copy=False)


# --------------------------------------------------------------------------- metal / blob (bright focal)
def _irregular_blob_mask(shape, center, radius, rng, lumpiness):
    """Boolean mask of an IRREGULAR (non-spherical) blob around ``center``.

    Builds a base sphere then perturbs its radius by a low-frequency angular
    field (random spherical-harmonic-ish lumpiness), so the metal core is never
    a clean ball. ``lumpiness`` in [0,1] scales the radial perturbation.
    """
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dz = zz - center[0]
    dy = yy - center[1]
    dx = xx - center[2]
    dist = np.sqrt(dz * dz + dy * dy + dx * dx).astype(np.float32)
    # angular perturbation: a few random sinusoidal lobes in each axis-angle
    # (cheap, broadcast-friendly proxy for a random star-shaped boundary)
    pert = np.zeros_like(dist)
    eps = 1e-6
    rxy = np.sqrt(dx * dx + dy * dy) + eps
    az = np.arctan2(dy, dx)            # azimuth
    el = np.arctan2(dz, rxy)           # elevation
    n_lobes = int(rng.integers(2, 7))
    for _ in range(n_lobes):
        ka = int(rng.integers(1, 6))
        ke = int(rng.integers(1, 6))
        pha = float(rng.uniform(0, 2 * np.pi))
        phe = float(rng.uniform(0, 2 * np.pi))
        amp = float(rng.uniform(0.1, 0.5))
        pert = pert + amp * np.sin(ka * az + pha) * np.cos(ke * el + phe)
    eff_r = radius * (1.0 + lumpiness * pert)
    return dist <= np.maximum(eff_r, 1.0)


@register(
    "metal",
    kind="focal",
    severity_range=(1.0, 5.0),
    dims="3d",
    label_preserving=True,
    has_detector=True,
)
def metal(arr, *, severity, rng, mask=None, n_metals=None, phi=None, k_arc=None, **kw):
    """Rich decomposed metal-susceptibility artifact (canonical).

    Decomposes the real metal artifact into its low-level constituents and
    OVER-GENERATES each (wide/randomized ranges, beyond-realistic extremes) so
    the synthetic distribution is a SUPERSET enveloping reality + a margin:

      * **signal void** — a dark/near-zero irregular core (susceptibility
        dephasing), shape perturbed by a random star-field (never a clean ball).
      * **bright pile-up rim** — a hyperintense shell hugging the void
        (frequency mis-mapping piles signal at the void edge).
      * **blooming** — the void+rim are Gaussian-blurred (over a wide sigma
        range) so the disturbance bleeds outward.
      * **multiplicity** — 1..N implants, each with independent size / location /
        void-depth / pile-up-intensity.

    NO geometric warp: an earlier version added an elastic displacement field per
    implant, but its localization envelope (rim_r*2.5) went brain-wide at high
    severity and deformed the whole head (user 2026-08-29: metal must not warp
    the brain). Metal is intensity-only: void + rim + bloom.

    ``severity`` (the battery's ``amp``, advisory range ~1..5, but ANY value is
    accepted) scales count, size, void depth, pile-up brightness, blooming, and
    distortion together. Works WITH or WITHOUT ``mask`` (falls back to an
    intensity head mask, matching the battery's mask-free call). Single ndarray
    out, rng-driven, same signature — callers unchanged.
    """
    from scipy.ndimage import gaussian_filter

    data = np.asarray(arr, dtype=np.float32)
    sev = float(severity)
    if sev <= 0.0:
        return data.copy()
    if data.ndim == 4:
        out = np.empty_like(data)
        for t in range(data.shape[-1]):
            out[..., t] = metal(data[..., t], severity=sev, rng=rng, mask=mask, **kw)
        return out
    if data.ndim != 3:
        raise ValueError(f"metal expects 3D or 4D, got {data.ndim}D")

    if mask is not None and np.asarray(mask).any():
        head = np.asarray(mask).astype(bool)
    else:
        head, _ = _head(data)
    coords = np.argwhere(head)
    if coords.size == 0:
        return data.copy()

    shape = data.shape
    vmax = float(data.max()) if data.max() > 0 else 1.0
    # normalized severity in ~[0,1+] driving every knob (over-generates past 1)
    s = sev / 5.0

    # multiplicity: 1..(2 + ~6*s) implants, randomized (override for testing)
    n_max = max(1, int(round(2 + 6.0 * s)))
    n_metals = int(rng.integers(1, n_max + 1)) if n_metals is None else int(n_metals)

    out = data.copy()
    minside = min(shape)

    for _ in range(n_metals):
        c = coords[rng.integers(len(coords))]
        # size: wide range, scaled by severity (over-generated up to ~min/6)
        r = float(rng.uniform(2.0, 3.0 + (minside / 7.0) * (0.4 + s)))
        lumpiness = float(rng.uniform(0.2, 0.9))
        void_mask = _irregular_blob_mask(shape, c, r, rng, lumpiness)
        if not void_mask.any():
            continue

        # --- signal void: drive core toward zero (depth randomized & sev-scaled)
        void_depth = float(rng.uniform(0.0, 0.25)) * (1.0 - 0.6 * s)
        out[void_mask] = out[void_mask] * np.clip(void_depth, 0.0, 1.0)

        # --- bright pile-up rim: dilate the void, take the shell, light it up.
        # Susceptibility frequency mis-mapping piles signal along the readout /
        # frequency-encode axis only -> the lit rim is a DIRECTIONAL crescent,
        # not a symmetric shell. Angularly weight the rim toward a per-implant
        # readout azimuth ``phi`` with over-generated arc sharpness ``k``:
        # weight = (0.5 + 0.5*cos(theta - phi))**k. k=0 recovers the full
        # isotropic shell (the legacy behaviour, still REACHABLE inside the
        # broadened distribution), while typical k>0 draws yield realistic
        # one-sided crescents. ``phi`` lives in the (dy,dx) plane (the in-slice
        # readout direction); the crescent is the load-bearing TRAINED feature.
        rim_r = r * float(rng.uniform(1.15, 1.6))
        rim_mask = _irregular_blob_mask(shape, c, rim_r, rng, lumpiness) & ~void_mask
        pile_amp = (1.5 + 4.0 * s) * float(rng.uniform(0.7, 1.6)) * vmax
        phi_i = float(rng.uniform(0.0, 2.0 * np.pi)) if phi is None else float(phi)
        # over-generate arc sharpness: 0 (full shell) -> ~4 (tight crescent)
        k_arc_i = float(rng.uniform(0.0, 4.0)) if k_arc is None else float(k_arc)
        # angular weight only needs to be evaluated on the rim voxels; compute
        # the readout azimuth in the (axis-1, axis-2) plane from voxel coords.
        ri, rj, rk = np.nonzero(rim_mask)
        theta = np.arctan2(rj.astype(np.float32) - c[1],
                           rk.astype(np.float32) - c[2])
        ang_w = np.power(np.clip(0.5 + 0.5 * np.cos(theta - phi_i), 0.0, 1.0),
                         k_arc_i).astype(np.float32)
        # per-voxel lit field over the FULL volume (needed for the bloom blur)
        pile = np.zeros(shape, np.float32)
        pile[ri, rj, rk] = (pile_amp * ang_w).astype(np.float32)
        # MAX (not assign): on the UNLIT arc ang_w->0 so pile->0; a plain assignment overwrote the
        # underlying tissue with ~0 -> a synthetic DARK CRESCENT the detector could learn instead of
        # the bright pile-up. np.maximum lights the bright arc while leaving the unlit side as tissue
        # (true susceptibility DARKENING is supplied separately by the void above).
        out[rim_mask] = np.maximum(out[rim_mask], pile[rim_mask])

        # --- blooming: blur the local disturbance so it bleeds outward
        bloom_sigma = float(rng.uniform(0.5, 1.0 + 4.0 * s))
        if bloom_sigma > 0.3:
            # additive bloom halo around the (angularly-weighted) rim only
            halo = gaussian_filter(np.where(rim_mask, pile, 0.0).astype(np.float32),
                                   bloom_sigma)
            halo[void_mask] = 0.0
            out = np.maximum(out, halo * float(rng.uniform(0.3, 0.8)))

    return np.nan_to_num(out, nan=0.0, posinf=vmax * 6.0, neginf=0.0).astype(np.float32)


@register(
    "blob",
    kind="focal",
    severity_range=(0.0, 1.0),
    dims="3d",
    label_preserving=True,
    has_detector=True,
)
def blob(arr, *, severity, rng, mask=None, radius=None, **kw):
    """Bright focal in-brain blob(s) — DECOMPOSED into {bright core + soft halo}
    and OVER-GENERATED: ``severity`` (0-1, registry convention) scales the count,
    size, and brightness so the synthetic distribution is a superset of real focal
    hyperintensities. (Old form took ``severity`` as a 1-5x multiplier on a tiny
    fixed-radius ball at the deepest point -> at 0-1 it was dimmer than tissue and
    invisible.) ``radius`` still accepted as an absolute-voxel override. Needs mask.
    """
    scan = np.asarray(arr, dtype=np.float32)
    if mask is None:
        raise ValueError("blob requires a brain mask")
    m = np.asarray(mask).astype(bool)
    if not m.any():
        return scan.copy()
    sev = float(np.clip(severity, 0.0, 1.0))
    if sev <= 0.0:                       # severity 0 = identity (no lesion injected)
        return scan.copy()
    p99 = float(np.percentile(scan[m], 99))
    edt = ndi.distance_transform_edt(m)
    emax = float(edt.max())
    extent = emax * 2.0
    # placement candidate bands: DEEP interior (legacy) vs a juxtacortical /
    # marginal EDT band (~3-8 vox from the brain edge). Real focal lesions
    # (WMH, metastases, juxtacortical foci) are NOT all deep-interior; over-
    # generate by drawing a fraction of blobs from the margin band so the
    # placement distribution is a superset of the old deep-only behaviour.
    deep = np.argwhere(edt > 0.3 * emax)
    margin = np.argwhere((edt >= 3.0) & (edt <= 8.0))
    zz, yy, xx = np.ogrid[:scan.shape[0], :scan.shape[1], :scan.shape[2]]
    out = scan.copy().astype(np.float32)
    n_blobs = int(rng.integers(1, 2 + int(round(3 * sev))))  # 1..~4, more with severity
    for _ in range(max(1, n_blobs)):
        # ~40% juxtacortical/marginal placement, else deep interior
        if margin.size and rng.random() < 0.4:
            c = margin[int(rng.integers(len(margin)))]
        elif deep.size:
            c = deep[int(rng.integers(len(deep)))]
        else:
            c = np.array(np.unravel_index(int(np.argmax(edt)), m.shape))
        rad = (radius if radius is not None
               else max(2.0, extent * (0.03 + 0.10 * sev) * float(rng.uniform(0.7, 1.3))))
        bright = (1.4 + 3.0 * sev * float(rng.uniform(0.8, 1.2))) * p99
        d2 = (zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2
        # randomize edge sharpness: sigma fraction of the radius (sharp focal
        # core -> diffuse halo), over-generated per blob.
        sharp = float(rng.uniform(0.35, 0.85))
        prof = np.exp(-d2 / (2.0 * (rad * sharp) ** 2)).astype(np.float32)
        # irregular-shape fraction: gate the Gaussian profile by a lobed blob
        # footprint so ~35% of blobs are non-round (still bright, never dark).
        if rng.random() < 0.35:
            lobed = _irregular_blob_mask(scan.shape, c, rad * 1.3,
                                         rng, float(rng.uniform(0.2, 0.7)))
            prof = prof * lobed.astype(np.float32)
        out = out + np.clip(m * prof * (bright - out), 0.0, None)
    return out.astype(np.float32)


# --------------------------------------------------------------------------- void (dark focal)
@register(
    "void",
    kind="focal",
    severity_range=(0.0, 1.0),
    dims="3d",
    label_preserving=True,
    has_detector=True,
)
def void(arr, *, severity, rng, mask=None, radius=None, **kw):
    """Dark signal void at the deepest mask point — DECOMPOSED into
    {lobed boundary + smooth radial dephasing} and OVER-GENERATED.

    Real susceptibility / flow voids are NOT hard binary spheres with a flat
    rim — they are LOBED (irregular boundary) with a CONTINUOUS radial signal
    drop-off (smooth dephasing from the core outward). This transform replaces
    the old {hard sphere core + uniform-attenuated shell} with:

      * an irregular (star-perturbed) blob boundary via ``_irregular_blob_mask``
        (never a clean ball), and
      * a continuous attenuation ``atten = 1 - exp(-4*(d/boundary)**p)`` applied
        inside that boundary — ``d`` is the radial distance from the void centre,
        ``boundary`` the lobed-edge radius, and ``p ~ U(1,4)`` over-generates the
        falloff sharpness: low ``p`` -> soft gradual flow-void dephasing, high ``p``
        -> crisp near-binary microbleed. At the centre (d=0) atten=0 (null core); it
        reaches ~0.98 AT the lobed boundary for EVERY p, so there is no hard rim step
        (normalizing to ``boundary`` rather than ``rad`` is what removes the step).

    radius scales with ``severity`` (0-1) so strong voids span enough of the
    brain to be detectable. ``radius`` accepted as an absolute-voxel override.
    Requires ``mask``.
    """
    scan = np.asarray(arr, dtype=np.float32)
    if mask is None:
        raise ValueError("void requires a brain mask")
    m = np.asarray(mask).astype(bool)
    if not m.any():
        return scan.copy()
    sev = float(np.clip(severity, 0.0, 1.0))
    if sev <= 0.0:                       # severity 0 = identity (no void injected)
        return scan.copy()
    edt = ndi.distance_transform_edt(m)
    extent = float(edt.max()) * 2.0
    rad = float(radius) if radius is not None else max(3.0, extent * (0.05 + 0.18 * sev))
    # Sample the void CENTER from PLAUSIBLE DEEP candidates instead of always THE single deepest
    # voxel (np.argmax(edt) collapsed the location to a fixed brain-centroid the detector could
    # learn as "void = centre"). Draw uniformly among interior voxels at least depth_q of the max
    # depth; the deepest point stays reachable (depth_q near 1) as a tail, so the location knob is
    # added without dropping coverage.
    edt_max = float(edt.max())
    depth_q = float(rng.uniform(0.45, 0.92))
    cand = np.argwhere(edt >= depth_q * edt_max)
    if cand.shape[0] == 0:
        c = np.unravel_index(int(np.argmax(edt)), m.shape)
    else:
        c = tuple(int(v) for v in cand[int(rng.integers(cand.shape[0]))])
    # lobed (irregular) boundary instead of a clean sphere; size it ~1.45x the
    # nominal radius so the smooth falloff has reached ~1 by the lobed edge.
    lumpiness = float(rng.uniform(0.2, 0.7))
    blob = _irregular_blob_mask(scan.shape, c, rad * 1.45, rng, lumpiness)
    region = blob & m
    out = scan.copy().astype(np.float32)
    zz, yy, xx = np.ogrid[:scan.shape[0], :scan.shape[1], :scan.shape[2]]
    dist = np.sqrt((zz - c[0]) ** 2 + (yy - c[1]) ** 2 +
                   (xx - c[2]) ** 2).astype(np.float32)
    # Continuous dephasing normalized to the BLOB BOUNDARY (rad*1.45), so atten
    # reaches ~1 AT the lobed edge for EVERY p (the old (d/rad)**p left a hard rim
    # step at low p: at the boundary it was only ~0.77, jumping to 1.0 outside).
    bound = rad * 1.45
    frac = np.maximum(dist, 0.0) / max(bound, 1e-3)     # 0 at core -> 1 at boundary
    p = float(rng.uniform(1.0, 4.0))   # over-generated dephasing sharpness
    atten = 1.0 - np.exp(-4.0 * np.power(frac, p))      # 0 at core -> ~0.98 at boundary
    atten = np.clip(atten, 0.0, 1.0).astype(np.float32)
    out[region] = out[region] * atten[region]   # 0 at core -> ~1 at lobed edge (no step)
    return out


# --------------------------------------------------------------------------- droppatch (enclosed in-brain voids)
@register(
    "droppatch",
    kind="focal",
    severity_range=(0.0, 1.0),
    dims="3d",
    label_preserving=True,
    has_detector=True,
)
def droppatch(arr, *, severity, rng, mask=None, **kw):
    """Fully-enclosed in-brain signal-drop patches — DECOMPOSED and
    OVER-GENERATED beyond the old axis-aligned hard-zero cuboids.

    The legacy battery drew axis-aligned cuboids zeroed to EXACTLY 0.0 with hard
    edges — one of the clearest giveaway-fakes (no real dropout has a perfectly
    sharp axis-aligned box edge or a dead-flat 0.0 floor). This version makes the
    REALISTIC patch the DEFAULT draw while keeping the old hard box as a minority
    tail inside the broadened distribution:

      * **feathered edge** — each patch boundary is softened with a 1-2 voxel
        Gaussian ramp (per-patch ``edge_sigma ~ U(1,2)``) so the drop blends in.
      * **partial fill** — the floor is a randomized residual ``fill ~ U(0,0.3)``
        (fraction of the original signal kept) instead of always-zero; deep
        ``fill≈0`` dropouts stay reachable at the low end.
      * **irregular shape** — for a fraction of patches the footprint is a lobed
        ``_irregular_blob_mask`` blob rather than a cuboid.
      * **hard-box minority tail** — ~20% of patches keep the legacy
        axis-aligned hard-edge box at fill 0.0 (the old behaviour, still
        REACHABLE).

    ``severity`` is the ``sev`` arg. Uses an internal crude head mask (ignores
    ``mask``) to match the battery's placement.
    """
    data = np.asarray(arr, dtype=np.float32)
    sev = float(severity)
    if sev <= 0.0:                       # severity 0 = identity (no dropout injected)
        return data.copy()
    head, _ = _head(data)
    margin = 12 if sev < 0.5 else 16
    dt = ndi.distance_transform_edt(head)
    coords = np.argwhere(dt > margin)
    if len(coords) == 0:
        coords = np.argwhere(head)
    out = data.copy()
    n = 3 if sev < 0.5 else 5
    half = (6, 10) if sev < 0.5 else (8, 14)
    shape = data.shape
    for _ in range(n):
        c = coords[rng.integers(len(coords))]
        h = rng.integers(half[0], half[1], size=3)
        sl = tuple(slice(max(0, int(c[k] - h[k])),
                         min(shape[k], int(c[k] + h[k]))) for k in range(3))
        hard_box = rng.random() < 0.20          # legacy minority tail
        if hard_box:
            out[sl] = 0.0                        # verbatim old behaviour
            continue
        # realistic default: partial fill + feathered (optionally lobed) edge
        fill = float(rng.uniform(0.0, 0.3))     # fraction of signal retained
        if rng.random() < 0.5:
            # lobed irregular footprint
            rad = float(np.mean(h))
            patch = _irregular_blob_mask(shape, c, rad,
                                         rng, float(rng.uniform(0.2, 0.7)))
            m_drop = patch.astype(np.float32)
        else:
            # axis-aligned footprint (but feathered + partial, not a hard box)
            m_drop = np.zeros(shape, np.float32)
            m_drop[sl] = 1.0
        edge_sigma = float(rng.uniform(1.0, 2.0))
        soft = ndi.gaussian_filter(m_drop, edge_sigma)   # 1-2 vox feather
        soft = np.clip(soft, 0.0, 1.0).astype(np.float32)
        # blend: keep ``fill`` of signal inside, full signal outside, ramp across
        keep = 1.0 - (1.0 - fill) * soft
        out = (out * keep).astype(np.float32)
    return out
