"""k-space augmentations (motion, ghosting, gibbs, spike/herringbone, aliasing,
zipper, central-DC, PE-line undersampling, continuous-trajectory motion).

Canonical picks per the Stage 2 hard rules:

- ``motion``  -> rigid rotation+translation k-space-line model (Shaw/TorchIO
  style), an UPGRADE from the translation-only ``mri_qc.augmentations.motion``.
- ``ghosting`` -> moved verbatim from ``mri_qc.augmentations.ghosting``.
- ``gibbs``   -> ``_gibbs_realistic.gibbs_realistic`` (sharp separable rect
  truncation = true axis-aligned ringing), the canonical realistic Gibbs.

New simulators added (mark ``has_detector`` per AUGMENTATION_RESEARCH.md):
``spike`` / ``herringbone`` (k-space point + conjugate mirror), ``aliasing``
(PE-line fold-over wrap-around, near-equal-intensity fold), ``zipper`` (a
localized 1-3 px dotted RF feed-through line, NOT a full-FOV grating),
``dc_offset`` (central bright/dark image-space dot = zero-frequency point
artifact), ``pe_undersample`` (PE-line dropout -> structured ghosts),
``motion_continuous`` (smooth per-PE-line drift), ``slab_wrap`` (3D
slice/partition-direction wrap-around, generalizes the fold to all 3 axes,
slab-biased), ``nyquist_ghost`` (EPI N/2 ghost: alternating per-ky-line phase
ramp -> FOV/2-locked replica).

Canonical signature: ``fn(arr, *, severity, rng, mask=None, **kw)``.
"""
from __future__ import annotations

import numpy as np

from augmentations.registry import register
from augmentations.numerics import iter_4d
from augmentations.artifacts._realistic_ringing import realistic_ring


# --------------------------------------------------------------------------- motion (rigid: rotation + translation)
def _rigid_pose_2d(slice2d, dy, dx, ang_deg):
    """Rotate (image space) by ang_deg then return the FFT of the posed slice."""
    from scipy.ndimage import rotate as _rot

    posed = slice2d
    if abs(ang_deg) > 1e-6:
        posed = _rot(slice2d, ang_deg, reshape=False, order=1, mode="nearest")
    k = np.fft.fftshift(np.fft.fft2(posed))
    ny, nx = slice2d.shape
    ky = (np.arange(ny) - ny // 2) / ny
    kx = (np.arange(nx) - nx // 2) / nx
    phase = np.exp(-2j * np.pi * (dy * ky[:, None] + dx * kx[None, :]))
    return k * phase


def _motion_slice_worker(z, work, poses, band_edges, centric, clo, chi,
                         cpose, magnitude):
    """Return one independently reconstructed float32 slice for ``motion``."""
    sl = work[:, :, z]
    kacc = np.fft.fftshift(np.fft.fft2(sl)).copy()
    for i, (dy, dx, ang) in enumerate(poses):
        lo, hi = band_edges[i], band_edges[i + 1]
        if hi <= lo:
            continue
        kacc[lo:hi, :] = _rigid_pose_2d(sl, dy, dx, ang)[lo:hi, :]
    if centric:
        lo, hi = max(0, clo), min(work.shape[0], chi)
        kacc[lo:hi, :] = _rigid_pose_2d(sl, *cpose)[lo:hi, :]
    reconstructed = np.fft.ifft2(np.fft.ifftshift(kacc))
    result = np.empty_like(sl, dtype=np.float32)
    result[...] = (np.abs(reconstructed) if bool(magnitude)
                   else np.real(reconstructed))
    return z, result


def _motion_slice_worker_count(slice_workers, batched_3d):
    """Validate and normalize the deliberately narrow slice-worker API."""
    if slice_workers is None or slice_workers is False:
        workers = 1
    elif type(slice_workers) is int and slice_workers in (0, 1, 2, 4):
        workers = max(1, slice_workers)
    else:
        raise ValueError(
            "slice_workers must be None, False, or an exact built-in int "
            "in {0, 1, 2, 4}")
    if workers > 1 and bool(batched_3d):
        raise ValueError(
            "slice_workers=2/4 is mutually exclusive with batched_3d")
    return workers


@register(
    "motion",
    kind="kspace",
    severity_range=(0.0, 1.0),
    dims="3d",
    label_preserving=True,
    has_detector=True,
)
def motion(arr, *, severity, rng, mask=None, n_poses=8, pe_axis=None,
           magnitude=False, batched_3d=False, slice_workers=None, **kw):
    """Rigid (rotation + translation) k-space-line motion model.

    For each axial slice we draw ``n_poses`` rigid poses (small rotation +
    translation, magnitude scaled by ``severity``), assign each pose a contiguous
    band of phase-encode (PE) lines, and assemble the corrupted k-space by taking
    each PE band from its pose's FFT. By default iFFT -> real part, preserving the
    historical byte contract. ``magnitude=True`` instead takes the complex-iFFT
    magnitude before clipping, matching a reconstructed magnitude MR image and
    avoiding signed-reconstruction zero clipping. ``batched_3d=True`` executes
    the same fixed pose schedule over all slices with volume-wise rotations and
    FFTs; its default is False so omitted/False calls preserve the historical
    byte and RNG contract. This is the Shaw/TorchIO rigid model and a strict
    upgrade over the translation-only proxy.

    Realism (superset): the PE axis is no longer hard-wired to rows. When
    ``pe_axis`` is None (the default / dataset path) it is sampled per volume from
    {0, 1} so ghosts are not always vertical -- PE direction follows protocol, not
    always the image rows; pass ``pe_axis=0`` to recover the old always-vertical
    behavior. ~40% of volumes use a *centric* ordering that acquires the central
    (highest-energy) PE lines from a single near-clean pose, so the DC band is
    usually LESS corrupted (real centric/elliptical k-space ordering); the rest use
    the original i.i.d.-per-band ordering. Both modes live inside one broadened
    distribution.
    """
    slice_worker_count = _motion_slice_worker_count(
        slice_workers, batched_3d)
    data = np.asarray(arr, dtype=np.float32)
    sev = float(severity)
    if sev <= 0.0:
        return data.copy()
    if data.ndim == 4:
        return iter_4d(motion, data, severity, rng, mask, n_poses=n_poses,
                        pe_axis=pe_axis, magnitude=magnitude,
                        batched_3d=batched_3d,
                        slice_workers=slice_worker_count, **kw)
    if data.ndim != 3:
        raise ValueError(f"motion expects 3D or 4D, got {data.ndim}D")
    # PE axis: sample per volume when not pinned (so ghosts aren't always vertical)
    ax = int(rng.integers(0, 2)) if pe_axis is None else int(pe_axis)
    # work in a frame where the PE axis is axis-0 (rows), then move it back
    work = data if ax == 0 else np.swapaxes(data, 0, 1)
    out = np.empty_like(work)
    ny, nx, nz = work.shape
    n_poses = max(1, min(int(n_poses), ny))
    band_edges = np.linspace(0, ny, n_poses + 1, dtype=int)
    max_shift = sev * min(ny, nx) * 0.1
    max_rot = sev * 6.0  # degrees
    # centric ordering: a contiguous central band of PE lines is filled by ONE
    # near-clean pose (acquired first, while still), sparing the DC center.
    centric = bool(rng.random() < 0.4)
    cband = max(1, int(0.12 * ny))
    clo, chi = ny // 2 - cband, ny // 2 + cband + 1
    # Draw the rigid-pose SCHEDULE ONCE for the whole volume — the head moves as one body during the
    # scan, so every slice's k-space is corrupted by the SAME pose-per-PE-band. Drawing fresh poses per
    # slice made each axial slice independent, and that slice-to-slice discontinuity read as STRAIGHT
    # LINES in the sagittal/coronal planes (the user's complaint). One schedule = 3D-coherent ghosting
    # (and correct for 3D sequences like MPRAGE, where motion does corrupt the whole volume coherently).
    poses = [(float(rng.uniform(-max_shift, max_shift)),
              float(rng.uniform(-max_shift, max_shift)),
              float(rng.uniform(-max_rot, max_rot))) for _ in range(n_poses)]
    cpose = (float(rng.uniform(-max_shift, max_shift)) * 0.25,
             float(rng.uniform(-max_shift, max_shift)) * 0.25,
             float(rng.uniform(-max_rot, max_rot)) * 0.25)   # near-clean central-line pose
    if bool(batched_3d):
        from scipy.ndimage import rotate as _rot

        # Keep the same complex128 FFT/phase arithmetic as the legacy slice
        # loop, but execute each fixed 2-D pose over every z slice at once.
        # fftshift/ifftshift are restricted to the two encoded axes so z is
        # never shifted.  The original FFT is retained as the conservative
        # baseline even though the clamped pose bands cover every PE row.
        kacc = np.fft.fftshift(
            np.fft.fft2(work, axes=(0, 1)), axes=(0, 1)).copy()
        ky = (np.arange(ny) - ny // 2) / ny
        kx = (np.arange(nx) - nx // 2) / nx
        for i, (dy, dx, ang) in enumerate(poses):
            lo, hi = band_edges[i], band_edges[i + 1]
            if hi <= lo:
                continue
            posed = work
            if abs(ang) > 1e-6:
                posed = _rot(work, ang, axes=(1, 0), reshape=False,
                              order=1, mode="nearest")
            posed_k = np.fft.fftshift(
                np.fft.fft2(posed, axes=(0, 1)), axes=(0, 1))
            phase = np.exp(-2j * np.pi * (
                dy * ky[:, None] + dx * kx[None, :]))
            kacc[lo:hi, :, :] = (
                posed_k[lo:hi, :, :] * phase[lo:hi, :, None])
            del posed_k
            if posed is not work:
                del posed
        if centric:
            lo, hi = max(0, clo), min(ny, chi)
            dy, dx, ang = cpose
            posed = work
            if abs(ang) > 1e-6:
                posed = _rot(work, ang, axes=(1, 0), reshape=False,
                              order=1, mode="nearest")
            posed_k = np.fft.fftshift(
                np.fft.fft2(posed, axes=(0, 1)), axes=(0, 1))
            phase = np.exp(-2j * np.pi * (
                dy * ky[:, None] + dx * kx[None, :]))
            kacc[lo:hi, :, :] = (
                posed_k[lo:hi, :, :] * phase[lo:hi, :, None])
            del posed_k
            if posed is not work:
                del posed
        reconstructed = np.fft.ifft2(
            np.fft.ifftshift(kacc, axes=(0, 1)), axes=(0, 1))
        out[...] = (np.abs(reconstructed) if bool(magnitude)
                    else np.real(reconstructed))
        del reconstructed, kacc
        out = out if ax == 0 else np.swapaxes(out, 0, 1)
        out = np.clip(out, 0.0, 1.0)
        return np.ascontiguousarray(out).astype(np.float32)
    if slice_worker_count > 1:
        from concurrent.futures import ThreadPoolExecutor

        worker_poses = tuple(poses)
        worker_band_edges = tuple(int(edge) for edge in band_edges)
        worker_cpose = tuple(cpose)
        max_in_flight = min(nz, 2 * slice_worker_count)
        executor = ThreadPoolExecutor(
            max_workers=slice_worker_count,
            thread_name_prefix="hardtail-motion")
        pending = {}
        next_submit = 0
        try:
            while next_submit < max_in_flight:
                pending[next_submit] = executor.submit(
                    _motion_slice_worker, next_submit, work, worker_poses,
                    worker_band_edges, centric, clo, chi, worker_cpose,
                    magnitude)
                next_submit += 1
            for next_collect in range(nz):
                future = pending.pop(next_collect)
                result_z, result_slice = future.result()
                if result_z != next_collect:
                    raise RuntimeError(
                        "motion slice worker returned out-of-order result")
                out[:, :, next_collect] = result_slice
                if next_submit < nz:
                    pending[next_submit] = executor.submit(
                        _motion_slice_worker, next_submit, work, worker_poses,
                        worker_band_edges, centric, clo, chi, worker_cpose,
                        magnitude)
                    next_submit += 1
        except BaseException:
            for future in pending.values():
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True, cancel_futures=False)
    else:
        for z in range(nz):
            sl = work[:, :, z]
            kacc = np.fft.fftshift(np.fft.fft2(sl)).copy()
            for i, (dy, dx, ang) in enumerate(poses):
                lo, hi = band_edges[i], band_edges[i + 1]
                if hi <= lo:
                    continue
                kacc[lo:hi, :] = _rigid_pose_2d(sl, dy, dx, ang)[lo:hi, :]
            if centric:
                lo, hi = max(0, clo), min(ny, chi)
                kacc[lo:hi, :] = _rigid_pose_2d(sl, *cpose)[lo:hi, :]
            reconstructed = np.fft.ifft2(np.fft.ifftshift(kacc))
            out[:, :, z] = (np.abs(reconstructed) if bool(magnitude)
                            else np.real(reconstructed))
    out = out if ax == 0 else np.swapaxes(out, 0, 1)
    out = np.clip(out, 0.0, 1.0)  # iFFT real part can over/undershoot [0,1]
    return np.ascontiguousarray(out).astype(np.float32)


# --------------------------------------------------------------------------- motion_continuous (smooth drift/nod)
@register(
    "motion_continuous",
    kind="kspace",
    severity_range=(0.0, 1.0),
    dims="3d",
    label_preserving=True,
    has_detector=True,
)
def motion_continuous(arr, *, severity, rng, mask=None, **kw):
    """Continuous-trajectory motion: a smooth zero-mean per-PE-line translation.

    Models slow drift / nodding: each PE line gets a translation sampled from a
    smooth (low-pass random) trajectory -> directional blur + coherent ghost.
    """
    data = np.asarray(arr, dtype=np.float32)
    sev = float(severity)
    if sev <= 0.0:
        return data.copy()
    if data.ndim == 4:
        return iter_4d(motion_continuous, data, severity, rng, mask, **kw)
    if data.ndim != 3:
        raise ValueError(f"motion_continuous expects 3D or 4D, got {data.ndim}D")
    ny, nx, nz = data.shape
    amp = sev * min(ny, nx) * 0.12
    ky = (np.arange(ny) - ny // 2) / ny
    kx = (np.arange(nx) - nx // 2) / nx
    # ONE smooth zero-mean trajectory for the WHOLE volume (the head drifts as one body during the
    # scan) -> 3D-coherent. Drawing a fresh trajectory per z-slice made each axial slice independent,
    # and that read as STRAIGHT-LINE striping in the sagittal/coronal planes (same bug as `motion`).
    from scipy.ndimage import gaussian_filter1d as _g1
    traj_y = _g1(np.cumsum(rng.standard_normal(ny)), sigma=ny / 8.0)
    traj_x = _g1(np.cumsum(rng.standard_normal(ny)), sigma=ny / 8.0)
    traj_y -= traj_y.mean(); traj_x -= traj_x.mean()
    sy = traj_y / (np.abs(traj_y).max() + 1e-9) * amp
    sx = traj_x / (np.abs(traj_x).max() + 1e-9) * amp
    phase_rows = [np.exp(-2j * np.pi * (sy[r] * ky[r] + sx[r] * kx)) for r in range(ny)]  # z-independent
    out = np.empty_like(data)
    for z in range(nz):
        k = np.fft.fftshift(np.fft.fft2(data[:, :, z]))
        for r in range(ny):
            k[r, :] = k[r, :] * phase_rows[r]
        out[:, :, z] = np.real(np.fft.ifft2(np.fft.ifftshift(k)))
    return np.clip(out, 0.0, 1.0).astype(np.float32)   # iFFT real part can over/undershoot [0,1]


# --------------------------------------------------------------------------- ghosting (verbatim)
@register(
    "ghosting",
    kind="kspace",
    severity_range=(0.0, 1.0),
    dims="3d",
    label_preserving=True,
    has_detector=True,
)
def ghosting(arr, *, severity, rng, mask=None, n_ghosts=None, axis=1, **kw):
    """Periodic PE ghosts via shifted-copy superposition.

    Realism (superset): real PE ghosting is dominated by a SINGLE discrete replica
    at FOV/2 (period-2 motion / aliasing of a coherent moving structure); a regular
    even ladder with monotone ``1/(i+2)`` decay is the predictable special case, not
    the typical look. When ``n_ghosts`` is None (the default / dataset path):

      * the ghost count is drawn weighted heavily toward 1 (single FOV/2 ghost),
        with a short tail up to ~4;
      * each ghost gets a CONTINUOUS fractional-FOV offset (a real fraction of the
        FOV, rounded to an integer roll), the first biased to ~FOV/2;
      * per-ghost intensities are randomized and NON-monotone (not a clean decay).

    Passing an explicit integer ``n_ghosts`` recovers the original even-ladder +
    monotone-decay mode byte-for-byte, so the old behavior stays reachable as one
    point inside the broadened distribution.
    """
    data = np.asarray(arr)
    if float(severity) <= 0.0:
        return data.copy()
    if data.ndim == 4:
        return iter_4d(ghosting, data, severity, rng, mask, n_ghosts=n_ghosts, axis=axis, **kw)
    if data.ndim != 3:
        raise ValueError(f"ghosting expects 3D or 4D, got {data.ndim}D")
    dtype = data.dtype
    work = data.astype(np.float32, copy=False)
    if axis < 0:
        axis = work.ndim + axis
    if not 0 <= axis < work.ndim:
        raise ValueError(f"axis {axis} out of bounds for {work.ndim}D array")
    n_along = work.shape[axis]
    out = work.copy()
    sev = float(severity)
    if n_ghosts is None:
        # superset default: weighted toward a single FOV/2 ghost, short tail to 4
        k = int(rng.choice([1, 2, 3, 4], p=[0.55, 0.25, 0.13, 0.07]))
        for i in range(k):
            if i == 0:
                # dominant replica near FOV/2 (continuous fractional offset, jittered)
                frac = float(rng.uniform(0.45, 0.55))
            else:
                frac = float(rng.uniform(0.12, 0.88))
            shift = int(round(frac * n_along))
            # randomized, non-monotone intensity (no clean 1/(i+2) decay)
            inten = float(rng.uniform(0.30, 1.00)) * (1.0 if i == 0 else float(rng.uniform(0.25, 0.85)))
            out = out + sev * inten * np.roll(work, shift=shift, axis=axis)
        # ROBUST RE-WINDOW instead of a hard clip: additive ghost replicas pile past 1.0 where they
        # overlap, and np.clip pinned ~1/3 of those voxels to a FLAT white plateau -- a saturation
        # cue the detector learns INSTEAD of the replica structure. Rescale by the high percentile so
        # overshoot maps to GRADED near-white; only triggers when there IS overshoot (mild unchanged).
        pos = out[out > 0.0]
        hi = float(np.percentile(pos, 99.5)) if pos.size else 1.0
        if hi > 1.0:
            out = out / hi
        return np.clip(out, 0.0, 1.0).astype(dtype, copy=False)
    # legacy even-ladder mode (explicit n_ghosts) -- byte-identical to the original
    n_ghosts = max(1, int(n_ghosts))
    base_step = max(1, n_along // (2 * n_ghosts))
    for i in range(n_ghosts):
        shift = int(base_step * (i + 1) + rng.integers(-1, 2))
        decay = 1.0 / (i + 2)
        out = out + sev * decay * np.roll(work, shift=shift, axis=axis)
    return out.astype(dtype, copy=False)


# --------------------------------------------------------------------------- eye / structured ghosting
@register(
    "eye_ghosting",
    kind="kspace",
    severity_range=(0.0, 1.0),
    dims="3d",
    label_preserving=True,
    has_detector=True,
    extra_parameters=(
        "base_amp", "beam_mode", "decay", "decay_frac",
        "disc_gain", "floor_amp", "ghost_step_frac", "lr_axis",
        "n_beams", "n_discs", "n_ghosts", "outside_amp",
        "outside_decay_frac", "outside_reach_frac", "reach_frac", "ripple_frac",
        "round_discs", "si_axis", "sigma_lr", "sigma_si",
        "step_frac", "tail_floor",
    ),
)
def eye_ghosting(arr, *, severity, rng, mask=None, axis=0, anterior_low=True,
                 bright_thr=0.30, **kw):
    """Eye-motion PE ghosting: vertical beams from the bright ANTERIOR structures
    (the eyes / orbital + facial fat) marching along the anterior->posterior axis INTO
    the brain as decaying, textured phase-encode bands.

    Unlike ``ghosting`` (which rolls the WHOLE volume -> a full-brain replica overlay),
    the ghost SOURCE is masked to the bright anterior EXTRACRANIAL tissue, so the beams
    read as eye/face ghosts laid over the brain -- the realistic look of eye movement during
    a phase-encode-along-A-P acquisition. A real reference (``eye_ghosting_real.jpg``) shows
    the L-R variant as horizontal bands AT the eyes; this is the A-P variant whose bands
    march INTO the brain.

    Orientation (defaults = NFBS axcodes 'P','I','R'): ``axis`` is the anterior<->posterior
    in-plane axis (axis 0) and ``anterior_low`` marks the anterior end as the LOW index, so
    ghosts shift toward the brain (away from the face). ``severity`` scales replica count,
    amplitude and reach. Pass ``axis``/``anterior_low`` to override for other orientations.
    """
    data = np.asarray(arr)
    sev = float(severity)
    if sev <= 0.0:
        return data.copy()
    if data.ndim == 4:
        return iter_4d(eye_ghosting, data, severity, rng, mask, axis=axis,
                        anterior_low=anterior_low, bright_thr=bright_thr, **kw)
    if data.ndim != 3:
        raise ValueError(f"eye_ghosting expects 3D or 4D, got {data.ndim}D")
    work = data.astype(np.float32, copy=False)
    if axis < 0:
        axis = work.ndim + axis
    if not 0 <= axis < work.ndim:
        raise ValueError(f"axis {axis} out of bounds for {work.ndim}D array")
    n_along = work.shape[axis]

    # brain region (mask if given, else a bright-foreground proxy)
    if mask is not None and np.asarray(mask).any():
        brain = np.asarray(mask) > 0.5
    else:
        brain = work > 0.5 * float(np.nanmax(work) or 1.0)
    if not brain.any():
        return work.copy()

    # ANTERIOR bright extracranial source = eyes / orbital + facial fat. "Anterior" is
    # the side of the brain centroid toward the low (anterior_low) / high index along axis.
    from scipy.ndimage import binary_dilation
    bc = float(np.array(np.nonzero(brain)).mean(axis=1)[axis])
    shp = [1, 1, 1]; shp[axis] = n_along
    coord = np.arange(n_along).reshape(shp)
    anterior = (coord < bc) if anterior_low else (coord > bc)
    extracranial = ~binary_dilation(brain, iterations=2)
    src_mask = (work > float(bright_thr)) & extracranial & anterior
    if int(src_mask.sum()) < 200:
        # skull-stripped / no anterior face tissue -> use any anterior bright tissue so the
        # effect still appears (degrades gracefully); empty -> no-op (visibility filter drops it).
        src_mask = (work > float(bright_thr)) & anterior
    if int(src_mask.sum()) < 50:
        return work.copy()

    # Isolate ONLY THE EYES (not the whole anterior fat) so the ghosts are THIN BEAMS coming
    # out of the eyes, not wide face-wide bands. Three restrictions on the bright anterior
    # source, with NFBS-default in-plane axes (si_axis = I-S = the axial-slice axis, lr_axis =
    # L-R), overridable via kw:
    #   * A-P SLAB  — a thin slab at the eyes' A-P position, used as the beam origin.
    #   * L-R BAND  — two lateral eye columns about the L-R midline (drops the central nose and
    #                 the outer scalp), so the beams are narrow and PAIRED.
    #   * S-I BAND  — the orbit level (+ a little superior), so only eye-level slices ghost and
    #                 the forehead fat above does not paint beams onto the upper brain slices.
    si_axis = int(kw.get("si_axis", 1))
    lr_axis = int(kw.get("lr_axis", 2))

    def _band_mask(band_axis, lo, hi):
        n = work.shape[band_axis]
        keep = (np.arange(n) >= lo) & (np.arange(n) <= hi)
        shp = [1, 1, 1]; shp[band_axis] = n
        return keep.reshape(shp)

    # Localize the eyes by brain-BBOX GEOMETRY, NOT by "densest anterior bright" (which grabs
    # the forehead/scalp fat at mid S-I, not the orbits). The orbits sit just ANTERIOR of the
    # brain's front edge, in the INFERIOR portion of its S-I span, in two LATERAL columns.
    nz = [np.nonzero(brain.any(axis=tuple(j for j in range(brain.ndim) if j != a)))[0]
          for a in range(brain.ndim)]
    blo = [int(z[0]) for z in nz]; bhi = [int(z[-1]) for z in nz]
    bspan = [max(1, bhi[a] - blo[a]) for a in range(brain.ndim)]
    # A-P: at/just anterior of the brain front edge (eyes protrude ahead of the brain)
    if anterior_low:
        ap_keep = _band_mask(axis, 0, blo[axis] + int(round(0.05 * bspan[axis])))
    else:
        ap_keep = _band_mask(axis, bhi[axis] - int(round(0.05 * bspan[axis])), n_along - 1)
    # S-I: the orbit band — inferior half-ish of the brain S-I span (its SUPERIOR end still has
    # temporal-lobe/midbrain tissue, which is the slice we display).
    si_keep = _band_mask(si_axis, blo[si_axis] + int(round(0.52 * bspan[si_axis])),
                         blo[si_axis] + int(round(0.88 * bspan[si_axis])))
    # L-R: two eye columns about the L-R midline (drop the central nose + the outer scalp)
    lr_mid = 0.5 * (blo[lr_axis] + bhi[lr_axis])
    d_lr = np.abs(np.arange(work.shape[lr_axis]) - lr_mid)
    lr_keep = ((d_lr >= 0.05 * bspan[lr_axis]) & (d_lr <= 0.26 * bspan[lr_axis]))
    shp_lr = [1, 1, 1]; shp_lr[lr_axis] = work.shape[lr_axis]

    src_mask = src_mask & ap_keep & si_keep & lr_keep.reshape(shp_lr)
    if int(src_mask.sum()) < 30:
        return work.copy()

    into_brain = 1 if anterior_low else -1
    mode = str(kw.get("beam_mode", "beams"))
    old_modes = {"streak", "discrete", "discs"}

    # The legacy ghost SOURCE that gets replicated. Explicit old modes can still use:
    #   * round_discs=True: replace the blocky orbital-fat patch with a clean SOFT
    #     ELLIPSOID disc stamped at each eye's centroid (left/right split about the L-R midline),
    #     so the replicas read as distinct round DISCS rather than ragged fat blobs.
    #   * round_discs=False: use the raw masked image values (faithful but irregular).
    if mode in old_modes and bool(kw.get("round_discs", mode == "discs")):
        idx = np.array(np.nonzero(src_mask))                 # (3, N) eye-voxel coords
        lr_coord = idx[lr_axis]
        mid = 0.5 * (float(lr_coord.min()) + float(lr_coord.max()))
        grids = np.ogrid[tuple(slice(0, s) for s in work.shape)]
        source = np.zeros_like(work)
        for side in (lr_coord <= mid, lr_coord > mid):       # left eye, right eye
            if int(side.sum()) < 5:
                continue
            sub = idx[:, side]
            cen = sub.mean(axis=1)
            rad = [max(2.0, 0.5 * (float(sub[a].max()) - float(sub[a].min())) + 1.0)
                   for a in range(work.ndim)]
            # disc peak = the BRIGHT orbital-fat intensity (high percentile, not the median, so the
            # ghost discs are as bright as the real fat that casts them — else they wash out on brain)
            bright = float(kw.get("disc_gain", 1.8)) * float(np.percentile(work[tuple(sub)], 95))
            ee = sum(((grids[a] - cen[a]) / rad[a]) ** 2 for a in range(work.ndim))
            source = source + bright * np.exp(-2.2 * ee).astype(np.float32)
        source = source.astype(np.float32)
        if not source.any():                                 # degenerate split -> fall back
            source = np.where(src_mask, work, 0.0).astype(np.float32)
    else:
        source = np.where(src_mask, work, 0.0).astype(np.float32)

    # Render or replicate the eye source along the PE axis. The default draws
    # continuous textured vertical beams; legacy modes still roll source replicas.
    # Looks (``beam_mode``):
    #   * "beams"    - soft vertical PE beams from each eye (default).
    #   * "discs"    - legacy periodic disc train spanning the FOV.
    #   * "streak"   - legacy continuous bright trail into the brain.
    #   * "discrete" - legacy separated replicas.
    # All knobs are kw-overridable so the look can be swept without touching callers.
    # A shallow curve makes mid-tier eye ghosting visibly corrupting while letting
    # severe tiers climb hard without turning the mild tier into a failure case.
    sev_curve = float(np.clip(sev, 0.0, 1.0) ** 0.85)
    base_amp = float(kw.get("base_amp", 0.080 + 1.15 * sev_curve))
    out = work.copy()
    if mode in {"beams", "vertical", "vertical_beams"}:
        from scipy.ndimage import gaussian_filter

        idx = np.array(np.nonzero(src_mask))
        lr_coord = idx[lr_axis]
        mid = 0.5 * (float(lr_coord.min()) + float(lr_coord.max()))
        grids = np.ogrid[tuple(slice(0, s) for s in work.shape)]

        coord_axis = grids[axis].astype(np.float32)
        tex = rng.normal(0.0, 1.0, n_along).astype(np.float32)
        tex = gaussian_filter(tex, sigma=max(1.0, 0.010 * n_along), mode="nearest")
        tex = (tex - float(tex.min())) / (float(tex.max() - tex.min()) + 1e-6)
        period = max(5.0, float(kw.get("ripple_frac", 0.055)) * n_along)
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        ripple = 0.88 + 0.12 * np.sin(2.0 * np.pi * np.arange(n_along) / period + phase)
        tex = np.clip((0.72 + 0.34 * tex) * ripple, 0.48, 1.08).astype(np.float32)
        tshp = [1, 1, 1]; tshp[axis] = n_along
        tex = tex.reshape(tshp)

        beam = np.zeros_like(work, dtype=np.float32)
        reach = float(kw.get("reach_frac", 0.58 + 0.40 * sev_curve)) * n_along
        decay_len = max(4.0, float(kw.get("decay_frac", 0.30)) * n_along)
        outside_decay = max(4.0, float(kw.get("outside_decay_frac", 0.40)) * n_along)
        outside_amp = float(kw.get("outside_amp", 0.48 + 0.38 * sev_curve))
        outside_reach_scale = float(kw.get("outside_reach_frac", 1.0))
        n_lr = work.shape[lr_axis]

        for side in (lr_coord <= mid, lr_coord > mid):
            if int(side.sum()) < 5:
                continue
            sub = idx[:, side]
            cen = sub.mean(axis=1)
            start = float(np.percentile(sub[axis], 65 if anterior_low else 35))
            d = into_brain * (coord_axis - start)
            tail_floor = float(kw.get("tail_floor", 0.10 + 0.26 * sev_curve))
            long_core = np.exp(-np.maximum(d, 0.0) / decay_len)
            inside_tail = np.maximum(tail_floor, long_core)
            outside_edge = start if anterior_low else (n_along - 1 - start)
            outside_reach = max(1.0, outside_reach_scale * outside_edge)
            outside_tail = outside_amp * np.exp(-np.maximum(-d, 0.0) / outside_decay)
            long = np.where((d < 0.0) & (d >= -outside_reach), outside_tail,
                            np.where((d >= 0.0) & (d <= reach), inside_tail, 0.0))

            lr_extent = max(1.0, float(sub[lr_axis].max() - sub[lr_axis].min() + 1))
            si_extent = max(1.0, float(sub[si_axis].max() - sub[si_axis].min() + 1))
            widen = 1.0 + 0.25 * sev_curve
            sigma_lr = float(kw.get("sigma_lr", min(max(4.0, 0.45 * lr_extent + 1.8),
                                                     max(5.0, 0.045 * n_lr)) * widen))
            sigma_si = float(kw.get("sigma_si", max(1.8, 0.42 * si_extent + 0.8)))
            lr_d = grids[lr_axis] - cen[lr_axis]
            lr_prof = (np.exp(-0.5 * (lr_d / sigma_lr) ** 2) +
                       0.16 * np.exp(-0.5 * (lr_d / (2.7 * sigma_lr)) ** 2))
            si_prof = np.exp(-0.5 * ((grids[si_axis] - cen[si_axis]) / sigma_si) ** 2)

            bright = float(np.percentile(work[tuple(sub)], 92))
            beam = beam + bright * long.astype(np.float32) * lr_prof.astype(np.float32) * si_prof.astype(np.float32) * tex

        if beam.any():
            gsig = [0.45, 0.45, 0.45]
            gsig[axis] = 1.10
            grain = rng.normal(0.0, 1.0, work.shape).astype(np.float32)
            grain = gaussian_filter(grain, sigma=gsig, mode="nearest")
            grain = grain / (float(grain.std()) + 1e-6)
            beam *= np.clip(1.0 + 0.10 * grain, 0.72, 1.22)
            out = out + base_amp * beam
    elif mode == "streak":
        decay = float(kw.get("decay", 0.90))
        reach = max(2, int(round(float(kw.get("reach_frac", 0.22 + 0.18 * sev)) * n_along)))
        for d in range(1, reach + 1):
            out = out + base_amp * (decay ** (d - 1)) * np.roll(source, shift=into_brain * d, axis=axis)
        for i in range(int(kw.get("n_ghosts", 2))):
            off = into_brain * int(round((i + 1) * float(kw.get("ghost_step_frac", 0.17)) * n_along))
            out = out + 0.45 * base_amp * (0.7 ** i) * np.roll(source, shift=off, axis=axis)
    elif mode == "discrete":
        decay = float(kw.get("decay", 0.72))
        k = int(kw.get("n_beams", 2 + int(round(3.0 * sev))))
        step = float(kw.get("step_frac", 0.11 + 0.03 * sev)) * n_along
        for i in range(k):
            off = into_brain * int(round((i + 1) * step + float(rng.uniform(-2.0, 2.0))))
            out = out + base_amp * (decay ** i) * float(rng.uniform(0.9, 1.0)) * np.roll(source, shift=off, axis=axis)
    else:  # "discs" - legacy PE-ghost disc TRAIN of the eyes across the whole FOV
        decay = float(kw.get("decay", 0.86))               # per-replica falloff from the source
        floor = float(kw.get("floor_amp", 0.32))           # distant discs stay visible (fraction of base)
        step = float(kw.get("step_frac", 0.16)) * n_along  # GAP between discs (distance between them)
        nrep = int(kw.get("n_discs", max(2, int(round(n_along / max(1.0, step))) - 1)))  # span FOV (wraps)
        for i in range(nrep):
            off = into_brain * int(round((i + 1) * step + float(rng.uniform(-1.5, 1.5))))
            amp = base_amp * max(floor, decay ** i) * float(rng.uniform(0.92, 1.0))
            out = out + amp * np.roll(source, shift=off, axis=axis)
    return np.clip(out, 0.0, 1.0).astype(data.dtype, copy=False)


# --------------------------------------------------------------------------- gibbs (= gibbs_realistic)
@register(
    "gibbs",
    kind="kspace",
    severity_range=(0.0, 1.0),
    dims="3d",
    label_preserving=True,
    has_detector=True,
    extra_parameters=("axial_axis",),
)
def gibbs(arr, *, severity, rng, mask=None, **kw):
    """Realistic ringing — oval rings ADDED onto the SHARP volume using image-derived support,
    tracking the head's PER-AXIAL-SLICE outline (2-D ellipse fit per slice), with
    uneven spacing, random curvature, partial arcs and an OCCASIONAL (probabilistic)
    aliasing/motion fold-over. See ``_realistic_ringing.realistic_ring``.

    Replaces the old sharp k-space rectangular truncation (which BLURRED the image — a
    look the user rejected: real ringing is rings on an otherwise-sharp brain, not blur).
    The axial slicing axis is auto-detected from the image foreground's bilateral-symmetry axis
    (pass ``axial_axis`` to override). The supplied ``mask`` is intentionally ignored: using a
    segmentation target to place/constrain ringing exposes the answer boundary to the masker. The
    ringing engine instead derives its support from the image via an Otsu foreground estimate.

    QC class: ``graded_steep`` (magnitude-gated, SENSITIVE) in augment_qc_dataset — only
    the faintest ringing stays near-clean; visible rings score moderate, extreme -> severe.
    """
    data = np.asarray(arr)
    sev = float(severity)
    if sev <= 0.0:
        return data.copy()
    if data.ndim == 4:
        return iter_4d(gibbs, data, severity, rng, mask, **kw)
    if data.ndim != 3:
        raise ValueError(f"gibbs expects 3D or 4D, got {data.ndim}D")
    out = realistic_ring(data.astype(np.float32, copy=False), severity=sev, rng=rng,
                         brain=None, axial_axis=kw.get("axial_axis"))
    return out.astype(data.dtype, copy=False)


# --------------------------------------------------------------------------- spike / herringbone


def _add_kspace_spikes_3d(vol, n_spikes, amp_frac, rng):
    """3D-COHERENT k-space spikes: one 3D FFT, off-DC 3D points (+ Hermitian mirror). The corduroy /
    herringbone grating is then coherent in ALL THREE planes — the per-slice 2D version made each axial
    slice independent, which read as random speckle in the coronal/sagittal views (same striping class
    as the old motion)."""
    k = np.fft.fftshift(np.fft.fftn(vol))
    kmax = float(np.abs(k).max())
    c = [s // 2 for s in vol.shape]
    for _ in range(int(n_spikes)):
        off = [int(rng.integers(2, max(3, vol.shape[i] // 2))) * (1 if rng.random() < 0.5 else -1)
               for i in range(3)]
        p = [c[i] + off[i] for i in range(3)]
        if not all(0 <= p[i] < vol.shape[i] for i in range(3)):
            continue
        val = amp_frac * kmax
        k[p[0], p[1], p[2]] += val
        mp = [2 * c[i] - p[i] for i in range(3)]
        if all(0 <= mp[i] < vol.shape[i] for i in range(3)):
            k[mp[0], mp[1], mp[2]] += val
    return np.real(np.fft.ifftn(np.fft.ifftshift(k)))


@register(
    "spike",
    kind="kspace",
    severity_range=(0.0, 1.0),
    dims="3d",
    label_preserving=True,
    has_detector=True,
)
def spike(arr, *, severity, rng, mask=None, n_spikes=1, **kw):
    """Single k-space spike (RF interference): one off-DC point + conjugate mirror.

    Produces a sinusoidal corduroy pattern across the image. Validates the orphan
    ``mri_qc ... single_spike_rf`` detector (detector existed, no simulator).
    """
    data = np.asarray(arr, dtype=np.float32)
    sev = float(severity)
    if sev <= 0.0:
        return data.copy()
    if data.ndim == 4:
        return iter_4d(spike, data, severity, rng, mask, n_spikes=n_spikes, **kw)
    if data.ndim != 3:
        raise ValueError(f"spike expects 3D or 4D, got {data.ndim}D")
    amp = 0.05 + 0.45 * sev
    return _add_kspace_spikes_3d(data, n_spikes, amp, rng).astype(np.float32)   # 3D-coherent (no per-slice striping)


@register(
    "herringbone",
    kind="kspace",
    severity_range=(0.0, 1.0),
    dims="3d",
    label_preserving=True,
    has_detector=True,
)
def herringbone(arr, *, severity, rng, mask=None, n_spikes=4, **kw):
    """Herringbone / crosshatch: multiple k-space spikes (n_spikes 2-5).

    Same core as ``spike`` with several points; the overlapping corduroy gratings
    read as a herringbone weave.
    """
    return spike(arr, severity=severity, rng=rng, mask=mask, n_spikes=n_spikes, **kw)


# --------------------------------------------------------------------------- zipper (RF line)
@register(
    "zipper",
    kind="kspace",
    severity_range=(0.0, 1.0),
    dims="3d",
    label_preserving=True,
    has_detector=True,
)
def zipper(arr, *, severity, rng, mask=None, axis=0, **kw):
    """Zipper / RF-line artifact: a localized 1-3 px DOTTED line of alternating teeth.

    The physical zipper is RF feed-through into a *single* k-space line: it lands as
    a narrow line (1-3 px thick) of alternating bright/dark "teeth" running across
    the FOV at a center-biased position -- NOT a full-FOV grating (those belong to
    ``spike``/``herringbone``, which keep the family visually distinct).

    Realism (superset): the old implementation synthesized a full-FOV cosine *band*
    spanning every row/column. This now confines the modulation to ``1 + round(2*sev)``
    narrow lines, each at a jittered center-biased position, with alternating
    bright/dark teeth (a per-pixel +/- comb) at a tissue-referenced amplitude. The
    teeth/orientation/sign/count/amplitude are over-generated so faint single dotted
    lines through full crossing-zippers are all reachable.
    """
    data = np.asarray(arr, dtype=np.float32)
    sev = float(severity)
    if sev <= 0.0:
        return data.copy()
    if data.ndim == 4:
        return iter_4d(zipper, data, severity, rng, mask, axis=axis, **kw)
    if data.ndim != 3:
        raise ValueError(f"zipper expects 3D or 4D, got {data.ndim}D")
    ny, nx, nz = data.shape
    # BRIGHT-tissue reference (full-head scans are dominated by near-zero background,
    # so median(positives) collapses to the air floor -- use the brain median when a
    # mask is given, else a high percentile of the positive voxels).
    m = np.asarray(mask).astype(bool) if mask is not None else None
    if m is not None and m.any():
        ref = float(np.median(data[m]))
    else:
        pos = data[np.isfinite(data) & (data > 0)]
        ref = float(np.percentile(pos, 85)) if pos.size else 1.0
    amp = (0.20 + 0.80 * sev) * ref               # tooth amplitude: faint -> strong
    n_lines = 1 + int(round(2 * sev))             # 1..3 lines with severity
    out = data.astype(np.float32).copy()
    for _ in range(n_lines):
        # orient the dotted line across the FOV: pick the run axis (0 or 1)
        run_axis = int(rng.integers(0, 2))
        pos_axis = 1 - run_axis
        span_run = ny if run_axis == 0 else nx
        span_pos = ny if pos_axis == 0 else nx
        thick = int(rng.integers(1, 4))           # 1-3 px line thickness
        # center-biased position (RF lines cluster near k-space center artifacts)
        center = span_pos // 2
        loc = int(np.clip(center + rng.normal(0, span_pos * 0.18), 0, span_pos - 1))
        lo = max(0, loc - thick // 2)
        hi = min(span_pos, lo + thick)
        # alternating bright/dark teeth along the run axis
        sign0 = 1.0 if rng.random() < 0.5 else -1.0
        run = np.arange(span_run, dtype=np.float32)
        teeth = sign0 * amp * np.where((np.floor(run) % 2) == 0, 1.0, -1.0)
        if run_axis == 0:   # line runs down rows; spans columns [lo:hi]
            out[:, lo:hi, :] += teeth[:, None, None]
        else:               # line runs across cols; spans rows [lo:hi]
            out[lo:hi, :, :] += teeth[None, :, None]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- central DC-offset dot
@register(
    "dc_offset",
    kind="kspace",
    severity_range=(0.0, 1.0),
    dims="3d",
    label_preserving=True,
    has_detector=True,
)
def dc_offset(arr, *, severity, rng, mask=None, **kw):
    """Zero-frequency point artifact: a bright/dark Gaussian DOT at the FOV center.

    The real "DC"/central-point artifact is a single bright (or dark) pixel at the
    center of the FOV: a constant offset across ALL k-samples is a delta at image
    CENTER, so it shows as a localized dot, not a flat field. (Bumping the k-space
    DC *bin* would instead add a flat brightness offset over the whole image -- the
    literal Fourier dual -- which ``robust_normalize`` largely removes; that was the
    old bug, now fixed.)

    Realism (superset): the default draws an image-space dot
    ``sign * amp * exp(-r^2 / 2 sigma^2)`` at the slice center, sigma ~1-3 px,
    over-generating sign (bright + dark), amplitude, sub-pixel jitter, and an
    occasional second replica. A faint flat-offset mode is still reachable as a
    low-probability tail (~15%), so the broadened distribution spans dot -> flat.
    """
    data = np.asarray(arr, dtype=np.float32)
    sev = float(severity)
    if sev <= 0.0:
        return data.copy()
    if data.ndim == 4:
        return iter_4d(dc_offset, data, severity, rng, mask, **kw)
    if data.ndim != 3:
        raise ValueError(f"dc_offset expects 3D or 4D, got {data.ndim}D")
    ny, nx, nz = data.shape
    cy, cx = ny / 2.0, nx / 2.0
    out = data.astype(np.float32).copy()
    # faint flat-offset mode (low-prob tail; spans dot -> flat in the superset)
    if rng.random() < 0.15:
        sign = 1.0 if rng.random() < 0.5 else -1.0
        off = sign * (0.02 + 0.10 * sev) * float(rng.uniform(0.5, 1.0))
        return np.clip(out + off, 0.0, 1.0).astype(np.float32)
    # central bright/dark dot in IMAGE space
    yy, xx = np.indices((ny, nx)).astype(np.float32)
    n_dots = 1 + (1 if rng.random() < 0.25 else 0)  # occasional replica
    amp_base = 0.30 + 0.70 * sev                     # dot amplitude (normalized)
    for _ in range(n_dots):
        sign = 1.0 if rng.random() < 0.5 else -1.0
        sigma = float(rng.uniform(1.0, 3.0))
        jy = float(rng.uniform(-1.5, 1.5)); jx = float(rng.uniform(-1.5, 1.5))
        amp = sign * amp_base * float(rng.uniform(0.7, 1.0))
        r2 = (yy - (cy + jy)) ** 2 + (xx - (cx + jx)) ** 2
        dot = (amp * np.exp(-r2 / (2.0 * sigma * sigma))).astype(np.float32)
        out = out + dot[:, :, None]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- PE-line undersampling -> aliasing
@register(
    "pe_undersample",
    kind="kspace",
    severity_range=(0.0, 1.0),
    dims="3d",
    label_preserving=True,
    has_detector=True,
)
def pe_undersample(arr, *, severity, rng, mask=None, magnitude=False, **kw):
    """PE-line dropout undersampling -> structured wrap/aliasing ghosts.

    Zero a fraction of phase-encode (ky) lines (regular decimation + jitter); the
    missing lines produce coherent fold-over replicas (parallel-imaging style).
    The historical default reconstructs the real component byte-identically.
    ``magnitude=True`` reconstructs the complex magnitude and clips to [0,1],
    matching a magnitude image without converting negative ringing lobes to
    exact zeros in a downstream clip.
    """
    data = np.asarray(arr, dtype=np.float32)
    sev = float(severity)
    if sev <= 0.0:
        return data.copy()
    if data.ndim == 4:
        return iter_4d(pe_undersample, data, severity, rng, mask,
                        magnitude=magnitude, **kw)
    if data.ndim != 3:
        raise ValueError(f"pe_undersample expects 3D or 4D, got {data.ndim}D")
    ny, nx, nz = data.shape
    # acceleration factor R: 2..(2 + ~3*sev); keep 1/R of the lines
    R = 2 + int(round(3 * sev))
    keep_mask = np.zeros(ny, bool)
    keep_mask[::R] = True
    # always keep a central autocalibration band
    band = max(1, int(0.08 * ny))
    keep_mask[ny // 2 - band: ny // 2 + band + 1] = True
    out = np.empty_like(data)
    for z in range(nz):
        k = np.fft.fftshift(np.fft.fft2(data[:, :, z]))
        k[~keep_mask, :] = 0.0
        reconstructed = np.fft.ifft2(np.fft.ifftshift(k))
        out[:, :, z] = (np.abs(reconstructed) if bool(magnitude)
                        else np.real(reconstructed))
    if bool(magnitude):
        out = np.clip(out, 0.0, 1.0)
    return out.astype(np.float32)


@register(
    "aliasing",
    kind="kspace",
    severity_range=(0.0, 1.0),
    dims="3d",
    label_preserving=True,
    has_detector=True,
)
def aliasing(arr, *, severity, rng, mask=None, axis=0, **kw):
    """Wrap-around / fold-over aliasing along the PE axis.

    Reduced-FOV PE wrap: a fraction of the image folds from one edge onto the
    opposite edge (the classic aliasing where the nose wraps onto the back of the
    head). Validates ``mri_qc ... aliasing_detection`` (detector existed, no sim).

    Realism (superset): real wrap-around is real signal folded in at ~EQUAL coil
    sensitivity, so the fold reads as anatomy, not a dim ghost. The amplitude is now
    ``0.8 + 0.2*sev`` (near-equal at all severities) so SEVERITY drives the wrap
    EXTENT, not the dimness (the old ``0.5 + 0.5*sev`` made low severity a faint
    ghost indistinguishable from ``ghosting``). The roll direction is randomized, and
    a fraction of calls fold from BOTH edges (top+bottom wrap) -- so the broadened
    distribution still reaches the old single-direction fold but no longer defaults
    to a dim one-sided ghost.
    """
    data = np.asarray(arr, dtype=np.float32)
    sev = float(severity)
    if sev <= 0.0:
        return data.copy()
    if data.ndim == 4:
        return iter_4d(aliasing, data, severity, rng, mask, axis=axis, **kw)
    if data.ndim != 3:
        raise ValueError(f"aliasing expects 3D or 4D, got {data.ndim}D")
    n = data.shape[axis]
    # TRUE reduced-FOV fold: only the out-of-FOV EDGE STRIP (width `wrap`, severity-driven) aliases
    # onto the opposite edge -- NOT the whole volume. The old np.roll blended a shifted copy of the
    # ENTIRE image into itself, so the in-FOV centre was doubled too; that global self-blend dominated
    # the metric (RMS plateaued past band 0.2) and entangled wrap-EXTENT with global contrast. A
    # localized strip fold makes the affected fraction (hence the degree) scale monotonically with the
    # wrap extent, leaving the in-FOV anatomy clean -- the way real wrap-around looks.
    wrap = max(1, min(n // 2, int(round((0.06 + 0.44 * sev) * n))))
    amp = 0.8 + 0.2 * sev                              # near-equal-intensity fold (reads as anatomy)
    out = data.copy().astype(np.float32)

    def _fold(dst_lo, dst_hi, src_lo, src_hi):
        d = [slice(None)] * data.ndim; s = [slice(None)] * data.ndim
        d[axis] = slice(dst_lo, dst_hi); s[axis] = slice(src_lo, src_hi)
        out[tuple(d)] = (data[tuple(d)] + amp * data[tuple(s)]) / (1.0 + amp)

    direction = 1 if rng.random() < 0.5 else -1
    both = rng.random() < 0.5                          # reduced FOV usually clips BOTH ends
    if direction > 0 or both:                          # high strip wraps onto the low edge
        _fold(0, wrap, n - wrap, n)
    if direction < 0 or both:                          # low strip wraps onto the high edge
        _fold(n - wrap, n, 0, wrap)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- 3D slab / partition-direction wrap-around
@register(
    "slab_wrap",
    kind="kspace",
    severity_range=(0.0, 1.0),
    dims="3d",
    label_preserving=True,
    has_detector=True,
)
def slab_wrap(arr, *, severity, rng, mask=None, axis=None, **kw):
    """3D slice/partition-direction wrap-around (generalizes ``aliasing`` to all axes).

    3D-encoded sequences (MPRAGE / SPACE / 3D-FLAIR) phase-encode the slice/partition
    axis too; a too-thin slab folds the top of the head onto the bottom partitions (or
    one in-plane edge onto the other). This generalizes the in-plane-only ``aliasing``
    fold so the wrap axis is sampled from ALL THREE axes, biased toward the
    slab/partition direction (the third axis), which is the regime ``aliasing`` and
    ``pe_undersample`` (both in-plane) never reach -- a real threat to skull-strip / QC
    because folded scalp/nose can land inside the brain.

    Realism (superset): same near-equal fold amplitude as ``aliasing`` (``0.8+0.2*sev``
    so severity drives wrap EXTENT not dimness); over-generates fold axis, fraction,
    direction, and single/both-edge.
    """
    data = np.asarray(arr, dtype=np.float32)
    sev = float(severity)
    if sev <= 0.0:
        return data.copy()
    if data.ndim == 4:
        return iter_4d(slab_wrap, data, severity, rng, mask, axis=axis, **kw)
    if data.ndim != 3:
        raise ValueError(f"slab_wrap expects 3D or 4D, got {data.ndim}D")
    if axis is None:
        # slab-biased: axis 2 (partition/through-slab) ~60%, in-plane axes the rest
        ax = int(rng.choice([0, 1, 2], p=[0.2, 0.2, 0.6]))
    else:
        ax = int(axis)
    n = data.shape[ax]
    if n < 2:
        return data.copy()
    # Severity drives BOTH the wrap EXTENT and the fold amplitude (each centered on sev with a
    # +/-15% random tail), so the artifact degree is monotone in `severity` instead of rng-only
    # (the old uniform(0.12,0.5)/uniform(0.35,1.0) made mild and severe statistically identical).
    wrap = max(1, min(n - 1, int(round((0.10 + 0.40 * sev) * n * float(rng.uniform(0.85, 1.15))))))
    amp = float(np.clip((0.45 + 0.50 * sev) * float(rng.uniform(0.85, 1.15)), 0.2, 1.0))
    direction = 1 if rng.random() < 0.5 else -1
    acc = amp * np.roll(data, shift=direction * wrap, axis=ax)
    w = amp
    if rng.random() < 0.35:
        acc = acc + amp * np.roll(data, shift=-direction * wrap, axis=ax)
        w += amp
    # CONVEX combine (renormalize), not additive-then-clip — see `aliasing`: the self-overlap would
    # otherwise saturate the whole head to white. Divide by (1+w) -> fold stays a ~50% ghost, stays <=1.
    out = (data + acc) / (1.0 + w)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- EPI Nyquist (N/2) ghost
@register(
    "nyquist_ghost",
    kind="kspace",
    severity_range=(0.0, 1.0),
    dims="3d",
    label_preserving=True,
    has_detector=False,
)
def nyquist_ghost(arr, *, severity, rng, mask=None, axis=None, **kw):
    """EPI Nyquist (N/2) ghost: alternating per-ky-line phase error -> FOV/2 replica.

    EPI reverses the readout gradient polarity on alternate ky lines; residual
    eddy-current / timing mismatch leaves a phase error on every OTHER line. A phase
    term that alternates +/- per ky line is a modulation by ``cos(pi*ky)`` (period 2
    in k-space) -> a replica shifted by exactly FOV/2 in image space, with intensity
    tied to the odd/even mismatch. This is mechanistically distinct from ``ghosting``
    (a generic shifted-copy ladder): the ghost is LOCKED to FOV/2 and its strength is
    set by the alternating phase amplitude.

    Realism (superset): over-generates the per-line phase-error amplitude (-> partial
    to strong cancellation/replica), the PE axis (in-plane 0 or 1), and a constant
    phase offset. Lower priority for the structural corpus (EPI is not the core), but
    closes the EPI N/2 coverage gap.
    """
    data = np.asarray(arr, dtype=np.float32)
    sev = float(severity)
    if sev <= 0.0:
        return data.copy()
    if data.ndim == 4:
        return iter_4d(nyquist_ghost, data, severity, rng, mask, axis=axis, **kw)
    if data.ndim != 3:
        raise ValueError(f"nyquist_ghost expects 3D or 4D, got {data.ndim}D")
    # PE (ky) axis in-plane; sample per volume when not pinned
    pe = int(rng.integers(0, 2)) if axis is None else int(axis)
    work = data if pe == 0 else np.swapaxes(data, 0, 1)
    ny, nx, nz = work.shape
    # alternating per-ky-line phase error: phi(ky) = (-1)^ky * delta + phi0.
    # A sign that flips every other ky line is a modulation by cos/sin terms with
    # period-2 in k-space, which shifts a fraction of the signal by exactly FOV/2 in
    # image space -> the N/2 ghost. The phased reconstruction is genuinely complex
    # (the ghost is in quadrature with the main image), so the final MAGNITUDE image
    # carries both; np.real would discard the ghost. We work in natural (unshifted)
    # FFT order so the (-1)^ky alternation maps cleanly to the FOV/2 image shift.
    delta = (0.15 + 1.2 * sev) * float(rng.uniform(0.7, 1.0))   # radians, per-line swing
    phi0 = float(rng.uniform(0.0, 0.5 * np.pi))                 # constant offset
    alt = np.where((np.arange(ny) % 2) == 0, 1.0, -1.0).astype(np.float32)
    phase = np.exp(1j * (alt * delta + phi0)).astype(np.complex64)  # (ny,)
    out = np.empty_like(work)
    for z in range(nz):
        k = np.fft.fft2(work[:, :, z])
        k = k * phase[:, None]
        out[:, :, z] = np.abs(np.fft.ifft2(k))   # magnitude image carries the ghost
    out = out if pe == 0 else np.swapaxes(out, 0, 1)
    out = np.maximum(out, 0.0)
    return np.ascontiguousarray(np.clip(out, 0.0, 1.0)).astype(np.float32)
