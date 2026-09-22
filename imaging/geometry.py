"""Validated NIfTI spatial geometry shared by training and inference."""

import warnings

import numpy as np


def nifti_affine_mm(img) -> np.ndarray:
    """Return a validated working affine in mm without changing the source image/header.

    Trailing singleton image dimensions are permitted; callers loading such images must
    remove those dimensions before resampling. Unknown units retain the legacy mm
    interpretation with an explicit warning.
    """
    shape = tuple(img.shape)
    if len(shape) < 3 or any(int(n) <= 0 for n in shape) or any(n != 1 for n in shape[3:]):
        raise ValueError(f"Expected a 3D NIfTI (optional trailing singleton dimensions), got {shape}")
    affine = np.array(img.affine, dtype=np.float64, copy=True)
    if (affine.shape != (4, 4) or not np.isfinite(affine).all()
            or not np.allclose(affine[3], [0, 0, 0, 1], rtol=0, atol=1e-8)):
        raise ValueError("NIfTI affine must be a finite homogeneous 4x4 matrix")
    sign, logdet = np.linalg.slogdet(affine[:3, :3])
    if sign == 0 or not np.isfinite(logdet):
        raise ValueError("NIfTI affine must have a nonsingular spatial transform")
    unit = img.header.get_xyzt_units()[0]
    scales = {"mm": 1.0, "meter": 1000.0, "micron": 0.001}
    if unit == "unknown":
        warnings.warn("NIfTI spatial units are unknown; assuming millimetres", UserWarning, stacklevel=2)
        scale = 1.0
    elif unit not in scales:
        raise ValueError(f"Unsupported NIfTI spatial units: {unit!r}")
    else:
        scale = scales[unit]
    affine[:3, :] *= scale
    if not np.isfinite(affine).all():
        raise ValueError("NIfTI affine is not finite after conversion to millimetres")
    return affine


def assert_scan_mask_grid(scan_affine, scan_shape, mask_affine, mask_shape, *, context,
                          check_shape=True, require_full_affine=False, zoom_atol=1e-2):
    """Check that paired scan and mask arrays share shape, orientation, and spacing.

    Conform resampling additionally requires matching origins and obliquity.
    Array-index operations permit translated but otherwise matching affines."""
    from nibabel import aff2axcodes
    from nibabel.affines import voxel_sizes
    sa = np.asarray(scan_affine, dtype=np.float64)
    ma = np.asarray(mask_affine, dtype=np.float64)
    if check_shape and tuple(scan_shape) != tuple(mask_shape):
        raise ValueError(
            f"[{context}] scan/mask SHAPE mismatch {tuple(scan_shape)} vs {tuple(mask_shape)} â€” they are "
            "indexed together (vol[mask]) before resampling. Check --mask-suffix (a wrong/absent suffix "
            "makes discover_pairs fall back to an unrelated *brainmask*.nii.gz on a different grid).")
    if aff2axcodes(sa) != aff2axcodes(ma):
        raise ValueError(
            f"[{context}] scan/mask ORIENTATION mismatch {aff2axcodes(sa)} vs {aff2axcodes(ma)} â€” the mask "
            "is on a transposed/flipped grid, so vol[mask] (and independent resize) misaligns it silently. "
            "Reorient the mask to the scan's frame, or fix --mask-suffix.")
    if not np.allclose(voxel_sizes(sa), voxel_sizes(ma), rtol=0, atol=zoom_atol):
        raise ValueError(
            f"[{context}] scan/mask SPACING mismatch {tuple(np.round(voxel_sizes(sa), 4))} vs "
            f"{tuple(np.round(voxel_sizes(ma), 4))} mm (atol {zoom_atol}).")
    if require_full_affine and not np.allclose(sa, ma, rtol=0, atol=1e-3):
        raise ValueError(
            f"[{context}] scan/mask AFFINE mismatch (max|d|={float(np.abs(sa - ma).max()):.4g} mm) â€” "
            "conform mode resamples through the affine, so origin/obliquity must match too.")
