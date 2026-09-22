"""Label-driven synthesis for masker training.

The realism-vs-large-anatomy contradiction (you can't give a REAL scan big region-size changes and
keep its real texture — it smears) is dissolved by working on a LABEL MAP, not the image:

  1. make_label_map  : segment a T1 + brain mask into a whole-head label map (CSF/GM/WM/ventricle +
                       skull/scalp + background) using only k-means + morphology (no external tool).
  2. deform_labels   : LARGE-SCALE anatomy on the LABELS (warp = region sizes; morphology = ventricle
                       grow, atrophy, cortical thickness, skull thickness). Clean + unbounded, because
                       you move boundaries, not pixels — nothing to smear.
  3. paint_labels    : paint each label with an intensity (a SOMEWHAT-REALISTIC sequence profile most
                       of the time, full-random a fraction for coverage) + per-tissue grain + bias +
                       partial-volume blur + noise. The "improve on SynthStrip" = tissue-matched grain
                       + realistic profiles, not flat fills.

The brain MASK is just the union of the brain-tissue labels, so it co-deforms with the anatomy for
free. n-D: the same code runs on a 2D slice (review montage) and a 3D volume (training).
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy import ndimage as ndi

# label codes
BG, CSF, GM, WM, VENT, SKULL, SCALP, LESION = 0, 1, 2, 3, 4, 5, 6, 7
BRAIN_LABELS = (CSF, GM, WM, VENT)          # deform/connectivity operate on these (LESION added later)
MASK_LABELS = (CSF, GM, WM, VENT, LESION)   # the brain MASK = union of these (a tumour/WMH is intracranial)
_HEAD_LABELS = (SKULL, SCALP)

# somewhat-realistic per-sequence tissue profiles (csf, gm, wm, vent, skull, scalp).
_PROFILES = [
    dict(csf=0.12, gm=0.45, wm=0.72, vent=0.10, skull=0.85, scalp=0.80),   # T1 / MPRAGE
    dict(csf=0.90, gm=0.44, wm=0.24, vent=0.90, skull=0.70, scalp=0.78),   # T2 (WM dark, not washed out)
    dict(csf=0.08, gm=0.62, wm=0.48, vent=0.08, skull=0.50, scalp=0.58),   # FLAIR (CSF nulled)
    dict(csf=0.50, gm=0.70, wm=0.60, vent=0.50, skull=0.80, scalp=0.82),   # PD
    dict(csf=0.15, gm=0.34, wm=0.85, vent=0.12, skull=0.60, scalp=0.62),   # PSIR (WM bright)
]

# PEDIATRIC / infant profiles: the defining feature is INCOMPLETE MYELINATION -> reversed GM-WM contrast
# vs the adult (infant T1: WM darker than GM; infant T2: WM brighter than GM), plus proportionally large
# CSF. Approximate (warped-adult), not faithful neonatal anatomy — but covers the masker's young-brain look.
_PEDIATRIC_PROFILES = [
    dict(csf=0.22, gm=0.58, wm=0.40, vent=0.20, skull=0.82, scalp=0.80),   # infant T1 (WM < GM, reversed)
    dict(csf=0.92, gm=0.40, wm=0.62, vent=0.92, skull=0.65, scalp=0.76),   # infant T2 (WM > GM, reversed)
]


def _kmeans_ordered(vals: np.ndarray, k: int, iters: int = 15) -> np.ndarray:
    """1-D k-means; returns a per-value class in 0..k-1 ORDERED by ascending intensity."""
    vals = np.asarray(vals, np.float32).reshape(-1)
    c = np.quantile(vals, np.linspace(0, 1, k + 2)[1:-1]).astype(np.float32)
    lab = np.zeros(vals.size, np.int64)
    for _ in range(iters):
        lab = np.argmin(np.abs(vals[:, None] - c[None, :]), axis=1)
        nc = np.array([vals[lab == i].mean() if np.any(lab == i) else c[i] for i in range(k)], np.float32)
        if np.allclose(nc, c):
            break
        c = nc
    order = np.argsort(c)
    rank = np.argsort(order)               # class id -> ascending rank
    return rank[lab]


# --------------------------------------------------------------------------- label map
def make_label_map(scan01: np.ndarray, brain: np.ndarray) -> np.ndarray:
    """Whole-head label map from a T1 + brain mask (k-means tissue + morphology; no external tool)."""
    scan01 = np.asarray(scan01, np.float32)
    b = np.asarray(brain) > 0.5
    lab = np.zeros(scan01.shape, np.int32)
    if b.sum() < 50:
        return lab
    # pre-smooth so the tissue split keys on structure, not voxel noise (cleaner CSF/GM/WM regions).
    cl = _kmeans_ordered(ndi.gaussian_filter(scan01, 0.7)[b], 3)   # 0=csf,1=gm,2=wm
    flat = np.zeros(scan01.shape, np.int32); flat[b] = cl + 1
    lab[flat == 1] = CSF; lab[flat == 2] = GM; lab[flat == 3] = WM
    # median-clean the INTERIOR tissue labels (kill salt-and-pepper misclassification) without
    # disturbing the brain boundary (only reassign eroded-interior voxels).
    inner = ndi.binary_erosion(b, iterations=1)
    if inner.any():
        med = ndi.median_filter(lab, size=3)
        lab[inner] = med[inner]
    # ventricles: sizable, CENTRAL CSF connected components
    csf = lab == CSF
    cc, n = ndi.label(csf)
    if n:
        com = np.array(ndi.center_of_mass(b), np.float32)
        scale = np.sqrt(b.sum()) / 2.0 + 1e-6
        for i in range(1, n + 1):
            comp = cc == i
            if comp.sum() < max(15, 0.0008 * b.sum()):
                continue
            d = np.linalg.norm(np.array(ndi.center_of_mass(comp), np.float32) - com) / scale
            if d < 0.55:
                lab[comp] = VENT
    # non-brain head foreground: skull (inner band) + scalp (outer)
    head = (scan01 > 0.06) & (~b)
    lab[head] = SCALP
    lab[ndi.binary_dilation(b, iterations=6) & head] = SKULL
    return lab


def mask_of(lab: np.ndarray) -> np.ndarray:
    """Brain mask = union of the intracranial labels (brain tissue + any LESION); co-deforms for free.
    A tumour/WMH (LESION) and a resection cavity (CSF) are intracranial, so they stay INSIDE the mask."""
    return np.isin(lab, MASK_LABELS)


# --------------------------------------------------------------------------- large-scale deformation (on labels)


def deform_labels(lab: np.ndarray, rng: np.random.Generator, strength: float = 1.0) -> np.ndarray:
    """LARGE-SCALE anatomy on the LABEL map (nearest warp + label morphology -> crisp, unbounded)."""
    shape = lab.shape
    out = lab.copy()
    sc = float(min(shape)) / 128.0
    # 0) named anatomical modes (scan_morph morph3d): big CLEAN shape changes — width/height/taper/
    #    bend/twist/asymmetry/ventricle/atrophy/regional-lobe-bulge — connectivity-guarded so the
    #    brain is never split. On the LABEL map (repainted after), so there is nothing to smear.
    if rng.random() < 0.85:
        from augmentations.anatomy import deform_anatomy
        out = deform_anatomy(out, rng, strength=float(rng.uniform(0.6, 1.3)),   # superset-wide anatomy
                             brain_labels=BRAIN_LABELS, fragment_label=SCALP)
    # NOTE: the old whole-frame smooth-warp (region sizes) is GONE — it warped the SKULL too, which
    # both looks unreal and breaks the real-skull texture transfer in paint_labels. morph3d above does
    # the same region-size/shape changes but CONFINED to the brain (skull/scalp/bg stay put).
    # 2) ventricle dilation (grow VENT into WM)
    if rng.random() < 0.7:
        it = int(rng.uniform(0, 7) * strength)
        if it and (out == VENT).any():
            out[ndi.binary_dilation(out == VENT, iterations=it) & (out == WM)] = VENT
    # 3) atrophy (erode brain, widen CSF/sulci)
    if rng.random() < 0.5:
        it = int(rng.uniform(0, 5) * strength)
        if it:
            brain = np.isin(out, BRAIN_LABELS)
            freed = brain & ~ndi.binary_erosion(brain, iterations=it)
            out[freed & (out != VENT)] = CSF
    # 4) cortical thickness (GM<->WM boundary)
    if rng.random() < 0.5:
        it = int(rng.uniform(1, 4))
        if rng.random() < 0.5:
            out[ndi.binary_dilation(out == GM, iterations=it) & (out == WM)] = GM
        else:
            out[ndi.binary_dilation(out == WM, iterations=it) & (out == GM)] = WM
    # (skull/scalp thickness morphology removed — it moved the skull labels off the real skull, so the
    #  real-texture transfer in paint_labels could no longer fill them; the skull stays as segmented.)
    # 6) partial FOV (cropped acquisition): zero a slab off one face -> BG (mask shrinks with it)
    if rng.random() < 0.3:
        ax = int(rng.integers(len(shape))); n = shape[ax]
        w = int(rng.uniform(0.05, 0.22) * n)
        if w > 0:
            sl = [slice(None)] * len(shape)
            sl[ax] = slice(0, w) if rng.random() < 0.5 else slice(n - w, n)
            out[tuple(sl)] = BG
    return out


# --------------------------------------------------------------------------- painting (labels -> image)
def _bias_field(shape, rng, max_strength=0.6):
    sig = float(rng.uniform(0.12, 0.3)) * float(min(shape))
    f = ndi.gaussian_filter(rng.standard_normal(shape).astype(np.float32), sig)
    f = f / (f.std() or 1.0) * float(rng.uniform(0.0, max_strength))
    return np.exp(np.clip(f, -1.2, 1.2))


def _profile(rng: np.random.Generator, realistic: bool, force: Dict = None) -> Dict[int, float]:
    if force is not None:                                     # a specific profile (e.g. pediatric)
        j = lambda v: float(np.clip(v + rng.uniform(-0.06, 0.06), 0.0, 1.0))
        p = {CSF: j(force["csf"]), GM: j(force["gm"]), WM: j(force["wm"]), VENT: j(force["vent"]),
             SKULL: j(force["skull"]), SCALP: j(force["scalp"]), BG: float(rng.uniform(0.0, 0.12))}
    elif realistic and rng.random() < 0.7:                    # somewhat-realistic sequence look
        base = _PROFILES[int(rng.integers(len(_PROFILES)))]
        j = lambda v: float(np.clip(v + rng.uniform(-0.10, 0.10), 0.0, 1.0))
        p = {CSF: j(base["csf"]), GM: j(base["gm"]), WM: j(base["wm"]), VENT: j(base["vent"]),
             SKULL: j(base["skull"]), SCALP: j(base["scalp"]), BG: float(rng.uniform(0.0, 0.15))}
    else:                                                     # full-random (coverage / superset)
        p = {L: float(rng.uniform(0, 1)) for L in (CSF, GM, WM, VENT, SKULL, SCALP)}
        p[BG] = float(rng.uniform(0.0, 0.4))
    p[LESION] = float(rng.uniform(0.55, 0.95))                # tumour / WMH: hyperintense (variable)
    return p


def paint_labels(lab: np.ndarray, rng: np.random.Generator, realistic: bool = True,
                 real_scan: np.ndarray = None, p_real_skull: float = 0.75,
                 force_profile: Dict = None) -> np.ndarray:
    """Paint each label -> intensity (per-tissue mean + matched grain) -> bias -> partial-volume blur
    -> noise. The improvement over flat SynthStrip fills: tissue-matched grain + realistic profiles.

    REAL SKULL: when `real_scan` is given (and with prob `p_real_skull`), the SKULL/SCALP labels keep
    the scan's OWN texture (real bone structure -> looks REAL) but each gets an INDEPENDENT random
    intensity remap (contrast scale + brightness offset + occasional polarity invert). That is the
    crucial detail: copying the skull verbatim would re-couple skull brightness to "is-not-brain" and
    bring back the over-inclusion failure (and the NFBS benchmark can't see that regression). The remap
    decorrelates the skull's BRIGHTNESS from the brain while preserving its TEXTURE, so we keep the
    realistic look AND the SynthStrip mechanism. The minority with a fully randomized (textureless)
    skull keeps superset coverage of unusual bone (fat-sat, bright-marrow, etc.)."""
    shape = lab.shape
    out = np.zeros(shape, np.float32)
    p = _profile(rng, realistic, force=force_profile)
    real_skull = real_scan is not None and rng.random() < float(p_real_skull)
    rs = np.asarray(real_scan, np.float32) if real_skull else None
    for L in np.unique(lab):
        m = lab == L
        if real_skull and int(L) in (SKULL, SCALP):                # REAL bone texture, RANDOMIZED brightness
            v = rs[m].astype(np.float32)
            if rng.random() < 0.25:                                # contrast-invert (texture kept, polarity flipped)
                v = float(v.max()) + float(v.min()) - v
            out[m] = float(rng.uniform(0.5, 1.7)) * v + float(rng.uniform(-0.25, 0.5))
            continue
        mean = p.get(int(L), float(rng.uniform(0, 1)))
        sd = float(rng.uniform(0.08, 0.18)) if int(L) == LESION else float(rng.uniform(0.01, 0.06))  # tumour heterogeneous
        out[m] = rng.normal(mean, sd, int(m.sum())).astype(np.float32)
    out = out * _bias_field(shape, rng)
    lm = lab == LESION                                             # cap LESION below a flat-white plateau
    if lm.any():                                                   #   (bias could push the brightest tissue to 1.0)
        bm = np.isin(lab, BRAIN_LABELS)
        cap = float(np.percentile(out[bm], 99)) if bm.any() else 1.0
        out[lm] = np.minimum(out[lm], cap * float(rng.uniform(0.85, 1.05)))
    out = ndi.gaussian_filter(out, float(rng.uniform(0.6, 1.6)))    # partial-volume boundaries
    out = out + rng.normal(0.0, float(rng.uniform(0.0, 0.04)), shape).astype(np.float32)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- pathology / pediatric (label edits)
def _ellipsoid(shape, center, radii, rng, rough: float = 0.35) -> np.ndarray:
    """A rough (noisy-border) ellipsoid boolean blob at `center` with per-axis `radii`."""
    g = np.indices(shape, dtype=np.float32)
    r2 = sum(((g[i] - center[i]) / max(float(radii[i]), 1e-3)) ** 2 for i in range(len(shape)))
    if rough > 0:                                            # roughen the border (not a perfect sphere)
        r2 = r2 + rough * ndi.gaussian_filter(rng.standard_normal(shape).astype(np.float32), 2.0)
    return r2 <= 1.0


def add_pathology(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Synthesize focal PATHOLOGY on the label map so the masker tolerates abnormal brains (a local
    label edit; the mask — incl. LESION + CSF cavities — stays the intracranial region):
      * resection cavity / infarct: a brain region -> CSF (fluid / tissue loss); stays IN the mask.
      * tumour: a hyperintense LESION blob (in the mask; near the rim it bulges the boundary = mass effect).
      * WMH: several small LESION spots in WM."""
    out = labels.copy()
    brain = np.isin(out, BRAIN_LABELS)
    if not brain.any():
        return out
    idx = np.argwhere(brain); shape = out.shape; mn = float(min(shape))
    for _ in range(int(rng.integers(1, 3))):                 # 1-2 lesions
        kind = str(rng.choice(["cavity", "tumor", "wmh", "infarct"]))
        if kind == "wmh":
            wm = np.argwhere(out == WM)
            if len(wm) == 0:
                continue
            for _ in range(int(rng.integers(3, 10))):        # several small spots
                cc = wm[int(rng.integers(len(wm)))].astype(np.float32)
                rr = [float(rng.uniform(0.012, 0.03) * mn)] * len(shape)
                out[_ellipsoid(shape, cc, rr, rng) & (out == WM)] = LESION
            continue
        c = idx[int(rng.integers(len(idx)))].astype(np.float32)
        rr = [float(rng.uniform(0.04, 0.12) * mn) for _ in range(len(shape))]
        blob = _ellipsoid(shape, c, rr, rng)
        if kind in ("cavity", "infarct"):
            out[blob & brain] = CSF                          # fluid / tissue loss -> CSF (IN the mask)
        else:                                                # tumour core -> LESION (hyperintense)
            out[blob & (np.isin(out, BRAIN_LABELS))] = LESION
    return out


def _pediatric_morphology(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Bend adult anatomy toward an INFANT look: enlarged ventricles + proportionally large extra-axial
    CSF + a thinner cortical ribbon. Approximate (not faithful neonatal anatomy), but the masker-relevant
    young-brain morphology (big CSF spaces, small parenchyma)."""
    out = labels.copy()
    brain0 = np.isin(out, BRAIN_LABELS)
    vfrac = float((out == VENT).sum()) / max(1, int(brain0.sum()))
    if (out == VENT).any() and vfrac < 0.10:                 # enlarge ventricles — but NOT if already large
        out[ndi.binary_dilation(out == VENT, iterations=int(rng.integers(1, 4))) & (out == WM)] = VENT
    brain = np.isin(out, BRAIN_LABELS)                       # widen subarachnoid CSF (large extra-axial space)
    freed = brain & ~ndi.binary_erosion(brain, iterations=int(rng.integers(2, 5)))
    out[freed & (out != VENT)] = CSF
    if rng.random() < 0.5:                                   # thin the cortex a touch (immature)
        out[ndi.binary_dilation(out == CSF, iterations=1) & (out == GM)] = CSF
    return out


def synthesize_from_labels(scan01: np.ndarray, brain: np.ndarray, rng: np.random.Generator,
                           strength: float = 1.0, realistic: bool = True,
                           p_pathology: float = 0.22, p_pediatric: float = 0.12) -> Tuple[np.ndarray, np.ndarray]:
    """Full label-driven synthesis: segment -> deform LABELS (big clean anatomy, skull preserved) ->
    optional PEDIATRIC morphology/profile + optional PATHOLOGY -> paint (real skull kept). Returns
    (scan01 in [0,1], co-deformed mask). `scan01` is passed as `real_scan` so skull/scalp keep real texture."""
    lab = deform_labels(make_label_map(scan01, brain), rng, strength=strength)
    pediatric = rng.random() < float(p_pediatric)
    if pediatric:
        lab = _pediatric_morphology(lab, rng)
    if rng.random() < float(p_pathology):
        lab = add_pathology(lab, rng)
    force = _PEDIATRIC_PROFILES[int(rng.integers(len(_PEDIATRIC_PROFILES)))] if pediatric else None
    img = paint_labels(lab, rng, realistic=realistic, real_scan=np.asarray(scan01, np.float32),
                       force_profile=force)
    return img, mask_of(lab)
