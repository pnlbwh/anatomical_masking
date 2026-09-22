"""Unified augmentation registry.

Training-time transforms are described by an
:class:`AugSpec` and registered with the :func:`register` decorator. They are
invoked through :func:`apply` and enumerated with :func:`list_augmentations`.

Canonical transform signature
------------------------------
Every registered transform conforms to::

    fn(arr, *, severity: float, rng: np.random.Generator,
       mask=None, **params) -> np.ndarray

``arr`` is a 2D slice OR 3D volume (per the spec's ``dims``); the return has
the same shape. ``rng`` is a NumPy ``Generator``. ``mask`` is an optional
boolean array (brain mask) for transforms that need an in-brain region.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple

import numpy as np

# Allowed metadata vocabularies. ``kind`` has 7 values even though the package
# only ships 6 kind-modules: ``noise`` transforms live inside ``intensity.py``.
KINDS = frozenset(
    {"geometry", "intensity", "noise", "kspace", "focal", "structure", "meta"}
)
DIMS = frozenset({"2d", "3d", "either"})


@dataclass(frozen=True)
class AugSpec:
    """Descriptor for one registered augmentations.

    Attributes
    ----------
    name: registry key (unique).
    fn: the transform callable (canonical signature, see module docstring).
    kind: one of :data:`KINDS`.
    severity_range: ``(lo, hi)`` advisory severity bounds (metadata only;
        :func:`apply` does NOT clamp/rescale).
    dims: one of :data:`DIMS` — whether ``fn`` operates on a 2D slice, a 3D
        volume, or either.
    label_preserving: True if the transform does not move anatomy relative to
        its paired label/mask (intensity/noise/kspace are typically True;
        geometry that warps space is typically False).
    has_detector: True if a matching QC detector exists for this artifact.
    modalities: tuple of applicable modalities (empty = unrestricted).
    extra_parameters: JSON parameters consumed by an adapter through **kwargs.
    excluded_parameters: callable-only or ignored legacy signature parameters.
    """

    name: str
    fn: Callable[..., np.ndarray]
    kind: str
    severity_range: Tuple[float, float] = (0.0, 1.0)
    dims: str = "either"
    label_preserving: bool = True
    has_detector: bool = False
    modalities: Tuple[str, ...] = field(default_factory=tuple)
    extra_parameters: Tuple[str, ...] = ()
    excluded_parameters: Tuple[str, ...] = ()


# The single global registry: name -> AugSpec.
REGISTRY: Dict[str, AugSpec] = {}


def register(
    name: str,
    *,
    kind: str,
    severity_range: Tuple[float, float] = (0.0, 1.0),
    dims: str = "either",
    label_preserving: bool = True,
    has_detector: bool = False,
    modalities: Tuple[str, ...] = (),
    extra_parameters: Tuple[str, ...] = (),
    excluded_parameters: Tuple[str, ...] = (),
):
    """Decorator factory: register ``fn`` under ``name`` and return it unchanged.

    Validates ``kind`` and ``dims`` against the allowed vocabularies and rejects
    duplicate names so typos surface immediately when registering transforms.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}; must be one of {sorted(KINDS)}")
    if dims not in DIMS:
        raise ValueError(f"unknown dims {dims!r}; must be one of {sorted(DIMS)}")
    if name in REGISTRY:
        raise ValueError(f"augmentation {name!r} already registered")

    def _decorator(fn: Callable[..., np.ndarray]) -> Callable[..., np.ndarray]:
        REGISTRY[name] = AugSpec(
            name=name,
            fn=fn,
            kind=kind,
            severity_range=tuple(severity_range),  # type: ignore[arg-type]
            dims=dims,
            label_preserving=label_preserving,
            has_detector=has_detector,
            modalities=tuple(modalities),
            extra_parameters=tuple(extra_parameters),
            excluded_parameters=tuple(excluded_parameters),
        )
        return fn

    return _decorator


def apply(
    name: str,
    vol: np.ndarray,
    severity: float,
    rng=None,
    *,
    mask=None,
    **kw,
) -> np.ndarray:
    """Look up ``name`` and run its transform on ``vol``.

    ``severity`` is passed through verbatim (no clamping/rescaling — the
    ``severity_range`` on the spec is advisory metadata). ``rng`` may be a
    ``Generator``, an int seed, or ``None``; it is normalized with
    ``np.random.default_rng`` (a ``Generator`` passes through unchanged).
    """
    if name not in REGISTRY:
        raise KeyError(
            f"unknown augmentation {name!r}; "
            f"available: {sorted(REGISTRY)}"
        )
    spec = REGISTRY[name]
    generator = np.random.default_rng(rng)
    result = spec.fn(vol, severity=severity, rng=generator, mask=mask, **kw)
    # Contract: apply() returns a single ndarray (the transformed image). A few legacy
    # transforms (create_ring/metal_paint/noise_overlay/random_erasing/signal_drop_band)
    # return a (image, mask/orig, count) tuple — normalize to the image so every caller
    # gets an array regardless of the transform.
    if isinstance(result, (tuple, list)):
        result = result[0]
    return result


def list_augmentations():
    """Return registered augmentation names, sorted."""
    return sorted(REGISTRY)
