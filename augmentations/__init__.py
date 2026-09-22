"""MRI augmentations registered for online training and evaluation."""
from .registry import REGISTRY, AugSpec, apply, list_augmentations, register

# Import in dependency order to register each operator once.
from .artifacts import kspace, geometry, intensity, focal, structure, meta, operators, physics, recon
from .protocols import mp2rage
from .artifacts import motion, ringing, metal
from .protocols import t2
from .artifacts import volume

__all__ = ["REGISTRY", "AugSpec", "apply", "list_augmentations", "register"]
