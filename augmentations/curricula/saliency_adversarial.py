"""Opt-in saliency-guided adversarial examples for 3-D masking training.

The attack is hard-thresholded projected gradient ascent (PGD).  At every inner
step it differentiates the current masking loss with respect to the input,
smooths the absolute input gradient into a saliency map, keeps only the
brightest ``top_fraction`` of voxels, and takes the locally worst L-infinity
step there.  The support is projected again after every step, so it never grows
beyond that fraction merely because the saliency map moved.

This computes the exact steepest *first-order local* change under the stated
L0/L-infinity constraints; it is not a proof of the globally worst possible MRI
artifact.  Realistic non-differentiable motion/ghosting therefore still belongs
in the ordinary synthesis pipeline.

The module intentionally imports PyTorch lazily.  ``normalize_config`` is used
by the lightweight Colab preflight before dependencies may have been imported.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Mapping
from typing import Any, Callable, Dict, Optional, Tuple


IDENTITY = "masker-saliency-hard-pgd-v1"

DEFAULT_CONFIG: Dict[str, Any] = {
    # Final share of training batches attacked after the curriculum ramp.
    "batch_fraction": 0.25,
    # Epoch indices are zero based: 5 means five clean epochs, then start.
    "start_epoch": 5,
    "ramp_epochs": 20,
    # Final PGD depth.  The active depth also ramps from one to this value.
    "steps": 2,
    # Both are in the z-scored intensity units consumed by the masker.
    "epsilon": 0.15,
    "step_size": 0.075,
    # Exact per-step L0 support (apart from ceil-to-one-voxel rounding).
    "top_fraction": 0.10,
    "smooth_kernel": 7,
    # Selected batches train on this clean/adversarial loss mixture.
    "clean_weight": 0.50,
}


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number, not bool")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer, not bool")
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    try:
        exact = float(value)
    except (TypeError, ValueError):
        exact = float(out)
    if not math.isfinite(exact) or exact != float(out):
        raise ValueError(f"{name} must be an integer")
    return out


def normalize_config(config: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """Validate and canonicalize an enabled attack configuration.

    ``None`` is the only disabled representation.  Supplying a mapping opts in,
    and omitted fields receive conservative defaults.
    """
    if config is None:
        return None
    if not isinstance(config, Mapping):
        raise ValueError("saliency_adversarial must be a mapping or None")
    unknown = sorted(set(config) - set(DEFAULT_CONFIG))
    if unknown:
        raise ValueError(f"unknown saliency_adversarial setting(s): {unknown}")

    out = dict(DEFAULT_CONFIG)
    out.update(dict(config))
    for name in ("batch_fraction", "epsilon", "step_size", "top_fraction", "clean_weight"):
        out[name] = _finite_number(out[name], f"saliency_adversarial[{name!r}]")
    for name in ("start_epoch", "ramp_epochs", "steps", "smooth_kernel"):
        out[name] = _integer(out[name], f"saliency_adversarial[{name!r}]")

    if not 0.0 < out["batch_fraction"] <= 1.0:
        raise ValueError("saliency_adversarial['batch_fraction'] must be in (0, 1]")
    if out["start_epoch"] < 0:
        raise ValueError("saliency_adversarial['start_epoch'] must be >= 0")
    if out["ramp_epochs"] < 1:
        raise ValueError("saliency_adversarial['ramp_epochs'] must be >= 1")
    if not 1 <= out["steps"] <= 8:
        raise ValueError("saliency_adversarial['steps'] must be in [1, 8]")
    if not 0.0 < out["epsilon"] <= 2.0:
        raise ValueError("saliency_adversarial['epsilon'] must be in (0, 2] z-score units")
    if not 0.0 < out["step_size"] <= out["epsilon"]:
        raise ValueError(
            "saliency_adversarial['step_size'] must be in (0, epsilon]")
    if not 0.0 < out["top_fraction"] <= 1.0:
        raise ValueError("saliency_adversarial['top_fraction'] must be in (0, 1]")
    if not 1 <= out["smooth_kernel"] <= 31 or out["smooth_kernel"] % 2 != 1:
        raise ValueError(
            "saliency_adversarial['smooth_kernel'] must be an odd integer in [1, 31]")
    if not 0.0 <= out["clean_weight"] < 1.0:
        raise ValueError("saliency_adversarial['clean_weight'] must be in [0, 1)")
    return out


def curriculum_state(config: Mapping[str, Any], epoch: int) -> Dict[str, Any]:
    """Return deterministic effective strength/frequency for one zero-based epoch."""
    cfg = normalize_config(config)
    assert cfg is not None
    epoch = _integer(epoch, "epoch")
    if epoch < cfg["start_epoch"]:
        progress = 0.0
        active_steps = 0
    else:
        progress = min(1.0, (epoch - cfg["start_epoch"] + 1) / cfg["ramp_epochs"])
        active_steps = max(1, int(math.ceil(cfg["steps"] * progress)))
    return {
        "progress": progress,
        "batch_fraction": cfg["batch_fraction"] * progress,
        "epsilon": cfg["epsilon"] * progress,
        "step_size": cfg["step_size"] * progress,
        "steps": active_steps,
        "top_fraction": cfg["top_fraction"],
        "smooth_kernel": cfg["smooth_kernel"],
        "clean_weight": cfg["clean_weight"],
    }


def _splitmix64(value: int) -> int:
    mask = (1 << 64) - 1
    z = (value + 0x9E3779B97F4A7C15) & mask
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
    return (z ^ (z >> 31)) & mask


def should_attack(config: Mapping[str, Any], epoch: int, batch_index: int,
                  n_batches: int, seed: int) -> bool:
    """Stateless deterministic batch gate that does not consume training RNG state."""
    state = curriculum_state(config, epoch)
    fraction = float(state["batch_fraction"])
    if fraction <= 0.0 or n_batches <= 0:
        return False
    if fraction >= 1.0:
        return True
    if not 0 <= int(batch_index) < int(n_batches):
        raise ValueError("batch_index must be in [0, n_batches)")
    mask = (1 << 64) - 1
    key = (
        (int(seed) & mask)
        ^ (((int(epoch) + 1) * 0xD1B54A32D192ED03) & mask)
        ^ (((int(batch_index) + 1) * 0x94D049BB133111EB) & mask)
        ^ ((int(n_batches) * 0x9E3779B97F4A7C15) & mask)
    )
    # Use the high 53 bits, exactly representable as a Python float in [0, 1).
    unit = (_splitmix64(key) >> 11) / float(1 << 53)
    return unit < fraction


def _to_fp32(value: Any) -> Any:
    import torch

    if torch.is_tensor(value):
        return value.float()
    if isinstance(value, dict):
        return {key: _to_fp32(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_fp32(item) for item in value)
    if isinstance(value, list):
        return [_to_fp32(item) for item in value]
    return value


def _smooth_3d(value: Any, kernel: int) -> Any:
    if kernel == 1:
        return value
    import torch.nn.functional as F

    return F.avg_pool3d(value, kernel_size=kernel, stride=1, padding=kernel // 2)


def _top_fraction_mask(saliency: Any, fraction: float) -> Any:
    """Binary [B,1,D,H,W] mask containing each sample's brightest voxels."""
    import torch

    if saliency.ndim != 5 or saliency.shape[1] != 1:
        raise ValueError("saliency must have shape [B, 1, D, H, W]")
    flat = saliency.reshape(saliency.shape[0], -1)
    keep = max(1, min(flat.shape[1], int(math.ceil(float(fraction) * flat.shape[1]))))
    indices = torch.topk(flat, keep, dim=1, largest=True, sorted=False).indices
    support = torch.zeros_like(flat, dtype=torch.bool)
    support.scatter_(1, indices, True)
    return support.reshape_as(saliency)


def generate_adversarial(
    model: Any,
    scan: Any,
    target: Any,
    criterion: Callable[[Any, Any], Any],
    *,
    forward_fn: Callable[[Any, Any], Any],
    autocast_factory: Optional[Callable[[], Any]],
    config: Mapping[str, Any],
    epoch: int,
    return_saliency: bool = False,
) -> Tuple[Any, Dict[str, Any]]:
    """Build one bounded, model-dependent hard example without touching parameter grads.

    The model is temporarily put in evaluation mode so dropout/running-state changes do not make
    the attack itself stochastic or mutate accumulated training state.  Its original mode is
    restored before returning.  The original image is returned if the final candidate does not
    raise the deterministic attack-mode loss, so a failed inner optimization cannot make a selected
    batch easier.
    """
    import torch

    cfg = normalize_config(config)
    assert cfg is not None
    state = curriculum_state(cfg, epoch)
    if scan.ndim != 5:
        raise ValueError("saliency adversarial input must have shape [B, C, D, H, W]")
    if state["steps"] == 0 or state["epsilon"] <= 0.0:
        zero = scan.new_zeros(())
        return scan.detach(), {
            "accepted": zero,
            "base_loss": zero,
            "attack_loss": zero,
            "loss_increase": zero,
            "support_fraction": zero,
            "max_abs_delta": zero,
            "steps": 0,
        }

    autocast_factory = autocast_factory or contextlib.nullcontext
    original = scan.detach()
    delta = torch.zeros_like(original)
    last_saliency = torch.zeros(
        (scan.shape[0], 1, *scan.shape[2:]), device=scan.device, dtype=scan.dtype)
    last_support = torch.zeros_like(last_saliency, dtype=torch.bool)
    base_loss = None
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.enable_grad():
            for _ in range(int(state["steps"])):
                candidate_input = (original + delta).detach().requires_grad_(True)
                with autocast_factory():
                    prediction = forward_fn(model, candidate_input)
                prediction = _to_fp32(prediction)
                attack_loss = criterion(prediction, target)
                if attack_loss.ndim:
                    attack_loss = attack_loss.mean()
                if base_loss is None:
                    base_loss = attack_loss.detach()
                gradient = torch.autograd.grad(
                    attack_loss, candidate_input, only_inputs=True, retain_graph=False,
                    create_graph=False)[0].detach()
                gradient = torch.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)
                last_saliency = _smooth_3d(
                    gradient.abs().mean(dim=1, keepdim=True), int(state["smooth_kernel"]))
                last_saliency = torch.nan_to_num(
                    last_saliency, nan=0.0, posinf=0.0, neginf=0.0)
                last_support = _top_fraction_mask(last_saliency, float(state["top_fraction"]))
                direction = _smooth_3d(gradient, int(state["smooth_kernel"])).sign()
                proposed = delta + float(state["step_size"]) * direction
                proposed = proposed.clamp(-float(state["epsilon"]), float(state["epsilon"]))
                # Hard projection onto the current brightest-voxel support.  Recomputing this mask
                # each iteration is what lets the attack follow a moving saliency hotspot without
                # accumulating an ever-growing union of modified voxels.
                delta = (proposed * last_support.expand_as(proposed)).detach()

        adversarial = (original + delta).detach()
        with torch.no_grad():
            with autocast_factory():
                final_prediction = forward_fn(model, adversarial)
            final_prediction = _to_fp32(final_prediction)
            final_loss = criterion(final_prediction, target)
            if final_loss.ndim:
                final_loss = final_loss.mean()
        assert base_loss is not None
        accepted = torch.isfinite(final_loss) & (final_loss >= base_loss)
        adversarial = torch.where(accepted, adversarial, original).detach()
        effective_delta = adversarial - original
        effective_support = effective_delta.abs().amax(dim=1, keepdim=True) > 0
        info: Dict[str, Any] = {
            "accepted": accepted.to(dtype=scan.dtype).detach(),
            "base_loss": base_loss.detach(),
            "attack_loss": torch.where(accepted, final_loss, base_loss).detach(),
            "loss_increase": torch.where(
                accepted, (final_loss - base_loss).clamp_min(0.0),
                torch.zeros_like(base_loss)).detach(),
            "support_fraction": effective_support.float().mean().detach(),
            "max_abs_delta": effective_delta.abs().amax().detach(),
            "steps": int(state["steps"]),
        }
        if return_saliency:
            info["saliency_map"] = last_saliency.detach()
            info["support_mask"] = effective_support.detach()
        return adversarial, info
    finally:
        model.train(was_training)


__all__ = [
    "DEFAULT_CONFIG",
    "IDENTITY",
    "curriculum_state",
    "generate_adversarial",
    "normalize_config",
    "should_attack",
]
