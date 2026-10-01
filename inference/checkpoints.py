"""Safe checkpoint loading and preprocessing metadata discovery for inference."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from anatomical_masking.models.architectures import build_model


class MaskingCheckpoint:
    """Lazily deserialize a checkpoint once, sharing its metadata and model weights.

    CPU loading avoids a second GPU copy while discovering preprocessing. The model
    loader moves the constructed model to its requested device after strict loading.
    """

    def __init__(self, path):
        self.path = Path(path)
        self._loaded = False
        self._payload = None

    def read(self):
        if not self._loaded:
            self._payload = self._read_safe()
            self._loaded = True
        return self._payload

    def _read_safe(self):
        model_path = self.path
        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {model_path}")
        # Never fall back to arbitrary pickle unpickling for deploy checkpoints.
        try:
            obj = torch.load(str(model_path), map_location="cpu", weights_only=True)
        except Exception as exc:  # pickle.UnpicklingError and torch's safe-load refusals
            raise RuntimeError(
                f"Refusing to load {model_path} with weights_only=True (safe load). The checkpoint is not "
                "a plain tensor state-dict — it requires arbitrary unpickling, which is unsafe on a shared "
                "server. Re-save it as a pure state-dict via torch.save(model.state_dict(), ...). Do NOT "
                "disable weights_only to load an untrusted checkpoint."
            ) from exc
        return obj


# ---------------------------------------------------------------------------- checkpoint loading
def _extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    """Pull a flat ``name -> tensor`` state-dict out of whatever ``torch.load`` returned.

    Accept the deployment bundle's wrapped weights and historical bare state-dicts,
    including the common wrappers used by other training harnesses.
    """
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "weights"):
            inner = obj.get(key)
            if isinstance(inner, dict) and any(torch.is_tensor(v) for v in inner.values()):
                return dict(inner)
        if any(torch.is_tensor(v) for v in obj.values()):
            return dict(obj)
    raise ValueError(
        "Could not find a model state-dict in the checkpoint. Expected a state-dict saved by "
        "train.py (torch.save(model.state_dict(), ...)) or a dict containing one under "
        "'state_dict'/'model_state_dict'/'model'."
    )


def _strip_module_prefix(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Drop a leading ``module.`` (DataParallel/DDP) from every key if uniformly present."""
    if sd and all(k.startswith("module.") for k in sd):
        return {k[len("module."):]: v for k, v in sd.items()}
    return sd


def load_masking_model(model_path, device=None, model_kwargs: Optional[Dict[str, Any]] = None,
                       *, checkpoint: Optional[MaskingCheckpoint] = None) -> torch.nn.Module:
    """Build the configured masker and load its checkpoint (eval mode), failing loud on mismatch.

    Reused by `BrainMasker` and direct callers. `device` defaults to cuda-if-available. Explicit
    ``model_kwargs`` are forwarded to ``build_model("masking", ...)``; when they are ``None``, a
    modern deploy-v1 checkpoint's embedded ``preproc_config.model_kwargs`` is used automatically.
    A bare legacy state-dict still falls back to the historical DynUNet defaults.
    """
    model_path = Path(model_path)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = checkpoint or MaskingCheckpoint(model_path)
    obj = checkpoint.read()
    if model_kwargs is None and isinstance(obj, dict):
        embedded = obj.get("preproc_config")
        kw = dict(embedded.get("model_kwargs") or {}) if isinstance(embedded, dict) else {}
    else:
        kw = dict(model_kwargs or {})
    model = build_model("masking", **kw)
    sd = _strip_module_prefix(_extract_state_dict(obj))
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError as exc:
        # Match-or-fail-loud: a mismatch means the built architecture != the trained one, which would
        # yield a meaningless mask. Surface a clear diff instead of silently mis-loading. Compute the
        # diff by hand — strict=False would itself raise on a SHAPE mismatch (the different-`filters`
        # case), which is exactly the situation we want to report cleanly.
        model_sd = model.state_dict()
        missing = [k for k in model_sd if k not in sd]
        unexpected = [k for k in sd if k not in model_sd]
        mismatched = [k for k in sd if k in model_sd and tuple(sd[k].shape) != tuple(model_sd[k].shape)]
        if not missing and not unexpected and not mismatched:
            # Keys+shapes all matched, so this RuntimeError came from the tensor-copy phase, not an
            # architecture mismatch (e.g. CUDA OOM / a device error). Surface it as-is.
            raise
        raise RuntimeError(
            f"State-dict does not match the configured masking model (**{kw!r}): "
            f"{len(missing)} missing, {len(unexpected)} unexpected, {len(mismatched)} shape-mismatched key(s). "
            "If this checkpoint was trained with non-default architecture kwargs (architecture / "
            "filters / deep_supervision / sdt_auxiliary), pass them via model_kwargs (or place the training "
            "results JSON next to the model so they are read automatically).\n"
            f"  missing:    {missing[:3]}\n"
            f"  unexpected: {unexpected[:3]}\n"
            f"  mismatched: {mismatched[:3]}"
        ) from exc
    model.to(dev).eval()
    return model


def _checkpoint_name(path) -> str:
    """Extract a recorded basename even when a sidecar came from another OS."""
    return str(path).replace("\\", "/").rsplit("/", 1)[-1]


def _find_sidecar_config(model_path: Path, *, checkpoint=None) -> Optional[Dict[str, Any]]:
    """Prefer embedded metadata; automatically adopt only compatible legacy sidecars."""
    embedded = _read_embedded_masking_config(model_path, checkpoint=checkpoint)
    if embedded is not None:
        return embedded
    rejected = set()

    def read_compatible(path):
        if path.resolve() in rejected:
            return None
        cfg = _read_masking_config(path)
        recorded = (cfg or {}).get("__model_path__")
        if recorded and os.path.normcase(_checkpoint_name(recorded)) != os.path.normcase(model_path.name):
            rejected.add(path.resolve())
            print(f"[warn] {path.name} records a different model ({_checkpoint_name(recorded)}); "
                  "ignoring it. Pass --config to select it explicitly.", file=sys.stderr, flush=True)
            return None
        return cfg

    candidates = [model_path.with_suffix(".json"),
                  model_path.with_name(model_path.stem + "_results.json"),
                  model_path.with_name(model_path.stem + ".results.json"),
                  model_path.with_name("results.json")]
    for candidate in candidates:
        cfg = read_compatible(candidate)
        if cfg is not None:
            return cfg
    try:
        matches = [cfg for path in sorted(model_path.parent.glob("*.json"))
                   if (cfg := read_compatible(path)) is not None]
    except OSError:
        matches = []
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        identified = [cfg for cfg in matches if cfg.get("__model_path__")]
        if len(identified) == 1:
            return identified[0]
        print(f"[warn] Multiple compatible masking sidecars next to {model_path.name}; "
              "pass --config to select one explicitly.", file=sys.stderr, flush=True)
    return None


def _read_embedded_masking_config(model_path: Path, *, checkpoint=None) -> Optional[Dict[str, Any]]:
    """Safely read and validate a deploy_v1 checkpoint's self-describing masker config."""
    try:
        obj = (checkpoint or MaskingCheckpoint(model_path)).read()
    except Exception:
        # The model loader owns the detailed safe-load error. Config discovery remains
        # best-effort for legacy/bare checkpoints and must not mask that clearer diagnostic.
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("preproc_config"), dict):
        return None
    cfg = dict(obj["preproc_config"])
    if cfg.get("model_type") != "masking":
        return None
    stored_hash = obj.get("preproc_hash")
    if stored_hash:
        from anatomical_masking.imaging.metadata import preproc_config_hash
        actual_hash = preproc_config_hash(cfg)
        if actual_hash != stored_hash:
            raise RuntimeError(
                f"{model_path.name}: embedded preprocessing config hash {actual_hash} does not match "
                f"the checkpoint hash {stored_hash}; refusing a corrupted/internally inconsistent model")
    cfg["__source__"] = f"{model_path}::preproc_config"
    cfg["__model_path__"] = str(model_path)
    return cfg


def _read_masking_config(path: Path) -> Optional[Dict[str, Any]]:
    """Return the `config` block of a results JSON iff it is a masking-model results file."""
    if not path.exists():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if isinstance(doc, dict) and doc.get("model_type") == "masking" and isinstance(doc.get("config"), dict):
        cfg = dict(doc["config"])
        cfg["__source__"] = str(path)
        if doc.get("model_path"):  # the checkpoint this run trained (for identity disambiguation)
            cfg["__model_path__"] = str(doc["model_path"])
        return cfg
    return None


