"""Safe checkpoint initialization, atomic publication, and full-state continuation."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch


class CheckpointManager:
    """Checkpoint IO for one run; training supplies its current signature and scaler."""

    def __init__(self, settings, *, signature, scaler, selection_metric):
        self.settings = settings
        self.signature = signature
        self.scaler = scaler
        self.selection_metric = selection_metric

    def save_train_state(self, model, opt, sched, epoch, best_score, best_val, best_epoch, history,
                          ema_model=None, best_model_state=None) -> None:
        """Atomically persist the state needed to continue after the last completed epoch.

        Online augmentation workers are reconstructed on launch, so continuation is semantically faithful
        (model/optimizer/scheduler/EMA/best selection), but is not promised bit-for-bit sample-order replay.
        """
        if (best_epoch < 0 or not isinstance(best_model_state, dict)
                or not np.isfinite(float(best_score)) or not np.isfinite(float(best_val))):
            raise RuntimeError(
                "cannot save resumable state without a finite selected epoch and its best-model weights"
            )
        expected_state = model.state_dict()
        if set(best_model_state) != set(expected_state) or any(
                not torch.is_tensor(best_model_state[key])
                or tuple(best_model_state[key].shape) != tuple(expected_state[key].shape)
                or best_model_state[key].dtype != expected_state[key].dtype
                for key in expected_state):
            raise RuntimeError(
                "cannot save resumable state with an incompatible selected best-model snapshot"
            )
        np_rng = np.random.get_state()
        state = {
            "format": "train_resume_v1",
            "model_type": self.settings.model_type,
            "config": self.signature,         # checked against the resuming run for compatibility
            "output_paths": {
                "model": str(self.settings.model_out_path.resolve()),
                "results": str(self.settings.results_out_path.resolve()),
                "train_state": str(self.settings.train_state_path.resolve()),
            },
            "epoch": int(epoch),              # last COMPLETED epoch
            "model_state_dict": model.state_dict(),
            "ema_state_dict": (ema_model.state_dict() if ema_model is not None else None),
            "optimizer_state_dict": opt.state_dict(),   # the AdamW moments — the real "full resume"
            "scheduler_state_dict": sched.state_dict(),
            "best_score": float(best_score),
            "best_val": float(best_val),
            "best_epoch": int(best_epoch),
            "best_model_state_dict": best_model_state,
            "history": history,
            "rng": {
                "torch": torch.get_rng_state(),
                "numpy": {"bit_generator": np_rng[0], "keys": np_rng[1].tolist(),
                          "position": int(np_rng[2]), "has_gauss": int(np_rng[3]),
                          "cached_gaussian": float(np_rng[4])},
                "cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
            },
        }
        scaler = self.scaler
        if scaler.is_enabled():
            state["grad_scaler_state_dict"] = scaler.state_dict()
        self.settings.train_state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.settings.train_state_path.with_suffix(self.settings.train_state_path.suffix + ".tmp")
        torch.save(state, tmp)
        tmp.replace(self.settings.train_state_path)  # atomic publish


    def load_train_state(self, model, opt, sched, history, ema_model=None):
        """Restore a compatible ``train_resume_v1`` state, including EMA, scaler and selected weights."""
        # Load on CPU so the embedded selected-best snapshot does not consume a second model's worth of
        # VRAM; load_state_dict moves raw/EMA/optimizer tensors to their destination devices as needed.
        try:
            ckpt = torch.load(self.settings.resume_from, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError(
                "Cannot safely load full training state. For a trusted legacy pickle checkpoint, "
                "start a new run with init_from and allow_unsafe_init=True.") from exc
        if not (isinstance(ckpt, dict) and ckpt.get("format") == "train_resume_v1"):
            raise ValueError(
                f"--resume {self.settings.resume_from} is not a resumable training checkpoint (expected one written "
                "by this trainer, format 'train_resume_v1'). The bare deployment .pt holds only weights and "
                "cannot resume the optimizer — use the sibling '<model>.train_state.pt'.")
        prev, cur = ckpt.get("config", {}), self.signature
        if not isinstance(prev, dict):
            raise ValueError("Cannot resume: checkpoint config is missing or malformed.")
        # Full-state continuation is deliberately strict: every data, optimization, augmentation, and
        # selection setting (including --epochs) must match. Only the execution device may move.
        compared = sorted((set(prev) | set(cur)) - {"device"})
        mism = {k: (prev.get(k), cur.get(k)) for k in compared if prev.get(k) != cur.get(k)}
        if mism:
            raise ValueError(
                f"Cannot resume: training/data settings differ from the checkpoint {mism}. "
                "Re-run with matching values, or use --init-from for a new weights-only run.")
        expected_paths = {
            "model": str(self.settings.model_out_path.resolve()),
            "results": str(self.settings.results_out_path.resolve()),
            "train_state": str(self.settings.train_state_path.resolve()),
        }
        if ckpt.get("output_paths") != expected_paths:
            raise ValueError(
                "Cannot resume: checkpoint output paths do not match this invocation; continue the "
                "original run in place, or use --init-from for a new output location."
            )
        scaler = self.scaler
        saved_scaler = ckpt.get("grad_scaler_state_dict")
        if scaler.is_enabled():
            if not isinstance(saved_scaler, dict) or not saved_scaler:
                raise ValueError("Cannot resume FP16 training: checkpoint has no gradient-scaler state. "
                                 "Use --init-from for a new weights-only run.")
            scaler.load_state_dict(saved_scaler)
        elif saved_scaler:
            raise ValueError("Cannot resume scaled FP16 state with a different AMP dtype. "
                             "Use --init-from for a new weights-only run.")
        model.load_state_dict(ckpt["model_state_dict"])
        saved_ema = ckpt.get("ema_state_dict")
        if self.settings.ema:
            if ema_model is None or saved_ema is None:
                raise ValueError(
                    "Cannot resume EMA training: checkpoint has no EMA shadow state. Restart this rung "
                    "fresh; silently resetting EMA would invalidate the ablation."
                )
            ema_model.load_state_dict(saved_ema)  # includes AveragedModel.n_averaged
        elif saved_ema is not None:
            raise ValueError("Cannot resume: a non-EMA run was given a checkpoint containing EMA state.")
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        sched.load_state_dict(ckpt["scheduler_state_dict"])
        # RNG restore is BEST-EFFORT (the critical resume state is model+optimizer+scheduler+epoch); a
        # serialized RNG tensor can come back in a form set_rng_state rejects, so never let it abort.
        rng = ckpt.get("rng") or {}
        try:
            ts = rng.get("torch")
            if ts is not None:
                torch.set_rng_state(ts.clone().cpu().to(torch.uint8) if torch.is_tensor(ts) else ts)
            if rng.get("numpy") is not None:
                ns = rng["numpy"]
                if isinstance(ns, dict):
                    ns = (ns["bit_generator"], np.asarray(ns["keys"], dtype=np.uint32),
                          int(ns["position"]), int(ns["has_gauss"]), float(ns["cached_gaussian"]))
                np.random.set_state(ns)
            if rng.get("cuda") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([s.clone().cpu().to(torch.uint8) for s in rng["cuda"]])
        except Exception as e:  # non-fatal — proceed with the current RNG
            print(f"[resume] RNG state not fully restored ({type(e).__name__}: {e}); continuing.", flush=True)
        saved_history = ckpt.get("history", [])
        saved_epoch = int(ckpt["epoch"])
        if not isinstance(saved_history, list):
            raise ValueError("Cannot resume: checkpoint history is not a list.")
        saved_ids = [h.get("epoch") if isinstance(h, dict) else None for h in saved_history]
        expected_ids = list(range(saved_epoch + 1))
        if saved_ids != expected_ids:
            raise ValueError(
                "Cannot resume: checkpoint epoch/history are inconsistent; "
                f"checkpoint epoch={saved_epoch}, history ids={saved_ids[:10]}"
                f"{'...' if len(saved_ids) > 10 else ''}, expected 0..{saved_epoch}."
            )
        start_epoch = saved_epoch + 1
        if start_epoch > self.settings.epochs:
            raise ValueError(
                f"Cannot resume: checkpoint already completed {start_epoch} epochs, greater than the "
                f"requested --epochs {self.settings.epochs}. Start a fresh run (use --init-from to fine-tune)."
            )
        best_epoch = int(ckpt.get("best_epoch", -1))
        if not 0 <= best_epoch <= saved_epoch:
            raise ValueError(
                f"Cannot resume: best_epoch={best_epoch} is inconsistent with saved epoch={saved_epoch}."
            )
        try:
            best_score = float(ckpt["best_score"])
            best_val = float(ckpt["best_val"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError("Cannot resume: best-score metadata is missing or invalid.") from e
        if not np.isfinite(best_score) or not np.isfinite(best_val):
            raise ValueError("Cannot resume: best-score metadata is non-finite.")
        best_model_state = ckpt.get("best_model_state_dict")
        if not isinstance(best_model_state, dict) or not best_model_state:
            raise ValueError(
                "Cannot resume: checkpoint does not own the selected best-model weights. This is a "
                "legacy/incomplete state; restart the rung rather than evaluating a possibly stale .pt."
            )
        expected_state = model.state_dict()
        if set(best_model_state) != set(expected_state) or any(
                not torch.is_tensor(best_model_state[k])
                or tuple(best_model_state[k].shape) != tuple(expected_state[k].shape)
                or best_model_state[k].dtype != expected_state[k].dtype
                for k in expected_state):
            raise ValueError("Cannot resume: selected best-model snapshot is incompatible with the model.")
        # Roll back a deployment .pt that may have advanced one epoch farther if a prior process
        # crashed after publishing a new best but before committing that epoch's full train state.
        self.publish_deploy_state(best_model_state)
        history.extend(saved_history)
        print(
            f"[resume] {self.settings.resume_from.name}: continuing at epoch {start_epoch} "
            f"(best so far: epoch {ckpt.get('best_epoch')}, "
            f"{self.selection_metric}={float(ckpt.get('best_score', 0.0)):+.4f}).",
            flush=True,
        )
        return start_epoch, best_score, best_val, best_epoch, best_model_state


    def load_init_weights(self, model) -> None:
        """Weights-only WARM-START from any checkpoint (a bare deployment `.pt`, or a `train_state.pt`):
        load the model weights, then train with a FRESH optimizer + LR schedule from epoch 0. This is
        the most you can do when continuing a pre-existing bare checkpoint — it has no saved optimizer
        state, so a true full resume is impossible; the AdamW moments simply restart (a few epochs to
        rebuild momentum). Use `resume_from` instead when a `train_state.pt` exists."""
        try:
            obj = torch.load(self.settings.init_from, map_location=self.settings.device, weights_only=True)
        except Exception as exc:
            if not self.settings.allow_unsafe_init:
                raise ValueError(
                    "Initialization checkpoint cannot be loaded safely. Prefer a weights-only or "
                    "current bundle checkpoint. Only for a trusted legacy pickle file, explicitly "
                    "set allow_unsafe_init=true or pass --allow-unsafe-init.") from exc
            warnings.warn("Loading trusted legacy initialization with unrestricted pickle; "
                          "the checkpoint can execute code", UserWarning, stacklevel=2)
            obj = torch.load(self.settings.init_from, map_location=self.settings.device, weights_only=False)
        sd = None
        if isinstance(obj, dict):
            for k in ("model_state_dict", "state_dict", "model", "weights"):  # wrapped forms
                if isinstance(obj.get(k), dict):
                    sd = obj[k]
                    break
            if sd is None and any(torch.is_tensor(v) for v in obj.values()):  # bare state_dict
                sd = obj
        if sd is None:
            raise ValueError(
                f"--init-from {self.settings.init_from}: no model state-dict found (expected a bare "
                "torch.save(model.state_dict()) or a checkpoint with 'model_state_dict'/'state_dict').")
        sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
        if self.settings.init_partial:
            # Load matching tensors; retain fresh initialization for changed layers.
            msd = model.state_dict()
            loadable = {k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}
            expanded = []
            for k, v in sd.items():
                # MASK-OUTPUT expansion (legacy 1-channel masker -> mask + SDT auxiliary). Preserve
                # the trained segmentation row EXACTLY in row 0 and retain the freshly initialized
                # SDT row 1. This applies to the main output conv and every deep-supervision head,
                # including their 1D biases. Without it --init-partial would discard all mask heads
                # and throw away the most valuable part of a 96%-Dice warm-start.
                is_mask_head = (
                    "output_block" in k or "deep_supervision_heads" in k
                )
                if (is_mask_head and k in msd and msd[k].shape != v.shape
                        and v.dim() >= 1 and msd[k].dim() == v.dim()
                        and v.shape[0] == 1 and msd[k].shape[0] == 2
                        and msd[k].shape[1:] == v.shape[1:]):
                    w = msd[k].detach().clone()
                    w[:1] = v.to(dtype=w.dtype, device=w.device)
                    loadable[k] = w
                    expanded.append(k)
            reinit = [k for k in msd if k not in loadable]
            model.load_state_dict(loadable, strict=False)
            print(
                f"[init-from PARTIAL] warm-started {len(loadable)}/{len(msd)} tensors from "
                f"{self.settings.init_from.name} (output-expanded tensors: {expanded or 'none'}); "
                f"RE-INITIALIZED {len(reinit)} fresh: "
                f"{reinit[:6]}{' ...' if len(reinit) > 6 else ''}. Optimizer + LR start FRESH.",
                flush=True,
            )
            return
        try:
            model.load_state_dict(sd, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(
                f"--init-from {self.settings.init_from}: weights do not match the model architecture "
                f"({self.settings.model_type}, model_kwargs={self.settings.model_kwargs}). Warm-start needs the SAME "
                "architecture, or pass --init-partial to load only the matching tensors. Underlying "
                "error:\n" + str(exc)) from exc
        print(
            f"[init-from] warm-started WEIGHTS from {self.settings.init_from.name}; optimizer + LR schedule start "
            "FRESH from epoch 0 (no optimizer momentum is carried - that needs a train_state.pt resume).",
            flush=True,
        )


    def deploy_checkpoint(self, state_dict) -> Dict[str, Any]:
        """Wrap a state_dict as a self-describing deploy_v1 checkpoint: the FULL preprocessing config plus
        a hash over the preprocessing-CRITICAL subset. A reader recomputes the subset hash over its
        resolved sidecar and WARNS on mismatch — catching a wrong/edited/foreign sidecar that would run
        the wrong spatial/normalization path with a shape-identical head (no load error otherwise). Loads
        cleanly under weights_only=True (primitives + tensors); readers recover the tensors via
        _extract_state_dict('state_dict'). Bare legacy checkpoints (no wrapper) still load, hash-unverified."""
        from imaging.metadata import preproc_config_hash
        cfg = self.signature
        return {"format": "deploy_v1", "state_dict": dict(state_dict),
                "preproc_hash": preproc_config_hash(cfg), "preproc_config": cfg}


    def publish_deploy_state(self, state_dict) -> None:
        """Atomically publish selected deployment weights owned by the full training state."""
        self.settings.model_out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.settings.model_out_path.with_suffix(self.settings.model_out_path.suffix + ".tmp")
        torch.save(self.deploy_checkpoint(state_dict), tmp)
        tmp.replace(self.settings.model_out_path)


