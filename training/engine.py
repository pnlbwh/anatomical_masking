"""Masking training engine: setup, epochs, evaluation, and checkpoint selection.

Use ``Trainer(TrainingConfig(...))`` to construct a validated training run.
Dataset preparation and checkpoint IO have their own modules.
"""
from __future__ import annotations

# Must be first: this module establishes thread caps before numerical imports.
import training.runtime as training_runtime
import contextlib
import copy
import json
import sys
import time
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import nibabel as nib
import torch
from models.architectures import build_model, postprocess_mask
from models.losses import build_loss
from training.config import TrainingConfig
from training.checkpoints import CheckpointManager
from training.provenance import training_code_fingerprint
import training.data as training_data
def _jsonable(v) -> Optional[float]:
    """Round for the results JSON; map nan/inf -> None (json.dumps emits invalid NaN/Infinity otherwise)."""
    if v is None:
        return None
    v = float(v)
    return round(v, 6) if (v == v and abs(v) != float("inf")) else None


class Trainer:
    """Train a masker with validated settings and explicit data/checkpoint boundaries."""

    def __init__(self, config: TrainingConfig):
        if not isinstance(config, TrainingConfig):
            raise TypeError("Trainer requires a TrainingConfig")
        # The run owns mutable settings; never mutate the caller's configuration.
        self.__dict__.update(vars(copy.deepcopy(config)))
        self._val_aug_loaders: Dict[str, Any] = {}
        self._last_saliency_adversarial_stats: Dict[str, Any] = {}
        self._configure_amp()

    def train(self) -> Dict[str, Any]:
        if self.resume_from is not None and not self.resume_from.exists():
            raise FileNotFoundError(f"--resume checkpoint not found: {self.resume_from}")
        # Fail fast BEFORE a long training run if an output would be overwritten (shared-server safety).
        # Resuming a run INTENDS to advance its own outputs, so it implies overwrite for them.
        if not self.overwrite and self.resume_from is None:
            existing = [str(p) for p in (self.model_out_path, self.results_out_path, self.train_state_path)
                        if Path(p).exists()]
            if existing:
                raise FileExistsError(
                    "Refusing to overwrite existing output(s) (pass overwrite=True to allow): "
                    + ", ".join(existing)
                )
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # Startup heartbeat (flushed) so a long quiet stretch is diagnosable, not mistaken for a hang.
        t0 = time.time()
        train_dl, val_dl, test_dl = self._dataloaders()
        n_test = len(test_dl.dataset) if test_dl is not None else 0
        print(
            f"[setup +{time.time() - t0:4.1f}s] {len(train_dl.dataset)} train / {len(val_dl.dataset)} val / "
            f"{n_test} test samples; building '{self.model_type}' model on {self.device} "
            f"(num_workers={self.num_workers}) — first build imports the backbone (MONAI for masking) "
            f"+ inits CUDA ...",
            flush=True,
        )
        if self.amp:
            print(f"[amp] forward dtype={self._amp_dtype}; "
                  f"gradient scaling={'enabled' if self._grad_scaler.is_enabled() else 'disabled'}",
                  flush=True)
        model = build_model("masking", **self.model_kwargs).to(self.device)
        criterion = build_loss(self.model_type, **self.loss_kwargs)
        opt = self._optimizer(model)
        sched = self._scheduler(opt)
        # OPT-IN weight EMA: a shadow model averaged after every optimizer step; evaluated + checkpointed
        # in place of the raw weights (InstanceNorm/GroupNorm are stateless, so no BN-stat recalibration).
        ema_model = None
        if self.ema:
            _ema_decay = self.ema_decay
            ema_model = torch.optim.swa_utils.AveragedModel(
                model, avg_fn=lambda avg_p, p, _n: _ema_decay * avg_p + (1.0 - _ema_decay) * p)

        history: List[Dict[str, float]] = []
        best_score = float("-inf")   # selection score (higher == better); see _selection_score
        best_val = float("inf")      # val_loss AT the selected epoch (reported for continuity)
        best_epoch = -1
        best_model_state = None       # CPU snapshot owned by train_state; avoids stale external best .pt
        start_epoch = 0
        # ---- RESUME: warm-start model + FULL optimizer/scheduler state from a prior run, continue.
        if self.resume_from is not None:
            start_epoch, best_score, best_val, best_epoch, best_model_state = self._checkpoints().load_train_state(
                model, opt, sched, history, ema_model=ema_model)
        # ---- INIT-FROM: load WEIGHTS only from any checkpoint; fresh optimizer/schedule from epoch 0.
        elif self.init_from is not None:
            self._checkpoints().load_init_weights(model)
        print(
            f"[setup +{time.time() - t0:4.1f}s] model ready; starting epoch {start_epoch} — the first epoch "
            f"loads every volume from disk (slowest epoch; with num_workers=0 it is serial and silent until "
            f"done).",
            flush=True,
        )
        if start_epoch >= self.epochs:
            print(
                f"[resume] checkpoint already completed {start_epoch} epoch(s) >= --epochs {self.epochs}; "
                "no epochs will be repeated; final test/results publication will run now.",
                flush=True,
            )
        self.model_out_path.parent.mkdir(parents=True, exist_ok=True)
        self.results_out_path.parent.mkdir(parents=True, exist_ok=True)

        def _save_results(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
            res = {
                "model_type": self.model_type,
                "config": self._config(),
                # Live results are checkpoint/progress telemetry, not proof that the requested run
                # completed. Only the post-loop/test atomic write overrides this to True.
                "training_complete": False,
                "best_epoch": best_epoch,
                "best_val_loss": round(best_val, 6) if best_epoch >= 0 else None,
                "selection_metric": self._selection_metric_name(),
                "best_selection_score": round(best_score, 6) if best_epoch >= 0 else None,
                "epochs_done": len(history),
                "epochs_total": self.epochs,
                "history": history,
                "model_path": str(self.model_out_path),
                "train_state_path": str(self.train_state_path),  # pass to --resume to continue this run
            }
            res.update(extra or {})
            # Atomic write (temp + replace) so a live reader never catches a half-written file.
            tmp = self.results_out_path.with_suffix(self.results_out_path.suffix + ".tmp")
            tmp.write_text(json.dumps(res, indent=2), encoding="utf-8")
            tmp.replace(self.results_out_path)
            return res

        for epoch in range(start_epoch, self.epochs):
            te = time.time()
            # Capture the LR BEFORE stepping so `row["lr"]` is the rate this epoch ACTUALLY ran at
            # (sched.step() below advances it to the NEXT epoch's rate).
            epoch_lr = opt.param_groups[0]["lr"]
            tr_loss = self._run_epoch(
                model, train_dl, criterion, opt, ema_model=ema_model, epoch=epoch)
            # Evaluate + checkpoint the EMA shadow when enabled, else the raw model.
            eval_model = ema_model.module if ema_model is not None else model
            val_loss, val_metrics = self._evaluate(eval_model, val_dl, criterion)
            # Diagnostic breakdown, appended AFTER the selection-relevant metrics are computed so it
            # cannot influence them (`_selection_score` reads only `dice`/`deployed_dice`).
            val_metrics.update(self._val_aug_metrics(eval_model, criterion, epoch))
            sched.step()

            row = {
                "epoch": epoch,
                "train_loss": round(tr_loss, 6),
                "val_loss": round(val_loss, 6),
                # Masking Dice and any enabled per-category metrics.
                **{f"val_{k}": _jsonable(v) for k, v in val_metrics.items()},
                "lr": epoch_lr,
                "secs": round(time.time() - te, 1),
            }
            if self.saliency_adversarial is not None:
                row.update(self._last_saliency_adversarial_stats)
            history.append(row)
            # Select using the configured masking objective.
            score = self._selection_score(val_loss, val_metrics)
            if not all(np.isfinite(float(x)) for x in (tr_loss, val_loss, score)):
                raise FloatingPointError(
                    f"non-finite epoch result at epoch {epoch}: train_loss={tr_loss}, "
                    f"val_loss={val_loss}, selection_score={score}. The epoch was not committed; "
                    "fix numerical instability and resume from the prior train_state checkpoint."
                )
            if score > best_score:
                best_score, best_val, best_epoch = score, val_loss, epoch
                # Keep an immutable CPU snapshot inside the full train-state checkpoint. The external
                # deployment .pt alone is not sufficient provenance: a crash can occur after replacing
                # it but before the epoch's optimizer/history state is committed.
                best_model_state = {
                    k: (v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v))
                    for k, v in eval_model.state_dict().items()
                }
                self._checkpoints().publish_deploy_state(best_model_state)
            _save_results()  # live: results JSON reflects progress after EVERY epoch
            # Persist FULL training state each epoch so a crash/preemption can `--resume` from here.
            self._checkpoints().save_train_state(model, opt, sched, epoch, best_score, best_val, best_epoch, history,
                                   ema_model=ema_model, best_model_state=best_model_state)
            print(
                f"epoch {epoch:3d}/{self.epochs - 1}  train {tr_loss:.4f}  val {val_loss:.4f}  "
                f"{self._fmt_metrics(val_metrics)}  ({row['secs']}s)"
                + ("  *best*" if best_epoch == epoch else ""),
                flush=True,
            )

        # A normal run (including a valid full-state resume) must carry one contiguous history row for
        # every requested epoch. Reject a malformed/foreign resume state or any accidental short loop
        # before final test/evaluation can make a partial checkpoint look complete.
        history_epochs = [h.get("epoch") for h in history]
        expected_epochs = list(range(self.epochs))
        if history_epochs != expected_epochs:
            raise RuntimeError(
                "training ended without a complete contiguous history: "
                f"got epochs={history_epochs[:10]}{'...' if len(history_epochs) > 10 else ''} "
                f"(n={len(history_epochs)}), expected 0..{self.epochs - 1} (n={self.epochs}). "
                "The results JSON remains training_complete=false and must not be evaluated."
            )

        # One-shot TEST eval of the BEST checkpoint — never used for selection, so it stays unbiased.
        test_loss, test_metrics = None, None
        if test_dl is not None and best_epoch >= 0 and self.model_out_path.exists():
            _obj = torch.load(self.model_out_path, map_location=self.device, weights_only=True)
            # deploy_v1 checkpoints wrap the state_dict; unwrap inline (avoid importing a reader -> the
            # deployment checkpoints already carry their preprocessing metadata).
            _sd = _obj["state_dict"] if isinstance(_obj, dict) and "state_dict" in _obj else _obj
            model.load_state_dict(_sd)
            test_loss, test_metrics = self._evaluate(model, test_dl, criterion)
            print(
                f"\n[test] best-model (epoch {best_epoch}) on held-out test: loss {test_loss:.4f}  "
                f"{self._fmt_metrics(test_metrics)}  (n={len(test_dl.dataset)}, "
                f"source={'--test-dir' if self.test_dir else 'carved subjects'})",
                flush=True,
            )

        extra = {"training_complete": True,
                 "completed_at_unix": time.time(),
                 "test_n": (len(test_dl.dataset) if test_dl is not None else 0),
                 "test_loss": _jsonable(test_loss)}
        if test_metrics:
            extra.update({f"test_{k}": _jsonable(v) for k, v in test_metrics.items()})
        results = _save_results(extra)
        print(
            f"\nBest epoch {best_epoch} (val_loss {best_val:.4f}; selected on "
            f"{self._selection_metric_name()}={best_score:+.4f}). Model -> {self.model_out_path}\n"
            f"Resume this run with: --resume {self.train_state_path}",
            flush=True,
        )
        return results


    def _optimizer(self, model):
        """AdamW decays learned matrices, while bias and normalization vectors stay free."""
        # AdamW weight decay should NOT hit norm/bias params: GroupNorm/InstanceNorm affine weights and
        # every bias are 1D (ndim<=1); decaying them fights the norm's own scale/shift and shrinks
        # biases for no benefit. Split into a decayed group (ndim>=2: conv/linear weight matrices) and
        # a decay-free group (ndim<=1: biases + norm affine).
        decay, no_decay = [], []
        for prm in model.parameters():
            if not prm.requires_grad:
                continue
            (decay if prm.ndim >= 2 else no_decay).append(prm)
        return torch.optim.AdamW(
            [{"params": decay, "weight_decay": self.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=self.lr,
        )

    def _scheduler(self, opt):
        """Optional linear warmup followed by the same cosine schedule."""
        if self.warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=0.1, total_iters=self.warmup_epochs)
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, self.epochs - self.warmup_epochs))
            return torch.optim.lr_scheduler.SequentialLR(
                opt, [warmup, cosine], milestones=[self.warmup_epochs])
        else:
            return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)

    def _run_epoch(self, model, dl, criterion, opt, ema_model=None,
                       epoch: int = 0) -> float:
        """One training epoch with gradient accumulation and optional masker hard examples.

        The disabled path is the historical single-forward update.  When saliency adversarial
        training is enabled, a deterministic subset of masking batches first runs hard-thresholded
        PGD against the current model.  Clean and adversarial training forwards are then backpropagated
        sequentially, rather than retaining both 3-D activation graphs at once, to keep peak VRAM
        close to the ordinary path.  Validation and test loaders never enter this method.

        AMP keeps losses in fp32 and scales FP16 gradients.
        """
        model.train()
        scaler = self._get_grad_scaler()
        # Keep epoch totals and attack telemetry on-device, synchronizing once at epoch end.
        total, n = torch.zeros((), device=self.device), 0
        accum = self.accum_steps
        n_batches = len(dl)

        saliency_state = None
        saliency_generate = None
        saliency_gate = None
        saliency_batches = 0
        # accepted, attack-mode loss increase, support fraction, max |delta|
        saliency_telemetry = torch.zeros(4, device=self.device, dtype=torch.float32)
        if self.saliency_adversarial is not None:
            from augmentations.curricula.saliency_adversarial import (
                curriculum_state,
                generate_adversarial,
                should_attack,
            )
            saliency_state = curriculum_state(self.saliency_adversarial, epoch)
            saliency_generate = generate_adversarial
            saliency_gate = should_attack

        opt.zero_grad(set_to_none=True)
        window_samples = 0
        for i, (scan, target) in enumerate(dl):
            scan = scan.to(self.device, non_blocking=True)
            target = self._to_device(target)
            batch_samples = int(scan.size(0))
            if batch_samples < 1:
                raise ValueError("Training batches must contain at least one sample")
            window_samples += batch_samples
            use_saliency = bool(
                self.saliency_adversarial is not None
                and saliency_gate(self.saliency_adversarial, epoch, i, n_batches, self.seed))

            if use_saliency:
                # Inner input gradients are requested with autograd.grad(inputs=scan), so they do not
                # write model parameter .grad fields or disturb gradients already being accumulated.
                adversarial_scan, attack_info = saliency_generate(
                    model,
                    scan,
                    target,
                    criterion,
                    forward_fn=self._train_forward,
                    autocast_factory=self._autocast,
                    config=self.saliency_adversarial,
                    epoch=epoch,
                )
                clean_weight = float(self.saliency_adversarial["clean_weight"])

                # Backpropagate the two terms one at a time. Holding both 128^3 graphs concurrently
                # can double activation memory and OOM a 16-GB Colab GPU.
                with self._autocast():
                    clean_pred = self._train_forward(model, scan)
                if self.amp:
                    clean_pred = self._to_fp32(clean_pred)
                clean_loss = criterion(clean_pred, target)
                scaler.scale(clean_weight * clean_loss * batch_samples).backward()
                del clean_pred

                with self._autocast():
                    adversarial_pred = self._train_forward(model, adversarial_scan)
                if self.amp:
                    adversarial_pred = self._to_fp32(adversarial_pred)
                adversarial_loss = criterion(adversarial_pred, target)
                scaler.scale((1.0 - clean_weight) * adversarial_loss * batch_samples).backward()
                del adversarial_pred

                loss = (clean_weight * clean_loss.detach()
                        + (1.0 - clean_weight) * adversarial_loss.detach())
                saliency_telemetry = saliency_telemetry + torch.stack([
                    attack_info["accepted"].float(),
                    attack_info["loss_increase"].float(),
                    attack_info["support_fraction"].float(),
                    attack_info["max_abs_delta"].float(),
                ])
                saliency_batches += 1
            else:
                with self._autocast():
                    pred = self._train_forward(model, scan)
                if self.amp:
                    pred = self._to_fp32(pred)  # keep numerically sensitive loss terms in FP32
                loss = criterion(pred, target)
                scaler.scale(loss * batch_samples).backward()

            total = total + loss.detach() * scan.size(0)
            n += scan.size(0)
            if ((i + 1) % accum == 0) or ((i + 1) == n_batches):
                scaler.unscale_(opt)
                # Losses are sample means; accumulate sums and normalize once by the actual
                # window size, including short tails and unequal DataLoader micro-batches.
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(window_samples)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 12.0)
                old_scale = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                # GradScaler skips optimizer.step on overflow and reduces its scale. Do not advance
                # EMA on that skipped step; a successful step keeps or increases the scale.
                stepped = not scaler.is_enabled() or scaler.get_scale() >= old_scale
                opt.zero_grad(set_to_none=True)
                window_samples = 0
                if ema_model is not None and stepped:
                    ema_model.update_parameters(model)

        if self.saliency_adversarial is not None:
            accepted, loss_gain, support, max_delta = saliency_telemetry.detach().cpu().tolist()
            denom = max(saliency_batches, 1)
            self._last_saliency_adversarial_stats = {
                "saliency_adv_batches": int(saliency_batches),
                "saliency_adv_batch_fraction": round(float(saliency_state["batch_fraction"]), 6),
                "saliency_adv_epsilon": round(float(saliency_state["epsilon"]), 6),
                "saliency_adv_steps": int(saliency_state["steps"]),
                "saliency_adv_acceptance": round(accepted / denom, 6),
                "saliency_adv_attack_loss_increase": round(loss_gain / denom, 6),
                "saliency_adv_support_fraction": round(support / denom, 6),
                "saliency_adv_mean_max_abs_delta": round(max_delta / denom, 6),
            }
        else:
            self._last_saliency_adversarial_stats = {}
        return float((total / max(n, 1)).item())


    def _configure_amp(self):
        """Use native BF16 where available; T4/older CUDA GPUs require scaled FP16.

        The default is_bf16_supported() includes emulation in recent PyTorch. Emulated BF16
        Conv3d can allocate a huge unfolded convolution buffer, even with ample free VRAM.
        """
        self._amp_dtype = torch.bfloat16
        if self.amp and self.device.type == "cuda":
            with torch.cuda.device(self.device):
                try:
                    native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
                except TypeError:  # PyTorch versions before the including_emulation argument
                    native_bf16 = (torch.version.hip is not None or
                                   torch.cuda.get_device_properties(self.device).major >= 8)
            if not native_bf16:
                self._amp_dtype = torch.float16
        use_scaler = bool(self.amp and self.device.type == "cuda" and
                          self._amp_dtype == torch.float16)
        if hasattr(torch.amp, "GradScaler"):
            self._grad_scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        else:  # PyTorch 2.0-2.2
            self._grad_scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)


    def _get_grad_scaler(self):
        # Also supports lightweight callers/tests that construct Trainer without __init__.
        if not hasattr(self, "_grad_scaler"):
            self._configure_amp()
        return self._grad_scaler


    def _autocast(self):
        """Hardware-appropriate mixed precision when AMP is on; losses remain in FP32."""
        if self.amp:
            if not hasattr(self, "_amp_dtype"):
                self._configure_amp()
            return torch.autocast(device_type=self.device.type, dtype=self._amp_dtype)
        return contextlib.nullcontext()


    @staticmethod
    def _to_fp32(pred):
        """Upcast a model output (tensor or {name: tensor} dict) so loss/metrics stay in FP32
        under BF16 or FP16 autocast."""
        if isinstance(pred, dict):
            return {k: (v.float() if torch.is_tensor(v) else v) for k, v in pred.items()}
        return pred.float() if torch.is_tensor(pred) else pred


    def _train_forward(self, model, scan: torch.Tensor):
        """Forward one batch of masking volumes or patches."""
        return model(scan)


    @torch.no_grad()
    def _evaluate(self, model, dl, criterion, *, deployed=None) -> Tuple[float, Dict[str, float]]:
        """Loss + metrics over one loader. `deployed` overrides whether the (slow, scipy-per-volume)
        post-processed Dice is also computed; None == follow `select_on_deployed_dice`. The diagnostic
        per-category loaders pass False — they are a breakdown of the raw Dice, and paying the
        post-processing cost three extra times per epoch is not worth it."""
        model.eval()
        total, n = 0.0, 0
        metric = 0.0
        dep = 0.0
        want_dep = self.select_on_deployed_dice if deployed is None else bool(deployed)
        want_posterior = (
            getattr(getattr(dl, "dataset", None), "eval_category", None)
            == "mp2rage_posterior_fossa")
        posterior_metric = 0.0
        for scan, target in dl:
            scan = scan.to(self.device, non_blocking=True)
            target = self._to_device(target)
            with self._autocast():
                pred = self._eval_forward(model, scan)
            if self.amp:
                pred = self._to_fp32(pred)
            total += criterion(pred, target).item() * scan.size(0)
            metric += self._metric(pred, target) * scan.size(0)
            if want_posterior:
                posterior_metric += self._posterior_fossa_metric(pred, target) * scan.size(0)
            if want_dep:
                dep += self._deployed_dice(pred, target) * scan.size(0)
            n += scan.size(0)
        out = {"dice": metric / max(n, 1)}
        if want_dep:
            out["deployed_dice"] = dep / max(n, 1)
        if want_posterior:
            out["posterior_fossa_dice"] = posterior_metric / max(n, 1)
        return total / max(n, 1), out


    def _val_aug_metrics(self, model, criterion, epoch: int) -> Dict[str, float]:
        """Per-augmentation-category validation Dice, measured on frozen synthesized
        sets over the SAME val subjects. The posterior-fossa category additionally
        reports global Dice while exposing regional posterior/inferior Dice as its
        checkpoint-selection score.

        Answers the question the aggregate Dice cannot: WHICH part of the training mix the model is
        weak on. `benign` and `nonbenign` differ in exactly one thing (the artifact overlay always
        fires for the latter), so the gap between them is the cost of the artifact overlays;
        `mp2rage` isolates the one sequence a benign-only lineage was tuned for, and so doubles as the
        regression guard when artifacts are introduced to the training mix.

        Runs every `masker_val_aug_every` epochs plus the final one. Returns {} on skipped epochs, so
        the history rows simply omit the keys rather than carrying a stale or NaN value.
        """
        if not self._val_aug_loaders:
            return {}
        full = (epoch % self.masker_val_aug_every == 0) or (epoch == self.epochs - 1)
        sel = getattr(self, "masker_select_category", None)
        # The SELECTION category is measured every epoch (its Dice is the checkpoint score, and a
        # score that exists only every Nth epoch would leave the run picking from a handful of
        # candidates). The others stay on the cheap diagnostic cadence.
        if not full and sel is None:
            return {}
        out: Dict[str, float] = {}
        for cat, dl in self._val_aug_loaders.items():
            if not full and cat != sel:
                continue
            _, m = self._evaluate(model, dl, criterion, deployed=False)
            if cat == "mp2rage_posterior_fossa":
                # This key is also the checkpoint-selection metric. Use the regional
                # proxy score so a localized posterior/inferior failure cannot hide under
                # an excellent whole-brain Dice; retain the global diagnostic beside it.
                out[f"dice_{cat}"] = m.get("posterior_fossa_dice", m["dice"])
                out[f"global_dice_{cat}"] = m["dice"]
            else:
                out[f"dice_{cat}"] = m["dice"]
        return out


    def _eval_forward(self, model, scan: torch.Tensor):
        """Predict whole volumes directly or stitch patch logits with Gaussian blending."""
        if self.patch_size is None:
            return model(scan)
        from monai.inferers import sliding_window_inference

        # mode="gaussian": weight each window by a Gaussian so overlap seams are blended smoothly
        # (MONAI's default "constant" hard-averages and can leave grid seams). Matches the gaussian
        # blending predict_mask.py uses at inference, so train-eval and serve stitch identically.
        return sliding_window_inference(
            scan, self.patch_size, self.sw_batch_size, model, overlap=self.sw_overlap,
            mode="gaussian",
        )


    def _selection_score(self, val_loss: float, val_metrics: Dict[str, float]) -> float:
        """Checkpoint-selection score, higher is better.

        Use negative validation loss by default, Dice for composite objectives,
        or the explicitly requested deployed/category Dice.
        """
        sel = getattr(self, "masker_select_category", None)
        if sel is not None:
            v = val_metrics.get(f"dice_{sel}")
            # -inf when the category was not measured this epoch: an unmeasured epoch must never
            # win, and falling back to another metric would compare two different scales.
            return float(v) if v is not None else float("-inf")
        if self.select_on_deployed_dice and val_metrics.get("deployed_dice") is not None:
            return float(val_metrics["deployed_dice"])   # optimize the mask that actually ships
        select_mask_dice = getattr(
            self, "mask_dice_selection", getattr(self, "sdt_supervision", False))
        if select_mask_dice and val_metrics.get("dice") is not None:
            # With an auxiliary/composite task, total val_loss can improve because its extra term
            # fell while the actual binary mask got worse. Preserve the primary objective.
            return float(val_metrics["dice"])
        return -float(val_loss)


    def _selection_metric_name(self) -> str:
        sel = getattr(self, "masker_select_category", None)
        if sel is not None:
            return f"val_dice_{sel}"
        if self.select_on_deployed_dice:
            return "deployed_dice"
        select_mask_dice = getattr(
            self, "mask_dice_selection", getattr(self, "sdt_supervision", False))
        return "val_dice" if select_mask_dice else "neg_val_loss"


    def _metric(self, pred, target) -> float:
        # Masking Dice at the deployed 0.4 threshold.
        target = target["mask"] if isinstance(target, dict) else target
        pred = pred[:, :1]  # channel 1, when present, is linear SDT regression (never a mask logit)
        p = (torch.sigmoid(pred) >= 0.4).float()
        dims = tuple(range(1, p.dim()))
        inter = (p * target).sum(dims)
        dice = (2 * inter + 1e-6) / (p.sum(dims) + target.sum(dims) + 1e-6)
        return float(dice.mean())


    def _posterior_fossa_metric(self, pred, target) -> float:
        """Deployed-mask Dice in a target-derived posterior-inferior RAS proxy box.

        Posterior-fossa validation samples are conformed to canonical RAS before
        synthesis, so low array indices along Y and Z are posterior and inferior.
        This is deliberately named a proxy: whole-brain targets cannot isolate the
        cerebellum. The box extends beyond the target by 6 mm so tentorial and
        inferior extracranial false positives are counted too. Predictions are
        postprocessed exactly as deployment does before the regional crop, so a
        disconnected region removed by component filtering is penalized.
        """

        target = target["mask"] if isinstance(target, dict) else target
        prob = torch.sigmoid(pred[:, :1].float()).detach().cpu()
        if prob.dim() == 4:
            prob = prob.unsqueeze(1)
        p = postprocess_mask(prob, threshold=0.4, dilate_iters=1).float()
        t = (target.detach().cpu() >= 0.5).float()
        if t.dim() == 4:
            t = t.unsqueeze(1)

        conform_mm = float(getattr(self, "conform_mm", 1.0) or 1.0)
        margin = max(1, int(round(6.0 / conform_mm)))
        scores = []
        for pi, ti in zip(p, t):
            # Drop the singleton channel; spatial order is R/L, A/P, S/I.
            pi = pi[0]
            ti = ti[0]
            occupied = ti > 0.5
            # Axis projections recover the bounding box without materializing an
            # N-brain-voxels x 3 coordinate tensor for every 224^3 validation case.
            x_idx = torch.nonzero(occupied.any(dim=2).any(dim=1), as_tuple=False).flatten()
            y_idx = torch.nonzero(occupied.any(dim=2).any(dim=0), as_tuple=False).flatten()
            z_idx = torch.nonzero(occupied.any(dim=1).any(dim=0), as_tuple=False).flatten()
            if x_idx.numel() == 0:
                inter = (pi * ti).sum()
                scores.append((2 * inter + 1e-6) / (pi.sum() + ti.sum() + 1e-6))
                continue

            lo_i = [int(x_idx[0].item()), int(y_idx[0].item()), int(z_idx[0].item())]
            hi_i = [int(x_idx[-1].item()) + 1, int(y_idx[-1].item()) + 1,
                    int(z_idx[-1].item()) + 1]
            extent_i = [hi_i[i] - lo_i[i] for i in range(3)]
            # Cover all left/right tissue, the posterior 50%, and inferior 42%
            # of the subject's brain bounding box. This is a deliberately coarse
            # posterior-fossa proxy, not a cerebellum segmentation. The physical
            # margin captures nearby false positives without using the answer as a gate.
            x0 = max(0, lo_i[0] - margin)
            x1 = min(pi.shape[0], hi_i[0] + margin)
            y0 = max(0, lo_i[1] - margin)
            y1 = min(pi.shape[1], lo_i[1] + int(np.ceil(0.50 * extent_i[1])) + margin)
            z0 = max(0, lo_i[2] - margin)
            z1 = min(pi.shape[2], lo_i[2] + int(np.ceil(0.42 * extent_i[2])) + margin)
            pr = pi[x0:x1, y0:y1, z0:z1]
            tr = ti[x0:x1, y0:y1, z0:z1]
            inter = (pr * tr).sum()
            scores.append((2 * inter + 1e-6) / (pr.sum() + tr.sum() + 1e-6))

        return float(torch.stack(scores).mean()) if scores else float("nan")


    def _deployed_dice(self, pred, target) -> float:
        """Dice of the POST-PROCESSED (deployed) mask vs GT — CC-keep + fill-holes + dilate at the
        deployment defaults (threshold 0.4, 1-voxel dilate) — not raw thresholded logits. Selecting on
        this optimizes the mask that actually SHIPS (the review's point). Slower (scipy per volume); the
        val set is small and it is opt-in (--select-on-deployed-dice)."""
        target = target["mask"] if isinstance(target, dict) else target
        prob = torch.sigmoid(pred[:, :1].float()).detach().cpu()
        if prob.dim() == 4:                              # [B,D,H,W] -> [B,1,D,H,W]
            prob = prob.unsqueeze(1)
        m = postprocess_mask(prob, threshold=0.4, dilate_iters=1).float()   # [B,1,D,H,W] uint8->float
        t = target.detach().cpu().float()
        if t.dim() == 4:
            t = t.unsqueeze(1)
        dims = tuple(range(1, m.dim()))
        inter = (m * t).sum(dims)
        dice = (2 * inter + 1e-6) / (m.sum(dims) + t.sum(dims) + 1e-6)
        return float(dice.mean())


    def _fmt_metrics(self, m: Dict[str, float]) -> str:
        line = f"dice {m.get('dice', float('nan')):.4f}"
        # Per-category breakdown, on the epochs it was measured (`dice_<category>`; note
        # `deployed_dice` deliberately does not match this prefix).
        per_cat = "  ".join(f"{k[len('dice_'):]} {v:.4f}"
                            for k, v in m.items() if k.startswith("dice_"))
        return f"{line}  [{per_cat}]" if per_cat else line


    def _to_device(self, target):
        if isinstance(target, dict):
            return {k: v.to(self.device, non_blocking=True) for k, v in target.items()}
        return target.to(self.device, non_blocking=True)


    def _runtime_versions(self) -> Dict[str, Optional[str]]:
        cached = getattr(self, "_resume_runtime_versions", None)
        if cached is not None:
            return dict(cached)
        def dist(name):
            try:
                return importlib_metadata.version(name)
            except importlib_metadata.PackageNotFoundError:
                return None
        versions = {
            "python": sys.version.split()[0],
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
            "nibabel": str(nib.__version__),
            "cuda_runtime": str(torch.version.cuda),
            "scipy": dist("scipy"),
            "monai": dist("monai"),
            "torchvision": dist("torchvision"),
            "opencv_python": dist("opencv-python"),
            "opencv_python_headless": dist("opencv-python-headless"),
            "scikit_image": dist("scikit-image"),
        }
        self._resume_runtime_versions = versions
        return dict(versions)


    def _config(self) -> Dict[str, Any]:
        cfg = TrainingConfig.resume_settings(self)
        cfg.update(
            resume_signature_version=2,
            training_code_fingerprint=self._training_code_fingerprint(),
            runtime_versions=self._runtime_versions(),
            data_split_fingerprint=getattr(self, "_resume_data_fingerprint", None),
        )
        # AMP precision and prepared-data metadata are observations of this run.
        if self.amp and getattr(self, "_amp_dtype", torch.bfloat16) == torch.float16:
            cfg["amp_dtype"] = "float16"
        if getattr(self, "_train_epoch_accounting", None) is not None:
            cfg["training_epoch_accounting"] = dict(self._train_epoch_accounting)
        if self.conform_mm is None:
            orientation = training_data._data_orientation(self)
            if orientation:
                cfg["canonical_orientation"] = orientation
        return cfg

    def _training_code_fingerprint(self):
        if not getattr(self, '_resume_code_fingerprint', None):
            self._resume_code_fingerprint = training_code_fingerprint(Path(__file__).resolve().parents[1])
        return self._resume_code_fingerprint

    def _dataloaders(self):
        prepared = training_data.prepare_data(self, self._selection_metric_name())
        self._resume_data_fingerprint = prepared.fingerprint
        self._train_epoch_accounting = prepared.epoch_accounting
        self._val_aug_loaders = prepared.validation_categories
        return prepared.train, prepared.validation, prepared.test

    def _checkpoints(self):
        return CheckpointManager(self, signature=self._config(),
                                 scaler=self._get_grad_scaler(),
                                 selection_metric=self._selection_metric_name())
