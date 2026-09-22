"""Train a brain masker from run_config.json. See training/engine.py for the training loop."""
from __future__ import annotations

# Support both direct execution and existing package-style imports using local siblings.
import sys
from pathlib import Path
_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Establish opt-in caps before any numerical/model/data dependency is imported.
import training.runtime as training_runtime
import contextlib
import json
from training.engine import Trainer
from training.config import TrainingConfig


def main(argv=None):
    """Run the masking trainer using the single editable run configuration."""
    import argparse
    from augmentations.config import load_run_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path(__file__).parent / "configuration/run_config.json")
    parser.add_argument("--validate-config", action="store_true",
                        help="Validate and print resolved settings without loading data or training.")
    parser.add_argument("--data-dir", help="Override the input data directory.")
    parser.add_argument("--device", help="Override device, e.g. cpu or cuda:0.")
    parser.add_argument("--resume", type=Path, help="Resume this bundle's full training-state checkpoint.")
    parser.add_argument("--init-from", type=Path, help="Initialize weights from an existing checkpoint.")
    parser.add_argument("--allow-unsafe-init", action="store_true",
                        help="Permit unrestricted pickle only for a trusted legacy --init-from file.")
    args = parser.parse_args(argv)
    try:
        kwargs = load_run_config(args.config)
        if args.data_dir:
            kwargs["data_dir"] = str(Path(args.data_dir).resolve())
        if args.device:
            kwargs["device"] = args.device
        if args.resume:
            kwargs["resume_from"] = str(args.resume.resolve())
        if args.init_from:
            kwargs["init_from"] = str(args.init_from.resolve())
        if args.allow_unsafe_init:
            kwargs["allow_unsafe_init"] = True
        if kwargs.get("resume_from") and kwargs.get("init_from"):
            raise ValueError("Choose either resume_from or init_from, not both")
        if args.validate_config:
            # Constructor validates cross-field requirements without loading data or building a model.
            with contextlib.redirect_stdout(sys.stderr):
                Trainer(TrainingConfig(**kwargs))
            print(json.dumps(kwargs, indent=2))
            return 0
        Trainer(TrainingConfig(**kwargs)).train()
        return 0
    except (ValueError, TypeError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
