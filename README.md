# Brain-mask workflow

Train MRI brain-segmentation models, generate masks, and evaluate predictions.

## Folders

- `training/`: Training, datasets, checkpoints, and the `Train_On_Colab.ipynb` notebook.
- `inference/`: Load trained models and generate brain masks.
- `evaluation/`: Reference-mask evaluation, augmentation benchmarks, and failed-trial recovery.
- `models/`: U-Net architectures and training losses.
- `imaging/`: Shared NIfTI geometry, normalization, and metadata.
- `configuration/`: Training and benchmark run settings.
- `configuration/augmentation_presets/`: Shared augmentation settings.
- `augmentations/`: Training-sample synthesis, routing, anatomy, and appearance variation.
- `augmentations/artifacts/`: Motion, noise, metal, ringing, and other MRI defects.
- `augmentations/protocols/`: MRI acquisition and contrast simulation.
- `augmentations/curricula/`: Optional MP2RAGE, hard-artifact, and adversarial training modes.

## Setup

Use Python 3.12. Run all subsequent commands from `deployment_condensed`:

```sh
cd deployment_condensed
python -m pip install -r requirements.txt
```

## Train

Edit `configuration/run_config.json`: set `training.data_dir` and output paths.
JSON paths resolve from the configuration file's folder; CLI paths resolve from the working directory.
Defaults pair `*_T1w.nii.gz` scans with `*_T1w_brainmask.nii.gz` masks in subject subfolders.
Use aligned scan/mask pairs and BIDS/NFBS subject IDs, or set `training.subject_id_regex`.

```sh
python train.py --config configuration/run_config.json --validate-config
python train.py --config configuration/run_config.json
# Resume the same run:
python train.py --config configuration/run_config.json --resume runs/mask.train_state.pt
```

The default outputs are `runs/mask.pt` (model), `runs/mask.json` (results), and resumable training state.
Add `--device cpu` or `--device cuda:0` to select a device; otherwise selection is automatic.
For Colab, open `training/Train_On_Colab.ipynb` and follow its cells.

## Generate a mask

```sh
python generate_mask.py --model runs/mask.pt --scan /path/to/scan.nii.gz --out /path/to/brainmask.nii.gz
```

Replace example paths with your own. Output is a binary NIfTI in the input scan's grid.

## Evaluate

```sh
# Independently held-out scan/mask pairs:
python evaluate.py --model runs/mask.pt --data-dir /path/to/heldout_pairs --output-dir reports/evaluation
# Augmentation benchmark on the saved training test split:
python -m evaluation.training_split --model runs/mask.pt --output training_test_manifest.json
python evaluate_augmentations.py --model runs/mask.pt --manifest training_test_manifest.json --output-dir reports/augmentations --repeats 1
```

Add `--help` to any command for all options.
