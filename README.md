# Condensed brain-mask workflow

Train a brain-segmentation model, evaluate it against reference masks, and generate binary masks for new MRI scans. The command-line entry points use focused modules for configuration, data preparation, training, and inference.

The top-level Python files are small command-line entry points; implementations live in the folders for their respective concerns. Copy the entire `deployment_condensed` folder when moving to another machine. Supply your dataset separately. For inference, also supply a trained checkpoint.

## Files

```text
deployment_condensed/
  train.py                    Training command
  generate_mask.py            Mask-generation command
  evaluate.py                 Reference-mask evaluation command
  evaluate_augmentations.py   Augmentation benchmark command
  training/
    engine.py                 Training and validation loop
    config.py                 Typed settings and validation
    data.py                   Datasets, crops, splits, and loaders
    checkpoints.py            Saving, resume, and warm starts
    runtime.py                Thread setup and runtime checks
    provenance.py             Source fingerprints
    Train_On_Colab.ipynb       Colab training, backups, and evaluation
  inference/
    masker.py                 In-memory prediction and mask output
    checkpoints.py            Weights and metadata loading
  evaluation/
    standard.py               Evaluation against reference masks
    augmentation.py           Augmentation robustness benchmark
    transforms.py             Reproducible benchmark conditions
  models/
    architectures.py          Masking U-Net definitions
    losses.py                 Masking losses and deep supervision
  imaging/
    geometry.py               NIfTI coordinates and scan/mask grid checks
    normalization.py          Shared image normalization
    metadata.py               Preprocessing fingerprints
  configuration/
    run_config.json           Training experiment settings
    run_config.augmented.json Benchmark experiment settings
    presets.py                Preset loading and named overrides
    augmentation_presets/
      standard.json           Shared augmentation inventory
  augmentations/
    pipeline.py               Sample routing, rendering, and postprocessing
    config.py                 Stage defaults and configured switches
    sampling_policy.py        Sample probabilities and curriculum rules
    registry.py               Named operator registration and dispatch
    numerics.py               Shared numerical operations for renderers
    anatomy.py                Paired image/mask morphology
    appearance.py             Realistic intensity and resolution variation
    label_synthesis.py        Label-driven synthetic anatomy
    artifacts/                Noise, motion, metal, ringing, and other defects
    protocols/                MRI acquisition and contrast renderers
    curricula/                Optional MP2RAGE, hard-tail, and adversarial modes
  requirements.txt
  README.md
```

## Working on the code

Start with the module responsible for the behavior you want to change:

- `training/config.py`: training defaults, cross-field validation, and recorded settings. Construct runs with `Trainer(TrainingConfig(...))`; import the two classes from `training.engine` and `training.config`.
- `training/engine.py`: optimization, validation, and checkpoint selection. Dataset preparation belongs in `training/data.py`; checkpoint persistence belongs in `training/checkpoints.py`.
- `augmentations/pipeline.py`: the sequence from a source scan/mask pair to a training sample. `sampling_policy.py` chooses route probabilities; `config.py` owns stage defaults and configured switches.
- `augmentations/artifacts/`: individual corruption operators, including `operators.py` and the shared volume renderers in `volume.py`. MRI protocol simulation belongs in `augmentations/protocols/`; optional training curricula belong in `augmentations/curricula/`.
- `imaging/`: normalization, physical coordinates, grid validation, and preprocessing metadata shared by training and inference.
- `inference/masker.py`: prediction and mask output. `inference/checkpoints.py` interprets checkpoint weights and metadata. Inference has no dependency on training or augmentation modules.

For routine augmentation changes, edit the run JSON or shared preset first. Change a renderer only when its scientific behavior needs to change; paired geometry must transform the reference mask with the scan. The verification commands at the end of this guide check configuration, imports, rendering, and the complete workflow.

Command-line entry points are unchanged. `train.py` is the CLI; Python callers should import from the owning modules (`training`, `models`, or `imaging`). Custom Python imports must use the new module paths: for example, `augmentations.pipeline` replaces `augmentations.synth_masker_dataset`, and shared normalization comes from `imaging.normalization`.

For Python training code:

```python
from training.config import TrainingConfig
from training.engine import Trainer

config = TrainingConfig(
    model_type="masking", data_dir="data",
    model_out_path="runs/mask.pt", results_out_path="runs/mask.json",
)
Trainer(config).train()
```

The MP2RAGE renderer in `augmentations/artifacts/volume.py` separates tissue mapping, extracranial contrast, background noise, foreground blending, and output composition into named stages. Shared numerical operations live in `augmentations/numerics.py`.

For Python evaluation code, use the in-memory API:

```python
from inference.masker import BrainMasker

masker = BrainMasker(model_path="runs/mask.pt", device="cpu")
prediction = masker.predict(scan_path="scan.nii.gz")
mask = prediction.mask  # uint8 mask in the original NIfTI grid
metadata = prediction.record
masker.save_prediction(prediction, output_path="brainmask.nii.gz")
```

`masker.run(scan_path=..., output_path=...)` remains the convenience method that predicts and saves. Evaluation uses in-memory predictions and writes masks only when requested. Checkpoint weights and embedded settings are read once per masker instance.

## Quick start on your computer

The workflow was verified with Python 3.12. Run these commands in `deployment_condensed`, preferably in a virtual environment:

```sh
python -m pip install -r requirements.txt
```

Edit `configuration/run_config.json` for your dataset and output location, then validate it before training:

```sh
python train.py --config configuration/run_config.json --validate-config
python train.py --config configuration/run_config.json
```

Validation checks configuration values and their compatibility. It does not load the dataset or train a model. Training performs dataset checks when it starts.

The supplied configuration runs 100 epochs. The Colab notebook overrides it with 128-cube training crops from the full 256-cube, 1 mm scan, one sliding-window prediction at a time for full-scan validation/test, and two data workers. Crops reduce GPU memory without downsampling the scan, at the cost of less spatial context per training example. `PATCH_SIZE=None` selects whole-volume training for a larger GPU. Use a new run name when changing these settings.

With `amp=true`, training uses BF16 on CUDA GPUs with native BF16 support, FP16 with gradient scaling on other CUDA GPUs, and BF16 on CPU. The selected precision is printed at startup; FP16 scaling state is saved for resume. Losses remain FP32. Model weights remain compatible with the previous architecture.

CUDA is selected when available. To choose a device explicitly:

```sh
python train.py --config configuration/run_config.json --device cuda:0
python train.py --config configuration/run_config.json --device cpu
```

After training, generate a mask or evaluate the saved model:

```sh
python generate_mask.py --model runs/mask.pt --scan /path/to/scan.nii.gz --out /path/to/predicted_mask.nii.gz
python evaluate.py --model runs/mask.pt --data-dir /path/to/heldout_pairs --output-dir reports/evaluation
```

Replace example paths with your own. Quote paths containing spaces. In JSON, forward slashes work on Windows, for example `"D:/MRI/data"`; literal backslashes must be doubled.

## Prepare your data

Training needs raw NIfTI scans and aligned reference masks. The supplied file-matching settings are:

```json
{
  "scan_glob": "*_T1w.nii.gz",
  "mask_suffix": "_brainmask"
}
```

These are fields inside `training`, not a replacement for the full configuration. A matching dataset can look like this:

```text
my_dataset/
  subject_01/
    sub-01_ses-01_T1w.nii.gz
    sub-01_ses-01_T1w_brainmask.nii.gz
  subject_02/
    sub-02_ses-01_T1w.nii.gz
    sub-02_ses-01_T1w_brainmask.nii.gz
```

Discovery searches subject subfolders. Reference masks must share the scan's physical grid. Use binary masks with `0` for background and `1` for brain; the evaluator requires those exact values. For masks ending in `_brainmask_refined_crf`, change `training.mask_suffix` accordingly and pass `--mask-suffix _brainmask_refined_crf` to directory-based evaluation.

The supplied `data_dir` is `../../NFBS_Dataset/NFBS_Dataset`, which selects one copy of the 125-subject dataset in this workspace. Update it when moving the bundle.

### Subject separation

All sessions and acquisitions of a subject stay in the same training, validation, or test split. Subject IDs are derived from BIDS `sub-...` or NFBS `A########` filenames. At least two distinct subjects are required.

For another naming convention, set `training.subject_id_regex`. For example:

```json
"subject_id_regex": "(?P<subject_id>patient[0-9]+)"
```

This groups `patient001_session1_T1w.nii.gz` and `patient001_session2_T1w.nii.gz` together. The regex searches the scan path with forward slashes. The named `subject_id` group takes precedence, then the first capture group, then the full match. Unrecognized identities fail before training. Byte-identical acquisition copies are deduplicated; conflicting copies with the same identity are rejected.

## Configure a training run

`configuration/run_config.json` holds experiment settings and refers to a shared augmentation preset:

| Section | Purpose |
| --- | --- |
| `training` | Data/output paths, model, optimizer, geometry, splits, and runtime settings |
| `sampling` | Override sample probabilities and optional curricula from the preset |
| `augmentation_preset` | Path to the complete shared augmentation inventory |
| `augmentation_overrides` | Optional changes to named augmentations' `enabled` switches and `settings` |

Relative paths inside the JSON resolve from the JSON file's directory. Relative command-line paths resolve from the directory where you run the command.

Both supplied runs use `configuration/augmentation_presets/standard.json` (referenced relative to each run JSON). Each run can override only the settings it changes. Nested settings dictionaries merge; lists replace their preset values. Training outputs record the full resolved augmentation configuration, including disabled entries, so interpreting a checkpoint does not require its original preset file. Existing JSON files containing a complete `augmentations` list remain supported when their inventory matches the current registry. Older full-inventory files must remove these retired entries: `rotate`, `resize`, `warp`, `elastic_warp`, `invert_brain`, `invert_skull`, `brain_blackspace`, `skull_blackspace`, `motion_smear`, `ghost_mirror`, `blur_2d`, and `skull_textures`. Paired training geometry and the benchmark rotation/resize conditions remain available.

Common `training` settings in the supplied configuration:

| Setting | Supplied value | Meaning |
| --- | --- | --- |
| `data_dir` | `../../NFBS_Dataset/NFBS_Dataset` | Root containing scan/reference pairs |
| `model_out_path` | `../runs/mask.pt` | Selected checkpoint for inference |
| `results_out_path` | `../runs/mask.json` | Training results and recorded configuration |
| `epochs` | `100` | Total planned epochs |
| `target_shape` | `[256, 256, 256]` | Full conformed volume dimensions |
| `conform_mm` | `1.0` | Isotropic working voxel size in millimeters |
| `batch_size` | `1` | Samples per loader batch |
| `accum_steps` | `4` | Microbatches per optimizer step; partial windows use their actual sample count |
| `num_workers` | `0` | Data-loader worker processes |
| `lr` | `0.0003` | Learning rate |
| `val_fraction` / `test_fraction` | `0.20` / `0.15` | Approximate fractions of subjects held out |
| `seed` | `0` | Split seed; online training synthesis draws fresh randomness |
| `device` | `null` | Automatic device selection |
| `model_kwargs.architecture` | `resenc` | Residual encoder segmentation model |
| `loss_kwargs.profile` | `recall-boundary` | Training loss profile |
| `ema` / `ema_decay` | `true` / `0.99` | Exponential moving average of model weights |

Choose distinct output paths for each experiment. An augmentation change is a new experiment; the trainer records the complete configuration, including disabled entries.

### Enable, disable, and configure augmentations

The supplied `configuration/run_config.json` enables the full standard augmentation mix, including
all broad benign families and all 36 supported 3D artifact-overlay transforms.
The benign families include:
canonical **MPRAGE, MP2RAGE, T2 (SPACE_T2), and FLAIR**; broad acquisition variation
across all 10 configured sequences, fields, vendors, and reconstruction styles;
realistic appearance, tone mapping, bias fields, mild noise; label-driven synthetic
anatomy (including the configured pathology/pediatric variation); paired morphology,
orientation/resize, and resolution changes. The 22% canonical-protocol branch is
split equally across its four protocols (about 5.5% each); broad acquisition and
synthetic draws provide additional contrast coverage.

Nonbenign artifact overlays are enabled at both the stage and individual-transform
levels, with `p_artifact=0.30` and one or two transforms per selected draw. The pool
includes motion, ghosting, ringing, aliasing, noise, bias, dropout, metal effects,
and the remaining supported artifacts. This gate applies to the broad-benign and
synthetic branches (60% of all draws), so about 18% of all draws reach the artifact
overlay; individual operators still retain their validity/compounding guards.
The optional hard-artifact tail, adversarial perturbations, and dedicated MP2RAGE
stress curricula remain off. MP2RAGE morphology/noise-superset switches match the
standard preset but are only used if a dedicated MP2RAGE route is selected later.
Donor histogram transfer and standalone
2D/unpaired geometry transforms remain unsupported; paired geometry is provided by
the enabled morphology/orientation/resolution stages. Clean and canonical-protocol
anchors retain their geometry. Use a new run name and a weights-only warm start
after changing a run's augmentation configuration.

Add an `augmentation_overrides` object to the run JSON to change specific augmentations. Set `enabled` to `false` to disable one. For example:

```json
{
  "augmentation_overrides": {
    "noise": {
      "enabled": true,
      "settings": {
        "severity_range": [0.1, 0.34],
        "params": {"mode": "rician"}
      }
    }
  }
}
```

Add this field alongside `training`, `sampling`, and `augmentation_preset` in the existing run. Unmentioned augmentations inherit the preset. Ordinary noise overlays also need `artifact_overlay.enabled=true`, a positive `sampling.p_artifact`, and a sample route eligible for overlays. `enabled=true` permits an operation when its route is selected; it does not apply that operation to every sample.

The inventory lists 66 registered primitives and 20 pipeline stages. Entries marked `supported: false` must remain disabled in this 3D training workflow. Some are 2D-only, require a donor, or change geometry without updating the reference mask. Paired morphology/orientation stages handle geometric changes for training.

Stage switches also have dependencies:

- Disabling `standard_protocols` turns that stage's sample share into clean samples.
- Disabling `label_synthesis` starts that branch from the source pair; eligible downstream stages can still run.
- `tone_mapping`, `realistic_bias_field`, and `realistic_noise` run within `realistic_appearance`. The registry `noise` overlay is a separate operation.
- Disabling `artifact_overlay` disables ordinary overlays. An independently enabled hard-artifact-tail curriculum is separate.
- Disabling every augmentation entry makes augmentation an identity operation. Spatial preprocessing and normalization still run.

Invalid parameters, failed operators, nonfinite outputs, and incorrect output shapes raise errors. They do not silently become clean training samples. Optional per-category augmented validation uses a fixed diagnostic distribution independently of the training switches.

### Sampling probabilities

With both dedicated MP2RAGE curriculum settings left at `null`, the supplied mix is:

| Sample type | Probability |
| --- | --- |
| Clean | `p_clean = 0.18` |
| Standard protocol | `p_standard = 0.22` |
| Broad benign variation | `p_benign = 0.30` |
| Label synthesis | Remaining `0.30` |

The first three probabilities must sum to at most `1`. `p_artifact = 0.30` is the separate artifact gate on eligible benign/synthetic samples; it is not a fifth sample category. `configuration/run_config.augmented.json` supplies the same full standard augmentation profile for benchmarking; the training geometry in `configuration/run_config.json` remains 256-cube at 1 mm.

`mixed_mp2rage_fraction` and `benign_only_mp2rage_fraction` select alternative curricula and replace the ordinary top-level mix. They are mutually exclusive. **`null` disables a curriculum; `0` still selects its alternative distribution.** MP2RAGE phenotype fractions apply within the dedicated MP2RAGE share and need their corresponding stages enabled. Run `--validate-config` after changing these settings; it checks exposure limits and geometry requirements.

## Training outputs and restarting

With the supplied output paths, training writes:

```text
runs/
  mask.pt                  Selected deployment weights and embedded model/preprocessing settings
  mask.json                Training history, selection metric, and recorded configuration
  mask.train_state.pt      Full state for continuing the same run
```

Keep these files together. Use `mask.pt` for inference. The training-state file contains the model, optimizer, scheduler, EMA when enabled, selected-best weights, and RNG state.

Continue the same run:

```sh
python train.py --config configuration/run_config.json --resume runs/mask.train_state.pt
```

Full resume requires the original data/output paths, compatible source/runtime identity, and unchanged settings, including the planned total `epochs`. It resumes after the last completed epoch. It is not a guarantee of identical data-loader sample order after a process restart.

This bundle supports masking only; QC models, losses, and training settings have been removed. Existing masking weights remain compatible. Full training states saved before source changes, including removal of unused tools, require a weights-only warm start because the resume policy checks source-file fingerprints. Full states produced by the updated bundle continue to support resume with the same sources and settings.

Start a new experiment from compatible model weights:

```sh
python train.py --config configuration/run_config.json --init-from /path/to/previous_mask.pt
```

First set new output paths in the JSON. A warm start creates fresh optimizer/scheduler state. The checkpoint architecture must match the requested model. Full states from the original deployment or an older source version generally require a warm start instead of full resume.

New runs protect existing weights, results, and full-state outputs. To intentionally replace earlier outputs, add `"overwrite": true` inside `training`. A valid resume advances its original outputs automatically.

Checkpoint loading uses weights-only deserialization. A trusted legacy file requiring pickle can be initialized with `--allow-unsafe-init`; this explicit opt-in may execute code from the checkpoint. Full resume always uses safe loading.

## Evaluate a model

Use scans that were independently held out from training, with aligned binary references:

```sh
python evaluate.py --model runs/mask.pt --data-dir /path/to/heldout_pairs --output-dir reports/evaluation --save-masks
```

Directory discovery uses `*_T1w.nii.gz` and `_brainmask` by default. For an explicit case list, use a JSON list, JSONL, or CSV manifest. Example JSON:

```json
[
  {
    "id": "subject_01",
    "scan": "heldout/sub-01_T1w.nii.gz",
    "mask": "heldout/sub-01_T1w_brainmask.nii.gz"
  }
]
```

Manifest paths resolve relative to the manifest file. Run:

```sh
python evaluate.py --model runs/mask.pt --manifest cases.json --output-dir reports/evaluation --save-masks
```

Outputs are `reports/evaluation/results.json`, `reports/evaluation/cases.csv`, and optional predictions in `reports/evaluation/masks/`. Reports contain per-case and aggregate Dice, IoU, precision, and recall. Invalid cases are listed as failed and excluded from aggregates; the command returns a nonzero exit code if any case fails. The evaluator does not establish whether your supplied cases were excluded from training.

Evaluation runs the same preprocessing and postprocessing as mask generation. Internal trainer metrics retain the original reference-mask normalization protocol; use this standalone evaluator to assess the deployed inference workflow.

## Evaluate augmentation robustness

Section 13 of `training/Train_On_Colab.ipynb` evaluates any selected checkpoint on an explicit
held-out manifest or directory. Set `EVAL_MODEL` and `EVAL_MANIFEST` or
`EVAL_DATA_DIR`; do not point it at the entire training dataset. The default
`MAX_CASES=5` is a reproducible pilot; set it to `None` for the full cohort.
Run validation and test cohorts separately, keeping the test cohort out of tuning.

The four tiers are clean, mild geometry/resolution, mild plus synthetic MRI
protocols, and the full supported augmentation suite. Mild conditions isolate
rotation, anatomical resizing, and resolution degradation, with a combined condition.
Protocol conditions include T2, MP2RAGE, FLAIR and other supported contrasts.
The full tier adds isolated 3D artifacts, paired anatomy/appearance synthesis,
and a combined acquisition/geometry/multi-artifact condition. The full tier tests
coverage across conditions; it does not apply all artifacts simultaneously.
Specialized or unsupported operations are listed with reasons in the report.

The runner uses the deployed `BrainMasker` path for every condition and preserves
its checkpoint preprocessing. Synthetic reference masks never guide inference
normalization. Geometry transforms scan and reference together; contrast and
artifact transforms retain the reference. Synthetic protocol scores are stress
tests, not a substitute for real acquired protocol cohorts.

Reports (`results.json`, `cases.csv`, `summary.csv`) update after each condition;
the notebook writes directly to a new timestamped Drive directory. Mean Dice is
averaged across repeats, then conditions, then cases. Augmented tier averages
exclude clean scans. Failed, pending, or unchanged renders exclude that case from
its condition/tier mean; counts expose incomplete coverage. A separately labeled
available-case mean reports partial results over successful draws. Clean runs once per
case. Fixed seeds allow paired checkpoint comparisons.

The same benchmark is available outside Colab:

```sh
python evaluate_augmentations.py --model runs/mask.pt --manifest heldout.json --output-dir augmentation_evaluation --max-cases 5 --repeats 1
```

Omit `--max-cases` for all supplied cases. `--tiers clean mild` runs a shorter
benchmark; `--augmentations rotation resolution` selects exact named conditions.
Use `--help` for severity ranges and shared inference settings. Nonzero exit
status means at least one failed trial; unchanged trials are reported separately; inspect the saved reports.
The source augmentation configuration remains unchanged by evaluation.

## Generate a mask

```sh
python generate_mask.py --model runs/mask.pt --scan /path/to/scan.nii.gz --out /path/to/brainmask.nii.gz
```

The output is a binary `uint8` NIfTI in the scan's native 3D grid, preserving its spatial form codes and units. Working coordinates are converted to millimeters for mm, meter, and micron inputs. Unknown units produce a warning and are treated as millimeters. Extra trailing singleton dimensions are accepted; multi-volume time series are not.

Common options shared by generation and evaluation:

| Option | Default | Effect |
| --- | --- | --- |
| `--threshold` | `0.60` | Probability cutoff for the binary mask |
| `--component-policy` | `largest` | Choose `largest`, `keep-large` (at least 5% of the largest component), or `keep-all` |
| `--dilate-mm` | Unset | Grow the mask by a physical distance; overrides `--dilate-iters` |
| `--no-tta` | TTA enabled | Disable flip averaging; conform modes already skip it |
| `--refine-normalization` | Off | Enable a second pass using the initial mask for normalization |
| `--device` | Automatic | Select CPU or a CUDA device |
| `--overwrite` | Off | Permit replacing previous outputs |

Use the same operating point for evaluation and final masks. Generation prints its settings, mask size, and review flags. Nonfinite predictions fail before a mask is published. Even with `--overwrite`, output paths cannot alias input scans, model/configuration files, or evaluation references/manifests, including through hardlinks.

Modern checkpoints embed their architecture and preprocessing settings. For an older bare state dictionary, supply its matching training results JSON:

```sh
python generate_mask.py --model /path/to/legacy.pt --config /path/to/training_results.json --scan /path/to/scan.nii.gz --out /path/to/brainmask.nii.gz
```

**The inference `--config` is a training-results sidecar, not `configuration/run_config.json`.** Automatic sidecars that name another checkpoint are ignored.

## Google Colab

Open `training/Train_On_Colab.ipynb` in Colab and follow its cells in order. The notebook uses the bundle's `configuration/run_config.json` and shared preset for training and augmentation settings.

Upload a ZIP of the complete `deployment_condensed` folder and your dataset ZIP or folder to Google Drive.

The staging cell prints the exact selected archive, its hash, and the internal bundle root. It accepts flat and nested ZIP layouts and rejects ambiguous multiple bundles. Optional augmentation-evaluation helpers do not block training; missing files are listed and are required only when running evaluation. If Drive has several similarly named ZIPs, set `CODE_SRC` to the exact one you intend to use. The notebook mounts Drive, stages the code/data onto local runtime storage, validates the run configuration, trains, and backs up resumable state to Drive. Training uses fixed local paths so a restored run can satisfy the trainer's path identity checks.

Use the same run name, configuration, sources, and local paths to continue after a disconnect. A new configuration needs a new run name. Resume remains strict about runtime compatibility: a changed Colab software environment can require restoring compatible versions or starting a new run from weights. Backup and restore do not relax the trainer's validation.

For a new run initialized from previous weights, set `INIT_FROM` in the notebook settings to the previous compatible `mask.pt` and choose a new `RUN_NAME`. The training cell reloads its saved configuration from disk and resumes the new run automatically once its training state exists.

The notebook also includes progress inspection and optional mask-generation/evaluation cells. Check its backup status before ending a session; work after the latest complete backup can be lost if the runtime stops abruptly.

## Troubleshooting

| Message or symptom | What to check |
| --- | --- |
| No scan/mask pairs found | `data_dir`, `scan_glob`, `mask_suffix`, and the directory level containing the dataset |
| Cannot determine subject identity | BIDS/NFBS filenames or `training.subject_id_regex` |
| Conflicting copies or overlapping test subjects | Duplicate acquisitions and subject identities across dataset folders |
| Refusing to overwrite outputs | Choose new output paths, resume the existing run, or explicitly enable overwrite |
| Cannot resume because settings differ | Restore the original configuration/paths/runtime, or warm-start a new run |
| Unknown augmentation or invalid parameters | Check named overrides against the shared inventory and validate the JSON after editing |
| Slow online synthesis or memory pressure | Check CPU load, workers, volume/patch dimensions, batch size, and enabled augmentations; changing run settings requires a new experiment |
| Missing preprocessing metadata for an old checkpoint | Supply the matching training-results JSON with inference `--config` |
| Scan/reference grid mismatch | Align the reference mask to the scan before evaluation |

For the complete command-line interfaces:

```sh
python train.py --help
python evaluate.py --help
python evaluate_augmentations.py --help
python generate_mask.py --help
```

## Verification

The current bundle passed 343 regression tests, including notebook staging and inference import isolation, and a copied-bundle CPU workflow covering one training epoch, safe full-state resume, native mask generation, and evaluation. This cleanup preserved all 66 registered operators and 70 benchmark definitions. Seeded comparisons matched 440 MP2RAGE renders, 44 shared-helper artifact renders, 14 benchmark renders, 16 training samples, and 10 posterior samples, including RNG states where recorded. Default/severity settings, validation errors, and disk resampling outputs also matched the previous implementation. These synthetic checks establish execution behavior, not trained-model accuracy. Full dataset training and CUDA validation were not performed during this refactor.

The workspace's `_condensed_validation` folder contains the regression suite and `smoke_workflow.py`; they are development checks, not runtime dependencies. From the workspace root:

```sh
python -m pytest _condensed_validation -q
python _condensed_validation/smoke_workflow.py
```
