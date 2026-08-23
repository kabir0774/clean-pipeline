# Genuine RSNA Knee Pipeline — SSH/H100 Setup

## Purpose

This folder trains and evaluates a genuine MRI model for RSNA Knee Abnormality Detection. It uses the V2 and GPT report labels only as fold-specific soft training supervision; reports are not model inputs at test time.

The pipeline excludes the old notebook’s test-specific calibration payload, test-series regression, fixed public-LB priors, inherited fold heads, copied predictions, and public-LB-tuned blend coefficients.

The official competition metric is macro ROC-AUC across 12 targets. A public score is not a substitute for gold out-of-fold validation.

## Files to upload

Upload the entire local folder `C:\kabir\RSNA_Knee_AI\genuine_pipeline` to the SSH server. It must contain:

```text
build_rsna_fold_safe_labels.py
train_rsna_soft_oof.py
rsna_knee_genuine.py
rsna_backbone_adapters.py
fold_safe_labels/
  gold_outer_folds.csv
  labels_fold_0.csv ... labels_fold_4.csv
  label_builder_manifest.json
RSNA_Fold_Safe_H100_Runbook.md
RSNA_Knee_Static_Code_Audit.md
SSH_H100_SETUP_AND_RUN.md
```

Do not upload the old Kaggle submission notebook, compressed calibration payload, inherited submission CSV, or unverified fold-head bundle into the genuine run directory.

## Required competition data on the SSH server

Keep the competition data in a separate directory, for example:

```text
<DATA_ROOT>/
  train.csv
  test.csv
  train_series.csv
  test_series.csv
  train_series/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm
  test_series/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm
```

Do not modify or redistribute competition data. The final `--root` argument must point to the directory containing `train.csv` and `train_series/`.

## Optional model assets

### DINOv2

Upload or mount the complete local DINOv2 model directory, including its `config.json` and model weights. Set its path in `--dino-source`.

Do not use a directory containing only a partial checkpoint. The loader freezes the DINOv2 backbone except the last four transformer blocks and uses the CLS feature.

### RadImageNet ResNet-50

Upload only a full ResNet-50 encoder checkpoint whose license permits the competition use. You must know its SHA-256 hash. The loader requires:

```text
--rad-checkpoint /path/to/full_radimagenet_resnet50.pt
--rad-sha256 <exact_sha256>
```

The loader uses `weights_only=True`, removes the classifier, verifies ResNet-50 convolutional keys, and rejects ambiguous head-only/fold bundles. Do not pass `v52_*_heads.pt` files as the encoder unless their manifest proves they contain the full compatible encoder.

## Environment setup

Use a fresh environment on the SSH server. Example:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install a PyTorch build matching the server CUDA driver. Check first:

```bash
nvidia-smi
python --version
```

Then install:

```bash
pip install torch torchvision
pip install numpy pandas scikit-learn pydicom
pip install transformers safetensors huggingface_hub
```

Confirm:

```bash
python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda', torch.cuda.is_available())
print('gpus', torch.cuda.device_count())
if torch.cuda.is_available(): print(torch.cuda.get_device_name(0))
PY
```

## First run: build/verify labels

The supplied `fold_safe_labels/` was already generated from the local train CSV, V2 CSV, and GPT CSV. Rebuild it on the server only if the uploaded source files and paths are identical:

```bash
python build_rsna_fold_safe_labels.py \
  --train-csv <DATA_ROOT>/train.csv \
  --v2-csv <LABEL_ROOT>/llm_labels_v2.csv \
  --gpt-csv <LABEL_ROOT>/report_labels_gpt56sol.csv \
  --out <RUN_ROOT>/fold_safe_labels
```

Verify the manifest and confirm it says `gold_rows: 58`, `folds: 5`. Never replace gold labels with weak labels.

## First H100 model run

Run the baseline first:

```bash
python train_rsna_soft_oof.py \
  --root <DATA_ROOT> \
  --labels-dir <RUN_ROOT>/fold_safe_labels \
  --output <RUN_ROOT>/runs/resnet18 \
  --architecture resnet18 \
  --epochs 12 \
  --batch-size 3 \
  --workers 8 \
  --image-size 256 \
  --slices-per-slot 12 \
  --compile
```

Expected outputs:

```text
<RUN_ROOT>/runs/resnet18/
  resnet18_fold_0.pt ... resnet18_fold_4.pt
  resnet18_gold_oof.csv
  resnet18_metrics.json
```

`resnet18_metrics.json` is the first real model result. It must contain macro gold OOF AUC and all 12 per-target AUCs.

## DINOv2 run

After the ResNet baseline completes:

```bash
python train_rsna_soft_oof.py \
  --root <DATA_ROOT> \
  --labels-dir <RUN_ROOT>/fold_safe_labels \
  --output <RUN_ROOT>/runs/dinov2_small \
  --architecture dinov2_small \
  --dino-source <DINO_ROOT> \
  --epochs 12 \
  --batch-size 2 \
  --workers 8 \
  --image-size 256 \
  --slices-per-slot 12 \
  --compile
```

DINOv2 is promoted only if its gold OOF result is better than ResNet on repeated runs and does not cause unexplained damage to individual targets.

## RadImageNet run

First calculate the checkpoint hash:

```bash
sha256sum <RAD_CHECKPOINT>
```

Then run:

```bash
python train_rsna_soft_oof.py \
  --root <DATA_ROOT> \
  --labels-dir <RUN_ROOT>/fold_safe_labels \
  --output <RUN_ROOT>/runs/radimagenet_resnet50 \
  --architecture radimagenet_resnet50 \
  --rad-checkpoint <RAD_CHECKPOINT> \
  --rad-sha256 <SHA256> \
  --epochs 12 \
  --batch-size 1 \
  --workers 8 \
  --image-size 256 \
  --slices-per-slot 12
```

This branch is not automatically accepted into an ensemble. It must first pass the OOF comparison and its checkpoint license/provenance must be recorded.

## What counts as a valid improvement

For each architecture, inspect:

```bash
cat <RUN_ROOT>/runs/<ARCH>/*_metrics.json
```

Promote a model only when:

- macro gold OOF AUC improves by at least 0.002–0.003;
- the improvement appears in at least 3 of 5 folds;
- no target drops by more than 0.005 without a documented reason;
- the result repeats with a second seed;
- the exact model/checkpoint/config is recorded.

Do not select an architecture because the public leaderboard increased once.

## Final ensemble policy

Only after all branches have complete OOF files should an ensemble be created. Blend weights must be fit from nested OOF predictions or use equal weights. There must be no test-label or public-LB optimization. The final package should contain:

```text
model_manifest.json
fold assignments
OOF predictions for every member
per-target AUC table
checkpoint hashes and licenses
exact environment/package versions
final inference script
```

## Current unresolved items

Before the first run, fill in:

- `<DATA_ROOT>`
- `<RUN_ROOT>`
- `<DINO_ROOT>`
- `<RAD_CHECKPOINT>`
- RadImageNet checkpoint SHA-256 and license
- whether the server has one H100 or multiple GPUs

The code is prepared for these values but does not guess them.
