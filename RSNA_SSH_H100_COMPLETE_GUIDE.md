# RSNA Knee Abnormality Detection
## Genuine Pipeline — Complete SSH/H100 Upload and Run Guide

**Prepared for:** Kabir Verma  
**Pipeline directory:** `C:\kabir\RSNA_Knee_AI\genuine_pipeline`  
**Purpose:** train genuine MRI models and select them using held-out gold out-of-fold macro ROC-AUC.

---

## 1. What this folder does

This folder is a clean replacement for the large Kaggle inference notebook. It combines:

1. Audited training labels from the 58 studies with adjudicated labels.
2. V2 and GPT-5.6-Sol report-derived labels for the remaining studies as soft training supervision.
3. Study-level MRI loading with DICOM-aware slice ordering.
4. Multi-plane and fat-suppression slot construction.
5. Soft-label confidence weighting.
6. Held-out gold OOF evaluation.
7. Multiple independently trainable visual architectures:
   - ResNet18 baseline
   - ConvNeXt-Tiny
   - EfficientNetV2-S
   - DINOv2
   - RadImageNet ResNet-50
8. Checkpoint, fold, metric, and label-manifest outputs.

The folder does **not** promise a 0.95 or 0.96 score. It produces evidence for whether an architecture genuinely improves the competition metric.

The competition metric is macro ROC-AUC across 12 targets. Binary accuracy and threshold selection are not the primary metric.

---

## 2. Loopholes deliberately removed

The old Kaggle notebook contains an inherited test-time ensemble branch, fixed public-LB-oriented coefficients, external heads, and a compressed calibration payload. This clean folder excludes:

- Public-leaderboard-fitted calibration.
- Test-series metadata regression.
- Hidden/test-label derivation.
- Report text as a test-time model input.
- Copied predictions or fixed competitor blend coefficients.
- Globally chosen V4 weak-label blending inside fold validation.
- Unverified head-only checkpoints masquerading as full models.

Public competitor notebooks may inspire preprocessing or pooling ideas, but their predictions, hidden outputs, and unexplained blend constants are not used.

---

## 3. File-by-file inventory

### `build_rsna_fold_safe_labels.py`

Builds five fold-specific weak-label files from:

- `llm_labels_v2.csv`
- `report_labels_gpt56sol.csv`
- `train.csv` gold labels

For each outer fold, it selects each target’s V2/GPT mixture using only the other gold folds. The held-out gold fold is disabled from training. It also enforces balanced fold capacity for 58 gold studies and fails if any target has only one class in a fold.

### `fold_safe_labels/`

Generated label package:

```text
gold_outer_folds.csv
labels_fold_0.csv
labels_fold_1.csv
labels_fold_2.csv
labels_fold_3.csv
labels_fold_4.csv
label_builder_manifest.json
```

Each fold file contains:

- `StudyInstanceUID`
- `is_gold`
- `train_enabled`
- `outer_fold`
- 12 soft/hard target columns
- 12 `__confidence` columns

Gold labels are hard targets with confidence 1.0. Weak labels are soft targets. Confidence is based on `2 * abs(p - 0.5)`.

### `rsna_knee_genuine.py`

Core reusable utilities:

- Dataset table loading.
- Study-level grouped multilabel fold construction.
- DICOM metadata and geometry helpers.
- DICOM slice ordering.
- MRI decoding, normalization, and resizing.
- Plane/sequence slot selection.
- Study dataset class.
- Baseline study MIL model.
- Weighted soft-label BCE.
- Macro and per-target ROC-AUC functions.

### `train_rsna_soft_oof.py`

Main H100 training program. It:

1. Reads one fold-safe label file at a time.
2. Excludes held-out gold studies from training.
3. Trains one architecture per command.
4. Selects the best epoch using the held-out gold fold.
5. Writes a checkpoint for each fold.
6. Concatenates all held-out predictions into one gold OOF file.
7. Writes macro and per-target AUC metrics.

Supported values:

```text
resnet18
convnext_tiny
efficientnet_v2_s
dinov2
dinov2_small
radimagenet_resnet50
```

### `rsna_backbone_adapters.py`

Safe adapters for external backbones:

- `load_dinov2()` loads a complete DINOv2 model and enables only the configured final transformer blocks.
- `load_radimagenet_resnet50()` loads a full ResNet-50 encoder only.
- RadImageNet checkpoint SHA-256 can be required.
- Checkpoints are loaded with `weights_only=True`.
- Head-only and incompatible checkpoints are rejected.

### `RSNA_Knee_Static_Code_Audit.md`

Detailed audit of the original 3,675-line exported Kaggle script, including the inherited RadImageNet branch, hidden calibration payload, fixed blend choices, label handling, and retained safe components.

### `RSNA_Fold_Safe_H100_Runbook.md`

Short technical runbook for the first baseline and architecture comparisons.

### `RSNA_SSH_H100_COMPLETE_GUIDE.md`

This document. It is the complete operator guide.

### `requirements-h100.txt`

High-level Python dependencies. Install a PyTorch build compatible with the server’s CUDA driver before installing the remaining packages.

---

## 4. What to upload to the SSH server

Upload the complete folder:

```text
C:\kabir\RSNA_Knee_AI\genuine_pipeline
```

It must contain:

```text
build_rsna_fold_safe_labels.py
train_rsna_soft_oof.py
rsna_knee_genuine.py
rsna_backbone_adapters.py
requirements-h100.txt
RSNA_Fold_Safe_H100_Runbook.md
RSNA_Knee_Static_Code_Audit.md
RSNA_SSH_H100_COMPLETE_GUIDE.md
fold_safe_labels/
  gold_outer_folds.csv
  labels_fold_0.csv
  labels_fold_1.csv
  labels_fold_2.csv
  labels_fold_3.csv
  labels_fold_4.csv
  label_builder_manifest.json
```

Also upload or mount separately:

```text
<DATA_ROOT>/train.csv
<DATA_ROOT>/test.csv
<DATA_ROOT>/train_series.csv
<DATA_ROOT>/test_series.csv
<DATA_ROOT>/train_series/
<DATA_ROOT>/test_series/
```

For rebuilding labels, upload separately:

```text
llm_labels_v2.csv
report_labels_gpt56sol.csv
```

For DINOv2, upload the complete model directory containing its configuration and full weights.

For RadImageNet, upload the full compatible ResNet-50 encoder checkpoint, its SHA-256, and its license/provenance note. Do not assume a file named `v52_*_heads.pt` is a full encoder.

---

## 5. Files not to place in the genuine run directory

Do not copy these into the clean run directory unless separately audited:

- Old `submission.csv` files.
- Compressed calibration payloads.
- Public-LB-selected coefficients.
- Competitor fold predictions.
- Inherited E10/E11/E13 heads without provenance.
- Test-derived labels.
- Any checkpoint whose training rows or license are unknown.

The original Kaggle notebook remains useful for comparison, but it is not the genuine training pipeline.

---

## 6. Suggested SSH directory layout

```text
~/rsna_knee/
  pipeline/
    build_rsna_fold_safe_labels.py
    train_rsna_soft_oof.py
    rsna_knee_genuine.py
    rsna_backbone_adapters.py
    requirements-h100.txt
    *.md
    fold_safe_labels/
  data/
    train.csv
    test.csv
    train_series.csv
    test_series.csv
    train_series/
    test_series/
  labels/
    llm_labels_v2.csv
    report_labels_gpt56sol.csv
  models/
    dinov2/
    radimagenet_resnet50.pt
  runs/
```

Use a separate `runs/` directory for checkpoints and metrics. Never overwrite the input data.

---

## 7. Environment setup

Check the machine first:

```bash
nvidia-smi
python3 --version
```

Create an isolated environment:

```bash
python3 -m venv ~/rsna_knee/.venv
source ~/rsna_knee/.venv/bin/activate
python -m pip install --upgrade pip
```

Install PyTorch matching the CUDA driver. Then:

```bash
pip install numpy pandas scikit-learn pydicom
pip install transformers safetensors huggingface_hub
```

Or use the package list:

```bash
pip install -r ~/rsna_knee/pipeline/requirements-h100.txt
```

Verify GPU access:

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('gpu count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
PY
```

---

## 8. Rebuild and verify fold-safe labels

If the prebuilt label folder is uploaded unchanged, it can be used directly. To rebuild it:

```bash
python ~/rsna_knee/pipeline/build_rsna_fold_safe_labels.py \
  --train-csv ~/rsna_knee/data/train.csv \
  --v2-csv ~/rsna_knee/labels/llm_labels_v2.csv \
  --gpt-csv ~/rsna_knee/labels/report_labels_gpt56sol.csv \
  --out ~/rsna_knee/pipeline/fold_safe_labels
```

Verify:

```bash
cat ~/rsna_knee/pipeline/fold_safe_labels/label_builder_manifest.json
```

Expected properties:

- 58 gold rows.
- Five folds.
- Fold sizes 12, 12, 12, 11, 11.
- No single-class target in any gold fold.
- Held-out gold rows have `train_enabled=0` for their fold.

---

## 9. First H100 run: ResNet18 baseline

Run this before DINOv2 or RadImageNet:

```bash
python ~/rsna_knee/pipeline/train_rsna_soft_oof.py \
  --root ~/rsna_knee/data \
  --labels-dir ~/rsna_knee/pipeline/fold_safe_labels \
  --output ~/rsna_knee/runs/resnet18 \
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
~/rsna_knee/runs/resnet18/
  resnet18_fold_0.pt ... resnet18_fold_4.pt
  resnet18_gold_oof.csv
  resnet18_metrics.json
```

---

## 10. DINOv2 run

Use a complete local DINOv2 directory:

```bash
python ~/rsna_knee/pipeline/train_rsna_soft_oof.py \
  --root ~/rsna_knee/data \
  --labels-dir ~/rsna_knee/pipeline/fold_safe_labels \
  --output ~/rsna_knee/runs/dinov2 \
  --architecture dinov2 \
  --dino-source ~/rsna_knee/models/dinov2 \
  --dino-train-last-blocks 4 \
  --epochs 12 \
  --batch-size 2 \
  --workers 8 \
  --image-size 256 \
  --slices-per-slot 12 \
  --compile
```

Before a full run, verify the directory contains `config.json` and the complete model weights. The DINOv2 branch has passed syntax checks but must pass a real GPU forward pass before a full run is trusted.

Confirm the checkpoint’s processor/normalization configuration. The current pipeline uses ImageNet normalization; verify this against the actual model configuration before final training.

---

## 11. RadImageNet ResNet-50 run

First calculate the exact hash:

```bash
sha256sum ~/rsna_knee/models/radimagenet_resnet50.pt
```

Run with the full encoder checkpoint:

```bash
python ~/rsna_knee/pipeline/train_rsna_soft_oof.py \
  --root ~/rsna_knee/data \
  --labels-dir ~/rsna_knee/pipeline/fold_safe_labels \
  --output ~/rsna_knee/runs/radimagenet_resnet50 \
  --architecture radimagenet_resnet50 \
  --radimagenet-checkpoint ~/rsna_knee/models/radimagenet_resnet50.pt \
  --radimagenet-sha256 <EXACT_SHA256> \
  --epochs 12 \
  --batch-size 1 \
  --workers 8 \
  --image-size 256 \
  --slices-per-slot 12
```

If the checkpoint is a trained head bundle rather than a full ResNet-50 encoder, the script should reject it. Do not bypass that check.

---

## 12. Architecture comparison protocol

Run one architecture at a time with the same:

- Fold files.
- Gold validation rows.
- Image size.
- Slice count.
- Epoch budget.
- Seed.
- Metric calculation.

Compare:

```text
resnet18_metrics.json
convnext_tiny_metrics.json
efficientnet_v2_s_metrics.json
dinov2_metrics.json
radimagenet_resnet50_metrics.json
```

Promote a branch only if:

1. Macro gold OOF AUC improves by at least 0.002–0.003.
2. The gain appears in at least 3 of 5 folds.
3. No target has an unexplained fall larger than 0.005.
4. A second seed reproduces the gain.
5. The checkpoint and training data provenance are documented.

A public leaderboard increase alone is not sufficient.

---

## 13. Ensemble policy

After all branches have gold OOF files:

1. Join predictions by `StudyInstanceUID`.
2. Check per-target residual correlation.
3. Fit blend weights only from nested OOF data, or start with equal weights.
4. Do not use public leaderboard feedback to select weights.
5. Save the final blend weights and OOF score in a manifest.
6. Refit each selected branch on the approved full training set only after the configuration is frozen.

The final submission must be generated from the frozen model recipe, not from a manual score adjustment.

---

## 14. Troubleshooting

### `DINOv2 source lacks encoder.layer`

The model directory is not the expected Hugging Face DINOv2 transformer contract. Inspect `config.json` and use a complete compatible DINOv2 model.

### `RadImageNet checkpoint SHA-256 mismatch`

The file differs from the documented checkpoint. Recalculate the hash and update the manifest only after verifying provenance.

### `checkpoint is not a compatible full ResNet-50 encoder`

The file may be a head-only checkpoint or use a different architecture. Do not force-load it.

### `too few positives for folds`

Reduce the number of folds only with justification. Do not allow single-class gold validation folds.

### `OOF is incomplete`

Check that every gold study belongs to exactly one fold and that all DICOM study directories exist.

### GPU out of memory

Reduce batch size first, then image size or slices per slot. Do not change the validation protocol while solving memory issues.

### Training is slow

Use cached DICOM preprocessing, persistent workers, local SSD storage, bfloat16/AMP, and a verified H100. Check disk I/O before increasing model complexity.

---

## 15. What has and has not been verified

Verified in the local environment:

- V2 and GPT label schemas match the 4,407 training studies.
- Fold-safe label generation completed for 58 gold studies.
- Fold builder and trainer pass Python syntax checks.
- The V2/GPT fold-specific label builder produces five output files.
- The project contains a DINOv2 adapter and a strict RadImageNet adapter.

Not verified without the SSH environment:

- Full H100 training result.
- DINOv2 GPU forward pass with the exact uploaded model directory.
- RadImageNet checkpoint compatibility for the exact uploaded file.
- Final private leaderboard score.
- Whether any configuration reaches 0.95 or 0.96 macro AUC.

That distinction is intentional: the pipeline reports measured results rather than inventing them.

---

## 16. Information needed to finalize server commands

Replace these placeholders when the SSH server is known:

```text
SSH host:
SSH username:
project directory:
competition data directory:
label directory:
DINOv2 directory:
RadImageNet checkpoint path:
RadImageNet SHA-256:
output directory:
GPU count:
CUDA/PyTorch version:
```
