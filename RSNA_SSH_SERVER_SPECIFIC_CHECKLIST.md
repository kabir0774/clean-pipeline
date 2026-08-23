# RSNA Knee Abnormality Detection
## Server-specific clean deployment checklist

Prepared for Kabir from the verified SSH layout pasted from:
`/home/harleen_ece/rsna_knee_ai`

## Current server facts

The server already contains:

- Competition data under `~/rsna_knee_ai/DATA/`
- DINOv2-small under `~/rsna_knee_ai/DINOv2/dinov2-small/`
- 4,407 training studies and test DICOM data
- `train.csv`, `test.csv`, `train_series.csv`, `test_series.csv`, and `sample_submission.csv`

The GPU is currently unavailable. Do not start full training until `nvidia-smi` works again.

## Clean boundary

Use a new directory:

```bash
mkdir -p ~/rsna_knee_ai/clean_pipeline/{labels,pretrained/radimagenet,fold_safe_labels,runs}
```

Do not use any of these existing artifact directories for the genuine run:

```text
AI-MODEL/
rsnav4_ssh/
rsna_aryan_ssh/
parser/
rsnav4_ssh.zip
kaggle_run/
```

Do not load their `.pt`, `.pkl`, embeddings, caches, submissions, fold heads, or calibration files. The suspicious directory `AI-MODEL/C:/kabir/...` is especially excluded.

## What to copy from the Windows project

Copy the contents of:

```text
C:\kabir\RSNA_Knee_AI\genuine_pipeline
```

to:

```text
~/rsna_knee_ai/clean_pipeline/
```

The clean folder must contain:

```text
build_rsna_fold_safe_labels.py
train_rsna_soft_oof.py
rsna_knee_genuine.py
rsna_backbone_adapters.py
requirements-h100.txt
fold_safe_labels/                 # optional until rebuilt on server
*.md
```

Also copy the two label-source files into `labels/` if labels will be rebuilt on the server:

```text
llm_labels_v2.csv
report_labels_gpt56sol.csv
```

The RadImageNet file is not present yet. Upload only a complete, license-cleared ResNet-50 encoder checkpoint—not a head-only `v52_*_heads.pt` file—and record its SHA-256.

## Verify the exact data layout first

Run these CPU-safe commands:

```bash
cd ~/rsna_knee_ai
find DATA -maxdepth 2 -type d | head -30
find DATA/train_series -mindepth 1 -maxdepth 2 -type d | head -10
ls -lh DATA/train.csv DATA/test.csv DATA/train_series.csv DATA/test_series.csv
ls -lh DINOv2/dinov2-small/pytorch_model.bin DINOv2/dinov2-small/config.json
```

The pasted listing suggests the training DICOM path may be nested as:

```text
~/rsna_knee_ai/DATA/train_series/train_series/<StudyInstanceUID>/<SeriesInstanceUID>/
```

If so, set the data root or create a symlink so the clean loader sees the expected study directories. Do not guess; verify with `find` first.

## Rebuild labels cleanly

From the clean directory, use the actual server paths:

```bash
cd ~/rsna_knee_ai/clean_pipeline
python3 build_rsna_fold_safe_labels.py \
  --train-csv ~/rsna_knee_ai/DATA/train.csv \
  --v2-csv labels/llm_labels_v2.csv \
  --gpt-csv labels/report_labels_gpt56sol.csv \
  --output-dir fold_safe_labels
```

This must produce 58 gold studies, five balanced outer folds, and no single-class target in any gold validation fold. Gold labels remain hard labels; report-derived labels are training-only soft supervision.

## CPU-only preflight while the GPU is off

Do not run model training. First run syntax and import checks:

```bash
cd ~/rsna_knee_ai/clean_pipeline
python3 -m py_compile \
  rsna_knee_genuine.py \
  rsna_backbone_adapters.py \
  build_rsna_fold_safe_labels.py \
  train_rsna_soft_oof.py

python3 - <<'EOF'
from pathlib import Path
import pandas as pd

root = Path.home() / 'rsna_knee_ai'
train = pd.read_csv(root/'DATA/train.csv')
targets = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
print('train studies:', len(train))
print('gold studies:', int(train[targets].notna().all(axis=1).sum()))
print('DINO files:', sorted(p.name for p in (root/'DINOv2/dinov2-small').glob('*')))
EOF
```

Do not call `from_pretrained` or decode DICOMs on CPU unless specifically testing one file; the full run is GPU work.

## H100 run order after GPU restoration

1. `nvidia-smi` and CUDA/PyTorch compatibility check.
2. Verify one DICOM study and one DINOv2 forward pass.
3. Run the DINOv2-only baseline and save OOF predictions.
4. Upload and verify the RadImageNet ResNet-50 checkpoint.
5. Run the RadImageNet-only baseline with a required SHA-256.
6. Compare both models using the same frozen folds and gold OOF macro-AUC.
7. Fit a simple equal-weight or globally constrained blend only on training OOF predictions; never use public-LB feedback.
8. Run final full-data training and produce the submission.

Example DINOv2 baseline command (adjust `--data-root` after the layout check):

```bash
python3 train_rsna_soft_oof.py \
  --data-root ~/rsna_knee_ai/DATA \
  --labels-dir fold_safe_labels \
  --architecture dinov2 \
  --dino-source ~/rsna_knee_ai/DINOv2/dinov2-small \
  --output-dir runs/dinov2_baseline \
  --device cuda
```

Do not run the RadImageNet command until the checkpoint path and hash are known.

## What counts as a genuine result

A result is accepted only if:

- folds are fixed before training;
- no study or patient group crosses train/validation;
- all learned transforms and blend weights are fit without the held-out gold fold;
- reports are used only to create training supervision because test reports are absent;
- OOF predictions are saved for every gold study;
- macro ROC-AUC is calculated from the complete OOF vector across all 12 targets;
- per-target AUC, positive counts, fold variation, and bootstrap uncertainty are reported;
- no calibration payload, test metadata regression, copied competitor prediction, or public-LB-tuned constant is used.

A 0.96 score is a target, not a guarantee. If the clean OOF score does not support it, we report that honestly rather than manufacture an improvement.

## Immediate status

Completed: clean scripts, clean label logic, balanced fold patch, DINOv2/RadImageNet architecture wiring, and this server-specific checklist.

Blocked: GPU is off; RadImageNet checkpoint has not been uploaded; the exact nested DICOM path needs one `find` confirmation; no H100 forward pass or training result has yet been verified.
