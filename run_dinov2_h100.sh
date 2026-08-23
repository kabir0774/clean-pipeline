#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/harleen_ece/rsna_knee_ai"
PIPE="$ROOT/clean_pipeline"
DATA="$ROOT/DATA"
DINO="$ROOT/DINOv2/dinov2-small"

cd "$PIPE"
mkdir -p runs logs

python3 -m py_compile \
  rsna_knee_genuine.py \
  rsna_backbone_adapters.py \
  train_rsna_soft_oof.py

python3 train_rsna_soft_oof.py \
  --root "$DATA" \
  --labels-dir "$PIPE/fold_safe_labels" \
  --architecture dinov2 \
  --dino-source "$DINO" \
  --slice-sampling middle \
  --middle-fraction 0.60 \
  --slices-per-slot 12 \
  --image-size 256 \
  --batch-size 3 \
  --workers 6 \
  --output "$PIPE/runs/dinov2_sixslot_laterality" 2>&1 | tee "$PIPE/logs/dinov2_sixslot_laterality.log"
