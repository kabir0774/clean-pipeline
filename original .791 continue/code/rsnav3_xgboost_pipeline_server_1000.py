#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/harleen_ece/rsna_knee_ai"
EXP="$ROOT/original .791 continue"
CODE_DIR="$EXP/code"
CACHE_DIR="$EXP/cache_1000"
OUTPUT_DIR="$EXP/outputs_1000"
LOG_DIR="$EXP/logs"
SCRIPT="$CODE_DIR/rsnav3_xgboost_pipeline_server_1000.py"
mkdir -p "$CODE_DIR" "$CACHE_DIR" "$OUTPUT_DIR" "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_1000_$(date +%Y%m%d_%H%M%S).log"

export RSNA_SERVER_ROOT="$ROOT"
export RSNA_DATA_ROOT="$ROOT/DATA"
export RSNA_MODEL_PATH="$ROOT/MedSigLIP"
export RSNA_LABELS_CSV="$ROOT/AI-MODEL/final_labels_real_plus_generated.csv"
export RSNA_EXPERIMENT_DIR="$EXP"
export RSNA_CACHE_DIR="$CACHE_DIR"
export RSNA_WORK_DIR="$OUTPUT_DIR"

export RSNA_N_TOTAL_STUDIES="1000"
export RSNA_SAMPLE_SEED="42"
export RSNA_BATCH_SIZE="4"
export RSNA_RUN_STAGE_A0="1"
export RSNA_STAGE_A0_BLOCKS="2"
export RSNA_STAGE_A0_EPOCHS="1"
export RSNA_STAGE_A0_MAX_IMAGES="4"
export RSNA_RUN_TEST_INFERENCE="0"
export RSNA_REQUIRE_CUDA="1"
export PYTHONUNBUFFERED="1"

for p in "$SCRIPT" "$RSNA_DATA_ROOT/train.csv" "$RSNA_DATA_ROOT/train_series.csv" "$RSNA_DATA_ROOT/train_series" "$RSNA_MODEL_PATH/config.json" "$RSNA_LABELS_CSV"; do
  [[ -e "$p" ]] || { echo "Missing required input: $p" >&2; exit 1; }
done

python3 - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; verify the GPU allocation before training.")
print("GPU:", torch.cuda.get_device_name(0))
PY

python3 - <<'PY'
import os
from pathlib import Path
import pandas as pd
p=Path(os.environ["RSNA_LABELS_CSV"])
df=pd.read_csv(p,dtype={"StudyInstanceUID":str})
targets=["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
missing=[c for c in ["StudyInstanceUID","is_real_label",*targets] if c not in df.columns]
if missing: raise SystemExit(f"Label CSV missing columns: {missing}")
flag=df["is_real_label"]
if flag.dtype != bool:
    flag=flag.astype(str).str.strip().str.lower().map({"true":True,"false":False,"1":True,"0":False})
if flag.isna().any(): raise SystemExit("is_real_label contains unsupported values")
gold=df[flag].StudyInstanceUID.nunique(); weak=df[~flag].StudyInstanceUID.nunique()
print("Gold studies:",gold); print("Weak studies:",weak)
if gold != 58: raise SystemExit(f"Expected exactly 58 gold studies, found {gold}")
if weak < 942: raise SystemExit(f"Need at least 942 weak studies, found {weak}")
PY

echo "Experiment: $EXP"
echo "Code:       $CODE_DIR"
echo "Cache:      $CACHE_DIR"
echo "Outputs:    $OUTPUT_DIR"
echo "Live log:   $LOG_FILE"
echo "Fresh 1000-study cache: $CACHE_DIR (the old 4349-study cache is not used)."
echo "Restart the same command after an error; completed 1000-study embeddings and fold checkpoints will be reused."
python3 -u "$SCRIPT" 2>&1 | tee -a "$LOG_FILE"
