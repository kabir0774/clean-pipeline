#!/usr/bin/env bash
set -euo pipefail
ROOT="/home/harleen_ece/rsna_knee_ai"
EXP="$ROOT/original .791 continue"
SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$EXP/code" "$EXP/cache" "$EXP/outputs" "$EXP/logs"
cp -f "$SOURCE_DIR/code/rsnav3_xgboost_pipeline_server_1000.py" "$EXP/code/"
cp -f "$SOURCE_DIR/code/run_server_medsiglip_1000.sh" "$EXP/code/"
chmod +x "$EXP/code/run_server_medsiglip_1000.sh"
echo "Installed experiment at: $EXP"
echo "Run with:"
echo "  cd \"$EXP\""
echo "  tmux new -s original791_1000"
echo "  ./code/run_server_medsiglip_1000.sh"
