# original .791 continue — SSH GPU 1,000-study run

This directory is the complete GitHub package for the resumable 1,000-study MedSigLIP experiment.

## Server layout created by setup

```text
/home/harleen_ece/rsna_knee_ai/original .791 continue/
├── code/
│   ├── rsnav3_xgboost_pipeline_server_1000.py
│   ├── run_server_medsiglip_1000.sh
│   └── experiment_config.json
├── cache/
│   ├── embeddings/
│   ├── models/
│   ├── train_dicom_index.csv
│   ├── train_slots_cache.pkl
│   └── train_embedding_index.csv
├── outputs/
└── logs/
```

The run uses 1,000 studies: all 58 official gold studies plus 942 deterministic weak-labelled studies. Stage A0 and Stage A are weak-only. Stage B performs leakage-safe five-fold gold OOF.

## Restart behavior

Rerun the same launcher after an error. It reuses:

- completed per-series MedSigLIP embedding files;
- the DICOM index and slot cache;
- Stage A0 progress every 100 studies;
- completed neural Stage A folds;
- completed XGBoost Stage A folds;
- completed neural and XGBoost Stage B folds;
- final OOF predictions when all five folds are complete.

Important cache and checkpoint writes are atomic, so interrupted `.tmp` files are never treated as finished artifacts. If the MedSigLIP encoder changes, downstream embeddings and model checkpoints are invalidated automatically while the encoder-independent DICOM and slot caches are preserved.

## Install from GitHub after `git pull`

From the cloned repository root, run:

```bash
chmod +x "original .791 continue/setup_original_791_continue.sh"
"original .791 continue/setup_original_791_continue.sh"
```

Then start training:

```bash
cd "/home/harleen_ece/rsna_knee_ai/original .791 continue"
tmux new -s original791_1000
./code/run_server_medsiglip_1000.sh
```

Detach with `Ctrl-b`, then `d`. Reattach with:

```bash
tmux attach -t original791_1000
```

The launcher requires CUDA and checks the known data, MedSigLIP and best-label paths before training. Test inference is disabled for this experiment.
