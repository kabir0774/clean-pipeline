"""
Diagnostic: run this on the server, same environment as the main pipeline,
to find out WHY Stage A XGBoost's PCA produced NaN explained-variance and
a near-random (0.463) OOF AUC.

Usage (from the same shell/env as run_server_medsiglip_1000.sh):
    python3 diagnose_embeddings.py
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path

WORK_DIR = Path(os.environ.get("RSNA_WORK_DIR",
    "/home/harleen_ece/rsna_knee_ai/original .791 continue/outputs_1000"))

emb_idx_path = WORK_DIR / "train_embedding_index.csv"
print(f"Reading: {emb_idx_path}")
emb_df = pd.read_csv(emb_idx_path, dtype=str)
print(f"Rows in embedding index: {len(emb_df)}")
print(f"Unique studies in embedding index: {emb_df['StudyInstanceUID'].nunique()}")

# Check for missing embedding_file paths on disk
missing = 0
for f in emb_df["embedding_file"].unique():
    if not Path(f).exists():
        missing += 1
print(f"Embedding files referenced but MISSING from disk: {missing} / {emb_df['embedding_file'].nunique()}")

# Check per-study slot coverage
slots_per_study = emb_df.groupby("StudyInstanceUID").size()
print(f"\nSlots-per-study distribution:")
print(slots_per_study.value_counts().sort_index())
zero_slot_studies = (slots_per_study == 0).sum()
print(f"\nStudies appearing in labels but with 0 rows in embedding index "
      f"(would produce an all-zero pooled vector): check against labels CSV separately.")

# Try loading a sample of embedding files to check for NaN/Inf inside them
import torch
print("\nSpot-checking 20 random embedding files for NaN/Inf...")
sample_files = emb_df["embedding_file"].drop_duplicates().sample(
    min(20, emb_df["embedding_file"].nunique()), random_state=0)
bad = 0
for f in sample_files:
    try:
        d = torch.load(f, map_location="cpu", weights_only=False)
        t = d["embeddings"]
        if torch.isnan(t).any() or torch.isinf(t).any():
            print(f"  BAD (NaN/Inf): {f}")
            bad += 1
    except Exception as e:
        print(f"  BAD (failed to load): {f} -- {e}")
        bad += 1
print(f"Bad files in sample: {bad} / {len(sample_files)}")
