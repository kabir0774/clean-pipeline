"""
Diagnostic v2: reproduces the EXACT build_tabular_matrix + fit_pca path
Stage A XGBoost uses, to find which specific studies/rows introduce NaN --
rather than spot-checking random individual embedding files (which can
miss the culprit if it's a small subset).

Usage: run from the same folder/env as the main pipeline so imports work:
    python3 code/diagnose_embeddings_v2.py
"""
import sys
sys.path.insert(0, "code")  # so we can import from the main pipeline file

import numpy as np
import pandas as pd
import torch
from pathlib import Path

# Import the exact functions the real pipeline uses, so this is a true
# reproduction, not a re-implementation that could differ subtly.
from rsnav3_xgboost_pipeline_server_1000 import (
    WORK_DIR, PARSED_LABELS_CSV, EMBED_DIM, N_SLOT, SLOT_NAMES,
    load_embedding, study_pooled_embedding, build_tabular_matrix,
)

emb_idx_path = WORK_DIR / "train_embedding_index.csv"
emb_df = pd.read_csv(emb_idx_path, dtype=str)
print(f"Embedding index rows: {len(emb_df)}  studies: {emb_df['StudyInstanceUID'].nunique()}")

labeled = pd.read_csv(PARSED_LABELS_CSV, dtype={"StudyInstanceUID": str})
labeled_ids = set(labeled["StudyInstanceUID"].astype(str))
emb_ids = set(emb_df["StudyInstanceUID"].astype(str))
common = labeled_ids & emb_ids
pretrain_ids = np.array(sorted(common))
print(f"Common (labeled + embedded) studies: {len(pretrain_ids)}")

emb_filtered = emb_df[emb_df["StudyInstanceUID"].astype(str).isin(common)].copy()

print("\nBuilding pooled matrix per-study (this reproduces the real Stage A XGBoost path)...")
bad_studies = []
raw_embed, presence = build_tabular_matrix(pretrain_ids, emb_filtered)
print(f"raw_embed shape: {raw_embed.shape}")

nan_rows = np.isnan(raw_embed).any(axis=1)
inf_rows = np.isinf(raw_embed).any(axis=1)
zero_rows = np.all(raw_embed == 0, axis=1)

print(f"\nRows with NaN:  {nan_rows.sum()}")
print(f"Rows with Inf:  {inf_rows.sum()}")
print(f"All-zero rows:  {zero_rows.sum()}")

if nan_rows.sum() > 0:
    print("\nStudies with NaN in pooled embedding:")
    for sid in pretrain_ids[nan_rows][:20]:
        print(f"  {sid}")
if zero_rows.sum() > 0:
    print("\nStudies with ALL-ZERO pooled embedding (no matching slots found):")
    for sid in pretrain_ids[zero_rows][:20]:
        print(f"  {sid}")

# Cross-check: for a zero/nan study, what does emb_filtered actually contain?
if nan_rows.sum() > 0 or zero_rows.sum() > 0:
    bad_idx = np.where(nan_rows | zero_rows)[0][0]
    bad_sid = pretrain_ids[bad_idx]
    print(f"\nDetail for one bad study ({bad_sid}):")
    rows = emb_filtered[emb_filtered["StudyInstanceUID"] == bad_sid]
    print(rows[["StudyInstanceUID", "SeriesInstanceUID", "slot_name", "embedding_file", "presence_mask"]].to_string())
    for _, r in rows.iterrows():
        try:
            t = load_embedding(r["embedding_file"])
            print(f"  {r['slot_name']}: shape={tuple(t.shape)} "
                  f"nan={torch.isnan(t).any().item()} inf={torch.isinf(t).any().item()} "
                  f"mean={t.mean().item():.4f}")
        except Exception as e:
            print(f"  {r['slot_name']}: FAILED TO LOAD -- {e}")

print("\nDone.")
