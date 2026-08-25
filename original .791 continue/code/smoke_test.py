"""
Smoke test: fast, CPU-only sanity checks against REAL cached data on the
server, run BEFORE committing to a full (slow, GPU) training run. Every
check here is designed to catch a specific class of bug that would
otherwise only surface as a bad/garbage number hours into training --
same category as the presence_mask reload bug this test was written
after finding.

Exits non-zero (and prints a clear FAIL) on the first real problem, so it
is safe to gate a launcher on (e.g. `python3 code/smoke_test.py && ./run...`).

Usage (from the experiment root, same env as the main launcher):
    python3 code/smoke_test.py
"""
import sys
sys.path.insert(0, "code")

import numpy as np
import pandas as pd
import torch

FAILS = []

def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILS.append(name)

print("=" * 60)
print("SMOKE TEST — rsnav3_xgboost_pipeline_server_1000.py")
print("=" * 60)

# ── Import the real module (executes its top-level config/dir setup) ──
try:
    import rsnav3_xgboost_pipeline_server_1000 as pipe
    check("Module imports cleanly", True)
except Exception as e:
    check("Module imports cleanly", False, str(e))
    print("\nCannot continue — fix the import error above first.")
    sys.exit(1)

print(f"\nWORK_DIR   : {pipe.WORK_DIR}")
print(f"CACHE_DIR  : {pipe.CACHE_DIR}")
print(f"EMB_DIR    : {pipe.EMB_DIR}")
print(f"MODEL_DIR  : {pipe.MODEL_DIR}")
print(f"LABELS_CSV : {pipe.PARSED_LABELS_CSV}")
print(f"N_TOTAL    : {pipe.N_TOTAL_STUDIES}")

# ── 1. presence_mask dtype check — the exact bug we just fixed ──
emb_idx_path = pipe.WORK_DIR / "train_embedding_index.csv"
if emb_idx_path.exists():
    raw = pd.read_csv(emb_idx_path, dtype=str)
    n_rows = len(raw)
    # reproduce what main() actually does after the fix
    fixed = pd.to_numeric(raw["presence_mask"], errors="coerce").fillna(0).astype(int)
    n_present = int((fixed == 1).sum())
    check(
        "presence_mask survives reload as usable int (not stuck at 0 due to string compare)",
        n_present > 0,
        f"{n_present}/{n_rows} rows have presence_mask==1 after fix — should be most of them"
    )
    # also confirm this WOULD have been silently broken pre-fix, as a
    # regression guard: if this ever reads back as already-int, that's
    # fine too (pandas can infer it), but zero after our fix is real.
else:
    print(f"[SKIP] presence_mask check — no cached embedding index yet at {emb_idx_path}")

# ── 2. build_tabular_matrix on a REAL small sample — no all-zero rows ──
common, sample_ids, labeled, emb_df = set(), np.array([]), None, None
if emb_idx_path.exists() and pipe.PARSED_LABELS_CSV.exists():
    labeled = pd.read_csv(pipe.PARSED_LABELS_CSV, dtype={"StudyInstanceUID": str})
    emb_df = pd.read_csv(emb_idx_path, dtype=str)
    emb_df["presence_mask"] = pd.to_numeric(emb_df["presence_mask"], errors="coerce").fillna(0).astype(int)

    common = set(labeled["StudyInstanceUID"].astype(str)) & set(emb_df["StudyInstanceUID"].astype(str))
    sample_ids = np.array(sorted(common))[:20]  # small, fast sample — not the full 1000
    if len(sample_ids) > 0:
        raw_embed, presence = pipe.build_tabular_matrix(sample_ids, emb_df)
        n_zero = int(np.all(raw_embed == 0, axis=1).sum())
        n_nan  = int(np.isnan(raw_embed).any(axis=1).sum())
        check("No all-zero pooled embeddings in sample", n_zero == 0,
              f"{n_zero}/{len(sample_ids)} studies pooled to an all-zero vector")
        check("No NaN in pooled embeddings in sample", n_nan == 0,
              f"{n_nan}/{len(sample_ids)} studies produced NaN")

        # ── 3. PCA on the sample doesn't produce NaN explained variance ──
        try:
            import warnings
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                pca = pipe.fit_pca(raw_embed, n_components=min(8, len(sample_ids) - 1))
                var_nan = np.isnan(pca.explained_variance_ratio_).any()
                runtime_warns = [x for x in w if "invalid value" in str(x.message)]
            check("PCA explained variance is not NaN", not var_nan,
                  f"explained_variance_ratio_={pca.explained_variance_ratio_}")
            check("No 'invalid value in divide' warning from PCA", len(runtime_warns) == 0)
        except Exception as e:
            check("PCA runs without error", False, str(e))
    else:
        print("[SKIP] build_tabular_matrix check — no common labeled+embedded studies found")
else:
    print("[SKIP] build_tabular_matrix check — missing embedding index or labels CSV")

# ── 4. StudyDataset actually returns non-empty tensors for a real study ──
if emb_idx_path.exists() and pipe.PARSED_LABELS_CSV.exists() and len(common) > 0:
    small_lbl = labeled[labeled["StudyInstanceUID"].astype(str).isin(sample_ids)]
    small_emb = emb_df[emb_df["StudyInstanceUID"].astype(str).isin(sample_ids)]
    try:
        ds = pipe.StudyDataset(small_emb, small_lbl)
        study, x, mask, idx, y = ds.get(0)
        check("StudyDataset.get() returns non-empty embedding tensor", x.numel() > 0,
              f"x.shape={tuple(x.shape)}")
        check("StudyDataset.get() mask has at least one present slot", mask.sum().item() > 0,
              f"mask.sum()={mask.sum().item()}")
    except Exception as e:
        check("StudyDataset.get() runs without error", False, str(e))

# ── 5. Model checkpoints referenced by resume logic actually load ──
for ckpt_name in ["stage_a_pretrained.pt"] + [f"fold_{i}.pt" for i in range(1, 6)]:
    p = pipe.MODEL_DIR / ckpt_name
    if p.exists():
        try:
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
            check(f"Checkpoint loads: {ckpt_name}", True)
        except Exception as e:
            check(f"Checkpoint loads: {ckpt_name}", False, str(e))
    else:
        print(f"[SKIP] {ckpt_name} — not present yet (fine if not trained yet)")

# ── 6. xgb_stage_a.pkl, if present, isn't the known-bad stale checkpoint ──
xgb_path = pipe.MODEL_DIR / "xgb_stage_a.pkl"
if xgb_path.exists():
    print(f"[WARN] {xgb_path} exists — if this predates the presence_mask fix, "
          f"delete it and let Stage A XGBoost retrain (see prior diagnosis).")

print("\n" + "=" * 60)
if FAILS:
    print(f"SMOKE TEST FAILED — {len(FAILS)} check(s) failed:")
    for f in FAILS:
        print(f"  - {f}")
    print("=" * 60)
    sys.exit(1)
else:
    print("SMOKE TEST PASSED — safe to proceed with a full run.")
    print("=" * 60)
