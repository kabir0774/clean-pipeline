# RSNA Knee DINOsaur V3 — Static Code Audit

**Source audited:** `rsna-knee-dinosaur-v3.py` (3,675 lines) in the user-approved Downloads folder.

**Audit boundary:** static source review only. I did not execute the script, load its weights, access the competition data, or test the claimed public leaderboard score.

## Executive assessment

This is primarily a complex **test-time inference and ensemble script**, not a self-contained, from-scratch, reproducible training system. It contains useful MRI preprocessing ideas—DICOM geometry-aware ordering, plane/sequence slotting, laterality normalization, and multi-instance pooling—but its reported public performance rests materially on inherited model bundles, fixed blend choices, and a hidden compressed calibration payload.

A genuine route toward higher macro AUC should keep the defensible image-processing components while replacing the inherited/payload-driven prediction stack with a new, fold-isolated training and OOF-selected ensemble pipeline.

## High-priority findings

### 1. The central prediction block explicitly reproduces an external deployed recipe

- **Lines 2458–2466:** Cell 25 describes itself as a “surgical reproduction of V48’s deployed prediction branch.”
- **Lines 2493–2499, 2633–2716:** it requires pinned SHA-256-matched external encoder/head artifacts: `ResNet50.pt`, `v52_radimagenet_heads.pt`, and `v52_e11_heads.pt`.
- **Lines 2793–2950:** it evaluates two external five-fold head families and combines them with fixed weights.

**Risk:** a public score from this branch is not evidence that this project independently trained a model or that it will generalize. The script alone does not establish the original data split, labels, training code, licenses, or whether all validation studies were excluded from every external artifact.

**Action:** remove this block from the genuine training path unless every artifact has a manifest covering public source, license, training data, exact fold assignment, code commit, and OOF predictions. Do not call it “our model.”

### 2. The calibration block is hidden, test-metadata-driven, and not supported by visible OOF fitting

- **Lines 2459–2461:** `_RAD_CAL_PAYLOAD` is a compressed/base64 embedded payload.
- **Lines 3116–3181:** it decodes a stored regression, builds features from `test_series.csv` (series counts by plane, fat suppression, fluid sensitivity), generates adjusted target rankings, and blends them at 40% for gated targets.
- **Lines 3054–3064:** the code itself says the per-finding priors have no fitting set and may be public-leaderboard fitting.

**Risk:** the payload’s provenance and fitting data cannot be audited from this script. Test metadata is an allowed input only if the model was fitted solely from training-fold data; however, payload constants selected against leaderboard movement are not a valid generalization method.

**Action:** delete `_RAD_CAL_PAYLOAD`, `_RAD_CAL`, and `_rad_calibrate` from the baseline. If metadata is tested, build features from train and validation separately inside each fold; report image-only vs image+metadata OOF AUC and site/scanner-proxy stress tests. Do not use isotonic calibration as an AUC shortcut.

### 3. Fixed per-target and adaptive blend choices are not shown to be OOF-selected

- **Lines 2491–2499:** fixed E10/E13 weights and excluded labels.
- **Lines 3020–3107:** per-label stability/diversity adjustments, fixed bounds, a 0.65 “exact share,” and imported per-target priors.
- **Lines 2910–2931:** two targets are held to raw parent values by design.

**Risk:** ranking/blending can be valid for macro AUC, but only when target-specific choices and weights are selected using nested OOF data. In this code the key constants are fixed and provenance is external.

**Action:** build an OOF prediction matrix for every candidate. Choose equal-weight averaging as the starting point; permit target-specific non-negative blend weights only via outer-fold/nested OOF. Store the OOF objective, weights, confidence intervals, and final frozen configuration.

### 4. Report-derived labels are useful but insufficiently auditable

- **Lines 488–531:** labels come from a custom multilingual rule parser over `train.csv` reports, optionally replaced by a mounted `report_labels*.csv` table; table rows replace parser labels where IDs overlap.

**What is good:** the source logs when an external label table is found and refuses a silent fallback when a label dataset is mounted but malformed.

**Risks:** this path does not preserve a label-source field per target/study, it does not force uncertain/not-mentioned targets to be masked, and it does not prove that the external label table is train-only or rule-compliant.

**Action:** create `label_source`, `label_confidence`, and `label_valid` columns for every target. Benchmark parser performance against expert/adjudicated labels per language and target. Use weighted/masked weak supervision; do not use report text at test inference unless the official test schema includes reports.

### 5. The script does not itself prove valid model selection

- **Lines 1425–1427:** `macro_auc` is correctly defined as mean per-label ROC-AUC.
- **Lines 3299–3655:** V16 has an apparent OOF gating mechanism through a separate manifest, but the manifest and its data provenance are not included. It uses hard-coded target caps and can overwrite the submission with an inherited “specialist” model.

**Action:** create a versioned fold file first. For every experiment write: `study_id`, fold, true label, label source, per-fold OOF prediction, per-label AUC, macro AUC, config hash, model hash, and seed. Require a second-seed replication for ensemble entrants.

## Additional code-quality and safety findings

| Location | Finding | Recommended change |
|---|---|---|
| 3219 | `_rad_main()` runs unconditionally at cell execution/import. | Put execution under a top-level `if __name__ == '__main__':` or explicit notebook switch. |
| 2665 and 3343–3347 | Some checkpoint loads set `weights_only=False`. | Prefer `safetensors` or `torch.load(..., weights_only=True)` with strict schema validation. Never load unknown pickle checkpoints. |
| 3670–3674 | The final stage deletes receipts/diagnostic files. | Preserve `oof_predictions`, manifests, timings, and diagnostics; only remove temporary files. |
| 824–894 | DICOM ordering is robust where geometry is available, but can fall back to arbitrary file order when the ordering budget is exceeded. | Fail loudly or mark affected studies; do not silently use arbitrary order for a final model. |
| 805–806 | Per-series percentile normalization is reasonable. | Test it against fold-fitted normalization and retain only the OOF winner. |
| 813–818 | Right-knee canonicalization is applied. | Verify orientation conventions with visual QC and a small, fixed set of test studies before using it in full training. |

## Components worth retaining and testing fairly

1. DICOM position/orientation-based slice ordering (lines 706–757).
2. Series classification into plane/fat-saturation/fluid-sensitive slots (lines 662–695).
3. Per-study multi-series cache with mask-aware attention pooling (lines 824–951).
4. Study-level macro-AUC calculation (lines 1425–1427).
5. Submission ID/schema assertions (lines 2765–2773 and 1484–1497).

These are engineering foundations, not evidence that the current scores are valid.

## Replacement implementation order

1. **Freeze grouped multilabel folds.** One row per study; group by patient where possible; audit duplicate report/image clusters.
2. **Train a clean image-only 2.5D baseline.** Use the retained DICOM loader but train every fold from scratch or with documented public pretrained weights.
3. **Save complete OOF output.** Macro plus all 12 target AUCs; patient-bootstrap uncertainty.
4. **Add audited report weak labels for training only.** Confidence-weighted/masked BCE baseline first.
5. **Run controlled ablations.** Physical crop, number of slices, pooling, loss (BCE then ASL), and MRI-safe augmentation—one variable at a time.
6. **Add models only for OOF-proven diversity.** Use 2–4 members and nested OOF blend selection.
7. **Prepare a clean Kaggle inference notebook.** No hidden payloads, external private artifacts, or tuning against public LB; include model cards, hashes, training command, and runtime dry run.

## What I need before writing the replacement code

- Exact `train.csv` columns and a small schema sample (no patient data need be shared publicly).
- Confirmation of available VRAM/H100 count and whether training will occur on the local machine, SSH server, or Kaggle.
- Whether you want a clean independent model only, or a controlled comparison against the inherited branch for research (never blend it into the final genuine model without provenance and rule review).
