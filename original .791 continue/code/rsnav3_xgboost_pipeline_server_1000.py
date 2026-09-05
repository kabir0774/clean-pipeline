# ============================================================
# RSNA KNEE ABNORMALITY DETECTION
# Full pipeline: DICOM → MedSigLIP → Attention Model → Submission
#
# Kaggle paths:
#   Data    : /kaggle/input/rsna-knee-abnormality-2024/
#   MedSigLIP: /kaggle/input/medsiglip/
#   Output  : /kaggle/working/
# ============================================================

import os
import re
import math
import random
import pickle
import argparse
import subprocess
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold


def report_groups(study_ids, data_root):
    """
    Group key for GroupKFold. Grouping by study UID leaks: 49 reports in
    train.csv are shared verbatim by 183 studies (one report by 37 of them), and
    a shared report yields ONE derived target vector for all of them. Split by
    study and those identical-target rows land on both sides of the fold, so OOF
    reads optimistically. Grouping by report hash keeps them together.
    """
    import hashlib
    p = Path(data_root) / "train.csv"
    if not p.exists():
        return list(study_ids)
    t = pd.read_csv(p, dtype={"StudyInstanceUID": str})
    if "Report" not in t.columns:
        return list(study_ids)
    h = {r.StudyInstanceUID: hashlib.md5(str(r.Report or "").strip().encode()).hexdigest()
         for r in t.itertuples(index=False)}
    g = [h.get(str(s), str(s)) for s in study_ids]
    n_st, n_gr = len(set(map(str, study_ids))), len(set(g))
    if n_gr < n_st:
        print(f"  fold groups: {n_gr} report-groups from {n_st} studies "
              f"({n_st - n_gr} studies share a report with another)")
    return g
from sklearn.decomposition import PCA

# ── config ────────────────────────────────────────────────────────────────────
IS_KAGGLE     = os.path.exists("/kaggle/input")

# Linux SSH/GPU server (not Kaggle, not the local Windows dev machine).
# Same server layout used by RSNA v4.
IS_SERVER     = (not IS_KAGGLE) and os.name == "posix"
SERVER_ROOT   = Path("/home/harleen_ece/rsna_knee_ai")

def _find_data_root():
    # Handles both the raw competition dataset and a manually-renamed copy.
    candidates = [
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-2024"),
    ]
    for c in candidates:
        if (c / "train_series.csv").exists() or (c / "train.csv").exists():
            print(f"  Data root found at: {c}")
            return c
    print("  Scanning /kaggle/input (max depth 3) for competition data (train_series.csv)...")
    # bounded-depth scan instead of unbounded rglob — avoids crawling into
    # the (huge) train_series/<study>/<series>/*.dcm tree, which is what
    # made the old unbounded rglob effectively hang for minutes.
    base = Path("/kaggle/input")
    for depth in range(0, 4):
        pattern = "/".join(["*"] * depth + ["train_series.csv"]) if depth else "train_series.csv"
        for p in sorted(base.glob(pattern)):
            print(f"  Data root found at: {p.parent}")
            return p.parent
    raise RuntimeError(
        "Competition data not found under /kaggle/input.\n"
        "Attach the RSNA Knee Abnormality Detection competition data to this notebook."
    )

if IS_KAGGLE:
    DATA_ROOT = _find_data_root()
elif IS_SERVER:
    DATA_ROOT = Path("/home/harleen_ece/rsna_knee_ai/DATA")
else:
    DATA_ROOT = Path("C:/kabir/RSNA_Knee_AI/DATA")
# auto-find MedSigLIP on Kaggle — handles subfolder variations
def _find_medsiglip():
    # kabirverma01/medsiglip dataset — files are in root
    candidates = [
        Path("/kaggle/input/datasets/kabirverma01/medsiglip/MedSigLIP"),  # confirmed actual path
        Path("/kaggle/input/datasets/kabirverma01/medsiglip"),
        Path("/kaggle/input/medsiglip"),           # dataset slug = medsiglip
        Path("/kaggle/input/medsiglip/MedSigLIP"), # if Kaggle adds subfolder
    ]
    for c in candidates:
        if (c / "config.json").exists():
            print(f"  MedSigLIP found at: {c}")
            return c
    # fallback: bounded-depth scan of /kaggle/input (avoids crawling into
    # the huge train_series/<study>/<series>/*.dcm tree)
    print("  Scanning /kaggle/input (max depth 3) for MedSigLIP...")
    base = Path("/kaggle/input")
    for depth in range(0, 4):
        pattern = "/".join(["*"] * depth + ["config.json"]) if depth else "config.json"
        for p in sorted(base.glob(pattern)):
            if (p.parent / "model.safetensors").exists():
                print(f"  MedSigLIP found at: {p.parent}")
                return p.parent
    raise RuntimeError(
        "MedSigLIP not found.\n"
        "Attach dataset kabirverma01/medsiglip to this notebook."
    )

if IS_KAGGLE:
    MODEL_PATH = _find_medsiglip()
elif IS_SERVER:
    MODEL_PATH = Path("/home/harleen_ece/rsna_knee_ai/MedSigLIP")
else:
    MODEL_PATH = Path("C:/kabir/RSNA_Knee_AI/MedSigLIP")

if IS_KAGGLE:
    WORK_DIR = Path("/kaggle/working")
elif IS_SERVER:
    WORK_DIR = SERVER_ROOT / "rsnav4_ssh"  # all cache/output goes here
else:
    WORK_DIR = Path("C:/kabir/RSNA/kaggle_run")
TRAIN_SERIES  = DATA_ROOT / "train_series"
TEST_SERIES   = DATA_ROOT / "test_series"
EMB_DIR       = WORK_DIR / "embeddings"
MODEL_DIR     = WORK_DIR / "models"
WORK_DIR.mkdir(parents=True, exist_ok=True)
EMB_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def _kaggle_dataset_variants(slug):
    """
    Kaggle sometimes mounts an attached dataset at /kaggle/input/<slug>
    and sometimes nests it under /kaggle/input/datasets/<username>/<slug>
    (confirmed for this account: MedSigLIP landed at
    /kaggle/input/datasets/kabirverma01/medsiglip). Returns both shapes so
    callers can just check each path.
    """
    base = Path("/kaggle/input")
    variants = [base / slug]
    datasets_dir = base / "datasets"
    if datasets_dir.exists():
        for user_dir in datasets_dir.iterdir():
            if user_dir.is_dir():
                variants.append(user_dir / slug)
    return variants

# If a previous Kaggle run's WORK_DIR (embeddings, models, index CSVs) was
# saved as its own Dataset and re-attached, copy everything back into place
# up front. embed_slots() and the Stage A / Stage B resume checks each skip
# recomputation when their target files already exist, so this picks up
# exactly where a previous run left off instead of starting over.
if IS_KAGGLE:
    # confirmed cache dataset path: kabirverma01/rsna-v4-xgboost-cache
    _resume_candidates = [
        Path("/kaggle/input/datasets/kabirverma01/rsna-v4-xgboost-cache/rsnav4_ssh"),
        Path("/kaggle/input/rsna-v4-xgboost-cache/rsnav4_ssh"),
        Path("/kaggle/input/rsna-v4-xgboost-cache"),
    ]
    # also check old slugs as fallback
    for _slug in ["rsnav3-trained", "rsna-embeddings-cache"]:
        for v in _kaggle_dataset_variants(_slug):
            _resume_candidates.append(v)

    _resume_root = None
    for _c in _resume_candidates:
        if Path(_c).exists() and (
            (Path(_c) / "embeddings").exists() or
            (Path(_c) / "train_embedding_index.csv").exists()
        ):
            _resume_root = Path(_c)
            print(f"  Cache found at: {_resume_root}")
            break

    if _resume_root:
        import shutil as _shutil
        print(f"  Restoring cache from {_resume_root} into {WORK_DIR} ...")

        # embeddings/ and models/ subfolders
        for _sub in ("embeddings", "models"):
            _src = _resume_root / _sub
            if _src.exists():
                _dst = WORK_DIR / _sub
                _shutil.copytree(_src, _dst, dirs_exist_ok=True)
                _n = sum(1 for _ in _dst.rglob("*") if _.is_file())
                print(f"    restored {_sub}/  ({_n} files)")

        # index CSVs + slot cache at dataset root
        for _fname in ("train_dicom_index.csv", "train_embedding_index.csv", "train_slots_cache.pkl"):
            _src = _resume_root / _fname
            if _src.exists():
                _shutil.copy2(_src, WORK_DIR / _fname)
                print(f"    restored {_fname}")

        print("  Cache restore complete — scan/slot/embed steps will be skipped.")
    else:
        print("  No cache dataset found — starting fresh (full pipeline will run).")

# Big zip holding the (mostly-unlabeled) train series that are NOT already
# unzipped on disk. Only used LOCALLY — on Kaggle, data is already unzipped
# (mounted read-only from the attached Dataset), so this path is unused there.
if IS_KAGGLE:
    TRAIN_SERIES_ZIP = None
elif IS_SERVER:
    TRAIN_SERIES_ZIP = None  # data already extracted on SSH server
else:
    TRAIN_SERIES_ZIP = Path(r"D:\rsna-knee-abnormality-detection.zip")

# Parser output (weak labels for everyone + real labels for 58) produced by
# train_and_predict.py / report_parser.py. This REPLACES train.csv as the
# label source. On Kaggle, upload this CSV as its own small Dataset (or
# alongside train_series.csv in the main data dataset) and attach it —
# code auto-finds it under /kaggle/input if the exact path below isn't found.
if IS_KAGGLE:
    _parsed_candidates = [
        # confirmed new path: kabirverma01/final-labels-real-plus-generated-best
        Path("/kaggle/input/datasets/kabirverma01/final-labels-real-plus-generated-best/final_labels_real_plus_generated best.csv"),
        Path("/kaggle/input/final-labels-real-plus-generated-best/final_labels_real_plus_generated best.csv"),
        # old slug fallbacks
        Path("/kaggle/input/datasets/kabirverma01/final-labels-real-plus-generated/final_labels_real_plus_generated.csv"),
        Path("/kaggle/input/final-labels-real-plus-generated/final_labels_real_plus_generated.csv"),
    ]
    _parsed_candidates += [v / "final_labels_real_plus_generated best.csv"
                           for v in _kaggle_dataset_variants("final-labels-real-plus-generated-best")]
    _parsed_candidates += [v / "final_labels_real_plus_generated.csv"
                           for v in _kaggle_dataset_variants("final-labels-real-plus-generated")]
    _parsed_candidates += [v / "final_labels_real_plus_generated.csv"
                           for v in _kaggle_dataset_variants("rsna-knee-labels")]
    _parsed_candidates.append(DATA_ROOT / "final_labels_real_plus_generated.csv")
    _found_labels = next((p for p in _parsed_candidates if p.exists()), None)
    if _found_labels is None:
        # scan all of /kaggle/input for any matching filename
        import os as _os
        for _root, _dirs, _files in _os.walk("/kaggle/input"):
            _dirs[:] = [d for d in _dirs if d not in ("train_series", "test_series")]
            for _f in _files:
                if "final_labels_real_plus_generated" in _f and _f.endswith(".csv"):
                    _found_labels = Path(_root) / _f
                    break
            if _found_labels:
                break
    if _found_labels is None:
        raise FileNotFoundError(
            "Cannot find final_labels_real_plus_generated*.csv\n"
            "Attach dataset kabirverma01/final-labels-real-plus-generated-best"
        )
    print(f"  Labels CSV: {_found_labels}")
    PARSED_LABELS_CSV = _found_labels
elif IS_SERVER:
    PARSED_LABELS_CSV = Path(
        "/home/harleen_ece/rsna_knee_ai/AI-MODEL/final_labels_real_plus_generated.csv"
    )
else:
    PARSED_LABELS_CSV = Path(r"C:\kabir\RSNA_Knee_AI\parser\output\final_labels_real_plus_generated.csv")

N_TOTAL_STUDIES = int(os.environ.get("RSNA_N_STUDIES", 4407))  # FIX: was 4340; dataset has 4,407
SAMPLE_SEED     = int(os.environ.get("RSNA_SEED", 42))

# ── constants ─────────────────────────────────────────────────────────────────
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]

SLOT_NAMES = [
    "SAG_FLUID_FS",
    "COR_FLUID_FS",
    "AX_FLUID_FS",
    "AX_T1",          # FIX: was SAG_FLUID_NOFS -- 0 rows in train_series.csv
    "COR_T1",
    "SAG_T1",
]
N_SLOT    = len(SLOT_NAMES)
EMBED_DIM_BASE = 1152
PROJ_DIM  = 256

# Patch-level pooling. get_image_features returns ONE pooled vector per image,
# which averages every patch in the field. A meniscal tear occupies a small part
# of that field, so a plain mean over ~256 patches dilutes it by roughly two
# orders of magnitude -- the same dilution problem as slice pooling, one level
# down. "cls_mean_focal" concatenates the CLS token, the patch mean, and the
# mean of the top eighth of each channel's responses across patches, so a strong
# localised response survives instead of being averaged into the background.
#   "pooled"          -> 1x EMBED_DIM  (previous behaviour, get_image_features)
#   "cls_mean"        -> 2x
#   "cls_mean_focal"  -> 3x            (default)
PATCH_POOL = os.environ.get("RSNA_PATCH_POOL", "cls_mean_focal").lower()
POOL_PARTS = {"pooled": 1, "cls_mean": 2, "cls_mean_focal": 3}[PATCH_POOL]
EMBED_DIM  = EMBED_DIM_BASE * POOL_PARTS
# Raised to process more of each series. 12 slices from the middle 60%
# used only ~31% of the DICOMs on disk. 20 slices from the middle 76%
# roughly doubles coverage. Encoding time scales linearly with this.
MAX_SLICES  = int(os.environ.get("RSNA_MAX_SLICES", 20))
BATCH_SIZE  = int(os.environ.get("RSNA_BATCH_SIZE", 16))
# 6-94%, not 12-88%. The collateral ligaments and the lateral meniscus sit in
# the peripheral slices a tighter band discards, and those are two of the
# weakest classes in this pipeline.
SLICE_BAND  = (float(os.environ.get("RSNA_BAND_LO", 0.06)),
               float(os.environ.get("RSNA_BAND_HI", 0.94)))
LAT_OFFSET  = 20.0
PRIOR_STRENGTH = 0.55
WINDOW_PCT     = (0.5, 99.5)     # FIX #16 (was 1, 99)

# RUN A switches. All default ON; set the env var to 0 to disable one and
# isolate its effect. None of these require a re-encode.
TWO_LEVEL_ATT  = os.environ.get("RSNA_TWO_LEVEL", "1") == "1"
LEARNABLE_PRIOR= os.environ.get("RSNA_LEARN_PRIOR", "1") == "1"
SLICE_POSENC   = os.environ.get("RSNA_SLICE_POSENC", "1") == "1"

# Within-slot pooling. The original Stanford MRNet MAX-pools across slices
# rather than using softmax attention, and that is not incidental: softmax
# weights must sum to 1, so a finding visible on 3 of 20 slices is diluted by
# the 17 showing nothing. Max keeps the strongest evidence. Your worst classes
# are the focal ones (ACL, both menisci); your best is Effusion, which is
# diffuse and bright on every slice -- exactly the pattern dilution predicts.
#   "max" -> pure max over slices
#   "att" -> softmax attention (Run A behaviour)
#   "mix" -> 0.5*max + 0.5*attention  (default)
SLOT_POOL = os.environ.get("RSNA_SLOT_POOL", "mix").lower()

# Physical-millimetre cropping. Until now every slice was handed to the
# processor whole and resized to 384, so a wide field-of-view scanner produced a
# small knee in a large frame and a tight one produced a large knee -- the same
# anatomy arriving at different scales, leaving the encoder to absorb a
# scale-invariance it should never have needed. Cropping a fixed CROP_MM box
# around the image centre using PixelSpacing makes the knee occupy the same
# fraction of the frame whatever the scanner did.
CROP_MM = float(os.environ.get("RSNA_CROP_MM", 140.0))   # 0 disables

# Three neighbouring slices in the three colour channels. SigLIP wants three
# channels and a grey slice was being copied into all three, so two thirds of
# the input carried no information. Filling them with the slices above and
# below gives the encoder local depth context at no extra cost -- most of what
# a 3D network buys, for the price of a 2D one.
RGB_MODE = os.environ.get("RSNA_RGB_MODE", "neighbours").lower()   # or "gray"

# Sliding-window TTA. The embedding cache already holds every slice, so looking
# at more of them at inference costs forward passes and no extra decoding. The
# model is read over each consecutive window of TTA_GROUP slices per slot and
# the results averaged. Logit-space averaging is a geometric mean of odds,
# probability-space an arithmetic mean of risk; they order studies differently,
# and macro-AUC reads only order, so this is a real choice rather than a detail.
TTA_WINDOWS = int(os.environ.get("RSNA_TTA_WINDOWS", 0))   # 0 = off
TTA_GROUP   = int(os.environ.get("RSNA_TTA_GROUP", 12))
TTA_POOL    = os.environ.get("RSNA_TTA_POOL", "logit").lower()

# Fingerprinting. Weights loaded through the wrong preprocessing produce
# predictions, not errors: the submission is well formed, the log says nothing,
# and no output of the run reveals the difference. A checkpoint therefore
# carries the answer it gave to a seeded synthetic input, recomputed before use.
# GPU numeric noise moves that by ~1e-5; any real preprocessing difference moves
# it by order one, so the tolerance sits between them.
FINGERPRINT_TOL = 2e-3

# Index 3 is now AX_T1 (axial T1), not SAG_FLUID_NOFS. Axial T1 shows the
# patellofemoral joint and tibiofibular articulation well; it is poor for
# the cruciates and the meniscal body, so the priors differ from the old
# sagittal slot they replace.
SLOT_PRIOR = {
    #                   SAG_FS COR_FS AX_FS AX_T1 COR_T1 SAG_T1
    "ACL":              [1,     0,     0,    0,    0,     1],
    "MCL":              [0,     1,     0,    0,    1,     0],
    "Medial Meniscus":  [1,     1,     0,    0,    1,     0],
    "Lateral Meniscus": [1,     1,     0,    0,    1,     0],
    "Medial OA":        [0,     1,     0,    0,    1,     0],
    "Lateral OA":       [0,     1,     0,    0,    1,     0],
    "PF OA":            [1,     0,     1,    1,    0,     1],
    "Effusion":         [1,     0,     1,    1,    0,     0],
    "Synovitis":        [1,     0,     1,    1,    0,     0],
    "Baker's":          [1,     0,     1,    0,    0,     0],
    "Contusion":        [1,     1,     0,    0,    1,     0],
    "Fracture":         [1,     1,     1,    1,    1,     1],
}

# FIX: verified against train_series.csv (24,371 series). Only six
# (plane, fluid, fat) combinations exist in the data. (Sagittal,1,0) has
# ZERO rows -- the old SAG_FLUID_NOFS slot was permanently empty, a dead
# input channel. Meanwhile (Axial,0,0) had 1,179 series in 857 studies
# matching no slot at all, silently discarded. Swapping the two recovers
# them and removes the dead channel.
SLOTS = [
    ("SAG_FLUID_FS", "Sagittal", 1, 1),
    ("COR_FLUID_FS", "Coronal",  1, 1),
    ("AX_FLUID_FS",  "Axial",    1, 1),
    ("AX_T1",        "Axial",    0, 0),
    ("COR_T1",       "Coronal",  0, 0),
    ("SAG_T1",       "Sagittal", 0, 0),
]

# Localizer / scout rejection. A 3-plane localizer often has MORE slices
# than the diagnostic scan, so "most slices wins" actively selects junk.
LOCALIZER_RE   = re.compile(r"loc|scout|survey|localiz|localis|3pl|surview|plan|smartbrain", re.I)
MIN_SERIES_SLICES = 6      # below this, not a real diagnostic sequence
MAX_SLICE_THICK   = 8.0    # mm; knee MRI is typically 3-4mm
MIN_INPLANE_DIM   = 160    # px; scouts are low-res


# Run-to-run variance has never been measured here, so every comparison so far
# has been one number against one number with no idea how much a number moves on
# its own. Same config trained twice gives two different models: weights start
# random, dropout drops different units, and cuDNN sums in a nondeterministic
# order. Run one config at several seeds and the spread IS the noise floor.
RUN_SEED = int(os.environ.get("RSNA_SEED", 42))


def seed_everything(s=None):
    s = RUN_SEED if s is None else s
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 0 — SELECTIVE UNZIP (only for studies we actually need, only once)
# ══════════════════════════════════════════════════════════════════════════════
def ensure_studies_unzipped(study_ids, series_root, zip_path):
    """
    Make sure `series_root` contains a folder for every StudyInstanceUID in
    `study_ids`. Studies whose folder already exists on disk are left
    untouched (no re-extraction). Anything missing is pulled out of the big
    `zip_path` archive, and only that study's files are extracted — nothing
    else. Safe to call every run: on a second run everything is already
    there so it just does a fast existence check and returns.
    """
    import zipfile

    series_root = Path(series_root)
    series_root.mkdir(parents=True, exist_ok=True)

    study_ids = set(str(s) for s in study_ids)
    already = {p.name for p in series_root.iterdir() if p.is_dir()}
    missing = study_ids - already

    if not missing:
        print(f"  All {len(study_ids)} requested studies already on disk — nothing to unzip.")
        return

    if not Path(zip_path).exists():
        print(f"  [WARN] {len(missing)} studies missing and zip not found at {zip_path} — skipping unzip.")
        return

    print(f"  {len(already)} studies already on disk, {len(missing)} need extracting from {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Archive layout is train_series/<StudyInstanceUID>/<SeriesInstanceUID>/*.dcm
        # (a "train_series" folder at the zip root, not study folders at root).
        # Find that prefix once, then pull out only the members whose study
        # folder is one we're missing.
        names = zf.namelist()
        prefix = ""
        for n in names:
            parts = n.split("/")
            if len(parts) > 1 and parts[0].lower() == "train_series":
                prefix = "train_series/"
                break

        wanted_members = [n for n in names
                          if n[len(prefix):].split("/")[0] in missing]
        print(f"  Extracting {len(wanted_members)} files for {len(missing)} studies "
              f"(archive prefix: '{prefix}')...")
        for member in tqdm(wanted_members, desc="  Unzipping"):
            # extract, then relocate out from under the "train_series/" prefix
            # so the result lands directly as series_root/<StudyInstanceUID>/...
            zf.extract(member, path=series_root)
            if prefix:
                extracted_path = series_root / member
                rel = Path(member[len(prefix):])
                target_path = series_root / rel
                target_path.parent.mkdir(parents=True, exist_ok=True)
                if extracted_path != target_path:
                    extracted_path.replace(target_path)

        # clean up the now-empty "train_series" subfolder left behind by extraction
        if prefix:
            leftover = series_root / "train_series"
            if leftover.exists():
                for root, dirs, files in os.walk(leftover, topdown=False):
                    for d in dirs:
                        try:
                            os.rmdir(os.path.join(root, d))
                        except OSError:
                            pass
                try:
                    os.rmdir(leftover)
                except OSError:
                    pass

    still_missing = missing - {p.name for p in series_root.iterdir() if p.is_dir()}
    if still_missing:
        print(f"  [WARN] {len(still_missing)} studies not found in zip either: "
              f"{sorted(still_missing)[:5]}{'...' if len(still_missing) > 5 else ''}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — SCAN DICOMS
# ══════════════════════════════════════════════════════════════════════════════
def scan_dicoms(series_root, series_csv, study_filter=None):
    """
    Scan DCMs, join with series CSV for slot metadata.
    study_filter: optional iterable of StudyInstanceUIDs to restrict the
    scan to — goes straight to each study's folder instead of walking the
    entire series_root tree, which matters a lot when series_root holds
    far more studies than we're actually training on.
    """
    series_root = Path(series_root)
    if study_filter is not None:
        study_filter = list(study_filter)
        paths = []
        for uid in study_filter:
            study_dir = series_root / str(uid)
            if study_dir.is_dir():
                paths.extend(study_dir.rglob("*.dcm"))
        print(f"  Restricting scan to {len(study_filter)} study folders → {len(paths)} DCMs")
    else:
        paths = list(series_root.rglob("*.dcm"))
    print(f"  DCM files found: {len(paths)}")

    # FIX: one header pass for everything. The old code read every header
    # here, again in detect_laterality, and a third time in sort_slices.
    # Geometry, quality fields and the image-centre X are all computed once
    # and carried in the dataframe. Drops are counted by reason, not silent.
    def _read_header(p):
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
        except Exception as e:
            return ("ERR", type(e).__name__)
        study  = str(getattr(ds, "StudyInstanceUID",  "") or "").strip()
        series = str(getattr(ds, "SeriesInstanceUID", "") or "").strip()
        if not (study and series):
            return ("NOUID", None)
        sop  = str(getattr(ds, "SOPInstanceUID", "") or "").strip()
        inst = getattr(ds, "InstanceNumber", None)
        try: inst = int(inst) if inst is not None else None
        except Exception: inst = None

        def _f(v, d=None):
            try: return float(v)
            except Exception: return d

        ipp = getattr(ds, "ImagePositionPatient", None)
        iop = getattr(ds, "ImageOrientationPatient", None)
        ps  = getattr(ds, "PixelSpacing", None)
        rws = _f(getattr(ds, "Rows", None)); cls = _f(getattr(ds, "Columns", None))
        px = py = pz = cx = None
        if ipp is not None and len(ipp) >= 3:
            px, py, pz = _f(ipp[0]), _f(ipp[1]), _f(ipp[2])
        # image-centre X in patient coords -> +X is patient LEFT
        if None not in (px, rws, cls) and iop is not None and ps is not None:
            try:
                o = [float(x) for x in iop[:6]]; s = [float(x) for x in ps[:2]]
                cx = px + o[0]*s[1]*cls/2.0 + o[3]*s[0]*rws/2.0
            except Exception:
                cx = None
        return dict(filepath=str(p), StudyInstanceUID=study,
                    SeriesInstanceUID=series, SOPInstanceUID=sop,
                    InstanceNumber=inst,
                    pos_x=px, pos_y=py, pos_z=pz, centre_x=cx,
                    Rows=rws, Columns=cls,
                    SliceThickness=_f(getattr(ds, "SliceThickness", None)),
                    SeriesDescription=str(getattr(ds, "SeriesDescription", "") or ""))

    from concurrent.futures import ThreadPoolExecutor
    print(f"  Reading {len(paths)} DICOM headers (parallel)...")
    with ThreadPoolExecutor(max_workers=int(os.environ.get("RSNA_SCAN_WORKERS", 16))) as ex:
        results = list(ex.map(_read_header, paths))

    rows, n_err, n_nouid, err_kinds = [], 0, 0, {}
    for r in results:
        if isinstance(r, dict):
            rows.append(r)
        elif r[0] == "ERR":
            n_err += 1; err_kinds[r[1]] = err_kinds.get(r[1], 0) + 1
        else:
            n_nouid += 1
    if n_err or n_nouid:
        print(f"  [DROP] unreadable={n_err} missing-UID={n_nouid} "
              f"of {len(paths)} ({100.0*(n_err+n_nouid)/max(len(paths),1):.2f}%)")
        if err_kinds:
            print(f"  [DROP] error kinds: {err_kinds}")
    else:
        print(f"  [DROP] none -- all {len(paths)} headers read cleanly")

    df = pd.DataFrame(rows)
    meta = pd.read_csv(series_csv, dtype=str)
    meta["Fluid_Sensitive"] = meta["Fluid_Sensitive"].astype(int)
    meta["Fat_Suppression"] = meta["Fat_Suppression"].astype(int)

    merged = df.merge(meta, on=["StudyInstanceUID", "SeriesInstanceUID"], how="left")

    # FIX: the old fallback set Fluid_Sensitive=0 and Fat_Suppression=0 for
    # any series absent from the CSV. A fat-sat T2 landing there was shoved
    # into a T1 slot with no warning. Now the guess is flagged so it can be
    # counted, and how many series hit this path is printed.
    merged["meta_guessed"] = False
    unmatched = merged["Anatomical_Plane"].isna()
    n_unmatched_series = merged.loc[unmatched, "SeriesInstanceUID"].nunique()
    if unmatched.sum() > 0:
        for sid, sub in merged[unmatched].groupby("SeriesInstanceUID"):
            idx0 = sub.index[0]
            plane = "Unknown"
            try:
                ds = pydicom.dcmread(merged.loc[idx0, "filepath"],
                                     stop_before_pixels=True, force=True)
                plane = _plane_from_iop(ds)
            except Exception:
                pass
            merged.loc[sub.index, "Anatomical_Plane"] = plane
            merged.loc[sub.index, "Fluid_Sensitive"]  = 0
            merged.loc[sub.index, "Fat_Suppression"]  = 0
            merged.loc[sub.index, "meta_guessed"]     = True
        print(f"  [WARN] {n_unmatched_series} series absent from series CSV -- "
              f"plane inferred from geometry, fluid/fat GUESSED as 0/0")

    n_unknown = merged.loc[merged["Anatomical_Plane"].isin(["Unknown"]) |
                           merged["Anatomical_Plane"].isna(),
                           "SeriesInstanceUID"].nunique()
    if n_unknown:
        print(f"  [WARN] {n_unknown} series have no usable plane -- "
              f"they will match no slot and be dropped")

    print(f"  Studies: {merged['StudyInstanceUID'].nunique()}")
    print(f"  Series : {merged['SeriesInstanceUID'].nunique()}")
    return merged


def _plane_from_iop(ds):
    try:
        iop = [float(x) for x in ds.ImageOrientationPatient]
        r, c = iop[:3], iop[3:]
        n = [r[1]*c[2]-r[2]*c[1], r[2]*c[0]-r[0]*c[2], r[0]*c[1]-r[1]*c[0]]
        a = [abs(x) for x in n]
        if a[0] > a[1] and a[0] > a[2]: return "Sagittal"
        if a[1] > a[0] and a[1] > a[2]: return "Coronal"
        if a[2] > a[0] and a[2] > a[1]: return "Axial"
    except Exception:
        pass
    return "Unknown"


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — BUILD SLOTS + LATERALITY
# ══════════════════════════════════════════════════════════════════════════════
def detect_laterality(dicom_df):
    """
    FIX: centre_x is now computed once inside scan_dicoms and carried in the
    dataframe. The old version re-opened every DICOM header from disk and
    iterated with iterrows() -- minutes wasted on the 58-study set, far more
    on the full 4,407. This is a groupby over a column already in memory.
    """
    if "centre_x" not in dicom_df.columns:
        raise RuntimeError("dicom_df has no centre_x column -- stale cached "
                           "train_dicom_index.csv. Delete it and re-scan.")
    cx = pd.to_numeric(dicom_df["centre_x"], errors="coerce")
    med = cx.groupby(dicom_df["StudyInstanceUID"]).median()

    result = {}
    for su, m in med.items():
        if pd.isna(m):
            result[su] = None
        else:
            result[su] = "R" if m < -LAT_OFFSET else ("L" if m > LAT_OFFSET else None)

    vals = list(result.values())
    print(f"  Laterality: L={vals.count('L')} R={vals.count('R')} "
          f"unknown={vals.count(None)} of {len(vals)} studies")
    return result


def _series_quality(series_df):
    """
    Rank a series as diagnostic (1) or junk (0), plus a tiebreak score.

    "Most slices wins" is not a quality measure: a 3-plane localizer often
    carries MORE slices than the diagnostic scan it beats. Flag the obvious
    junk, then break remaining ties on slice count.
    """
    desc  = series_df.get("SeriesDescription", pd.Series("", index=series_df.index)).fillna("").astype(str)
    thick = pd.to_numeric(series_df.get("SliceThickness"), errors="coerce")
    rows_ = pd.to_numeric(series_df.get("Rows"),    errors="coerce")
    cols_ = pd.to_numeric(series_df.get("Columns"), errors="coerce")
    nsl   = pd.to_numeric(series_df["n_slices"],    errors="coerce").fillna(0)

    is_loc   = desc.str.contains(LOCALIZER_RE, na=False)
    too_thick = thick.notna() & (thick > MAX_SLICE_THICK)
    too_few   = nsl < MIN_SERIES_SLICES
    too_small = (rows_.notna() & (rows_ < MIN_INPLANE_DIM)) | \
                (cols_.notna() & (cols_ < MIN_INPLANE_DIM))
    # a "series" whose slices span more than one plane is a 3-plane scout
    multi_plane = series_df.get("n_planes", pd.Series(1, index=series_df.index)) > 1

    junk = is_loc | too_thick | too_few | too_small | multi_plane
    return (~junk).astype(int), {
        "localizer_name": int(is_loc.sum()), "too_thick": int(too_thick.sum()),
        "too_few_slices": int(too_few.sum()), "low_res": int(too_small.sum()),
        "multi_plane": int(multi_plane.sum()),
    }


def assign_slots(dicom_df, reject_junk=True, verbose=True):
    slice_counts = dicom_df.groupby("SeriesInstanceUID").size().to_dict()
    # how many distinct planes appear within one series -> scout detector
    n_planes = (dicom_df.groupby("SeriesInstanceUID")["Anatomical_Plane"]
                .nunique().to_dict())

    series_df = (dicom_df.groupby(["StudyInstanceUID", "SeriesInstanceUID"])
                 .first().reset_index())
    series_df["n_slices"] = series_df["SeriesInstanceUID"].map(slice_counts)
    series_df["n_planes"] = series_df["SeriesInstanceUID"].map(n_planes).fillna(1)

    quality, junk_counts = _series_quality(series_df)
    series_df["quality"] = quality if reject_junk else 1
    if verbose:
        n_junk = int((series_df["quality"] == 0).sum())
        print(f"  Quality filter: {n_junk}/{len(series_df)} series flagged as junk "
              f"({100.0*n_junk/max(len(series_df),1):.1f}%) -> {junk_counts}")

    rows, n_orphan = [], 0
    for study, grp in series_df.groupby("StudyInstanceUID"):
        claimed = set()
        for slot_name, plane, fluid, fat in SLOTS:
            mask = ((grp["Anatomical_Plane"] == plane) &
                    (grp["Fluid_Sensitive"].fillna(0).astype(int) == fluid) &
                    (grp["Fat_Suppression"].fillna(0).astype(int) == fat))
            # FIX: diagnostic series outrank junk; slice count only breaks ties
            cands = grp[mask].sort_values(["quality", "n_slices"], ascending=[False, False])
            if len(cands) == 0:
                rows.append({"StudyInstanceUID": study, "SeriesInstanceUID": "",
                             "slot_name": slot_name,
                             "Anatomical_Plane": plane,   # FIX: carry the plane
                             "n_slices": 0, "presence_mask": 0,
                             "alt_series": ""})
            else:
                best = cands.iloc[0]
                claimed.update(cands["SeriesInstanceUID"].tolist())
                # FIX: runners-up are no longer discarded -- they are recorded
                # so embed_slots can pool them into the same slice budget.
                alts = [s for s in cands["SeriesInstanceUID"].tolist()[1:]]
                rows.append({"StudyInstanceUID": study,
                             "SeriesInstanceUID": best["SeriesInstanceUID"],
                             "slot_name": slot_name,
                             "Anatomical_Plane": str(best["Anatomical_Plane"]),
                             "n_slices": int(best["n_slices"]),
                             "presence_mask": 1,
                             "alt_series": "|".join(alts)})
        n_orphan += int((~grp["SeriesInstanceUID"].isin(claimed)).sum())

    if verbose and n_orphan:
        print(f"  [WARN] {n_orphan} series matched no slot in any study "
              f"({100.0*n_orphan/max(len(series_df),1):.1f}%)")
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — MEDSIGLIP EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════════
def sort_slices(paths, pos_lookup=None):
    """
    FIX: pos_lookup maps filepath -> (x, y, z, InstanceNumber), built once in
    scan_dicoms. The old code re-read every DICOM header here -- a third full
    pass over the corpus after scan_dicoms and detect_laterality. Falls back
    to reading from disk only if the lookup is missing an entry.
    """
    meta = []
    for p in paths:
        rec = pos_lookup.get(str(p)) if pos_lookup else None
        if rec is not None:
            x, y, z, inst = rec
            pos = None
            if x is not None and np.isfinite([x, y, z]).all():
                pos = np.array([x, y, z], dtype=float)
            meta.append((p, pos, inst))
            continue
        try:
            ds   = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
            ipp  = getattr(ds, "ImagePositionPatient", None)
            inst = getattr(ds, "InstanceNumber", None)
            pos  = None
            if ipp is not None:
                coords = np.array([float(x) for x in ipp[:3]])
                if np.isfinite(coords).all(): pos = coords
            meta.append((p, pos, inst))
        except Exception:
            meta.append((p, None, None))

    positioned = [(p, pos, inst) for p, pos, inst in meta if pos is not None]
    if len(positioned) >= max(2, int(0.8 * len(meta))):
        xyz  = np.stack([pos for _, pos, _ in positioned])
        axis = int(np.argmax(np.ptp(xyz, axis=0)))
        spare = float(np.nanmedian(xyz[:, axis]))
        meta.sort(key=lambda x: (float(x[1][axis]) if x[1] is not None else spare,
                                  float(x[2]) if x[2] is not None else float("inf")))
    else:
        meta.sort(key=lambda x: (float(x[2]) if x[2] is not None else float("inf"),))
    return [p for p, _, _ in meta]


def select_band(paths):
    """
    FIX: the old version always cut the outer 40%, even when there were only
    10 slices to begin with -- leaving 6 and throwing away usable anatomy for
    no gain. Banding exists to drop edge slices when there is a surplus; if
    the series already has no surplus, keep all of it.
    """
    n = len(paths)
    if n <= MAX_SLICES:
        return paths                      # no surplus -- keep everything
    lo = int(np.floor(n * SLICE_BAND[0]))
    hi = int(np.ceil(n  * SLICE_BAND[1]))
    band = paths[lo:hi]
    if len(band) < MAX_SLICES:            # banding overshot -- widen back out
        band = paths
    if len(band) <= MAX_SLICES:
        return band
    idx = np.unique(np.round(np.linspace(0, len(band)-1, MAX_SLICES)).astype(int))
    return [band[i] for i in idx]


def normalise_laterality(imgs, plane, lat):
    if lat != "R": return imgs
    if plane in ("Coronal", "Axial"):
        return [img.transpose(Image.FLIP_LEFT_RIGHT) for img in imgs]
    return imgs[::-1]


def dicom_to_array(path, want_spacing=False):
    """Raw rescaled float array for one slice, polarity corrected.

    want_spacing also returns the in-plane PixelSpacing in mm, which the
    millimetre crop needs and which is otherwise thrown away here.
    """
    ds  = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)
    if str(getattr(ds, "PhotometricInterpretation", "")).strip() == "MONOCHROME1":
        # FIX #7: invert against the declared bit depth, not the observed max.
        # arr.max() makes the inversion image-dependent, so two slices in one
        # series invert onto different scales and per-series windowing then
        # normalises an inconsistent mix.
        try:
            _bits = int(getattr(ds, "BitsStored", 0) or 0)
        except Exception:
            _bits = 0
        arr = ((2 ** _bits - 1) - arr) if _bits > 0 else (arr.max() - arr)
    slope     = float(getattr(ds, "RescaleSlope",     1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    out = arr * slope + intercept
    if not want_spacing:
        return out
    ps = None
    try:
        v = getattr(ds, "PixelSpacing", None)
        if v is not None and len(v) >= 2:
            ps = (float(v[0]) + float(v[1])) / 2.0     # row/col spacing, mm
    except Exception:
        ps = None
    return out, ps


def mm_crop(a, spacing, crop_mm=None):
    """Centre-crop a fixed physical box, then leave resizing to the processor.

    spacing is mm per pixel, so crop_mm / spacing is the box in pixels. A slice
    whose field of view is already smaller than the box is returned untouched
    rather than padded -- padding would invent tissue at the edge of the frame,
    and the edge of this crop is where the popliteal fossa sits.
    """
    crop_mm = CROP_MM if crop_mm is None else crop_mm
    if not crop_mm or spacing is None or not np.isfinite(spacing) or spacing <= 0:
        return a
    px = int(round(crop_mm / spacing))
    h, w = a.shape[:2]
    if px >= min(h, w) or px < 32:
        return a
    cy, cx = h // 2, w // 2
    y0, x0 = max(0, cy - px // 2), max(0, cx - px // 2)
    return a[y0:y0 + px, x0:x0 + px]


def arrays_to_pils(arrays):
    """
    FIX: window once per SERIES, not per slice.

    The old dicom_to_pil computed 1st/99th percentiles on each slice
    independently, so a slice full of bright effusion was rescaled until the
    fluid looked like ordinary tissue in a neighbouring slice. Relative
    brightness across the series -- which is exactly the signal for Effusion
    and Synovitis -- was destroyed. Percentiles are now taken over the whole
    stack and one window is applied to every slice.
    """
    if not arrays:
        return []
    flat = np.concatenate([a.ravel() for a in arrays])
    # FIX #16: 1/99 clipped 2% of pixels; in a knee with a small bright
    # effusion the effusion can BE the top 1%. 0.5/99.5 still removes hot
    # pixels and metal artefact but keeps small bright findings intact.
    lo, hi = np.percentile(flat, WINDOW_PCT)
    if hi <= lo:
        lo, hi = float(flat.min()), float(flat.max())
    u8s = []
    for a in arrays:
        if hi <= lo:
            u8s.append(np.zeros(a.shape, dtype=np.uint8))
        else:
            u8s.append((np.clip((a - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8))

    if RGB_MODE != "neighbours" or len(u8s) < 2:
        return [Image.fromarray(u) .convert("RGB") for u in u8s]

    # Channels are slice n-1, n, n+1. Neighbours are only stacked when they are
    # the same shape -- a series whose slices differ in size would otherwise
    # silently mis-register the channels against each other, which is worse
    # than having no depth context at all.
    out, N = [], len(u8s)
    for i, mid in enumerate(u8s):
        prv, nxt = u8s[max(0, i - 1)], u8s[min(N - 1, i + 1)]
        if prv.shape != mid.shape or nxt.shape != mid.shape:
            out.append(Image.fromarray(mid).convert("RGB"))
        else:
            out.append(Image.fromarray(np.stack([prv, mid, nxt], axis=-1), mode="RGB"))
    return out


def dicom_to_pil(path):
    """Single-slice path, kept for callers outside embed_slots."""
    return arrays_to_pils([dicom_to_array(path)])[0]


def load_medsiglip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading MedSigLIP | device={device}")
    processor = AutoProcessor.from_pretrained(str(MODEL_PATH))
    model = AutoModel.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()
    return processor, model, device


@torch.inference_mode()
def encode_images(images, processor, model, device):
    feats = []
    for i in range(0, len(images), BATCH_SIZE):
        batch  = images[i:i + BATCH_SIZE]
        inputs = processor(images=batch, return_tensors="pt")
        pixels = inputs["pixel_values"].to(device)
        if getattr(device, "type", str(device)) == "cuda":
            pixels = pixels.to(dtype=torch.float16)
        if PATCH_POOL == "pooled":
            out = model.get_image_features(pixel_values=pixels)
            if not torch.is_tensor(out):
                if hasattr(out, "pooler_output"):       out = out.pooler_output
                elif hasattr(out, "image_embeds"):      out = out.image_embeds
                elif hasattr(out, "last_hidden_state"): out = out.last_hidden_state.mean(1)
            out = F.normalize(out.float(), dim=-1)
        else:
            # Reach into the vision tower for the patch grid. get_image_features
            # discards it, and the patch grid is where a focal finding lives.
            vt = getattr(model, "vision_model", model)
            hs = vt(pixel_values=pixels).last_hidden_state      # [B, 1+P, D] or [B, P, D]
            if hs.shape[1] % 2 == 1:                            # CLS present
                cls, patch = hs[:, 0], hs[:, 1:]
            else:                                               # no CLS token
                cls, patch = hs.mean(1), hs
            parts = [cls, patch.mean(1)]
            if PATCH_POOL == "cls_mean_focal":
                k = max(1, patch.shape[1] // 8)
                parts.append(patch.topk(k, dim=1).values.mean(1))
            out = torch.cat([F.normalize(p.float(), dim=-1) for p in parts], dim=-1)
        if out.shape[-1] != EMBED_DIM:
            raise RuntimeError(f"encoder returned width {out.shape[-1]}, "
                               f"EMBED_DIM is {EMBED_DIM}")
        feats.append(out.cpu())
    return torch.cat(feats, dim=0)


def embed_slots(slots_df, dicom_df, processor, model, device,
                lat_map, out_dir, force=False, use_alts=True):
    series_to_files = (dicom_df.groupby("SeriesInstanceUID")["filepath"]
                       .apply(list).to_dict())

    # FIX: position lookup so sort_slices never re-reads headers from disk
    pos_lookup = {}
    if {"pos_x", "pos_y", "pos_z"} <= set(dicom_df.columns):
        _sub = dicom_df[["filepath", "pos_x", "pos_y", "pos_z", "InstanceNumber"]]
        for fp, x, y, z, inst in _sub.itertuples(index=False):
            try: inst = int(inst) if pd.notna(inst) else None
            except Exception: inst = None
            pos_lookup[str(fp)] = (
                float(x) if pd.notna(x) else None,
                float(y) if pd.notna(y) else None,
                float(z) if pd.notna(z) else None, inst)

    # FIX: the old code read Anatomical_Plane off slots_df, but assign_slots
    # never wrote that column -- so plane was ALWAYS "Unknown" and every right
    # knee fell through to the sagittal branch of normalise_laterality. Coronal
    # and axial right knees were never mirrored, leaving medial/lateral swapped
    # for four of the twelve targets. assign_slots now emits the column; this
    # is a hard check so the bug cannot come back silently.
    if "Anatomical_Plane" not in slots_df.columns:
        raise RuntimeError("slots_df has no Anatomical_Plane column -- stale "
                           "train_slots_cache.pkl. Delete it and rebuild slots.")

    present = slots_df[slots_df["presence_mask"] == 1].copy()
    index_rows = []
    done = failed = skipped = n_pooled = n_degraded = 0

    for _, row in tqdm(present.iterrows(), total=len(present), desc="  Embedding"):
        study  = str(row["StudyInstanceUID"])
        series = str(row["SeriesInstanceUID"])
        slot   = str(row["slot_name"])
        plane  = str(row.get("Anatomical_Plane", "Unknown"))
        lat    = lat_map.get(study)

        out_path = out_dir / study / f"{series}__{slot}.pt"
        if out_path.exists() and not force:
            skipped += 1
            index_rows.append({"StudyInstanceUID": study, "SeriesInstanceUID": series,
                                "slot_name": slot, "embedding_file": str(out_path),
                                "presence_mask": 1})
            continue

        # FIX: runners-up for this slot are no longer thrown away. The slice
        # budget is shared between the winning series and its alternates, so
        # the same MAX_SLICES images are drawn from more of the study. Costs
        # nothing extra to encode.
        srcs = [series]
        if use_alts:
            alt = str(row.get("alt_series", "") or "")
            srcs += [s for s in alt.split("|") if s]

        per_src = max(1, MAX_SLICES // max(len(srcs), 1))
        paths = []
        for si, s in enumerate(srcs):
            sp = [Path(p) for p in series_to_files.get(s, []) if Path(p).is_file()]
            if not sp:
                continue
            sp = sort_slices(sp, pos_lookup=pos_lookup)
            budget = MAX_SLICES if len(srcs) == 1 else per_src
            n_sp = len(sp)
            if n_sp > budget:
                lo = int(np.floor(n_sp * SLICE_BAND[0]))
                hi = int(np.ceil(n_sp  * SLICE_BAND[1]))
                band = sp[lo:hi] if (hi - lo) >= budget else sp
                idx = np.unique(np.round(np.linspace(0, len(band)-1, budget)).astype(int))
                sp = [band[i] for i in idx]
            paths.extend(sp)
        # FIX #5: the old test compared the final slice count against the
        # winner's TOTAL file count, so a 40-file winner capped to 20 never
        # registered even when pooling really happened.
        if len(srcs) > 1 and len(paths) > 0:
            n_pooled += 1
        if not paths:
            paths = sort_slices(
                [Path(p) for p in series_to_files.get(series, []) if Path(p).is_file()],
                pos_lookup=pos_lookup)
            paths = select_band(paths)

        # FIX: window per series, not per slice (see arrays_to_pils)
        # FIX #6: per-slice decode failures were swallowed silently, so a slot
        # where 18 of 20 slices failed still wrote an embedding marked present.
        arrays, n_bad = [], 0
        for p in paths:
            try:
                a, ps = dicom_to_array(p, want_spacing=True)
                arrays.append(mm_crop(a, ps))
            except Exception:
                n_bad += 1
        if n_bad and (n_bad / max(len(paths), 1)) > 0.20:
            print(f"\n  [WARN] {study[:16]}/{slot}: {n_bad}/{len(paths)} slices "
                  f"failed to decode")
            n_degraded += 1
        images = arrays_to_pils(arrays)

        if not images:
            failed += 1
            # FIX: still record the slot so the embedding index and the slot
            # table cannot silently disagree about what is present.
            index_rows.append({"StudyInstanceUID": study, "SeriesInstanceUID": series,
                               "slot_name": slot, "embedding_file": "",
                               "presence_mask": 0})
            continue

        images = normalise_laterality(images, plane, lat)

        try:
            feats = encode_images(images, processor, model, device)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"embeddings": feats, "slot_name": slot,
                        "study_uid": study, "series_uid": series,
                        "laterality": lat, "plane": plane,
                        "n_slices": len(images)}, out_path)
            done += 1
            index_rows.append({"StudyInstanceUID": study, "SeriesInstanceUID": series,
                                "slot_name": slot, "embedding_file": str(out_path),
                                "presence_mask": 1})
        except Exception as e:
            print(f"\n  [WARN] {study[:20]}/{slot}: {e}")
            failed += 1
            index_rows.append({"StudyInstanceUID": study, "SeriesInstanceUID": series,
                               "slot_name": slot, "embedding_file": "",
                               "presence_mask": 0})

    print(f"  Embedded={done} Skipped={skipped} Failed={failed} "
          f"SlotsPooledFromMultipleSeries={n_pooled} Degraded={n_degraded}")
    return pd.DataFrame(index_rows)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — MODEL
# ══════════════════════════════════════════════════════════════════════════════
def fingerprint(model, dev=None):
    """Model output on a fixed synthetic bag — a portable identity for the map.

    Seeded rather than read, so it is identical on any machine, and pushed
    through the whole forward path (projection, positional encoding, both
    attention levels, the heads). Any of those differing moves the value.
    """
    dev = dev or next(model.parameters()).device
    g = torch.Generator().manual_seed(1234)
    x = torch.randn(2 * N_SLOT, EMBED_DIM, generator=g).to(dev)
    idx = torch.arange(N_SLOT, device=dev).repeat_interleave(2)
    mask = torch.ones(N_SLOT, device=dev)
    mask[-1] = 0.0                      # exercise the masked branch of the softmax
    was = model.training
    model.eval()
    with torch.no_grad():
        out = model(x, mask, idx).float().cpu().numpy()
    if was:
        model.train()
    return out


def check_fingerprint(model, expected, tag=""):
    if expected is None:
        return
    got = fingerprint(model)
    d = float(np.abs(np.asarray(got) - np.asarray(expected)).max())
    if d > FINGERPRINT_TOL:
        raise RuntimeError(
            f"fingerprint mismatch{(' [' + tag + ']') if tag else ''}: max diff "
            f"{d:.2e} > {FINGERPRINT_TOL:.0e}. The weights load but do not compute "
            f"what they computed when fitted — check PATCH_POOL, MAX_SLICES, "
            f"SLICE_BAND, WINDOW_PCT, SLOT_POOL and the Run A switches.")
    print(f"  fingerprint ok{(' [' + tag + ']') if tag else ''} (max diff {d:.2e})")


class SlotAttentionModel(nn.Module):
    """
    RUN A changes, all in this class. No re-encode needed.

    1. TWO-LEVEL ATTENTION.
       The old forward ran ONE softmax over every slice of every slot at once
       (`dim=1` spans the whole concatenated stack). Attention mass therefore
       tracked SLICE COUNT: a study with 20 sagittal and 5 coronal slices gave
       sagittal 4x the mass regardless of which view was informative, and a
       focal ACL tear visible on 3 slices competed against ~100 slices from
       every other slot at once. Raising MAX_SLICES made that strictly worse,
       which is the most likely reason ACL fell to 0.762 when slices went
       12 -> 20.

       Now: pool WITHIN each slot (softmax over that slot's own slices), then
       pool ACROSS the six slots. Each slot contributes exactly one vector, so
       slice count no longer buys influence and the anatomical prior operates
       where it was always meant to -- at the slot level.

    2. LEARNABLE PRIOR.
       SLOT_PRIOR is hand-written anatomy that was never validated, and the
       AX_T1 row is a guess about a slot that did not exist last week. It is
       now an nn.Parameter initialised to those values: the model starts from
       the anatomy and corrects it from data. Still additive-before-softmax,
       so a disease can still draw on a non-preferred view -- it is a lean,
       not a filter.

    3. SLICE POSITION ENCODING.
       normalise_laterality reverses slice order for sagittal right knees, but
       a permutation-invariant sum ignores order entirely, so that branch was
       dead code and ~40% of slots were never laterality-normalised. A learned
       positional embedding over normalised slice depth makes order matter, so
       the reversal finally does something.
    """
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(EMBED_DIM),
            nn.Linear(EMBED_DIM, PROJ_DIM),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.att = nn.Linear(PROJ_DIM, len(TARGETS), bias=False)          # within-slot
        self.slot_att = nn.Linear(PROJ_DIM, len(TARGETS), bias=False)     # across-slot
        self.heads = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(PROJ_DIM), nn.Linear(PROJ_DIM, 64),
                          nn.GELU(), nn.Dropout(0.15), nn.Linear(64, 1))
            for _ in TARGETS
        ])

        prior = torch.zeros(len(TARGETS), N_SLOT)
        for t, target in enumerate(TARGETS):
            for sl, val in enumerate(SLOT_PRIOR[target]):
                prior[t, sl] = val * PRIOR_STRENGTH
        if LEARNABLE_PRIOR:
            self.slot_prior = nn.Parameter(prior)
        else:
            self.register_buffer("slot_prior", prior)

        # depth -> PROJ_DIM, 32 buckets over the normalised 0..1 slice position
        self.n_pos = 32
        self.pos_emb = nn.Embedding(self.n_pos, PROJ_DIM)
        nn.init.zeros_(self.pos_emb.weight)   # starts as a no-op

    def _add_posenc(self, h, slot_indices):
        """Depth within each slot, bucketed to 0..n_pos-1.

        Vectorised. The original looped over slots and called int(m.sum()) per
        slot, which pulls a scalar off the GPU and stalls the pipeline on every
        sample -- it is also what broke torch.compile, and it is why Stage A
        folds went from ~25 minutes to ~72. Rank-within-slot is computed here
        with a scatter and a cumulative sum, entirely on device.
        """
        # Raw counts, NOT clamped: an absent slot must contribute zero rows to
        # the offsets, otherwise every slot after it is shifted and the depth
        # ranks come out wrong for exactly the studies that are missing a slot.
        counts = torch.bincount(slot_indices, minlength=N_SLOT)
        # slot_indices is sorted ascending by construction (rows are appended
        # slot by slot), so a row's rank within its slot is its absolute
        # position minus where that slot starts.
        order = torch.arange(h.shape[0], device=h.device)
        starts = torch.cumsum(
            torch.cat([torch.zeros(1, dtype=counts.dtype, device=h.device),
                       counts[:-1]]), 0)
        rank = order - starts[slot_indices]
        denom = (counts[slot_indices] - 1).clamp(min=1).to(h.dtype)
        frac = rank.to(h.dtype) / denom
        frac = torch.where(counts[slot_indices] > 1, frac,
                           torch.zeros_like(frac))
        pos = (frac * (self.n_pos - 1)).round().long().clamp(0, self.n_pos - 1)
        return h + self.pos_emb(pos)

    def forward(self, x, mask, slot_indices):
        h = self.proj(x)
        if SLICE_POSENC:
            h = self._add_posenc(h, slot_indices)

        absent = (mask[slot_indices] < 0.5)

        if not TWO_LEVEL_ATT:
            scores = self.att(h).T + self.slot_prior[:, slot_indices]
            scores = scores.masked_fill(absent.unsqueeze(0), -1e4)
            w = torch.softmax(scores, dim=1)
            return torch.stack([self.heads[t]((w[t, :, None] * h).sum(0)).squeeze()
                                for t in range(len(TARGETS))])

        T = len(TARGETS)
        raw = self.att(h).T                       # [T, n_slices]
        raw = raw.masked_fill(absent.unsqueeze(0), -1e4)

        present = [int(sl) for sl in torch.unique(slot_indices)
                   if mask[int(sl)] >= 0.5]
        if not present:                            # nothing usable in this study
            pooled_all = h.mean(0, keepdim=True).expand(T, -1)
            return torch.stack([self.heads[t](pooled_all[t]).squeeze()
                                for t in range(T)])

        # ── level 1: within each slot, over that slot's own slices ──
        # Vectorised over slots. The loop below used to run once per slot per
        # sample with a host sync inside it; this does all six at once with a
        # masked softmax and two scatter-adds, and never leaves the device.
        sid = torch.tensor(present, device=h.device, dtype=torch.long)
        S = sid.shape[0]
        # [S, n] membership, so every reduction below is one op over all slots
        memb = (slot_indices.unsqueeze(0) == sid.unsqueeze(1))       # [S, n]

        # softmax over each slot's own slices: mask the rest to -inf per slot
        sc = raw.unsqueeze(1).expand(T, S, -1).masked_fill(
            ~memb.unsqueeze(0), float("-inf"))                       # [T, S, n]
        w = torch.softmax(sc, dim=2)
        w = torch.nan_to_num(w, nan=0.0)          # a slot with no rows -> zeros
        att = torch.einsum("tsn,nd->tsd", w, h)                      # [T, S, D]

        if SLOT_POOL in ("max", "mix"):
            big = h.unsqueeze(0).masked_fill(~memb.unsqueeze(-1), -1e4)
            mx = big.max(dim=1).values                               # [S, D]
            mx = mx.unsqueeze(0).expand(T, -1, -1)
            V = mx if SLOT_POOL == "max" else 0.5 * att + 0.5 * mx
        else:
            V = att

        # ── level 2: across slots, prior applied here ──
        sc = torch.einsum("tsd,dt->ts", V, self.slot_att.weight.T)
        sc = sc + self.slot_prior[:, sid]
        wS = torch.softmax(sc, dim=1)                      # [T, n_present]
        pooled = torch.einsum("ts,tsd->td", wS, V)         # [T, PROJ]

        return torch.stack([self.heads[t](pooled[t]).squeeze()
                            for t in range(T)])


def load_embedding(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "embeddings" in obj: x = obj["embeddings"]
    elif torch.is_tensor(obj): x = obj
    else:
        cands = [v for v in (obj.values() if isinstance(obj, dict) else [])
                 if torch.is_tensor(v)]
        if not cands: raise ValueError(f"No tensor in {path}")
        x = cands[0]
    x = x.float()
    if x.ndim == 1: x = x.unsqueeze(0)
    if x.ndim != 2 or x.shape[1] != EMBED_DIM:
        raise ValueError(f"{path}: expected [N,{EMBED_DIM}]")
    return x


class StudyDataset:
    def __init__(self, emb_df, labels_df):
        self.emb_df = emb_df
        self.labels = labels_df.set_index("StudyInstanceUID")
        self.ids    = sorted(emb_df["StudyInstanceUID"].unique())

    def __len__(self): return len(self.ids)

    def get(self, i):
        study = self.ids[i]
        rows  = self.emb_df[self.emb_df["StudyInstanceUID"] == study]
        slot_to_file = {r["slot_name"]: r["embedding_file"]
                        for _, r in rows.iterrows() if str(r["presence_mask"]) in ("1", "1.0", "True", "true")}
        tensors, slot_indices, mask = [], [], torch.zeros(N_SLOT)
        for s_idx, slot_name in enumerate(SLOT_NAMES):
            if slot_name in slot_to_file:
                try:
                    x = load_embedding(slot_to_file[slot_name])
                    tensors.append(x)
                    slot_indices.extend([s_idx] * len(x))
                    mask[s_idx] = 1.0
                except Exception as e:
                    print(f"  [WARN] {study[:20]}/{slot_name}: {e}")
        if not tensors:
            tensors = [torch.zeros(1, EMBED_DIM)]
            slot_indices = [0]
        x   = torch.cat(tensors, dim=0)
        idx = torch.tensor(slot_indices, dtype=torch.long)
        y   = torch.tensor(self.labels.loc[study, TARGETS].astype(float).values,
                           dtype=torch.float32)
        return study, x, mask, idx, y


def load_state_dict_safe(model, state_dict):
    """
    Load a state_dict into model, transparently handling the '_orig_mod.'
    prefix that torch.compile() adds to every parameter name. Without this,
    loading a compiled model's weights into a fresh (uncompiled) model — or
    vice versa — fails with "Missing/Unexpected key(s)" even though the
    underlying weights are identical.
    """
    sd = state_dict
    model_keys = set(model.state_dict().keys())
    if not any(k in model_keys for k in sd.keys()):
        # keys don't match at all — try stripping/adding the compile prefix
        if all(k.startswith("_orig_mod.") for k in sd.keys()):
            sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
        elif not any(k.startswith("_orig_mod.") for k in sd.keys()):
            # model is compiled but checkpoint isn't — add the prefix
            if all(("_orig_mod." + k) in model_keys for k in list(sd.keys())[:1]):
                sd = {"_orig_mod." + k: v for k, v in sd.items()}
    model.load_state_dict(sd)
    return model


def auc_mean(y_true, y_pred):
    """
    AUC averaged over targets. y_true may contain continuous weak-label
    scores (from the report parser) instead of strict 0/1 — sklearn's
    roc_auc_score requires binary ground truth, so we threshold at 0.5
    to get a clean binary label for the AUC check. This only affects
    which epoch gets picked as "best" during weak-label pretraining;
    Stage B (fine-tuning on the 58 real studies) always uses exact 0/1
    ground truth, so its reported AUC is unaffected by this.
    """
    y_true_bin = (y_true >= 0.5).astype(np.float32)
    vals = []
    for i in range(len(TARGETS)):
        if len(np.unique(y_true_bin[:, i])) > 1:
            vals.append(roc_auc_score(y_true_bin[:, i], y_pred[:, i]))
    return float(np.mean(vals)) if vals else float("nan")


def train_fold(train_ds, val_ds, device, epochs):
    model   = SlotAttentionModel().to(device)
    # torch.compile gives ~15% speedup on PyTorch 2.0+ (safe, no result change)
    try:
        model = torch.compile(model)
    except Exception:
        pass  # older PyTorch — skip compile
    opt     = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler  = torch.amp.GradScaler("cuda", enabled=device.type=="cuda")
    yy      = np.vstack([train_ds.labels.loc[s, TARGETS].astype(float).values
                         for s in train_ds.ids])
    pos     = yy.sum(axis=0)
    pw      = np.maximum((len(yy) - pos) / np.maximum(pos, 1), 1.0).astype(np.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=device))
    best_state, best_auc = None, -np.inf

    for epoch in range(epochs):
        model.train()
        losses = []
        for i in np.random.permutation(len(train_ds)):
            _, x, mask, idx, y = train_ds.get(i)
            x, mask, idx, y = x.to(device, non_blocking=True), mask.to(device, non_blocking=True), idx.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=device.type=="cuda"):
                loss = loss_fn(model(x, mask, idx), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.item()))

        model.eval()
        Y, P = [], []
        with torch.no_grad():
            for i in range(len(val_ds)):
                _, x, mask, idx, y = val_ds.get(i)
                P.append(torch.sigmoid(model(x.to(device), mask.to(device),
                                             idx.to(device))).cpu().numpy())
                Y.append(y.numpy())
        auc = auc_mean(np.vstack(Y), np.vstack(P))
        print(f"  epoch {epoch+1:02d}  loss={np.mean(losses):.5f}  val_auc={auc:.5f}")
        if not np.isnan(auc) and auc > best_auc:
            best_auc   = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state: model.load_state_dict(best_state)
    return model


def finetune_fold(model, train_ds, val_ds, device, epochs, lr=2e-5):
    """
    Same loop as train_fold, but takes an already-initialized model (e.g.
    Stage A weak-label weights) and fine-tunes it with a smaller LR instead
    of training from scratch. Used for Stage B (58 real labels).
    """
    opt     = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler  = torch.amp.GradScaler("cuda", enabled=device.type=="cuda")
    yy      = np.vstack([train_ds.labels.loc[s, TARGETS].astype(float).values
                         for s in train_ds.ids])
    pos     = yy.sum(axis=0)
    pw      = np.maximum((len(yy) - pos) / np.maximum(pos, 1), 1.0).astype(np.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=device))
    best_state, best_auc = None, -np.inf

    for epoch in range(epochs):
        model.train()
        losses = []
        for i in np.random.permutation(len(train_ds)):
            _, x, mask, idx, y = train_ds.get(i)
            x, mask, idx, y = x.to(device, non_blocking=True), mask.to(device, non_blocking=True), idx.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=device.type=="cuda"):
                loss = loss_fn(model(x, mask, idx), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.item()))

        model.eval()
        Y, P = [], []
        with torch.no_grad():
            for i in range(len(val_ds)):
                _, x, mask, idx, y = val_ds.get(i)
                P.append(torch.sigmoid(model(x.to(device), mask.to(device),
                                             idx.to(device))).cpu().numpy())
                Y.append(y.numpy())
        auc = auc_mean(np.vstack(Y), np.vstack(P))
        print(f"  [finetune] epoch {epoch+1:02d}  loss={np.mean(losses):.5f}  val_auc={auc:.5f}")
        if not np.isnan(auc) and auc > best_auc:
            best_auc   = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state: model.load_state_dict(best_state)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
class InferDataset:
    def __init__(self, emb_df):
        self.emb_df = emb_df
        self.ids    = sorted(emb_df["StudyInstanceUID"].unique())

    def __len__(self): return len(self.ids)

    def get(self, i):
        study = self.ids[i]
        rows  = self.emb_df[self.emb_df["StudyInstanceUID"] == study]
        slot_to_file = {r["slot_name"]: r["embedding_file"]
                        for _, r in rows.iterrows() if str(r["presence_mask"]) in ("1", "1.0", "True", "true")}
        tensors, slot_indices, mask = [], [], torch.zeros(N_SLOT)
        for s_idx, slot_name in enumerate(SLOT_NAMES):
            if slot_name in slot_to_file:
                try:
                    x = load_embedding(slot_to_file[slot_name])
                    tensors.append(x)
                    slot_indices.extend([s_idx] * len(x))
                    mask[s_idx] = 1.0
                except Exception:
                    pass
        if not tensors:
            tensors = [torch.zeros(1, EMBED_DIM)]
            slot_indices = [0]
        return study, torch.cat(tensors, dim=0), mask, torch.tensor(slot_indices, dtype=torch.long)


def run_inference(emb_df, model_paths, device):
    ds      = InferDataset(emb_df)
    all_preds = np.zeros((len(ds), len(TARGETS)), dtype=np.float32)

    for mp in model_paths:
        model = SlotAttentionModel().to(device)
        ckpt  = torch.load(mp, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        preds = np.zeros((len(ds), len(TARGETS)), dtype=np.float32)
        with torch.no_grad():
            for i in range(len(ds)):
                study, x, mask, idx = ds.get(i)
                logits = model(x.to(device), mask.to(device), idx.to(device))
                preds[i] = torch.sigmoid(logits).cpu().numpy()
        all_preds += preds / len(model_paths)
        print(f"  Applied: {Path(mp).name}")

    study_ids = ds.ids
    return study_ids, all_preds


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# XGBOOST STACKING LAYER
#
# Second, structurally-different model trained on the SAME cached MedSigLIP
# embeddings the neural attention model already uses (no re-encoding needed).
# Trees make different mistakes than the attention model, so blending the
# two typically adds a small but real AUC gain from diversity alone.
#
# Pipeline:
#   pooled embedding per study (mean over all present slots/slices)
#     -> PCA to PCA_COMPONENTS dims (fit once on the Stage A pool)
#     -> + slot presence mask
#     -> Stage A: 12 per-disease XGBoost models, 5-fold CV on weak+real pool
#     -> Stage A out-of-fold predictions become extra meta-features
#     -> Stage B: 12 per-disease XGBoost models, 5-fold CV on the 58 real
#        studies only (same folds as the neural net's Stage B loop)
#     -> blend weight (alpha) picked automatically from OOF AUC
#
# Every block below is wrapped so that any failure (missing package,
# degenerate fold, etc.) disables the stacking layer and falls back to
# NN-only predictions — this addition can only help or do nothing, never
# make the submission worse than before.
# ══════════════════════════════════════════════════════════════════════════════
XGB_AVAILABLE = True
try:
    import xgboost as xgb
except ImportError:
    try:
        print("Installing xgboost...")
        subprocess.run(["pip", "install", "xgboost", "-q"], check=True)
        import xgboost as xgb
    except Exception as e:
        print(f"[WARN] xgboost unavailable ({e}) — stacking layer disabled.")
        XGB_AVAILABLE = False

PCA_COMPONENTS = 64
XGB_PARAMS = dict(
    max_depth=3, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.7, min_child_weight=3,
    reg_lambda=2.0, reg_alpha=0.5, eval_metric="auc",
    tree_method="hist", verbosity=0,
)
XGB_N_ESTIMATORS = 300
ALPHA_CANDIDATES = [1.0, 0.85, 0.70, 0.55, 0.40]   # 1.0 = NN only, fallback-safe


def study_pooled_embedding(study, emb_df):
    """
    Mean-pool ALL slice embeddings for one study across all present slots.
    Reuses the exact same cached .pt files the neural net loads — zero
    extra MedSigLIP forward passes needed.
    Returns (EMBED_DIM,) vector + (N_SLOT,) presence mask.
    """
    rows = emb_df[emb_df["StudyInstanceUID"] == study]
    slot_to_file = {r["slot_name"]: r["embedding_file"]
                    for _, r in rows.iterrows() if str(r["presence_mask"]) in ("1", "1.0", "True", "true")}
    tensors = []
    mask = np.zeros(N_SLOT, dtype=np.float32)
    for s_idx, slot_name in enumerate(SLOT_NAMES):
        if slot_name in slot_to_file:
            try:
                tensors.append(load_embedding(slot_to_file[slot_name]))
                mask[s_idx] = 1.0
            except Exception:
                pass
    if not tensors:
        return np.zeros(EMBED_DIM, dtype=np.float32), mask
    return torch.cat(tensors, dim=0).mean(dim=0).numpy(), mask


def build_tabular_matrix(study_ids, emb_df):
    """[n, EMBED_DIM] pooled embeddings + [n, N_SLOT] presence masks, in order."""
    feats, masks = [], []
    for s in study_ids:
        f, m = study_pooled_embedding(s, emb_df)
        feats.append(f)
        masks.append(m)
    return np.stack(feats).astype(np.float32), np.stack(masks).astype(np.float32)


def fit_pca(X, n_components=PCA_COMPONENTS):
    n_components = max(1, min(n_components, X.shape[0] - 1, X.shape[1]))
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X)
    return pca


def make_features(raw_embed, mask, pca, extra=None):
    """PCA-reduce pooled embeddings, concat presence mask + optional extra meta-features."""
    reduced = pca.transform(raw_embed)
    parts = [reduced, mask]
    if extra is not None:
        parts.append(extra)
    return np.concatenate(parts, axis=1).astype(np.float32)


def soft_label_weight(y_soft, is_real, conf=None):
    """
    Confidence weight per (study, target) pair.
      Real (official) label -> 3.0, full trust.
      Weak label with conf  -> scaled by the labeller's OWN confidence.
      Weak label without    -> old fallback: distance from the undecided 0.5.

    FIX: the v2 label file reports confidence directly -- 0.95 when the report
    asserts a finding, 0.85 when it denies it, 0.05 when the report never
    mentions it. That is a far better weight than guessing from distance to
    0.5, because an UNK sits at 0.28 and the old formula would read that as a
    fairly confident negative when it actually means "no information".
    """
    if conf is not None:
        w = np.where(is_real[:, None], 3.0, 0.15 + 1.05 * np.asarray(conf, dtype=np.float32).clip(0, 1))
    else:
        w = np.where(
            is_real[:, None],
            3.0,
            0.25 + 0.75 * (2.0 * np.abs(y_soft - 0.5)).clip(0, 1),
        )
    return w.astype(np.float32)


def train_xgb_per_target(X, Y, W):
    """
    One XGBoost binary classifier per disease target.
    Y is thresholded at 0.5 for the binary label; W is the per-sample,
    per-target confidence weight from soft_label_weight().
    Targets with fewer than 2 classes in this fold are skipped (returns
    None for that target — predict_xgb() falls back to 0.5 for it).
    """
    models = {}
    for t_idx, target in enumerate(TARGETS):
        y = (Y[:, t_idx] >= 0.5).astype(int)
        w = W[:, t_idx]
        if len(np.unique(y)) < 2:
            models[target] = None
            continue
        dtrain = xgb.DMatrix(X, label=y, weight=w)
        models[target] = xgb.train(
            XGB_PARAMS, dtrain, num_boost_round=XGB_N_ESTIMATORS,
            verbose_eval=False,
        )
    return models


def predict_xgb(models, X):
    """[n, N_TARGETS] predicted probabilities; 0.5 for any skipped target."""
    n = X.shape[0]
    P = np.full((n, len(TARGETS)), 0.5, dtype=np.float32)
    dtest = xgb.DMatrix(X)
    for t_idx, target in enumerate(TARGETS):
        m = models.get(target)
        if m is not None:
            P[:, t_idx] = m.predict(dtest)
    return P


def main():
    seed_everything()          # honours RSNA_SEED
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # local flag (not the module-level XGB_AVAILABLE) so any failure inside
    # main() can disable the stacking layer for this run only, cleanly
    xgb_ok = XGB_AVAILABLE

    print("=" * 60)
    print("RSNA KNEE ABNORMALITY DETECTION")
    print(f"RUN A  two_level_att={TWO_LEVEL_ATT}  learnable_prior={LEARNABLE_PRIOR}  "
          f"slice_posenc={SLICE_POSENC}")
    print(f"       MAX_SLICES={MAX_SLICES}  BAND={SLICE_BAND}  WINDOW={WINDOW_PCT}")
    print(f"       slot_pool={SLOT_POOL}  patch_pool={PATCH_POOL} "
          f"(embed_dim={EMBED_DIM})")
    print(f"       crop_mm={CROP_MM}  rgb_mode={RGB_MODE}")
    if TTA_WINDOWS:
        print(f"       tta_windows={TTA_WINDOWS} group={TTA_GROUP} pool={TTA_POOL}")
    print(f"Device  : {device}")
    print(f"Kaggle  : {IS_KAGGLE}")
    print("=" * 60)

    # ── load labels: parser output, NOT train.csv ──────────────────
    test_df         = pd.read_csv(DATA_ROOT / "test.csv")
    test_series_csv  = DATA_ROOT / "test_series.csv"
    train_series_csv = DATA_ROOT / "train_series.csv"

    print(f"Using parsed labels: {PARSED_LABELS_CSV}")
    parsed = pd.read_csv(PARSED_LABELS_CSV, dtype={"StudyInstanceUID": str})
    parsed[TARGETS] = parsed[TARGETS].fillna(0)

    # FIX: the v2 label file has no is_real_label column -- it is a pure
    # report-derived file (score / __conf / __verdict per target). Deriving
    # the flag from train.csv is safer than hard-failing: the 58 official
    # studies are the ones with non-null labels there.
    if "is_real_label" not in parsed.columns:
        gold_ids = set()
        _train_csv = DATA_ROOT / "train.csv"
        if _train_csv.exists():
            _t = pd.read_csv(_train_csv, dtype={"StudyInstanceUID": str})
            _tt = [c for c in TARGETS if c in _t.columns]
            if _tt:
                gold_ids = set(_t.loc[_t[_tt].notna().any(axis=1),
                                      "StudyInstanceUID"].astype(str))
                # official 0/1 truth overrides the report guess for these
                _g = _t[_t["StudyInstanceUID"].astype(str).isin(gold_ids)]
                _g = _g.set_index("StudyInstanceUID")[_tt].astype(float)
                parsed = parsed.set_index("StudyInstanceUID")
                common = parsed.index.intersection(_g.index)
                parsed.loc[common, _tt] = _g.loc[common, _tt].values
                for c in _tt:
                    if c + "__conf" in parsed.columns:
                        parsed.loc[common, c + "__conf"] = 1.0
                parsed = parsed.reset_index()
        parsed["is_real_label"] = parsed["StudyInstanceUID"].astype(str).isin(gold_ids)
        print(f"  No is_real_label column -- derived {len(gold_ids)} gold studies "
              f"from train.csv and overrode their scores with official 0/1")

    parsed["is_real_label"] = parsed["is_real_label"].astype(bool)
    real_df = parsed[parsed["is_real_label"]].copy()
    weak_df = parsed[~parsed["is_real_label"]].copy()
    print(f"Real-labeled studies (true 0/1): {len(real_df)}")
    print(f"Weak-labeled studies (parser)  : {len(weak_df)}")

    # FIX: N_TOTAL_STUDIES defaults to 4407 now, so this uses every study
    # instead of randomly discarding ~67 of them.
    n_weak_needed = max(0, N_TOTAL_STUDIES - len(real_df))
    if n_weak_needed >= len(weak_df):
        weak_sample = weak_df
    else:
        weak_sample = weak_df.sample(n=n_weak_needed, random_state=SAMPLE_SEED)

    labeled = pd.concat([real_df, weak_sample], ignore_index=True)
    print(f"Training pool total            : {len(labeled)} "
          f"({len(real_df)} real + {len(weak_sample)} weak)")

    # Per-target confidence straight from the labeller, when present.
    CONF_COLS = [t + "__conf" for t in TARGETS]
    HAS_CONF  = all(c in labeled.columns for c in CONF_COLS)
    print(f"Per-target __conf columns      : {'yes' if HAS_CONF else 'no (fallback)'}")
    print(f"Test studies    : {len(test_df)}")

    # ══ TRAIN PIPELINE ══════════════════════════════════════════
    if IS_KAGGLE:
        print("\n── TRAIN: On Kaggle — data already unzipped from attached Dataset, skipping unzip ──")
    else:
        print("\n── TRAIN: Ensure studies unzipped ──")
        ensure_studies_unzipped(labeled["StudyInstanceUID"], TRAIN_SERIES, TRAIN_SERIES_ZIP)

    print("\n── TRAIN: Scan DICOMs ──")
    _cached_dcm_idx = WORK_DIR / "train_dicom_index.csv"
    _cached_emb_idx = WORK_DIR / "train_embedding_index.csv"

    # FIX: every cached artifact below was produced by the OLD slot config,
    # the OLD per-slice windowing and the broken laterality path, so reusing
    # any of it would silently keep the bugs alive. RSNA_FORCE_RESCAN=1
    # deletes them and rebuilds from the DICOMs.
    if os.environ.get("RSNA_FORCE_RESCAN", "0") == "1":
        print("\n  RSNA_FORCE_RESCAN=1 -- clearing cached scan/slot/embedding artifacts")
        for _p in [_cached_dcm_idx, _cached_emb_idx,
                   WORK_DIR / "train_slots_cache.pkl",
                   WORK_DIR / "test_slots_cache.pkl",
                   WORK_DIR / "test_dicom_index.csv",
                   WORK_DIR / "test_embedding_index.csv"]:
            if _p.exists():
                _p.unlink(); print(f"    removed {_p.name}")
        for _d in [EMB_DIR / "train", EMB_DIR / "test"]:
            if _d.exists():
                shutil.rmtree(_d, ignore_errors=True); print(f"    removed {_d}")
        if os.environ.get("RSNA_FORCE_RETRAIN", "1") == "1":
            for _p in list(MODEL_DIR.glob("*.pt")) + list(MODEL_DIR.glob("*.pkl")):
                _p.unlink(); print(f"    removed {_p.name}")

    # Anything that changes the PIXELS fed to MedSigLIP must invalidate the
    # embedding cache. Reuse was gated only on the two index files existing, so
    # changing MAX_SLICES, the band, the window, the millimetre crop or the
    # channel layout would silently reuse embeddings built under the old
    # settings: the run completes, the numbers move, and nothing says why.
    _sig_path = WORK_DIR / "embedding_signature.txt"
    _sig = "|".join(str(v) for v in [
        "medsiglip", EMBED_DIM_BASE, PATCH_POOL, MAX_SLICES, SLICE_BAND,
        WINDOW_PCT, CROP_MM, RGB_MODE, tuple(SLOT_NAMES),
    ])
    if _cached_dcm_idx.exists() and _cached_emb_idx.exists():
        _old = _sig_path.read_text().strip() if _sig_path.exists() else None
        if _old is not None and _old != _sig:
            raise RuntimeError(
                "cached embeddings were built with different preprocessing.\n"
                f"  cached : {_old}\n"
                f"  current: {_sig}\n"
                "Re-run with RSNA_FORCE_RESCAN=1, or restore the old settings.")
        if _old is None:
            print("  [WARN] cache has no signature (predates this check) — "
                  "cannot verify it matches the current preprocessing")
        print("  Found existing DICOM + embedding index — skipping scan/slot/embed steps")
        train_dcm     = pd.read_csv(_cached_dcm_idx, dtype=str)
        train_emb_idx = pd.read_csv(_cached_emb_idx, dtype=str)
        if "centre_x" not in train_dcm.columns:
            raise RuntimeError(
                "Cached train_dicom_index.csv predates the geometry fix (no "
                "centre_x column). Re-run with RSNA_FORCE_RESCAN=1.")
    else:
        if _cached_dcm_idx.exists():
            print(f"  Reusing cached DICOM index: {_cached_dcm_idx}")
            train_dcm = pd.read_csv(_cached_dcm_idx, dtype=str)
        else:
            train_study_filter = set(labeled["StudyInstanceUID"].astype(str).tolist())
            train_dcm = scan_dicoms(TRAIN_SERIES, train_series_csv, study_filter=train_study_filter)
            train_dcm.to_csv(_cached_dcm_idx, index=False)

        print("\n── TRAIN: Build slots ──")
        _cached_slots = WORK_DIR / "train_slots_cache.pkl"
        if _cached_slots.exists():
            print(f"  Reusing cached slots: {_cached_slots}")
            with open(_cached_slots, "rb") as f:
                train_slots, train_lat = pickle.load(f)
        else:
            train_lat   = detect_laterality(train_dcm)
            train_slots = assign_slots(train_dcm)
            train_slots["laterality"] = train_slots["StudyInstanceUID"].map(train_lat)
            with open(_cached_slots, "wb") as f:
                pickle.dump((train_slots, train_lat), f)
            print(f"  Cached slots to: {_cached_slots}")

        print("\n── TRAIN: Embed ──")
        processor, model_enc, device_enc = load_medsiglip()
        train_emb_idx = embed_slots(train_slots, train_dcm, processor, model_enc,
                                     device_enc, train_lat, EMB_DIR / "train")
        train_emb_idx.to_csv(WORK_DIR / "train_embedding_index.csv", index=False)
        _sig_path.write_text(_sig)      # so the next run can verify the cache
    if "model_enc" in dir(): del model_enc
    torch.cuda.empty_cache()

    # ══ STAGE A — PRETRAIN ON WEAK LABELS (all 4340: 58 real + weak) ═══
    print("\n── STAGE A: Pretrain on weak labels (4340 studies) ──")
    labeled_ids = set(labeled["StudyInstanceUID"].astype(str))
    emb_ids     = set(train_emb_idx["StudyInstanceUID"].astype(str))
    common      = labeled_ids & emb_ids
    print(f"Studies with labels + embeddings: {len(common)}")

    if len(common) < 2:
        print("[STOP] Not enough labeled studies.")
        return

    lbl = labeled[labeled["StudyInstanceUID"].astype(str).isin(common)].copy()
    emb = train_emb_idx[train_emb_idx["StudyInstanceUID"].astype(str).isin(common)].copy()

    real_ids_common = set(real_df["StudyInstanceUID"].astype(str)) & common
    pretrain_ids = np.array(sorted(common))
    print(f"Pretrain pool: {len(pretrain_ids)} studies "
          f"({len(real_ids_common)} of them real-labeled)")

    # Stage A: 5-fold CV on all weak+real studies
    # Best fold model (highest val AUC) used as starting point for Stage B
    #
    # RESUME SUPPORT: if stage_a_pretrained.pt already exists (either from
    # this session or copied in from a re-attached Kaggle Dataset of a
    # previous run's /kaggle/working/models folder), skip Stage A entirely
    # and go straight to Stage B using those weights.
    pretrain_path = MODEL_DIR / "stage_a_pretrained.pt"
    if pretrain_path.exists():
        print(f"\n── STAGE A: Found existing checkpoint at {pretrain_path} — skipping Stage A training ──")
        ckpt = torch.load(pretrain_path, map_location=device)
        pretrained_model = SlotAttentionModel().to(device)
        pretrained_model = load_state_dict_safe(pretrained_model, ckpt["model_state_dict"])
        best_stage_a_auc = None
    else:
        print(f"Stage A: 5-fold CV on {len(pretrain_ids)} studies")
        stage_a_table  = pd.DataFrame({"study": pretrain_ids})
        stage_a_gkf    = GroupKFold(n_splits=5)
        stage_a_models = []
        best_stage_a_auc   = -1
        best_stage_a_model = None

        for sa_fold, (sa_tri, sa_vi) in enumerate(
            stage_a_gkf.split(stage_a_table, groups=report_groups(stage_a_table.study, DATA_ROOT)), 1
        ):
            sa_tr_ids = pretrain_ids[sa_tri]
            sa_va_ids = pretrain_ids[sa_vi]
            pre_tr_ds = StudyDataset(emb[emb["StudyInstanceUID"].isin(sa_tr_ids)],
                                      lbl[lbl["StudyInstanceUID"].isin(sa_tr_ids)])
            pre_va_ds = StudyDataset(emb[emb["StudyInstanceUID"].isin(sa_va_ids)],
                                      lbl[lbl["StudyInstanceUID"].isin(sa_va_ids)])
            print(f"\nStage A FOLD {sa_fold}  train={len(pre_tr_ds)}  val={len(pre_va_ds)}")

            sa_model = train_fold(pre_tr_ds, pre_va_ds, device, epochs=30)

            # evaluate this fold
            sa_model.eval()
            sa_Y, sa_P = [], []
            with torch.no_grad():
                for i in range(len(pre_va_ds)):
                    _, x, mask, idx, y = pre_va_ds.get(i)
                    pred = torch.sigmoid(sa_model(x.to(device), mask.to(device),
                                                  idx.to(device))).cpu().numpy()
                    sa_Y.append(y.numpy()); sa_P.append(pred)
            sa_auc = auc_mean(np.vstack(sa_Y), np.vstack(sa_P))
            print(f"[Stage A FOLD {sa_fold}] val_auc={sa_auc:.5f}")

            # save each fold model
            sa_path = MODEL_DIR / f"stage_a_fold_{sa_fold}.pt"
            torch.save({"model_state_dict": sa_model.state_dict(),
                        "targets": TARGETS, "slot_names": SLOT_NAMES,
                        "embed_dim": EMBED_DIM, "proj_dim": PROJ_DIM}, sa_path)
            stage_a_models.append(sa_model)

            if not np.isnan(sa_auc) and sa_auc > best_stage_a_auc:
                best_stage_a_auc   = sa_auc
                best_stage_a_model = sa_model

        # use best Stage A fold as pretrained_model for Stage B
        pretrained_model = best_stage_a_model
        torch.save({"model_state_dict": pretrained_model.state_dict(),
                    "targets": TARGETS, "slot_names": SLOT_NAMES,
                    "embed_dim": EMBED_DIM, "proj_dim": PROJ_DIM},
                   pretrain_path)
        print(f"\nBest Stage A AUC: {best_stage_a_auc:.5f}")
        print(f"Saved best Stage A weights: {pretrain_path}")

    # ══ STAGE A (XGBOOST) — SAME WEAK+REAL POOL, TREE-BASED MODEL ═══════
    xgb_pca              = None
    xgb_stage_a_models   = []      # one dict-of-boosters per fold, for test-time averaging
    stage_a_xgb_oof      = None    # [len(pretrain_ids), N_TARGETS], meta-feature for Stage B
    if xgb_ok:
        try:
            xgb_stage_a_path = MODEL_DIR / "xgb_stage_a.pkl"
            print("\n── STAGE A (XGBoost): pooling embeddings + PCA ──")
            raw_embed, presence = build_tabular_matrix(pretrain_ids, emb)
            xgb_pca = fit_pca(raw_embed)
            print(f"  PCA: {raw_embed.shape[1]} -> {xgb_pca.n_components_} dims "
                  f"(explained var {xgb_pca.explained_variance_ratio_.sum():.2%})")

            _li = lbl.set_index("StudyInstanceUID")
            Y_pool = _li.loc[pretrain_ids, TARGETS].values.astype(np.float32)
            is_real_pool = np.array([sid in real_ids_common for sid in pretrain_ids])
            # FIX: feed the labeller's own per-target confidence when the file has it
            _cc = [t + "__conf" for t in TARGETS]
            C_pool = (_li.loc[pretrain_ids, _cc].values.astype(np.float32)
                      if all(c in _li.columns for c in _cc) else None)
            W_pool = soft_label_weight(Y_pool, is_real_pool, conf=C_pool)
            X_pool = make_features(raw_embed, presence, xgb_pca)

            if xgb_stage_a_path.exists():
                print(f"  Found existing checkpoint at {xgb_stage_a_path} — skipping Stage A XGBoost training")
                with open(xgb_stage_a_path, "rb") as f:
                    saved = pickle.load(f)
                xgb_pca            = saved["pca"]
                xgb_stage_a_models = saved["fold_models"]
                stage_a_xgb_oof    = saved["oof"]
                X_pool = make_features(raw_embed, presence, xgb_pca)  # rebuild with loaded pca (should match)
            else:
                stage_a_xgb_oof = np.zeros((len(pretrain_ids), len(TARGETS)), dtype=np.float32)
                xgb_sa_gkf = GroupKFold(n_splits=5)
                xgb_sa_table = pd.DataFrame({"study": pretrain_ids})
                for xfold, (xtri, xvi) in enumerate(
                    xgb_sa_gkf.split(xgb_sa_table, groups=report_groups(xgb_sa_table.study, DATA_ROOT)), 1
                ):
                    fold_models = train_xgb_per_target(X_pool[xtri], Y_pool[xtri], W_pool[xtri])
                    stage_a_xgb_oof[xvi] = predict_xgb(fold_models, X_pool[xvi])
                    xgb_stage_a_models.append(fold_models)
                    print(f"  [Stage A XGBoost FOLD {xfold}] trained "
                          f"({sum(m is not None for m in fold_models.values())}/{len(TARGETS)} targets)")

                sa_xgb_auc = auc_mean(Y_pool, stage_a_xgb_oof)
                print(f"  Stage A XGBoost OOF mean AUC: {sa_xgb_auc:.5f}")

                with open(xgb_stage_a_path, "wb") as f:
                    pickle.dump({"pca": xgb_pca, "fold_models": xgb_stage_a_models,
                                "oof": stage_a_xgb_oof}, f)
                print(f"  Saved: {xgb_stage_a_path}")
        except Exception as e:
            print(f"[WARN] Stage A XGBoost failed ({e}) — stacking layer disabled for this run.")
            xgb_ok = False

    # ══ STAGE B — FINE-TUNE ON THE 58 REAL LABELS (5-fold CV) ══════════
    print("\n── STAGE B: Fine-tune on 58 real-labeled studies ──")
    real_lbl = lbl[lbl["StudyInstanceUID"].isin(real_ids_common)].copy()
    real_emb = emb[emb["StudyInstanceUID"].isin(real_ids_common)].copy()

    ids      = np.array(sorted(real_ids_common))
    table    = pd.DataFrame({"study": ids})
    n_splits = min(5, len(ids))
    gkf      = GroupKFold(n_splits=n_splits)
    oof      = np.zeros((len(ids), len(TARGETS)), dtype=np.float32)

    # ── XGBoost Stage B: build features once, train per-fold inside the loop below ──
    xgb_stage_b_models = []
    xgb_oof            = None
    if xgb_ok and xgb_pca is not None:
        try:
            raw_embed_b, presence_b = build_tabular_matrix(ids, real_emb)
            # Stage A XGBoost OOF predictions, reordered to match `ids`, used as
            # meta-features — this is the actual "stacking" part: Stage B trees
            # get to see what the Stage A model (trained 4340 studies) thought.
            pretrain_pos = {sid: i for i, sid in enumerate(pretrain_ids)}
            meta_b = np.stack([stage_a_xgb_oof[pretrain_pos[sid]] for sid in ids]).astype(np.float32)
            X_stage_b = make_features(raw_embed_b, presence_b, xgb_pca, extra=meta_b)
            Y_stage_b = real_lbl.set_index("StudyInstanceUID").loc[ids, TARGETS].values.astype(np.float32)
            xgb_oof   = np.zeros((len(ids), len(TARGETS)), dtype=np.float32)
        except Exception as e:
            print(f"[WARN] XGBoost Stage B feature build failed ({e}) — stacking layer disabled for this run.")
            xgb_ok = False

    # ── FULL CHECKPOINT RESUME: if ALL fold models + oof_predictions.csv exist,
    #    load everything from cache — no re-training, no re-evaluation on wrong splits
    _oof_cache       = MODEL_DIR / "oof_predictions.csv"
    _all_folds_exist = all((MODEL_DIR / f"fold_{f}.pt").exists() for f in range(1, n_splits+1))
    _all_xgb_exist   = all((MODEL_DIR / f"xgb_fold_{f}.pkl").exists() for f in range(1, n_splits+1))

    if _all_folds_exist and _oof_cache.exists():
        print("\n  All fold checkpoints + OOF file found — loading from cache, skipping all training")

        # load OOF from saved file (correct splits from SSH training)
        _oof_df = pd.read_csv(_oof_cache, dtype={"StudyInstanceUID": str})
        _oof_df = _oof_df.set_index("StudyInstanceUID").reindex(ids).reset_index()
        oof = _oof_df[TARGETS].values.astype(np.float32)
        mean_auc = auc_mean(real_lbl.set_index("StudyInstanceUID").loc[ids, TARGETS].values, oof)
        print(f"  Loaded OOF AUC (from SSH training): {mean_auc:.5f}")

        # load all fold models for inference
        fold_models_loaded = []
        for fold in range(1, n_splits+1):
            ckpt = torch.load(MODEL_DIR / f"fold_{fold}.pt", map_location=device, weights_only=False)
            m = SlotAttentionModel().to(device)
            m = load_state_dict_safe(m, ckpt["model_state_dict"])
            m.eval()
            fold_models_loaded.append(m)
            print(f"  Loaded fold_{fold}.pt")

        # load XGBoost folds if available
        if xgb_ok and xgb_oof is not None and _all_xgb_exist:
            try:
                for fold in range(1, n_splits+1):
                    with open(MODEL_DIR / f"xgb_fold_{fold}.pkl", "rb") as f:
                        fold_xgb_models = pickle.load(f)
                    xgb_stage_b_models.append(fold_xgb_models)
                    xgb_oof[np.array([i for i, s in enumerate(ids)
                                      if s in ids])] = predict_xgb(fold_xgb_models,
                                      X_stage_b[[i for i,_ in enumerate(ids)]])
                print(f"  Loaded {len(xgb_stage_b_models)} XGBoost fold checkpoints")
            except Exception as e:
                print(f"[WARN] XGBoost load failed ({e}) — NN-only for blend")
                xgb_ok = False

    else:
        # ── normal training loop ──────────────────────────────────────────────
        for fold, (tri, vi) in enumerate(gkf.split(table, groups=report_groups(table.study, DATA_ROOT)), 1):
            fold_ckpt_path = MODEL_DIR / f"fold_{fold}.pt"
            tr_fold_ids = ids[tri]; va_fold_ids = ids[vi]
            tr_ds = StudyDataset(real_emb[real_emb["StudyInstanceUID"].isin(tr_fold_ids)],
                                  real_lbl[real_lbl["StudyInstanceUID"].isin(tr_fold_ids)])
            va_ds = StudyDataset(real_emb[real_emb["StudyInstanceUID"].isin(va_fold_ids)],
                                  real_lbl[real_lbl["StudyInstanceUID"].isin(va_fold_ids)])

            if fold_ckpt_path.exists():
                print(f"\nFOLD {fold}  — checkpoint already exists, loading and skipping training")
                ckpt = torch.load(fold_ckpt_path, map_location=device, weights_only=False)
                fold_model = SlotAttentionModel().to(device)
                fold_model = load_state_dict_safe(fold_model, ckpt["model_state_dict"])
            else:
                print(f"\nFOLD {fold}  train={len(tr_ds)}  val={len(va_ds)}")
                fold_model = SlotAttentionModel().to(device)
                fold_model = load_state_dict_safe(fold_model, pretrained_model.state_dict())
                fold_model = finetune_fold(fold_model, tr_ds, va_ds, device, epochs=15, lr=2e-5)

            fold_model.eval()
            Y, P = [], []
            with torch.no_grad():
                for i in range(len(va_ds)):
                    study, x, mask, idx_t, y = va_ds.get(i)
                    pred = torch.sigmoid(fold_model(x.to(device), mask.to(device),
                                                    idx_t.to(device))).cpu().numpy()
                    oof[np.where(ids == study)[0][0]] = pred
                    Y.append(y.numpy()); P.append(pred)

            fold_auc = auc_mean(np.vstack(Y), np.vstack(P))
            print(f"[FOLD {fold}] AUC={fold_auc:.5f}")
            # The reading travels with the member: a checkpoint that loads under
            # different preprocessing runs cleanly and returns a submission
            # computed from the wrong pixels, so the config and a fingerprint of
            # the fitted map are stored alongside the weights.
            torch.save({"model_state_dict": fold_model.state_dict(),
                        "targets": TARGETS, "slot_names": SLOT_NAMES,
                        "embed_dim": EMBED_DIM, "proj_dim": PROJ_DIM,
                        "fingerprint": fingerprint(fold_model),
                        "config": {"PATCH_POOL": PATCH_POOL, "MAX_SLICES": MAX_SLICES,
                                   "SLICE_BAND": SLICE_BAND, "WINDOW_PCT": WINDOW_PCT,
                                   "SLOT_POOL": SLOT_POOL,
                                   "TWO_LEVEL_ATT": TWO_LEVEL_ATT,
                                   "LEARNABLE_PRIOR": LEARNABLE_PRIOR,
                                   "SLICE_POSENC": SLICE_POSENC}},
                       MODEL_DIR / f"fold_{fold}.pt")

            if xgb_ok and xgb_oof is not None:
                try:
                    xgb_fold_path = MODEL_DIR / f"xgb_fold_{fold}.pkl"
                    if xgb_fold_path.exists():
                        with open(xgb_fold_path, "rb") as f:
                            fold_xgb_models = pickle.load(f)
                        print(f"  [XGB FOLD {fold}] loaded existing checkpoint")
                    else:
                        fold_xgb_models = train_xgb_per_target(
                            X_stage_b[tri], Y_stage_b[tri],
                            np.ones_like(Y_stage_b[tri])
                        )
                        with open(xgb_fold_path, "wb") as f:
                            pickle.dump(fold_xgb_models, f)
                    xgb_oof[vi] = predict_xgb(fold_xgb_models, X_stage_b[vi])
                    xgb_stage_b_models.append(fold_xgb_models)
                    xgb_fold_auc = auc_mean(Y_stage_b[vi], xgb_oof[vi])
                    print(f"  [XGB FOLD {fold}] AUC={xgb_fold_auc:.5f}")
                except Exception as e:
                    print(f"[WARN] XGBoost Stage B fold {fold} failed ({e})")
                    xgb_ok = False

        oof_df = pd.DataFrame(oof, columns=TARGETS)
        oof_df.insert(0, "StudyInstanceUID", ids)
        oof_df.to_csv(MODEL_DIR / "oof_predictions.csv", index=False)
        mean_auc = auc_mean(real_lbl[TARGETS].values, oof)

    print(f"\nOOF Mean AUC (fine-tuned, on 58 real labels): {mean_auc:.5f}")
    print("\nPer-target AUC:")
    for i, t in enumerate(TARGETS):
        col = real_lbl.set_index("StudyInstanceUID").loc[ids, TARGETS].values[:, i]
        if len(np.unique(col)) > 1:
            a = roc_auc_score(col, oof[:, i])
            print(f"  {t:22s}: {a:.4f}")
        else:
            print(f"  {t:22s}: (no positives in OOF)")

    # ── Pick blend weight (alpha) between NN and XGBoost using OOF AUC ──
    best_alpha = 1.0
    if xgb_ok and xgb_oof is not None and len(xgb_stage_b_models) > 0:
        try:
            xgb_only_auc = auc_mean(
                real_lbl.set_index("StudyInstanceUID").loc[ids, TARGETS].values, xgb_oof)
            print(f"\nXGBoost-only OOF AUC: {xgb_only_auc:.5f}")
            print("\nBlend search (alpha = NN weight, 1-alpha = XGBoost weight):")
            best_blend_auc = mean_auc
            for a in ALPHA_CANDIDATES:
                blended = a * oof + (1 - a) * xgb_oof
                blend_auc = auc_mean(
                    real_lbl.set_index("StudyInstanceUID").loc[ids, TARGETS].values, blended)
                marker = ""
                if not np.isnan(blend_auc) and blend_auc > best_blend_auc:
                    best_blend_auc = blend_auc
                    best_alpha     = a
                    marker = "  <- best so far"
                print(f"  alpha={a:.2f}  blended_auc={blend_auc:.5f}{marker}")
            print(f"\nChosen alpha={best_alpha:.2f}  "
                  f"(NN-only={mean_auc:.5f} -> blended={best_blend_auc:.5f})")
        except Exception as e:
            print(f"[WARN] Blend search failed ({e}) — using NN-only predictions.")
            best_alpha = 1.0

    # ══ TEST PIPELINE ═══════════════════════════════════════════
    print("\n── TEST: Scan DICOMs ──")
    test_dcm = scan_dicoms(TEST_SERIES, test_series_csv)

    print("\n── TEST: Build slots ──")
    test_lat   = detect_laterality(test_dcm)
    test_slots = assign_slots(test_dcm)
    test_slots["laterality"] = test_slots["StudyInstanceUID"].map(test_lat)

    print("\n── TEST: Embed ──")
    processor, model_enc, device_enc = load_medsiglip()
    test_emb_idx = embed_slots(test_slots, test_dcm, processor, model_enc,
                                device_enc, test_lat, EMB_DIR / "test")
    if "model_enc" in dir(): del model_enc
    torch.cuda.empty_cache()

    print("\n── TEST: Inference ──")
    model_paths = sorted(MODEL_DIR.glob("fold_*.pt"))
    study_ids, preds = run_inference(test_emb_idx, model_paths, device)

    # ── TEST: XGBoost stacking prediction + blend with NN ──
    final_preds = preds
    if xgb_ok and best_alpha < 1.0 and xgb_pca is not None and xgb_stage_b_models:
        try:
            print("\n── TEST: XGBoost stacking prediction ──")
            raw_embed_t, presence_t = build_tabular_matrix(study_ids, test_emb_idx)

            # Stage A meta-features for test studies — average across Stage A folds
            X_stage_a_t = make_features(raw_embed_t, presence_t, xgb_pca)
            meta_t = np.zeros((len(study_ids), len(TARGETS)), dtype=np.float32)
            for fold_models in xgb_stage_a_models:
                meta_t += predict_xgb(fold_models, X_stage_a_t) / len(xgb_stage_a_models)

            # Stage B prediction — average across Stage B folds
            X_stage_b_t = make_features(raw_embed_t, presence_t, xgb_pca, extra=meta_t)
            xgb_test_preds = np.zeros((len(study_ids), len(TARGETS)), dtype=np.float32)
            for fold_models in xgb_stage_b_models:
                xgb_test_preds += predict_xgb(fold_models, X_stage_b_t) / len(xgb_stage_b_models)

            final_preds = best_alpha * preds + (1 - best_alpha) * xgb_test_preds
            print(f"  Blended NN + XGBoost predictions (alpha={best_alpha:.2f})")
        except Exception as e:
            print(f"[WARN] XGBoost test-time prediction failed ({e}) — using NN-only predictions.")
            final_preds = preds

    # ══ SUBMISSION ══════════════════════════════════════════════
    sub = pd.DataFrame(final_preds, columns=TARGETS)
    sub.insert(0, "StudyInstanceUID", study_ids)

    # add any test studies with no embeddings at 0.5 (default)
    missing = set(test_df["StudyInstanceUID"].astype(str)) - set(study_ids)
    if missing:
        print(f"[WARN] {len(missing)} test studies had no embeddings — defaulting to 0.5")
        filler = pd.DataFrame([[sid] + [0.5]*len(TARGETS) for sid in missing],
                               columns=["StudyInstanceUID"] + TARGETS)
        sub = pd.concat([sub, filler], ignore_index=True)

    # reorder to match sample submission
    sub = sub.set_index("StudyInstanceUID").reindex(
        test_df["StudyInstanceUID"].astype(str)).reset_index()
    sub.columns = ["StudyInstanceUID"] + TARGETS

    out = WORK_DIR / "submission.csv"
    sub.to_csv(out, index=False)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"OOF AUC    : {mean_auc:.5f}")
    print(f"Submission : {out}")
    print(f"Shape      : {sub.shape}")
    print(sub.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
