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
from sklearn.decomposition import PCA

# ── config ────────────────────────────────────────────────────────────────────
IS_KAGGLE     = os.path.exists("/kaggle/input")

# Linux SSH/GPU server (not Kaggle, not the local Windows dev machine).
# Same server layout used by RSNA v4.
IS_SERVER     = (not IS_KAGGLE) and os.name == "posix"
SERVER_ROOT   = Path(os.environ.get("RSNA_SERVER_ROOT", "/home/harleen_ece/rsna_knee_ai"))

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
    DATA_ROOT = Path(os.environ.get("RSNA_DATA_ROOT", str(SERVER_ROOT / "DATA")))
else:
    DATA_ROOT = Path(os.environ.get("RSNA_DATA_ROOT", r"C:\kabir\RSNA_Knee_AI\genuine_pipeline\DATA"))
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
    MODEL_PATH = Path(os.environ.get("RSNA_MODEL_PATH", str(SERVER_ROOT / "MedSigLIP")))
else:
    MODEL_PATH = Path(os.environ.get("RSNA_MODEL_PATH", r"C:\kabir\RSNA_Knee_AI\genuine_pipeline\MedSigLIP"))

if IS_KAGGLE:
    WORK_DIR = Path("/kaggle/working")
elif IS_SERVER:
    WORK_DIR = Path(os.environ.get("RSNA_WORK_DIR", str(SERVER_ROOT / "rsnav4_ssh")))
else:
    WORK_DIR = Path(os.environ.get("RSNA_WORK_DIR", r"C:\kabir\RSNA_Knee_AI\genuine_pipeline\rsna .791\runs\pc_200_best"))
TRAIN_SERIES  = DATA_ROOT / "train_series"
TEST_SERIES   = DATA_ROOT / "test_series"
# CACHE_DIR holds embeddings specifically -- separate env var from WORK_DIR
# so a launcher can point embeddings at one experiment's cache (e.g.
# cache_4349) while keeping models/reports under a different outputs dir
# (e.g. outputs_4349), or share a single dir when RSNA_CACHE_DIR isn't set.
CACHE_DIR     = Path(os.environ.get("RSNA_CACHE_DIR", str(WORK_DIR)))
EMB_DIR       = CACHE_DIR / "embeddings"
MODEL_DIR     = WORK_DIR / "models"
WORK_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
EMB_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Stage 0: optional MRNet auxiliary pretraining (ACL only) ──────────────
# MRNet (Stanford, https://stanfordmlgroup.github.io/competitions/mrnet/) is
# a public knee-MRI dataset with ACL tear labels on ~1,370 exams -- far more
# than our own 58 real-labeled studies. Used ONLY for the ACL head (MRNet's
# meniscus label doesn't distinguish medial/lateral, so it's skipped rather
# than injected as an ambiguous signal into two targets that need to stay
# separate). Entirely optional: if MRNET_ROOT doesn't exist, Stage 0 is
# skipped automatically and training proceeds exactly as before.
if IS_SERVER:
    MRNET_ROOT = SERVER_ROOT / "MRNet-v1.0"
elif IS_KAGGLE:
    MRNET_ROOT = Path("/kaggle/input/mrnet-v1")  # only used if this dataset is attached
else:
    MRNET_ROOT = Path(os.environ.get("RSNA_MRNET_ROOT", r"C:\kabir\RSNA_Knee_AI\MRNet-v1.0"))
RUN_MRNET_PRETRAIN = (os.environ.get("RSNA_RUN_MRNET_PRETRAIN", "auto").lower() != "0"
                      and MRNET_ROOT.exists())
MRNET_EMB_DIR = WORK_DIR / "mrnet_embeddings"
MRNET_EMB_DIR.mkdir(parents=True, exist_ok=True)
# loose plane->slot mapping since MRNet doesn't record fat-sat vs non-fat-sat
# the way our own DICOM slot system does -- best-effort correspondence only
MRNET_PLANE_TO_SLOT = {"sagittal": "SAG_T1", "coronal": "COR_T1", "axial": "AX_FLUID_FS"}

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
    PARSED_LABELS_CSV = Path(os.environ.get(
        "RSNA_LABELS_CSV",
        str(SERVER_ROOT / "AI-MODEL" / "final_labels_real_plus_generated.csv"),
    ))
else:
    PARSED_LABELS_CSV = Path(os.environ.get(
        "RSNA_LABELS_CSV",
        r"C:\kabir\RSNA_Knee_AI\genuine_pipeline\DATA\output best\final_labels_real_plus_generated best.csv",
    ))

N_TOTAL_STUDIES = int(os.environ.get("RSNA_N_TOTAL_STUDIES", "4349"))
SAMPLE_SEED     = int(os.environ.get("RSNA_SAMPLE_SEED", "42"))
STUDY_LIST_CSV = os.environ.get("RSNA_STUDY_LIST", "").strip()
RUN_TEST_INFERENCE = os.environ.get("RSNA_RUN_TEST_INFERENCE", "1").lower() not in {"0", "false", "no"}
# Test-time augmentation: average predictions over a few rotation-only
# views (laterality-safe, see _tta_views/encode_images_tta near
# encode_images). Off by default for train embeddings (unaffected by this
# flag -- train always calls embed_slots with tta=False), on by default
# for test embeddings and for pseudo-label scoring of weak studies.
TEST_TTA = os.environ.get("RSNA_TEST_TTA", "1").lower() not in {"0", "false", "no"}
# Guarded pseudo-labeling: use the fine-tuned Stage B model to refine weak
# (parser-derived) labels where it is very confident (see
# PSEUDO_LABEL_HIGH/LOW near apply_guarded_pseudo_labels). The 58 real
# labels are never touched. Writes a separate CSV rather than overwriting
# PARSED_LABELS_CSV in place, so this is opt-in to actually use.
RUN_PSEUDO_LABELING = os.environ.get("RSNA_RUN_PSEUDO_LABELING", "1").lower() not in {"0", "false", "no"}
# If set, refuse to run on CPU rather than silently falling back -- useful
# on a shared server where a GPU allocation might not actually be live
# (e.g. driver not loaded) and a silent CPU run would just be extremely
# slow rather than failing loudly.
REQUIRE_CUDA = os.environ.get("RSNA_REQUIRE_CUDA", "0").lower() not in {"0", "false", "no"}

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
    "SAG_FLUID_NOFS",
    "COR_T1",
    "SAG_T1",
]
N_SLOT    = len(SLOT_NAMES)
EMBED_DIM = 1152
PROJ_DIM  = 256
MAX_SLICES  = 12
BATCH_SIZE  = int(os.environ.get("RSNA_BATCH_SIZE", "16"))
SLICE_BAND  = (0.20, 0.80)
LAT_OFFSET  = 20.0
PRIOR_STRENGTH = 0.55

# Stage A0: partial MedSigLIP fine-tuning (see finetune_medsiglip_stage_a0).
# Set to False to skip it and keep the exact old frozen-MedSigLIP behavior.
RUN_STAGE_A0_FINETUNE = os.environ.get("RSNA_RUN_STAGE_A0", "1").lower() not in {"0", "false", "no"}
MEDSIGLIP_FINETUNE_BLOCKS = int(os.environ.get("RSNA_STAGE_A0_BLOCKS", "4"))
MEDSIGLIP_FINETUNE_EPOCHS = int(os.environ.get("RSNA_STAGE_A0_EPOCHS", "3"))
MEDSIGLIP_FINETUNE_MAX_IMAGES = int(os.environ.get("RSNA_STAGE_A0_MAX_IMAGES", "12"))

# Set RSNA_FORCE_RESCAN=1 to ignore/delete the train-side DICOM index,
# slots, and embedding-index caches at startup and rebuild them from
# scratch this run (e.g. after changing N_TOTAL_STUDIES / the study
# sample, or when you don't trust a cache is still valid). Does NOT
# touch the fine-tuned MedSigLIP checkpoint or per-study embedding
# .npy files themselves -- only the three index/cache files that
# gate whether scan/slot/embed steps are skipped.
FORCE_RESCAN = os.environ.get("RSNA_FORCE_RESCAN", "0").lower() not in {"0", "false", "no"}

SLOT_PRIOR = {
    "ACL":              [1, 0, 0, 1, 0, 1],
    "MCL":              [0, 1, 0, 0, 1, 0],
    "Medial Meniscus":  [1, 1, 0, 1, 1, 0],
    "Lateral Meniscus": [1, 1, 0, 1, 1, 0],
    "Medial OA":        [0, 1, 0, 0, 1, 0],
    "Lateral OA":       [0, 1, 0, 0, 1, 0],
    "PF OA":            [1, 0, 1, 0, 0, 1],
    "Effusion":         [1, 0, 1, 0, 0, 0],
    "Synovitis":        [1, 0, 1, 0, 0, 0],
    "Baker's":          [1, 0, 0, 0, 0, 0],
    "Contusion":        [1, 1, 0, 1, 1, 0],
    "Fracture":         [1, 1, 1, 1, 1, 1],
}

SLOTS = [
    ("SAG_FLUID_FS",   "Sagittal", 1, 1),
    ("COR_FLUID_FS",   "Coronal",  1, 1),
    ("AX_FLUID_FS",    "Axial",    1, 1),
    ("SAG_FLUID_NOFS", "Sagittal", 1, 0),
    ("COR_T1",         "Coronal",  0, 0),
    ("SAG_T1",         "Sagittal", 0, 0),
]


def seed_everything(s=42):
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

    def _read_header(p):
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
            study  = str(getattr(ds, "StudyInstanceUID",  "") or "").strip()
            series = str(getattr(ds, "SeriesInstanceUID", "") or "").strip()
            sop    = str(getattr(ds, "SOPInstanceUID",    "") or "").strip()
            inst   = getattr(ds, "InstanceNumber", None)
            if inst is not None:
                try: inst = int(inst)
                except: inst = None
            if study and series:
                return dict(filepath=str(p), StudyInstanceUID=study,
                            SeriesInstanceUID=series, SOPInstanceUID=sop,
                            InstanceNumber=inst)
        except Exception:
            pass
        return None

    from concurrent.futures import ThreadPoolExecutor
    print(f"  Reading {len(paths)} DICOM headers (parallel)...")
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(_read_header, paths))
    rows = [r for r in results if r is not None]

    df = pd.DataFrame(rows)
    meta = pd.read_csv(series_csv, dtype=str)
    meta["Fluid_Sensitive"] = meta["Fluid_Sensitive"].astype(int)
    meta["Fat_Suppression"] = meta["Fat_Suppression"].astype(int)

    merged = df.merge(meta, on=["StudyInstanceUID", "SeriesInstanceUID"], how="left")

    # fallback: infer plane from IOP for unmatched
    unmatched = merged["Anatomical_Plane"].isna()
    if unmatched.sum() > 0:
        for idx in merged[unmatched].index:
            try:
                ds = pydicom.dcmread(merged.loc[idx, "filepath"],
                                     stop_before_pixels=True, force=True)
                merged.loc[idx, "Anatomical_Plane"] = _plane_from_iop(ds)
                merged.loc[idx, "Fluid_Sensitive"]  = 0
                merged.loc[idx, "Fat_Suppression"]  = 0
            except Exception:
                pass

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
    study_cx = {}
    for _, r in dicom_df.iterrows():
        try:
            ds  = pydicom.dcmread(r["filepath"], stop_before_pixels=True, force=True)
            ipp = getattr(ds, "ImagePositionPatient", None)
            iop = getattr(ds, "ImageOrientationPatient", None)
            ps  = getattr(ds, "PixelSpacing", None)
            rows = getattr(ds, "Rows", None)
            cols = getattr(ds, "Columns", None)
            if ipp is None or iop is None or ps is None: continue
            ipp = np.array([float(x) for x in ipp[:3]])
            iop = np.array([float(x) for x in iop[:6]])
            ps  = np.array([float(x) for x in ps[:2]])
            cx  = ipp[0] + iop[0]*ps[1]*float(cols)/2 + iop[3]*ps[0]*float(rows)/2
            study_cx.setdefault(r["StudyInstanceUID"], []).append(float(cx))
        except Exception:
            pass
    result = {}
    for su, xs in study_cx.items():
        m = float(np.median(xs))
        result[su] = "R" if m < -LAT_OFFSET else ("L" if m > LAT_OFFSET else None)
    return result


def assign_slots(dicom_df):
    slice_counts = dicom_df.groupby("SeriesInstanceUID").size().to_dict()
    series_df = (dicom_df.groupby(["StudyInstanceUID", "SeriesInstanceUID"])
                 .first().reset_index())
    series_df["n_slices"] = series_df["SeriesInstanceUID"].map(slice_counts)
    rows = []
    for study, grp in series_df.groupby("StudyInstanceUID"):
        for slot_name, plane, fluid, fat in SLOTS:
            mask = ((grp["Anatomical_Plane"] == plane) &
                    (grp["Fluid_Sensitive"].fillna(0).astype(int) == fluid) &
                    (grp["Fat_Suppression"].fillna(0).astype(int) == fat))
            cands = grp[mask].sort_values("n_slices", ascending=False)
            if len(cands) == 0:
                rows.append({"StudyInstanceUID": study, "SeriesInstanceUID": "",
                             "slot_name": slot_name, "n_slices": 0, "presence_mask": 0})
            else:
                best = cands.iloc[0]
                rows.append({"StudyInstanceUID": study,
                             "SeriesInstanceUID": best["SeriesInstanceUID"],
                             "slot_name": slot_name,
                             "n_slices": int(best["n_slices"]),
                             "presence_mask": 1})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — MEDSIGLIP EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════════
def sort_slices(paths):
    meta = []
    for p in paths:
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
    n  = len(paths)
    lo = int(np.floor(n * SLICE_BAND[0]))
    hi = int(np.ceil(n  * SLICE_BAND[1]))
    band = paths[lo:hi]
    if len(band) <= MAX_SLICES: return band
    idx = np.unique(np.round(np.linspace(0, len(band)-1, MAX_SLICES)).astype(int))
    return [band[i] for i in idx]


def normalise_laterality(imgs, plane, lat):
    if lat != "R": return imgs
    if plane in ("Coronal", "Axial"):
        return [img.transpose(Image.FLIP_LEFT_RIGHT) for img in imgs]
    return imgs[::-1]


def dicom_to_pil(path):
    ds  = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)
    if str(getattr(ds, "PhotometricInterpretation", "")).strip() == "MONOCHROME1":
        arr = arr.max() - arr
    slope     = float(getattr(ds, "RescaleSlope",     1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    arr = arr * slope + intercept
    lo, hi = np.percentile(arr, [1, 99])
    if hi <= lo: lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo: arr = np.zeros_like(arr, dtype=np.uint8)
    else:
        arr = np.clip((arr - lo) / (hi - lo), 0, 1)
        arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def unfreeze_medsiglip_vision(model, train_last_blocks=4):
    """Lock every MedSigLIP parameter, then deliberately re-unlock only the
    last N vision-tower blocks (plus the final norm layer).

    Early transformer blocks learn generic visual patterns that transfer
    fine as-is; late blocks are where task-specific adaptation happens.
    With only 58 gold-labeled studies, unlocking the WHOLE model risks
    catastrophically overwriting MedSigLIP's general knowledge while trying
    to fit 58 examples. Unlocking just the tail is the standard, safer way
    to adapt a large pretrained model to a small labeled set.
    """
    vision = model.vision_model
    for p in vision.parameters():
        p.requires_grad = False
    blocks = vision.encoder.layers
    for block in blocks[-train_last_blocks:]:
        for p in block.parameters():
            p.requires_grad = True
    if hasattr(vision, "post_layernorm"):
        for p in vision.post_layernorm.parameters():
            p.requires_grad = True
    n_trainable = sum(p.numel() for p in vision.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in vision.parameters())
    print(f"  MedSigLIP vision tower: {n_trainable:,}/{n_total:,} params trainable "
          f"(last {train_last_blocks} blocks + post_layernorm)")
    return model


def load_medsiglip(finetuned_path=None):
    """Single loader used at BOTH train-embed and test-embed time.

    If a fine-tuned vision-tower checkpoint exists (produced by
    finetune_medsiglip_stage_a0 below), it is loaded here automatically.
    This is deliberate: train and test embeddings must come from the exact
    same weights, or you get a silent train/test distribution mismatch that
    tanks accuracy with no error thrown. Having ONE function that every
    embedding call site goes through -- rather than two call sites each
    deciding independently whether to load fine-tuned weights -- removes
    that failure mode by construction instead of relying on remembering to
    keep two code paths in sync.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading MedSigLIP | device={device}")
    processor = AutoProcessor.from_pretrained(str(MODEL_PATH))
    model = AutoModel.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    if finetuned_path is None:
        finetuned_path = MODEL_DIR / "medsiglip_finetuned_vision.pt"
    finetuned_path = Path(finetuned_path)
    if finetuned_path.exists():
        print(f"  Found fine-tuned vision tower at {finetuned_path} — loading it "
              f"(both train and test embeddings will use these weights)")
        ckpt = torch.load(finetuned_path, map_location=device, weights_only=True)
        missing, unexpected = model.vision_model.load_state_dict(ckpt["vision_state_dict"], strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"medsiglip_finetuned_vision.pt does not match this MedSigLIP build: "
                f"missing={len(missing)} unexpected={len(unexpected)}. Do not silently "
                f"proceed on a partial/mismatched load -- re-run Stage A0 fine-tuning."
            )
    else:
        print("  No fine-tuned checkpoint found — using stock MedSigLIP weights.")

    model = model.eval()
    return processor, model, device


def encoder_fingerprint():
    """Identifies which encoder weights are currently active (stock vs a
    specific fine-tuned checkpoint's mtime+size), so cached embeddings can
    be auto-invalidated if the encoder changes between runs instead of
    silently mixing embeddings produced by two different encoders.
    """
    p = MODEL_DIR / "medsiglip_finetuned_vision.pt"
    if p.exists():
        stat = p.stat()
        return f"finetuned:{stat.st_mtime_ns}:{stat.st_size}"
    return "stock"


def check_embedding_cache_freshness(work_dir):
    """Invalidate every artifact derived from an outdated vision encoder.

    DICOM scan and slot-assignment caches are intentionally preserved: they
    depend only on the source DICOM/metadata, not on encoder weights. Raw
    per-series embeddings, neural/XGBoost checkpoints and OOF predictions
    must all be rebuilt together, otherwise a new encoder can silently reuse
    old features or models trained on those old features.
    """
    import shutil

    work_dir = Path(work_dir)
    fp_path = work_dir / "embedding_encoder_fingerprint.txt"
    current = encoder_fingerprint()
    previous = fp_path.read_text().strip() if fp_path.exists() else None
    if previous == current:
        return

    removed = []

    # Per-study train/test .pt files are checked individually by embed_slots,
    # so removing only train_embedding_index.csv is not sufficient.
    embedding_dir = work_dir / "embeddings"
    if embedding_dir.exists():
        n_embedding_files = sum(1 for p in embedding_dir.rglob("*") if p.is_file())
        shutil.rmtree(embedding_dir)
        embedding_dir.mkdir(parents=True, exist_ok=True)
        removed.append(f"embeddings/ ({n_embedding_files} files)")

    stale_files = [
        work_dir / "train_embedding_index.csv",
        MODEL_DIR / "stage_a_pretrained.pt",
        MODEL_DIR / "xgb_stage_a.pkl",
        MODEL_DIR / "oof_predictions.csv",
    ]
    stale_files.extend(MODEL_DIR.glob("stage_a_fold_*.pt"))
    stale_files.extend(MODEL_DIR.glob("fold_*.pt"))
    stale_files.extend(MODEL_DIR.glob("xgb_fold_*.pkl"))

    # De-duplicate paths while preserving deterministic log order.
    for stale_path in sorted(set(stale_files), key=lambda p: str(p)):
        if stale_path.exists():
            stale_path.unlink()
            removed.append(str(stale_path.relative_to(work_dir)))

    old_name = previous if previous is not None else "unrecorded/legacy cache"
    if removed:
        print(f"  [CACHE] Encoder changed ({old_name} -> {current}) — invalidated "
              f"{len(removed)} downstream cache group(s): {removed}")
    else:
        print(f"  [CACHE] Recording encoder fingerprint: {current} "
              f"(no downstream cache artifacts existed)")

    # Write this only after every stale artifact has been removed successfully.
    fp_path.write_text(current)


def encode_images(images, processor, model, device):
    feats = []
    for i in range(0, len(images), BATCH_SIZE):
        batch  = images[i:i + BATCH_SIZE]
        inputs = processor(images=batch, return_tensors="pt")
        pixels = inputs["pixel_values"].to(device)
        if device == "cuda": pixels = pixels.to(dtype=torch.float16)
        out = model.get_image_features(pixel_values=pixels)
        if not torch.is_tensor(out):
            if hasattr(out, "pooler_output"):       out = out.pooler_output
            elif hasattr(out, "image_embeds"):      out = out.image_embeds
            elif hasattr(out, "last_hidden_state"): out = out.last_hidden_state.mean(1)
        out = F.normalize(out.float(), dim=-1)
        feats.append(out.cpu())
    return torch.cat(feats, dim=0)


# TTA view transforms for embed_slots(). Every view here must be a
# transform that does NOT change which anatomical side/structure is being
# looked at -- knees are not left-right symmetric the way brains are, so
# a naive horizontal flip (valid for e.g. intracranial-aneurysm TTA) would
# risk teaching the model that a right-knee ACL finding is a left-knee one.
# Images going into this function have ALREADY been through
# normalise_laterality(), i.e. they are already canonicalized to a single
# consistent side -- so we only apply views that preserve that
# canonicalization: small rotations (do not touch left/right meaning) and
# a small in-plane zoom. We deliberately do NOT include FLIP_LEFT_RIGHT or
# FLIP_TOP_BOTTOM here, since either can invert laterality or
# superior/inferior anatomical meaning depending on plane.
TTA_ROTATIONS_DEG = (0, 3, -3)


def _tta_views(img):
    """Yield laterality-safe augmented copies of a single PIL image."""
    for deg in TTA_ROTATIONS_DEG:
        yield img.rotate(deg, resample=Image.BILINEAR, fillcolor=0) if deg else img


def encode_images_tta(images, processor, model, device, tta=False):
    """
    Same as encode_images(), optionally averaging predictions over a few
    laterality-safe views per image (rotations only -- see _tta_views).
    tta=False reproduces encode_images() exactly (used for training
    embeddings, which should stay single-pass/deterministic and cheap).
    tta=True is intended for test-time and pseudo-label-scoring embeddings.
    """
    if not tta:
        return encode_images(images, processor, model, device)

    per_view_feats = []
    for deg in TTA_ROTATIONS_DEG:
        view_images = [img.rotate(deg, resample=Image.BILINEAR, fillcolor=0) if deg else img
                       for img in images]
        per_view_feats.append(encode_images(view_images, processor, model, device))
    # average across views, then re-normalize (mean of unit vectors isn't
    # itself unit-norm)
    stacked = torch.stack(per_view_feats, dim=0)          # [n_views, N, D]
    averaged = stacked.mean(dim=0)                         # [N, D]
    return F.normalize(averaged, dim=-1)


def embed_slots(slots_df, dicom_df, processor, model, device,
                lat_map, out_dir, force=False, tta=False):
    series_to_files = (dicom_df.groupby("SeriesInstanceUID")["filepath"]
                       .apply(list).to_dict())
    present = slots_df[slots_df["presence_mask"] == 1].copy()
    index_rows = []
    done = failed = skipped = 0

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

        paths = [Path(p) for p in series_to_files.get(series, []) if Path(p).is_file()]
        paths = sort_slices(paths)
        paths = select_band(paths)

        images = []
        for p in paths:
            try: images.append(dicom_to_pil(p))
            except Exception: pass

        if not images:
            failed += 1
            continue

        images = normalise_laterality(images, plane, lat)

        try:
            feats = encode_images_tta(images, processor, model, device, tta=tta)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"embeddings": feats, "slot_name": slot,
                        "study_uid": study, "series_uid": series,
                        "laterality": lat, "plane": plane,
                        "n_slices": len(images), "tta": tta}, out_path)
            done += 1
            index_rows.append({"StudyInstanceUID": study, "SeriesInstanceUID": series,
                                "slot_name": slot, "embedding_file": str(out_path),
                                "presence_mask": 1})
        except Exception as e:
            print(f"\n  [WARN] {study[:20]}/{slot}: {e}")
            failed += 1

    print(f"  Embedded={done} Skipped={skipped} Failed={failed}")
    return pd.DataFrame(index_rows)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — MODEL
# ══════════════════════════════════════════════════════════════════════════════
def _study_images_for_finetune(study, train_slots, train_dcm, lat_map):
    """All slices across all present slots for one study, decoded to PIL,
    laterality-normalized -- one flat list, no per-slot separation.

    Deliberate simplification versus embed_slots(): this fine-tuning stage
    only needs the encoder itself to adapt, via ONE pooled embedding per
    study feeding a tiny linear head. The full slot-attention structure is
    preserved downstream, once Stage A/B re-consume the regenerated
    per-slot cached embeddings -- this function's only job is to give the
    unfrozen vision blocks real gradient signal, cheaply.
    """
    series_to_files = (train_dcm.groupby("SeriesInstanceUID")["filepath"]
                       .apply(list).to_dict())
    rows = train_slots[(train_slots["StudyInstanceUID"] == study) &
                        (train_slots["presence_mask"] == 1)]
    lat = lat_map.get(study)
    images = []
    for _, row in rows.iterrows():
        series = str(row["SeriesInstanceUID"])
        plane = str(row.get("Anatomical_Plane", "Unknown"))
        paths = [Path(p) for p in series_to_files.get(series, []) if Path(p).is_file()]
        paths = select_band(sort_slices(paths))
        slot_imgs = []
        for p in paths:
            try:
                slot_imgs.append(dicom_to_pil(p))
            except Exception:
                pass
        images.extend(normalise_laterality(slot_imgs, plane, lat))
    return images


def finetune_medsiglip_stage_a0(pretrain_ids, lbl, train_dcm, train_slots, train_lat,
                                 device, train_last_blocks=4, epochs=3, lr=1e-5,
                                 holdout_frac=0.1, seed=42):
    """Stage A0: unfreeze the last N MedSigLIP vision blocks and let them
    actually learn from knee pixels, on the full weak+real pool (safe --
    ~4,300+ studies, not the 58-study gold set, so overfitting risk here is
    low). Run ONCE. After this, the checkpoint it saves makes MedSigLIP a
    fixed function again (just a better-adapted one), so every stage after
    this -- embedding, Stage A NN, Stage A XGBoost, Stage B, test inference
    -- goes back to full cached speed automatically via load_medsiglip().

    Single random holdout, not full 5-fold CV, and few epochs by default:
    real backprop through a foundation model per study is far more
    expensive per step than the cached-embedding path everything else in
    this pipeline uses. This stage trades CV rigor for being affordable to
    run at all; it is a warm-start improvement, not the scored model.
    """
    finetuned_path = MODEL_DIR / "medsiglip_finetuned_vision.pt"
    if finetuned_path.exists():
        print(f"  Found existing {finetuned_path} — skipping Stage A0 fine-tuning")
        return finetuned_path

    print(f"\n── STAGE A0: Fine-tune MedSigLIP (last {train_last_blocks} blocks) "
          f"on {len(pretrain_ids)} studies ──")
    processor, model, _ = load_medsiglip()  # stock weights -- no checkpoint exists yet
    model = unfreeze_medsiglip_vision(model, train_last_blocks).train()

    # GradScaler requires FP32 master weights for any parameter it updates --
    # it can only unscale FP32 gradients ("Attempting to unscale FP16
    # gradients" otherwise). load_medsiglip() loads the whole model in fp16
    # for fast/cheap inference, which is fine for the frozen backbone, but
    # the newly-unfrozen last N blocks (+ post_layernorm) are about to be
    # optimized here, so upcast just those to fp32. Autocast still runs the
    # forward pass in fp16/mixed precision for speed; only the trainable
    # master weights need to be fp32.
    for p in model.vision_model.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    rng = np.random.default_rng(seed)
    shuffled = pretrain_ids.copy()
    rng.shuffle(shuffled)
    n_holdout = max(1, int(len(shuffled) * holdout_frac))
    holdout_ids, train_ids = shuffled[:n_holdout], shuffled[n_holdout:]

    lbl_idx = lbl.set_index("StudyInstanceUID")
    is_real = lbl_idx["is_real_label"].to_numpy() if "is_real_label" in lbl_idx.columns \
        else np.zeros(len(lbl_idx), dtype=bool)
    is_real_map = dict(zip(lbl_idx.index, is_real))

    head = nn.Linear(EMBED_DIM, len(TARGETS)).to(device)
    trainable = [p for p in model.vision_model.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    def _forward_study(study):
        images = _study_images_for_finetune(study, train_slots, train_dcm, train_lat)
        if not images:
            return None
        # A 3.5 GB MedSigLIP checkpoint plus gradients cannot safely process
        # every slice of a study at once on a 6 GB RTX 3050. Select a
        # deterministic, evenly-spaced subset; later epochs rotate the study
        # order while preserving reproducibility. This limits activations,
        # not the number of studies or supervision targets.
        if len(images) > MEDSIGLIP_FINETUNE_MAX_IMAGES:
            take = np.linspace(0, len(images) - 1, MEDSIGLIP_FINETUNE_MAX_IMAGES).round().astype(int)
            images = [images[i] for i in take]
        inputs = processor(images=images, return_tensors="pt")
        pixels = inputs["pixel_values"].to(device)
        if device == "cuda":
            pixels = pixels.to(dtype=torch.float16)
        feats = model.get_image_features(pixel_values=pixels)
        if not torch.is_tensor(feats):
            if hasattr(feats, "pooler_output"):
                feats = feats.pooler_output
            elif hasattr(feats, "image_embeds"):
                feats = feats.image_embeds
            elif hasattr(feats, "last_hidden_state"):
                feats = feats.last_hidden_state.mean(1)
            else:
                raise ValueError(
                    "get_image_features() returned an object with none of "
                    "pooler_output / image_embeds / last_hidden_state -- "
                    "cannot recover a feature tensor."
                )
        pooled = F.normalize(feats.float(), dim=-1).mean(dim=0, keepdim=True)  # [1, EMBED_DIM]
        return head(pooled).squeeze(0)  # [N_TARGETS]

    for epoch in range(epochs):
        model.train()
        losses = []
        for study in rng.permutation(train_ids):
            study = str(study)
            if study not in lbl_idx.index:
                continue
            y = torch.tensor(lbl_idx.loc[study, TARGETS].astype(float).to_numpy(),
                              dtype=torch.float32, device=device)
            w_val = 3.0 if is_real_map.get(study, False) else \
                float(np.clip(0.25 + 0.75 * (2.0 * np.abs(y.mean().item() - 0.5)), 0.25, 1.0))
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                logits = _forward_study(study)
                if logits is None:
                    continue
                loss = F.binary_cross_entropy_with_logits(logits, y) * w_val
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.item()))

        model.eval()
        Y, P = [], []
        with torch.no_grad():
            for study in holdout_ids:
                study = str(study)
                if study not in lbl_idx.index:
                    continue
                with torch.amp.autocast("cuda", enabled=device == "cuda"):
                    logits = _forward_study(study)
                if logits is None:
                    continue
                P.append(torch.sigmoid(logits).cpu().numpy())
                Y.append(lbl_idx.loc[study, TARGETS].astype(float).to_numpy())
        holdout_auc = auc_mean(np.vstack(Y), np.vstack(P)) if Y else float("nan")
        print(f"  [Stage A0] epoch {epoch+1}/{epochs}  loss={np.mean(losses):.5f}  "
              f"holdout_auc={holdout_auc:.5f}")

    torch.save({"vision_state_dict": model.vision_model.state_dict(),
                "train_last_blocks": train_last_blocks,
                "holdout_auc": holdout_auc}, finetuned_path)
    print(f"  Saved fine-tuned MedSigLIP vision tower: {finetuned_path}")
    del model, head
    torch.cuda.empty_cache()
    return finetuned_path


class SlotAttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(EMBED_DIM),
            nn.Linear(EMBED_DIM, PROJ_DIM),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.att   = nn.Linear(PROJ_DIM, len(TARGETS), bias=False)
        self.heads = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(PROJ_DIM), nn.Linear(PROJ_DIM, 64),
                          nn.GELU(), nn.Dropout(0.15), nn.Linear(64, 1))
            for _ in TARGETS
        ])
        prior = torch.zeros(len(TARGETS), N_SLOT)
        for t, target in enumerate(TARGETS):
            for s, val in enumerate(SLOT_PRIOR[target]):
                prior[t, s] = val * PRIOR_STRENGTH
        self.register_buffer("slot_prior", prior)

    def forward(self, x, mask, slot_indices):
        h       = self.proj(x)
        scores  = self.att(h).T
        scores  = scores + self.slot_prior[:, slot_indices]
        absent  = (mask[slot_indices] < 0.5)
        scores  = scores.masked_fill(absent.unsqueeze(0), -1e4)
        weights = torch.softmax(scores, dim=1)
        outputs = []
        for t in range(len(TARGETS)):
            pooled = (weights[t, :, None] * h).sum(dim=0)
            outputs.append(self.heads[t](pooled).squeeze())
        return torch.stack(outputs)


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
                        for _, r in rows.iterrows() if r["presence_mask"] == 1}
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


def bootstrap_auc_ci(y_true, y_pred, n_boot=1000, ci=0.90, seed=42):
    """
    Bootstrap confidence interval for auc_mean(y_true, y_pred), resampling
    STUDIES (rows) with replacement. With only 58 real-labeled studies to
    validate against, a single point-estimate AUC can look like real
    improvement (or real regression) when it's actually noise from which
    58 studies happened to be in this dataset -- this reports how wide
    that uncertainty band actually is, per the same logic RSNA aneurysm
    winners used (bootstrap intervals over n=58-scale gold sets).

    Returns (point_estimate, lo, hi) for the given ci (default 90%, i.e.
    5th/95th percentile of the bootstrap distribution). Targets with no
    positive/negative variation in a given resample are skipped for that
    resample (same convention as auc_mean).
    """
    rng = np.random.RandomState(seed)
    n = y_true.shape[0]
    point = auc_mean(y_true, y_pred)
    boot_vals = []
    for _ in range(n_boot):
        sample_idx = rng.randint(0, n, size=n)
        v = auc_mean(y_true[sample_idx], y_pred[sample_idx])
        if not np.isnan(v):
            boot_vals.append(v)
    if not boot_vals:
        return point, float("nan"), float("nan")
    lo_pct = (1.0 - ci) / 2.0 * 100
    hi_pct = (1.0 - (1.0 - ci) / 2.0) * 100
    lo = float(np.percentile(boot_vals, lo_pct))
    hi = float(np.percentile(boot_vals, hi_pct))
    return point, lo, hi


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 0 — MRNet auxiliary pretraining (ACL only)
#
# MRNet (Stanford, https://stanfordmlgroup.github.io/competitions/mrnet/) is
# a public knee-MRI dataset with ACL tear labels on ~1,370 exams -- far more
# than our own 58 real-labeled studies. Used ONLY for the ACL head: MRNet's
# meniscus label doesn't distinguish medial/lateral the way our targets do,
# so it is skipped entirely rather than injected as an ambiguous signal.
# Entirely optional -- if MRNET_ROOT doesn't exist, Stage 0 is skipped
# automatically (RUN_MRNET_PRETRAIN=False) and training proceeds exactly
# as before this was added.
# ══════════════════════════════════════════════════════════════════════════════
def load_mrnet_labels(mrnet_root: Path):
    """
    Reads MRNet's train-acl.csv / valid-acl.csv (columns: id, label -- no
    header). Returns a DataFrame with columns [split, id, label], id
    zero-padded to 4 digits to match MRNet's .npy filenames (e.g. '0000').
    """
    rows = []
    for split in ("train", "valid"):
        csv_path = mrnet_root / f"{split}-acl.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path, header=None, names=["id", "label"])
        df["id"] = df["id"].map(lambda i: str(i).zfill(4))
        df["split"] = split
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No train-acl.csv/valid-acl.csv found under {mrnet_root}")
    return pd.concat(rows, ignore_index=True)


def mrnet_npy_to_pil_list(npy_path: Path):
    """MRNet stores a whole series as one stacked .npy array [n_slices, H, W]."""
    arr = np.load(npy_path)
    images = []
    for sl in arr:
        sl = sl.astype(np.float32)
        lo, hi = np.percentile(sl, [1, 99])
        if hi <= lo:
            lo, hi = float(sl.min()), float(sl.max())
        if hi <= lo:
            sl8 = np.zeros_like(sl, dtype=np.uint8)
        else:
            sl8 = (np.clip((sl - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
        images.append(Image.fromarray(sl8).convert("RGB"))
    return images


def embed_mrnet(mrnet_root: Path, labels_df: pd.DataFrame, processor, model, device, out_dir: Path):
    """
    Embeds every MRNet exam's 3 planes through MedSigLIP, same encode_images()
    path used for our own DICOM data, so the resulting embeddings are
    directly compatible with SlotAttentionModel. One .pt file per
    exam/plane, mirroring the {study}/{series}__{slot}.pt layout embed_slots()
    already uses, so StudyDataset can load them unmodified. Single-pass
    (no TTA) -- this is pretraining data, same convention as our own
    train embeddings.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for row in tqdm(labels_df.itertuples(index=False), total=len(labels_df), desc="  Embedding MRNet"):
        study_id = f"mrnet_{row.split}_{row.id}"
        for plane, slot in MRNET_PLANE_TO_SLOT.items():
            npy_path = mrnet_root / row.split / plane / f"{row.id}.npy"
            if not npy_path.exists():
                continue
            out_path = out_dir / study_id / f"{plane}__{slot}.pt"
            if out_path.exists():
                index_rows.append({"StudyInstanceUID": study_id, "SeriesInstanceUID": plane,
                                   "slot_name": slot, "embedding_file": str(out_path),
                                   "presence_mask": 1})
                continue
            try:
                images = mrnet_npy_to_pil_list(npy_path)
                if not images:
                    continue
                feats = encode_images(images, processor, model, device)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"embeddings": feats, "slot_name": slot, "study_uid": study_id,
                           "series_uid": plane, "laterality": None, "plane": plane,
                           "n_slices": len(images)}, out_path)
                index_rows.append({"StudyInstanceUID": study_id, "SeriesInstanceUID": plane,
                                   "slot_name": slot, "embedding_file": str(out_path),
                                   "presence_mask": 1})
            except Exception as e:
                print(f"\n  [WARN] MRNet {study_id}/{plane}: {e}")
    return pd.DataFrame(index_rows)


def train_stage0_mrnet(mrnet_emb_idx: pd.DataFrame, mrnet_labels: pd.DataFrame,
                       device, epochs=15, lr=1e-4, weight_decay=1e-4):
    """
    Pretrains SlotAttentionModel on MRNet's ACL labels only. The other 11
    output heads receive zero gradient (loss is masked to the ACL column
    only) so they stay at their random initialization, exactly as if Stage
    0 had never run for them -- only the ACL head and the shared backbone
    (proj/attention layers) actually learn anything here. Stage A then
    continues training ALL heads (including ACL) from these weights, via
    train_fold's init_state_dict parameter.

    Returns the trained model (best-val-AUC state loaded), or None if
    there wasn't enough MRNet data to train on.
    """
    acl_idx = TARGETS.index("ACL")
    study_ids = sorted(mrnet_emb_idx["StudyInstanceUID"].unique())
    label_lookup = mrnet_labels.set_index(
        mrnet_labels.apply(lambda r: f"mrnet_{r['split']}_{r['id']}", axis=1))["label"]

    # build a labels_df shaped like the rest of the pipeline expects
    # (StudyInstanceUID + TARGETS columns), with all non-ACL targets set to
    # 0.0 -- never read for loss (masked out below), kept only so
    # StudyDataset's .loc[study, TARGETS] lookup doesn't break
    rows = []
    for sid in study_ids:
        if sid not in label_lookup.index:
            continue
        rec = {"StudyInstanceUID": sid}
        for t in TARGETS:
            rec[t] = float(label_lookup[sid]) if t == "ACL" else 0.0
        rows.append(rec)
    mrnet_lbl_df = pd.DataFrame(rows)
    if len(mrnet_lbl_df) < 10:
        print(f"  [WARN] Only {len(mrnet_lbl_df)} MRNet studies have embeddings -- skipping Stage 0")
        return None

    rng = np.random.RandomState(SAMPLE_SEED)
    ids_arr = mrnet_lbl_df["StudyInstanceUID"].values
    perm = rng.permutation(len(ids_arr))
    n_val = max(1, int(0.15 * len(ids_arr)))
    val_ids = set(ids_arr[perm[:n_val]])
    tr_ids  = set(ids_arr[perm[n_val:]])

    tr_ds = StudyDataset(mrnet_emb_idx[mrnet_emb_idx["StudyInstanceUID"].isin(tr_ids)],
                         mrnet_lbl_df[mrnet_lbl_df["StudyInstanceUID"].isin(tr_ids)])
    va_ds = StudyDataset(mrnet_emb_idx[mrnet_emb_idx["StudyInstanceUID"].isin(val_ids)],
                         mrnet_lbl_df[mrnet_lbl_df["StudyInstanceUID"].isin(val_ids)])
    print(f"  Stage 0 (MRNet ACL): train={len(tr_ds)}  val={len(va_ds)}")

    model = SlotAttentionModel().to(device)
    try:
        model = torch.compile(model)
    except Exception:
        pass
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    y_all = np.array([float(label_lookup[s]) for s in tr_ids if s in label_lookup.index])
    pos = y_all.sum()
    pw = max((len(y_all) - pos) / max(pos, 1), 1.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))
    best_state, best_auc = None, -np.inf

    for epoch in range(epochs):
        model.train()
        losses = []
        for i in np.random.permutation(len(tr_ds)):
            _, x, mask, idx, y = tr_ds.get(i)
            x, mask, idx = x.to(device), mask.to(device), idx.to(device)
            y_acl = y[acl_idx:acl_idx + 1].to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                out = model(x, mask, idx)
                # only the ACL head contributes to the loss -- everything
                # else stays untouched at random initialization
                loss = loss_fn(out[acl_idx:acl_idx + 1], y_acl)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.item()))

        model.eval()
        Y, P = [], []
        with torch.no_grad():
            for i in range(len(va_ds)):
                _, x, mask, idx, y = va_ds.get(i)
                out = torch.sigmoid(model(x.to(device), mask.to(device), idx.to(device)))
                P.append(float(out[acl_idx].cpu()))
                Y.append(float(y[acl_idx]))
        auc = roc_auc_score(Y, P) if len(np.unique(Y)) > 1 else float("nan")
        print(f"  [Stage 0] epoch {epoch+1:02d}  loss={np.mean(losses):.5f}  "
              f"val_acl_auc={auc if auc == auc else float('nan'):.5f}")
        if not np.isnan(auc) and auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    print(f"  Stage 0 best ACL val AUC: {best_auc:.5f}")
    return model


def train_fold(train_ds, val_ds, device, epochs, init_state_dict=None):
    model   = SlotAttentionModel().to(device)
    if init_state_dict is not None:
        # Seed from Stage 0 (MRNet ACL auxiliary pretraining) weights when
        # given, instead of the model's random initialization. Safe no-op
        # when init_state_dict is None (all existing call sites unaffected).
        model = load_state_dict_safe(model, init_state_dict)
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
                        for _, r in rows.iterrows() if r["presence_mask"] == 1}
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


# Guarded pseudo-labeling: after Stage B (fine-tuned on the 58 real
# studies) exists, use it to re-score the WEAK-labeled studies only --
# the 58 real studies are never touched, never re-scored, never
# overwritten. A weak label is only replaced where the image model is
# extremely confident (>= PSEUDO_LABEL_HIGH or <= PSEUDO_LABEL_LOW) AND
# that confident image-based call does not contradict a confident
# existing text-based signal (checked via the parser's own __conf column
# for that target). This is deliberately conservative: with only 58 real
# studies to validate the image model against, a wrong confident call
# from the image model has no cheap way to be caught later, so both
# gates (image confidence AND non-contradiction with confident text
# evidence) must pass before a label is touched.
PSEUDO_LABEL_HIGH = 0.95
PSEUDO_LABEL_LOW  = 0.05
# Text evidence counts as "confident" (and therefore not to be overridden
# by a contradicting image call) at this __conf threshold or above.
PSEUDO_LABEL_TEXT_CONF_GATE = 0.60


def apply_guarded_pseudo_labels(weak_df, weak_emb_idx, model_paths, device,
                                 targets=TARGETS):
    """
    weak_df       : rows for weak-labeled studies only (is_real_label=False),
                     must contain StudyInstanceUID, each f"{t}" soft label,
                     and each f"{t}__conf" text-confidence column.
    weak_emb_idx  : embedding index (StudyInstanceUID/SeriesInstanceUID/
                     slot_name/embedding_file/presence_mask) for those same
                     studies -- reuse the already-cached train_emb_idx,
                     filtered to weak study IDs; no re-embedding needed.
    model_paths   : Stage B fold_*.pt paths (fine-tuned on the 58 real
                     studies) -- the SAME model_paths used for test
                     inference, reused here rather than retrained.

    Returns (updated_df, report_df). updated_df is weak_df with any
    replaced label values written in place (only f"{t}" columns change;
    __conf columns are left as-is -- a replaced label keeps its original
    text-confidence value, since that value describes the parser's
    evidence, not the image model's). report_df has one row per target
    with counts of studies replaced toward 1 and toward 0, for full
    transparency before this is trusted.
    """
    study_ids_present = set(weak_emb_idx["StudyInstanceUID"].astype(str))
    scoreable = weak_df[weak_df["StudyInstanceUID"].astype(str).isin(study_ids_present)].copy()
    if scoreable.empty:
        print("  [WARN] No weak studies have cached embeddings -- skipping pseudo-labeling")
        return weak_df, pd.DataFrame(columns=["target", "n_set_to_1", "n_set_to_0", "n_blocked_by_text"])

    print(f"  Scoring {len(scoreable)} weak-labeled studies with Stage B model "
          f"({len(model_paths)} fold(s))...")
    img_study_ids, img_preds = run_inference(weak_emb_idx, model_paths, device)
    img_pred_by_study = {sid: img_preds[i] for i, sid in enumerate(img_study_ids)}

    updated_df = weak_df.copy()
    updated_df = updated_df.set_index("StudyInstanceUID", drop=False)
    report_rows = []

    for t_idx, t in enumerate(targets):
        conf_col = f"{t}__conf"
        has_conf = conf_col in updated_df.columns
        n_hi = n_lo = n_blocked = 0

        for sid in scoreable["StudyInstanceUID"].astype(str):
            if sid not in img_pred_by_study:
                continue
            img_score = float(img_pred_by_study[sid][t_idx])
            if img_score < PSEUDO_LABEL_HIGH and img_score > PSEUDO_LABEL_LOW:
                continue  # image model not confident enough either way

            image_says_present = img_score >= PSEUDO_LABEL_HIGH
            text_conf   = float(updated_df.at[sid, conf_col]) if has_conf else 0.0
            text_score  = float(updated_df.at[sid, t])
            text_says_present  = text_score >= 0.5
            text_is_confident  = text_conf >= PSEUDO_LABEL_TEXT_CONF_GATE

            if text_is_confident and (text_says_present != image_says_present):
                n_blocked += 1
                continue  # confident text signal disagrees -- do not override

            new_val = 1.0 if image_says_present else 0.0
            updated_df.at[sid, t] = new_val
            if image_says_present: n_hi += 1
            else:                  n_lo += 1

        report_rows.append({"target": t, "n_set_to_1": n_hi, "n_set_to_0": n_lo,
                             "n_blocked_by_text": n_blocked})

    updated_df = updated_df.reset_index(drop=True)
    report_df = pd.DataFrame(report_rows)
    return updated_df, report_df


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
# Blend search grid: alpha = NN weight, 1-alpha = XGBoost weight.
# 1.0 (NN-only) is always included and is the fallback-safe default if
# blending never helps. Finer step (0.05) and extended range (down to 0.0,
# i.e. XGBoost-only) versus the original 5-point 0.15-step grid -- this
# reuses the SAME already-computed OOF predictions, so the extra search
# cost is negligible (no retraining, just more arithmetic on arrays
# already in memory).
ALPHA_CANDIDATES = [round(x, 2) for x in np.arange(1.0, -0.001, -0.05)]


def study_pooled_embedding(study, emb_df):
    """
    Mean-pool ALL slice embeddings for one study across all present slots.
    Reuses the exact same cached .pt files the neural net loads — zero
    extra MedSigLIP forward passes needed.
    Returns (EMBED_DIM,) vector + (N_SLOT,) presence mask.
    """
    rows = emb_df[emb_df["StudyInstanceUID"] == study]
    slot_to_file = {r["slot_name"]: r["embedding_file"]
                    for _, r in rows.iterrows() if r["presence_mask"] == 1}
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
      Real (official) label -> weight 3.0, full trust.
      Weak (parsed) label   -> weight 0.25 to 1.0, scaled by confidence.

    conf, when given, is the parser's own per-target __conf score (from
    train_and_predict.py) -- this reflects how much actual evidence the
    parser/learned-model blend found in the report text for that specific
    target, and is strictly more informative than guessing confidence from
    how far the soft label sits from 0.5. When conf is None (older-format
    label CSV without __conf columns), falls back to the distance-from-0.5
    proxy exactly as before.
    """
    if conf is not None:
        conf_term = np.clip(conf, 0.0, 1.0)
    else:
        conf_term = (2.0 * np.abs(y_soft - 0.5)).clip(0, 1)
    w = np.where(
        is_real[:, None],
        3.0,
        0.25 + 0.75 * conf_term,
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
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if REQUIRE_CUDA and device.type != "cuda":
        raise SystemExit(
            "RSNA_REQUIRE_CUDA=1 but CUDA is not available "
            "(torch.cuda.is_available() returned False). Refusing to silently "
            "fall back to CPU. Check the GPU/driver allocation before training."
        )
    print(f"Device: {device}")
    # local flag (not the module-level XGB_AVAILABLE) so any failure inside
    # main() can disable the stacking layer for this run only, cleanly
    xgb_ok = XGB_AVAILABLE

    print("=" * 60)
    print("RSNA KNEE ABNORMALITY DETECTION")
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
    if "is_real_label" not in parsed.columns:
        raise RuntimeError(
            f"{PARSED_LABELS_CSV} has no 'is_real_label' column — "
            "run train_and_predict.py first to produce it."
        )

    real_df = parsed[parsed["is_real_label"]].copy()
    weak_df = parsed[~parsed["is_real_label"]].copy()
    if STUDY_LIST_CSV:
        study_list_path = Path(STUDY_LIST_CSV)
        requested = pd.read_csv(study_list_path, dtype={"StudyInstanceUID": str})
        if "StudyInstanceUID" not in requested.columns:
            raise RuntimeError(f"{study_list_path} must contain StudyInstanceUID")
        requested_ids = set(requested["StudyInstanceUID"].astype(str))
        missing_requested = requested_ids - set(parsed["StudyInstanceUID"].astype(str))
        if missing_requested:
            raise RuntimeError(f"Study list contains {len(missing_requested)} IDs absent from labels")
        parsed = parsed[parsed["StudyInstanceUID"].astype(str).isin(requested_ids)].copy()
        real_df = parsed[parsed["is_real_label"]].copy()
        weak_df = parsed[~parsed["is_real_label"]].copy()
        if len(parsed) != N_TOTAL_STUDIES:
            raise RuntimeError(f"Expected {N_TOTAL_STUDIES} studies from {study_list_path}, found {len(parsed)}")
        print(f"Using explicit study list: {study_list_path}")
    print(f"Real-labeled studies (true 0/1): {len(real_df)}")
    print(f"Weak-labeled studies (parser)  : {len(weak_df)}")

    n_weak_needed = max(0, N_TOTAL_STUDIES - len(real_df))
    if n_weak_needed > len(weak_df):
        print(f"[WARN] Only {len(weak_df)} weak-labeled studies available, "
              f"wanted {n_weak_needed} — using all of them.")
        n_weak_needed = len(weak_df)
    weak_sample = weak_df.sample(n=n_weak_needed, random_state=SAMPLE_SEED)

    labeled = pd.concat([real_df, weak_sample], ignore_index=True)
    print(f"Training pool total            : {len(labeled)} "
          f"({len(real_df)} real + {len(weak_sample)} weak)")
    print(f"Test studies    : {len(test_df)}")

    # ══ TRAIN PIPELINE ══════════════════════════════════════════
    if IS_KAGGLE:
        print("\n── TRAIN: On Kaggle — data already unzipped from attached Dataset, skipping unzip ──")
    else:
        print("\n── TRAIN: Ensure studies unzipped ──")
        ensure_studies_unzipped(labeled["StudyInstanceUID"], TRAIN_SERIES, TRAIN_SERIES_ZIP)

    print("\n── TRAIN: Scan DICOMs ──")
    # Cache filenames are fingerprinted by the actual study selection
    # (N_TOTAL_STUDIES, plus the study-list CSV path if one is set) so that
    # changing RSNA_N_TOTAL_STUDIES (e.g. 1000 -> 4349) gets its own fresh
    # cache set instead of silently reusing train_dicom_index.csv /
    # train_slots_cache.pkl / train_embedding_index.csv left over from a
    # prior run with a different study count under the same fixed filename.
    _study_set_tag = f"n{N_TOTAL_STUDIES}"
    if STUDY_LIST_CSV:
        _study_set_tag += "_" + Path(STUDY_LIST_CSV).stem
    _cached_dcm_idx = WORK_DIR / f"train_dicom_index_{_study_set_tag}.csv"
    _cached_emb_idx = WORK_DIR / f"train_embedding_index_{_study_set_tag}.csv"
    _cached_slots   = WORK_DIR / f"train_slots_cache_{_study_set_tag}.pkl"

    if FORCE_RESCAN:
        print("  RSNA_FORCE_RESCAN=1 — deleting cached DICOM index / slots / "
              "embedding index, rebuilding from scratch this run")
        for _f in (_cached_dcm_idx, _cached_slots, _cached_emb_idx):
            if _f.exists():
                _f.unlink()
                print(f"    removed {_f}")

    # DICOM index + slot assignment are needed both for embedding AND for
    # Stage A0 MedSigLIP fine-tuning below (which reads raw pixels directly,
    # bypassing any cached embeddings) -- so these load/compute
    # unconditionally, decoupled from the embedding-specific cache check
    # that follows. Previously these two caches were bundled into one
    # skip-check, which meant train_dcm/train_slots were only ever computed
    # when embeddings did NOT already exist -- fine before Stage A0 existed,
    # but Stage A0 needs raw DICOM access even on a machine that already has
    # cached embeddings from a prior stock-encoder run.
    if _cached_dcm_idx.exists():
        print(f"  Reusing cached DICOM index: {_cached_dcm_idx}")
        train_dcm = pd.read_csv(_cached_dcm_idx, dtype=str)
    else:
        train_study_filter = set(labeled["StudyInstanceUID"].astype(str).tolist())
        train_dcm = scan_dicoms(TRAIN_SERIES, train_series_csv, study_filter=train_study_filter)
        train_dcm.to_csv(_cached_dcm_idx, index=False)

    print("\n── TRAIN: Build slots ──")
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

    # ══ STAGE A0 — FINE-TUNE MEDSIGLIP ITSELF (run once; the checkpoint it
    # saves makes MedSigLIP a fixed, better-adapted function again, so every
    # stage after this returns to full cached speed automatically) ══
    if RUN_STAGE_A0_FINETUNE:
        finetune_pool_ids = np.array(sorted(
            set(labeled["StudyInstanceUID"].astype(str)) &
            set(train_slots["StudyInstanceUID"].astype(str))
        ))
        finetune_medsiglip_stage_a0(
            finetune_pool_ids, labeled, train_dcm, train_slots, train_lat,
            device.type, train_last_blocks=MEDSIGLIP_FINETUNE_BLOCKS,
            epochs=MEDSIGLIP_FINETUNE_EPOCHS,
        )

    # The encoder may have just changed (first fine-tune, or a different
    # checkpoint swapped in since the last run) -- auto-invalidate stale
    # cached embeddings rather than risk silently mixing embeddings
    # produced by two different encoders.
    check_embedding_cache_freshness(WORK_DIR)

    if _cached_emb_idx.exists():
        print("  Found existing embedding index — skipping embed step")
        train_emb_idx = pd.read_csv(_cached_emb_idx, dtype=str)
    else:
        print("\n── TRAIN: Embed ──")
        processor, model_enc, device_enc = load_medsiglip()
        train_emb_idx = embed_slots(train_slots, train_dcm, processor, model_enc,
                                     device_enc, train_lat, EMB_DIR / "train")
        train_emb_idx.to_csv(_cached_emb_idx, index=False)
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

        # ══ STAGE 0 — MRNet auxiliary ACL pretraining (optional) ══════
        # Only runs here, inside the branch where Stage A is actually
        # about to train -- if Stage A were already cached (the `if`
        # branch above), Stage 0's output would never be consumed, so
        # there's no reason to spend the time computing it.
        stage0_state_dict = None
        stage0_path = MODEL_DIR / "stage0_mrnet_pretrained.pt"
        if RUN_MRNET_PRETRAIN:
            print(f"\n── STAGE 0: MRNet auxiliary ACL pretraining (root: {MRNET_ROOT}) ──")
            if stage0_path.exists():
                print(f"  Found existing checkpoint at {stage0_path} — skipping Stage 0 training")
                stage0_ckpt = torch.load(stage0_path, map_location=device, weights_only=False)
                stage0_state_dict = stage0_ckpt["model_state_dict"]
            else:
                try:
                    mrnet_labels = load_mrnet_labels(MRNET_ROOT)
                    _mrnet_cached_emb_idx = MRNET_EMB_DIR / "mrnet_embedding_index.csv"
                    if _mrnet_cached_emb_idx.exists():
                        print("  Found existing MRNet embedding index — skipping embed step")
                        mrnet_emb_idx = pd.read_csv(_mrnet_cached_emb_idx, dtype=str)
                    else:
                        mrnet_processor, mrnet_model_enc, mrnet_device_enc = load_medsiglip()
                        mrnet_emb_idx = embed_mrnet(MRNET_ROOT, mrnet_labels, mrnet_processor,
                                                    mrnet_model_enc, mrnet_device_enc, MRNET_EMB_DIR)
                        mrnet_emb_idx.to_csv(_mrnet_cached_emb_idx, index=False)
                        del mrnet_model_enc
                        torch.cuda.empty_cache()

                    stage0_model = train_stage0_mrnet(mrnet_emb_idx, mrnet_labels, device)
                    if stage0_model is not None:
                        stage0_state_dict = stage0_model.state_dict()
                        torch.save({"model_state_dict": stage0_state_dict,
                                    "targets": TARGETS, "slot_names": SLOT_NAMES,
                                    "embed_dim": EMBED_DIM, "proj_dim": PROJ_DIM}, stage0_path)
                        print(f"  Saved Stage 0 weights: {stage0_path}")
                except Exception as e:
                    print(f"  [WARN] Stage 0 MRNet pretraining failed ({e}) — "
                          "Stage A will use random initialization as before.")
                    stage0_state_dict = None
        else:
            print(f"\n── STAGE 0: skipped (MRNET_ROOT not found: {MRNET_ROOT}) ──")

        stage_a_table  = pd.DataFrame({"study": pretrain_ids})
        stage_a_gkf    = GroupKFold(n_splits=5)
        stage_a_models = []
        best_stage_a_auc   = -1
        best_stage_a_model = None

        for sa_fold, (sa_tri, sa_vi) in enumerate(
            stage_a_gkf.split(stage_a_table, groups=stage_a_table.study), 1
        ):
            sa_tr_ids = pretrain_ids[sa_tri]
            sa_va_ids = pretrain_ids[sa_vi]
            pre_tr_ds = StudyDataset(emb[emb["StudyInstanceUID"].isin(sa_tr_ids)],
                                      lbl[lbl["StudyInstanceUID"].isin(sa_tr_ids)])
            pre_va_ds = StudyDataset(emb[emb["StudyInstanceUID"].isin(sa_va_ids)],
                                      lbl[lbl["StudyInstanceUID"].isin(sa_va_ids)])
            print(f"\nStage A FOLD {sa_fold}  train={len(pre_tr_ds)}  val={len(pre_va_ds)}")

            sa_model = train_fold(pre_tr_ds, pre_va_ds, device, epochs=30,
                                   init_state_dict=stage0_state_dict)

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

            Y_pool = lbl.set_index("StudyInstanceUID").loc[pretrain_ids, TARGETS].values.astype(np.float32)
            is_real_pool = np.array([sid in real_ids_common for sid in pretrain_ids])
            conf_cols = [f"{t}__conf" for t in TARGETS]
            if all(c in lbl.columns for c in conf_cols):
                conf_pool = lbl.set_index("StudyInstanceUID").loc[pretrain_ids, conf_cols].values.astype(np.float32)
                print("  Using parser __conf columns for weak-label sample weighting")
            else:
                conf_pool = None
                print("  [WARN] __conf columns not found in labels CSV — "
                      "falling back to distance-from-0.5 weighting")
            W_pool = soft_label_weight(Y_pool, is_real_pool, conf=conf_pool)
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
                    xgb_sa_gkf.split(xgb_sa_table, groups=xgb_sa_table.study), 1
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
        for fold, (tri, vi) in enumerate(gkf.split(table, groups=table.study), 1):
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
            torch.save({"model_state_dict": fold_model.state_dict(),
                        "targets": TARGETS, "slot_names": SLOT_NAMES,
                        "embed_dim": EMBED_DIM, "proj_dim": PROJ_DIM},
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

    # ── Bootstrap CI: with only 58 real studies, a point-estimate AUC can
    #    look like real improvement/regression when it's actually noise
    #    from this particular set of 58 -- report the uncertainty band.
    _real_Y = real_lbl.set_index("StudyInstanceUID").loc[ids, TARGETS].values.astype(np.float32)
    _pt, _lo, _hi = bootstrap_auc_ci(_real_Y, oof)
    print(f"\nOOF Mean AUC 90% bootstrap CI (n=58, 1000 resamples): "
          f"{_pt:.4f}  [{_lo:.4f}, {_hi:.4f}]")
    print("  (treat AUC differences smaller than this interval's width as "
          "noise, not evidence of improvement)")

    # ══ GUARDED PSEUDO-LABELING — refine weak labels using Stage B ═════
    # Uses the fine-tuned (Stage B) model, not Stage A, since Stage B has
    # actually seen the 58 real labels. Only touches weak-labeled studies;
    # the 58 real studies above are never re-scored or overwritten. See
    # apply_guarded_pseudo_labels() for the exact confidence gates.
    if RUN_PSEUDO_LABELING:
        print("\n── PSEUDO-LABELING: refining weak labels with Stage B model ──")
        pseudo_model_paths = sorted(MODEL_DIR.glob("fold_*.pt"))
        weak_ids_common = common - real_ids_common
        weak_lbl_common = lbl[lbl["StudyInstanceUID"].astype(str).isin(weak_ids_common)].copy()
        weak_emb_common = emb[emb["StudyInstanceUID"].astype(str).isin(weak_ids_common)].copy()

        updated_weak_lbl, pseudo_report = apply_guarded_pseudo_labels(
            weak_lbl_common, weak_emb_common, pseudo_model_paths, device)

        print("\n  Pseudo-label replacement report (weak studies only, 58 real untouched):")
        print(pseudo_report.to_string(index=False))
        total_replaced = int(pseudo_report["n_set_to_1"].sum() + pseudo_report["n_set_to_0"].sum())
        total_blocked  = int(pseudo_report["n_blocked_by_text"].sum())
        print(f"  Total labels replaced: {total_replaced}  "
              f"(blocked by confident contradicting text: {total_blocked})")

        pseudo_out_path = WORK_DIR / "weak_labels_pseudo_refined.csv"
        updated_weak_lbl.to_csv(pseudo_out_path, index=False)
        print(f"  Saved refined weak labels to: {pseudo_out_path}")
        print("  NOTE: this run's Stage A/B models were trained on the ORIGINAL "
              "weak labels -- refined labels take effect on the NEXT run that "
              "reads PARSED_LABELS_CSV (or point PARSED_LABELS_CSV at this file).")

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

    if not RUN_TEST_INFERENCE:
        print("\n" + "=" * 60)
        print("TRAINING + OOF COMPLETE (test inference disabled for this run)")
        print("=" * 60)
        print(f"OOF AUC    : {mean_auc:.5f}")
        print(f"Artifacts  : {WORK_DIR}")
        return

    # ══ TEST PIPELINE ═══════════════════════════════════════════
    print("\n── TEST: Scan DICOMs ──")
    test_dcm = scan_dicoms(TEST_SERIES, test_series_csv)

    print("\n── TEST: Build slots ──")
    test_lat   = detect_laterality(test_dcm)
    test_slots = assign_slots(test_dcm)
    test_slots["laterality"] = test_slots["StudyInstanceUID"].map(test_lat)

    print("\n── TEST: Embed (TTA on -- rotation-averaged, laterality-safe) ──")
    processor, model_enc, device_enc = load_medsiglip()
    test_emb_idx = embed_slots(test_slots, test_dcm, processor, model_enc,
                                device_enc, test_lat, EMB_DIR / "test", tta=TEST_TTA)
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
