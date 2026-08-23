#!/usr/bin/env python3
"""RSNA Knee: genuine, OOF-selected image model.

This script deliberately excludes public-LB calibration, test-derived features,
report-text inference, hidden payloads, and inherited ensemble recipes.

Required inputs
---------------
--root: folder containing train.csv, test.csv, train_series.csv, test_series.csv,
        train_series/<study>/<series>/*.dcm and test_series/<study>/<series>/*.dcm
--labels-csv: training-only labels with StudyInstanceUID and the 12 target columns.
              Optional: PatientID (or another --group-col) and <target>__confidence.

The labels file must be auditable. Do not use labels derived from test reports or
weights trained with held-out-fold records.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]
SLOTS = [
    ("SAG_FS", "Sagittal", True),
    ("COR_FS", "Coronal", True),
    ("AX_FS", "Axial", True),
    ("SAG_NFS", "Sagittal", False),
    ("COR_NFS", "Coronal", False),
    ("AX_NFS", "Axial", False),
]


@dataclass
class Config:
    root: str
    labels_csv: str
    output: str = "runs/genuine_rsna"
    group_col: str = "StudyInstanceUID"
    folds: int = 5
    seed: int = 20260822
    image_size: int = 256
    slices_per_slot: int = 12
    slice_sampling: str = "middle"  # middle or full
    middle_fraction: float = 0.60
    canonicalize_laterality: bool = True
    geometry_log: str = "runs/geometry_fallbacks.jsonl"
    batch_size: int = 3
    workers: int = 6
    epochs: int = 12
    lr: float = 2e-4
    weight_decay: float = 1e-4
    pretrained: bool = True
    compile: bool = False
    resume: bool = False


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require_columns(df: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def load_tables(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path(cfg.root)
    train = pd.read_csv(root / "train.csv", dtype={"StudyInstanceUID": str})
    test = pd.read_csv(root / "test.csv", dtype={"StudyInstanceUID": str})
    tr_series = pd.read_csv(root / "train_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    te_series = pd.read_csv(root / "test_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    labels_path = Path(cfg.labels_csv)
    labels = pd.read_csv(labels_path, dtype={"StudyInstanceUID": str})
    require_columns(labels, ["StudyInstanceUID", *TARGETS], "labels CSV")
    if labels.StudyInstanceUID.duplicated().any():
        raise ValueError("labels CSV must contain one row per StudyInstanceUID")
    # Fail closed: labels must exactly cover train studies; no silent report parser fallback.
    if set(train.StudyInstanceUID) != set(labels.StudyInstanceUID):
        only_train = len(set(train.StudyInstanceUID) - set(labels.StudyInstanceUID))
        only_labels = len(set(labels.StudyInstanceUID) - set(train.StudyInstanceUID))
        raise ValueError(f"labels/train ID mismatch: train_only={only_train}, labels_only={only_labels}")
    labels[TARGETS] = labels[TARGETS].apply(pd.to_numeric, errors="raise")
    if not labels[TARGETS].isin([0, 1]).all().all():
        raise ValueError("all supervised labels must be binary 0/1; mask uncertain labels before training")
    train = train[["StudyInstanceUID"]].merge(labels, on="StudyInstanceUID", how="inner", validate="one_to_one")
    if cfg.group_col not in train.columns:
        # Study-level grouping is the conservative fallback when no patient ID exists.
        train[cfg.group_col] = train.StudyInstanceUID
        print(f"WARNING: {cfg.group_col!r} absent; grouping at study level.")
    return train, test, tr_series, te_series


def group_multilabel_folds(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Greedy, deterministic group-wise multilabel allocation.

    This never splits a group. It balances rare labels first; inspect the saved
    fold audit before using results in a final experiment.
    """
    groups = df.groupby(cfg.group_col, sort=True)
    rows = []
    for gid, g in groups:
        y = g[TARGETS].max(axis=0).to_numpy(np.int64)
        rows.append((str(gid), y, list(g.StudyInstanceUID.astype(str))))
    if len(rows) < cfg.folds:
        raise ValueError("fewer groups than folds")
    total = np.sum([r[1] for r in rows], axis=0)
    if (total < cfg.folds).any():
        bad = [TARGETS[i] for i, n in enumerate(total) if n < cfg.folds]
        raise ValueError(f"too few positives for {cfg.folds} folds: {bad}; lower --folds")
    rng = np.random.default_rng(cfg.seed)
    # Rare-label burden first; random tie-break remains deterministic from seed.
    rarity = 1.0 / np.maximum(total, 1)
    order = sorted(rows, key=lambda r: (-(r[1] * rarity).sum(), -r[1].sum(), rng.random()))
    counts = np.zeros((cfg.folds, len(TARGETS)), dtype=np.float64)
    sizes = np.zeros(cfg.folds, dtype=np.int64)
    assigned: Dict[str, int] = {}
    desired = total / cfg.folds
    for gid, y, ids in order:
        costs = []
        for fold in range(cfg.folds):
            new = counts.copy(); new[fold] += y
            # Strong label balance; mild study-count balance.
            label_cost = ((new - desired) ** 2 / np.maximum(desired, 1)).sum()
            size_cost = ((sizes[fold] + len(ids)) - len(df) / cfg.folds) ** 2 / max(1, len(df) / cfg.folds)
            costs.append(label_cost + 0.1 * size_cost)
        fold = int(np.argmin(costs))
        counts[fold] += y; sizes[fold] += len(ids); assigned[gid] = fold
    out = df[["StudyInstanceUID", cfg.group_col, *TARGETS]].copy()
    out["fold"] = out[cfg.group_col].astype(str).map(assigned).astype(int)
    # Hard anti-leakage assertion.
    assert out.groupby(cfg.group_col).fold.nunique().max() == 1
    return out


def dicom_sort_key(path: Path) -> Tuple[float, float, str]:
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True,
                              specific_tags=["ImagePositionPatient", "ImageOrientationPatient", "InstanceNumber"])
        ipp = np.asarray(getattr(ds, "ImagePositionPatient", []), dtype=np.float64)
        iop = np.asarray(getattr(ds, "ImageOrientationPatient", []), dtype=np.float64)
        if len(ipp) == 3 and len(iop) == 6 and np.isfinite(ipp).all() and np.isfinite(iop).all():
            normal = np.cross(iop[:3], iop[3:])
            return (float(np.dot(ipp, normal)), float(getattr(ds, "InstanceNumber", 0)), path.name)
        # Caller records this as a geometry fallback; InstanceNumber is only a fallback order.
        return (0.0, float(getattr(ds, "InstanceNumber", 0)), path.name)
    except Exception:
        return (0.0, 0.0, path.name)


def read_dicom(path: Path, size: int) -> np.ndarray:
    ds = pydicom.dcmread(str(path), force=True)
    x = ds.pixel_array.astype(np.float32)
    x = x * float(getattr(ds, "RescaleSlope", 1.0) or 1.0) + float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    lo, hi = np.percentile(x, [1, 99])
    x = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
    t = torch.from_numpy(x)[None, None]
    return F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)[0, 0].numpy()


def choose_slots(series: pd.DataFrame) -> Dict[str, str]:
    s = series.copy()
    for col in ("Fat_Suppression", "Fluid_Sensitive"):
        if col not in s:
            s[col] = 0
        s[col] = pd.to_numeric(s[col], errors="coerce").fillna(0).astype(int) > 0
    if "Anatomical_Plane" not in s:
        raise ValueError("series table requires Anatomical_Plane")
    result: Dict[str, str] = {}
    for name, plane, fs in SLOTS:
        candidates = s[(s.Anatomical_Plane.astype(str) == plane) & (s.Fat_Suppression == fs)]
        if len(candidates) == 0 and not fs:
            candidates = s[(s.Anatomical_Plane.astype(str) == plane) & (~s.Fat_Suppression)]
        if len(candidates):
            result[name] = str(candidates.iloc[0].SeriesInstanceUID)
    return result


def canonicalize_laterality(image: np.ndarray, laterality: str, enabled: bool = True) -> np.ndarray:
    """Map right-knee pixels to the canonical left-knee frame."""
    if enabled and str(laterality).upper().strip() in {"R", "RIGHT"}:
        return image[..., ::-1].copy()
    return image


def dicom_orientation_info(path: Path) -> Tuple[str, bool]:
    """Return laterality and whether geometry metadata is complete."""
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True,
                              specific_tags=["Laterality", "ImagePositionPatient", "ImageOrientationPatient"])
        lat = str(getattr(ds, "Laterality", "")).upper().strip()
        ipp = np.asarray(getattr(ds, "ImagePositionPatient", []), dtype=np.float64)
        iop = np.asarray(getattr(ds, "ImageOrientationPatient", []), dtype=np.float64)
        return lat, bool(len(ipp) == 3 and len(iop) == 6 and np.isfinite(ipp).all() and np.isfinite(iop).all())
    except Exception:
        return "", False


class KneeStudyDataset(Dataset):
    def __init__(self, studies: pd.DataFrame, series: pd.DataFrame, root: Path, split: str, cfg: Config, train: bool = False):
        self.df = studies.reset_index(drop=True)
        self.series = {str(k): v for k, v in series.groupby("StudyInstanceUID", sort=False)}
        self.root, self.split, self.cfg, self.train = root, split, cfg, train

    def __len__(self) -> int:
        return len(self.df)

    def _load_slot(self, study: str, series_id: str) -> Tuple[np.ndarray, np.ndarray]:
        files = sorted((self.root / self.split / study / series_id).glob("*.dcm"), key=dicom_sort_key)
        k, h = self.cfg.slices_per_slot, self.cfg.image_size
        out = np.zeros((k, h, h), np.float32); valid = np.zeros(k, np.float32)
        if not files:
            return out, valid
        lat, geometry_ok = dicom_orientation_info(files[len(files)//2])
        if not geometry_ok:
            log = Path(self.cfg.geometry_log)
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"study_id": study, "series_id": series_id, "n_files": len(files), "reason": "missing_or_invalid_geometry"}) + "\n")
        if self.cfg.slice_sampling == "middle":
            frac = min(max(float(self.cfg.middle_fraction), 0.1), 1.0)
            lo = int(round((1.0-frac) * 0.5 * (len(files)-1)))
            hi = len(files) - 1 - lo
        elif self.cfg.slice_sampling == "full":
            lo, hi = 0, len(files)-1
        else:
            raise ValueError("slice_sampling must be 'middle' or 'full'")
        idx = np.linspace(lo, hi, k).round().astype(int)
        for i, j in enumerate(idx):
            try:
                out[i] = read_dicom(files[int(j)], h); valid[i] = 1.0
                out[i] = canonicalize_laterality(out[i], lat, self.cfg.canonicalize_laterality)

            except Exception:
                pass
        return out, valid

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        study = str(row.StudyInstanceUID)
        images = np.zeros((len(SLOTS), self.cfg.slices_per_slot, self.cfg.image_size, self.cfg.image_size), np.float32)
        valid = np.zeros((len(SLOTS), self.cfg.slices_per_slot), np.float32)
        slot_map = choose_slots(self.series.get(study, pd.DataFrame(columns=["Anatomical_Plane", "SeriesInstanceUID"])))
        for slot_i, (name, _, _) in enumerate(SLOTS):
            if name in slot_map:
                images[slot_i], valid[slot_i] = self._load_slot(study, slot_map[name])
        # Do not randomly mirror studies: that swaps medial/lateral anatomy
        # without swapping target semantics. Laterality canonicalization above
        # is deterministic and is the only horizontal orientation transform.
        item = {"image": torch.from_numpy(images), "valid": torch.from_numpy(valid), "study_id": study}
        if all(t in row.index for t in TARGETS):
            item["target"] = torch.tensor(row[TARGETS].to_numpy(np.float32))
            conf = np.ones(len(TARGETS), np.float32)
            for j, t in enumerate(TARGETS):
                c = f"{t}__confidence"
                if c in row.index and pd.notna(row[c]):
                    conf[j] = float(row[c])
            item["confidence"] = torch.tensor(np.clip(conf, 0, 1))
        return item


class StudyMIL(nn.Module):
    """2.5D ResNet + target-query attention over all slot/slice tokens."""
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet18(weights=weights)
        self.encoder = nn.Sequential(*list(net.children())[:-1])
        dim = 512
        self.slot = nn.Parameter(torch.randn(len(SLOTS), dim) * 0.02)
        self.pos = nn.Parameter(torch.randn(64, dim) * 0.02)
        self.query = nn.Parameter(torch.randn(len(TARGETS), dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads=8, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(0.2), nn.Linear(dim, len(TARGETS)))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def forward(self, image: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        # image [B, slots, slices, H, W]; form 3-slice local context at each slice.
        b, s, k, h, w = image.shape
        pad = F.pad(image, (0, 0, 0, 0, 1, 1), mode="replicate")
        tri = torch.stack([pad[:, :, i:i + k] for i in range(3)], dim=3).reshape(b * s * k, 3, h, w)
        tri = (tri - self.mean) / self.std
        feat = self.encoder(tri).flatten(1).reshape(b, s * k, -1)
        token_valid = valid.reshape(b, s * k).bool()
        # Make completely absent studies numerically safe while preserving their mask.
        empty = ~token_valid.any(1)
        if empty.any():
            token_valid = token_valid.clone(); token_valid[empty, 0] = True
        slot_index = torch.arange(s, device=image.device).repeat_interleave(k)
        pos_index = torch.arange(k, device=image.device).repeat(s)
        feat = feat + self.slot[slot_index][None] + self.pos[pos_index][None]
        q = self.query[None].expand(b, -1, -1)
        z, _ = self.attn(q, feat, feat, key_padding_mask=~token_valid, need_weights=False)
        return self.head(z).diagonal(dim1=1, dim2=2)


def weighted_bce(logits: torch.Tensor, target: torch.Tensor, confidence: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none", pos_weight=pos_weight)
    return (raw * confidence).sum() / confidence.sum().clamp_min(1.0)


def macro_auc(y: np.ndarray, p: np.ndarray) -> Tuple[float, Dict[str, float]]:
    scores: Dict[str, float] = {}
    for i, t in enumerate(TARGETS):
        if np.unique(y[:, i]).size != 2:
            raise ValueError(f"cannot compute ROC-AUC for {t}: only one class in OOF")
        scores[t] = float(roc_auc_score(y[:, i], p[:, i]))
    return float(np.mean(list(scores.values()))), scores


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[List[str], np.ndarray]:
    model.eval(); ids: List[str] = []; out: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            x, valid = batch["image"].to(device, non_blocking=True), batch["valid"].to(device, non_blocking=True)
            with autocast(enabled=device.type == "cuda"):
                logits = model(x, valid)
            out.append(torch.sigmoid(logits).float().cpu().numpy()); ids.extend(batch["study_id"])
    return ids, np.concatenate(out, axis=0)


def train_fold(fold: int, train_df: pd.DataFrame, val_df: pd.DataFrame, tr_series: pd.DataFrame, root: Path, cfg: Config, out: Path) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_ds = KneeStudyDataset(train_df, tr_series, root, "train_series", cfg, train=True)
    va_ds = KneeStudyDataset(val_df, tr_series, root, "train_series", cfg, train=False)
    tr_loader = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers, pin_memory=True, persistent_workers=cfg.workers > 0)
    va_loader = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers, pin_memory=True, persistent_workers=cfg.workers > 0)
    model = StudyMIL(cfg.pretrained).to(device)
    if cfg.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    positives = train_df[TARGETS].sum().to_numpy(np.float32)
    negatives = len(train_df) - positives
    # Conservative, clipped weighting; log and compare it by OOF, not assumption.
    pos_weight = torch.tensor(np.clip(np.sqrt(negatives / np.maximum(positives, 1)), 1.0, 5.0), device=device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs)
    scaler = GradScaler(enabled=device.type == "cuda")
    best, best_p = -np.inf, None
    for epoch in range(cfg.epochs):
        model.train()
        for batch in tr_loader:
            x = batch["image"].to(device, non_blocking=True); valid = batch["valid"].to(device, non_blocking=True)
            y = batch["target"].to(device, non_blocking=True); conf = batch["confidence"].to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                loss = weighted_bce(model(x, valid), y, conf, pos_weight)
            scaler.scale(loss).backward(); scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim); scaler.update()
        sched.step()
        _, p = predict(model, va_loader, device)
        auc, _ = macro_auc(val_df[TARGETS].to_numpy(np.int8), p)
        print(f"fold={fold} epoch={epoch + 1}/{cfg.epochs} val_macro_auc={auc:.5f}")
        if auc > best:
            best, best_p = auc, p.copy()
            torch.save({"fold": fold, "state_dict": model.state_dict(), "config": asdict(cfg), "val_macro_auc": auc}, out / f"fold_{fold}.pt")
    assert best_p is not None
    return best_p


def train(cfg: Config) -> None:
    seed_everything(cfg.seed)
    out = Path(cfg.output); out.mkdir(parents=True, exist_ok=True)
    train_df, test_df, tr_series, te_series = load_tables(cfg)
    folds = group_multilabel_folds(train_df, cfg)
    folds.to_csv(out / "folds.csv", index=False)
    audit = folds.groupby("fold")[TARGETS].agg(["sum", "mean"])
    audit.to_csv(out / "fold_audit.csv")
    (out / "run_config.json").write_text(json.dumps({"config": asdict(cfg), "labels_sha256": sha256(Path(cfg.labels_csv))}, indent=2))
    oof = np.full((len(train_df), len(TARGETS)), np.nan, np.float32)
    for fold in range(cfg.folds):
        va_ids = set(folds.loc[folds.fold == fold, "StudyInstanceUID"])
        tr = train_df[~train_df.StudyInstanceUID.isin(va_ids)].reset_index(drop=True)
        va = train_df[train_df.StudyInstanceUID.isin(va_ids)].reset_index(drop=True)
        assert set(tr.StudyInstanceUID).isdisjoint(set(va.StudyInstanceUID))
        if set(tr[cfg.group_col].astype(str)) & set(va[cfg.group_col].astype(str)):
            raise RuntimeError("group leakage detected")
        p = train_fold(fold, tr, va, tr_series, Path(cfg.root), cfg, out)
        positions = train_df.index[train_df.StudyInstanceUID.isin(va_ids)].to_numpy()
        # Restore exact train_df order rather than trusting arbitrary validation ordering.
        lookup = {sid: i for i, sid in enumerate(va.StudyInstanceUID.astype(str))}
        oof[positions] = np.stack([p[lookup[str(sid)]] for sid in train_df.loc[positions, "StudyInstanceUID"]])
    if not np.isfinite(oof).all():
        raise RuntimeError("OOF is incomplete")
    macro, per_target = macro_auc(train_df[TARGETS].to_numpy(np.int8), oof)
    oof_df = train_df[["StudyInstanceUID", cfg.group_col, *TARGETS]].copy()
    oof_df["fold"] = folds.set_index("StudyInstanceUID").loc[oof_df.StudyInstanceUID, "fold"].to_numpy()
    for j, t in enumerate(TARGETS): oof_df[f"pred_{t}"] = oof[:, j]
    oof_df.to_csv(out / "oof_predictions.csv", index=False)
    (out / "metrics.json").write_text(json.dumps({"macro_oof_auc": macro, "per_target_auc": per_target}, indent=2))
    print(json.dumps({"macro_oof_auc": macro, "per_target_auc": per_target}, indent=2))
    print("Training complete. Do not tune against public LB; compare subsequent changes against this OOF file.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True); p.add_argument("--labels-csv", required=True)
    p.add_argument("--output", default="runs/genuine_rsna"); p.add_argument("--group-col", default="StudyInstanceUID")
    p.add_argument("--folds", type=int, default=5); p.add_argument("--seed", type=int, default=20260822)
    p.add_argument("--image-size", type=int, default=256); p.add_argument("--slices-per-slot", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=3); p.add_argument("--workers", type=int, default=6)
    p.add_argument("--epochs", type=int, default=12); p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4); p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--compile", action="store_true")
    args = p.parse_args()
    cfg = Config(**{k: v for k, v in vars(args).items() if k not in {"no_pretrained"}})
    cfg.pretrained = not args.no_pretrained
    train(cfg)


if __name__ == "__main__":
    main()
