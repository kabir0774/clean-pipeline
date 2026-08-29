#!/usr/bin/env python3
"""
RSNA Knee 3D branch: volumetric CNN trained alongside (not instead of) the
MedSigLIP pipeline, producing an OOF matrix that plugs into the existing
alpha blend as an additional prediction stream.

DESIGN NOTES
------------
1. Why a separate branch, not a replacement.
   The MedSigLIP path encodes each slice independently -- through-plane
   reasoning happens only afterwards, on top of already-independent
   features. A 3D convolution sees through-plane texture directly. Those
   are different failure modes, which is exactly what makes an ensemble
   worth building: variance reduction needs uncorrelated errors, not more
   models. If this branch turns out weak, the existing OOF-driven blend
   search weights it toward zero automatically, so the downside is bounded.

2. Why (study, slot) volumes rather than one volume per study.
   Each study contributes up to 6 volumes (one per named slot), each
   carrying that study's labels. This multiplies the training pool roughly
   4x -- the binding constraint here is data, not capacity -- and it makes
   the auxiliary slot-classification task meaningful, which it would not be
   if every sample came from the same slot. At inference a study's slot
   predictions are averaged back into one study-level prediction.

3. Why auxiliary supervision.
   Across the 2021-2025 RSNA winners, auxiliary localization/segmentation
   supervision produced larger measured gains than backbone choice (the
   2025 ablation put it at 0.026 AUC). We have no segmentation masks for
   knee, but slot identity is a free label already computed by the main
   pipeline's slot assignment. Predicting "which of the 6 acquisition types
   is this" forces the encoder to represent plane, fat-saturation and
   fluid-weighting explicitly, rather than only whatever separates diseased
   from healthy on the dominant slot.

4. Honest limitation.
   58 gold labels. A 3D ResNet-34 is ~63M parameters. Stage B fine-tuning
   on 58 studies will overfit regardless of initialization; the weak-label
   Stage A pool is doing most of the real work. Expect this branch to be
   useful as a blend member, not as a standalone winner.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Dataset

from rsna_volume3d import load_volume

TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
           'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
           'Contusion', 'Fracture']
SLOT_NAMES = ['SAG_FLUID_FS', 'COR_FLUID_FS', 'AX_FLUID_FS',
              'SAG_FLUID_NOFS', 'COR_T1', 'SAG_T1']
N_SLOT = len(SLOT_NAMES)


# ═══════════════════════════════════════════════════════════════════════════
# BACKBONES
# ═══════════════════════════════════════════════════════════════════════════
class BasicBlock3D(nn.Module):
    """MedicalNet-compatible 3D basic block (matches Tencent/MedicalNet's
    resnet.py so their published checkpoints load without renaming)."""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet3D(nn.Module):
    """3D ResNet matching MedicalNet's layer naming, so published Med3D
    checkpoints load directly. Encoder only -- no segmentation decoder."""

    def __init__(self, layers, in_channels=1):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7,
                               stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.out_dim = 512

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm3d(planes))
        layers = [BasicBlock3D(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        layers += [BasicBlock3D(planes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return F.adaptive_avg_pool3d(x, 1).flatten(1)


def load_medicalnet(arch="resnet34", checkpoint=None):
    """Build a 3D ResNet and, if a MedicalNet checkpoint is given, load it.

    Reports the fraction of parameters that actually matched and REFUSES a
    load below 50%. A silent partial load is the dangerous failure here:
    the model would train from mostly-random weights while the log claimed
    it was pretrained, and the resulting weaker OOF would be blamed on the
    architecture rather than on the load having failed.
    """
    layers = {"resnet18": [2, 2, 2, 2], "resnet34": [3, 4, 6, 3]}[arch]
    model = ResNet3D(layers, in_channels=1)
    if not checkpoint:
        print(f"  [3D] {arch}: random init (no MedicalNet checkpoint given)")
        return model, model.out_dim

    p = Path(checkpoint)
    if not p.is_file():
        raise FileNotFoundError(f"MedicalNet checkpoint not found: {p}")
    payload = torch.load(p, map_location="cpu", weights_only=False)
    state = payload.get("state_dict", payload)
    state = {str(k).removeprefix("module."): v for k, v in state.items()}

    own = model.state_dict()
    usable = {k: v for k, v in state.items()
              if k in own and tuple(own[k].shape) == tuple(v.shape)}
    matched = len(usable) / max(len(own), 1)
    model.load_state_dict(usable, strict=False)
    print(f"  [3D] {arch}: loaded {len(usable)}/{len(own)} tensors "
          f"({matched:.1%}) from {p.name}")

    # 0.9, not 0.5. ResNet-18 and ResNet-34 share identical shapes through
    # their early blocks, so loading an 18 checkpoint into a 34 still matches
    # ~56% of tensors -- comfortably past a 50% bar while leaving every
    # deeper block randomly initialized. A genuine checkpoint for the right
    # architecture matches essentially everything, so anything short of that
    # means the wrong file.
    if matched < 0.9:
        unmatched = sorted(set(own) - set(usable))
        raise ValueError(
            f"Only {matched:.1%} of {arch} parameters matched {p.name}. "
            f"A correct checkpoint matches ~100%; this is the wrong architecture "
            f"or the wrong depth. Refusing rather than training from partly-random "
            f"weights while reporting 'pretrained'. "
            f"First unmatched keys: {unmatched[:6]}")
    return model, model.out_dim


def load_r2plus1d():
    """torchvision R(2+1)D-18, Kinetics-400 pretrained.

    Factorizes each 3D conv into a 2D spatial then 1D temporal conv, which
    suits anisotropic MRI (fine in-plane, coarse through-plane) better than
    isotropic 3D kernels. Caveat: Kinetics is human action video, so the
    pretraining is not domain-matched the way MedicalNet's is.
    """
    from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
    net = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
    dim = net.fc.in_features
    net.fc = nn.Identity()
    print(f"  [3D] r2plus1d_18: Kinetics-400 pretrained, feature dim {dim}")
    return net, dim


# ═══════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════
class Volume3DNet(nn.Module):
    def __init__(self, backbone="medicalnet_resnet34", checkpoint=None,
                 aux_slot_weight=0.3, dropout=0.3):
        super().__init__()
        if backbone.startswith("medicalnet_"):
            self.encoder, dim = load_medicalnet(backbone.split("_", 1)[1], checkpoint)
            self.in_ch = 1
        elif backbone == "r2plus1d_18":
            self.encoder, dim = load_r2plus1d()
            self.in_ch = 3      # video backbones expect RGB
        else:
            raise ValueError(f"unknown backbone {backbone!r}")
        self.backbone_name = backbone
        self.head = nn.Sequential(
            nn.LayerNorm(dim), nn.Dropout(dropout),
            nn.Linear(dim, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, len(TARGETS)))
        # Auxiliary head: which of the 6 acquisition types is this volume.
        # Free label, and it forces the encoder to represent plane /
        # fat-saturation / fluid-weighting rather than only disease signal.
        self.aux_slot = nn.Linear(dim, N_SLOT)
        self.aux_slot_weight = aux_slot_weight

    def forward(self, x):
        # x: [B, 1, D, H, W] float in [0,1]
        if self.in_ch == 3:
            x = x.repeat(1, 3, 1, 1, 1)
        f = self.encoder(x)
        return self.head(f), self.aux_slot(f)


# ═══════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════
class VolumeDataset(Dataset):
    """One sample = one (study, slot) volume plus that study's labels.

    index_df columns: StudyInstanceUID, slot_name, volume_file
    labels_df: StudyInstanceUID + TARGETS (+ optional per-target
    __confidence columns and is_real_label).
    """

    def __init__(self, index_df, labels_df, train=False, augment=True):
        self.rows = index_df.reset_index(drop=True)
        self.lbl = labels_df.set_index("StudyInstanceUID")
        self.train = train
        self.augment = augment and train
        self.conf_cols = [f"{t}__confidence" for t in TARGETS
                          if f"{t}__confidence" in labels_df.columns]

    def __len__(self):
        return len(self.rows)

    def _aug(self, vol):
        # MRI-safe only. No vertical flip (anatomically implausible) and no
        # horizontal flip: laterality was already normalized during volume
        # construction, so mirroring here would undo it and teach the model
        # that left and right knees are interchangeable.
        if np.random.rand() < 0.5:      # intensity scale/shift ~ scanner variation
            vol = np.clip(vol * np.random.uniform(0.9, 1.1)
                          + np.random.uniform(-0.05, 0.05), 0, 1)
        if np.random.rand() < 0.3:      # depth jitter: drop/duplicate an end slice
            k = np.random.choice([-1, 1])
            vol = np.roll(vol, k, axis=0)
        return vol

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        vol = load_volume(r["volume_file"]).astype(np.float32) / 255.0
        if self.augment:
            vol = self._aug(vol)
        x = torch.from_numpy(np.ascontiguousarray(vol))[None]     # [1,D,H,W]

        sid = r["StudyInstanceUID"]
        y = torch.tensor(self.lbl.loc[sid, TARGETS].astype(float).to_numpy(),
                         dtype=torch.float32)
        if self.conf_cols:
            w = torch.tensor(self.lbl.loc[sid, self.conf_cols].astype(float).to_numpy(),
                             dtype=torch.float32)
        else:
            w = torch.ones(len(TARGETS))
        slot_idx = SLOT_NAMES.index(r["slot_name"])
        return x, y, w, torch.tensor(slot_idx), sid


def collate(batch):
    xs, ys, ws, slots, sids = zip(*batch)
    return (torch.stack(xs), torch.stack(ys), torch.stack(ws),
            torch.stack(slots), list(sids))


# ═══════════════════════════════════════════════════════════════════════════
# TRAIN / EVAL
# ═══════════════════════════════════════════════════════════════════════════
def macro_auc(y, p):
    aucs = []
    for j in range(y.shape[1]):
        col = y[:, j]
        if len(np.unique(col)) < 2:
            continue
        aucs.append(roc_auc_score(col, p[:, j]))
    return (float(np.mean(aucs)) if aucs else float("nan")), aucs


@torch.no_grad()
def predict_studies(model, loader, device):
    """Predict per volume, then average a study's slots back to one row.

    Simple mean rather than a learned pooling: with 58 gold studies there
    is not enough signal to fit slot-weighting reliably on top of
    everything else, and an unweighted mean cannot overfit.
    """
    model.eval()
    acc, cnt = {}, {}
    for x, _, _, _, sids in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits, _ = model(x)
        p = torch.sigmoid(logits.float()).cpu().numpy()
        for k, sid in enumerate(sids):
            acc[sid] = acc.get(sid, 0) + p[k]
            cnt[sid] = cnt.get(sid, 0) + 1
    ids = sorted(acc)
    return ids, np.stack([acc[s] / cnt[s] for s in ids])


def train_one(model, tr_loader, va_loader, va_labels, device, epochs, lr,
              weight_decay=1e-4, pos_weight=None, tag=""):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best, best_state, best_pred = -np.inf, None, None

    for ep in range(epochs):
        model.train()
        losses = []
        for x, y, w, slot, _ in tr_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            w = w.to(device, non_blocking=True)
            slot = slot.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits, aux = model(x)
                main = F.binary_cross_entropy_with_logits(
                    logits, y, weight=w, pos_weight=pos_weight, reduction="mean")
                loss = main + model.aux_slot_weight * F.cross_entropy(aux, slot)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.item()))
        sched.step()

        msg = f"  [3D{tag}] epoch {ep+1}/{epochs} loss={np.mean(losses):.5f}"
        if va_loader is not None:
            ids, pred = predict_studies(model, va_loader, device)
            truth = va_labels.set_index("StudyInstanceUID").loc[ids, TARGETS] \
                             .astype(float).to_numpy()
            auc, _ = macro_auc(truth, pred)
            msg += f" val_macro_auc={auc:.5f}"
            if auc > best:
                best = auc
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                best_pred = (ids, pred)
        print(msg)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best, best_pred


def compute_pos_weight(labels_df, ids, device, cap=5.0):
    sub = labels_df[labels_df.StudyInstanceUID.isin(ids)]
    pos = sub[TARGETS].astype(float).to_numpy().sum(0)
    neg = len(sub) - pos
    pw = np.clip(np.sqrt(neg / np.maximum(pos, 1)), 1.0, cap)
    return torch.tensor(pw, dtype=torch.float32, device=device)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume-index", required=True,
                    help="CSV from cache_volumes.py: StudyInstanceUID, slot_name, volume_file")
    ap.add_argument("--labels", required=True, help="parsed labels CSV")
    ap.add_argument("--output", required=True)
    ap.add_argument("--backbone", default="medicalnet_resnet34",
                    choices=["medicalnet_resnet34", "medicalnet_resnet18", "r2plus1d_18"])
    ap.add_argument("--medicalnet-checkpoint", default=None)
    ap.add_argument("--stage-a-epochs", type=int, default=12)
    ap.add_argument("--stage-b-epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--lr-a", type=float, default=1e-4)
    ap.add_argument("--lr-b", type=float, default=2e-5)
    ap.add_argument("--aux-weight", type=float, default=0.3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    idx = pd.read_csv(a.volume_index, dtype={"StudyInstanceUID": str})
    lbl = pd.read_csv(a.labels, dtype={"StudyInstanceUID": str})
    idx = idx[idx.StudyInstanceUID.isin(set(lbl.StudyInstanceUID))].reset_index(drop=True)

    if "is_real_label" in lbl.columns:
        real_mask = lbl["is_real_label"].astype(bool)
    else:
        # Fall back to "every target is exactly 0 or 1" -- weak labels are soft.
        vals = lbl[TARGETS].astype(float).to_numpy()
        real_mask = np.all((vals == 0) | (vals == 1), axis=1)
    real_ids = set(lbl.loc[real_mask, "StudyInstanceUID"])
    print(f"Volumes: {len(idx)} across {idx.StudyInstanceUID.nunique()} studies")
    print(f"Real-labeled: {len(real_ids)}   Weak: {lbl.StudyInstanceUID.nunique()-len(real_ids)}")

    # ── STAGE A: all studies (weak + real) ────────────────────────────────
    stage_a_path = out / "stage_a_3d.pt"
    model = Volume3DNet(a.backbone, a.medicalnet_checkpoint, a.aux_weight).to(device)
    if stage_a_path.exists():
        print(f"\n── STAGE A (3D): found {stage_a_path.name}, loading ──")
        model.load_state_dict(torch.load(stage_a_path, map_location=device,
                                         weights_only=True)["state_dict"])
    else:
        print(f"\n── STAGE A (3D): pretrain on {len(idx)} volumes ──")
        # Hold out a slice of studies (not volumes) so a study's own slots
        # cannot appear on both sides of the split.
        rng = np.random.default_rng(a.seed)
        all_ids = np.array(sorted(idx.StudyInstanceUID.unique()))
        perm = rng.permutation(len(all_ids))
        n_val = max(1, int(0.1 * len(all_ids)))
        va_ids, tr_ids = set(all_ids[perm[:n_val]]), set(all_ids[perm[n_val:]])
        tr = idx[idx.StudyInstanceUID.isin(tr_ids)]
        va = idx[idx.StudyInstanceUID.isin(va_ids)]
        tr_dl = DataLoader(VolumeDataset(tr, lbl, train=True), batch_size=a.batch_size,
                           shuffle=True, num_workers=a.workers, collate_fn=collate,
                           pin_memory=True, drop_last=True)
        va_dl = DataLoader(VolumeDataset(va, lbl), batch_size=a.batch_size,
                           num_workers=a.workers, collate_fn=collate, pin_memory=True)
        pw = compute_pos_weight(lbl, tr_ids, device)
        model, best, _ = train_one(model, tr_dl, va_dl, lbl, device,
                                   a.stage_a_epochs, a.lr_a, pos_weight=pw, tag=" A")
        torch.save({"state_dict": model.state_dict(), "backbone": a.backbone,
                    "val_auc": best}, stage_a_path)
        print(f"  Saved {stage_a_path}")

    # ── STAGE B: 5-fold GroupKFold on the real-labeled studies ────────────
    print(f"\n── STAGE B (3D): {a.folds}-fold on {len(real_ids)} real-labeled studies ──")
    real_idx = idx[idx.StudyInstanceUID.isin(real_ids)].reset_index(drop=True)
    groups = real_idx.StudyInstanceUID.to_numpy()
    stage_a_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    oof_rows, fold_scores = [], []
    gkf = GroupKFold(n_splits=a.folds)
    for f, (tr_i, va_i) in enumerate(gkf.split(real_idx, groups=groups), 1):
        tr, va = real_idx.iloc[tr_i], real_idx.iloc[va_i]
        assert not (set(tr.StudyInstanceUID) & set(va.StudyInstanceUID)), \
            "study leaked across the fold boundary"
        fm = Volume3DNet(a.backbone, None, a.aux_weight).to(device)
        fm.load_state_dict(stage_a_state)     # warm-start from Stage A
        tr_dl = DataLoader(VolumeDataset(tr, lbl, train=True), batch_size=a.batch_size,
                           shuffle=True, num_workers=a.workers, collate_fn=collate,
                           pin_memory=True)
        va_dl = DataLoader(VolumeDataset(va, lbl), batch_size=a.batch_size,
                           num_workers=a.workers, collate_fn=collate, pin_memory=True)
        pw = compute_pos_weight(lbl, set(tr.StudyInstanceUID), device)
        fm, best, best_pred = train_one(fm, tr_dl, va_dl, lbl, device,
                                        a.stage_b_epochs, a.lr_b,
                                        pos_weight=pw, tag=f" B/f{f}")
        torch.save({"state_dict": fm.state_dict(), "fold": f, "val_auc": best},
                   out / f"fold_{f}_3d.pt")
        fold_scores.append(best)
        if best_pred is not None:
            ids, pred = best_pred
            for k, sid in enumerate(ids):
                oof_rows.append({"StudyInstanceUID": sid,
                                 **{f"pred_{t}": pred[k, j] for j, t in enumerate(TARGETS)}})
        del fm
        torch.cuda.empty_cache()

    oof = pd.DataFrame(oof_rows).drop_duplicates("StudyInstanceUID")
    oof.to_csv(out / "oof_predictions_3d.csv", index=False)

    truth = lbl.set_index("StudyInstanceUID").loc[oof.StudyInstanceUID, TARGETS] \
               .astype(float).to_numpy()
    pred = oof[[f"pred_{t}" for t in TARGETS]].to_numpy()
    macro, per = macro_auc(truth, pred)
    print(f"\n{'='*60}\n3D BRANCH OOF AUC : {macro:.5f}\n{'='*60}")
    for t, v in zip(TARGETS, per):
        print(f"  {t:22s}: {v:.4f}")
    (out / "metrics_3d.json").write_text(json.dumps(
        {"macro_oof_auc": macro, "per_target": dict(zip(TARGETS, per)),
         "fold_val_aucs": fold_scores, "backbone": a.backbone}, indent=2))
    print(f"\nOOF written to {out/'oof_predictions_3d.csv'} — blend it against the "
          f"MedSigLIP branch's OOF by the same alpha search, on OOF only.")


if __name__ == "__main__":
    main()
