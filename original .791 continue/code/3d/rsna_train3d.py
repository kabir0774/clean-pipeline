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

from rsna_volume3d import load_volume, build_volume, VOL_DEPTH, VOL_SIZE, CROP_MM

TARGETS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
           'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
           'Contusion', 'Fracture']
SLOT_NAMES = ['SAG_FLUID_FS', 'COR_FLUID_FS', 'AX_FLUID_FS',
              'SAG_FLUID_NOFS', 'COR_T1', 'SAG_T1']
N_SLOT = len(SLOT_NAMES)

# Intensity normalization scheme -- see VolumeDataset.__getitem__.
# "unit" (default, matches the 0.859 run) or "zscore" (MONAI/MedicalNet convention).
NORM_MODE = os.environ.get("RSNA_VOL3D_NORM", "unit").lower()
if NORM_MODE not in {"unit", "zscore"}:
    raise ValueError(f"RSNA_VOL3D_NORM must be 'unit' or 'zscore', got {NORM_MODE!r}")

# ── Kaggle environment detection, matching the main pipeline's exact pattern ─
# On Kaggle, this script needs only the 5 fold checkpoints (+ stage_a_3d.pt),
# NOT the 4096 training volumes -- inference only needs to build volumes for
# the 3 test studies fresh, from the competition's own test DICOM data,
# which is already present under /kaggle/input with no download needed.
IS_KAGGLE = os.path.exists("/kaggle/input")


def find_kaggle_dataset(*name_fragments):
    """Search /kaggle/input (depth-limited) for a directory whose path
    contains ALL given fragments, case-insensitive. Mirrors the main
    pipeline's dataset auto-discovery so the same notebook works whether the
    checkpoint dataset is named "rsna-3d-branch-cache" or something the user
    renamed it to on upload -- Kaggle dataset names are not guaranteed to
    match what a script expects, so searching by content fragment is more
    robust than hardcoding one exact path.
    """
    base = Path("/kaggle/input")
    if not base.is_dir():
        return None
    for depth in range(1, 4):
        for p in base.glob("/".join(["*"] * depth)):
            if p.is_dir():
                low = str(p).lower()
                if all(f.lower() in low for f in name_fragments):
                    return p
    return None


# ═══════════════════════════════════════════════════════════════════════════
# BACKBONES
# ═══════════════════════════════════════════════════════════════════════════
class BasicBlock3D(nn.Module):
    """MedicalNet-compatible 3D basic block (matches Tencent/MedicalNet's
    resnet.py so their published checkpoints load without renaming). Used
    for ResNet-18/34."""
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


class Bottleneck3D(nn.Module):
    """MedicalNet-compatible 3D bottleneck block. Used for ResNet-50/101/152 --
    NOT interchangeable with BasicBlock3D: different internal structure
    (1x1 -> 3x3 -> 1x1 with 4x channel expansion) and a checkpoint trained
    with one cannot be loaded into the other regardless of matching layer
    counts, since the actual tensor shapes at each layer differ.
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet3D(nn.Module):
    """3D ResNet matching MedicalNet's layer naming, so published Med3D
    checkpoints load directly. Encoder only -- no segmentation decoder.

    block: BasicBlock3D (resnet18/34) or Bottleneck3D (resnet50/101/152).
    Passing the wrong block for a given layer count produces a model that
    LOOKS plausible but has completely different internal tensor shapes --
    this is exactly why load_medicalnet's checkpoint-match check below
    verifies actual tensor shapes, not just that loading didn't error.
    """

    def __init__(self, layers, block=None, in_channels=1):
        super().__init__()
        block = block or BasicBlock3D
        self.inplanes = 64
        self.block = block
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7,
                               stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.out_dim = 512 * block.expansion

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        out_planes = planes * self.block.expansion
        if stride != 1 or self.inplanes != out_planes:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, out_planes, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_planes))
        layers = [self.block(self.inplanes, planes, stride, downsample)]
        self.inplanes = out_planes
        layers += [self.block(self.inplanes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return F.adaptive_avg_pool3d(x, 1).flatten(1)


def load_medicalnet(arch="resnet34", checkpoint=None, quiet=False):
    """Build a 3D ResNet and, if a MedicalNet checkpoint is given, load it.

    Reports the fraction of parameters that actually matched and REFUSES a
    load below 50%. A silent partial load is the dangerous failure here:
    the model would train from mostly-random weights while the log claimed
    it was pretrained, and the resulting weaker OOF would be blamed on the
    architecture rather than on the load having failed.
    """
    arch_config = {
        "resnet18": ([2, 2, 2, 2], BasicBlock3D),
        "resnet34": ([3, 4, 6, 3], BasicBlock3D),
        "resnet50": ([3, 4, 6, 3], Bottleneck3D),
    }
    if arch not in arch_config:
        raise ValueError(f"unknown MedicalNet arch {arch!r}; choose from {list(arch_config)}")
    layers, block = arch_config[arch]
    model = ResNet3D(layers, block=block, in_channels=1)
    if not checkpoint:
        if not quiet:
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


def load_monai(arch="seresnet50", depth_hint=None):
    """MONAI 3D classification backbones, classifier stripped so our own
    12-disease + 6-slot heads attach instead.

    IMPORTANT -- random init. MONAI ships architectures, not general-purpose
    pretrained 3D classification weights. Its Model Zoo bundles are
    task-specific (organ segmentation), not a MedicalNet-style generic
    pretrain. So these start from scratch exactly like the ResNet-34 run
    that scored 0.859. The reason to use them is ARCHITECTURAL DIVERSITY
    for the blend -- SE blocks and dense connectivity make different errors
    than plain residual blocks, and decorrelated errors are what make an
    ensemble worth more than its members.

    DEPTH CONSTRAINT (verified empirically, not from docs): DenseNet121-3D
    downsamples the depth axis 32x, so it CRASHES on volumes with depth < 32
    ("input image smaller than kernel size") -- our depth-18 cache fails on
    it. SEResNet50 handles depth 18 fine. The guard below fails fast with a
    clear message rather than after a long run has already started.
    """
    import torch.nn as nn
    if arch == "densenet121":
        if depth_hint is not None and depth_hint < 32:
            raise ValueError(
                f"MONAI DenseNet121-3D requires volume depth >= 32, but the volumes "
                f"are depth={depth_hint}. It downsamples the depth axis 32x and will "
                f"crash mid-training otherwise. Either rebuild volumes with "
                f"--depth 32 (expect heavy duplicate-padding, ~88% at your slice "
                f"counts) or use --backbone monai_seresnet50, which works at depth 18.")
        from monai.networks.nets import DenseNet121
        net = DenseNet121(spatial_dims=3, in_channels=1, out_channels=12)
        dim = net.class_layers.out.in_features
        net.class_layers.out = nn.Identity()
    elif arch == "seresnet50":
        from monai.networks.nets import SEResNet50
        net = SEResNet50(spatial_dims=3, in_channels=1, num_classes=12)
        dim = net.last_linear.in_features
        net.last_linear = nn.Identity()
    else:
        raise ValueError(f"unknown MONAI arch {arch!r}; use densenet121 or seresnet50")
    print(f"  [3D] MONAI {arch}: random init (MONAI ships no general 3D "
          f"classification pretrain), feature dim {dim}")
    return net, dim


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
                 aux_slot_weight=0.3, dropout=0.3, quiet=False, depth_hint=None):
        """quiet=True suppresses the init message. Used by Stage B, which
        builds an empty model purely as a shell and immediately overwrites
        every weight via load_state_dict(stage_a_state) -- printing
        "random init" there is actively misleading, since the model never
        trains from those random values."""
        super().__init__()
        if backbone.startswith("medicalnet_"):
            self.encoder, dim = load_medicalnet(backbone.split("_", 1)[1], checkpoint,
                                                quiet=quiet)
            self.in_ch = 1
        elif backbone.startswith("monai_"):
            self.encoder, dim = load_monai(backbone.split("_", 1)[1], depth_hint=depth_hint)
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
            vol = vol * np.random.uniform(0.9, 1.1) + np.random.uniform(-0.05, 0.05)
            # Clip only under "unit" normalization. z-scored volumes are
            # legitimately negative and unbounded, so clipping to [0,1] there
            # would destroy roughly half the signal.
            if NORM_MODE == "unit":
                vol = np.clip(vol, 0, 1)
        if np.random.rand() < 0.3:      # depth jitter: drop/duplicate an end slice
            k = np.random.choice([-1, 1])
            vol = np.roll(vol, k, axis=0)
        return vol

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        vol = load_volume(r["volume_file"]).astype(np.float32)
        # Intensity normalization. Two schemes, because they suit different
        # initializations:
        #   "unit"   -- divide by 255 -> [0,1]. Fine for random init, which
        #               learns whatever scale it is given.
        #   "zscore" -- per-volume (x - mean) / std, the MONAI/MedicalNet
        #               convention. Pretrained medical weights were trained on
        #               z-scored inputs, so feeding them [0,1] data puts every
        #               filter and BatchNorm running-stat off-distribution --
        #               a prime suspect for why MedicalNet transfer scored
        #               WORSE (0.764) than random init (0.859) here.
        if NORM_MODE == "zscore":
            m, s = float(vol.mean()), float(vol.std())
            vol = (vol - m) / (s if s > 1e-6 else 1.0)
        else:
            vol = vol / 255.0
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
def macro_auc(y, p, threshold=0.5):
    """AUC per target, averaged, skipping any target with a degenerate (single-
    class) column in this batch.

    y may be SOFT (continuous parser probabilities), not just hard 0/1 --
    Stage A's held-out split is a random slice of the whole weak+real pool,
    so it will contain weak-labeled studies whose "ground truth" is itself a
    confidence score, not a clean label. sklearn's roc_auc_score refuses
    continuous-valued y outright ("continuous format is not supported"),
    which is what crashed here originally.

    Thresholding at 0.5 turns that into an approximate monitoring metric:
    "does the model's ranking agree with the parser's best guess." This is
    NOT a real evaluation -- parser noise means the thresholded weak label
    isn't ground truth either, just a proxy for early-stopping/model
    selection during Stage A. The only genuine evaluation in this file is
    Stage B's OOF AUC, computed against the 58 real gold labels, which are
    already hard 0/1 and pass through this threshold unchanged.
    """
    aucs = []
    for j in range(y.shape[1]):
        col = (y[:, j] >= threshold).astype(int)
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
def run_test_inference_inmemory(fold_paths, test_dcm_df, test_slots_df, lat_map,
                                backbone, out_path, depth, size, crop_mm, device):
    """Same as run_test_inference, but takes already-loaded DataFrames instead
    of file paths.

    THIS is the intended Kaggle entry point: append this call as one more
    cell in your EXISTING notebook, right after the main pipeline's own
    "TEST: Scan DICOMs" / "TEST: Build slots" steps, passing whatever
    variable names it already produced (e.g. test_dcm, test_slots, test_lat).
    No separate cache file needs to exist for this -- the main pipeline's
    test scan is cheap (3 studies) and already runs fresh every time; this
    just reuses its in-memory output directly instead of requiring it to
    have been saved to disk somewhere first.
    """
    from cache_volumes import sort_slices

    if "presence_mask" in test_slots_df.columns:
        test_slots_df = test_slots_df.copy()
        test_slots_df["presence_mask"] = test_slots_df["presence_mask"].astype(int)

    series_files = test_dcm_df.groupby("SeriesInstanceUID")["filepath"].apply(list).to_dict()
    rows = test_slots_df[(test_slots_df.slot_name == "SAG_FLUID_FS") &
                         (test_slots_df.presence_mask == 1)]

    print(f"  Building test volumes for {rows.StudyInstanceUID.nunique()} studies "
          f"(fresh, not cached -- only a few studies, cheap to rebuild every run)")
    study_vols = {}
    for r in rows.itertuples(index=False):
        study = str(r.StudyInstanceUID)
        paths = [Path(p) for p in series_files.get(str(r.SeriesInstanceUID), [])
                 if Path(p).is_file()]
        if not paths:
            print(f"  [WARN] {study}: no files found for its SAG_FLUID_FS series")
            continue
        ordered, _ = sort_slices(paths)
        plane = str(getattr(r, "Anatomical_Plane", "Sagittal") or "Sagittal")
        is_right = str(lat_map.get(study, "L")).upper().startswith("R")
        vol, _ = build_volume(ordered, plane=plane, is_right=is_right,
                              depth=depth, size=size, crop_mm=crop_mm)
        if vol is not None:
            v = vol.astype(np.float32)
            # MUST match training normalization, or the model sees a
            # different input distribution at test time than it learned on.
            if NORM_MODE == "zscore":
                _m, _s = float(v.mean()), float(v.std())
                v = (v - _m) / (_s if _s > 1e-6 else 1.0)
            else:
                v = v / 255.0
            study_vols[study] = v

    if not study_vols:
        raise RuntimeError(
            "No test volumes could be built from the given test_dcm_df/test_slots_df -- "
            "check that SAG_FLUID_FS is actually present for these studies. Refusing to "
            "write an all-default submission silently; an empty submission is a bug "
            "worth seeing, not hiding.")

    ids = sorted(study_vols)
    x = torch.stack([torch.from_numpy(study_vols[s])[None] for s in ids]).to(device)

    all_fold_preds = []
    for fp in fold_paths:
        m = Volume3DNet(backbone, None, 0.3).to(device)
        ckpt = torch.load(fp, map_location=device, weights_only=True)
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits, _ = m(x)
        all_fold_preds.append(torch.sigmoid(logits.float()).cpu().numpy())
        del m
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pred = np.mean(all_fold_preds, axis=0)
    sub = pd.DataFrame({"StudyInstanceUID": ids})
    for j, t in enumerate(TARGETS):
        sub[t] = pred[:, j]
    sub.to_csv(out_path, index=False)
    print(f"  Wrote {out_path}  shape={sub.shape}")
    print(sub.to_string(index=False))
    return sub


def run_test_inference(fold_paths, test_dicom_index, test_slots_cache, backbone,
                       out_path, depth, size, crop_mm, device):
    """File-path convenience wrapper around run_test_inference_inmemory, for
    the standalone-script / non-notebook usage (--test-only from the CLI).
    Requires test_dicom_index.csv and test_slots_cache.pkl to actually exist
    as files -- if your test-scan step doesn't persist those (many don't,
    since scanning 3 test studies is cheap enough to just redo every run),
    use run_test_inference_inmemory directly instead, passing the DataFrames
    your test-scan step already produced in memory.
    """
    import pickle
    dcm = pd.read_csv(test_dicom_index, dtype=str)
    with open(test_slots_cache, "rb") as f:
        slots_df, lat_map = pickle.load(f)
    return run_test_inference_inmemory(fold_paths, dcm, slots_df, lat_map, backbone,
                                       out_path, depth, size, crop_mm, device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume-index", required=False, default=None,
                    help="CSV from cache_volumes.py: StudyInstanceUID, slot_name, volume_file. "
                         "Not required with --test-only.")
    ap.add_argument("--test-only", action="store_true",
                    help="Skip Stage A/B entirely; load existing fold checkpoints and run "
                         "inference on the test studies only. This is the Kaggle submission path.")
    ap.add_argument("--test-dicom-index", default=None,
                    help="DICOM index covering the TEST studies (required with --test-only)")
    ap.add_argument("--test-slots-cache", default=None,
                    help="Slot cache covering the TEST studies (required with --test-only)")
    ap.add_argument("--submission-out", default=None,
                    help="Where to write the 3D branch's predictions (default: <output>/submission_3d.csv)")
    ap.add_argument("--labels", required=True, help="parsed labels CSV")
    ap.add_argument("--output", required=True)
    ap.add_argument("--backbone", default="medicalnet_resnet34",
                    choices=["medicalnet_resnet34", "medicalnet_resnet18",
                             "medicalnet_resnet50", "monai_seresnet50",
                             "monai_densenet121", "r2plus1d_18"])
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

    if a.test_only:
        # Kaggle path: skip Stage A/B entirely. Auto-discover the checkpoint
        # dataset and the test DICOM index/slots if not given explicitly --
        # on Kaggle, dataset mount paths are auto-generated
        # (/kaggle/input/<dataset-slug>/...) and the exact slug depends on
        # what the user named it at upload time, so searching by content
        # fragment is more robust than requiring an exact hardcoded path.
        ckpt_dir = out
        if IS_KAGGLE and not list(ckpt_dir.glob("fold_*_3d.pt")):
            found = find_kaggle_dataset("3d")
            if found is None:
                raise FileNotFoundError(
                    "--test-only on Kaggle: no fold_*_3d.pt found under --output and "
                    "no dataset containing '3d' found under /kaggle/input. Attach the "
                    "checkpoint dataset (fold_1_3d.pt..fold_5_3d.pt) before running.")
            ckpt_dir = found
            print(f"  Kaggle: found checkpoint dataset at {ckpt_dir}")

        fold_paths = sorted(ckpt_dir.glob("fold_*_3d.pt"),
                            key=lambda p: int(p.stem.split("_")[1]))
        if not fold_paths:
            raise FileNotFoundError(f"No fold_*_3d.pt files found under {ckpt_dir}")
        print(f"  Loading {len(fold_paths)} fold checkpoint(s): "
              f"{[p.name for p in fold_paths]}")

        test_dcm = a.test_dicom_index
        test_slots = a.test_slots_cache
        if IS_KAGGLE and (test_dcm is None or test_slots is None):
            comp_root = find_kaggle_dataset("rsna", "knee") or find_kaggle_dataset("competitions")
            if comp_root is None:
                raise FileNotFoundError(
                    "--test-only on Kaggle: --test-dicom-index/--test-slots-cache not given "
                    "and no competition dataset auto-detected under /kaggle/input. Pass them "
                    "explicitly, or point at wherever the main pipeline's test scan/slot "
                    "outputs are attached as a dataset.")
            raise FileNotFoundError(
                f"Found competition data at {comp_root}, but this script does not scan "
                f"DICOMs itself -- pass --test-dicom-index/--test-slots-cache pointing at "
                f"the outputs of the main pipeline's own test-scan step (the same "
                f"train_dicom_index.csv/train_slots_cache.pkl-equivalent files, built for "
                f"the TEST studies), not raw competition DICOMs.")

        submission_out = Path(a.submission_out) if a.submission_out else out / "submission_3d.csv"
        run_test_inference(fold_paths, test_dcm, test_slots, a.backbone,
                           submission_out, VOL_DEPTH, VOL_SIZE, CROP_MM, device)
        return

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
    # Read the depth from an actual cached volume rather than trusting
    # VOL_DEPTH -- cache_volumes.py takes --depth as a CLI arg, so the
    # constant here can easily disagree with what is actually on disk. The
    # MONAI DenseNet guard depends on this being the true value.
    _detected_depth = None
    try:
        _detected_depth = int(load_volume(idx.iloc[0]["volume_file"]).shape[0])
        print(f"Detected volume depth from cache: {_detected_depth}")
    except Exception as e:
        print(f"[WARN] could not detect volume depth ({e}); "
              f"falling back to VOL_DEPTH={VOL_DEPTH}")
        _detected_depth = VOL_DEPTH

    model = Volume3DNet(a.backbone, a.medicalnet_checkpoint, a.aux_weight,
                        depth_hint=_detected_depth).to(device)
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
        fm = Volume3DNet(a.backbone, None, a.aux_weight, quiet=True,
                         depth_hint=_detected_depth).to(device)
        fm.load_state_dict(stage_a_state)     # warm-start from Stage A
        if f == 1:
            print(f"  [3D] Stage B folds warm-start from Stage A weights "
                  f"({len(stage_a_state)} tensors), which carry whatever "
                  f"initialization Stage A used.")
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
