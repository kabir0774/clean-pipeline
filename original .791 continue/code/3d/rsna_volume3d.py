#!/usr/bin/env python3
"""
Volume construction for the RSNA Knee 3D branch.

WHY THIS IS A SEPARATE MODULE
-----------------------------
A 3D CNN cannot use the MedSigLIP embedding cache -- that cache holds
per-slice feature vectors, and a 3D convolution needs raw voxels. So this
is a genuinely parallel data path: its own volume construction, its own
cache, its own training loop. The only things it reuses from the main
pipeline are the DICOM index and the slot assignment (which series belongs
to which of the 6 named roles).

WHAT "ROI CROP" MEANS HERE
--------------------------
The 2025 aneurysm winner cropped to a 224-cubed ROI around the target
anatomy, supported by segmentation decoders. We have no equivalent: the
RSNA knee data ships no localization or segmentation labels. (KneeMRI has
ACL boxes, but that is a different dataset with a different protocol.)

So "ROI" here means PHYSICAL-SIZE NORMALIZED CROPPING: take a fixed number
of millimetres around the image centre, using each slice's PixelSpacing.
This is not a learned ROI and this module does not pretend otherwise. What
it does buy is scanner invariance -- a 140mm crop covers the same anatomy
whether the scanner produced 0.3mm or 0.5mm pixels, whereas a fixed
224-pixel resize does not. The original DINOsaur pipeline did this and the
current pipeline lost it; the static audit flagged the loss.

If a real localizer is added later, crop_mm becomes the fallback path and
the learned box takes precedence -- the interface here is designed for
that swap.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pydicom


# ── configuration ────────────────────────────────────────────────────────────
CROP_MM = float(os.environ.get("RSNA_VOL3D_CROP_MM", "140"))
VOL_DEPTH = int(os.environ.get("RSNA_VOL3D_DEPTH", "32"))
VOL_SIZE = int(os.environ.get("RSNA_VOL3D_SIZE", "224"))
BAND_LO = float(os.environ.get("RSNA_VOL3D_BAND_LO", "0.2"))
BAND_HI = float(os.environ.get("RSNA_VOL3D_BAND_HI", "0.8"))


# ── DICOM reading ────────────────────────────────────────────────────────────
def _read_slice(path):
    try:
        ds = pydicom.dcmread(str(path), force=True)
        arr = ds.pixel_array.astype(np.float32)
    except Exception:
        return None, None

    if str(getattr(ds, "PhotometricInterpretation", "")).strip() == "MONOCHROME1":
        arr = arr.max() - arr
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    arr = arr * slope + intercept

    spacing = getattr(ds, "PixelSpacing", None)
    if spacing is not None and len(spacing) >= 2:
        try:
            spacing = (float(spacing[0]), float(spacing[1]))
        except Exception:
            spacing = None
    else:
        spacing = None
    return arr, spacing


def _physical_center_crop(arr, spacing, crop_mm=CROP_MM):
    if spacing is None:
        return arr, False
    row_mm, col_mm = spacing
    if not (row_mm > 0 and col_mm > 0):
        return arr, False

    h, w = arr.shape[:2]
    half_rows = int(round((crop_mm / 2.0) / row_mm))
    half_cols = int(round((crop_mm / 2.0) / col_mm))
    cy, cx = h // 2, w // 2

    y0, y1 = max(0, cy - half_rows), min(h, cy + half_rows)
    x0, x1 = max(0, cx - half_cols), min(w, cx + half_cols)
    if y1 - y0 < 16 or x1 - x0 < 16:
        return arr, False
    return arr[y0:y1, x0:x1], True


def _window_to_uint8(arr):
    lo, hi = np.percentile(arr, [1, 99])
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    return (np.clip((arr - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def _resize_2d(arr, size):
    from PIL import Image
    return np.asarray(Image.fromarray(arr).resize((size, size), Image.BILINEAR))


def _select_depth_indices(n_slices, depth):
    if n_slices <= 0:
        return []
    if n_slices >= depth:
        return np.linspace(0, n_slices - 1, depth).round().astype(int).tolist()
    idx = np.linspace(0, n_slices - 1, depth).round().astype(int)
    return idx.tolist()


def _apply_band(paths, lo=BAND_LO, hi=BAND_HI):
    n = len(paths)
    if n == 0 or (lo <= 0.0 and hi >= 1.0):
        return paths
    a, b = int(n * lo), int(np.ceil(n * hi))
    sub = paths[a:b]
    return sub if len(sub) >= 3 else paths


# ── volume construction ──────────────────────────────────────────────────────
def build_volume(ordered_paths, plane="Sagittal", is_right=False,
                 depth=VOL_DEPTH, size=VOL_SIZE, crop_mm=CROP_MM,
                 band=(BAND_LO, BAND_HI)):
    paths = _apply_band(list(ordered_paths), *band)
    if not paths:
        return None, {"reason": "no paths after band selection"}

    idxs = _select_depth_indices(len(paths), depth)
    stats = {"n_available": len(paths), "n_decoded": 0,
             "n_spacing_missing": 0, "n_cropped": 0, "padded": len(paths) < depth}

    slices = []
    cache = {}
    for i in idxs:
        if i in cache:
            slices.append(cache[i])
            continue
        arr, spacing = _read_slice(paths[i])
        if arr is None:
            continue
        if spacing is None:
            stats["n_spacing_missing"] += 1
        arr, was_cropped = _physical_center_crop(arr, spacing, crop_mm)
        if was_cropped:
            stats["n_cropped"] += 1
        sl = _resize_2d(_window_to_uint8(arr), size)
        cache[i] = sl
        slices.append(sl)
        stats["n_decoded"] += 1

    if not slices:
        return None, {**stats, "reason": "no slices decoded"}

    while len(slices) < depth:
        slices.append(slices[-1])
    vol = np.stack(slices[:depth], axis=0)

    if is_right:
        if str(plane).lower().startswith("sag"):
            vol = vol[::-1].copy()
        else:
            vol = vol[:, :, ::-1].copy()
    return vol, stats


def build_and_cache(study, slot_name, ordered_paths, out_dir, plane="Sagittal",
                    is_right=False, depth=VOL_DEPTH, size=VOL_SIZE,
                    crop_mm=CROP_MM, overwrite=False):
    """
    The cache filename ENCODES depth/size/crop_mm. Without this, rerunning
    with a different --depth against an existing cache directory would find
    files at the old filename and skip rebuilding them -- so a run intended
    to fix an "88% padded" warning would silently do nothing while reporting
    100% cache hits and no error. Different shape parameters produce
    different files; there is nothing to reuse between them.
    """
    out_dir = Path(out_dir)
    shape_tag = f"d{depth}_s{size}_c{int(crop_mm)}"
    out_path = out_dir / study / f"{study}__{slot_name}__{shape_tag}.npz"
    if out_path.exists() and not overwrite:
        return out_path, {"cached": True}
    vol, stats = build_volume(ordered_paths, plane=plane, is_right=is_right,
                              depth=depth, size=size, crop_mm=crop_mm)
    if vol is None:
        return None, stats
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, volume=vol, plane=str(plane),
                        laterality="R" if is_right else "L")
    return out_path, stats


def load_volume(path):
    with np.load(path, allow_pickle=False) as z:
        return z["volume"]
