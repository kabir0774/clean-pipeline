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
# Physical crop size in millimetres, per in-plane axis. An adult knee is
# roughly 100-150mm across at the joint line; 140mm keeps the joint plus
# enough surrounding tissue for findings that sit outside the joint proper
# (Baker's cyst sits posteriorly, MCL medially, contusions can be anywhere
# in the visible bone). Cropping tighter risks amputating exactly the
# findings that are hardest to detect.
CROP_MM = float(os.environ.get("RSNA_VOL3D_CROP_MM", "140"))

# Output volume shape: [DEPTH, SIZE, SIZE].
VOL_DEPTH = int(os.environ.get("RSNA_VOL3D_DEPTH", "32"))
VOL_SIZE = int(os.environ.get("RSNA_VOL3D_SIZE", "224"))

# Keep only the middle fraction of each stack, matching the 2D path's
# SLICE_BAND. The outer ends of an MRI stack are usually edge tissue or
# padding. Set to 0.0/1.0 to use the full stack.
BAND_LO = float(os.environ.get("RSNA_VOL3D_BAND_LO", "0.2"))
BAND_HI = float(os.environ.get("RSNA_VOL3D_BAND_HI", "0.8"))


# ── DICOM reading ────────────────────────────────────────────────────────────
def _read_slice(path):
    """Decode one DICOM slice to float32 plus its in-plane PixelSpacing.

    Returns (array, (row_mm, col_mm)) or (None, None) on failure. Spacing is
    None when the tag is absent -- callers must handle that rather than
    assuming a default, because silently assuming 1.0mm would produce a crop
    covering the wrong amount of anatomy without any error being raised.
    """
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
    """Crop a fixed number of millimetres around the image centre.

    Two slices from different scanners with different PixelSpacing produce
    crops covering the SAME physical extent, so the anatomy lands at a
    consistent scale after the later resize. Without this, a 224-pixel
    resize means different things on different scanners.

    Returns (cropped_array, was_cropped). When spacing is unknown the array
    is returned untouched and was_cropped is False -- the caller counts
    these so a systematically missing PixelSpacing tag shows up as a number
    rather than silently degrading every volume.
    """
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
    # A degenerate crop (absurd spacing value, tiny image) would destroy the
    # slice. Fall back to the full slice rather than emitting a 3-pixel image.
    if y1 - y0 < 16 or x1 - x0 < 16:
        return arr, False
    return arr[y0:y1, x0:x1], True


def _window_to_uint8(arr):
    """1st-99th percentile window, matching the 2D path's dicom_to_pil so the
    two branches see comparable intensity distributions."""
    lo, hi = np.percentile(arr, [1, 99])
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    return (np.clip((arr - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def _resize_2d(arr, size):
    """Bilinear resize to (size, size) without pulling in torch or cv2.

    Uses PIL, which is already a dependency of the main pipeline.
    """
    from PIL import Image
    return np.asarray(Image.fromarray(arr).resize((size, size), Image.BILINEAR))


def _select_depth_indices(n_slices, depth):
    """Choose `depth` slice indices from `n_slices` available.

    Picks evenly spaced ACTUAL slices rather than interpolating between
    them. MRI slice spacing is coarse (often 3-4mm) and neighbouring slices
    can differ substantially, so linear interpolation along Z invents
    tissue that was never imaged -- it produces smooth-looking volumes that
    are partly fabricated. Selecting real slices keeps every voxel
    something the scanner actually measured.

    When there are fewer slices than requested, indices repeat. The volume
    is then padded with duplicates rather than zeros, because a block of
    black slices would read to a 3D conv as a real anatomical edge.
    """
    if n_slices <= 0:
        return []
    if n_slices >= depth:
        return np.linspace(0, n_slices - 1, depth).round().astype(int).tolist()
    idx = np.linspace(0, n_slices - 1, depth).round().astype(int)
    return idx.tolist()


def _apply_band(paths, lo=BAND_LO, hi=BAND_HI):
    """Keep the middle fraction of an ordered slice list."""
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
    """Build one [depth, size, size] uint8 volume from ordered DICOM paths.

    ordered_paths MUST already be in physical order (the main pipeline's
    sort_slices does this by projecting ImagePositionPatient onto the
    dominant axis). Passing filename-ordered paths produces a volume whose
    depth axis is anatomically scrambled, which is worse than useless for a
    3D convolution -- there is no way to detect that here, so it is the
    caller's responsibility.

    Laterality follows the same convention as the 2D path's
    normalise_laterality: right knees are made to resemble left ones, by
    reversing slice order for sagittal series and mirroring left-right for
    coronal/axial.

    Returns (volume, stats_dict). Volume is None when nothing decoded.
    """
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

    # A mid-series decode failure leaves fewer slices than requested; repeat
    # the last good one so the output shape stays fixed. Shape must be
    # invariant for batching regardless of per-study decode failures.
    while len(slices) < depth:
        slices.append(slices[-1])
    vol = np.stack(slices[:depth], axis=0)

    if is_right:
        if str(plane).lower().startswith("sag"):
            vol = vol[::-1].copy()          # reverse through-plane direction
        else:
            vol = vol[:, :, ::-1].copy()    # mirror left-right
    return vol, stats


def build_and_cache(study, slot_name, ordered_paths, out_dir, plane="Sagittal",
                    is_right=False, depth=VOL_DEPTH, size=VOL_SIZE,
                    crop_mm=CROP_MM, overwrite=False):
    """Build one volume and cache it as compressed .npz.

    uint8 keeps this affordable: at 32x224x224 each volume is ~1.6MB raw,
    so ~7GB for 4,349 studies at one slot each, versus ~42GB for all six
    slots. Compression typically halves that again since MRI has large dark
    regions.
    """
    out_dir = Path(out_dir)
    out_path = out_dir / study / f"{study}__{slot_name}.npz"
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
    """Load a cached volume as uint8 [D, H, W]."""
    with np.load(path, allow_pickle=False) as z:
        return z["volume"]
