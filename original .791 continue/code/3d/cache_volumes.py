#!/usr/bin/env python3
"""
Build and cache 3D volumes for every (study, slot) pair, producing the
volume index that rsna_train3d.py consumes.

Reuses the main pipeline's DICOM index and slot cache -- the two artifacts
that describe which files exist and how series map to the 6 named slots.
Nothing else is shared: this writes its own volume cache and never touches
the MedSigLIP embeddings.

Run this once before training. It is resumable: volumes already on disk are
skipped, so an interrupted run can be restarted with the same command.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import rsna_volume3d as V

SLOT_NAMES = ['SAG_FLUID_FS', 'COR_FLUID_FS', 'AX_FLUID_FS',
              'SAG_FLUID_NOFS', 'COR_T1', 'SAG_T1']


def sort_slices(paths):
    """Order slices by physical position, mirroring the main pipeline.

    Projects ImagePositionPatient onto the slice-normal derived from
    ImageOrientationPatient. Falls back to InstanceNumber, then filename.
    The fallback is COUNTED and reported, because a volume built from
    filename-ordered slices has an anatomically scrambled depth axis --
    silently worse than useless for a 3D convolution, and invisible unless
    the fallback rate is surfaced.
    """
    import pydicom
    recs, n_geom = [], 0
    for p in paths:
        key, has_geom = None, False
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
            ipp = getattr(ds, "ImagePositionPatient", None)
            iop = getattr(ds, "ImageOrientationPatient", None)
            if ipp is not None and iop is not None and len(iop) >= 6:
                r = np.array([float(v) for v in iop[:3]])
                c = np.array([float(v) for v in iop[3:6]])
                normal = np.cross(r, c)
                key = float(np.dot(np.array([float(v) for v in ipp]), normal))
                has_geom = True
            elif getattr(ds, "InstanceNumber", None) is not None:
                key = float(ds.InstanceNumber)
        except Exception:
            pass
        if key is None:
            key = float(hash(p.name) % 10**6)
        recs.append((key, p))
        n_geom += int(has_geom)
    recs.sort(key=lambda t: t[0])
    return [p for _, p in recs], n_geom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom-index", required=True,
                    help="train_dicom_index_*.csv from the main pipeline")
    ap.add_argument("--slots-cache", required=True,
                    help="train_slots_cache_*.pkl from the main pipeline")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--slots", default="SAG_FLUID_FS",
                    help="comma-separated slot names, or 'all'")
    ap.add_argument("--depth", type=int, default=V.VOL_DEPTH)
    ap.add_argument("--size", type=int, default=V.VOL_SIZE)
    ap.add_argument("--crop-mm", type=float, default=V.CROP_MM)
    ap.add_argument("--limit", type=int, default=0, help="cap studies (smoke test)")
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slots = SLOT_NAMES if a.slots == "all" else [s.strip() for s in a.slots.split(",")]
    bad = [s for s in slots if s not in SLOT_NAMES]
    if bad:
        raise ValueError(f"unknown slot(s) {bad}; valid: {SLOT_NAMES}")

    dcm = pd.read_csv(a.dicom_index, dtype=str)
    with open(a.slots_cache, "rb") as f:
        slots_df, lat_map = pickle.load(f)
    # The main pipeline hit a real bug here: reading with dtype=str makes
    # presence_mask the STRING "1", so `== 1` is silently False and every
    # slot reads as absent. Coerce explicitly rather than trusting dtype.
    if "presence_mask" in slots_df.columns:
        slots_df["presence_mask"] = slots_df["presence_mask"].astype(int)

    series_files = dcm.groupby("SeriesInstanceUID")["filepath"].apply(list).to_dict()
    rows = slots_df[(slots_df.slot_name.isin(slots)) &
                    (slots_df.presence_mask == 1)].copy()
    if a.limit:
        keep = sorted(rows.StudyInstanceUID.unique())[:a.limit]
        rows = rows[rows.StudyInstanceUID.isin(keep)]

    print(f"Building {len(rows)} volumes across "
          f"{rows.StudyInstanceUID.nunique()} studies, slots={slots}")
    print(f"Shape [{a.depth}, {a.size}, {a.size}], physical crop {a.crop_mm}mm")

    index, stats = [], {"built": 0, "cached": 0, "failed": 0,
                        "spacing_missing": 0, "padded": 0, "geom_missing": 0,
                        "n_slices": []}
    for r in tqdm(rows.itertuples(index=False), total=len(rows), desc="volumes"):
        study, slot = str(r.StudyInstanceUID), str(r.slot_name)
        paths = [Path(p) for p in series_files.get(str(r.SeriesInstanceUID), [])
                 if Path(p).is_file()]
        if not paths:
            stats["failed"] += 1
            continue
        ordered, n_geom = sort_slices(paths)
        stats["geom_missing"] += len(ordered) - n_geom
        stats["n_slices"].append(len(ordered))
        plane = str(getattr(r, "Anatomical_Plane", "Sagittal") or "Sagittal")
        is_right = str(lat_map.get(study, "L")).upper().startswith("R")
        try:
            path, st = V.build_and_cache(study, slot, ordered, out_dir, plane=plane,
                                         is_right=is_right, depth=a.depth,
                                         size=a.size, crop_mm=a.crop_mm)
        except Exception as e:
            print(f"\n  [WARN] {study[:16]}/{slot}: {e}")
            stats["failed"] += 1
            continue
        if path is None:
            stats["failed"] += 1
            continue
        if st.get("cached"):
            stats["cached"] += 1
        else:
            stats["built"] += 1
            stats["spacing_missing"] += st.get("n_spacing_missing", 0)
            stats["padded"] += int(st.get("padded", False))
        index.append({"StudyInstanceUID": study, "slot_name": slot,
                      "volume_file": str(path)})

    idx_df = pd.DataFrame(index)
    idx_path = out_dir / "volume_index.csv"
    idx_df.to_csv(idx_path, index=False)

    ns = np.array(stats.pop("n_slices")) if stats["n_slices"] else np.array([0])
    print(f"\n{'='*60}")
    print(f"built={stats['built']} cached={stats['cached']} failed={stats['failed']}")
    print(f"slice counts per series: median={np.median(ns):.0f} "
          f"min={ns.min()} max={ns.max()}")
    if stats["padded"]:
        pct = stats["padded"] / max(stats["built"], 1)
        print(f"WARNING: {stats['padded']} volumes ({pct:.0%}) had fewer usable slices "
              f"than depth={a.depth} and were padded with duplicates.")
        print(f"         Median series has {np.median(ns):.0f} slices; the 0.2-0.8 band "
              f"keeps ~{np.median(ns)*0.6:.0f}. Consider --depth {int(np.median(ns)*0.6)} "
              f"to stop duplicating a large fraction of every volume.")
    if stats["spacing_missing"]:
        print(f"WARNING: {stats['spacing_missing']} slices had no PixelSpacing — those "
              f"were NOT physically cropped, so their scale is scanner-dependent.")
    if stats["geom_missing"]:
        print(f"WARNING: {stats['geom_missing']} slices lacked ImagePositionPatient/"
              f"ImageOrientationPatient and fell back to InstanceNumber or filename "
              f"ordering. Depth axis may be scrambled for those series.")
    print(f"\nIndex: {idx_path}  ({len(idx_df)} volumes)")
    total_mb = sum(Path(p).stat().st_size for p in idx_df.volume_file) / 1e6 \
               if len(idx_df) else 0
    print(f"Cache size: {total_mb/1000:.2f} GB")
    (out_dir / "cache_stats.json").write_text(json.dumps(
        {**stats, "median_slices": float(np.median(ns)),
         "depth": a.depth, "size": a.size, "crop_mm": a.crop_mm}, indent=2))


if __name__ == "__main__":
    main()
