"""
Per-class blend of two runs that fail in different places.

The situation this is for
-------------------------
Run D crops every slice to a fixed 140 mm box so the knee occupies the same
fraction of the frame whatever the scanner did. Nine of twelve findings improved
under it, all of them focal -- PF OA +0.084, medial meniscus +0.070, fracture
+0.063, ACL +0.047. Effusion fell 0.052, which is the same fact seen from the
other side: fluid is diffuse and can extend past a 140 mm box, so cropping cuts
signal for it while sharpening everything that sits inside the box.

Two models that are better at different findings is exactly the case where a
per-class weight is worth fitting, rather than picking one and discarding the
other.

Why the gate
------------
Twelve weights fitted on 58 studies with 9 to 35 positives each will overfit if
simply argmaxed -- that is how an OOF number climbs while the leaderboard falls,
which has already happened twice here. A per-class weight is adopted only if its
own 95% bootstrap interval clears zero; otherwise the class falls back to the
single global weight, and if the global weight does not clear zero either, the
blend is refused entirely and the better single model is recommended.

Ranks, not probabilities
------------------------
Macro-AUC reads only the order of predictions. Two models trained under
different preprocessing produce differently shaped probability distributions, so
averaging those directly lets the more confident one dominate for reasons that
have nothing to do with being more correct. Ranks remove that.

Usage:
    python blend_per_class.py OOF_A.csv OOF_B.csv TRAIN.csv [nameA nameB]
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
           "Contusion", "Fracture"]

MIN_GAIN = 0.005          # below this a per-class weight is not worth the risk
N_BOOT_GLOBAL = 3000
N_BOOT_CLASS = 1000
GRID = np.arange(0.0, 1.001, 0.05)
RNG = np.random.default_rng(0)


def macro_auc(y, p):
    out = []
    for j in range(y.shape[1]):
        yj = y[:, j].astype(int)
        if 0 < yj.sum() < len(yj):
            out.append(roc_auc_score(yj, p[:, j]))
    return float(np.mean(out)) if out else float("nan")


def to_ranks(p):
    r = np.empty_like(p, dtype=float)
    for j in range(p.shape[1]):
        r[:, j] = rankdata(p[:, j]) / len(p)
    return r


def main(fa, fb, ftrain, name_a="A", name_b="B"):
    a = pd.read_csv(fa, dtype={"StudyInstanceUID": str})
    b = pd.read_csv(fb, dtype={"StudyInstanceUID": str})
    t = pd.read_csv(ftrain, dtype={"StudyInstanceUID": str})

    gold = t[t[TARGETS].notna().any(axis=1)].set_index("StudyInstanceUID")
    ids = [i for i in gold.index
           if i in set(a.StudyInstanceUID) and i in set(b.StudyInstanceUID)]
    print(f"gold studies in both runs: {len(ids)} / {len(gold)}")
    if len(ids) < 40:
        print("[WARN] too few overlapping studies for this to mean much")

    Y = gold.loc[ids, TARGETS].values.astype(float)
    PA = a.set_index("StudyInstanceUID").loc[ids, TARGETS].values.astype(float)
    PB = b.set_index("StudyInstanceUID").loc[ids, TARGETS].values.astype(float)
    RA, RB = to_ranks(PA), to_ranks(PB)

    auc_a, auc_b = macro_auc(Y, PA), macro_auc(Y, PB)
    print(f"\n{name_a:>10s} : {auc_a:.5f}")
    print(f"{name_b:>10s} : {auc_b:.5f}")
    better = name_a if auc_a >= auc_b else name_b
    base_R = RA if auc_a >= auc_b else RB
    base_auc = max(auc_a, auc_b)

    print(f"\nper class ({name_a} / {name_b}):")
    for j, tg in enumerate(TARGETS):
        yj = Y[:, j].astype(int)
        if not (0 < yj.sum() < len(yj)):
            continue
        sa, sb = roc_auc_score(yj, PA[:, j]), roc_auc_score(yj, PB[:, j])
        mark = ""
        if abs(sa - sb) > 0.04:
            mark = f"  <- {name_a if sa > sb else name_b} clearly better"
        print(f"  {tg:18s} {sa:.4f} / {sb:.4f}  ({sb-sa:+.4f}){mark}")

    # ── global weight ────────────────────────────────────────────────────────
    scores = [macro_auc(Y, w * RA + (1 - w) * RB) for w in GRID]
    gi = int(np.argmax(scores))
    g_alpha, g_auc = float(GRID[gi]), float(scores[gi])
    print(f"\nglobal weight (alpha = {name_a} share):")
    for w, s in zip(GRID[::4], np.array(scores)[::4]):
        print(f"  alpha={w:.2f}  auc={s:.5f}")
    print(f"  best alpha={g_alpha:.2f}  auc={g_auc:.5f}   "
          f"(best single {better} {base_auc:.5f})")

    diffs = []
    for _ in range(N_BOOT_GLOBAL):
        k = RNG.integers(0, len(ids), len(ids))
        base = macro_auc(Y[k], base_R[k])
        bl = macro_auc(Y[k], g_alpha * RA[k] + (1 - g_alpha) * RB[k])
        if not (np.isnan(base) or np.isnan(bl)):
            diffs.append(bl - base)
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"  bootstrap gain over {better}: {diffs.mean():+.5f}  "
          f"95% CI [{lo:+.5f}, {hi:+.5f}]")
    adopt_global = (g_auc - base_auc) >= MIN_GAIN and lo > 0
    print(f"  global blend adopted: {adopt_global}")

    # ── per-class weights, each gated on its own interval ────────────────────
    print(f"\nper-class weights (adopted only if that class's CI clears zero):")
    alphas, n_adopt = {}, 0
    for j, tg in enumerate(TARGETS):
        yj = Y[:, j].astype(int)
        if not (0 < yj.sum() < len(yj)):
            alphas[tg] = g_alpha
            continue
        sc = [roc_auc_score(yj, w * RA[:, j] + (1 - w) * RB[:, j]) for w in GRID]
        bi = int(np.argmax(sc))
        cand, cand_s = float(GRID[bi]), float(sc[bi])
        base_s = roc_auc_score(yj, base_R[:, j])

        d = []
        for _ in range(N_BOOT_CLASS):
            k = RNG.integers(0, len(ids), len(ids))
            if not (0 < yj[k].sum() < len(k)):
                continue
            d.append(roc_auc_score(yj[k], cand * RA[k, j] + (1 - cand) * RB[k, j])
                     - roc_auc_score(yj[k], base_R[k, j]))
        clo = float(np.percentile(d, 2.5)) if d else -1.0
        ok = clo > 0 and (cand_s - base_s) >= MIN_GAIN
        alphas[tg] = cand if ok else (g_alpha if adopt_global else
                                      (1.0 if better == name_a else 0.0))
        n_adopt += int(ok)
        print(f"  {tg:18s} alpha={cand:.2f}  gain={cand_s-base_s:+.4f}  "
              f"CIlo={clo:+.4f}   {'ADOPT' if ok else 'fallback'}")

    PC = np.column_stack([alphas[t] * RA[:, j] + (1 - alphas[t]) * RB[:, j]
                          for j, t in enumerate(TARGETS)])
    pc_auc = macro_auc(Y, PC)
    print(f"\nper-class blend auc: {pc_auc:.5f}   ({n_adopt}/12 classes adopted)")

    out = Path("blend_weights.json")
    out.write_text(json.dumps({
        "name_a": name_a, "name_b": name_b,
        "auc_a": auc_a, "auc_b": auc_b,
        "global_alpha": g_alpha, "adopt_global": bool(adopt_global),
        "auc_global": g_auc, "auc_per_class": pc_auc,
        "per_class_alpha": alphas,
    }, indent=2))
    print(f"wrote {out}")

    print("\nverdict:")
    if not adopt_global and n_adopt == 0:
        print(f"  Do not blend. Submit {better} alone ({base_auc:.5f}); neither "
              f"the global nor any per-class weight clears its interval.")
    elif n_adopt:
        print(f"  Blend per class: {n_adopt} classes take a fitted weight, the "
              f"rest fall back. OOF {pc_auc:.5f} vs {base_auc:.5f} for "
              f"{better} alone.")
        print("  Confirm on the leaderboard before trusting it -- OOF has "
              "mispredicted LB four times in this project.")
    else:
        print(f"  Global blend only, alpha={g_alpha:.2f}. OOF {g_auc:.5f}.")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(1)
    main(*sys.argv[1:6])
