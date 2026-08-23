import numpy as np
import pandas as pd

from rsna_knee_genuine import SLOTS, choose_slots, canonicalize_laterality


def test_six_slots_are_defined():
    names = [x[0] for x in SLOTS]
    assert names == ["SAG_FS", "COR_FS", "AX_FS", "SAG_NFS", "COR_NFS", "AX_NFS"]


def test_choose_slots_selects_all_six():
    rows = []
    for i, (name, plane, fs) in enumerate(SLOTS):
        rows.append({"SeriesInstanceUID": f"s{i}", "Anatomical_Plane": plane,
                     "Fat_Suppression": int(fs), "Fluid_Sensitive": int(fs)})
    got = choose_slots(pd.DataFrame(rows))
    assert set(got) == {x[0] for x in SLOTS}
    assert got["SAG_FS"] == "s0"
    assert got["COR_NFS"] == "s4"


def test_right_knee_horizontal_canonicalization_direction():
    # A marker on the right edge must move to the left edge for a right knee.
    x = np.zeros((3, 4), dtype=np.float32)
    x[:, -1] = 1
    got = canonicalize_laterality(x, "R", enabled=True)
    assert np.array_equal(got[:, 0], np.ones(3))
    assert np.array_equal(got[:, -1], np.zeros(3))
    assert np.array_equal(canonicalize_laterality(x, "L"), x)


if __name__ == "__main__":
    test_six_slots_are_defined()
    test_choose_slots_selects_all_six()
    test_right_knee_horizontal_canonicalization_direction()
    print("clean pipeline tests passed")
