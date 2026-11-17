"""Local tests on synthetic labels only (no images, no real ids).

  python tests/test_folds_and_parse.py
"""

import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.utkface import list_utkface, parse_name  # noqa: E402
from src.folds import age_band, inner_val_fold, make_folds, strat_keys  # noqa: E402


def test_parse():
    assert parse_name("15_1_1_20170109.jpg.chip.jpg") == ("15_1_1_20170109", 15, "female", "Black")
    assert parse_name("39_1_20170116174525125.jpg.chip.jpg") is None   # missing race
    with tempfile.TemporaryDirectory() as td:
        for n in ["1_0_0_1.jpg.chip.jpg", "70_1_4_2.jpg.chip.jpg", "bad.jpg", "notes.txt"]:
            (Path(td) / n).touch()
        rows, skipped = list_utkface(td)
        assert [r["group"] for r in rows] == ["White", "Others"] and skipped == ["bad.jpg"]


def test_bands():
    assert [age_band(a) for a in (0, 9, 10, 17, 18, 59, 60, 116)] == \
        ["0-9", "0-9", "10-17", "10-17", "18-19", "40-59", "60+", "60+"]


def synthetic(n=6000, seed=1):
    rng = np.random.default_rng(seed)
    groups = rng.choice(["White", "Black", "Asian", "Indian", "Others"], n,
                        p=[.42, .19, .15, .17, .07]).tolist()
    ages = rng.integers(0, 100, n).astype(float).tolist()
    ids = [f"img{i:05d}" for i in range(n)]
    ages = [20.0 if a in (18.0, 19.0) else a for a in ages]
    groups[:3] = ["Others"] * 3; ages[:3] = [18.0, 18.0, 19.0]   # only 3 in cell -> merged
    return ids, groups, ages


def test_folds():
    ids, groups, ages = synthetic()
    f = make_folds(ids, groups, ages)
    assert set(f) == set(ids) and set(f.values()) == set(range(5))
    perm = np.random.default_rng(2).permutation(len(ids))
    f2 = make_folds([ids[i] for i in perm], [groups[i] for i in perm], [ages[i] for i in perm])
    assert f == f2, "order-dependent"
    assert f != make_folds(ids, groups, ages, seed=1), "seed ignored"
    sizes = Counter(f.values())
    assert max(sizes.values()) - min(sizes.values()) <= 5, sizes
    # every stratification cell spread evenly (difference <= 1 between folds)
    keys = strat_keys(groups, ages, 5)
    for k in set(keys):
        c = Counter(f[i] for i, kk in zip(ids, keys) if kk == k)
        vals = [c.get(x, 0) for x in range(5)]
        assert max(vals) - min(vals) <= 1, (k, vals)
    assert min(Counter(keys).values()) >= 5
    assert [inner_val_fold(k) for k in range(5)] == [1, 2, 3, 4, 0]


def test_rare_paths():
    # 6 tiny cells of 1-3 images -> '<group>|rare' or 'rare' pool, all >= 5 after merging
    g = ["White"] * 50 + ["Black"] * 3 + ["Asian"] * 2
    a = [30.0] * 50 + [18.0, 19.0, 65.0] + [18.0, 19.0]
    keys = strat_keys(g, a, 5)
    assert min(Counter(keys).values()) >= 5, Counter(keys)
    g2 = ["White"] * 50 + ["Black"] * 30 + ["Black", "Asian", "Asian"]
    a2 = [30.0] * 50 + [30.0] * 30 + [18.0, 18.0, 65.0]
    k2 = strat_keys(g2, a2, 5)
    assert min(Counter(k2).values()) >= 5 and k2[80] == "Black|20-39", Counter(k2)


if __name__ == "__main__":
    for t in (test_parse, test_bands, test_folds, test_rare_paths):
        t(); print("PASS", t.__name__)
