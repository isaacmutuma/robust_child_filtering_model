"""Local tests for models B and C and the worst-group selection rule (synthetic data only).

  python tests/test_models_bc.py
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import train as T  # noqa: E402
from src.folds import age_band  # noqa: E402

GROUPS = ["White", "Black", "Asian", "Indian", "Others"]


def synth(n=3000, seed=0, dim=24, noisy_black_teens=0.0):
    """Embeddings carrying an age signal; optional extra label noise for Black 10-17."""
    rng = np.random.default_rng(seed)
    g = rng.choice(GROUPS, n, p=[.42, .19, .15, .17, .07])
    age = rng.integers(0, 80, n).astype(float)
    w = rng.normal(size=dim)
    X = (np.outer(age - 40, w) / 40 + rng.normal(0, 0.3, (n, dim)) + 5).astype(np.float32)
    y = age.copy()
    bt = (g == "Black") & (age >= 10) & (age <= 17)
    y[bt] += rng.normal(0, noisy_black_teens, bt.sum())
    return X, np.clip(y, 0, None), g, age


def test_balanced_probs():
    _, _, g, age = synth(4000, seed=1)
    p = T.balanced_probs(g, age)
    assert np.isclose(p.sum(), 1)
    bands = np.array([age_band(a) for a in age])
    for b in np.unique(bands):
        m = bands == b
        assert np.isclose(p[m].sum(), m.mean())                    # band share kept
        per_race = defaultdict(float)
        for gi, pi in zip(g[m], p[m]):
            per_race[gi] += pi
        vals = np.array(list(per_race.values()))
        assert np.allclose(vals, vals[0])                           # races equal within band
        # within a cell every row has the same probability
        for r in per_race:
            assert np.ptp(p[m & (g == r)]) < 1e-15


def test_plain_path_bit_identical_to_committed_A():
    """The refactor must not change model A (committed results at 8cfc7aa)."""
    import importlib.util
    old = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if old is None or not old.exists():
        print("  (skipped: pass the committed train.py path as argv[1])")
        return
    spec = importlib.util.spec_from_file_location("train_old", old)
    O = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(O)
    X, y, g, _ = synth(1500, seed=2)
    folds = np.arange(len(y)) % 5
    grid = {"hidden": [0, 32], "lr": [3e-3], "weight_decay": [1e-4]}
    fixed_old = {"max_epochs": 8, "batch_size": 128, "dropout": 0.1, "loss": "l1"}
    p_old, r_old = O.oof_predictions(X, y, folds, grid, fixed_old, log=lambda *_: None)
    p_new, r_new = T.oof_predictions(X, y, folds, grid, {**fixed_old, "method": "plain"},
                                     log=lambda *_: None)
    assert np.array_equal(p_old, p_new), np.abs(p_old - p_new).max()
    assert [r["chosen"] for r in r_old] == [r["chosen"] for r in r_new]
    assert [r["epochs"] for r in r_old] == [r["epochs"] for r in r_new]


def test_rank_key():
    a = {"mae": 4.0, "worst_group_tpr": 0.80, "max_adult_fpr": 0.01, "feasible": True}
    b = {"mae": 4.5, "worst_group_tpr": 0.85, "max_adult_fpr": 0.02, "feasible": True}
    c = {"mae": 3.9, "worst_group_tpr": 0.99, "max_adult_fpr": 0.20, "feasible": False}
    d = {"mae": 4.4, "worst_group_tpr": 0.85, "max_adult_fpr": 0.03, "feasible": True}
    best = min([a, b, c, d], key=lambda s: T._rank_key(s, "worst_group"))
    assert best is d                       # c infeasible; b, d tie on TPR -> lower MAE
    assert min([a, b, c, d], key=lambda s: T._rank_key(s, "mae")) is c
    e = {**c, "max_adult_fpr": 0.10}
    assert min([c, e], key=lambda s: T._rank_key(s, "worst_group")) is e   # none feasible


def test_dro_upweights_hard_group():
    X, y, g, age = synth(4000, seed=3, noisy_black_teens=8.0)
    st = T.Standardiser().fit(X)
    s = {**T.FIXED_C, "hidden": 0, "lr": 3e-3, "weight_decay": 1e-4, "dro_eta": 0.05}
    _, _, info = T.fit_head(st(X), y, s, 10, 0, "cpu", None, g, age)
    top = list(info["dro_final_weights_top5"])
    print("  DRO top groups:", {k: round(v, 3) for k, v in info["dro_final_weights_top5"].items()},
          "uniform", round(info["dro_uniform_weight"], 3))
    assert top[0] == "Black|10-17"
    s0 = {**s, "dro_eta": 0.0}                    # eta 0 -> weights stay uniform
    _, _, info0 = T.fit_head(st(X), y, s0, 2, 0, "cpu", None, g, age)
    assert all(np.isclose(v, info0["dro_uniform_weight"])
               for v in info0["dro_final_weights_top5"].values())


def test_oof_worst_group_runs_for_B_and_C():
    # n large enough that each inner fold has hundreds of adults per group, as in UTKFace;
    # with ~20 adults per group one false positive = FPR 0.05 and only an untrained,
    # flag-nobody epoch stays feasible (seen with n = 2000)
    X, y, g, _ = synth(10000, seed=4)
    folds = np.arange(len(y)) % 5
    small = {"hidden": [0], "lr": [3e-3], "weight_decay": [1e-4]}
    for fixed, grid in [({**T.FIXED_B, "max_epochs": 8}, small),
                        ({**T.FIXED_C, "max_epochs": 8}, {**small, "dro_eta": [0.01]})]:
        p, rec = T.oof_predictions(X, y, folds, grid, fixed, groups=g, select="worst_group",
                                   log=lambda *_: None)
        mae = np.abs(p - y).mean()
        assert mae < 6 and all("inner_val_worst_group_tpr" in r for r in rec), mae
        assert min(r["inner_val_worst_group_tpr"] for r in rec) > 0.5
        print(f"  {fixed['method']}: OOF MAE {mae:.2f}, epochs {[r['epochs'] for r in rec]}")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
