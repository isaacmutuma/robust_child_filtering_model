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



# ── v2 selection rule and C2 (2026-11-18) ────────────────────────────────────
def _load(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_v1_paths_unchanged():
    """B and C under rule v1 must give the same predictions as the committed v1 code."""
    old = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    if old is None or not old.exists():
        print("  (skipped: pass the v1 train.py path as argv[2])")
        return
    O = _load(old, "train_v1")
    X, y, g, age = synth(4000, seed=5)
    folds = np.arange(len(y)) % 5
    small = {"hidden": [0], "lr": [3e-3], "weight_decay": [1e-4]}
    for fixed, grid in [({**T.FIXED_B, "max_epochs": 4}, small),
                        ({**T.FIXED_C, "max_epochs": 4}, {**small, "dro_eta": [0.1]})]:
        p_old, _ = O.oof_predictions(X, y, folds, grid, fixed, groups=g, select="worst_group",
                                     log=lambda *_: None)
        p_new, _ = T.oof_predictions(X, y, folds, grid, fixed, groups=g, select="worst_group",
                                     log=lambda *_: None)
        assert np.array_equal(p_old, p_new), (fixed["method"], np.abs(p_old - p_new).max())


def test_c2_eta0_equals_plain():
    X, y, g, age = synth(3000, seed=6)
    st = T.Standardiser().fit(X)
    base = {"hidden": 32, "lr": 3e-3, "weight_decay": 1e-4}
    pf_plain, _, _ = T.fit_head(st(X), y, {**T.FIXED_A, **base}, 5, 0, "cpu", None, g, age)
    pf_c2, _, info = T.fit_head(st(X), y, {**T.FIXED_C2, **base, "dro_eta": 0.0}, 5, 0, "cpu",
                                None, g, age)
    d = np.abs(pf_plain(st(X)) - pf_c2(st(X))).max()
    print(f"  C2 eta=0 vs plain: max prediction difference {d:.2e} years")
    assert d < 1e-3 and info["dro_init"] == "natural"
    assert all(abs(v - 1) < 1e-4 for v in info["dro_weight_over_share_top5"].values())
    # with eta > 0 the hard group gains weight relative to its share
    X2, y2, g2, a2 = synth(4000, seed=3, noisy_black_teens=8.0)
    st2 = T.Standardiser().fit(X2)
    _, _, info2 = T.fit_head(st2(X2), y2, {**T.FIXED_C2, **base, "dro_eta": 0.01}, 10, 0, "cpu",
                             None, g2, a2)
    top = list(info2["dro_weight_over_share_top5"])
    print("  C2 weight/share top:", {k: round(v, 2) for k, v in info2["dro_weight_over_share_top5"].items()})
    assert top[0] == "Black|10-17"


def test_v2_rule():
    ref = {"fpr": {"White": 0.006, "Black": 0.004, "Asian": 0.003, "Indian": 0.003}, "mae": 4.4}
    rng = np.random.default_rng(0)
    n = 6000
    g = rng.choice(GROUPS, n, p=[.42, .19, .15, .17, .07])
    y = rng.integers(0, 80, n).astype(float)
    good = y + rng.normal(0, 3, n)
    st = T.selection_stats_v2(good, y, g, ref)
    blk_ad = (g == "Black") & (y >= 20) & (y <= 59)
    bad = good.copy()
    bad[np.flatnonzero(blk_ad)[: int(0.05 * blk_ad.sum())]] = 10.0    # 5% of Black adults flagged
    sb = T.selection_stats_v2(bad, y, g, ref)
    assert not sb["feasible"] and sb["n_violations"] >= 1 and sb["fpr_excess"] > 0
    worse = good + rng.normal(0, 4, n)                                   # MAE much higher
    sw = T.selection_stats_v2(worse, y, g, {**ref, "mae": st["mae"]})
    assert sw["mae_excess"] > 0 and not sw["feasible"]
    feas = {**st, "feasible": True, "n_violations": 0, "fpr_excess": 0.0, "mae_excess": 0.0}
    assert min([sb, feas], key=lambda s: T._rank_key(s, "worst_group_v2")) is feas


def test_oof_v2_with_A_reference():
    X, y, g, _ = synth(10000, seed=4)
    folds = np.arange(len(y)) % 5
    gridA = {"hidden": [0, 32], "lr": [3e-3], "weight_decay": [1e-4]}
    saved = T.GRID_A
    T.GRID_A = gridA                     # reference uses A's grid order
    try:
        fixedA = {**T.FIXED_A, "max_epochs": 6}
        _, recA = T.oof_predictions(X, y, folds, gridA, fixedA, log=lambda *_: None)
        refA = [{"chosen": r["chosen"], "epochs": r["epochs"], "inner_val_mae": r["inner_val_mae"]}
                for r in recA]
        for fixed, grid in [({**T.FIXED_B, "max_epochs": 6}, {"hidden": [0], "lr": [3e-3],
                                                             "weight_decay": [1e-4]}),
                            ({**T.FIXED_C2, "max_epochs": 6}, {"hidden": [0], "lr": [3e-3],
                                                              "weight_decay": [1e-4],
                                                              "dro_eta": [0.01]})]:
            p, rec = T.oof_predictions(X, y, folds, grid, fixed, groups=g,
                                       select="worst_group_v2", ref_A=refA, log=lambda *_: None)
            diffs = [r["reference_A"]["mae_diff_vs_A_record"] for r in rec]
            assert max(abs(d) for d in diffs) < 1e-9, diffs        # exact on CPU
            print(f"  {fixed['method']}/{fixed.get('dro_init', '-')}: OOF MAE "
                  f"{np.abs(p - y).mean():.2f}, feasible folds "
                  f"{sum(r['inner_val_feasible'] for r in rec)}/5")
    finally:
        T.GRID_A = saved


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
