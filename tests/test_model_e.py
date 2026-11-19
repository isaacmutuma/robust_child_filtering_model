"""Local tests for model E-invariant and the leakage probe (synthetic data only).

  python tests/test_model_e.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import leakage as L  # noqa: E402
from src import train as T  # noqa: E402

GROUPS = ["White", "Black", "Asian", "Indian", "Others"]
BASE = {"hidden": 32, "lr": 3e-3, "weight_decay": 1e-4}


def synth(n=4000, seed=0, dim=24, race_dims=6, race_strength=2.0):
    """Age signal in every dim plus a race signal in `race_dims` dims (unrelated to age)."""
    rng = np.random.default_rng(seed)
    g = rng.choice(GROUPS, n, p=[.42, .19, .15, .17, .07])
    age = rng.integers(0, 80, n).astype(float)
    w = rng.normal(size=dim)
    X = np.outer(age - 40, w) / 40 + rng.normal(0, 0.3, (n, dim)) + 5
    centres = rng.normal(0, race_strength, (len(GROUPS), race_dims))
    X[:, :race_dims] += centres[np.searchsorted(GROUPS, g, sorter=np.argsort(GROUPS))]
    return X.astype(np.float32), age, g


def test_lambda0_is_model_A():
    """With adv_lambda = 0 the adversary must not change A's training at all."""
    X, y, g = synth(1500, seed=1)
    st = T.Standardiser().fit(X)
    pa, va, _ = T.fit_head(st(X), y, {**T.FIXED_A, **BASE}, 6, 3, "cpu", st(X[:300]), g, y)
    for steps in (1, 5):
        pe, ve, info = T.fit_head(st(X), y, {**T.FIXED_E, **BASE, "adv_lambda": 0.0,
                                             "adv_steps": steps}, 6, 3, "cpu", st(X[:300]), g, y)
        assert np.array_equal(pa(st(X)), pe(st(X)))
        assert np.array_equal(va, ve)
        assert len(info["adv_bal_acc_by_epoch"]) == 6
    # with lambda = 0 the adversary still learns race from A's z (well above chance)
    assert info["adv_bal_acc_by_epoch"][-1] > 0.5, info["adv_bal_acc_by_epoch"]


def test_probe_scale():
    rng = np.random.default_rng(0)
    g = rng.choice(GROUPS, 3000, p=[.42, .19, .15, .17, .07])
    onehot = (g[:, None] == np.array(GROUPS)[None]).astype(float) + rng.normal(0, .1, (3000, 5))
    noise = rng.normal(size=(3000, 8))
    tr, te = np.arange(3000) < 2000, np.arange(3000) >= 2000
    p_info = L.fit_predict_probe(onehot[tr], g[tr], onehot[te])
    p_noise = L.fit_predict_probe(noise[tr], g[tr], noise[te])
    assert L.balanced_accuracy(p_info, g[te]) > 0.99
    assert abs(L.balanced_accuracy(p_noise, g[te]) - 0.2) < 0.06
    # plain accuracy of "always White" is 0.42, balanced accuracy 0.2
    assert L.balanced_accuracy(np.full(1000, "White"), g[te]) == 0.2
    s = L.summarise({"model": p_noise, "A": p_info}, g[te], B=200)
    assert s["bal_acc"]["A"]["ci"][0] > 0.95 and s["chance"] == 0.2
    d = s["diff"]["model_minus_A"]
    assert d["ci"][1] < -0.6, d


def test_adversary_strength_matters():
    """Planted race signal: the fast adversary (FIXED_E) removes part of it from z; the slow
    one first approved (lr 1e-3, 1 step) is fooled while race stays fully readable."""
    X, y, g = synth(6000, seed=2, race_strength=3.0)
    tr, te = np.arange(len(y)) < 4500, np.arange(len(y)) >= 4500
    st = T.Standardiser().fit(X[tr])
    runs = {"lambda0": {"adv_lambda": 0.0},
            "fast_prior": {"adv_lambda": 0.3, "adv_target": "prior"},
            "slow_prior": {"adv_lambda": 0.3, "adv_target": "prior", "adv_lr": 1e-3,
                           "adv_steps": 1}}
    res = {}
    for name, kw in runs.items():
        s = {**T.FIXED_E, **BASE, "lr": 1e-3, **kw}
        pf, _, info = T.fit_head(st(X[tr]), y[tr], s, 60, 0, "cpu", None, g[tr], y[tr])
        p = L.fit_predict_probe(pf.encode(st(X[tr])), g[tr], pf.encode(st(X[te])))
        res[name] = (L.balanced_accuracy(p, g[te]), float(np.abs(pf(st(X[te])) - y[te]).mean()),
                     info["adv_bal_acc_by_epoch"][-1])
        print(f"  {name:10s} probe {res[name][0]:.3f}  MAE {res[name][1]:.2f}  "
              f"adversary last epoch {res[name][2]:.3f}")
    assert res["lambda0"][0] > 0.95                          # race readable from A-style z
    assert res["fast_prior"][0] < res["lambda0"][0] - 0.2    # partly removed
    assert res["slow_prior"][0] > 0.9 and res["slow_prior"][2] < 0.8   # fooled, not removed


def test_oof_E_runs_with_probe():
    X, y, g = synth(6000, seed=4)
    folds = np.arange(len(y)) % 5
    gridA = {"hidden": [32], "lr": [3e-3], "weight_decay": [1e-4]}
    saved, savedF = T.GRID_A, T.FIXED_A
    T.GRID_A = gridA
    try:
        _, recA = T.oof_predictions(X, y, folds, gridA, {**T.FIXED_A, "max_epochs": 6},
                                    log=lambda *_: None)
        refA = [{"chosen": r["chosen"], "epochs": r["epochs"], "inner_val_mae": r["inner_val_mae"],
                 "test_mae": r["test_mae"]} for r in recA]
        probe = {}
        p, rec = T.oof_predictions(X, y, folds, {**gridA, "adv_lambda": [1.0],
                                                 "adv_target": ["max_ce", "prior"]},
                                   {**T.FIXED_E, "max_epochs": 6}, groups=g,
                                   select="worst_group_v3", ref_A=refA, log=lambda *_: None,
                                   probe_out=probe)
        assert set(probe) == {"model", "A", "clip_raw"}
        for r in rec:
            lp = r["leakage_probe"]
            # A's own refit is reproduced exactly (same setting, epochs, seed, CPU)
            assert abs(lp["A_refit_test_mae_diff_vs_A_record"]) < 1e-9, lp
            assert len(r["refit_info"]["adv_bal_acc_by_epoch"]) == r["epochs"]
        s = L.summarise(probe, g, B=200)
        print("  OOF probe bal acc:", {k: round(v["est"], 3) for k, v in s["bal_acc"].items()},
              f"OOF MAE {np.abs(p - y).mean():.2f}")
    finally:
        T.GRID_A, T.FIXED_A = saved, savedF


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
