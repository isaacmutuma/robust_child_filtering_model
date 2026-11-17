"""Local tests on synthetic data only (no images, no embeddings, no real ids).

  python tests/test_metrics_and_train.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs import config as C  # noqa: E402
from src import metrics as M  # noqa: E402
from src.train import oof_predictions  # noqa: E402

GROUPS = ["White", "Black", "Asian", "Indian", "Others"]


def population(n=6000, seed=3):
    rng = np.random.default_rng(seed)
    g = rng.choice(GROUPS, n, p=[.42, .19, .15, .17, .07])
    age = rng.integers(0, 80, n).astype(float)
    return [f"s{i:05d}" for i in range(n)], g, age, rng


def preds_with_teen_bias(age, g, rng, black_teen_shift):
    """Noisy age predictions; Black teenagers pushed older by `black_teen_shift` years."""
    p = age + rng.normal(0, 2.5, len(age))
    p[(g == "Black") & (age >= 10) & (age <= 17)] += black_teen_shift
    return p


def test_point_estimates_by_hand():
    ids = [f"x{i}" for i in range(8)]
    g = ["White"] * 4 + ["Black"] * 4
    age = [5, 5, 15, 30, 5, 15, 15, 30]
    pred = [3, 20, 10, 25, 4, 30, 12, 10]          # flagged: 1,0,1,0 | 1,0,1,1
    d = M.prepare(M.frame(ids, g, age, pred))
    rep = M.model_report(d, M.BootPlan(d, B=200))
    w9, w17 = C.STRATUM_WEIGHTS["0-9"], C.STRATUM_WEIGHTS["10-17"]
    assert np.isclose(rep["by_group"]["White"]["tpr_strat"]["est"], w9 * 0.5 + w17 * 1.0)
    assert np.isclose(rep["by_group"]["Black"]["tpr_strat"]["est"], w9 * 1.0 + w17 * 0.5)
    assert rep["by_group"]["Black"]["fpr_adult"]["est"] == 1.0
    assert rep["by_group"]["White"]["fpr_adult"]["est"] == 0.0
    assert np.isclose(rep["tpr_gap_W_minus_B"]["est"], (w17 - w9) * 0.5)


def test_plan_is_deterministic_and_checked():
    ids, g, age, rng = population(800)
    d = M.prepare(M.frame(ids, g, age, age))
    p1, p2 = M.BootPlan(d, B=50), M.BootPlan(d.sample(frac=1, random_state=1)
                                             .sort_values("id").reset_index(drop=True), B=50)
    assert all(np.array_equal(p1.W[k], p2.W[k]) for k in p1.W)
    try:
        M.estimates(d.iloc[1:].reset_index(drop=True), p1)
        raise AssertionError("plan accepted a different image set")
    except ValueError:
        pass


def test_paired_identical_and_planted():
    ids, g, age, rng = population()
    pA = preds_with_teen_bias(age, g, rng, black_teen_shift=6.0)
    dA = M.prepare(M.frame(ids, g, age, pA))
    plan = M.BootPlan(dA, B=500)
    repA = M.model_report(dA, plan)
    assert repA["tpr_gap_W_minus_B"]["est"] > 0.05            # planted bias is visible

    same = M.paired_vs_A(dA, dA, plan)
    assert same["gap_reduction_A_minus_M"]["est"] == 0
    assert same["gap_reduction_A_minus_M"]["p_one_sided"] == 1.0
    assert same["mae_increase_M_minus_A"]["ci"] == [0.0, 0.0]

    pM = pA.copy()
    fix = (g == "Black") & (age >= 10) & (age <= 17)
    pM[fix] -= 6.0                                            # remove the bias
    dM = M.prepare(M.frame(ids, g, age, pM))
    repM = M.model_report(dM, plan)
    pr = M.paired_vs_A(dA, dM, plan)
    assert pr["gap_reduction_A_minus_M"]["est"] > 0.05
    assert pr["gap_reduction_A_minus_M"]["p_one_sided"] < 0.01
    v = M.verdict(repA, repM, pr, p_holm=pr["gap_reduction_A_minus_M"]["p_one_sided"])
    assert v["c1_fairer_holm"] and v["c3_adult_fpr"] and v["c4_mae_noninferior"]
    print("planted fix:", {k: v[k] for k in ("c2_small_gap", "outcome")},
          "M gap", repM["tpr_gap_W_minus_B"])

    # cost moved onto adults: flag every Black adult -> criterion 3 must fail
    pBad = pM.copy()
    pBad[(g == "Black") & (age >= 20) & (age <= 59)] = 10.0
    dB = M.prepare(M.frame(ids, g, age, pBad))
    vB = M.verdict(repA, M.model_report(dB, plan), M.paired_vs_A(dA, dB, plan), p_holm=0.001)
    assert not vB["c3_adult_fpr"] and not vB["c3_by_group"]["Black"] and vB["outcome"] == "neither"


def test_holm():
    adj = M.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert np.allclose([adj["a"], adj["c"], adj["b"]], [0.03, 0.06, 0.06])


def test_oof_training():
    rng = np.random.default_rng(0)
    n, dim = 1500, 32
    age = rng.integers(0, 80, n).astype(float)
    w = rng.normal(size=dim)
    X = (np.outer(age - 40, w) / 40 + rng.normal(0, 0.3, (n, dim)) + 5).astype(np.float32)
    folds = np.arange(n) % 5
    grid = {"hidden": [0, 64], "lr": [3e-3], "weight_decay": [1e-4]}
    fixed = {"max_epochs": 15, "batch_size": 128, "dropout": 0.1, "loss": "l1"}
    p1, rec = oof_predictions(X, age, folds, grid, fixed, log=lambda *_: None)
    p2, _ = oof_predictions(X, age, folds, grid, fixed, log=lambda *_: None)
    mae, base = np.abs(p1 - age).mean(), np.abs(age - np.median(age)).mean()
    print(f"synthetic OOF MAE {mae:.2f} vs median baseline {base:.2f}; "
          f"chosen {[r['chosen'] for r in rec][:2]}")
    assert mae < 0.3 * base
    assert np.allclose(p1, p2), "training is not deterministic"
    assert all(r["inner_val_fold"] == (r["test_fold"] + 1) % 5 for r in rec)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
