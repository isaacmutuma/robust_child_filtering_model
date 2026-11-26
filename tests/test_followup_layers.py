"""Local tests for follow-up 3 (scripts/followup_layers.py); random ViT + synthetic features, no downloads.

  python tests/test_followup_layers.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import followup_layers as FL  # noqa: E402
from src import leakage as LK  # noqa: E402
from src import metrics as M  # noqa: E402
from src.folds import make_folds  # noqa: E402
from src.lora_clip import finish, tokens_before  # noqa: E402

GROUPS = ["White", "Black", "Asian", "Indian", "Others"]


def test_layer_features_match_clip_forward():
    import open_clip
    torch.manual_seed(0)
    visual = open_clip.create_model("ViT-B-32", pretrained=None, force_quick_gelu=True).visual.eval()
    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        f, emb = FL.layer_features(visual, x)
        ref_emb = visual(x)
        tok7 = tokens_before(visual, x, 7)
        fin = finish(visual, tokens_before(visual, x, 3), 3)
    n_blocks = len(visual.transformer.resblocks)
    assert f.shape == (4, n_blocks + 1, 2, visual.transformer.width), f.shape
    assert torch.allclose(emb, ref_emb, atol=1e-4) and torch.allclose(emb, fin, atol=1e-4)
    assert torch.allclose(f[:, 7, 0], tok7[:, 0], atol=1e-5)               # depth l = tokens entering block l
    assert torch.allclose(f[:, 7, 1], tok7[:, 1:].mean(1), atol=1e-5)
    assert torch.allclose(f[0, 0, 0], f[1, 0, 0])                          # depth-0 CLS identical for all images
    print(f"  layer features: {n_blocks + 1} depths, final emb max diff {(emb - ref_emb).abs().max():.1e}")


def synthetic(n=6000, dim=32, seed=0):
    """Ages + groups; feature 'depths' where a Black-specific +4-year age offset is written from depth 3 on."""
    rng = np.random.default_rng(seed)
    groups = rng.choice(GROUPS, n, p=[0.42, 0.19, 0.15, 0.17, 0.07])
    y = np.where(rng.random(n) < 0.3, rng.integers(0, 18, n), rng.integers(18, 80, n)).astype(float)
    ids = np.array([f"img{i:05d}" for i in range(n)])
    race = np.eye(5)[np.searchsorted(sorted(GROUPS), groups)]
    noise = rng.normal(size=(n, dim))
    reps = {}
    for depth in range(6):
        age_signal = y + (4.0 * (groups == "Black") * (y < 25) if depth >= 3 else 0)
        F = noise.copy()
        F[:, 0] = age_signal / 10 + 0.2 * rng.normal(size=n)
        F[:, 1:6] = race * (1.0 if depth >= 1 else 0.0) + 0.3 * rng.normal(size=(n, 5))
        reps[depth] = F
    fold_of = make_folds(list(ids), list(groups), list(y))
    folds = np.array([fold_of[i] for i in ids])
    return ids, y, groups, folds, reps


def test_planted_bias_found_at_right_depth():
    ids, y, groups, folds, reps = synthetic()
    plan = M.BootPlan(M.prepare(M.frame(ids, groups, y, y)), B=300)
    res, race_out = {}, {}
    for depth, F in reps.items():
        r, race = FL.evaluate(f"cls_{depth}", F, ids, y, groups, folds, plan, "cpu")
        res[f"cls_{depth}"], race_out[f"cls_{depth}"] = r, race
    res["final_embedding"] = res["cls_5"]
    race = LK.summarise(race_out, groups, B=300)
    for d in range(6):
        r = res[f"cls_{d}"]
        print(f"  depth {d}: B-W err {r['age_err_gap_B_minus_W']['est']:+.2f} "
              f"age-matched {r['age_err_gap_B_minus_W_age_matched']['est']:+.2f} "
              f"gap {r['tpr_gap_W_minus_B']['est']:+.3f} race {race['bal_acc'][f'cls_{d}']['est']:.2f} "
              f"MAE {r['mae_overall']['est']:.2f}")
    for d in (0, 1, 2):
        assert abs(res[f"cls_{d}"]["age_err_gap_B_minus_W_age_matched"]["est"]) < 0.5
    for d in (3, 4, 5):
        assert res[f"cls_{d}"]["age_err_gap_B_minus_W_age_matched"]["est"] > 2.5
        assert res[f"cls_{d}"]["tpr_gap_W_minus_B"]["est"] > 0.04
    assert race["bal_acc"]["cls_0"]["est"] < 0.3 and race["bal_acc"]["cls_1"]["est"] > 0.8
    # reading rule on synthetic depths: map depth 2 -> 'F1_4 input' and depth 4 -> 'F1_1 input'
    FL.F_INPUT_DEPTH, FL.DEPTHS = {"F1_4": 2, "F1_1": 4}, 6
    rd = FL.reading(res, "cls")
    assert rd["age_err_gap_B_minus_W"]["first_depth_from_which_ci_excludes_0"] == 3, rd
    assert rd["case"] == "formed_in_F_blocks", rd
    FL.F_INPUT_DEPTH = {"F1_4": 4, "F1_1": 5}
    assert FL.reading(res, "cls")["case"] == "formed_before_F"
    print(f"  emergence depth 3 recovered; reading cases OK")


def test_age_matched_removes_age_mix_confound():
    """Same error curve for both groups, different age mixes: plain stratified gap != 0, age-matched ~0."""
    rng = np.random.default_rng(1)
    n = 8000
    groups = rng.choice(["White", "Black", "Asian", "Indian"], n)
    young = np.where(groups == "Black", 0.2, 0.8)
    base = np.where(rng.random(n) < 0.6, 0, 10)                       # stratum 0-9 or 10-17
    y = (base + np.where(rng.random(n) < young, rng.integers(0, 4, n), rng.integers(4, 8, n))).astype(float)
    pred = y + (10 - y) * 0.8 + rng.normal(0, 0.5, n)        # regression to the mean, same for all
    ids = np.array([f"i{i}" for i in range(n)])
    d = M.prepare(M.frame(ids, groups, y, pred))
    plan = M.BootPlan(d, B=200)
    am = FL.age_matched_gap(d, plan)
    plain = M.model_report(d, plan)["age_err_gap_B_minus_W"]["est"]
    print(f"  age-mix confound: stratified B-W {plain:+.2f}, age-matched {am['est']:+.2f}")
    assert abs(am["est"]) < 0.1 and abs(plain) > 0.5


def test_ridge_and_probe():
    rng = np.random.default_rng(2)
    n, dim = 3000, 20
    X = rng.normal(size=(n, dim))
    y = 3 * X[:, 0] - 2 * X[:, 1] + 40 + 0.1 * rng.normal(size=n)
    groups = np.array(GROUPS)[np.argmax(X[:, 2:7], 1)]
    folds = np.arange(n) % 5
    pred, rec = FL.ridge_oof(X, y, folds, "cpu")
    assert np.abs(pred - y).mean() < 0.2, np.abs(pred - y).mean()
    race = FL.logistic_probe_oof(X, groups, folds, "cpu")
    assert LK.balanced_accuracy(race, groups) > 0.9
    print(f"  ridge OOF MAE {np.abs(pred - y).mean():.3f} (alphas {[r['alpha'] for r in rec]}); "
          f"probe bal acc {LK.balanced_accuracy(race, groups):.3f}")


if __name__ == "__main__":
    for t in (test_layer_features_match_clip_forward, test_ridge_and_probe, test_age_matched_removes_age_mix_confound,
              test_planted_bias_found_at_right_depth):
        print(t.__name__)
        t()
    print("ALL PASS")
