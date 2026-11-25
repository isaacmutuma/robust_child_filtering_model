"""The submission notebook's code must give the same results as the tested repo code.

  python tests/test_submission_notebook.py
Builds notebooks/submission/DLP_term_project.ipynb, executes its library cells (not the
download / CLIP / experiment cells) on synthetic data and compares with src/.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")]
import build_submission_notebook as B  # noqa: E402
from src import folds as RF  # noqa: E402
from src import leace as RL  # noqa: E402
from src import train as T  # noqa: E402
from test_model_e import synth  # noqa: E402

LIB_MARKERS = ("from types import SimpleNamespace", "def age_band", "STRATA = list", "def fit_predict_probe",
               "# ── Age head", "def _onehot")


def notebook_namespace():
    nb = B.build()
    ns = {}
    for c in nb["cells"]:
        src = "".join(c["source"])
        if c["cell_type"] == "code" and any(m in src for m in LIB_MARKERS):
            lib = src.split("\nfold_of =")[0].split("\ndef mlp_probe")[0]   # drop experiment parts
            exec(compile(lib, "<notebook>", "exec"), ns)
    return ns


def test_library_matches_repo():
    ns = notebook_namespace()
    X, y, g = synth(2500, seed=3, race_strength=2.0)
    ids = [f"id{i:05d}" for i in range(len(y))]
    # folds: verbatim copy
    assert ns["make_folds"](ids, list(g), list(y)) == RF.make_folds(ids, list(g), list(y))
    folds = np.arange(len(y)) % 5
    grid = {"hidden": [0, 32], "lr": [3e-3], "weight_decay": [1e-4]}
    fixedA = {**T.FIXED_A, "max_epochs": 4}
    saved = T.GRID_A
    T.GRID_A, ns["GRID_A"] = grid, grid
    try:
        pA_repo, recA = T.oof_predictions(X, y, folds, grid, fixedA, groups=g, log=lambda *_: None)
        pA_nb, recA_nb = ns["oof_predictions"](X, y, folds, grid, fixedA, groups=g, log=lambda *_: None)
        assert np.array_equal(pA_repo, pA_nb), np.abs(pA_repo - pA_nb).max()
        assert [r["chosen"] for r in recA] == [r["chosen"] for r in recA_nb]
        ref = [{"chosen": r["chosen"], "epochs": r["epochs"], "inner_val_mae": r["inner_val_mae"],
                "test_mae": r["test_mae"]} for r in recA]
        gridE = {"hidden": [32], "lr": [3e-3], "weight_decay": [1e-4], "adv_lambda": [0.3, 1.0],
                 "adv_target": ["max_ce", "prior"]}
        fixedE = {**T.FIXED_E, "max_epochs": 4}
        pr_repo, pr_nb = {}, {}
        pE_repo, recE = T.oof_predictions(X, y, folds, gridE, fixedE, groups=g, select="worst_group_v3",
                                          ref_A=ref, probe_out=pr_repo, log=lambda *_: None)
        pE_nb, recE_nb = ns["oof_predictions"](X, y, folds, gridE, fixedE, groups=g, select="worst_group_v3",
                                               ref_A=ref, probe_out=pr_nb, log=lambda *_: None)
        assert np.array_equal(pE_repo, pE_nb), np.abs(pE_repo - pE_nb).max()
        assert [r["chosen"] for r in recE] == [r["chosen"] for r in recE_nb]
        for k in pr_repo:
            assert list(pr_repo[k]) == list(pr_nb[k]), k
        pL_repo, _ = T.oof_predictions(X, y, folds, grid, fixedA, groups=g, log=lambda *_: None,
                                       prep=RL.prep_factory(True))
        pL_nb, _ = ns["oof_predictions"](X, y, folds, grid, fixedA, groups=g, log=lambda *_: None,
                                         prep=ns["prep_factory"](True))
        assert np.array_equal(pL_repo, pL_nb)
        print("  A, E (v3 + probe) and LEACE out-of-fold predictions identical to the repo")
    finally:
        T.GRID_A = saved


def test_notebook_cells_parse_and_style():
    import ast
    nb = B.build()
    for c in nb["cells"]:
        src = "".join(c["source"])
        if c["cell_type"] == "code":
            ast.parse("\n".join(l for l in src.split("\n")
                                if not l.lstrip().startswith(("!", "%"))))
    text = json.dumps(nb)
    assert "[202555404] [Isaac Mutuma]" in text and "I predict, because" in text
    mounts = [c for c in nb["cells"] if "drive.mount" in "".join(c["source"])]
    assert len(mounts) == 1 and "RUN_ICCWD = False" in "".join(mounts[0]["source"])   # Drive only for the opt-in ICCWD part


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
