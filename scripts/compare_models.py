"""Holm-adjusted success verdicts for every finished model vs model A (SUCCESS_CRITERIA §4).

  CRF_ENV=colab python scripts/compare_models.py
Reads the model JSONs under DRIVE_ROOT/results/models/ (each holds its own paired
comparison with A on the shared bootstrap resamples). Holm is applied across the full
family in config.HOLM_FAMILY; members that have not run yet count as p = 1, so verdicts
are 'interim' (conservative) until the family is complete. config.CONTEXT_MODELS (B, C, C2)
are reported alongside with their unadjusted p only: they are not tested conditions.
Writes results/comparisons.json (aggregate only).
"""

import json
from pathlib import Path

from _common import REPO
from configs import config as C
from src import metrics as M
from src.provenance import write_result


def main():
    root = Path(C.DRIVE_ROOT) / "results" / "models"
    A = json.loads((root / "A.json").read_text())
    done = {m: json.loads((root / f"{m}.json").read_text())
            for m in C.HOLM_FAMILY if (root / f"{m}.json").exists()}
    for m, d in done.items():
        if d["git"]["dirty"] is not False:
            raise SystemExit(f"{m}.json came from modified code")
        if d["inputs"] != A["inputs"]:
            raise SystemExit(f"{m} used different embeddings/folds than A")
    pvals = {m: done[m]["paired_vs_A"]["gap_reduction_A_minus_M"]["p_one_sided"]
             if m in done else 1.0 for m in C.HOLM_FAMILY}
    adj = M.holm(pvals)
    out = {"family": C.HOLM_FAMILY, "complete": len(done) == len(C.HOLM_FAMILY),
           "missing_count_as_p1": [m for m in C.HOLM_FAMILY if m not in done],
           "A_gap": A["report"]["tpr_gap_W_minus_B"], "models": {}}
    for m, d in done.items():
        pr, rep = d["paired_vs_A"], d["report"]
        out["models"][m] = {
            "commit": d["git"]["commit"], "gap": rep["tpr_gap_W_minus_B"],
            "teen_gap": rep["tpr_teen_gap_W_minus_B"],
            "teen_tpr": {g: r["tpr_teen"] for g, r in rep["by_group"].items()},
            "adult_fpr": {g: r["fpr_adult"] for g, r in rep["by_group"].items()},
            "mae": rep["mae_overall"], "paired": pr, "p_holm": adj[m],
            "verdict": M.verdict(A["report"], rep, pr, adj[m])}
        if "leakage_probe" in d:
            out["models"][m]["leakage_probe_bal_acc"] = d["leakage_probe"]["bal_acc"]
    out["context_models"] = {}
    for m in C.CONTEXT_MODELS:
        f = root / f"{m}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        if d["git"]["dirty"] is not False or d["inputs"] != A["inputs"]:
            raise SystemExit(f"{m}.json: modified code or different inputs")
        pr, rep = d["paired_vs_A"], d["report"]
        out["context_models"][m] = {
            "commit": d["git"]["commit"], "gap": rep["tpr_gap_W_minus_B"],
            "teen_gap": rep["tpr_teen_gap_W_minus_B"], "mae": rep["mae_overall"], "paired": pr,
            "p_unadjusted": pr["gap_reduction_A_minus_M"]["p_one_sided"],
            "note": "context only, not in the Holm family; family-of-9 verdicts in "
                    "results/comparisons_family9.json"}
    res = write_result(REPO / "results" / "comparisons.json", out, C.as_dict(), REPO)
    tag = "" if out["complete"] else f"  (INTERIM: {len(out['missing_count_as_p1'])} of " \
        f"{len(C.HOLM_FAMILY)} family members not run, counted as p = 1)"
    print(f"A gap {out['A_gap']['est']:.3f}{tag}")
    for m, r in out["models"].items():
        v, g = r["verdict"], r["gap"]
        print(f"{m}: gap {g['est']:.3f} [{g['ci'][0]:.3f}, {g['ci'][1]:.3f}]  "
              f"p_holm {r['p_holm']:.4f}  c1 {v['c1_fairer_holm']} c2 {v['c2_small_gap']} "
              f"c3 {v['c3_adult_fpr']} c4 {v['c4_mae_noninferior']}  -> {v['outcome']}")
    for m, r in out["context_models"].items():
        g = r["gap"]
        print(f"{m} (context, not tested): gap {g['est']:.3f} [{g['ci'][0]:.3f}, {g['ci'][1]:.3f}]"
              f"  p unadjusted {r['p_unadjusted']:.4f}")
    print("->", res)


if __name__ == "__main__":
    main()
