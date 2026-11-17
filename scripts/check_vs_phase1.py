"""Cross-check: zero-shot CLIP L/14 margins recomputed from OUR UTKFace embeddings must
match the audit's Phase 1 QuickGELU run (per_image__clip_vitl14__utkface__full.csv on Drive).

  CRF_ENV=colab python scripts/check_vs_phase1.py
Same weights, preprocessing and activation => same embeddings. PASS needs all 23,705 ids
matched, max |margin difference| < 1e-3 and identical child decisions. Writes the
aggregate result to results/embeddings/check_vs_phase1.json.
"""

from pathlib import Path

import numpy as np

from _common import REPO
from configs import config as C
from src.provenance import write_result

TOL = 1e-3


def main():
    import pandas as pd
    import torch
    from src.clip_model import load_clip
    z = np.load(Path(C.EMB_DIR) / f"utkface__{C.EMB_TAG}.npz")
    model, _, tok, winfo = load_clip("cuda")
    prompts = C.AUDIT_CHILD_PROMPTS + C.AUDIT_ADULT_PROMPTS
    with torch.no_grad():
        t = model.encode_text(tok(prompts).cuda()).float().cpu().numpy()
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    f = z["emb"] / np.linalg.norm(z["emb"], axis=1, keepdims=True)
    sims = f @ t.T
    nc = len(C.AUDIT_CHILD_PROMPTS)
    ours = pd.DataFrame({"id": z["ids"], "margin": sims[:, :nc].max(1) - sims[:, nc:].max(1)})

    p1 = pd.read_csv(Path(C.AUDIT_PER_IMAGE_DIR) / "per_image__clip_vitl14__utkface__full.csv",
                     usecols=["id", "score", "is_child"], dtype={"id": str})
    m = ours.merge(p1, on="id", how="inner")
    d = np.abs(m["margin"].to_numpy() - m["score"].to_numpy())
    agree = float(((m["margin"] > 0) == m["is_child"].astype(bool)).mean())
    out = {"n_ours": len(ours), "n_phase1": len(p1), "n_matched": len(m),
           "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
           "pearson_r": float(np.corrcoef(m["margin"], m["score"])[0, 1]),
           "decision_agreement": agree, "tolerance": TOL, "weights": winfo}
    out["pass"] = bool(len(m) == C.SOURCES["utkface"]["n_records"] and d.max() < TOL
                       and agree == 1.0)
    res = write_result(REPO / "results" / "embeddings" / "check_vs_phase1.json", out,
                       C.as_dict(), REPO)
    print({k: v for k, v in out.items() if k != "weights"}, "\n->", res)
    if not out["pass"]:
        raise SystemExit("CROSS-CHECK FAILED: embeddings do not reproduce Phase 1 CLIP L/14")


if __name__ == "__main__":
    main()
