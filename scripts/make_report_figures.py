"""Report figures from the committed result files, with the SAME plotting code as the submission notebook.

  python scripts/make_report_figures.py      # -> figures/teen_tpr_by_model.png, figures/layers.png

Charts of aggregate numbers only (no images of people).
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from _common import REPO  # noqa: E402
import build_submission_notebook as B  # noqa: E402

OUT = REPO / "figures"


def save(code, ns, name):
    exec(code.replace("plt.show()", ""), ns)
    plt.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close("all")
    print("->", OUT / name)


def main():
    OUT.mkdir(exist_ok=True)
    res = {m: json.loads((REPO / f"results/models/{m}.json").read_text())["report"]
           for m in ["A", "E_inv", "F1_1", "F1_4", "F2"]}
    save(B.TEEN_FIG, {"np": np, "plt": plt, "FIG_RESULTS": res}, "teen_tpr_by_model.png")
    layers = json.loads((REPO / "results/followups/layers.json").read_text())["results"]
    save(B.LAYER_FIG, {"np": np, "plt": plt, "LAYERS": layers, "POOLS": ("cls", "mean"), "DEPTHS": 25},
         "layers.png")


if __name__ == "__main__":
    main()
