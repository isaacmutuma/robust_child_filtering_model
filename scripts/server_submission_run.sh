#!/usr/bin/env bash
# Full submission run on the course GPU server, unattended (one command, then log out of JupyterHub if you like).
#
#   bash scripts/server_submission_run.sh <GPU_ID> "<demo image numbers, comma-separated or empty>"
#   e.g. bash scripts/server_submission_run.sh 4 "12,345,6789"
#
# What it does (log: ~/submission_run.log; status: bash scripts/server_submission_status.sh):
#   1. fetches ICCWD into ~/iccwd/images with the dataset authors' pinned script (CPU; runs alongside step 2)
#   2. executes the submission notebook with GPU_ID, RUN["recheck"] = True, RUN_ICCWD = True,
#      ICCWD_DIR = ~/iccwd and DEMO_IMAGES; section 10 waits for the fetch to finish
#   3. deletes ~/iccwd (ethics rule of 26 Nov 2026: temporary copy only) and checks it is gone
# Output: ~/submission/202555404_IsaacMutuma.ipynb (OUTSIDE the repo: it contains the demo photos; never commit it).
set -uo pipefail
GPU_ID="${1:?usage: server_submission_run.sh <GPU_ID> \"<demo numbers>\"}"
DEMO="${2:-}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="$HOME/miniconda3/envs/crf/bin/python"
ICC="$HOME/iccwd"
OUT="$HOME/submission"
LOG="$HOME/submission_run.log"

if [[ "${INNER:-0}" != "1" ]]; then          # relaunch detached so the run survives a closed terminal
  INNER=1 setsid nohup bash "$0" "$GPU_ID" "$DEMO" > "$LOG" 2>&1 < /dev/null &
  echo "started (PID $!); follow with: tail -f $LOG"
  exit 0
fi

echo "== $(date) start; repo $(git -C "$REPO" rev-parse --short HEAD), dirty: $(git -C "$REPO" status --porcelain -- . ':!results' ':!figures' | wc -l) files"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader -i "$GPU_ID"
"$PY" -m pip -q install pdqhash nbconvert nbformat >/dev/null

# ---- 1. ICCWD fetch (background) ----------------------------------------------------------------------
mkdir -p "$ICC/images" "$OUT"
rm -f "$ICC/COMPLETE"
(
  trap 'touch "$ICC/COMPLETE"' EXIT                 # section 10 must never wait forever, even if the fetch fails
  set -e
  CSV=$("$PY" - <<'EOF'
import hashlib
from huggingface_hub import hf_hub_download
p = hf_hub_download("amcretu/iccwd", "image_caption_children_in_the_wild_dataset.csv", repo_type="dataset",
                    revision="9f76b200c40a8a88c19dc17af8b842581f10f5f5")
h = hashlib.sha256(open(p, "rb").read()).hexdigest()
assert h == "15f1d0142aac67740d6f41a233fe157d580c256f16705fd5c86cdfa8209c694e", "ICCWD label file changed"
print(p)
EOF
)
  rm -rf "$ICC/code" && git clone -q https://github.com/spring-epfl/iccwd.git "$ICC/code"
  git -C "$ICC/code" checkout -q ee6695fa063655039346a06d127a347321887047
  cd "$ICC" && "$PY" "$ICC/code/dataset_download_images_verify.py" --folder="$ICC/images" --dataframe="$CSV" > "$ICC/fetch.log" 2>&1
  echo "== $(date) ICCWD fetch done: $(ls "$ICC/images" | wc -l) images kept (Drive copy: 8,281)"
) || echo "!! ICCWD fetch failed; see $ICC/fetch.log" &
FETCH_PID=$!

# ---- 2. notebook -------------------------------------------------------------------------------------
"$PY" "$REPO/scripts/build_submission_notebook.py" >/dev/null
"$PY" - "$REPO" "$OUT" "$GPU_ID" "$DEMO" "$ICC" <<'EOF'
import json, sys
repo, out, gpu, demo, icc = sys.argv[1:]
nb = json.load(open(f"{repo}/notebooks/submission/DLP_term_project.ipynb"))
demo_list = [int(x) for x in demo.replace(" ", "").split(",") if x]
n = {"gpu": 0, "run": 0, "icc": 0}
for c in nb["cells"]:
    if c["cell_type"] != "code":
        continue
    s = "".join(c["source"])
    if "GPU_ID = None" in s:
        s = s.replace("GPU_ID = None", f'GPU_ID = "{gpu}"', 1); n["gpu"] += 1
    if '"recheck": False' in s:
        s = s.replace('"recheck": False', '"recheck": True', 1); n["run"] += 1
    if "RUN_ICCWD = False" in s:
        s = (s.replace("RUN_ICCWD = False", "RUN_ICCWD = True", 1)
              .replace("ICCWD_DIR = None", f'ICCWD_DIR = "{icc}"', 1)
              .replace("DEMO_IMAGES = []", f"DEMO_IMAGES = {demo_list}", 1)); n["icc"] += 1
    c["source"] = s.splitlines(keepends=True)
assert n == {"gpu": 1, "run": 1, "icc": 1}, n
json.dump(nb, open(f"{out}/202555404_IsaacMutuma.ipynb", "w"), indent=1, ensure_ascii=False)
print("settings:", {"GPU_ID": gpu, "recheck": True, "RUN_ICCWD": True, "ICCWD_DIR": icc, "DEMO_IMAGES": demo_list})
EOF
echo "== $(date) notebook start"
cd "$OUT" && "$PY" -m jupyter nbconvert --to notebook --execute --inplace --allow-errors \
  --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=crf "$OUT/202555404_IsaacMutuma.ipynb"
echo "== $(date) notebook done (exit $?)"
"$PY" - "$OUT/202555404_IsaacMutuma.ipynb" <<'EOF'
import json, sys
nb = json.load(open(sys.argv[1]))
errs = [(i, o.get("ename"), o.get("evalue", "")[:200]) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"
        for o in c.get("outputs", []) if o.get("output_type") == "error"]
print("cells with errors:", errs if errs else "none")
EOF

# ---- 3. delete the temporary ICCWD copy ------------------------------------------------------------
wait "$FETCH_PID"
rm -rf "$ICC"
[[ -e "$ICC" ]] && echo "!! $ICC still exists" || echo "== ICCWD copy deleted ($ICC gone)"
echo "== $(date) all done. Download ~/submission/202555404_IsaacMutuma.ipynb, then delete it from the server."
