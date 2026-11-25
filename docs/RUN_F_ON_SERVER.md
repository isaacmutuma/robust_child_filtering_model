# Running model F on the course GPU server

Needs: Python 3.11+, PyTorch with CUDA, internet access to Hugging Face, about 20 GB free disk (UTKFace + the stored block-20 tokens of ~12.5 GB).

```bash
git clone https://github.com/isaacmutuma/robust_child_filtering_model.git
cd robust_child_filtering_model
pip install open_clip_torch==3.3.0 huggingface_hub pandas scikit-learn
export CUDA_VISIBLE_DEVICES=<your assigned GPU id>     # use only the GPU you were given

# 1. time one fold first (prints seconds per epoch)
python scripts/run_model_f.py --work /path/with/20GB --configs F1_4 --folds 0

# 2. full runs: F1-1, F1-4, then F2 (picks the better F1 depth on the validation folds)
python scripts/run_model_f.py --work /path/with/20GB --configs F1_1,F1_4,F2
```

Outputs: `results/models/F1_1.json`, `F1_4.json`, `F2.json` (aggregate only; commit them) and `WORK/oof__*.csv` (per image; never commit). If the run stops, rerun: the stored tokens are reused.
