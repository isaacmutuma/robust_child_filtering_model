"""Pinned OpenAI CLIP ViT-L/14 loader, with sha256 and QuickGELU checks.

open_clip only switches to QuickGELU automatically for pretrained='openai'; loading the
same weights from a file path silently uses nn.GELU (cost the audit a full rerun,
2026-11-10). We force QuickGELU and refuse to return a model that has any nn.GELU.
"""

import hashlib

from configs import config as C


def sha256(path, bs=1 << 20):
    """Hex SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def assert_quick_gelu(model):
    """Return the number of QuickGELU layers; raise if none, or if any nn.GELU is present."""
    names = [type(m).__name__ for m in model.modules()]
    n_quick, n_gelu = names.count("QuickGELU"), names.count("GELU")
    if n_quick == 0 or n_gelu:
        raise RuntimeError(f"CLIP activation check failed: QuickGELU={n_quick}, GELU={n_gelu}")
    return n_quick


def load_clip(device="cuda"):
    """(model, preprocess, tokenizer, info) for the pinned ViT-L/14, eval mode on `device`."""
    import open_clip
    from huggingface_hub import hf_hub_download
    w = C.CLIP_WEIGHTS
    path = hf_hub_download(w["hf_repo"], w["file"], revision=w["revision"],
                           cache_dir=str(C.HF_CACHE_DIR))
    got = sha256(path)
    if got != w["sha256"]:
        raise RuntimeError(f"CLIP weights sha256 mismatch ({got}); delete {path} and rerun")
    model, _, preprocess = open_clip.create_model_and_transforms(
        C.CLIP_ARCH, pretrained=path, force_quick_gelu=True)
    n_quick = assert_quick_gelu(model)
    info = {**w, "open_clip_arch": C.CLIP_ARCH, "quick_gelu": True,
            "n_quick_gelu_layers": n_quick, "open_clip_version": open_clip.__version__}
    return model.eval().to(device), preprocess, open_clip.get_tokenizer(C.CLIP_ARCH), info
