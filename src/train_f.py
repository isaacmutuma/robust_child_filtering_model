"""Model F training: LoRA in CLIP's last blocks + model A's head, on stored block tokens.

FModel(tokens) = head( standardise( L2( finish(visual_with_LoRA, tokens) ) ) )
where `standardise` uses the training rows' mean/SD of the frozen CLIP embeddings (A's
Standardiser) and `head` starts as a copy of model A's trained head for the same rows.
With LoRA B = 0 the model reproduces A's predictions, so every change is the fine-tuning.

F1: L1 on the standardised age. F2: as F1 plus E's adversary on the head's hidden layer z
(alternating updates: adversary steps on z detached, then one step of LoRA + head on
L1 - lambda * CE(race | z)).
"""

import copy
import time

import numpy as np
import torch
import torch.nn as nn

from src.lora_clip import add_lora, finish

FIXED_F = {"rank": 8, "alpha": 16, "lora_dropout": 0.1, "lr": 1e-4, "weight_decay": 1e-4,
           "batch_size": 64, "max_epochs": 10,
           # F2 only (E's most-chosen setting)
           "adv_lambda": 1.0, "adv_target": "max_ce", "adv_hidden": 128, "adv_lr": 1e-2,
           "adv_steps": 5, "adv_weight_decay": 1e-4}


class FModel(nn.Module):
    def __init__(self, visual, start, lora_blocks, mu, sd, head, cfg):
        super().__init__()
        self.visual = copy.deepcopy(visual)
        self.lora_params = add_lora(self.visual, lora_blocks, cfg["rank"], cfg["alpha"], cfg["lora_dropout"])
        self.start = start
        self.register_buffer("mu", torch.as_tensor(mu, dtype=torch.float32))
        self.register_buffer("sd", torch.as_tensor(sd, dtype=torch.float32))
        self.head = copy.deepcopy(head)
        for p in self.head.parameters():
            p.requires_grad_(True)

    def standardised(self, tokens):
        e = finish(self.visual, tokens, self.start).float()
        e = e / e.norm(dim=1, keepdim=True)
        return (e - self.mu) / self.sd

    def z(self, tokens):
        return self.head[:2](self.standardised(tokens))

    def forward(self, tokens):
        return self.head(self.standardised(tokens)).squeeze(1)


def _batches(idx, bs, gen=None):
    order = idx[torch.randperm(len(idx), generator=gen).numpy()] if gen is not None else idx
    for b in range(0, len(order), bs):
        yield order[b:b + bs]


def _load(cache, rows, device):
    rows = np.sort(rows)   # memmap reads in order
    return rows, torch.from_numpy(np.asarray(cache[rows], dtype=np.float32)).to(device)


def predict_f(model, cache, rows, y_mu, y_sd, device, bs=256):
    model.eval()
    out = np.empty(len(rows))
    pos = {r: i for i, r in enumerate(rows)}
    with torch.no_grad():
        for b in _batches(np.asarray(rows), bs):
            srt, t = _load(cache, b, device)
            pred = model(t).float().cpu().numpy() * y_sd + y_mu
            for r, p_ in zip(srt, pred):
                out[pos[r]] = p_
    return out


def fit_f(cache, rows_tr, ytr, gtr, A_fn, mu, sd, visual, start, lora_blocks, cfg, epochs, seed, device,
          rows_val=None, adversary=False, log=print):
    """Train F on rows_tr of the token cache. A_fn: A's predict_fn for the same training rows
    (its .model is the head to start from, .y_mu/.y_sd its age scaling).
    Returns (predict_fn(rows) -> ages, per-epoch validation predictions or None, info)."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = FModel(visual, start, lora_blocks, mu, sd, A_fn.model.cpu(), cfg).to(device)
    params = model.lora_params + list(model.head.parameters())
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    y_mu, y_sd = A_fn.y_mu, A_fn.y_sd
    ystd = {r: (a - y_mu) / y_sd for r, a in zip(rows_tr, ytr)}
    loss_fn = nn.L1Loss()
    info = {"n_lora_params": int(sum(p.numel() for p in model.lora_params)), "epoch_seconds": []}
    if adversary:
        classes = sorted(set(np.asarray(gtr).tolist()))
        gid = {r: c for r, c in zip(rows_tr, np.searchsorted(classes, np.asarray(gtr)))}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 7919)
            adv = nn.Sequential(nn.Linear(model.head[0].out_features, cfg["adv_hidden"]), nn.ReLU(),
                                nn.Linear(cfg["adv_hidden"], len(classes)))
        adv = adv.to(device)
        opt_adv = torch.optim.AdamW(adv.parameters(), lr=cfg["adv_lr"], weight_decay=cfg["adv_weight_decay"])
        ce_fn = nn.CrossEntropyLoss()
        info["adv_bal_acc_by_epoch"] = []
    use_amp = device == "cuda"
    # fp16 autocast needs loss scaling: without it the LoRA gradients of earlier blocks
    # (~5e-5) fall under fp16's smallest normal number and lose their direction
    # (measured 26 Nov 2026: cosine with fp32 gradients 0.1 in block 20, 1.0 in block 23).
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    val_preds = [] if rows_val is not None else None
    rows_tr = np.asarray(rows_tr)
    for ep in range(epochs):
        t0 = time.time()
        model.train()
        conf = np.zeros((5, 5), int) if adversary else None
        for b in _batches(rows_tr, cfg["batch_size"], gen):
            srt, t = _load(cache, b, device)
            y = torch.tensor([ystd[r] for r in srt], dtype=torch.float32, device=device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                if adversary:
                    z = model.z(t).float()
                    rid = torch.tensor([gid[r] for r in srt], device=device)
                    with torch.autocast("cuda", enabled=False):      # adversary steps in fp32
                        for _ in range(cfg["adv_steps"]):
                            opt_adv.zero_grad()
                            ce_fn(adv(z.detach()), rid).backward()
                            opt_adv.step()
                    out = model.head[2:](z).squeeze(1)
                    logits = adv(z)
                    loss = loss_fn(out, y) - cfg["adv_lambda"] * ce_fn(logits, rid)
                    np.add.at(conf, (rid.cpu().numpy(), logits.argmax(1).cpu().numpy()), 1)
                else:
                    loss = loss_fn(model(t).float(), y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        info["epoch_seconds"].append(round(time.time() - t0, 1))
        if adversary:
            info["adv_bal_acc_by_epoch"].append(float(np.mean(np.diag(conf) / np.maximum(conf.sum(1), 1))))
        if rows_val is not None:
            val_preds.append(predict_f(model, cache, rows_val, y_mu, y_sd, device))
        log(f"  epoch {ep + 1}/{epochs}: {info['epoch_seconds'][-1]:.0f} s")

    def predict_fn(rows):
        return predict_f(model, cache, rows, y_mu, y_sd, device)
    predict_fn.model = model
    return predict_fn, (np.stack(val_preds) if val_preds is not None else None), info
