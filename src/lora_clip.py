"""Model F: LoRA inside CLIP's last vision blocks, and the stored-block-output trick.

open_clip's ViT blocks use torch.nn.MultiheadAttention with one packed q/k/v weight
W (3d x d). LoRAAttention replaces it with the same computation plus a low-rank update:

    qkv = W x + b + (alpha / r) * B A dropout(x)      A: r x d, B: 3d x r

W, b and every other CLIP weight stay frozen; only A and B train. B starts at zero, so an
untouched LoRAAttention gives the same output as the original layer (checked in tests).

Because blocks 0 .. start-1 never change, `tokens_before(visual, images, start)` runs them
once (the tokens are stored), and `finish(visual, tokens, start)` runs only the remaining
blocks + ln_post + projection: finish(tokens_before(x)) equals visual(x).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAAttention(nn.Module):
    """Self-attention (batch_first) = frozen nn.MultiheadAttention + LoRA on q/k/v."""

    def __init__(self, mha, rank=8, alpha=16, dropout=0.1):
        super().__init__()
        assert mha.batch_first and mha._qkv_same_embed_dim
        self.mha = mha
        for p in self.mha.parameters():
            p.requires_grad_(False)
        d = mha.embed_dim
        self.h, self.dh = mha.num_heads, d // mha.num_heads
        self.lora_A = nn.Parameter(torch.empty(rank, d))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.lora_B = nn.Parameter(torch.zeros(3 * d, rank))
        self.scale = alpha / rank
        self.drop = nn.Dropout(dropout)
        self.keep_cls_attention = False     # set True to record CLS-token attention maps
        self.cls_attention = None

    def forward(self, q_x, k_x=None, v_x=None, need_weights=False, attn_mask=None):
        x = q_x
        n, L, d = x.shape
        qkv = F.linear(x, self.mha.in_proj_weight, self.mha.in_proj_bias)
        qkv = qkv + self.scale * (self.drop(x) @ self.lora_A.t() @ self.lora_B.t())
        q, k, v = (t.view(n, L, self.h, self.dh).transpose(1, 2) for t in qkv.chunk(3, dim=-1))
        if self.keep_cls_attention:
            w = torch.softmax((q @ k.transpose(-2, -1)) / math.sqrt(self.dh), dim=-1)
            self.cls_attention = w[:, :, 0, :].detach()          # (n, heads, L): CLS query
            out = w @ v
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(n, L, d)
        return self.mha.out_proj(out), None


def add_lora(visual, blocks, rank=8, alpha=16, dropout=0.1):
    """Freeze all of `visual`, then put LoRA into the attention of the given block indices."""
    for p in visual.parameters():
        p.requires_grad_(False)
    for i in blocks:
        blk = visual.transformer.resblocks[i]
        blk.attn = LoRAAttention(blk.attn, rank, alpha, dropout)
    return [p for i in blocks for p in (visual.transformer.resblocks[i].attn.lora_A,
                                        visual.transformer.resblocks[i].attn.lora_B)]


def tokens_before(visual, images, start):
    """Token sequence entering block `start` (patch + position embedding, ln_pre, blocks < start)."""
    x = visual._embeds(images)
    for blk in visual.transformer.resblocks[:start]:
        x = blk(x)
    return x


def finish(visual, tokens, start):
    """Blocks start.., then ln_post on the CLS token and the projection: the CLIP embedding."""
    x = tokens
    for blk in visual.transformer.resblocks[start:]:
        x = blk(x)
    pooled, _ = visual._pool(x)
    return pooled @ visual.proj if visual.proj is not None else pooled


def n_trainable(params):
    return int(sum(p.numel() for p in params))
