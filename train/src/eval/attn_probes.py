"""Attention probes (spec §8 item 6): per-layer attention entropy and
attention mass beyond the SWA window, on real text.

Installs the probe hook on every RefitAttention (see model/refit.py), runs
documents through, and aggregates per layer. Both stats matter for the
hybrid's health checks: entropy shows specialization; mass-beyond-window on
SWA layers must be ~0 by construction (the mask enforces it — this is the
runtime check of invariant I7), and on GLOBAL/GATHER layers it shows
long-range usage (the p-RoPE retrieval behavior, spec §3.3).
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import torch


@torch.no_grad()
def attention_stats(
    model,
    docs: Sequence[Sequence[int]],
    device: str = "cuda",
    beyond: int = 2048,
    max_len: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Returns one dict per layer: {index, layer_type, entropy, mass_beyond}.

    entropy: mean over heads/positions of -sum(p log p) (nats).
    mass_beyond: mean attention mass on keys with offset > `beyond`.
    """
    model = model.to(device).eval()
    n_layers = len(model.model.layers)
    ent_sum = torch.zeros(n_layers)
    ent_n = torch.zeros(n_layers)
    beyond_sum = torch.zeros(n_layers)
    beyond_n = torch.zeros(n_layers)

    def make_probe(i: int):
        def probe(probs: torch.Tensor) -> None:  # [B, H, T, S]
            p = probs.to(torch.float32)
            ent = -(p * (p + 1e-12).log()).sum(-1)  # [B, H, T]
            ent_sum[i] += ent.sum().cpu()
            ent_n[i] += ent.numel()
            T, S = p.shape[-2], p.shape[-1]
            # Keys for query at row t are positions 0..t (causal); offset = t - j.
            row = torch.arange(T, device=p.device).unsqueeze(1)
            col = torch.arange(S, device=p.device).unsqueeze(0)
            far = (row - col) > beyond  # [T, S]
            mass = (p * far).sum(-1)  # [B, H, T]
            beyond_sum[i] += mass.sum().cpu()
            beyond_n[i] += mass.numel()
        return probe

    for i, layer in enumerate(model.model.layers):
        layer.self_attn.probe = make_probe(i)
    try:
        for doc in docs:
            ids = torch.tensor(list(doc), dtype=torch.long, device=device).unsqueeze(0)
            if max_len is not None:
                ids = ids[:, :max_len]
            model(ids, return_hidden=True)
    finally:
        for layer in model.model.layers:
            layer.self_attn.probe = None

    return [
        {
            "index": i,
            "layer_type": model.layer_types[i],
            "entropy": (ent_sum[i] / ent_n[i].clamp_min(1)).item(),
            "mass_beyond": (beyond_sum[i] / beyond_n[i].clamp_min(1)).item(),
        }
        for i in range(n_layers)
    ]
