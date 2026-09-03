"""Block Attention Residuals (spec §3.5) — one application point.

The load-bearing rules, verified against the official pseudocode
(arXiv 2603.15031) and LOCKED by the spec:

- Sources are sums of sublayer deltas, never residual-stream snapshots:
  blocks[0] = token embedding output e; blocks[b] = sum of sublayer deltas
  within completed block b; plus the current block's running partial b_n.
  The model (refit.py) owns that bookkeeping; this module just consumes
  the stacked sources.
- Values raw, keys-only RMSNorm: V = stack(sources); K = RMSNorm(V).
- One zero-initialized pseudo-query per application point -> uniform softmax
  at step 0 -> output = (e + sum(blocks) + b_n)/(N+1) = h_PreNorm/(N+1).
  RMSNorm is scale-invariant, so every sublayer sees exactly the donor's
  PreNorm input at step 0. Do not break this: no random query init, no
  normalizing values into the sum, no stream snapshots.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from train.src.model.decoder import LlamaRMSNorm


class BlockAttnRes(nn.Module):
    """AttnRes for a single application point (pre-attn or pre-mlp of a layer).

    forward(sources, h_prenorm) -> the sublayer input. `h_prenorm` is the
    current residual stream, used only by the optional [EXPERIMENTAL] gate.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-05, gate: bool = False) -> None:
        super().__init__()
        self.pseudo_query = nn.Parameter(torch.zeros(hidden_size))
        self.key_norm = LlamaRMSNorm(hidden_size, eps)
        # [EXPERIMENTAL] scalar gate: h = h_prenorm + g*(out - h_prenorm), g init 0.
        self.gate = nn.Parameter(torch.zeros(1)) if gate else None

    def forward(
        self,
        sources: list[torch.Tensor],
        h_prenorm: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # sources: N tensors [B, T, D] (embedding, completed block delta-sums,
        # current partial). At zero-init pseudo-query the softmax over the
        # source axis is exactly uniform.
        # Looped, NOT stacked: stacking the N sources ([N, B, T, D] ~670 MB
        # bf16 at 8k) + the fp32 key-norm upcast (~1.3 GB) + the weighted
        # product is a ~3.4 GiB transient at EACH of 66 application points,
        # which OOM'd the 96 GB box once optimizer states + bf16 grads were
        # resident (bring-up OOM #5, 2026-09-03). Per-source looping keeps
        # the transient at one source's norm (~0.2 GiB); same math, modulo
        # bf16 accumulation order in the final sum.
        logits = torch.stack([
            torch.einsum("d,...d->...", self.pseudo_query, self.key_norm(v))
            for v in sources
        ], dim=0)  # [N, B, T]
        alpha = torch.softmax(logits, dim=0)
        out = torch.zeros_like(sources[0])
        for a, v in zip(alpha, sources):
            out.add_(a.unsqueeze(-1) * v)  # values stay raw
        if self.gate is not None:
            if h_prenorm is None:
                raise ValueError("gated AttnRes requires h_prenorm")
            out = h_prenorm + self.gate * (out - h_prenorm)
        return out
