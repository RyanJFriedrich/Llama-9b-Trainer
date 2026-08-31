"""Held-out perplexity (spec §8 item 5). Inference-only; lm_head applied in
chunks so full logits never persist."""
from __future__ import annotations

from typing import Optional, Sequence

import torch


@torch.no_grad()
def perplexity(
    model,
    docs: Sequence[Sequence[int]],
    device: str = "cuda",
    chunk_positions: int = 256,
    max_len: Optional[int] = None,
) -> float:
    """Mean next-token perplexity over documents (final position of each doc
    excluded — no target)."""
    model = model.to(device).eval()
    W = model.lm_head.weight.to(torch.float32)
    total_nll, total_n = 0.0, 0
    for doc in docs:
        ids = torch.tensor(list(doc), dtype=torch.long, device=device).unsqueeze(0)
        if max_len is not None:
            ids = ids[:, :max_len]
        t = ids.shape[1]
        if t < 2:
            continue
        hidden = model(ids, return_hidden=True)
        for s in range(0, t - 1, chunk_positions):
            e = min(s + chunk_positions, t - 1)
            logp = torch.log_softmax(hidden[0, s:e].to(torch.float32) @ W.T, dim=-1)
            gold = ids[0, s + 1:e + 1]
            total_nll -= logp.gather(1, gold.unsqueeze(1)).sum().item()
            total_n += e - s
    return float(torch.exp(torch.tensor(total_nll / max(total_n, 1))).item())
