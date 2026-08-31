"""KL-to-donor metric — a spec §3.8/§8 diagnostic, not a gate.

KL(student || donor) per position on held-out text, mean over positions.
Logs the topology-shift magnitude at step 0 and drift during training; there
is no target curve (v2.0 cold-start premise). Both models run
inference-only; logits are chunked over positions so full [T, V] tensors
never persist. At dev-tiny scale both models fit anywhere; donor-scale runs
happen on the deploy box.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch


@torch.no_grad()
def kl_to_donor(
    student,
    donor,
    docs: Sequence[Sequence[int]],
    device: str = "cuda",
    chunk_positions: int = 256,
    max_len: Optional[int] = None,
) -> float:
    """Mean per-position KL(student || donor) over the given documents.

    Position i compares the two models' next-token distributions from
    context <= i; the final position of each document is excluded (no target
    follows it, matching the scorer's mask).
    """
    student = student.to(device).eval()
    donor = donor.to(device).eval()
    W_s = student.lm_head.weight.to(torch.float32)
    W_d = donor.lm_head.weight.to(torch.float32)

    total_kl, total_n = 0.0, 0
    for doc in docs:
        ids = torch.tensor(list(doc), dtype=torch.long, device=device).unsqueeze(0)
        if max_len is not None:
            ids = ids[:, :max_len]
        t = ids.shape[1]
        if t < 2:
            continue
        h_s = student(ids, return_hidden=True)
        h_d = donor(ids, return_hidden=True)
        for s in range(0, t - 1, chunk_positions):
            e = min(s + chunk_positions, t - 1)
            logp_s = torch.log_softmax(h_s[0, s:e].to(torch.float32) @ W_s.T, dim=-1)
            logp_d = torch.log_softmax(h_d[0, s:e].to(torch.float32) @ W_d.T, dim=-1)
            p_s = logp_s.exp()
            total_kl += (p_s * (logp_s - logp_d)).sum().item()
            total_n += e - s
    return total_kl / max(total_n, 1)
