"""Fused/chunked lumped-tail KD loss (spec §6.4) — LOCKED hard requirement.

    L_KD = -sum_i w_i * log p_i  -  tail_w * log(1 - sum_i p_i)
    L    = alpha * L_KD + (1 - alpha) * L_CE(gold)

student log-probs are gathered at the stored top-k indices plus a logsumexp
term. The full [T, vocab] logit tensor is NEVER materialized: hidden states
are multiplied by lm_head in chunks, each chunk's logits are freed after use,
and the backward pass recomputes them chunk-by-chunk (cut-cross-entropy
pattern). Standing rule 3: this is the only loss path for KD training.

Gradients (fp32, per chunk, letting p = softmax(z), z = h @ W.T / T_temp):
  top-k term:  dz_v = p_v * W_sum - w_v        (w_v = 0 for v not in top-k)
  tail term:   dz_v = tail * p_v               for v in top-k
               dz_v = -tail * p_v * (1-S)/S    for v not in top-k, S = tail mass
  CE term:     dz_v = p_v - 1[v == gold]
combined as alpha * (kd) + (1 - alpha) * (ce), masked, then dh = dz @ W and
dW += dz.T @ h.
"""
from __future__ import annotations

from typing import Optional

import torch

# Tail mass below this is treated as exact for log stability: with fp32
# softmax the student's true tail can underflow; clamping bounds the tail
# term's contribution at ~27 nats instead of inf/nan.
_TAIL_FLOOR = 1e-12


def _chunk_losses(
    z: torch.Tensor,  # [c, V] fp32, logits / temperature
    topk_idx: torch.Tensor,  # [c, k] long
    topk_w: torch.Tensor,  # [c, k] fp32
    tail_w: torch.Tensor,  # [c] fp32
    gold: torch.Tensor,  # [c] long
    alpha: float,
) -> torch.Tensor:
    """Per-position combined loss for one chunk. Returns [c] fp32."""
    lse = torch.logsumexp(z, dim=-1, keepdim=True)  # [c, 1]
    logp_topk = z.gather(1, topk_idx) - lse  # [c, k]
    p_topk = logp_topk.exp()
    student_tail = (1.0 - p_topk.sum(dim=-1)).clamp_min(_TAIL_FLOOR)

    l_kd = -(topk_w * logp_topk).sum(dim=-1) - tail_w * student_tail.log()
    l_ce = -(z.gather(1, gold.unsqueeze(1)).squeeze(1) - lse.squeeze(1))
    return alpha * l_kd + (1.0 - alpha) * l_ce


class _FusedKDLoss(torch.autograd.Function):
    """Sum of masked per-position losses; backward recomputes chunk logits."""

    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,  # [N, D] any float dtype
        weight: torch.Tensor,  # [V, D] lm_head
        topk_idx: torch.Tensor,  # [N, k] long
        topk_w: torch.Tensor,  # [N, k]
        tail_w: torch.Tensor,  # [N]
        gold: torch.Tensor,  # [N] long
        mask: torch.Tensor,  # [N] float/bool
        alpha: float,
        temperature: float,
        chunk_size: int,
    ) -> torch.Tensor:
        N, D = hidden.shape
        ctx.alpha, ctx.temperature, ctx.chunk_size = alpha, temperature, chunk_size
        ctx.save_for_backward(hidden, weight, topk_idx, topk_w, tail_w, gold, mask)

        total = torch.zeros((), dtype=torch.float32, device=hidden.device)
        with torch.no_grad():
            for s in range(0, N, chunk_size):
                e = min(s + chunk_size, N)
                m = mask[s:e].to(torch.float32)
                if not m.any():
                    continue
                z = (hidden[s:e].to(torch.float32) @ weight.to(torch.float32).T) / temperature
                losses = _chunk_losses(z, topk_idx[s:e], topk_w[s:e].to(torch.float32),
                                       tail_w[s:e].to(torch.float32), gold[s:e], alpha)
                total += (losses * m).sum()
        return total

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        hidden, weight, topk_idx, topk_w, tail_w, gold, mask = ctx.saved_tensors
        alpha, temperature, chunk_size = ctx.alpha, ctx.temperature, ctx.chunk_size
        N, D = hidden.shape
        V = weight.shape[0]

        grad_h = torch.zeros_like(hidden)
        grad_w = torch.zeros_like(weight, dtype=torch.float32)
        for s in range(0, N, chunk_size):
            e = min(s + chunk_size, N)
            m = mask[s:e].to(torch.float32)
            if not m.any():
                continue
            h_c = hidden[s:e].to(torch.float32)
            z = (h_c @ weight.to(torch.float32).T) / temperature
            p = torch.softmax(z, dim=-1)
            k = topk_idx.shape[1]

            # KD top-k term: p_v * W_sum - w_v
            w_sum = topk_w[s:e].to(torch.float32).sum(dim=-1, keepdim=True)  # [c,1]
            dz = p * w_sum
            dz.scatter_add_(1, topk_idx[s:e], -topk_w[s:e].to(torch.float32))

            # KD tail term: +tail*p in top-k, -tail*p*(1-S)/S outside.
            p_topk_sum = p.gather(1, topk_idx[s:e]).sum(dim=-1, keepdim=True)
            S = (1.0 - p_topk_sum).clamp_min(_TAIL_FLOOR)  # [c,1]
            tail = tail_w[s:e].to(torch.float32).unsqueeze(1)
            tail_grad = -tail * p * (1.0 - S) / S  # outside top-k value
            tail_grad.scatter_(1, topk_idx[s:e], (tail * p).gather(1, topk_idx[s:e]))
            dz += tail_grad

            # CE term.
            if alpha < 1.0:
                ce = p.clone()
                ce.scatter_add_(1, gold[s:e].unsqueeze(1),
                                -torch.ones(e - s, 1, dtype=torch.float32, device=p.device))
                dz = alpha * dz + (1.0 - alpha) * ce
            else:
                dz = alpha * dz

            dz = (dz * m.unsqueeze(1)) * (grad_out.to(torch.float32) / temperature)
            grad_h[s:e] = (dz @ weight.to(torch.float32)).to(hidden.dtype)
            grad_w += dz.T @ h_c

        return grad_h, grad_w.to(weight.dtype), None, None, None, None, None, None, None, None


def kd_loss(
    hidden: torch.Tensor,  # [B, T, D] final hidden states (pre-lm_head)
    lm_head_weight: torch.Tensor,  # [V, D]
    topk_idx: torch.Tensor,  # [B, T, k]
    topk_w: torch.Tensor,  # [B, T, k] teacher top-k probs (true mass, not renormalized)
    tail_w: torch.Tensor,  # [B, T] teacher tail mass, fp32, from the shard
    gold: torch.Tensor,  # [B, T] gold target tokens (for the CE term)
    loss_mask: torch.Tensor,  # [B, T] 1 where loss applies (spec §6.4: never across doc boundaries/padding)
    alpha: float = 1.0,  # Phase 0: pure KD (alpha=1, donor teacher); Phase 1 default 0.9
    temperature: float = 1.0,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Mean masked lumped-tail KD + CE loss. Never materializes [B*T, V]."""
    B, T, D = hidden.shape
    flat = lambda t: t.reshape(B * T, *t.shape[2:]) if t.dim() > 2 else t.reshape(B * T)
    mask_f = loss_mask.reshape(B * T).to(torch.float32)
    denom = mask_f.sum().clamp_min(1.0)
    total = _FusedKDLoss.apply(
        hidden.reshape(B * T, D), lm_head_weight,
        flat(topk_idx).long(), flat(topk_w), flat(tail_w),
        flat(gold).long(), mask_f, alpha, temperature, chunk_size,
    )
    return total / denom
