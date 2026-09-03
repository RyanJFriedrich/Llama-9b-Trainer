"""8-bit AdamW (spec §0.5 v1.2) — deploy-box optimizer.

fp32 AdamW states cost 8 B/param (≈132 GB at 8.25B) and do not fit a 96 GB
card; 8-bit states cost 2 B/param (≈82.5 GB with bf16 weights + fp32 master
+ bf16 grads) and do. This is the bitsandbytes pattern, hand-rolled to avoid
the dependency (bitsandbytes is unreliable on Windows dev boxes): optimizer
states are stored block-wise quantized (int8 + one fp32 scale per block of
2048), dequantized for each update, requantized after.

One deliberate deviation from vanilla block quantization: the second moment
is stored as **sqrt(v)**, not v. m ~ E[g] and sqrt(v) ~ RMS[g] share dynamic
range, and pointwise sqrt(v_i) >= |m_i| (Jensen), so a zero-rounded sqrt(v)
implies a near-zero m at the same position and Adam's m/(sqrt(v)+eps) ratio
stays bounded. Quantizing v directly breaks that alignment (v's range is
quadratic in g's), zeroing v while m stays large — updates explode by up to
1/eps and training diverges (reproduced on a toy quadratic).

Numerical character: quantization error ~1/254 relative per block — orders
below gradient noise; convergence parity with fp32 AdamW is covered by test.
Checkpoint round-trip is exact (int8 tensors serialize bit-exactly).
"""
from __future__ import annotations

import math
from typing import Any, Optional

import torch
from torch import nn
from torch.optim import Optimizer

BLOCK = 2048


def _quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-wise dynamic int8 quantization of a flat fp32 tensor."""
    n = x.numel()
    pad = (-n) % BLOCK
    if pad:
        x = torch.cat([x, x.new_zeros(pad)])
    blocks = x.reshape(-1, BLOCK)
    scale = blocks.abs().amax(dim=1).clamp_min(1e-12) / 127.0
    q = torch.round(blocks / scale.unsqueeze(1)).to(torch.int8)
    return q, scale


def _dequantize(q: torch.Tensor, scale: torch.Tensor, n: int) -> torch.Tensor:
    return (q.to(torch.float32) * scale.unsqueeze(1)).reshape(-1)[:n]


class AdamW8bit(Optimizer):
    """AdamW with block-wise 8-bit optimizer states. Same hyperparameters and
    update math as torch.optim.AdamW (decoupled weight decay, bias
    correction); second moment stored as sqrt(v) (see module docstring).
    """

    def __init__(self, params, lr=2e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.to(torch.float32).reshape(-1)
                n = g.numel()
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["q_m"], state["s_m"] = _quantize(torch.zeros(n, device=p.device))
                    state["q_r"], state["s_r"] = _quantize(torch.zeros(n, device=p.device))
                state["step"] += 1
                t = state["step"]

                m = _dequantize(state["q_m"], state["s_m"], n)
                r = _dequantize(state["q_r"], state["s_r"], n)  # running sqrt(v)
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                # v = beta2*v + (1-beta2)*g^2, stored as r = sqrt(v).
                # Everything below is in-place on m and r: the step's fp32
                # working set is 2 buffers per param tensor (~4.2 GB for the
                # 525M embed table) instead of ~7 (~14 GB) — that transient
                # pushed the step peak to 92 GB on the 96 GB box.
                r.square_().mul_(beta2).addcmul_(g, g, value=1 - beta2).sqrt_()
                state["q_m"], state["s_m"] = _quantize(m)
                state["q_r"], state["s_r"] = _quantize(r)

                m.div_(1 - beta1 ** t)                         # m_hat
                r.div_(math.sqrt(1 - beta2 ** t)).add_(eps)    # r_hat + eps
                update = m.div_(r)  # m_hat / (r_hat + eps)
                pf = p.to(torch.float32).reshape(-1)
                if wd != 0:
                    pf.mul_(1 - lr * wd)
                pf.add_(update, alpha=-lr)
                p.copy_(pf.reshape(p.shape).to(p.dtype))
        return loss
