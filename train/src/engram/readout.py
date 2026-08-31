"""Engram readout (spec §3.6, annex A1.5) — device-side projection.

Per order n, the readout is

    delta_n = g_n * U_n( RMSNorm( concat_k rows[n,k] ) )      [B, T, d_model]

summed over orders, with boundary-invalid positions masked to zero
(addressing.py `valid`). U is ZERO-INIT (invariant I1: the model at init is
bitwise identical to the no-Engram model — garbage or real rows, the
injection is exactly 0). The per-order scalar gate g inits to 1.0 and sits
in the no-weight-decay optimizer group (annex A1.7).

This module is an ordinary nn.Module living on the device with the rest of
the model (fp32 masters, bf16 autocast compute); only the TABLES are
host-resident (tables.py).
"""
from __future__ import annotations

import torch
from torch import nn

from train.src.config import EngramConfig
from train.src.engram.tables import GatherBatch
from train.src.model.decoder import LlamaRMSNorm


class EngramReadout(nn.Module):
    def __init__(self, cfg: EngramConfig, d_model: int, rms_eps: float) -> None:
        super().__init__()
        self.cfg = cfg
        self.d_model = d_model
        in_dim = cfg.heads_per_order * cfg.row_dim
        self.orders = list(cfg.orders)
        self.norms = nn.ModuleDict(
            {str(n): LlamaRMSNorm(in_dim, rms_eps) for n in self.orders}
        )
        self.proj = nn.ModuleDict(
            {str(n): nn.Linear(in_dim, d_model, bias=False) for n in self.orders}
        )
        self.gates = nn.ParameterDict(
            {str(n): nn.Parameter(torch.ones(())) for n in self.orders}
        )
        for n in self.orders:
            nn.init.zeros_(self.proj[str(n)].weight)  # I1: zero-init U

    def forward(self, gb: GatherBatch) -> torch.Tensor:
        """GatherBatch -> injection delta [B, T, d_model]."""
        B, T = gb.shape
        out: torch.Tensor | None = None
        for oi, n in enumerate(self.orders):
            pieces = []
            for ki, key in enumerate(gb.table_keys):
                if key[0] != n:
                    continue
                # position -> its staged unique row: [B*T, row_dim]
                pieces.append(gb.rows[ki][gb.inverse[ki]])
            x = torch.cat(pieces, dim=-1)  # [B*T, heads*row_dim]
            x = self.norms[str(n)](x)
            x = self.proj[str(n)](x) * self.gates[str(n)]
            x = x.view(B, T, self.d_model)
            x = x * gb.valid[:, :, oi].unsqueeze(-1).to(x.dtype)
            out = x if out is None else out + x
        assert out is not None
        return out
