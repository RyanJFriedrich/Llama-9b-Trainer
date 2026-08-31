"""Sparse host-side 8-bit AdamW for the Engram tables (annex A1.7).

The tables are host-resident and sparsely touched: a batch addresses only the
unique rows in its GatherBatch, so this optimizer updates exactly those rows
and leaves everything else bitwise untouched (unlike a dense optimizer pass
over ~10M rows that would also requantize untouched state every step).

States (m, sqrt(v) — same sqrt trick as optim8bit.py, see its docstring) are
dense host tensors, block-wise int8-quantized with one fp32 scale per
2048-element block. A sparse row update dequantizes only the AFFECTED blocks
(row_dim divides BLOCK, so a row never straddles a block boundary), applies
the Adam update to the touched rows' elements fp32, and requantizes those
blocks. Untouched rows inside a requantized block see the usual ~1/254
relative requantization noise — the same order the dense optimizer accepts
on every element every step.

Pinned choices (annex A1.7 + v1.1 cadence rule): row LR = lr_mult x base LR
(the TRAINER multiplies; this class takes the final lr), weight decay 0,
betas/eps as the main optimizer. Grads accumulate host-side in fp32 across
the gradient-accumulation window (accumulate()) and the Adam step runs once
per dense-equivalent step (step()) — the x lr_mult multiplier is per dense
step, never compounded by accum. Adam bias correction uses a per-table
global step counter incremented per dense step that touches the table
(recorded choice — the annex does not pin step counting).

The pending accumulation buffer is transient within a step and always empty
at step boundaries, which is where checkpoints happen — state_dict() carries
the optimizer moments only.

Rows whose staged gradient is exactly zero (e.g. addressed only at
boundary-invalid positions) are skipped: no state decay, no drift.

Telemetry: bf16 rounding-loss fraction — of the updated elements, how many
round back to the same bf16 value despite a nonzero update (sub-bf16-ulp
learning signal). Retrieved via pop_telemetry().
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from train.src.engram.tables import EngramTables, GatherBatch
from train.src.train.optim8bit import BLOCK


def _quant_block(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[nb, BLOCK] fp32 -> (int8 [nb, BLOCK], scale [nb]). Mirrors
    optim8bit._quantize for already block-shaped tensors."""
    scale = x.abs().amax(dim=1).clamp_min(1e-12) / 127.0
    return torch.round(x / scale.unsqueeze(1)).to(torch.int8), scale


class SparseRowAdamW8bit:
    def __init__(
        self,
        tables: EngramTables,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
    ) -> None:
        if tables.rows[tables.table_keys[0]].shape[1] > BLOCK or BLOCK % tables.cfg.row_dim:
            raise ValueError(f"row_dim must divide {BLOCK}")
        self.tables = tables
        self.betas = betas
        self.eps = eps
        self.state: dict[tuple[int, int], dict[str, Any]] = {}
        # Pending accumulated grads per table: list of (uniq int64, grad fp32)
        # fragments, one per backward'd micro-batch (annex v1.1 cadence rule).
        self._pending: dict[tuple[int, int], list[tuple[np.ndarray, torch.Tensor]]] = {}
        self._tel_rounded = 0
        self._tel_updated = 0

    def _state_for(self, key: tuple[int, int]) -> dict[str, Any]:
        st = self.state.get(key)
        if st is None:
            n = self.tables.moduli[key] * self.tables.cfg.row_dim
            nblocks = (n + BLOCK - 1) // BLOCK
            st = {
                "step": 0,
                "q_m": torch.zeros((nblocks, BLOCK), dtype=torch.int8),
                "s_m": torch.full((nblocks,), 1e-12 / 127.0),
                "q_r": torch.zeros((nblocks, BLOCK), dtype=torch.int8),
                "s_r": torch.full((nblocks,), 1e-12 / 127.0),
            }
            self.state[key] = st
        return st

    @torch.no_grad()
    def accumulate(self, gb: GatherBatch) -> None:
        """Buffer one backward'd micro-batch's staged grads host-side (fp32).

        Annex v1.1 cadence rule: table grads accumulate across the gradient-
        accumulation window; the Adam step happens once per dense-equivalent
        step (see step()). A row touched by several micro-batches gets ONE
        update with the SUMMED grad — the lr x lr_mult multiplier is defined
        per dense step, never compounded by accum."""
        for i, key in enumerate(gb.table_keys):
            g = gb.rows[i].grad
            if g is None:
                continue
            self._pending.setdefault(key, []).append(
                (gb.uniq[i], g.to(torch.float32).cpu())
            )

    @torch.no_grad()
    def step(self, lr: float) -> None:
        """Apply the sparse Adam update over the accumulated window. No-op
        when nothing accumulated (e.g. an all-masked batch)."""
        if not self._pending:
            return
        beta1, beta2 = self.betas
        D = self.tables.cfg.row_dim
        for key, frags in self._pending.items():
            # Collapse duplicates across the window: one grad per host row.
            all_uniq = np.concatenate([u for u, _ in frags])
            all_g = torch.cat([g for _, g in frags], dim=0)
            uniq, inv = np.unique(all_uniq, return_inverse=True)
            g_all = torch.zeros((len(uniq), D), dtype=torch.float32)
            g_all.index_add_(0, torch.from_numpy(inv), all_g)
            nz = g_all.abs().sum(dim=1) > 0  # skip zero-grad rows (no drift)
            if not bool(nz.any()):
                continue
            uniq = uniq[nz.numpy()]  # host row indices, int64
            g = g_all[nz]  # [Unz, D] fp32

            st = self._state_for(key)
            st["step"] += 1
            t = st["step"]

            rows_t = torch.from_numpy(uniq)
            old_bf16 = self.tables.rows[key][rows_t]
            old = old_bf16.to(torch.float32)

            # Affected blocks only: rows are block-aligned (row_dim | BLOCK).
            starts = uniq * D
            offs = torch.from_numpy((starts % BLOCK)[:, None] + np.arange(D))  # [Unz, D]
            ub, inv = np.unique(starts // BLOCK, return_inverse=True)
            ub_t = torch.from_numpy(ub)
            inv_t = torch.from_numpy(inv)

            m = st["q_m"][ub_t].to(torch.float32) * st["s_m"][ub_t].unsqueeze(1)
            r = st["q_r"][ub_t].to(torch.float32) * st["s_r"][ub_t].unsqueeze(1)
            m_e = m[inv_t.unsqueeze(1), offs]  # [Unz, D] states at touched rows
            r_e = r[inv_t.unsqueeze(1), offs]

            m_e = m_e * beta1 + g * (1 - beta1)
            v_e = r_e.square() * beta2 + g.square() * (1 - beta2)
            r_e = v_e.sqrt()

            m_hat = m_e / (1 - beta1**t)
            r_hat = r_e / math.sqrt(1 - beta2**t)
            new = old - lr * m_hat / (r_hat + self.eps)  # WD = 0 (annex A1.7)

            m[inv_t.unsqueeze(1), offs] = m_e
            r[inv_t.unsqueeze(1), offs] = r_e
            q_m, s_m = _quant_block(m)
            q_r, s_r = _quant_block(r)
            st["q_m"][ub_t] = q_m
            st["s_m"][ub_t] = s_m
            st["q_r"][ub_t] = q_r
            st["s_r"][ub_t] = s_r

            new_bf16 = new.to(torch.bfloat16)
            self.tables.rows[key][rows_t] = new_bf16
            self._tel_rounded += int((new_bf16 == old_bf16).sum())
            self._tel_updated += new_bf16.numel()
        self._pending = {}

    def pop_telemetry(self) -> dict[str, float]:
        """bf16 rounding-loss fraction since the last call (annex A1.8.6)."""
        frac = self._tel_rounded / self._tel_updated if self._tel_updated else 0.0
        out = {"engram_bf16_rounding_loss": frac}
        self._tel_rounded = 0
        self._tel_updated = 0
        return out

    def state_dict(self) -> dict[str, Any]:
        return {
            f"{n},{k}": st for (n, k), st in self.state.items()
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        for ks, st in sd.items():
            n, k = (int(x) for x in ks.split(","))
            self.state[(n, k)] = st
