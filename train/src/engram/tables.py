"""Engram host-resident tables (spec §3.6, annex A1.5/A1.6) — the memory itself.

Rows live in HOST RAM as bf16 [M, row_dim] tensors, one per (order, head);
only the unique rows addressed by the current batch are staged to the device
(annex A1.6 "gather, don't densify"). The staged copy is the autograd leaf:
backward lands a dense [n_unique, row_dim] gradient on it, which the sparse
host optimizer (sparse_opt.py) applies back to the addressed host rows only.

Addressing flow per batch: tokens -> canonical ids (canon.py) ->
address_batch (addressing.py) -> np.unique per table -> staged unique rows.

Init (annex A1.5): Uniform(-0.01, 0.01) in fp32, cast to bf16, from a
DEDICATED torch.Generator per table (seed = const64(n, k, "rowinit") ^
init_seed) — the global RNG is never touched (I9: model RNG state stays
bitwise resume-safe regardless of table builds).

Unigram modulus (annex A1.4 conflict resolution, recorded): the annex pins
canonical_id = min raw id in class AND M_uni = smallest prime >= |V'|.
Min-id representatives are not dense (max canonical id approaches |V|), so
the pinned injectivity theorem (A*c+B mod M, c < M) needs
M_uni = smallest prime >= max(canonical_id) + 1. With the identity fallback
P, max = vocab_size - 1, giving smallest prime >= vocab_size — exactly the
annex's stated fallback. rows_per_head[1] in the config is an explicit
override.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np
import torch

from train.src.engram import addressing
from train.src.engram.canon import canon_sha256, identity_canon, load_canon
from train.src.config import EngramConfig
from train.utils.log import log

ROW_INIT_LOW, ROW_INIT_HIGH = -0.01, 0.01  # annex A1.5


@dataclass
class GatherBatch:
    """One batch's staged Engram rows + the bookkeeping the optimizer needs.

    rows[i]    : staged unique rows for table_keys[i], [U_i, row_dim], on
                 device; the autograd leaf when requires_grad was set.
    inverse[i]: [B*T] long on device — position -> its slot in rows[i].
    uniq[i]   : [U_i] np.int64 host — which host rows are staged.
    valid     : [B, T, n_orders] bool on device — boundary mask (addressing.py).
    shape     : (B, T).
    """

    table_keys: list[tuple[int, int]]
    rows: list[torch.Tensor]
    inverse: list[torch.Tensor]
    uniq: list[np.ndarray]
    valid: torch.Tensor
    shape: tuple[int, int]
    addressed: dict = field(default_factory=dict)  # (idx, valid) host copies

    def grads(self) -> list[Optional[np.ndarray]]:
        """Host fp32 grads per table ([U_i, row_dim]) or None if untouched."""
        out = []
        for r in self.rows:
            g = r.grad
            out.append(None if g is None else g.detach().to(torch.float32).cpu().numpy())
        return out


class EngramTables:
    """The full set of (order, head) tables + gather/stage machinery.

    Plain host object (NOT an nn.Module): nothing here lives on the device,
    and it does not appear in model.state_dict() — the trainer checkpoints it
    separately (I9 covers tables + touch counters + canon checksum).
    """

    def __init__(self, cfg: EngramConfig, vocab_size: int, init_seed: int = 0) -> None:
        self.cfg = cfg
        self.vocab_size = vocab_size

        if cfg.canonical_compression:
            self.canon = load_canon(cfg.canon_path, cfg.canon_sha256)
        else:
            self.canon = identity_canon(vocab_size)
        if len(self.canon) != vocab_size:
            raise ValueError(
                f"canon map has {len(self.canon)} entries, model vocab is {vocab_size}"
            )
        self.canon_checksum = canon_sha256(self.canon)

        # Moduli: order 1 derived (see module docstring) unless overridden;
        # orders >= 2 always come from the pinned per-head primes.
        uni_m = addressing.smallest_prime_at_least(int(self.canon.max()) + 1)
        self.table_keys: list[tuple[int, int]] = [
            (n, k) for n in cfg.orders for k in range(cfg.heads_per_order)
        ]
        self.moduli: dict[tuple[int, int], int] = {}
        for n, k in self.table_keys:
            if n == 1:
                M = cfg.rows_per_head.get(1, [None] * cfg.heads_per_order)[k] or uni_m
            else:
                M = cfg.rows_per_head[n][k]
            self.moduli[(n, k)] = int(M)

        # Rows: bf16 host tensors, one dedicated generator per table (I9).
        self.rows: dict[tuple[int, int], torch.Tensor] = {}
        for key in self.table_keys:
            M = self.moduli[key]
            gen = torch.Generator().manual_seed(
                (addressing.const64(key[0], key[1], "rowinit") ^ init_seed)
                & 0x7FFFFFFFFFFFFFFF
            )
            w = torch.rand((M, cfg.row_dim), generator=gen, dtype=torch.float32)
            w.mul_(ROW_INIT_HIGH - ROW_INIT_LOW).add_(ROW_INIT_LOW)
            self.rows[key] = w.to(torch.bfloat16)

        # Lifetime touch counters (telemetry; annex A1.8.6), uint32 per row.
        self.touch: dict[tuple[int, int], np.ndarray] = {
            key: np.zeros(self.moduli[key], dtype=np.uint32) for key in self.table_keys
        }

        total_mb = sum(r.numel() * 2 for r in self.rows.values()) / 2**20
        log(
            f"engram tables: {len(self.table_keys)} tables "
            f"({[(k, self.moduli[k]) for k in self.table_keys]}), "
            f"row_dim={cfg.row_dim}, canon_sha256={self.canon_checksum[:16]}..., "
            f"{total_mb:.1f} MiB host RAM",
            print_console=True,
        )

    # -- addressing ---------------------------------------------------------

    def address(self, tokens: Union[torch.Tensor, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """tokens [B, T] integer (host or device tensor) -> (idx, valid) as in
        addressing.address_batch, with canonical ids resolved."""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.detach().cpu().numpy()
        canon_ids = self.canon[tokens.astype(np.int64)]
        return addressing.address_batch(
            canon_ids, self.cfg.orders, self.cfg.heads_per_order, self.moduli
        )

    # -- gather / stage -----------------------------------------------------

    def stage(
        self,
        idx: np.ndarray,
        valid: np.ndarray,
        device: torch.device,
        requires_grad: bool,
    ) -> GatherBatch:
        """Host-addressed batch -> staged device tensors. idx [B,T,O,H] uint32,
        valid [B,T,O] bool. np.unique collapses repeat addresses so each host
        row is staged (and later updated) exactly once per batch."""
        B, T = idx.shape[:2]
        rows_staged: list[torch.Tensor] = []
        inverse: list[torch.Tensor] = []
        uniqs: list[np.ndarray] = []
        for ki, key in enumerate(self.table_keys):
            n, k = key
            oi = self.cfg.orders.index(n)
            flat = idx[:, :, oi, k].reshape(-1).astype(np.int64)
            uniq, inv = np.unique(flat, return_inverse=True)
            self.touch[key][uniq] += 1
            staged = self.rows[key][torch.from_numpy(uniq)].to(device)
            if requires_grad:
                staged = staged.requires_grad_(True)
            rows_staged.append(staged)
            inverse.append(torch.from_numpy(inv).to(device))
            uniqs.append(uniq)
        return GatherBatch(
            table_keys=self.table_keys,
            rows=rows_staged,
            inverse=inverse,
            uniq=uniqs,
            valid=torch.from_numpy(valid).to(device),
            shape=(B, T),
            addressed={"idx": idx, "valid": valid},
        )

    def gather(
        self,
        tokens: Union[torch.Tensor, np.ndarray],
        device: torch.device,
        requires_grad: bool,
    ) -> GatherBatch:
        """Synchronous address+stage (fallback path; the trainer normally
        splits the two across the prefetch boundary)."""
        idx, valid = self.address(tokens)
        return self.stage(idx, valid, device, requires_grad)

    # -- telemetry ----------------------------------------------------------

    def touch_histogram(self) -> dict[str, Any]:
        """Per-table touched-row counts (annex A1.8.6 telemetry)."""
        return {
            f"{n},{k}": int((c > 0).sum()) for (n, k), c in self.touch.items()
        }

    # -- checkpointing (I9) -------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "rows": {f"{n},{k}": self.rows[(n, k)] for (n, k) in self.table_keys},
            "touch": {
                f"{n},{k}": torch.from_numpy(self.touch[(n, k)]) for (n, k) in self.table_keys
            },
            "canon_sha256": self.canon_checksum,
            "moduli": {f"{n},{k}": m for (n, k), m in self.moduli.items()},
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        if sd["canon_sha256"] != self.canon_checksum:
            raise ValueError(
                "engram checkpoint canon sha256 mismatch — the canon map "
                "determines all addressing; refusing to resume onto different tables"
            )
        for key in self.table_keys:
            ks = f"{key[0]},{key[1]}"
            rows = sd["rows"][ks]
            if rows.shape != self.rows[key].shape:
                raise ValueError(
                    f"engram table {ks}: checkpoint shape {tuple(rows.shape)} != "
                    f"config shape {tuple(self.rows[key].shape)}"
                )
            self.rows[key].copy_(rows)
            self.touch[key][:] = sd["touch"][ks].numpy()
