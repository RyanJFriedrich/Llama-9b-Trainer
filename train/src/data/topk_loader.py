"""Streaming mmap loader for TopK distillation shards — spec §6.2 / §6.3.

`TopKShard` opens one shard directory: `sidecar.json` for metadata, every
array via `np.load(..., mmap_mode='r')` — no whole-shard reads, ever. `k` is
read from the sidecar and validated against the `topk_idx` array shape (not
inferred from the shape alone), so shards with different k work unchanged.

`TopKLoader.iter_sequences()` packs each shard's token stream independently
into non-overlapping windows of `seq_len` and yields dicts of torch tensors:

    tokens    int64[seq_len]     input tokens
    topk_idx  int64[seq_len, k]  teacher top-k token ids (uint32 -> int64)
    topk_w    fp32[seq_len, k]   top-k probs (fp16 storage, upcast on load)
    tail_w    fp32[seq_len]      tail mass, consumed as stored (spec §6.3:
                                 never recomputed from the fp16 weights)
    doc_id    int64[seq_len]
    loss_mask bool[seq_len]

Packing / loss-mask policy (spec §6.4: never compute loss across document
boundaries or padding):
  - Window position j predicts stream token s+j+1 (standard shifted labels).
  - Position j is masked OFF when its target belongs to a different document
    than the position itself (doc_id[s+j+1] != doc_id[s+j]) — i.e. the first
    target position of each new doc inside the window is masked.
  - The final position of every window is masked OFF (its target lies outside
    the window, so the shifted label does not exist).
  - The shard's stored `loss_mask` is honored on top of both rules.
  - A shard's trailing tokens that do not fill a complete window are dropped
    (documented; callers size shards so the remainder is negligible).

Iteration is per shard in the given order (windows of one shard never mix k),
with optional seeded shuffle of the per-shard window order; same seed gives
the same order.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Union

import numpy as np
import torch

from train.src.data.topk_writer import ARRAY_DTYPES, SIDECAR_NAME
from train.utils.log import log


class TopKShard:
    """mmap view of one shard directory. All arrays stay on disk."""

    def __init__(self, shard_dir: Union[str, Path]) -> None:
        self.dir = Path(shard_dir)
        sidecar_path = self.dir / SIDECAR_NAME
        if not sidecar_path.exists():
            raise FileNotFoundError(f"no sidecar at {sidecar_path}")
        self.sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))

        self.k = int(self.sidecar["k"])
        if self.k < 1:
            raise ValueError(f"sidecar k must be >= 1, got {self.k}")
        self.total_tokens = int(self.sidecar["total_tokens"])
        self.teacher_id = self.sidecar.get("teacher_id", "")
        # spec §6.3 (v1.2): v1 = folded legacy shards (tail mass folded into
        # the gold token), v2 = unfolded + stored tail_w. Default v2.
        self.fold_version = self.sidecar.get("fold_version", "v2")

        for name, dtype in ARRAY_DTYPES.items():
            arr = np.load(self.dir / f"{name}.npy", mmap_mode="r")
            if not isinstance(arr, np.memmap):
                raise RuntimeError(f"{name}.npy did not open as memmap")
            if arr.dtype != dtype:
                raise ValueError(f"{name} dtype {arr.dtype} != expected {dtype}")
            expected_shape = (
                (self.total_tokens, self.k) if name in ("topk_idx", "topk_w")
                else (self.total_tokens,)
            )
            if arr.shape != expected_shape:
                raise ValueError(
                    f"{name} shape {arr.shape} != expected {expected_shape} "
                    f"(sidecar total_tokens={self.total_tokens}, k={self.k})"
                )
            setattr(self, name, arr)

    def __len__(self) -> int:
        return self.total_tokens

    def __repr__(self) -> str:
        return (
            f"TopKShard({self.dir}, k={self.k}, tokens={self.total_tokens}, "
            f"teacher={self.teacher_id!r})"
        )


class TopKLoader:
    """Packs shards into fixed-length sequences and streams them as tensors."""

    def __init__(
        self,
        shard_dirs: list[Union[str, Path]],
        seq_len: int,
        shuffle: bool = False,
        seed: int = 0,
        log_filename: str = "common.log",
    ) -> None:
        if seq_len < 2:
            raise ValueError(f"seq_len must be >= 2, got {seq_len}")
        self.seq_len = int(seq_len)
        self.shuffle = shuffle
        self.seed = int(seed)
        self.shards = [TopKShard(d) for d in shard_dirs]
        self._log_filename = log_filename

    def num_sequences(self) -> int:
        return sum(len(s) // self.seq_len for s in self.shards)

    def iter_sequences(self) -> Iterator[dict[str, torch.Tensor]]:
        rng = np.random.default_rng(self.seed)
        total_yielded = 0
        for shard in self.shards:
            n_win = len(shard) // self.seq_len
            order = np.arange(n_win)
            if self.shuffle:
                rng.shuffle(order)
            for w in order:
                yield self._window(shard, int(w))
                total_yielded += 1
        log(
            f"TopKLoader: streamed {total_yielded} sequences "
            f"(seq_len={self.seq_len}, {len(self.shards)} shards, "
            f"shuffle={self.shuffle}, seed={self.seed})",
            filename=self._log_filename,
        )

    def _window(self, shard: TopKShard, w: int) -> dict[str, torch.Tensor]:
        s, L = w * self.seq_len, self.seq_len
        sl = slice(s, s + L)

        # np.array copies: window slices are small, and torch.from_numpy on a
        # read-only memmap view yields non-writable tensors.
        tokens = np.array(shard.tokens[sl])
        topk_idx = np.array(shard.topk_idx[sl])
        topk_w = np.array(shard.topk_w[sl], dtype=np.float32)  # fp16 -> fp32
        tail_w = np.array(shard.tail_w[sl])
        doc_id = np.array(shard.doc_id[sl])
        stored_mask = np.array(shard.loss_mask[sl], dtype=bool)

        # Mask off any position whose TARGET (stream position s+j+1) belongs
        # to a different document, plus the final position (no target in the
        # window). doc_id is monotonically non-decreasing within a shard, so
        # doc_id[j+1] != doc_id[j] is exactly a document boundary.
        boundary = np.zeros(L, dtype=bool)
        boundary[:-1] = doc_id[1:] != doc_id[:-1]
        boundary[-1] = True
        loss_mask = stored_mask & ~boundary

        return {
            "tokens": torch.from_numpy(tokens.astype(np.int64)),
            "topk_idx": torch.from_numpy(topk_idx.astype(np.int64)),
            "topk_w": torch.from_numpy(topk_w),
            "tail_w": torch.from_numpy(tail_w),
            "doc_id": torch.from_numpy(doc_id.astype(np.int64)),
            "loss_mask": torch.from_numpy(loss_mask),
        }


def open_shards(shard_dirs: list[Union[str, Path]]) -> list[TopKShard]:
    """Convenience: open and validate a list of shard directories."""
    return [TopKShard(d) for d in shard_dirs]
