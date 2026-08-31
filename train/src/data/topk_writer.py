"""TopK distillation shard writer — spec §6.2 / §6.3 [LOCKED] storage format.

A shard is a *directory* of flat, memmap-able `.npy` arrays (written via
`numpy.save`, so `np.load(..., mmap_mode='r')` works) plus a `sidecar.json`:

    tokens.npy    uint32[T]    full token stream
    topk_idx.npy  uint32[T,k]  teacher top-k token ids (uint32 required:
                               vocab 128,256 > 65,535)
    topk_w.npy    float16[T,k] top-k probabilities, NOT renormalized —
                               true softmax mass
    tail_w.npy    float32[T]   tail mass = 1 - sum(topk_w), computed in FP32
                               AT STORAGE TIME from the fp32 probabilities;
                               never recomputed from the fp16 weights at load
                               (spec §6.3: fp16 eps near 1.0 ~ 1e-3 would make
                               the residual rounding-dominated)
    doc_id.npy    uint32[T]    shard-local document id per position
    loss_mask.npy uint8[T]     1 where KD/CE loss applies, 0 elsewhere
                               (document boundaries are handled by the loader;
                               this array is for padding / non-loss spans)
    sidecar.json  k, teacher id, quantization, chat-template version,
                  fold version, per-document offsets, total tokens T

Usage:

    writer = ShardWriter("path/to/shard_dir", k=10, teacher_id="llama-3.1-8b-instruct")
    writer.add_document(tokens, topk_probs)           # topk_probs fp32 [len, k]
    writer.add_document(tokens2, topk_probs2, loss_mask=mask2)
    sidecar = writer.finalize()

Documents are appended in call order and get consecutive shard-local doc ids
0, 1, 2, ...  Buffered arrays are concatenated at `finalize()`, so peak RAM is
proportional to shard size — size shards accordingly (~64 B/token at k=10,
spec §6.2). No pickles, no .npz, no full-vocab anything.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np

from train.utils.log import log

FORMAT_VERSION = 1

# Array file stems and their on-disk dtypes, in sidecar/loader contract order.
ARRAY_DTYPES: dict[str, np.dtype] = {
    "tokens": np.dtype(np.uint32),
    "topk_idx": np.dtype(np.uint32),
    "topk_w": np.dtype(np.float16),
    "tail_w": np.dtype(np.float32),
    "doc_id": np.dtype(np.uint32),
    "loss_mask": np.dtype(np.uint8),
}

SIDECAR_NAME = "sidecar.json"

# fp32 probability validation tolerances.
_PROB_TOL = 1e-6
_MASS_TOL = 1e-5


class ShardWriter:
    """Accumulates teacher top-k records per document and writes one shard."""

    def __init__(
        self,
        shard_dir: Union[str, Path],
        k: int = 10,
        teacher_id: str = "",
        quantization: str = "",
        chat_template_version: str = "",
        fold_version: str = "",
        vocab_size: int = 128256,
        log_filename: str = "common.log",
        text_source: str = "",
        logit_source: str = "",
        data_class: str = "",
        alpha_override: Optional[float] = None,
    ) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        self.shard_dir = Path(shard_dir)
        self.k = int(k)
        self.teacher_id = teacher_id
        self.quantization = quantization
        self.chat_template_version = chat_template_version
        self.fold_version = fold_version
        self.vocab_size = int(vocab_size)
        self._log_filename = log_filename
        # spec §6.1 sidecar metadata (v2.0): provenance + per-slice loss mix.
        self.text_source = text_source
        self.logit_source = logit_source
        self.data_class = data_class
        self.alpha_override = alpha_override

        self._tokens: list[np.ndarray] = []
        self._topk_idx: list[np.ndarray] = []
        self._topk_w: list[np.ndarray] = []
        self._tail_w: list[np.ndarray] = []
        self._loss_mask: list[np.ndarray] = []
        self._doc_lengths: list[int] = []
        self._finalized = False

    def add_document(
        self,
        tokens: np.ndarray,
        topk_ids: np.ndarray,
        topk_probs: np.ndarray,
        loss_mask: Optional[np.ndarray] = None,
    ) -> int:
        """Append one document. Returns its shard-local doc id.

        Args:
            tokens: [T_doc] token ids (any integer dtype; cast to uint32).
            topk_ids: [T_doc, k] teacher top-k token ids.
            topk_probs: fp32 [T_doc, k] teacher top-k probabilities aligned
                with `topk_ids` — true softmax mass, NOT renormalized.
            loss_mask: optional [T_doc] 0/1; defaults to all-ones.
        """
        if self._finalized:
            raise RuntimeError("add_document() after finalize()")

        tokens = np.ascontiguousarray(tokens)
        probs = np.ascontiguousarray(topk_probs, dtype=np.float32)
        ids = np.ascontiguousarray(topk_ids)

        t = tokens.shape[0]
        if t == 0:
            raise ValueError("empty document")
        if probs.shape != (t, self.k):
            raise ValueError(f"topk_probs shape {probs.shape} != ({t}, {self.k})")
        if ids.shape != (t, self.k):
            raise ValueError(f"topk_ids shape {ids.shape} != ({t}, {self.k})")

        if tokens.min() < 0 or tokens.max() >= self.vocab_size:
            raise ValueError("token id out of vocab range")
        if ids.min() < 0 or ids.max() >= self.vocab_size:
            raise ValueError("top-k token id out of vocab range")
        if (probs < -_PROB_TOL).any() or (probs > 1.0 + _PROB_TOL).any():
            raise ValueError("top-k probability outside [0, 1]")

        # spec §6.3: tail mass in fp32, at storage time, from fp32 probs.
        tail = 1.0 - probs.sum(axis=1, dtype=np.float32)
        if (tail < -_MASS_TOL).any():
            raise ValueError(
                f"top-k probabilities sum above 1 (min tail {tail.min()}); "
                "pass true softmax mass, not renormalized weights"
            )
        tail = np.clip(tail, 0.0, 1.0).astype(np.float32)

        if loss_mask is None:
            mask = np.ones(t, dtype=np.uint8)
        else:
            mask = np.ascontiguousarray(loss_mask)
            if mask.shape != (t,):
                raise ValueError(f"loss_mask shape {mask.shape} != ({t},)")
            if not np.isin(mask, (0, 1)).all():
                raise ValueError("loss_mask must be 0/1")
            mask = mask.astype(np.uint8)

        doc_id = len(self._doc_lengths)
        if doc_id > np.iinfo(np.uint32).max:
            raise ValueError("too many documents for uint32 doc_id")

        self._tokens.append(tokens.astype(np.uint32))
        self._topk_idx.append(ids.astype(np.uint32))
        self._topk_w.append(probs.astype(np.float16))
        self._tail_w.append(tail)
        self._loss_mask.append(mask)
        self._doc_lengths.append(int(t))
        return doc_id

    def finalize(self) -> dict:
        """Write all arrays + sidecar.json; returns the sidecar dict."""
        if self._finalized:
            raise RuntimeError("finalize() called twice")
        if not self._doc_lengths:
            raise ValueError("no documents added")
        self._finalized = True

        self.shard_dir.mkdir(parents=True, exist_ok=True)
        tokens = np.concatenate(self._tokens)
        topk_idx = np.concatenate(self._topk_idx)
        topk_w = np.concatenate(self._topk_w)
        tail_w = np.concatenate(self._tail_w)
        loss_mask = np.concatenate(self._loss_mask)
        total = int(tokens.shape[0])

        doc_offsets = np.zeros(len(self._doc_lengths), dtype=np.int64)
        np.cumsum(self._doc_lengths[:-1], out=doc_offsets[1:])
        doc_id = np.repeat(
            np.arange(len(self._doc_lengths), dtype=np.uint32), self._doc_lengths
        )

        arrays = {
            "tokens": tokens,
            "topk_idx": topk_idx,
            "topk_w": topk_w,
            "tail_w": tail_w,
            "doc_id": doc_id,
            "loss_mask": loss_mask,
        }
        for name, arr in arrays.items():
            assert arr.dtype == ARRAY_DTYPES[name], (name, arr.dtype)
            np.save(self.shard_dir / f"{name}.npy", arr)

        sidecar = {
            "format_version": FORMAT_VERSION,
            "k": self.k,
            "teacher_id": self.teacher_id,
            "quantization": self.quantization,
            "chat_template_version": self.chat_template_version,
            "fold_version": self.fold_version,
            "text_source": self.text_source,
            "logit_source": self.logit_source,
            "data_class": self.data_class,
            "alpha_override": self.alpha_override,
            "vocab_size": self.vocab_size,
            "total_tokens": total,
            "num_documents": len(self._doc_lengths),
            "doc_offsets": doc_offsets.tolist(),
            "doc_lengths": list(self._doc_lengths),
        }
        (self.shard_dir / SIDECAR_NAME).write_text(
            json.dumps(sidecar, indent=2) + "\n", encoding="utf-8"
        )
        log(
            f"ShardWriter: wrote {self.shard_dir} — {total} tokens, "
            f"{len(self._doc_lengths)} docs, k={self.k}, teacher={self.teacher_id}",
            filename=self._log_filename,
        )
        return sidecar
