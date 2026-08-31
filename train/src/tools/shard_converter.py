"""Converter: dense extraction intermediates -> spec §6.2 shards (M3).

The harness's ingest contract for the owner's extraction pipeline
(llama-server /completion prefill with n_probs, same machinery as the
reference Gemma project). One JSONL record per DOCUMENT:

    {
      "doc_id": "any string or int",          # informational; shard-local
                                               # uint32 ids are assigned
      "tokens": [int, ...],                    # FULL token stream, length T
      "topk_idx": [[int x k] x T],             # teacher top-k ids per position
      "topk_logprobs": [[float x k] x T],      # natural-log probs, true rank
                                               # order, full-vocab softmax
      "loss_mask": [0/1 x T]                   # optional; default all-ones
    }

Every record must use the same k (a shard is single-k by spec; use separate
shards for different k). tail_w = 1 - sum(exp(logprobs)) is computed in fp32
at conversion time (spec §6.3) — this converter is the "storage time".

Why not the old NPZ shards: they are lossy — probs renormalized to sum 1
(tail mass destroyed), GT token inserted at slot 0 scrambling id<->prob
pairing, k stored nowhere but shape, and sparse assistant-positions-only (no
token stream). The lossless source is the raw full-vocab logprobs at
extraction time, which is exactly the dense contract above. If the owner's
extractor currently stores assistant positions only, it must store all
positions — the converter needs the full stream for packing.

Usage:
    python -m train.src.tools.shard_converter intermediate.jsonl out_shard/ \
        --k 12 --teacher-id llama-3.1-8b-instruct
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Union

import numpy as np

from train.src.data.topk_writer import ShardWriter
from train.utils.log import log


def convert_jsonl(
    jsonl_path: Union[str, Path],
    shard_dir: Union[str, Path],
    k: int,
    teacher_id: str,
    quantization: str = "",
    chat_template_version: str = "",
    fold_version: str = "",
    max_docs: int | None = None,
    log_filename: str = "common.log",
) -> dict:
    """Convert a dense extraction JSONL to a spec shard. Returns the sidecar."""
    jsonl_path = Path(jsonl_path)
    writer = ShardWriter(
        shard_dir, k=k, teacher_id=teacher_id, quantization=quantization,
        chat_template_version=chat_template_version, fold_version=fold_version,
        log_filename=log_filename,
    )
    n_docs = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if max_docs is not None and n_docs >= max_docs:
                break
            rec = json.loads(line)

            tokens = np.asarray(rec["tokens"], dtype=np.uint32)
            t = tokens.shape[0]
            idx = np.asarray(rec["topk_idx"], dtype=np.uint32)
            logprobs = np.asarray(rec["topk_logprobs"], dtype=np.float32)
            if idx.shape != (t, k) or logprobs.shape != (t, k):
                raise ValueError(
                    f"{jsonl_path}:{line_no}: topk shapes {idx.shape}/{logprobs.shape} "
                    f"!= ({t}, {k}) — records must be dense (one entry per token) "
                    f"and single-k per shard"
                )
            if (logprobs > 1e-6).any():
                raise ValueError(
                    f"{jsonl_path}:{line_no}: positive logprobs — input must be "
                    "natural-log probabilities, not raw logits"
                )
            probs = np.exp(logprobs, dtype=np.float32)
            mask = rec.get("loss_mask")
            writer.add_document(
                tokens, idx, probs,
                loss_mask=np.asarray(mask, dtype=np.uint8) if mask is not None else None,
            )
            n_docs += 1

    sidecar = writer.finalize()
    log(
        f"shard_converter: {jsonl_path} -> {shard_dir}: {n_docs} docs, "
        f"{sidecar['total_tokens']} tokens, k={k}",
        print_console=True,
    )
    return sidecar


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("jsonl")
    p.add_argument("shard_dir")
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--teacher-id", required=True)
    p.add_argument("--quantization", default="")
    p.add_argument("--chat-template-version", default="")
    p.add_argument("--fold-version", default="")
    p.add_argument("--max-docs", type=int, default=None)
    args = p.parse_args()
    convert_jsonl(
        args.jsonl, args.shard_dir, k=args.k, teacher_id=args.teacher_id,
        quantization=args.quantization,
        chat_template_version=args.chat_template_version,
        fold_version=args.fold_version, max_docs=args.max_docs,
    )


if __name__ == "__main__":
    main()
