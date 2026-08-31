"""Donor scoring: produce top-k KD anchor shards (spec §6: the 8B donor is
the logit anchor for bulk text — text_source != logit_source is deliberate).

Prefill-only top-k extraction from the donor model. Feeds token-id documents
through the model once, takes per-position top-k log-probs from full-vocab
softmax, and writes spec §6.1 shards via ShardWriter (tail_w in fp32 at
storage time).

Tokenization lives in the caller (scripts/score_donor.py uses the HF
tokenizer from the donor checkpoint); this module is deliberately
tokenizer-free so tests can feed raw id sequences.

Logits are computed in chunks over positions and discarded immediately —
full [T, V] logits never persist (same memory rule as the loss path).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch

from train.src.data.topk_writer import ShardWriter
from train.utils.log import log


@torch.no_grad()
def score_documents(
    model,
    docs: Sequence[Sequence[int]],
    shard_dir: Union[str, Path],
    k: int = 10,
    teacher_id: str = "llama-3.1-8b-instruct",
    quantization: str = "none",
    device: str = "cuda",
    chunk_positions: int = 256,
    max_len: Optional[int] = None,
    log_filename: str = "common.log",
) -> dict:
    """Score documents with the model and write one spec shard.

    Position i's top-k describes the prediction of token i+1 from context
    <= i (standard teacher-forced prefill). The final position of each
    document has no target and gets loss_mask=0.
    """
    model = model.to(device).eval()
    writer = ShardWriter(shard_dir, k=k, teacher_id=teacher_id,
                         quantization=quantization, fold_version="v2",
                         log_filename=log_filename)

    for doc_i, doc in enumerate(docs):
        ids = torch.tensor(list(doc), dtype=torch.long, device=device).unsqueeze(0)
        if max_len is not None:
            ids = ids[:, :max_len]
        t = ids.shape[1]
        if t < 2:
            continue
        # Hidden states only — lm_head is applied chunk-by-chunk so the full
        # [T, V] logit tensor is never materialized (same rule as the loss).
        hidden = model(ids, return_hidden=True)  # [1, T, D]
        W = model.lm_head.weight.to(torch.float32)
        topk_idx = np.empty((t, k), dtype=np.uint32)
        topk_probs = np.empty((t, k), dtype=np.float32)
        for s in range(0, t, chunk_positions):
            e = min(s + chunk_positions, t)
            z = hidden[0, s:e].to(torch.float32) @ W.T
            lp = torch.log_softmax(z, dim=-1)
            vals, idxs = torch.topk(lp, k, dim=-1)
            topk_idx[s:e] = idxs.cpu().numpy().astype(np.uint32)
            topk_probs[s:e] = vals.exp().cpu().numpy().astype(np.float32)
        del hidden

        mask = np.ones(t, dtype=np.uint8)
        mask[-1] = 0  # no target beyond the document's final token
        writer.add_document(ids[0].cpu().numpy().astype(np.uint32),
                            topk_idx, topk_probs, loss_mask=mask)

    sidecar = writer.finalize()
    log(f"score_donor: {len(docs)} docs -> {shard_dir} "
        f"({sidecar['total_tokens']} tokens, k={k})",
        filename=log_filename, print_console=True)
    return sidecar
