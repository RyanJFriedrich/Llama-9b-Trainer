"""Pluggable scoring backends for Stage 4.

- `llama_server`: LlamaCPPBinaries/llama-server.exe with a Q8 GGUF.
- `pytorch`: HuggingFace-format donor checkpoint via the training repo's
  LlamaBaseModel (fallback if the server binary lacks prompt-logprob support).

Both backends produce (record, arrays) tuples where arrays is
(tokens, topk_idx, topk_probs, loss_mask) suitable for the spec ShardWriter.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import requests
import torch

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from train.src.tools.load_donor import load_donor


def _extract_topk_from_server(
    port: int,
    token_ids: list[int],
    k: int,
    doc_id: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Call llama-server /completion and return (topk_idx [T-1,k], topk_logprobs [T-1,k])."""
    n = len(token_ids)
    if n < 2:
        return None

    url = f"http://127.0.0.1:{port}/completion"
    payload = {
        "prompt": token_ids,
        "n_predict": 0,
        "n_probs": k,
        "prompt_logprobs": True,
        "stream": False,
        "cache_prompt": False,
    }
    response = requests.post(url, json=payload, timeout=300)
    response.raise_for_status()
    result = response.json()

    # The ReferenceCode path expected "prompt_probabilities". Current builds
    # return "completion_probabilities" for generated tokens only; prompt
    # logprobs are empty in those builds.
    prompt_probs = result.get("prompt_probabilities", [])
    expected = n - 1
    if len(prompt_probs) != expected:
        raise RuntimeError(
            f"{doc_id}: prompt_probabilities length {len(prompt_probs)} != expected {expected}. "
            "This llama-server build does not return per-prompt-position logprobs. "
            "Use backend=pytorch or a binary that supports prompt_logprobs."
        )

    topk_idx = np.zeros((expected, k), dtype=np.uint32)
    topk_logprobs = np.full((expected, k), -1e9, dtype=np.float32)

    for i, entry in enumerate(prompt_probs):
        top_entries = entry.get("top_logprobs", [])
        if len(top_entries) > k:
            top_entries = top_entries[:k]
        for j, e in enumerate(top_entries[:k]):
            topk_idx[i, j] = int(e["id"])
            topk_logprobs[i, j] = float(e["logprob"])

    return topk_idx, topk_logprobs


def _finalize_scores(
    token_ids: list[int],
    topk_idx: np.ndarray,
    topk_logprobs: np.ndarray,
    stored_mask: list[int] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert logprobs to probabilities and build the full-document arrays."""
    t = len(token_ids)
    k = topk_idx.shape[1]
    full_topk_idx = np.zeros((t, k), dtype=np.uint32)
    full_topk_logprobs = np.full((t, k), -1e9, dtype=np.float32)
    full_topk_idx[: t - 1] = topk_idx
    full_topk_logprobs[: t - 1] = topk_logprobs

    max_lp = np.max(full_topk_logprobs, axis=1, keepdims=True)
    exp_shifted = np.exp(np.clip(full_topk_logprobs - max_lp, -80, 0))
    probs = exp_shifted / np.maximum(exp_shifted.sum(axis=1, keepdims=True), 1e-12)

    loss_mask = np.ones(t, dtype=np.uint8)
    if stored_mask is not None:
        stored = np.asarray(stored_mask, dtype=np.uint8)
        loss_mask[: len(stored)] = stored[:t]
    loss_mask[-1] = 0

    return (
        np.asarray(token_ids, dtype=np.uint32),
        full_topk_idx,
        probs.astype(np.float32),
        loss_mask,
    )


def score_document_llama_server(
    port: int,
    record: dict[str, Any],
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Score one document with llama-server."""
    token_ids = record["tokens"]
    doc_id = record.get("id", "unknown")
    extracted = _extract_topk_from_server(port, token_ids, k, doc_id)
    if extracted is None:
        return None
    topk_idx, topk_logprobs = extracted
    return _finalize_scores(token_ids, topk_idx, topk_logprobs, record.get("loss_mask"))


def iter_llama_server_scores(
    port: int,
    records: list[dict[str, Any]],
    k: int,
) -> Iterator[tuple[dict[str, Any], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None]]:
    """Yield (record, arrays) for each document via llama-server."""
    for record in records:
        try:
            arrays = score_document_llama_server(port, record, k)
        except Exception:
            raise
        yield record, arrays


def _score_documents_pytorch(
    model,
    records: list[dict[str, Any]],
    k: int,
    device: str,
    chunk_positions: int = 256,
) -> Iterator[tuple[dict[str, Any], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
    """Score a list of documents with the PyTorch donor model."""
    model = model.to(device).eval()
    W = model.lm_head.weight.to(torch.float32)

    for record in records:
        token_ids = record["tokens"]
        t = len(token_ids)
        if t < 2:
            continue

        ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            hidden = model(ids, return_hidden=True)  # [1, T, D]

        topk_idx = np.zeros((t, k), dtype=np.uint32)
        topk_logprobs = np.full((t, k), -1e9, dtype=np.float32)

        for s in range(0, t, chunk_positions):
            e = min(s + chunk_positions, t)
            z = hidden[0, s:e].to(torch.float32) @ W.T
            lp = torch.log_softmax(z, dim=-1)
            vals, idxs = torch.topk(lp, k, dim=-1)
            topk_idx[s:e] = idxs.detach().cpu().numpy().astype(np.uint32)
            topk_logprobs[s:e] = vals.detach().cpu().numpy().astype(np.float32)

        # Position i predicts token i+1; the last position has no target.
        arrays = _finalize_scores(
            token_ids, topk_idx[: t - 1], topk_logprobs[: t - 1], record.get("loss_mask")
        )
        yield record, arrays


def build_pytorch_scorer(
    checkpoint_dir: str | Path,
    device: str,
    dtype_str: str = "bf16",
) -> Callable[[list[dict[str, Any]], int], Iterator]:
    """Load the donor model once and return a scorer callable."""
    dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float16
    model = load_donor(checkpoint_dir, device=device, dtype=dtype)

    def scorer(records: list[dict[str, Any]], k: int) -> Iterator:
        return _score_documents_pytorch(model, records, k, device)

    return scorer
