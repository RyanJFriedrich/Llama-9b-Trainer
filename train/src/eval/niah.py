"""Needle-in-a-haystack retrieval probe (spec §8 item 5): the global-layer
retrieval check at 4k/8k/16k/32k (tokenized lengths on the deploy box; toy
lengths at dev scale).

Design: the haystack is a filler token stream with a needle KEY:VALUE pair
embedded at a chosen depth. The query asks for the key; success = the model
assigns the needle's value a higher logprob than every distractor value.
Self-contained (token ids, no tokenizer dependency) so it runs on any
config; the real acceptance run uses tokenized natural text on the 8B.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch


@torch.no_grad()
def needle_accuracy(
    model,
    lengths: Sequence[int],
    depths: Sequence[float] = (0.1, 0.5, 0.9),
    n_distractors: int = 4,
    filler_id: int = 100,
    key_id: int = 50001,
    value_id: int = 90001,
    query_id: int = 60001,
    vocab_size: int = 128256,
    device: str = "cuda",
    seed: int = 0,
) -> dict[tuple[int, float], bool]:
    """For each (length, depth): True if the needle value outscores all
    distractor values at the query position.

    Sequence layout: [filler...] KEY VALUE [filler...] QUERY KEY -> the model
    must predict VALUE at the final position.
    """
    assert key_id < vocab_size and value_id < vocab_size and filler_id < vocab_size
    model = model.to(device).eval()
    g = torch.Generator().manual_seed(seed)
    results: dict[tuple[int, float], bool] = {}

    distractors = torch.randint(0, vocab_size, (n_distractors,), generator=g)
    while value_id in distractors:  # extremely unlikely; keep clean
        distractors = torch.randint(0, vocab_size, (n_distractors,), generator=g)

    W = model.lm_head.weight.to(torch.float32)
    for length in lengths:
        for depth in depths:
            pos = int(length * depth)
            ids = torch.full((length,), filler_id, dtype=torch.long)
            ids[pos] = key_id
            ids[pos + 1] = value_id
            ids[-2] = query_id
            ids[-1] = key_id
            ids = ids.unsqueeze(0).to(device)

            hidden = model(ids, return_hidden=True)
            logp = torch.log_softmax(hidden[0, -1].to(torch.float32) @ W.T, dim=-1)
            candidates = torch.cat([torch.tensor([value_id], device=device), distractors.to(device)])
            scores = logp[candidates]
            results[(length, depth)] = bool(scores[0].item() == scores.max().item()
                                            and scores[0].item() > scores[1:].max().item())
    return results
