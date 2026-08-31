"""M3 integration: converter (dense JSONL -> shard) + end-to-end smoke
(loader -> fused KD loss). Format unit tests live in test_topk_data.py and
test_kd_loss.py."""
import json
import math
from pathlib import Path

import numpy as np
import torch

from train.src.data.topk_loader import TopKLoader
from train.src.distill.kd_loss import kd_loss
from train.src.tools.shard_converter import convert_jsonl


def _write_dense_jsonl(path: Path, k: int, doc_lens: list[int], seed: int = 11,
                       vocab: int = 128256) -> None:
    rng = np.random.default_rng(seed)
    with path.open("w", encoding="utf-8") as f:
        for d, t in enumerate(doc_lens):
            tokens = rng.integers(0, vocab, size=t).tolist()
            ids = np.stack([rng.permutation(vocab)[:k] for _ in range(t)])
            raw = rng.random((t, k))
            probs = raw / raw.sum(axis=1, keepdims=True) * 0.85
            rec = {
                "doc_id": f"doc-{d}",
                "tokens": tokens,
                "topk_idx": ids.tolist(),
                "topk_logprobs": np.log(probs).tolist(),
            }
            f.write(json.dumps(rec) + "\n")


def test_converter_round_trip(tmp_path):
    src = tmp_path / "intermediate.jsonl"
    _write_dense_jsonl(src, k=5, doc_lens=[24, 18])
    sidecar = convert_jsonl(src, tmp_path / "shard", k=5,
                            teacher_id="synthetic", log_filename=str(tmp_path / "c.log"))
    assert sidecar["k"] == 5
    assert sidecar["total_tokens"] == 42
    assert sidecar["num_documents"] == 2
    assert sidecar["teacher_id"] == "synthetic"

    # Loader consumes the converted shard; mass invariant holds.
    loader = TopKLoader([tmp_path / "shard"], seq_len=10)
    for seq in loader.iter_sequences():
        mass = seq["topk_w"].sum(-1) + seq["tail_w"]
        assert torch.allclose(mass, torch.ones_like(mass), atol=1e-3)


def test_converter_rejects_sparse_or_multi_k(tmp_path):
    import pytest
    src = tmp_path / "bad.jsonl"
    rec = {"tokens": [1, 2, 3], "topk_idx": [[1, 2]], "topk_logprobs": [[-0.5, -0.5]]}
    src.write_text(json.dumps(rec) + "\n")
    with pytest.raises(ValueError, match="dense"):
        convert_jsonl(src, tmp_path / "s", k=2, teacher_id="t",
                      log_filename=str(tmp_path / "c.log"))

    src2 = tmp_path / "bad2.jsonl"
    rec = {"tokens": [1, 2], "topk_idx": [[1, 2], [3, 4]],
           "topk_logprobs": [[0.5, -0.5], [-0.5, -0.5]]}  # positive logprob
    src2.write_text(json.dumps(rec) + "\n")
    with pytest.raises(ValueError, match="logprobs"):
        convert_jsonl(src2, tmp_path / "s2", k=2, teacher_id="t",
                      log_filename=str(tmp_path / "c2.log"))


def test_end_to_end_shard_to_loss(tmp_path):
    """Shard -> loader window -> fused KD loss on a toy model: finite loss,
    finite grads, mask honored."""
    src = tmp_path / "intermediate.jsonl"
    _write_dense_jsonl(src, k=4, doc_lens=[40])
    convert_jsonl(src, tmp_path / "shard", k=4, teacher_id="synthetic",
                  log_filename=str(tmp_path / "c.log"))

    seq = next(TopKLoader([tmp_path / "shard"], seq_len=32).iter_sequences())
    V, D = 128256, 64
    g = torch.Generator().manual_seed(1)
    hidden = torch.randn(1, 32, D, generator=g, requires_grad=True)
    W = torch.randn(V, D, generator=g) * 0.02
    gold = torch.cat([seq["tokens"][1:], seq["tokens"][:1]]).unsqueeze(0)  # shifted labels

    loss = kd_loss(hidden.unsqueeze(0) if hidden.dim() == 2 else hidden, W,
                   seq["topk_idx"].unsqueeze(0), seq["topk_w"].unsqueeze(0),
                   seq["tail_w"].unsqueeze(0), gold,
                   seq["loss_mask"].float().unsqueeze(0),
                   alpha=1.0, chunk_size=8)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(hidden.grad).all()
    # The window's last position and no other boundary is masked (single doc):
    assert seq["loss_mask"][:-1].all() and not seq["loss_mask"][-1]
