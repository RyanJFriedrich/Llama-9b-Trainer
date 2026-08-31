"""NPZ -> spec-shard converter tests (docs/NPZFormat.md -> spec §6.1).

The load-bearing properties: the one-row shift between "row predicts
tokens[t]" (NPZ) and "row is the distribution at position t" (spec shard),
GT-at-slot-0 with its true prob preserved, chunk -> doc mapping, and the
mass invariant (I6) surviving conversion.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from train.src.data.topk_loader import TopKLoader, TopKShard
from train.src.distill.kd_loss import kd_loss
from train.src.tools.npz_converter import convert_npz

VOCAB = 500
K = 6


def _write_npz(path: Path, chunk_lens: list[int], seed: int = 7) -> dict:
    """Synthetic bulk NPZ faithful to docs/NPZFormat.md: slot 0 = GT with true
    prob, slots 1..K-1 shuffled intact pairs, unnormalized rows, loss_mask 0 at
    chunk starts (-1/zeros rows), chunks tiled contiguously."""
    rng = np.random.default_rng(seed)
    n = sum(chunk_lens)
    tokens = rng.integers(0, VOCAB, size=n).astype(np.uint32)
    teacher_ids = np.zeros((n, K), dtype=np.int32)
    teacher_probs = np.zeros((n, K), dtype=np.float32)
    loss_mask = np.ones(n, dtype=np.uint8)

    starts = np.zeros(len(chunk_lens), dtype=np.int64)
    for c, L in enumerate(chunk_lens[1:], 1):
        starts[c] = starts[c - 1] + chunk_lens[c - 1]
    loss_mask[starts] = 0  # chunk position 0: no distribution exists
    teacher_ids[starts, 0] = -1

    for t in range(n):
        if loss_mask[t] == 0:
            continue
        gt = int(tokens[t])
        gt_prob = float(rng.uniform(1e-4, 0.6))
        rest = rng.uniform(0.0, 0.05, size=K - 1).astype(np.float64)
        rest *= (0.85 - gt_prob) / rest.sum()  # row mass ~0.85, tail ~0.15
        ids = rng.permutation(VOCAB)[: K - 1]
        ids[ids == gt] = (ids[ids == gt] + 1) % VOCAB  # keep GT out of the shuffled slots
        pairs = list(zip(ids.tolist(), rest.tolist()))
        rng.shuffle(pairs)  # shuffled as intact (id, prob) pairs
        teacher_ids[t, 0] = gt
        teacher_probs[t, 0] = gt_prob
        for j, (i, p) in enumerate(pairs, 1):
            teacher_ids[t, j] = i
            teacher_probs[t, j] = p

    np.savez(
        path,
        tokens=tokens,
        teacher_ids=teacher_ids,
        teacher_probs=teacher_probs,
        loss_mask=loss_mask,
        chunk_start=starts,
        chunk_length=np.asarray(chunk_lens, dtype=np.int64),
    )
    return {
        "tokens": tokens, "teacher_ids": teacher_ids, "teacher_probs": teacher_probs,
        "loss_mask": loss_mask, "chunk_start": starts,
        "chunk_length": np.asarray(chunk_lens, dtype=np.int64),
    }


def test_convert_shift_and_gt_slot(tmp_path):
    """spec_row[t] == npz_row[t+1] within each chunk; gold sits at slot 0 with
    its true prob; each doc's final row is zero + masked."""
    src = _write_npz(tmp_path / "bulk_00000.npz", [10, 8, 5])
    sidecar = convert_npz(tmp_path / "bulk_00000.npz", tmp_path / "shard",
                          log_filename=str(tmp_path / "c.log"))
    assert sidecar["k"] == K
    assert sidecar["total_tokens"] == 23
    assert sidecar["num_documents"] == 3
    assert sidecar["fold_version"] == "v2"
    assert sidecar["data_class"] == "a"

    sh = TopKShard(tmp_path / "shard")
    # Token identity (I5 storage side): the stream survives verbatim.
    assert np.array_equal(np.asarray(sh.tokens), src["tokens"])
    doc_id = np.asarray(sh.doc_id)
    for c, (s, L) in enumerate(zip(src["chunk_start"], src["chunk_length"])):
        s, L = int(s), int(L)
        rows = np.nonzero(doc_id == c)[0]
        assert len(rows) == L
        idx = np.asarray(sh.topk_idx)[rows]
        w = np.asarray(sh.topk_w, dtype=np.float32)[rows]
        mask = np.asarray(sh.loss_mask)[rows]
        # Final position of the doc: zero row, masked off.
        assert mask[-1] == 0 and idx[-1].sum() == 0 and w[-1].sum() == 0
        # Every other row: slot 0 is the gold token (next stream token) with
        # the NPZ's true GT prob; full rows match the shifted NPZ rows.
        for t in range(L - 1):
            npz_row = s + t + 1
            assert mask[t] == 1
            assert idx[t, 0] == src["tokens"][npz_row]
            assert np.isclose(w[t, 0], src["teacher_probs"][npz_row, 0], atol=1e-3)
            assert np.array_equal(idx[t], src["teacher_ids"][npz_row])
            assert np.allclose(w[t], src["teacher_probs"][npz_row], atol=1e-3)


def test_convert_mass_invariant_and_loader(tmp_path):
    """I6 survives conversion, and the loader consumes the shard."""
    _write_npz(tmp_path / "b.npz", [16, 16])
    convert_npz(tmp_path / "b.npz", tmp_path / "shard",
                log_filename=str(tmp_path / "c.log"))
    loader = TopKLoader([tmp_path / "shard"], seq_len=8)
    n_seq = 0
    for seq in loader.iter_sequences():
        n_seq += 1
        mass = seq["topk_w"].sum(-1) + seq["tail_w"]
        assert torch.allclose(mass, torch.ones_like(mass), atol=1e-3)
        # Doc boundaries (chunk boundaries) are masked by the loader.
        boundary = seq["doc_id"][1:] != seq["doc_id"][:-1]
        assert not seq["loss_mask"][1:][boundary].any()
    assert n_seq == 4  # 32 tokens / 8


def test_convert_end_to_end_loss(tmp_path):
    """Converted shard -> loader window -> fused KD loss: finite loss + grads."""
    _write_npz(tmp_path / "b.npz", [40])
    convert_npz(tmp_path / "b.npz", tmp_path / "shard",
                log_filename=str(tmp_path / "c.log"))
    seq = next(TopKLoader([tmp_path / "shard"], seq_len=32).iter_sequences())
    V, D = 128256, 64
    g = torch.Generator().manual_seed(1)
    hidden = torch.randn(1, 32, D, generator=g, requires_grad=True)
    W = torch.randn(V, D, generator=g) * 0.02
    gold = torch.cat([seq["tokens"][1:], seq["tokens"][:1]]).unsqueeze(0)
    loss = kd_loss(hidden, W, seq["topk_idx"].unsqueeze(0), seq["topk_w"].unsqueeze(0),
                   seq["tail_w"].unsqueeze(0), gold, seq["loss_mask"].float().unsqueeze(0),
                   alpha=0.9, chunk_size=8)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(hidden.grad).all()
    # Gold sits in the anchor support at slot 0 on unmasked positions.
    unmasked = seq["loss_mask"].bool()
    assert (seq["topk_idx"][unmasked, 0] == gold[0][unmasked]).all()


def test_convert_rejects_contract_violations(tmp_path):
    src = _write_npz(tmp_path / "bad.npz", [8, 8])
    d = dict(np.load(tmp_path / "bad.npz"))
    d["teacher_ids"][3, 0] = (d["tokens"][3] + 1) % VOCAB  # GT slot corrupted
    np.savez(tmp_path / "bad_gt.npz", **d)
    with pytest.raises(ValueError, match="GT-slot"):
        convert_npz(tmp_path / "bad_gt.npz", tmp_path / "s1",
                    log_filename=str(tmp_path / "c.log"))

    d = dict(np.load(tmp_path / "bad.npz"))
    d["chunk_length"] = d["chunk_length"] + 1  # chunks no longer tile the stream
    np.savez(tmp_path / "bad_tile.npz", **d)
    with pytest.raises(ValueError, match="tile"):
        convert_npz(tmp_path / "bad_tile.npz", tmp_path / "s2",
                    log_filename=str(tmp_path / "c.log"))
