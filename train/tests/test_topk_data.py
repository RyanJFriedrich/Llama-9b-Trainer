"""M3 data-path tests: shard writer + mmap loader (spec §6.2/§6.3)."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from train.src.data.topk_loader import TopKLoader, TopKShard
from train.src.data.topk_writer import ARRAY_DTYPES, ShardWriter


def _doc(rng: np.random.Generator, t: int, k: int, vocab: int = 128256,
         top_mass: float = 0.9):
    """One synthetic document: tokens + teacher top-k with the given mass."""
    tokens = rng.integers(0, vocab, size=t).astype(np.uint32)
    ids = np.stack([rng.permutation(vocab)[:k] for _ in range(t)]).astype(np.uint32)
    raw = rng.random((t, k)).astype(np.float32)
    probs = (raw / raw.sum(axis=1, keepdims=True)) * top_mass
    return tokens, ids, probs.astype(np.float32)


def _write_shard(root: Path, name: str, k: int, doc_lens: list[int],
                 seed: int = 0, top_mass: float = 0.9) -> Path:
    rng = np.random.default_rng(seed)
    shard = root / name
    w = ShardWriter(shard, k=k, teacher_id="synthetic-teacher", quantization="none",
                    chat_template_version="test-v1", fold_version="test")
    for t in doc_lens:
        tokens, ids, probs = _doc(rng, t, k, top_mass=top_mass)
        w.add_document(tokens, ids, probs)
    w.finalize()
    return shard


def test_round_trip_variable_k(tmp_path):
    """Two shards with different k (4 and 10) round-trip exactly."""
    s1 = _write_shard(tmp_path, "k4", k=4, doc_lens=[20, 15])
    s2 = _write_shard(tmp_path, "k10", k=10, doc_lens=[30], seed=1)

    sh1, sh2 = TopKShard(s1), TopKShard(s2)
    assert sh1.k == 4 and sh2.k == 10  # k from sidecar, not shape
    assert len(sh1) == 35 and len(sh2) == 30

    loader = TopKLoader([s1, s2], seq_len=16)
    seqs = list(loader.iter_sequences())
    assert loader.num_sequences() == 35 // 16 + 30 // 16 == len(seqs)
    for seq in seqs:
        assert seq["tokens"].shape == (16,)
        assert seq["topk_idx"].dtype == torch.int64
        assert seq["topk_w"].dtype == torch.float32
        assert seq["tail_w"].dtype == torch.float32
    # Per-shard k preserved through packing.
    assert seqs[0]["topk_idx"].shape[1] == 4
    assert seqs[-1]["topk_idx"].shape[1] == 10


def test_topk_mass_invariant(tmp_path):
    """Mandated invariant: Σ topk_w + tail_w = 1 ± 1e-3 per position; k from
    sidecar; uint32 indices on disk."""
    shard = _write_shard(tmp_path, "s", k=7, doc_lens=[50], top_mass=0.75)
    sh = TopKShard(shard)
    # On-disk dtypes are the locked contract.
    assert sh.topk_idx.dtype == np.uint32
    assert sh.tokens.dtype == np.uint32
    assert sh.topk_w.dtype == np.float16
    assert sh.tail_w.dtype == np.float32

    w = np.asarray(sh.topk_w, dtype=np.float32)
    tail = np.asarray(sh.tail_w)
    mass = w.sum(axis=1) + tail
    assert np.allclose(mass, 1.0, atol=1e-3)
    # Release memmaps explicitly (Windows holds file locks on open memmaps).
    for name in ARRAY_DTYPES:
        setattr(sh, name, None)


def test_tail_w_fp32_fidelity(tmp_path):
    """A small tail (~1e-3) must survive exactly: computed in fp32 at storage
    time, NOT reconstructed from fp16-rounded weights (fp16 eps ~ 1e-3 near
    1.0 would swallow it)."""
    rng = np.random.default_rng(3)
    k, t = 6, 40
    shard = tmp_path / "s"
    wtr = ShardWriter(shard, k=k)
    tokens, ids, probs = _doc(rng, t, k)
    probs *= 0.999  # true tail ~ 1e-3
    wtr.add_document(tokens, ids, probs)
    wtr.finalize()

    sh = TopKShard(shard)
    stored_tail = np.asarray(sh.tail_w)
    expected = 1.0 - probs.astype(np.float32).sum(axis=1)
    assert np.allclose(stored_tail, expected, atol=1e-7)  # fp32 fidelity

    # Reconstructing from the fp16 weights would lose it — demonstrate why
    # the rule exists (not a loader behavior assertion).
    fp16_tail = 1.0 - np.asarray(sh.topk_w, dtype=np.float32).sum(axis=1)
    assert not np.allclose(fp16_tail, expected, atol=1e-4)


def test_doc_boundary_masking(tmp_path):
    """Mandated invariant: no loss across document boundaries or padding."""
    # Two docs of 10 tokens each; seq_len 16 -> first window spans the
    # boundary (doc0 fills 0..9, doc1 starts at 10).
    shard = _write_shard(tmp_path, "s", k=3, doc_lens=[10, 10])
    # Zero out some stored loss to prove stored mask is honored.
    loader = TopKLoader([shard], seq_len=16)
    seq = next(loader.iter_sequences())

    doc = seq["doc_id"]
    m = seq["loss_mask"]
    # Boundary: position 9 (last of doc 0) predicts position 10 (doc 1) -> off.
    assert doc[9].item() == 0 and doc[10].item() == 1
    assert m[9].item() is False or m[9].item() == 0
    # Last position of any window is off (target outside the window).
    assert m[-1].item() == 0
    # Interior positions of doc 0 are on.
    assert m[:9].all()
    # No masked-on position has a target in a different doc (target = j+1).
    for j in range(15):
        if m[j]:
            assert doc[j].item() == doc[j + 1].item()


def test_stored_loss_mask_honored(tmp_path):
    rng = np.random.default_rng(4)
    shard = tmp_path / "s"
    wtr = ShardWriter(shard, k=2)
    tokens, ids, probs = _doc(rng, 32, 2)
    mask = np.ones(32, dtype=np.uint8)
    mask[3:6] = 0  # a non-loss span (e.g. prompt tokens)
    wtr.add_document(tokens, ids, probs, loss_mask=mask)
    wtr.finalize()

    seq = next(TopKLoader([shard], seq_len=16).iter_sequences())
    assert not seq["loss_mask"][3:6].any()
    assert seq["loss_mask"][6:15].all()


def test_mmap_not_eager(tmp_path):
    shard = _write_shard(tmp_path, "s", k=4, doc_lens=[64])
    sh = TopKShard(shard)
    for name in ARRAY_DTYPES:
        assert isinstance(getattr(sh, name), np.memmap), name


def test_writer_rejects_bad_inputs(tmp_path):
    w = ShardWriter(tmp_path / "s", k=4)
    tokens = np.arange(10, dtype=np.uint32)
    ids = np.zeros((10, 4), dtype=np.uint32)
    with pytest.raises(ValueError, match="sum above 1"):
        w.add_document(tokens, ids, np.full((10, 4), 0.4, dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        w.add_document(tokens, ids[:, :3], np.zeros((10, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="vocab"):
        w.add_document(np.array([128256], dtype=np.uint32),
                       np.zeros((1, 4), dtype=np.uint32),
                       np.full((1, 4), 0.2, dtype=np.float32))


def test_shuffle_determinism(tmp_path):
    shard = _write_shard(tmp_path, "s", k=2, doc_lens=[128])
    a = [tuple(s["tokens"][:4].tolist()) for s in TopKLoader([shard], 16, shuffle=True, seed=7).iter_sequences()]
    b = [tuple(s["tokens"][:4].tolist()) for s in TopKLoader([shard], 16, shuffle=True, seed=7).iter_sequences()]
    c = [tuple(s["tokens"][:4].tolist()) for s in TopKLoader([shard], 16, shuffle=False).iter_sequences()]
    assert a == b  # same seed, same order
    assert sorted(a) == sorted(c)  # same windows either way
