"""Engram addressing acceptance tests (annex A1.8.1–A1.8.4) at dev_tiny scale
(identity canon, vocab 1024, unigram modulus 1031, bigram/trigram primes
4093/4091 and 4079/4073)."""
import numpy as np

from train.src.config import load_config
from train.src.engram import addressing
from train.src.engram.tables import EngramTables

DEV_TINY = "train/configs/model/dev_tiny.yaml"
VOCAB = 1024


def _tables() -> EngramTables:
    cfg = load_config(DEV_TINY).engram
    return EngramTables(cfg, vocab_size=VOCAB)


def test_unigram_injective_both_heads():
    """A1.8.1: (A*c + B) mod M_uni is a bijection restricted to the vocab —
    zero collisions across all 1024 ids, on BOTH unigram heads."""
    t = _tables()
    ids = np.arange(VOCAB, dtype=np.uint32)
    for k in range(t.cfg.heads_per_order):
        idx = addressing.unigram_addresses(ids, k, t.moduli[(1, k)])
        assert len(np.unique(idx)) == VOCAB
        assert idx.max() < t.moduli[(1, k)]


def test_ngram_head_collision_sets_decorrelate():
    """A1.8.2: distinct primes + salts per head — bigram addresses that
    collide on head 0 must collide on head 1 at chance level (~1/M)."""
    t = _tables()
    rng = np.random.default_rng(0)
    stream = rng.integers(0, VOCAB, size=8192, dtype=np.uint32)[None, :]
    a0, _ = addressing.ngram_addresses(stream, 2, 0, t.moduli[(2, 0)])
    a1, _ = addressing.ngram_addresses(stream, 2, 1, t.moduli[(2, 1)])
    a0, a1 = a0[0], a1[0]

    # Collision indicator per head via sorting (pairwise would be O(T^2)).
    def collided(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a, kind="stable")
        s = a[order]
        dup = np.zeros(len(a), dtype=bool)
        dup[1:] = s[1:] == s[:-1]
        dup[:-1] |= s[1:] == s[:-1]
        out = np.zeros(len(a), dtype=bool)
        out[order] = dup
        return out

    c0, c1 = collided(a0), collided(a1)
    both = (c0 & c1).sum()
    chance = c0.sum() * c1.sum() / len(a0)
    assert c0.sum() > 0, "expected SOME head-0 collisions at this scale"
    # Chance-level overlap: allow generous slack (fixed seed, deterministic).
    assert both <= max(4 * chance, 8), f"both-head collisions {both} vs chance {chance:.1f}"


def test_ngram_order_sensitive():
    """A1.8.3: the mix reads oldest-first — addr(a,b,c) != addr(c,b,a)."""
    t = _tables()
    rng = np.random.default_rng(1)
    for _ in range(20):
        a, b, c = rng.integers(0, VOCAB, size=3)
        if a == c:
            continue
        fwd = np.array([[a, b, c]], dtype=np.uint32)
        rev = np.array([[c, b, a]], dtype=np.uint32)
        for n in (2, 3):
            for k in range(t.cfg.heads_per_order):
                ia, _ = addressing.ngram_addresses(fwd, n, k, t.moduli[(n, k)])
                ib, _ = addressing.ngram_addresses(rev, n, k, t.moduli[(n, k)])
                assert ia[0, -1] != ib[0, -1]


def test_addressing_deterministic():
    """A1.8.4: fresh tables + fresh computations give identical addresses."""
    x = np.random.default_rng(2).integers(0, VOCAB, size=(4, 64), dtype=np.uint32)
    idx1, valid1 = _tables().address(x)
    idx2, valid2 = _tables().address(x)
    assert np.array_equal(idx1, idx2) and np.array_equal(valid1, valid2)


def test_boundary_validity():
    """Order-n grams need n tokens of context: positions t < n-1 are invalid;
    unigrams valid everywhere."""
    t = _tables()
    x = np.arange(24, dtype=np.uint32)[None, :] % VOCAB
    _, valid = t.address(x)
    assert valid.shape == (1, 24, 3)
    assert valid[0, :, 0].all()            # unigram
    assert not valid[0, 0, 1]              # bigram: t=0 invalid
    assert valid[0, 1:, 1].all()
    assert not valid[0, :2, 2].any()       # trigram: t<2 invalid
    assert valid[0, 2:, 2].all()


def test_table_init_distribution_and_determinism():
    """Annex A1.5: rows Uniform(-0.01, 0.01) in bf16, from dedicated
    per-table generators — building tables twice gives bitwise-identical
    rows and never disturbs the global RNG (I9)."""
    import torch

    # Table construction must not consume the global RNG (I9).
    torch.manual_seed(123)
    expected = torch.rand(4)
    torch.manual_seed(123)
    _tables()
    assert torch.equal(torch.rand(4), expected)

    t1, t2 = _tables(), _tables()
    for key in t1.table_keys:
        r1, r2 = t1.rows[key], t2.rows[key]
        assert torch.equal(r1, r2)
        f = r1.float()
        # bf16(0.01) rounds up to ~0.01001 — the bound is the bf16 grid.
        assert f.abs().max() <= 0.0101
        assert f.std() > 0.003  # uniform on [-.01,.01] -> std ~ .0058
