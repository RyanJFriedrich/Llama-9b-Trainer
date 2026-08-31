"""Engram addressing (annex A1.3/A1.4) — deterministic n-gram -> row indices.

All salt constants derive from a pinned SHA-256 stream, so the scheme extends
to any (order, head) without hand-picked magic numbers:

    const64(n, k, field) = sha256(f"engram9b.v1:n={n}:k={k}:{field}")[:8] le

Unigram (n=1) is INJECTIVE BY CONSTRUCTION: idx = (A*c + B) mod M_uni with
M_uni prime, A in [1, M-1], c < |V'| <= M_uni — a bijection restricted to the
vocab (annex A1.4; reproduces the 1B zero-collision measurement as a theorem).

n-gram (n>=2): 64-bit multiplicative-XOR mix over canonical ids (oldest token
first — order-sensitive), idx = state mod M[n,k], M prime and distinct per
(order, head) to decorrelate collision sets.

Everything here is pure CPU integer work on token ids (annex A1.6.1): no GPU
tensor is an input, so the dataloader/prefetch path computes indices.

Boundary rule (annex leaves this unpinned — recorded choice): the order-n
gram at position t spans tokens t-n+1..t; positions t < n-1 within a batch
window have no full context and contribute NOTHING for that order (the
`valid` mask). Unigrams are valid at every position.
"""
from __future__ import annotations

import hashlib

import numpy as np

# 2^64 / golden ratio (annex A1.4).
GOLD = np.uint64(0x9E3779B97F4A7C15)

LABEL_PREFIX = "engram9b.v1"  # bump only with a deliberate re-addressing decision


def const64(n: int, k: int, field: str) -> int:
    """64-bit constant from the pinned SHA-256 stream (annex A1.3)."""
    label = f"{LABEL_PREFIX}:n={n}:k={k}:{field}".encode()
    return int.from_bytes(hashlib.sha256(label).digest()[:8], "little")


def mul(n: int, k: int) -> np.uint64:
    return np.uint64(const64(n, k, "mul") | 1)  # forced odd


def seed(n: int, k: int) -> np.uint64:
    return np.uint64(const64(n, k, "seed"))


def aff_a(n: int, k: int, M: int) -> np.uint64:
    return np.uint64(const64(n, k, "affA") % (M - 1) + 1)  # in [1, M-1]


def aff_b(n: int, k: int) -> np.uint64:
    return np.uint64(const64(n, k, "affB"))


def smallest_prime_at_least(n: int) -> int:
    """Unigram modulus: smallest prime >= |V'| (annex A1.4)."""
    from train.src.config import _is_prime

    while not _is_prime(n):
        n += 1
    return n


def unigram_addresses(canon_ids: np.ndarray, k: int, M: int) -> np.ndarray:
    """idx = (A*c + B) mod M_uni — injective on {0, ..., |V'|-1} (annex A1.4).

    canon_ids: integer array any shape. Returns uint32 same shape.
    """
    c = canon_ids.astype(np.uint64)
    A = np.uint64(int(aff_a(1, k, M)))
    B = np.uint64(int(aff_b(1, k)))
    # A < M <= ~2^21 and c < M, so A*c + B < 2^42 — no uint64 overflow.
    return ((A * c + B) % np.uint64(M)).astype(np.uint32)


def ngram_addresses(canon_ids: np.ndarray, n: int, k: int, M: int) -> tuple[np.ndarray, np.ndarray]:
    """64-bit multiplicative-XOR mix over the n-gram ENDING at each position.

    canon_ids: [B, T] integer. Returns (idx uint32 [B, T], valid bool [B, T]);
    positions t < n-1 lack full context and are marked invalid (recorded
    boundary rule — see module docstring).
    """
    assert n >= 2
    ct = canon_ids.astype(np.uint64)
    B, T = ct.shape
    state = np.full((B, T), seed(n, k), dtype=np.uint64)
    m = mul(n, k)
    for j in range(n):  # c_0 is the OLDEST token in the window (order-sensitive)
        shift = n - 1 - j
        c = np.zeros((B, T), dtype=np.uint64)
        if shift == 0:
            c = ct
        elif shift < T:
            c[:, shift:] = ct[:, : T - shift]
        state = (state ^ (c + GOLD)) * m  # uint64 wraparound = mod 2^64
    valid = np.zeros((B, T), dtype=bool)
    if T >= n:
        valid[:, n - 1 :] = True
    return (state % np.uint64(M)).astype(np.uint32), valid


def address_batch(
    canon_ids: np.ndarray,
    orders: list[int],
    heads_per_order: int,
    moduli: dict[tuple[int, int], int],
) -> tuple[np.ndarray, np.ndarray]:
    """All (order, head) addresses for a batch of canonical-id windows.

    canon_ids: [B, T] uint32/uint64. moduli: (n, k) -> prime M.
    Returns (idx uint32 [B, T, n_orders, heads], valid bool [B, T, n_orders]).
    """
    B, T = canon_ids.shape
    n_orders = len(orders)
    idx = np.zeros((B, T, n_orders, heads_per_order), dtype=np.uint32)
    valid = np.zeros((B, T, n_orders), dtype=bool)
    for oi, n in enumerate(orders):
        for k in range(heads_per_order):
            M = moduli[(n, k)]
            if n == 1:
                idx[:, :, oi, k] = unigram_addresses(canon_ids, k, M)
                valid[:, :, oi] = True
            else:
                a, v = ngram_addresses(canon_ids, n, k, M)
                idx[:, :, oi, k] = a
                if k == 0:
                    valid[:, :, oi] = v
    return idx, valid
