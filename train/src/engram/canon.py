"""Canonical token-id compression (annex A1.2) — Engram stage 0.

Hashing operates on canonical ids, not raw token ids: surface variants
("Apple", "apple", "Ａｐｐｌｅ") merge into one memory row. The pinned
construction is exactly

    canon_string(token) = casefold(NFKC(text(token)))
    class(token)        = canon_string        (equivalence = identical string)
    canonical_id(token) = min { token_id | token_id in class(token) }

No whitespace stripping, no punctuation rules, no stemmer. The map
P: V -> V' is a data artifact built once from the Llama 3.1 tokenizer by
`train/scripts/build_engram_canon.py`, stored as uint32[V] with a sha256
checksum that the run config pins (`engram.canon_sha256`). Recompute only if
the tokenizer changes.

Fallback (compression vetoed / toy vocabs with no tokenizer): P = identity.
"""
from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Union

import numpy as np

from train.utils.log import log

SCHEME = "nfkc+casefold,min-id-representative"


def canon_string(text: str) -> str:
    """The pinned normalization: NFKC, then casefold. Nothing else."""
    return unicodedata.normalize("NFKC", text).casefold()


def build_canon_map(texts: list[str]) -> tuple[np.ndarray, dict]:
    """Build P from each token id's decoded surface text. Returns (P, stats).

    P[t] = canonical id of token t (uint32[V]); classes form on identical
    canon_string, representative = minimum raw id in the class.
    """
    representative: dict[str, int] = {}
    P = np.arange(len(texts), dtype=np.uint32)
    for t, s in enumerate(texts):
        key = canon_string(s)
        rep = representative.get(key)
        if rep is None:
            representative[key] = t
        else:
            P[t] = rep
    n_merged = len(texts) - len(representative)
    stats = {
        "vocab": len(texts),
        "canon_vocab": len(representative),
        "merged": n_merged,
        "empty_canon_class_size": int((P == representative.get("", -1)).sum()) if "" in representative else 0,
    }
    return P, stats


def canon_sha256(P: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(P, dtype=np.uint32).tobytes()).hexdigest()


def save_canon(P: np.ndarray, path: Union[str, Path], stats: dict) -> str:
    """Write the artifact (P.npy + sidecar json). Returns the sha256 to pin
    into the run config (`engram.canon_sha256`)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    P = np.ascontiguousarray(P, dtype=np.uint32)
    np.save(path, P)
    sha = canon_sha256(P)
    sidecar = {
        "scheme": SCHEME,
        "sha256": sha,
        **stats,
    }
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
    log(f"canon artifact written: {path} (|V'|={stats['canon_vocab']}, "
        f"merged {stats['merged']}, sha256={sha[:16]}...)", print_console=True)
    return sha


def load_canon(path: Union[str, Path], expected_sha256: str = "") -> np.ndarray:
    """Load the artifact; verify the pinned checksum when provided. A checksum
    mismatch is a hard error — a different map silently re-addresses every
    table row."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"canon artifact missing: {path} — build it with "
            "python -m train.scripts.build_engram_canon (or set "
            "engram.canonical_compression: false for the identity fallback)"
        )
    P = np.load(path)
    if P.dtype != np.uint32 or P.ndim != 1:
        raise ValueError(f"{path}: expected uint32[V], got {P.dtype}{P.shape}")
    if expected_sha256:
        actual = canon_sha256(P)
        if actual != expected_sha256:
            raise ValueError(
                f"{path}: sha256 mismatch (config pins {expected_sha256[:16]}..., "
                f"artifact is {actual[:16]}...) — the canon map determines all "
                "table addressing; do not swap artifacts mid-run"
            )
    return P


def identity_canon(vocab_size: int) -> np.ndarray:
    """The A1.2 fallback: P = identity (compression OFF)."""
    return np.arange(vocab_size, dtype=np.uint32)
