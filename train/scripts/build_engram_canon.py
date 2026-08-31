"""Build the Engram canonical-id artifact (annex A1.2) from the Llama 3.1
tokenizer. One-time data build; recompute only if the tokenizer changes.

Usage (from repo root):
    python -m train.scripts.build_engram_canon [--tokenizer OriginalModel] \
        [--out train/src/engram/assets/canon_llama31_v1.npy]

Prints the sha256 to pin into the model config's `engram.canon_sha256`.
"""
from __future__ import annotations

import argparse

from train.src.engram.canon import build_canon_map, save_canon
from train.utils.log import log


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tokenizer", default="OriginalModel")
    p.add_argument("--out", default="train/src/engram/assets/canon_llama31_v1.npy")
    args = p.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab = len(tok)  # 128,256 incl. special slots
    log(f"tokenizer loaded from {args.tokenizer} (vocab {vocab}); decoding "
        f"token surfaces...", print_console=True)

    # Single-token decode gives each id's surface text. Chunked to keep
    # progress visible; specials decode to their literal strings.
    texts: list[str] = []
    chunk = 8192
    for start in range(0, vocab, chunk):
        ids = [[t] for t in range(start, min(start + chunk, vocab))]
        texts.extend(tok.batch_decode(ids))

    P, stats = build_canon_map(texts)
    sha = save_canon(P, args.out, stats)
    log(
        f"canon build done: |V|={stats['vocab']} -> |V'|={stats['canon_vocab']} "
        f"({stats['merged']} merged; empty-canon class size "
        f"{stats['empty_canon_class_size']})\n"
        f"PIN THIS: engram.canon_sha256: \"{sha}\"",
        print_console=True,
    )


if __name__ == "__main__":
    main()
