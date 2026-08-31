"""Donor scoring entry point: produce top-k KD anchor shards (spec §6).

Usage (from repo root):
    python -m train.scripts.score_donor --input data/texts/*.txt --out shards/phase0/ ...
        --k 10 [--device cuda] [--max-len 4096]

Input files are plain UTF-8 text, one document per file (or per line with
--lines). Tokenized with the donor's HF tokenizer (OriginalModel/).
"""
import argparse
import glob
from pathlib import Path

from train.src.tools.donor_scorer import score_documents
from train.src.tools.load_donor import load_donor
from train.utils.log import log


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="glob of UTF-8 text files")
    p.add_argument("--out", required=True, help="output shard directory")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--donor", default="OriginalModel")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-len", type=int, default=4096)
    p.add_argument("--lines", action="store_true", help="one document per line")
    args = p.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.donor)

    docs: list[list[int]] = []
    for path in sorted(glob.glob(args.input)):
        text = Path(path).read_text(encoding="utf-8")
        chunks = text.splitlines() if args.lines else [text]
        for chunk in chunks:
            ids = tok.encode(chunk)
            if len(ids) >= 2:
                docs.append(ids)
    log(f"score_donor: {len(docs)} documents from {args.input}", print_console=True)

    model = load_donor(args.donor, device=args.device)
    score_documents(model, docs, args.out, k=args.k, device=args.device,
                    max_len=args.max_len)


if __name__ == "__main__":
    main()
