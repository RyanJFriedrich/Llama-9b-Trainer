"""M1 donor parity probe: our decoder vs HF on a frozen probe batch.

Usage (from repo root):
    python -m train.scripts.parity_probe [--dir OriginalModel] [--dtype bf16|fp32]

Loads the HF checkpoint in `--dir` into both HF's LlamaForCausalLM (eager
attention) and our LlamaBaseModel, runs the fixed-seed probe batch through
both, and logs the logit difference. This is the interactive companion of
train/tests/test_parity_hf_llama.py::test_parity_donor_8b.
"""
import argparse

import torch

from train.src.tools.load_donor import load_donor
from train.utils.log import log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="OriginalModel")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--seq-len", type=int, default=128)
    args = parser.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoModelForCausalLM

    log(f"parity_probe: loading HF reference from {args.dir} ({args.dtype}, {device})")
    hf = AutoModelForCausalLM.from_pretrained(
        args.dir, dtype=dtype, attn_implementation="eager"
    ).to(device).eval()
    ours = load_donor(args.dir, device=device, dtype=dtype).eval()

    g = torch.Generator().manual_seed(0xC0FFEE)
    ids = torch.randint(0, ours.config.vocab_size, (1, args.seq_len), generator=g).to(device)
    with torch.no_grad():
        hf_logits = hf(ids).logits
        our_logits = ours(ids)

    diff = (hf_logits - our_logits).abs()
    log(
        f"parity_probe: max abs diff {diff.max().item():.3e}, mean {diff.mean().item():.3e}, "
        f"logits |max| {hf_logits.abs().max().item():.1f}",
        print_console=True,
    )


if __name__ == "__main__":
    main()
