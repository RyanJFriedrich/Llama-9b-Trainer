"""Memory attribution diagnostic (deploy-box bring-up aid — not a test).

Builds the model at a model config's real scale and logs CUDA memory after
each phase of a training step — static fp32 masters, forward, backward,
optimizer step (8-bit states allocate here), and a steady-state second step —
once with the autocast weight-cast cache on and once off. Uses a synthetic
batch at the production shape: no data shards or donor weights needed
(scratch init; the memory footprint is init-independent).

Run from the repo root on the box:

    python -m train.scripts.mem_diag \
        --model-config train/configs/model/llama_9b_engram_v1.yaml

Paste the full output back. On OOM the script dumps
torch.cuda.memory_summary() before re-raising.
"""
from __future__ import annotations

import argparse

import torch

from train.src.config import load_config
from train.src.distill.kd_loss import kd_loss
from train.src.model.refit import RefitModel
from train.src.train.optim8bit import AdamW8bit
from train.utils.log import log

GIB = 2 ** 30


def mem(tag: str) -> None:
    free, total = torch.cuda.mem_get_info()
    log(f"mem[{tag}]: alloc {torch.cuda.memory_allocated() / GIB:.2f} GiB | "
        f"reserved {torch.cuda.memory_reserved() / GIB:.2f} GiB | "
        f"peak {torch.cuda.max_memory_allocated() / GIB:.2f} GiB | "
        f"device free {free / GIB:.2f}/{total / GIB:.2f} GiB",
        print_console=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-config", required=True)
    p.add_argument("--seq-len", type=int, default=8192)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--topk", type=int, default=32)
    p.add_argument("--precision", choices=["bf16", "fp8"], default="bf16")
    p.add_argument("--grad-checkpointing", type=int, default=1)
    p.add_argument("--loss-chunk", type=int, default=512)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("mem_diag needs a CUDA device")

    model_cfg = load_config(args.model_config)
    torch.manual_seed(0)

    try:
        model = RefitModel(model_cfg)  # scratch init — footprint is what matters
        n_params = sum(q.numel() for q in model.parameters())
        by_dtype: dict[str, int] = {}
        for q in model.parameters():
            by_dtype[str(q.dtype)] = by_dtype.get(str(q.dtype), 0) + q.numel() * q.element_size()
        log(f"mem_diag: {n_params / 1e9:.3f}B params; "
            + ", ".join(f"{k} {v / GIB:.2f} GiB" for k, v in sorted(by_dtype.items())),
            print_console=True)

        model.to("cuda")
        model.grad_checkpointing = bool(args.grad_checkpointing)
        if args.precision == "fp8":
            from train.src.train.fp8 import apply_fp8
            apply_fp8(model)
        mem("static fp32 masters on device")

        B, T, K, V = args.batch_size, args.seq_len, args.topk, model_cfg.vocab_size
        tokens = torch.randint(0, V, (B, T))
        gold = torch.cat([tokens[:, 1:], tokens[:, :1]], dim=1).cuda()
        topk_idx = torch.randint(0, V, (B, T, K), device="cuda")
        topk_w = torch.softmax(torch.randn(B, T, K), dim=-1).cuda()
        tail_w = torch.full((B, T), 0.1, device="cuda")
        loss_mask = torch.ones(B, T, device="cuda")
        loss_mask[:, -1] = 0

        optimizer = AdamW8bit(model.parameters(), lr=2e-4)  # states allocate at first step

        for cache in (True, False):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            gb = None
            if model_cfg.engram.enabled:
                idx, valid = model.engram_tables.address(tokens)
                gb = model.engram_tables.stage(idx, valid, torch.device("cuda"),
                                               requires_grad=True)
            mem(f"pre-step (autocast cache={cache})")
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=cache):
                hidden = model(tokens.cuda(), return_hidden=True, engram=gb)
            mem(f"after forward (autocast cache={cache})")
            loss = kd_loss(hidden.float(), model.lm_head.weight, topk_idx, topk_w,
                           tail_w, gold, loss_mask, alpha=1.0, temperature=1.0,
                           chunk_size=args.loss_chunk)
            loss.backward()
            mem(f"after backward (autocast cache={cache})")
            if cache:
                optimizer.step()  # first call allocates the 8-bit states
                mem("after first optimizer step (states now live)")

        log("mem_diag complete — final torch.cuda.memory_summary():\n"
            + torch.cuda.memory_summary(), print_console=True)
    except torch.OutOfMemoryError:
        log("mem_diag OOM — torch.cuda.memory_summary():\n"
            + torch.cuda.memory_summary(), print_console=True)
        raise


if __name__ == "__main__":
    main()
