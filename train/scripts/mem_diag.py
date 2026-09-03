"""Memory attribution diagnostic (deploy-box bring-up aid — not a test).

Builds the model at a model config's real scale and logs CUDA memory after
each phase of a training step — and after EVERY attention/MLP sublayer via
forward hooks, so per-layer growth (or a single sublayer that retains its
activations) shows up directly in the log. Runs the fwd/bwd pass twice:
autocast weight-cast cache OFF first, then ON (so a crash in one pass can't
hide the other). Synthetic batch: no data shards or donor weights needed
(scratch init; the memory footprint is init-independent).

Run from the repo root on the box:

    python -m train.scripts.mem_diag \
        --model-config train/configs/model/llama_9b_engram_v1.yaml

Everything is in common.log afterwards:

    grep -E "mem_diag:|mem\\[|OOM" ~/llama-9b/common.log | tail -160

On OOM the failing pass dumps torch.cuda.memory_summary() and the script
continues with the remaining pass.
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

    # Per-sublayer probes: fire after each attention/MLP call (also on
    # backward recompute under gradient checkpointing — those lines mark
    # where the recompute working set lands).
    handles = []
    for i, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.register_forward_hook(
            lambda m, a, o, i=i: mem(f"L{i:02d}.attn")))
        handles.append(layer.mlp.register_forward_hook(
            lambda m, a, o, i=i: mem(f"L{i:02d}.mlp")))
    handles.append(model.model.embed_tokens.register_forward_hook(
        lambda m, a, o: mem("embed")))

    B, T, K, V = args.batch_size, args.seq_len, args.topk, model_cfg.vocab_size
    tokens = torch.randint(0, V, (B, T))
    gold = torch.cat([tokens[:, 1:], tokens[:, :1]], dim=1).cuda()
    topk_idx = torch.randint(0, V, (B, T, K), device="cuda")
    topk_w = torch.softmax(torch.randn(B, T, K), dim=-1).cuda()
    tail_w = torch.full((B, T), 0.1, device="cuda")
    loss_mask = torch.ones(B, T, device="cuda")
    loss_mask[:, -1] = 0

    optimizer = AdamW8bit(model.parameters(), lr=2e-4)  # states allocate at first step

    # Cache-OFF pass first: if the autocast weight-cast cache is a large
    # contributor this pass survives and the comparison answers it; an OOM in
    # either pass is caught per-pass so the other still runs.
    for cache in (False, True):
        try:
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            gb = None
            if model_cfg.engram.enabled:
                idx, valid = model.engram_tables.address(tokens)
                gb = model.engram_tables.stage(idx, valid, torch.device("cuda"),
                                               requires_grad=True)
            mem(f"pass start (autocast cache={cache})")
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=cache):
                hidden = model(tokens.cuda(), return_hidden=True, engram=gb)
            mem(f"after forward (autocast cache={cache})")
            loss = kd_loss(hidden.float(), model.lm_head.weight, topk_idx, topk_w,
                           tail_w, gold, loss_mask, alpha=1.0, temperature=1.0,
                           chunk_size=args.loss_chunk)
            loss.backward()
            mem(f"after backward (autocast cache={cache})")
        except torch.OutOfMemoryError:
            log(f"mem_diag OOM in pass (autocast cache={cache}) — "
                "torch.cuda.memory_summary():\n" + torch.cuda.memory_summary(),
                print_console=True)

    try:
        optimizer.step()  # first call allocates the 8-bit states
        mem("after first optimizer step (states now live)")
    except torch.OutOfMemoryError:
        log("mem_diag OOM in optimizer step — torch.cuda.memory_summary():\n"
            + torch.cuda.memory_summary(), print_console=True)

    for h in handles:
        h.remove()
    log("mem_diag complete", print_console=True)


if __name__ == "__main__":
    main()
