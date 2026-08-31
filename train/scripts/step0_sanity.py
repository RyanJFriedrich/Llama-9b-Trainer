"""Donor-scale step-0 sanity (spec §3.8) — deploy box, before the first run.

With donor furniture init and all invariants holding, a fixed probe batch
must yield gold CE well below uniform (ln 128,256 ≈ 11.76 nats) and in a
sane band for a windowed/p-RoPE-perturbed donor. KL-to-donor on the same
probe set is logged as a topology-shift DIAGNOSTIC — not a gate (v2.0).

Usage (from repo root, on the deploy box):

    python -m train.scripts.step0_sanity
    python -m train.scripts.step0_sanity --model-config <engram-enabled yaml>
    python -m train.scripts.step0_sanity --skip-kl   # loss band only

The probe batch is built-in fixed text tokenized by the donor tokenizer —
deterministic, no shard dependency, same numbers on every box.
"""
from __future__ import annotations

import argparse
import math

import torch

from train.src.config import load_config
from train.src.model.refit import RefitModel
from train.src.tools.warm_start import warm_start_from_checkpoint
from train.utils.log import log

# Fixed probe texts (spec §3.8 "fixed probe batch"). Ordinary English prose;
# the donor should be far below uniform on these.
PROBE_TEXTS = [
    "The city of Paris is the capital of France. It sits on the river Seine "
    "and has been a major centre of art, science, and commerce for centuries. "
    "The Eiffel Tower, completed in 1889, remains its most famous landmark.",
    "To make bread, combine flour, water, salt, and yeast. Knead the dough "
    "until it is smooth and elastic, then leave it to rise in a warm place "
    "for about an hour. Bake in a hot oven until the crust is golden brown.",
    "The transformer architecture processes sequences using attention "
    "mechanisms. Each layer applies multi-head self-attention followed by a "
    "feed-forward network, with residual connections and normalization "
    "around each sublayer.",
]

# "Well below uniform" (§3.8): conservative PASS line at 3/4 of uniform.
# A windowed/perturbed donor on real prose should sit near 2-4 nats.
UNIFORM_MARGIN = 0.75


@torch.no_grad()
def gold_ce(model: RefitModel, ids: torch.Tensor, chunk: int = 128) -> float:
    """Mean next-token CE over the probe doc, chunked so the full-vocab
    logit tensor never persists (standing rule 3)."""
    W = model.lm_head.weight.to(torch.float32)
    h = model(ids, return_hidden=True).to(torch.float32)  # [1, T, d] — small
    total, n = 0.0, 0
    for s in range(0, ids.shape[1] - 1, chunk):
        e = min(s + chunk, ids.shape[1] - 1)
        logits = h[0, s:e] @ W.T
        gold = ids[0, s + 1 : e + 1]
        total += torch.nn.functional.cross_entropy(logits, gold, reduction="sum").item()
        n += e - s
    return total / max(n, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-config", default="train/configs/model/llama_8bpp_v1.yaml")
    p.add_argument("--donor", default="OriginalModel")
    p.add_argument("--max-positions", type=int, default=512)
    p.add_argument("--skip-kl", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_cfg = load_config(args.model_config)
    uniform = math.log(model_cfg.vocab_size)
    log(f"step0_sanity: config={args.model_config} device={device} "
        f"uniform={uniform:.3f} nats", print_console=True)

    torch.manual_seed(0)
    model = RefitModel(model_cfg).to(device).eval()
    report = warm_start_from_checkpoint(model, args.donor)
    log(f"warm start: {report['donor_consumed']}/{report['donor_tensors']} donor "
        f"tensors, {len(report['new_refit_params'])} new params", print_console=True)
    if model_cfg.engram.enabled:
        zero_u = all(
            int(p.abs().sum()) == 0 for p in model.engram.proj.parameters()
        )
        log(f"engram enabled; zero-init U (I1) at step 0: {zero_u}",
            print_console=True)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.donor)
    docs = [
        tok(t, add_special_tokens=True)["input_ids"][: args.max_positions]
        for t in PROBE_TEXTS
    ]

    losses = []
    for i, d in enumerate(docs):
        di = torch.tensor([d], dtype=torch.long, device=device)
        ce = gold_ce(model, di)
        losses.append(ce)
        log(f"probe[{i}] T={len(d)} CE={ce:.3f} nats", print_console=True)
    mean_ce = sum(losses) / len(losses)
    ok = mean_ce < UNIFORM_MARGIN * uniform
    log(f"step0_sanity: mean CE {mean_ce:.3f} vs uniform {uniform:.3f} "
        f"(pass line {UNIFORM_MARGIN * uniform:.3f}) -> {'PASS' if ok else 'FAIL'}",
        print_console=True)

    if not args.skip_kl:
        from train.src.eval.kl_donor import kl_to_donor
        from train.src.tools.load_donor import load_donor

        donor = load_donor(args.donor, device=device, dtype=torch.bfloat16)
        kl = kl_to_donor(model, donor, docs, device=device,
                         max_len=args.max_positions)
        log(f"step0_sanity: KL(student||donor) = {kl:.4f} nats/position "
            f"(diagnostic, not a gate)", print_console=True)

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
