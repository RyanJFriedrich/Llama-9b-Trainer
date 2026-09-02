"""Build the pre-initialized base checkpoint for HF upload (see
train/src/tools/base_ckpt.py for the format).

Usage (from repo root):

    python -m train.scripts.build_base_checkpoint \
        --model-config train/configs/model/llama_9b_engram_v1.yaml \
        --init warm --donor OriginalModel \
        --out exports/llama-9b-base-v1

Runs on CPU: builds the model directly in bf16 (halves host RAM; bitwise-
identical to the bf16 donor for donor-sourced params) and never touches the
GPU. ~35 GB peak host RAM at prod scale with warm init.
"""
from __future__ import annotations

import argparse

import torch

from train.src.config import load_config
from train.src.model.refit import RefitModel
from train.src.tools.base_ckpt import save_base_checkpoint
from train.utils.log import log


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-config", default="train/configs/model/llama_9b_engram_v1.yaml")
    p.add_argument("--init", default="warm", choices=["warm", "scratch"])
    p.add_argument("--donor", default="OriginalModel")
    p.add_argument("--out", default="exports/llama-9b-base-v1")
    args = p.parse_args()

    cfg = load_config(args.model_config)
    cfg.validate()
    log(f"build_base_checkpoint: {args.model_config} init={args.init} -> {args.out}",
        print_console=True)

    torch.set_default_dtype(torch.bfloat16)  # export dtype from construction
    torch.manual_seed(0)
    model = RefitModel(cfg)
    if args.init == "warm":
        from train.src.tools.warm_start import warm_start_from_checkpoint

        report = warm_start_from_checkpoint(model, args.donor)
        log(f"warm start: {report['donor_consumed']}/{report['donor_tensors']} donor "
            f"tensors, {len(report['new_refit_params'])} new params", print_console=True)

    save_base_checkpoint(model, args.out, dtype=torch.bfloat16,
                         note=f"init={args.init}, donor={args.donor}")
    log("done", print_console=True)


if __name__ == "__main__":
    main()
