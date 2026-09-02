"""Base-checkpoint export/import — the pre-initialized 9B artifact for HF.

The "base" is the model at step 0 of the v2.0 premise: donor furniture init
(warm start from the Llama-3.1-8B-Instruct checkpoint), gather identity
(zeroed o_proj/down_proj), all near-no-op inits (sink −10, AttnRes zero
queries, Engram zero U), and freshly initialized Engram tables
(Uniform(−0.01, 0.01) bf16, dedicated per-table generators).

Format (HF-idiomatic file layout; the architecture is custom, so stock
transformers cannot load it — the harness is the loader):

    <out>/
      config.json                    # our ModelConfig as JSON (not HF's schema)
      model-0000X-of-0000N.safetensors + model.safetensors.index.json
      engram.safetensors             # table rows, keys "n,k" (only if enabled)
      engram.json                    # canon sha256, moduli, scheme metadata
      README.md

Export dtype: bf16 is BITWISE SAFE — the donor weights are natively bf16 and
every new init constant is exactly representable, so a bf16 export round-
trips losslessly into the trainer's fp32 masters.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import torch
from safetensors.torch import load_file, save_file

from train.src.model.refit import RefitModel
from train.utils.log import log

SHARD_BYTES = 4 * 1024**3  # HF-style <=4 GiB shards


def save_base_checkpoint(
    model: RefitModel,
    out_dir: str | Path,
    dtype: torch.dtype = torch.bfloat16,
    note: str = "",
) -> Path:
    """Serialize model (sharded safetensors + index) + Engram tables."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = model.refit_config

    sd = {k: v.detach().to(dtype).cpu().contiguous() for k, v in model.state_dict().items()}

    # Pack tensors into <= SHARD_BYTES shards, in state_dict order.
    shards: list[dict[str, torch.Tensor]] = [dict()]
    weight_map: dict[str, str] = {}
    size = 0
    for name, t in sd.items():
        nbytes = t.numel() * t.element_size()
        if size + nbytes > SHARD_BYTES and shards[-1]:
            shards.append({})
            size = 0
        shards[-1][name] = t
        weight_map[name] = f"model-{len(shards):05d}-of-PLACEHOLDER.safetensors"
        size += nbytes
    n = len(shards)
    total = 0
    for i, shard in enumerate(shards, start=1):
        fname = f"model-{i:05d}-of-{n:05d}.safetensors"
        for name in shard:
            weight_map[name] = fname
        save_file(shard, str(out / fname), metadata={"format": "pt"})
        total += sum(t.numel() * t.element_size() for t in shard.values())
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=1)
        + "\n",
        encoding="utf-8",
    )
    (out / "config.json").write_text(
        json.dumps(cfg.to_dict(), indent=1) + "\n", encoding="utf-8"
    )

    if cfg.engram.enabled and model.engram_tables is not None:
        t = model.engram_tables
        rows = {f"{a},{b}": t.rows[(a, b)].contiguous() for (a, b) in t.table_keys}
        save_file(rows, str(out / "engram.safetensors"), metadata={"format": "pt"})
        (out / "engram.json").write_text(
            json.dumps(
                {
                    "canon_sha256": t.canon_checksum,
                    "moduli": {f"{a},{b}": m for (a, b), m in t.moduli.items()},
                    "scheme": "annex A1 v1.1",
                    "note": "rows bf16 host-resident; touch counters start at zero",
                },
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )

    (out / "README.md").write_text(
        "# Llama-9B base (untrained, step 0)\n\n"
        "Custom hybrid architecture (spec v2.1): NOT loadable by stock\n"
        "transformers. Load with `train.src.tools.base_ckpt.load_base_checkpoint`\n"
        "from the training harness (see repo AGENTS.md). Weights are bf16\n"
        "(bitwise-identical to the bf16 donor for donor-sourced params).\n"
        f"{note}\n",
        encoding="utf-8",
    )
    log(f"base checkpoint written: {out} ({n} model shards, "
        f"{total / 2**30:.1f} GiB)", print_console=True)
    return out


def load_base_checkpoint(model: RefitModel, path: str | Path) -> None:
    """Load a base export into an already-constructed RefitModel (any dtype —
    copy_ casts bf16 -> fp32 masters exactly). Verifies the config matches
    and, when Engram is enabled, the canon checksum before touching rows."""
    path = Path(path)
    exported = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if exported != model.refit_config.to_dict():
        raise ValueError(
            f"base checkpoint config mismatch: {path}/config.json was built from "
            "a different model config — refusing to load"
        )
    index = json.loads((path / "model.safetensors.index.json").read_text(encoding="utf-8"))
    state: dict[str, torch.Tensor] = {}
    for fname in sorted(set(index["weight_map"].values())):
        state.update(load_file(str(path / fname)))
    model.load_state_dict(state, strict=True)

    if model.refit_config.engram.enabled:
        tables = model.engram_tables
        meta = json.loads((path / "engram.json").read_text(encoding="utf-8"))
        if meta["canon_sha256"] != tables.canon_checksum:
            raise ValueError("base checkpoint engram canon sha256 mismatch")
        rows = load_file(str(path / "engram.safetensors"))
        for key in tables.table_keys:
            ks = f"{key[0]},{key[1]}"
            if rows[ks].shape != tables.rows[key].shape:
                raise ValueError(f"engram table {ks} shape mismatch in {path}")
            tables.rows[key].copy_(rows[ks])
    log(f"base checkpoint loaded: {path}", print_console=True)
