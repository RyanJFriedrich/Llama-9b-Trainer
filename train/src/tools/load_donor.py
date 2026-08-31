"""Donor checkpoint loader (M1).

Loads an HF-format Llama checkpoint (safetensors shards + index) into
`LlamaBaseModel` with an identity key mapping, and reports every donor
tensor as accounted for (M1 acceptance: "loader reports every donor tensor
accounted for"). Anything missing or unexpected is a hard error.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import torch
from safetensors.torch import load_file

from train.src.model.decoder import LlamaBaseConfig, LlamaBaseModel
from train.utils.log import log


def load_donor_state_dict(checkpoint_dir: Union[str, Path]) -> dict[str, torch.Tensor]:
    """Read all safetensors shards of an HF checkpoint into one state dict."""
    checkpoint_dir = Path(checkpoint_dir)
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shard_files = sorted(set(index["weight_map"].values()))
    else:
        shard_files = sorted(p.name for p in checkpoint_dir.glob("model*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"no safetensors shards found in {checkpoint_dir}")

    state_dict: dict[str, torch.Tensor] = {}
    for shard in shard_files:
        state_dict.update(load_file(checkpoint_dir / shard))
    return state_dict


def load_donor(
    checkpoint_dir: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    log_filename: str = "common.log",
) -> LlamaBaseModel:
    """Build a LlamaBaseModel from an HF checkpoint dir, fully accounted.

    Raises if any donor tensor is unused or any model tensor is unfilled.
    """
    from transformers import AutoConfig

    checkpoint_dir = Path(checkpoint_dir)
    hf_config = AutoConfig.from_pretrained(checkpoint_dir)
    config = LlamaBaseConfig.from_hf(hf_config)
    log(
        f"load_donor: {checkpoint_dir} (_name_or_path={getattr(hf_config, '_name_or_path', '?')}), "
        f"layers={config.num_hidden_layers} hidden={config.hidden_size} "
        f"heads={config.num_attention_heads}/{config.num_key_value_heads} "
        f"rope_theta={config.rope_theta} scaling={config.rope_scaling}",
        filename=log_filename,
    )

    state_dict = load_donor_state_dict(checkpoint_dir)
    n_donor = len(state_dict)

    model = LlamaBaseModel(config)
    state_dict = {k: v.to(dtype) for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if config.tie_word_embeddings:
        # Tied checkpoints legitimately omit lm_head.weight (shared parameter).
        missing = [k for k in missing if k != "lm_head.weight"]
    if unexpected or missing:
        raise ValueError(
            f"donor tensor accounting failed: "
            f"{len(unexpected)} unused donor tensors {sorted(unexpected)[:5]}..., "
            f"{len(missing)} unfilled model tensors {sorted(missing)[:5]}..."
        )

    n_model = len(model.state_dict())
    log(
        f"load_donor: {n_donor} donor tensors loaded, {n_model} model tensors filled, "
        f"0 missing / 0 unexpected — every donor tensor accounted for",
        filename=log_filename,
    )
    return model.to(device=device, dtype=dtype)
