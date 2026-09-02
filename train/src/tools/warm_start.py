"""Warm-start loader: donor state dict -> refit model (M2, spec §5).

Mapping is 1:1 for the donor's layers, embeddings, and lm_head — attention
type changes touch no weights. The gather layer is the only new transformer
layer: it receives a copy of the donor's FINAL layer with o_proj and
down_proj zeroed (LLaMA-Pro style identity init) so it is an exact no-op at
step 0. All other new parameters (sink logits, AttnRes pseudo-queries and
key norms) are already at their LOCKED start values by construction.

Accounting (M2 acceptance): every donor tensor must be consumed; every
non-donor refit parameter must be on the known-new whitelist. Surprises in
either direction are hard errors.
"""
from __future__ import annotations

from typing import Union

import torch

from train.src.model.refit import RefitModel
from train.src.tools.load_donor import load_donor_state_dict
from train.utils.log import log

# Parameter name fragments that are legitimately new in the refit (not donor).
# engram.: the §3.6 sidecar readout (zero-init U, unit gates/norms) — LOCKED
# start values by construction; the tables are host-side, not in state_dict.
NEW_PARAM_FRAGMENTS = ("sink_logit", "attn_res", "engram.")


def warm_start_state_dict(refit: RefitModel, donor_state: dict[str, torch.Tensor]) -> dict:
    """Load donor weights into the refit model. Returns an accounting report.

    donor_state: HF-style flat state dict (model.layers.N.*, ...), e.g. from
    load_donor_state_dict(). The donor's layer count is inferred from it.
    """
    cfg = refit.refit_config
    donor_layers = {
        int(k.split(".")[2]) for k in donor_state if k.startswith("model.layers.")
    }
    n_donor = max(donor_layers) + 1
    gather_pos = cfg.gather.position
    n_refit = len(refit.layer_types)
    if n_refit != n_donor + 1:
        raise ValueError(f"refit has {n_refit} layers, donor {n_donor} — expected donor + 1 (gather)")

    # Non-gather refit layers map to donor layers in order; the gather layer
    # copies the donor's final layer.
    mapping: dict[int, int] = {}
    donor_idx = 0
    for i, t in enumerate(refit.layer_types):
        if t == "gather":
            mapping[i] = n_donor - 1
        else:
            mapping[i] = donor_idx
            donor_idx += 1
    if donor_idx != n_donor:
        raise ValueError(f"mapped {donor_idx} non-gather layers but donor has {n_donor}")

    new_state: dict[str, torch.Tensor] = {}
    consumed: set[str] = set()
    for refit_i, donor_i in mapping.items():
        for name, tensor in donor_state.items():
            if name.startswith(f"model.layers.{donor_i}."):
                target = f"model.layers.{refit_i}." + name.split(f"model.layers.{donor_i}.", 1)[1]
                new_state[target] = tensor.clone()
                consumed.add(name)
    for name in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
        if name in donor_state:
            new_state[name] = donor_state[name].clone()
            consumed.add(name)

    missing, unexpected = refit.load_state_dict(new_state, strict=False)

    # Gather identity init: zero o_proj and down_proj of the gather layer.
    gather_prefix = f"model.layers.{gather_pos}."
    for key in (gather_prefix + "self_attn.o_proj.weight", gather_prefix + "mlp.down_proj.weight"):
        refit.state_dict()[key].zero_()
    # Reflect the zeroing in what we consider loaded (missing check used the
    # pre-zero state, which is fine — the keys exist either way).

    leftover_donor = sorted(set(donor_state) - consumed)
    if leftover_donor and not (
        cfg.tie_word_embeddings and leftover_donor == ["lm_head.weight"]
    ):
        raise ValueError(f"donor tensors not consumed by warm start: {leftover_donor[:5]}...")

    bad_missing = [
        k for k in missing
        if not any(frag in k for frag in NEW_PARAM_FRAGMENTS)
    ]
    if bad_missing:
        raise ValueError(f"refit tensors left unfilled by warm start: {bad_missing[:5]}...")
    if unexpected:
        raise ValueError(f"unexpected keys in mapped state: {unexpected[:5]}...")

    new_params = sorted(
        k for k in refit.state_dict()
        if any(frag in k for frag in NEW_PARAM_FRAGMENTS)
    )
    report = {
        "donor_tensors": len(donor_state),
        "donor_consumed": len(consumed),
        "new_refit_params": new_params,
        "gather_zeroed": [
            gather_prefix + "self_attn.o_proj.weight",
            gather_prefix + "mlp.down_proj.weight",
        ],
    }
    log(
        f"warm_start: {len(consumed)}/{len(donor_state)} donor tensors consumed; "
        f"{len(new_params)} new refit params (sink logits + AttnRes) at LOCKED start "
        f"values; gather layer {gather_pos} identity-initialized "
        f"(o_proj/down_proj zeroed)",
        print_console=True,
    )
    return report


def warm_start_from_checkpoint(
    refit: RefitModel,
    checkpoint_dir: Union[str, "Path"],
) -> dict:
    """Convenience: load donor shards from a directory and warm start."""
    return warm_start_state_dict(refit, load_donor_state_dict(checkpoint_dir))
