"""The dev-tiny config: a chopped-down v2.0 §3.1 variant for fast unit tests
on the local RTX 4090 (see AGENTS.md "Hardware"). It must stay loadable,
valid, and structurally analogous to the full 33-layer default so the model
unit tests exercise the real machinery at toy scale."""
from pathlib import Path

from train.src.config import load_config

DEV_TINY = Path(__file__).parents[1] / "configs" / "model" / "dev_tiny.yaml"


def test_dev_tiny_loads_and_validates():
    cfg = load_config(DEV_TINY)
    assert cfg.num_hidden_layers == 9
    assert cfg.layer_types == ["swa", "swa", "swa", "global"] * 2 + ["gather"]
    assert cfg.gather.position == 8
    # Two blocks + embedding + current partial = 4 AttnRes sources max.
    # (Full model: 8 blocks + embedding + partial = 10.)
    assert cfg.attn_res.enabled and cfg.attn_res.scope == "all_layers"
    assert cfg.swa.window == 32
    assert cfg.swa.rope_theta == 10000.0
    assert cfg.swa.rope_theta_warmstart_anneal is None  # final topology by default
    assert cfg.global_.rope_type == "prope" and cfg.global_.rope_fraction == 0.25
    assert cfg.gather.rope_type == "prope"
    assert cfg.engram.enabled is False


def test_dev_tiny_round_trip(tmp_path):
    cfg = load_config(DEV_TINY)
    out = tmp_path / "rt.json"
    out.write_text(cfg.to_json())
    assert load_config(out).to_dict() == cfg.to_dict()
