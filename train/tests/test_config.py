"""M0 acceptance tests: spec v2.0 §3.1 skeleton round-trip + every spec flag parses."""
import json
from pathlib import Path

import pytest

from train.src.config import ModelConfig, load_config

FIXTURE = Path(__file__).parent / "fixtures" / "spec_3_1_reference.json"
DEFAULT_YAML = Path(__file__).parents[1] / "configs" / "model" / "llama_8bpp_v1.yaml"


def test_spec_3_1_json_round_trip_unchanged():
    """M0 acceptance: the spec §3.1 skeleton JSON survives from_dict -> to_dict exactly."""
    spec = json.loads(FIXTURE.read_text(encoding="utf-8"))
    cfg = ModelConfig.from_dict(spec)
    assert cfg.to_dict() == spec
    # And a JSON string round-trip for good measure.
    assert json.loads(cfg.to_json()) == spec


def test_default_yaml_matches_spec_reference():
    """The shipped default config must be exactly the §3.1 reference skeleton."""
    spec = json.loads(FIXTURE.read_text(encoding="utf-8"))
    cfg = load_config(DEFAULT_YAML)
    assert cfg.to_dict() == spec


def test_default_layer_layout():
    """§3.1: 8 blocks of [swa,swa,swa,global] + final gather at index 32."""
    cfg = ModelConfig()
    lt = cfg.layer_types
    assert len(lt) == 33
    assert lt[:32] == ["swa", "swa", "swa", "global"] * 8
    assert lt[32] == "gather"
    assert cfg.gather.position == 32


def test_final_state_defaults():
    """Spec v2.0 §3.4/§7.2: training starts at FINAL topology. The locked
    near-no-op inits are the defaults; no anneal schedules are configured."""
    cfg = ModelConfig()
    assert cfg.swa.sink.init == -10.0                      # sink logits at -10 (I4)
    assert cfg.gather.init == "identity_from_layer_31"     # gather identity init
    assert cfg.attn_res.pseudo_query_init == "zeros"       # AttnRes zero-init (I3)
    # Final positional topology (spec §3.2/§3.3), no anneals by default.
    assert cfg.swa.window == 4096 and cfg.swa.rope_theta == 10000.0
    assert cfg.swa.rope_theta_warmstart_anneal is None
    assert cfg.swa.window_anneal is None
    for sub in (cfg.global_, cfg.gather):
        assert sub.rope_type == "prope"
        assert sub.rope_fraction == 0.25
        assert sub.rope_theta == 1000000.0
    # Engram schema present (spec §3.6 + annex A1), disabled by default.
    assert cfg.engram.enabled is False
    assert cfg.engram.orders == [1, 2, 3]
    assert cfg.engram.rows_per_head == {2: [1048573, 1048571], 3: [1048559, 1048549]}
    assert cfg.engram.injection_point == 3
    assert cfg.engram.canonical_compression is True
    assert cfg.engram.lr_mult == 5.0


def test_every_spec_flag_parses():
    """All §3.1/§7.2 knobs — including FLEXIBLE/EXPERIMENTAL ones — parse
    and re-serialize."""
    d = json.loads(FIXTURE.read_text(encoding="utf-8"))
    d["swa"]["window_anneal"] = {"from": "full", "schedule": "linear"}
    d["swa"]["rope_theta_warmstart_anneal"] = {"from": 500000.0, "schedule": "log_linear"}
    d["attn_res"]["scope"] = "globals_only"
    d["attn_res"]["gate"] = True
    d["gather"]["position"] = 32  # ablation value 30 would mismatch layer_types
    d["engram"]["enabled"] = True  # schema parses; module landed (see test_engram_*)
    cfg = ModelConfig.from_dict(d)
    assert cfg.to_dict() == d
    assert cfg.swa.window_anneal.start == "full"
    assert cfg.swa.rope_theta_warmstart_anneal.start == 500000.0
    assert cfg.attn_res.scope == "globals_only"


def test_ablation_arms_are_config_diffs():
    """Spec §10 ablation arms: expressible as dict diffs, no code changes."""
    base = json.loads(FIXTURE.read_text(encoding="utf-8"))

    full_rope_global = json.loads(FIXTURE.read_text())
    full_rope_global["global"]["rope_fraction"] = 1.0  # full-RoPE global variant (§10 sweep)
    assert ModelConfig.from_dict(full_rope_global).global_.rope_fraction == 1.0

    theta_anneal = json.loads(FIXTURE.read_text())
    theta_anneal["swa"]["rope_theta_warmstart_anneal"] = {"from": 500000.0}
    cfg = ModelConfig.from_dict(theta_anneal)
    assert cfg.swa.rope_theta_warmstart_anneal is not None

    no_attnres = json.loads(FIXTURE.read_text())
    no_attnres["attn_res"]["enabled"] = False
    assert ModelConfig.from_dict(no_attnres).attn_res.enabled is False

    tied = json.loads(FIXTURE.read_text())
    tied["tie_word_embeddings"] = True
    assert ModelConfig.from_dict(tied).tie_word_embeddings is True

    # Sanity: untouched base still parses.
    assert ModelConfig.from_dict(base).to_dict() == base


def test_unknown_keys_rejected():
    d = json.loads(FIXTURE.read_text())
    d["yarn_scaling"] = {"factor": 8}  # spec §10: long-context extension is deferred
    with pytest.raises(ValueError, match="unknown config keys"):
        ModelConfig.from_dict(d)


def test_shipped_engram_and_bringup_configs_parse():
    """The Engram-enabled prod model config and the bring-up smoke run config
    are shipped loadable (regression guard for hand-edited yamls)."""
    from train.src.config import load_train_config

    model = load_config(str(DEFAULT_YAML.parent / "llama_9b_engram_v1.yaml"))
    model.validate()
    assert model.engram.enabled
    assert model.engram.canon_sha256.startswith("03833370")
    assert model.engram.rows_per_head == {2: [1048573, 1048571], 3: [1048559, 1048549]}

    run = load_train_config(str(DEFAULT_YAML.parents[1] / "bringup_bf16_smoke.yaml"))
    assert run.bf16 and run.optimizer == "adamw8bit"
    assert run.seq_len == 8192 and run.init == "warm"

def test_validation_catches_bad_layouts():
    d = json.loads(FIXTURE.read_text())
    d["layer_types"][0] = "mamba"
    with pytest.raises(ValueError, match="invalid entries"):
        ModelConfig.from_dict(d)

    d = json.loads(FIXTURE.read_text())
    d["num_hidden_layers"] = 32
    with pytest.raises(ValueError, match="layer_types has 33"):
        ModelConfig.from_dict(d)

    d = json.loads(FIXTURE.read_text())
    d["gather"]["position"] = 30  # flag exists, but layer_types wasn't moved
    with pytest.raises(ValueError, match="does not match"):
        ModelConfig.from_dict(d)

    d = json.loads(FIXTURE.read_text())
    d["global"]["qk_norm"] = True  # LOCKED off (spec §3.2)
    with pytest.raises(ValueError, match="QK-norm"):
        ModelConfig.from_dict(d)

    d = json.loads(FIXTURE.read_text())
    d["global"]["rope_fraction"] = 0.3  # 128*0.3 = 38.4: not an integer slice
    with pytest.raises(ValueError, match="rope_fraction"):
        ModelConfig.from_dict(d)

    d = json.loads(FIXTURE.read_text())
    d["engram"]["rows_per_head"] = {"2": [1048573]}  # one head, heads_per_order is 2
    with pytest.raises(ValueError, match="heads_per_order"):
        ModelConfig.from_dict(d)

    d = json.loads(FIXTURE.read_text())
    d["engram"]["rows_per_head"]["2"] = [1048576, 1048571]  # 1048576 not prime
    with pytest.raises(ValueError, match="prime"):
        ModelConfig.from_dict(d)
