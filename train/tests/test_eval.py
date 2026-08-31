"""Eval tooling plumbing tests (M4/§8): perplexity, attention probes, NIAH.
These verify the harness at dev_tiny scale — capability numbers come from
the trained 8B on the deploy box."""
import torch

from train.src.config import load_config
from train.src.eval.attn_probes import attention_stats
from train.src.eval.niah import needle_accuracy
from train.src.eval.perplexity import perplexity
from train.src.model.refit import RefitModel

DEV_TINY = "train/configs/model/dev_tiny.yaml"


def _model():
    torch.manual_seed(5)
    return RefitModel(load_config(DEV_TINY)).eval()


def test_perplexity_finite_and_sane():
    model = _model()
    ppl = perplexity(model, [list(range(1, 40)), [5, 6, 7] * 10], device="cpu")
    assert ppl > 1.0
    # Random tiny model: ppl should be in the vicinity of vocab size.
    assert ppl < 100 * 1024


def test_attention_stats_structure_and_invariants():
    model = _model()
    stats = attention_stats(model, [list(range(1, 96))], device="cpu", beyond=32)
    assert len(stats) == 9
    for s in stats:
        assert s["entropy"] >= 0.0
        assert 0.0 <= s["mass_beyond"] <= 1.0
    # Default state is the final topology: window=32, so SWA mass-beyond-32
    # is exactly 0 (I7 runtime probe). The full-window ablation (window=None)
    # lets SWA layers attend far — sanity that the probe sees the difference.
    for s in stats:
        if s["layer_type"] == "swa":
            assert s["mass_beyond"] == 0.0, f"SWA layer {s['index']} attends beyond window"
    model.set_anneal_state(window=None, theta_progress=1.0)
    stats = attention_stats(model, [list(range(1, 96))], device="cpu", beyond=32)
    assert any(s["mass_beyond"] > 0.0 for s in stats if s["layer_type"] == "swa")


def test_probe_hook_removed_after_use():
    model = _model()
    attention_stats(model, [list(range(1, 40))], device="cpu", beyond=32)
    assert all(l.self_attn.probe is None for l in model.model.layers)


def test_niah_plumbing():
    model = _model()
    res = needle_accuracy(model, lengths=[64], depths=[0.5], n_distractors=3,
                          vocab_size=1024, key_id=500, value_id=900,
                          query_id=600, filler_id=3, device="cpu")
    assert list(res.keys()) == [(64, 0.5)]
    assert isinstance(res[(64, 0.5)], bool)  # random model: either way; plumbing only
