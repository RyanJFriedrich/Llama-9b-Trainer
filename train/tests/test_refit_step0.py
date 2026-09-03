"""Step-0 sanity + invariant tests, all at dev_tiny scale (AGENTS.md Hardware).

Spec v2.0: the old M2 donor-equivalence keystone is RETIRED (cold-start
premise — donor weights are furniture init, training starts at final
topology). What remains here:

- §3.8 step-0 sanity, toy-scale shape: finite loss in a sane band on a fixed
  probe batch. (The behavior-shaped band vs the real donor — loss well below
  uniform, KL-to-donor diagnostic — is a donor-scale check for the deploy
  box; at toy scale with random weights the correct value IS ~uniform.)
- Invariants I3/I4/I7 (AttnRes zero-init exactness, sink near-no-op, SWA
  offsets) and the gather identity init (spec §3.4).
- p-RoPE mechanics (spec §3.3): only the rotary slice rotates; globals and
  gather share one positional family.
- The anneal knobs (ablation tooling) are wired and move the output.

A tiny plain Llama plays the "donor" (same topology surgery, toy dims).
"""
import math

import torch
import torch.nn.functional as F

from train.src.config import ModelConfig, load_config
from train.src.model.attn_res import BlockAttnRes
from train.src.model.decoder import LlamaBaseConfig, LlamaBaseModel
from train.src.model.refit import RefitModel
from train.src.model.rope import PartialRotaryEmbedding, apply_partial_rotary_pos_emb
from train.src.tools.warm_start import warm_start_state_dict
from train.tests.test_parity_hf_llama import LLAMA3_SCALING
from train.utils.log import log

DEV_TINY = "train/configs/model/dev_tiny.yaml"
PROBE_SEED = 0xC0FFEE


def _tiny_donor(seed: int = 1234, std: float = 0.02) -> LlamaBaseModel:
    """An 8-layer plain Llama with dev_tiny's dims — the 'donor' for the
    9-layer model (8 block-layers + gather). std=0.02 keeps hidden magnitudes
    sane; the knob test uses a larger std so attention has signal
    (near-uniform attention at tiny init would mask RoPE changes)."""
    torch.manual_seed(seed)
    cfg = LlamaBaseConfig(
        vocab_size=1024, hidden_size=256, intermediate_size=512,
        num_hidden_layers=8, num_attention_heads=8, num_key_value_heads=4,
        head_dim=32, rms_norm_eps=1e-5, rope_theta=500000.0,
        rope_scaling=LLAMA3_SCALING, max_position_embeddings=8192,
    )
    model = LlamaBaseModel(cfg)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, std)
    return model.eval()


def _warm_refit(seed: int = 1234, std: float = 0.02, **cfg_overrides) -> tuple[LlamaBaseModel, RefitModel, dict]:
    donor = _tiny_donor(seed, std)
    d = load_config(DEV_TINY).to_dict()
    for section, kv in cfg_overrides.items():
        d[section].update(kv)
    refit = RefitModel(ModelConfig.from_dict(d))
    report = warm_start_state_dict(refit, donor.state_dict())
    return donor, refit.eval(), report


def _probe(vocab: int = 1024, batch: int = 2, seq_len: int = 64) -> torch.Tensor:
    g = torch.Generator().manual_seed(PROBE_SEED)
    return torch.randint(0, vocab, (batch, seq_len), generator=g)


def _ce_loss(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
        ids[:, 1:].reshape(-1),
    )


def test_step0_sanity():
    """Spec §3.8 (toy-scale shape): on a fixed probe batch the donor-init
    model at final topology yields finite logits and a loss in a sane band.
    With toy-random weights the correct band is ~uniform (ln 1024 ~ 6.931);
    the donor-scale 'well below uniform' band runs on the deploy box."""
    _, refit, _ = _warm_refit()
    ids = _probe()
    with torch.no_grad():
        logits = refit(ids)
    assert torch.isfinite(logits).all()
    loss = _ce_loss(logits, ids).item()
    uniform = math.log(1024)
    log(f"step0 sanity (toy): loss {loss:.6f}, uniform {uniform:.6f}", print_console=True)
    assert abs(loss - uniform) < 0.1, f"loss {loss} outside the sane band around {uniform}"
    # Default state is the final topology (spec v2.0 §3.4).
    assert refit.anneal_state == {"window": 32, "theta_progress": 1.0}


def test_each_knob_moves_loss():
    """Every anneal knob, moved off the final state, measurably moves the
    model's output (the knobs are wired, not decorative). Compares logits on
    the probe batch; std=0.1 init gives attention real signal (at std=0.02
    tiny-init attention is near-uniform and RoPE knobs register <1e-7).

    The theta knob only exists when the warm-start theta anneal is configured
    (ablation tooling), so that arm builds its model with the anneal block."""
    ids = _probe()

    def logits_of(**anneal):
        _, refit, _ = _warm_refit(std=0.1, swa={"rope_theta_warmstart_anneal":
                                                {"from": 500000.0, "schedule": "log_linear"}})
        refit.set_anneal_state(**anneal)
        with torch.no_grad():
            return refit(ids)

    base = logits_of(window=32, theta_progress=1.0)  # final state, explicit
    moved = {
        "window": (logits_of(window=16, theta_progress=1.0) - base).abs().max().item(),
        "theta": (logits_of(window=32, theta_progress=0.0) - base).abs().max().item(),
    }
    # Sink logits and AttnRes pseudo-queries are parameters, not anneal state.
    _, refit, _ = _warm_refit(std=0.1)
    with torch.no_grad():
        for layer in refit.model.layers:
            if layer.self_attn.sink_logit is not None:
                layer.self_attn.sink_logit.fill_(2.0)
        moved["sink"] = (refit(ids) - base).abs().max().item()

    _, refit, _ = _warm_refit(std=0.1)
    with torch.no_grad():
        g = torch.Generator().manual_seed(7)
        for ar in refit.model.attn_res:
            ar.pseudo_query.copy_(torch.randn(ar.pseudo_query.shape, generator=g) * 0.1)
        moved["attn_res"] = (refit(ids) - base).abs().max().item()

    log("knob movement (max |dlogits|): " +
        ", ".join(f"{k}={v:.3e}" for k, v in moved.items()), print_console=True)
    for name, d in moved.items():
        assert d > 1e-4, f"knob {name} did not move the model output (max |dlogits|={d})"


def test_swa_offset_invariant():
    """I7: SWA token-token relative offsets <= window - 1 < window; the sink
    carries no position; globals/gather have p-RoPE, not sinks."""
    _, refit, _ = _warm_refit()
    cfg = refit.refit_config
    seq_len, window = 96, cfg.swa.window  # 32

    mask = refit._mask_for("swa", window, seq_len, torch.float32, torch.device("cpu"))[0, 0]
    allowed = mask == 0
    i = torch.arange(seq_len).unsqueeze(1)
    j = torch.arange(seq_len).unsqueeze(0)
    offsets = (i - j)[allowed]
    assert offsets.min() == 0 and offsets.max() == window - 1 <= 4095

    for layer in refit.model.layers:
        attn = layer.self_attn
        if attn.layer_type == "swa":
            # One scalar per head: no position, no KV footprint.
            assert attn.sink_logit.shape == (cfg.num_attention_heads,)
        elif attn.layer_type in ("global", "gather"):
            assert attn.sink_logit is None
            assert isinstance(attn.rotary, PartialRotaryEmbedding)


def test_prope_rotates_only_the_slice():
    """p-RoPE (spec §3.3): dims >= rotary_dim pass through bitwise; on the
    rotated slice the standard RoPE relative-position property holds:
    (R_m q)·(R_n k) == (R_0 q)·(R_{n-m} k)."""
    head_dim, rd = 32, 8
    rope = PartialRotaryEmbedding(head_dim, fraction=0.25, theta=1000000.0)
    g = torch.Generator().manual_seed(0)
    T = 16
    q = torch.randn(1, 1, T, head_dim, generator=g)
    k = torch.randn(1, 1, T, head_dim, generator=g)
    cos, sin = rope(T)
    qr = apply_partial_rotary_pos_emb(q, cos, sin, rd)
    kr = apply_partial_rotary_pos_emb(k, cos, sin, rd)

    # Pass-through dims are bitwise unchanged.
    assert torch.equal(qr[..., rd:], q[..., rd:])
    assert torch.equal(kr[..., rd:], k[..., rd:])
    # The rotated slice actually rotates (position 0 is the identity angle).
    assert torch.equal(qr[0, 0, 0, :rd], q[0, 0, 0, :rd])
    assert not torch.allclose(qr[0, 0, 5, :rd], q[0, 0, 5, :rd])

    # Relative-position property: rotating the SAME content vectors at
    # positions (m, n) gives the same dot product as at (0, n-m):
    # (R_m Q)·(R_n K) == Q·(R_{n-m} K).
    m, n = 3, 11
    Q = torch.randn(1, 1, 1, head_dim, generator=g)
    K = torch.randn(1, 1, 1, head_dim, generator=g)
    qseq = torch.zeros(1, 1, T, head_dim)
    kseq = torch.zeros(1, 1, T, head_dim)
    qseq[0, 0, m] = Q[0, 0, 0]
    qseq[0, 0, 0] = Q[0, 0, 0]
    kseq[0, 0, n] = K[0, 0, 0]
    kseq[0, 0, n - m] = K[0, 0, 0]
    qs = apply_partial_rotary_pos_emb(qseq, cos, sin, rd)
    ks = apply_partial_rotary_pos_emb(kseq, cos, sin, rd)
    lhs = (qs[0, 0, m, :rd] * ks[0, 0, n, :rd]).sum()
    rhs = (qs[0, 0, 0, :rd] * ks[0, 0, n - m, :rd]).sum()
    assert torch.allclose(lhs, rhs, atol=1e-5)


def test_prope_uniform_family():
    """Spec §3.3: GLOBAL and GATHER layers share one positional family
    (identical inv_freq); SWA layers are a different, full-dim RoPE."""
    _, refit, _ = _warm_refit()
    cfg = refit.refit_config
    swa, prope = [], []
    for layer in refit.model.layers:
        attn = layer.self_attn
        if attn.layer_type == "swa":
            swa.append(attn.rotary)
        else:
            prope.append(attn.rotary)
    assert len(swa) == 6 and len(prope) == 3
    for r in prope:
        assert r.rotary_dim == cfg.head_dim // 4
        assert torch.equal(r.inv_freq, prope[0].inv_freq)
    for r in swa:
        assert r.inv_freq.shape == (cfg.head_dim // 2,)  # full-dim RoPE
    # Different families: the p-RoPE slice frequencies are computed over the
    # slice's own dim (NeoX convention), so they differ from the SWA head's.
    assert not torch.equal(prope[0].inv_freq, swa[0].inv_freq[: cfg.head_dim // 8])


def test_gather_identity_init():
    """The donor-init gather layer is an exact no-op: hidden stream bitwise
    unchanged across it (zeroed o_proj/down_proj produce exactly 0 deltas,
    regardless of its p-RoPE attention)."""
    _, refit, _ = _warm_refit()
    capture: dict = {}
    with torch.no_grad():
        refit(_probe(), capture=capture)
    gather_pos = refit.refit_config.gather.position  # 8
    assert torch.equal(capture["hiddens"][gather_pos - 1], capture["hiddens"][gather_pos])

    ga = refit.model.layers[gather_pos].self_attn
    assert torch.equal(ga.o_proj.weight, torch.zeros_like(ga.o_proj.weight))
    assert torch.equal(
        refit.model.layers[gather_pos].mlp.down_proj.weight,
        torch.zeros_like(refit.model.layers[gather_pos].mlp.down_proj.weight),
    )


def test_attnres_zero_init_uniform():
    """I3: at init, AttnRes weights are uniform over sources, and sources are
    delta-sums — never residual-stream snapshots (spec §3.5, load-bearing)."""
    # Direct module check: zero-init query -> uniform softmax -> mean of sources.
    ar = BlockAttnRes(hidden_size=8)
    sources = [torch.randn(2, 3, 8) for _ in range(4)]
    with torch.no_grad():
        out = ar(sources)
    assert torch.allclose(out, torch.stack(sources).mean(dim=0), atol=1e-6)

    # In-model check: at layer 4 pre-attn (first layer of block 2), sources
    # must be [embedding, sum_of_block1_deltas, partial=0]. If sources were
    # snapshots, source 1 would equal the stream after layer 3; as a
    # delta-sum it equals stream_minus_embedding.
    _, refit, _ = _warm_refit()
    capture: dict = {}
    with torch.no_grad():
        refit(_probe(), capture=capture)
    e = capture["embedding"]
    h3 = capture["hiddens"][3]  # stream after the block-1 global layer
    v0, v1, v2 = capture["attn_res_sources"][(4, "pre_attn")]
    assert torch.equal(v0, e)                       # blocks[0] = embedding output
    assert torch.allclose(v1, h3 - e, atol=1e-6)    # delta-sum of block 1
    assert not torch.allclose(v1, h3, atol=1e-3)    # ... which is NOT a snapshot
    assert torch.equal(v2, torch.zeros_like(v2))    # partial resets at block boundary

    # 2 blocks + embedding + partial = 4 sources max at dev_tiny scale
    # (full model: 8 blocks + embedding + partial = 10).
    n_sources = len(capture["attn_res_sources"][(8, "pre_attn")])  # gather layer
    assert n_sources == 4


def test_warmstart_accounting():
    """Every donor tensor consumed; new params are exactly the known set."""
    donor, refit, report = _warm_refit()
    assert report["donor_consumed"] == report["donor_tensors"] == len(donor.state_dict())
    new = report["new_refit_params"]
    assert new, "expected new params (sink logits + AttnRes)"
    for k in new:
        assert ("sink_logit" in k) or ("attn_res" in k), f"unexpected new param {k}"
    # 6 SWA layers -> 6 sink params; 9 layers x 2 points -> 18 AttnRes points
    # x (pseudo_query + key_norm.weight) = 36.
    n_sink = sum("sink_logit" in k for k in new)
    n_ar = sum("attn_res" in k for k in new)
    assert n_sink == 6 and n_ar == 36
    # Donor weights actually landed: layer 0 q_proj matches the donor's.
    assert torch.equal(
        refit.model.layers[0].self_attn.q_proj.weight,
        donor.model.layers[0].self_attn.q_proj.weight,
    )


def test_warmstart_accounting_with_engram_enabled():
    """Engram readout params are legitimate new params at LOCKED start values
    (zero U, unit gates/norms): warm start must whitelist them, not fail."""
    _, refit, report = _warm_refit(engram={"enabled": True})
    new = report["new_refit_params"]
    assert any(k.startswith("engram.") for k in new)
    for k in new:
        assert ("sink_logit" in k) or ("attn_res" in k) or k.startswith("engram."), k
    # Readout untouched by warm start: U still zero (I1), gates still 1.
    assert all(int(p.abs().sum()) == 0 for p in refit.engram.proj.parameters())
    assert all(float(p) == 1.0 for p in refit.engram.gates.parameters())


def test_engram_enabled_builds_with_zero_injection():
    """The Engram module is landed: enabled=true builds host tables + device
    readout, and zero-init U (I1) makes the injection exactly zero at init —
    logits bitwise-match the no-Engram model with the same weights."""
    d = load_config(DEV_TINY).to_dict()
    d["engram"]["enabled"] = True
    model = RefitModel(ModelConfig.from_dict(d))
    assert model.engram is not None and model.engram_tables is not None
    assert all(
        int(p.sum().abs()) == 0 for p in model.engram.proj.parameters()
    )  # I1: U zero-init

    base = RefitModel(ModelConfig.from_dict(load_config(DEV_TINY).to_dict()))
    base.load_state_dict(
        {k: v for k, v in model.state_dict().items() if not k.startswith("engram.")},
        strict=False,
    )
    x = torch.randint(0, model.refit_config.vocab_size, (2, 16))
    assert torch.equal(model(x), base(x))


def test_chunked_attention_matches_unchunked():
    """attn_query_chunk splits the [H, T, S] score matrix over query blocks
    without changing the math: chunked and unchunked dev_tiny models agree on
    logits and on grads (CPU fp32). The backward pass exercises the per-chunk
    gradient-checkpoint path (the memory fix that makes T=8192 fit the 96 GB
    box)."""
    d = load_config(DEV_TINY).to_dict()
    torch.manual_seed(0)
    m_full = RefitModel(ModelConfig.from_dict({**d, "attn_query_chunk": 0}))
    torch.manual_seed(0)
    m_chunk = RefitModel(ModelConfig.from_dict({**d, "attn_query_chunk": 16}))

    ids = torch.randint(0, d["vocab_size"], (2, 64))
    out_full = m_full(ids)
    out_chunk = m_chunk(ids)
    assert torch.allclose(out_full, out_chunk, atol=1e-4, rtol=1e-4), (
        f"logits differ: {(out_full - out_chunk).abs().max()}"
    )

    m_full(ids).sum().backward()
    m_chunk(ids).sum().backward()
    p_chunk = dict(m_chunk.named_parameters())
    for name, p in m_full.named_parameters():
        if p.grad is None:
            continue
        assert torch.allclose(p.grad, p_chunk[name].grad, atol=1e-4, rtol=1e-3), name
