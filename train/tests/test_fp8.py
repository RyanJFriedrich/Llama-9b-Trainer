"""FP8 compute path tests (spec §4): quantization semantics, emulated parity
vs bf16 at dev_tiny scale (CPU exercises the exact production quantization
via the emulation path), apply_fp8 scope/state-dict stability, trainer
overlay smoke, and the real _scaled_mm path when a capable GPU is present."""
import numpy as np
import pytest
import torch

from train.src.config import ModelConfig, TrainConfig, load_config
from train.src.model.refit import RefitModel
from train.src.train.fp8 import (
    E4M3, E4M3_MAX, FP8Linear, _dequantize, _quantize, apply_fp8,
    fp8_gemm_available,
)

DEV_TINY = "train/configs/model/dev_tiny.yaml"


def test_quantize_roundtrip_bounded_by_fp8_grid():
    """e4m3 has 3 mantissa bits: relative error <= 2^-4 per element."""
    torch.manual_seed(0)
    x = torch.randn(4096) * torch.logspace(-3, 3, 4096)
    q, s = _quantize(x, E4M3, E4M3_MAX)
    back = q.float() * s
    # Per-tensor scaling keys the grid to amax: elements far below amax
    # quantize to zero by design (absolute error ~ amax/448), so relative
    # error is only bounded where the element is a meaningful fraction of
    # amax. Below that, check the absolute bound instead.
    amax = x.abs().max()
    big = x.abs() > amax * 0.01
    rel = (back - x).abs()[big] / x.abs()[big]
    assert rel.max() <= 0.0626
    # Worst-case absolute error is the grid step at amax: 2^-4 * amax.
    assert (back - x).abs().max() <= amax * 0.0626
    assert s.dtype == torch.float32 and s.ndim == 0


def test_fp8linear_emulated_parity_cpu():
    """Emulated FP8 GEMM tracks the bf16 GEMM within fp8 tolerance, and
    grads flow to both input and weight."""
    torch.manual_seed(0)
    lin = FP8Linear(256, 512, bias=False)
    x = torch.randn(4, 256, requires_grad=True)
    out = lin(x)
    ref = torch.nn.functional.linear(x, lin.weight.to(x.dtype))
    rel = (out.float() - ref.float()).abs().max() / ref.abs().max()
    assert rel < 0.15, f"emulated fp8 vs bf16 rel err {float(rel):.4f}"
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert lin.weight.grad is not None and torch.isfinite(lin.weight.grad).all()
    assert lin.weight.grad.dtype == lin.weight.dtype  # fp32 masters get fp32 grads


def test_apply_fp8_scope_and_state_dict_stability():
    """Exactly the attention+FFN GEMMs swap (7 per layer); lm_head, embed,
    AttnRes, and Engram readout are untouched; state_dict keys are unchanged
    so warm start and checkpoints are format-stable."""
    d = load_config(DEV_TINY).to_dict()
    d["engram"]["enabled"] = True
    model = RefitModel(ModelConfig.from_dict(d))
    keys_before = set(model.state_dict())
    lm_head_w = model.lm_head.weight
    engram_u = model.engram.proj["1"].weight

    n = apply_fp8(model)
    assert n == 9 * 7  # 9 layers x (q,k,v,o + gate,up,down)
    assert set(model.state_dict()) == keys_before
    assert model.lm_head.weight is lm_head_w
    assert not isinstance(model.lm_head, FP8Linear)
    assert model.engram.proj["1"].weight is engram_u
    assert not isinstance(model.engram.proj["1"], FP8Linear)
    assert isinstance(model.model.layers[0].self_attn.q_proj, FP8Linear)
    assert isinstance(model.model.layers[0].mlp.gate_proj, FP8Linear)
    with pytest.raises(ValueError, match="twice"):
        apply_fp8(model)


def _learnable_shard(root, name, n_docs=8, t=128, k=4, vocab=1024, seed=3):
    from train.src.data.topk_writer import ShardWriter
    rng = np.random.default_rng(seed)
    w = ShardWriter(root / name, k=k, teacher_id="synthetic", vocab_size=vocab)
    for _ in range(n_docs):
        tokens = rng.integers(0, vocab, size=t).astype(np.uint32)
        ids = np.stack([rng.permutation(vocab)[:k] for _ in range(t)]).astype(np.uint32)
        ids[:-1, 0] = tokens[1:]
        probs = np.full((t, k), 0.15 / (k - 1), dtype=np.float32)
        probs[:, 0] = 0.8
        w.add_document(tokens, ids, probs, loss_mask=np.ones(t, dtype=np.uint8))
    w.finalize()
    return root / name


def _run(tmp_path, shard, precision, steps=30):
    from pathlib import Path
    from train.src.train.trainer import Trainer
    d = {
        "model": DEV_TINY, "data_shards": [str(shard)], "seq_len": 32,
        "steps": steps, "batch_size": 2, "lr": 3e-3, "warmup_steps": 5,
        "alpha": 1.0, "loss_chunk_size": 16, "bf16": False, "seed": 0,
        "precision": precision, "init": "scratch",
        "out_dir": str(tmp_path / precision), "log_every": 5,
        "log_filename": str(tmp_path / f"{precision}.log"),
    }
    Trainer(TrainConfig.from_dict(d), device="cpu").train()
    losses = [
        float(line.split("loss ")[1].split()[0])
        for line in Path(d["log_filename"]).read_text().splitlines()
        if "loss " in line
    ]
    return losses


def test_fp8_overlay_smoke(tmp_path):
    """Spec §4 bring-up logic at toy scale: the FP8 path learns (loss
    decreases like bf16) and the curve overlays the bf16 reference within a
    loose fp8-noise tolerance. On CPU this runs the emulation path — same
    quantization semantics as production."""
    shard = _learnable_shard(tmp_path, "s")
    bf16 = _run(tmp_path, shard, "bf16")
    fp8 = _run(tmp_path, shard, "fp8")
    early_b, late_b = sum(bf16[:2]) / 2, sum(bf16[-2:]) / 2
    early_f, late_f = sum(fp8[:2]) / 2, sum(fp8[-2:]) / 2
    assert late_b < early_b * 0.98, f"bf16 loss did not decrease: {early_b:.4f} -> {late_b:.4f}"
    # Toy-scale per-tensor fp8 is the worst case (256-dim tensors don't
    # average out like prod 4096-dim ones); the bar is "learns, does not
    # diverge", not bit-parity with bf16.
    assert late_f < early_f * 0.995, f"fp8 loss did not decrease: {early_f:.4f} -> {late_f:.4f}"
    gap = abs(fp8[-1] - bf16[-1]) / bf16[-1]
    assert gap < 0.25, f"fp8/bf16 final-loss gap {gap:.3f} exceeds overlay tolerance"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_real_scaled_mm_path_on_gpu():
    """The production _scaled_mm path (sm_89+): output tracks bf16 within
    per-tensor fp8 tolerance and backward runs all three FP8 GEMMs."""
    if not fp8_gemm_available():
        pytest.skip("_scaled_mm not supported on this device")
    torch.manual_seed(0)
    lin = FP8Linear(512, 1024, bias=False).to("cuda")
    x = torch.randn(8, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = lin(x)
    ref = torch.nn.functional.linear(x.float(), lin.weight.float())
    rel = (out.float() - ref).abs().max() / ref.abs().max()
    assert rel < 0.15
    out.sum().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(lin.weight.grad).all()
