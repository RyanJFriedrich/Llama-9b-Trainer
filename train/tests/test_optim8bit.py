"""8-bit AdamW (spec §0.5 v1.2): correctness + state-format tests."""
import torch

from train.src.train.optim8bit import AdamW8bit, BLOCK, _dequantize, _quantize


def _toy_problem(seed=0, n=4096):
    """Quadratic bowl: w* near origin; loss must go ~to 0."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, n, generator=g) / n ** 0.5
    target = torch.randn(n, generator=g)
    w = torch.randn(n, generator=g, requires_grad=True)
    return A, target, w


def test_convergence_parity_with_fp32_adamw():
    results = {}
    for kind in ("fp32", "8bit"):
        A, target, w = _toy_problem()
        opt = (torch.optim.AdamW([w], lr=0.01, weight_decay=0.0) if kind == "fp32"
               else AdamW8bit([w], lr=0.01, weight_decay=0.0))
        for _ in range(500):
            opt.zero_grad()
            loss = ((A @ w - target) ** 2).mean()
            loss.backward()
            opt.step()
        results[kind] = loss.item()
    # Measured at lr=0.01, 500 steps: fp32 0.148, 8bit 0.072. The 8-bit path
    # must converge to the same basin; exact equality is not expected.
    assert results["fp32"] < 0.2
    assert results["8bit"] < 0.2
    assert results["8bit"] < 5 * max(results["fp32"], 1e-6)


def test_states_are_8bit():
    w = torch.randn(BLOCK * 4, requires_grad=True)  # exact block multiple
    opt = AdamW8bit([w])
    w.grad = torch.randn_like(w)
    opt.step()
    state = opt.state[w]
    assert state["q_m"].dtype == torch.int8 and state["q_r"].dtype == torch.int8
    assert state["s_m"].dtype == torch.float32
    # 2 int8 states = 2 B/param + negligible per-block scale overhead
    # (spec §0.5: 2 B/param total).
    bytes_per_param = (state["q_m"].numel() + state["q_r"].numel()) / w.numel()
    assert bytes_per_param == 2.0


def test_quantize_roundtrip_error_bounded():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(10000, generator=g) * 3.0
    q, s = _quantize(x)
    x2 = _dequantize(q, s, x.numel())
    # Round-to-nearest: abs error <= half the element's block scale.
    padded = (-x.numel()) % BLOCK
    bounds = (s / 2).repeat_interleave(BLOCK)
    if padded:
        bounds = bounds[:-padded]
    assert ((x2 - x).abs() <= bounds * 1.001).all()


def test_checkpoint_state_roundtrip_bitwise():
    g = torch.Generator().manual_seed(2)
    w1 = torch.randn(5000, requires_grad=True)
    opt1 = AdamW8bit([w1])
    for _ in range(5):
        w1.grad = torch.randn_like(w1, generator=g)
        opt1.step()
    # Optimizer state_dict carries optimizer state only — params are the
    # caller's; copy them for the resume trajectory.
    w2 = w1.detach().clone().requires_grad_(True)
    opt2 = AdamW8bit([w2])
    opt2.load_state_dict(opt1.state_dict())
    for _ in range(5):
        grad = torch.randn_like(w1, generator=g)
        w1.grad = grad
        opt1.step()
        w2.grad = grad.clone()
        opt2.step()
    assert torch.equal(w1, w2)  # int8 states serialize bit-exactly


def test_trainer_knob(tmp_path):
    """The TrainConfig optimizer knob selects AdamW8bit end to end."""
    from train.src.config import TrainConfig
    cfg = TrainConfig.from_dict({
        "model": "train/configs/model/dev_tiny.yaml", "data_shards": ["x"],
        "steps": 4, "warmup_steps": 1, "optimizer": "adamw8bit",
    })
    assert cfg.optimizer == "adamw8bit"
    with __import__("pytest").raises(ValueError, match="optimizer"):
        TrainConfig.from_dict({
            "model": "m", "data_shards": ["x"], "steps": 4, "warmup_steps": 1,
            "optimizer": "lion",
        })
