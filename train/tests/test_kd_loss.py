"""M3 acceptance for the fused KD loss (spec §6.4, §8 item 4 [LOCKED]):
matches a naive full-logits reference on tiny batches, at a fraction of peak
memory. Standing rule 3: the reference implementation materializes [T, V]
logits deliberately for verification only — it is never used in training.
"""
import pytest
import torch
import torch.nn.functional as F

from train.src.distill.kd_loss import kd_loss

TAIL_FLOOR = 1e-12  # must match kd_loss._TAIL_FLOOR


def reference_loss(hidden, W, topk_idx, topk_w, tail_w, gold, loss_mask,
                   alpha=1.0, temperature=1.0):
    """Naive reference: materializes full [B, T, V] logits. Test-only."""
    z = (hidden.float() @ W.float().T) / temperature
    logp = F.log_softmax(z, dim=-1)
    logp_topk = logp.gather(2, topk_idx)
    student_tail = (1.0 - logp_topk.exp().sum(-1)).clamp_min(TAIL_FLOOR)
    l_kd = -(topk_w.float() * logp_topk).sum(-1) - tail_w.float() * student_tail.log()
    l_ce = -logp.gather(2, gold.unsqueeze(-1)).squeeze(-1)
    per = alpha * l_kd + (1.0 - alpha) * l_ce
    m = loss_mask.float()
    return (per * m).sum() / m.sum().clamp_min(1.0)


def _batch(B=2, T=33, V=997, D=64, k=4, seed=99, mask=True):
    g = torch.Generator().manual_seed(seed)
    dev = torch.device("cpu")
    hidden = torch.randn(B, T, D, generator=g) * 0.5
    W = torch.randn(V, D, generator=g) * 0.1
    topk_idx = torch.stack([torch.randperm(V, generator=g)[:k] for _ in range(B * T)]).reshape(B, T, k)
    raw = torch.rand(B, T, k, generator=g)
    topk_w = raw / raw.sum(-1, keepdim=True) * 0.9  # teacher top-k mass 0.9
    tail_w = 1.0 - topk_w.sum(-1)
    gold = torch.randint(0, V, (B, T), generator=g)
    loss_mask = torch.ones(B, T)
    if mask:
        loss_mask[0, :5] = 0  # padding/doc-boundary positions
    return hidden, W, topk_idx, topk_w, tail_w, gold, loss_mask


@pytest.mark.parametrize("alpha,temp", [(1.0, 1.0), (0.9, 1.0), (1.0, 2.0), (0.7, 1.5)])
def test_fused_loss_matches_reference(alpha, temp):
    hidden, W, idx, w, tail, gold, m = _batch()
    args = (idx, w, tail, gold, m)
    got = kd_loss(hidden, W, *args, alpha=alpha, temperature=temp, chunk_size=8)
    want = reference_loss(hidden, W, *args, alpha=alpha, temperature=temp)
    assert torch.allclose(got, want, atol=1e-5, rtol=1e-5), (got.item(), want.item())


def test_fused_loss_gradients_match_reference():
    hidden, W, idx, w, tail, gold, m = _batch()
    args = (idx, w, tail, gold, m)

    h1 = hidden.clone().requires_grad_(True)
    W1 = W.clone().requires_grad_(True)
    kd_loss(h1, W1, *args, alpha=0.9, temperature=1.3, chunk_size=8).backward()

    h2 = hidden.clone().requires_grad_(True)
    W2 = W.clone().requires_grad_(True)
    reference_loss(h2, W2, *args, alpha=0.9, temperature=1.3).backward()

    assert torch.allclose(h1.grad, h2.grad, atol=1e-4, rtol=1e-4)
    assert torch.allclose(W1.grad, W2.grad, atol=1e-4, rtol=1e-4)


def test_masked_positions_get_zero_gradient():
    hidden, W, idx, w, tail, gold, m = _batch(mask=True)
    h = hidden.clone().requires_grad_(True)
    W_ = W.clone().requires_grad_(True)
    kd_loss(h, W_, idx, w, tail, gold, m, chunk_size=8).backward()
    assert torch.equal(h.grad[0, :5], torch.zeros(5, h.shape[-1]))
    assert h.grad[0, 5:].abs().sum() > 0


def test_degenerate_tail_is_finite():
    """Student puts ~all mass inside the teacher's top-k: tail -> 0 must not NaN."""
    hidden, W, idx, w, tail, gold, m = _batch(B=1, T=4, mask=False)
    with torch.no_grad():
        # Point hidden straight at a top-k token's embedding row, strongly.
        h = hidden.clone()
        h[0, 0] = W[idx[0, 0, 0]] * 100.0
    loss = kd_loss(h, W, idx, w, tail, gold, m, chunk_size=2)
    assert torch.isfinite(loss)
    h.requires_grad_(True)
    kd_loss(h, W, idx, w, tail, gold, m, chunk_size=2).backward()
    assert torch.isfinite(h.grad).all()


def test_all_masked_returns_zero():
    hidden, W, idx, w, tail, gold, m = _batch()
    m = torch.zeros_like(m)
    h = hidden.clone().requires_grad_(True)
    loss = kd_loss(h, W, idx, w, tail, gold, m)
    assert loss.item() == 0.0
    loss.backward()
    assert torch.equal(h.grad, torch.zeros_like(h.grad))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="peak-memory test needs CUDA")
def test_fused_loss_peak_memory_ceiling():
    """LOCKED rationale (spec §0.5): full fp32 logits at 4k x 128256 are ~1 GB
    per copy; the fused path must run at a fraction of that."""
    B, T, V, D, k = 1, 2048, 128256, 256, 10
    g = torch.Generator(device="cuda").manual_seed(5)
    hidden = torch.randn(B, T, D, generator=g, device="cuda", dtype=torch.bfloat16)
    W = torch.randn(V, D, generator=g, device="cuda", dtype=torch.bfloat16)
    idx = torch.randint(0, V, (B, T, k), device="cuda")
    w = torch.rand(B, T, k, device="cuda") / k
    tail = torch.full((B, T), 0.05, device="cuda")
    gold = torch.randint(0, V, (B, T), device="cuda")
    m = torch.ones(B, T, device="cuda")

    torch.cuda.reset_peak_memory_stats()
    h1 = hidden.clone().requires_grad_(True)
    W1 = W.clone().requires_grad_(True)
    kd_loss(h1, W1, idx, w, tail, gold, m, chunk_size=128).backward()
    fused_peak = torch.cuda.max_memory_allocated()

    torch.cuda.reset_peak_memory_stats()
    h2 = hidden.clone().requires_grad_(True)
    W2 = W.clone().requires_grad_(True)
    reference_loss(h2, W2, idx, w, tail, gold, m).backward()
    ref_peak = torch.cuda.max_memory_allocated()

    print(f"\nfused peak {fused_peak/1e9:.2f} GB, reference peak {ref_peak/1e9:.2f} GB")
    assert fused_peak < 0.25 * ref_peak
    # Absolute ceiling: a handful of chunk-sized transients (chunk x V fp32 ~
    # 65 MB each at chunk=128) plus the fp32 grad accumulator for lm_head —
    # never the multi-GB [T, V] tensors the reference needs (3.6 GB here).
    assert fused_peak < 1.0e9
