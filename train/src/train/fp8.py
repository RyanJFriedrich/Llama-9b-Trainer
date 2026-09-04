"""FP8 compute path (spec §4) — attention/FFN GEMMs in float8, everything
else untouched.

Scope (spec §4, LOCKED): ONLY the attention projections (q/k/v/o) and the
SwiGLU FFN (gate/up/down) run as FP8 GEMMs. Embeddings, lm_head (loss path),
norms, AttnRes source-mixing, sink logits, and the Engram readout stay
bf16/fp32. FP8 WEIGHT STORAGE is rejected: master weights stay fp32 (bf16
under autocast at the GEMM), and are dynamically re-quantized per GEMM.

Recipe: per-tensor dynamic scaling, e4m3 for forward operands, e5m2 for the
outgoing gradient (the wide-range gradient format — the Transformer Engine
convention), forward/dgrad/wgrad all FP8 via torch._scaled_mm. cuBLASLt
layout rules: mat1 row-major, mat2 column-major — dgrad/wgrad arrange
layouts explicitly (verified on sm_89 and required on sm_120).

When no FP8 GEMM backend exists (CPU, or a GPU without _scaled_mm support),
the same autograd Function runs an EMULATED path: operands are quantized to
the fp8 grid and dequantized, then multiplied in bf16. Numerics match the
real path up to accumulation order, so dev-box/CI tests exercise exactly the
production quantization semantics.

Toggle: TrainConfig.precision = "bf16" | "fp8" — the trainer calls
apply_fp8() after model construction. FP8Linear subclasses nn.Linear and
reuses the SAME weight Parameter, so state_dict keys, warm start, optimizer
groups, and checkpoint format are all unchanged (one-variable-at-a-time:
precision is a run knob, never an architecture branch).
"""
from __future__ import annotations

import torch
from torch import nn

from train.utils.log import log

E4M3 = torch.float8_e4m3fn
E5M2 = torch.float8_e5m2
E4M3_MAX = 448.0
E5M2_MAX = 57344.0

_FP8_OK: bool | None = None


def fp8_gemm_available() -> bool:
    """Probe once whether torch._scaled_mm runs on this device."""
    global _FP8_OK
    if _FP8_OK is not None:
        return _FP8_OK
    if not torch.cuda.is_available():
        _FP8_OK = False
        return False
    try:
        a = torch.randn(16, 16, device="cuda").bfloat16()
        q, s = _quantize(a, E4M3, E4M3_MAX)
        torch._scaled_mm(q, q.t(), scale_a=s, scale_b=s, out_dtype=torch.bfloat16, use_fast_accum=True)
        _FP8_OK = True
    except Exception:
        _FP8_OK = False
    return _FP8_OK


def _quantize(x: torch.Tensor, dtype: torch.dtype, fmax: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor dynamic scaling: x_fp8 = x / scale, scale = amax / fmax.
    Returns (fp8 tensor, fp32 scalar scale)."""
    amax = x.abs().amax().float().clamp_min(1e-12)
    scale = amax / fmax
    return (x.float() / scale).clamp(-fmax, fmax).to(dtype), scale


def _dequantize(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.to(torch.bfloat16) * scale.to(torch.bfloat16)


class _FP8GEMM(torch.autograd.Function):
    """out[M,N] = x[M,K] @ w[N,K]^T with FP8 operands all three ways.

    Emulation mode reproduces the quantization exactly and multiplies in
    bf16 — the math the GPU path computes, minus the fp8 accumulation.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor, cached_wq: Optional[torch.Tensor] = None, cached_sw: Optional[torch.Tensor] = None) -> torch.Tensor:
        emulate = not fp8_gemm_available() if x.is_cuda else True
        xb = x.to(torch.bfloat16)
        xq, sx = _quantize(xb, E4M3, E4M3_MAX)
        if cached_wq is not None and cached_sw is not None:
            wq, sw = cached_wq, cached_sw
        else:
            wb = w.to(torch.bfloat16)
            wq, sw = _quantize(wb, E4M3, E4M3_MAX)
        ctx.save_for_backward(xq, wq, sx, sw)
        ctx.emulate = emulate
        ctx.x_dtype, ctx.w_dtype = x.dtype, w.dtype
        if emulate:
            return _dequantize(xq, sx) @ _dequantize(wq, sw).t()
        return torch._scaled_mm(xq, wq.t(), scale_a=sx, scale_b=sw,
                                out_dtype=torch.bfloat16, use_fast_accum=True)

    @staticmethod
    def backward(ctx, g: torch.Tensor):
        xq, wq, sx, sw = ctx.saved_tensors
        gq, sg = _quantize(g.to(torch.bfloat16), E5M2, E5M2_MAX)
        if ctx.emulate:
            gb = _dequantize(gq, sg)
            gx = gb @ _dequantize(wq, sw)
            gw = gb.t() @ _dequantize(xq, sx)
        else:
            # dgrad: g[M,N] @ w[N,K] — mat2 must be column-major storage.
            w_cm = wq.t().contiguous().t()
            gx = torch._scaled_mm(gq, w_cm, scale_a=sg, scale_b=sw,
                                  out_dtype=torch.bfloat16, use_fast_accum=True)
            # wgrad: g^T[N,M] @ x[M,K] — mat1 row-major, mat2 column-major.
            # cuBLASLt requires inner dims divisible by 16; M (token count)
            # is the only dim that can miss, so zero-pad it — zero rows
            # contribute exactly zero to the product.
            m = gq.shape[0]
            if m % 16:
                pad = 16 - m % 16
                gq_m = torch.cat([gq, gq.new_zeros(pad, gq.shape[1])], 0)
                xq_m = torch.cat([xq, xq.new_zeros(pad, xq.shape[1])], 0)
            else:
                gq_m, xq_m = gq, xq
            gt_rm = gq_m.t().contiguous()
            x_cm = xq_m.t().contiguous().t()
            gw = torch._scaled_mm(gt_rm, x_cm, scale_a=sg, scale_b=sx,
                                  out_dtype=torch.bfloat16, use_fast_accum=True)
        return gx.to(ctx.x_dtype), gw.to(ctx.w_dtype), None, None


class FP8Linear(nn.Linear):
    """nn.Linear with the GEMM in FP8 (spec §4). Same parameter object and
    state_dict keys as the Linear it replaces."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self._cached_wq: Optional[torch.Tensor] = None
        self._cached_sw: Optional[torch.Tensor] = None

    def cache_fp8_weight(self) -> None:
        wb = self.weight.to(torch.bfloat16)
        self._cached_wq, self._cached_sw = _quantize(wb, E4M3, E4M3_MAX)

    def clear_fp8_cache(self) -> None:
        self._cached_wq = None
        self._cached_sw = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        out = _FP8GEMM.apply(x.reshape(-1, shape[-1]), self.weight, self._cached_wq, self._cached_sw)
        return out.reshape(*shape[:-1], self.weight.shape[0])


def cache_model_fp8_weights(model: nn.Module) -> None:
    """Pre-quantize FP8Linear weights once per step across micro-batches."""
    for mod in model.modules():
        if isinstance(mod, FP8Linear):
            mod.cache_fp8_weight()


def clear_model_fp8_weights(model: nn.Module) -> None:
    """Clear cached FP8Linear weights after micro-batch accumulation."""
    for mod in model.modules():
        if isinstance(mod, FP8Linear):
            mod.clear_fp8_cache()


def apply_fp8(model: nn.Module) -> int:
    """Swap attention/FFN Linears for FP8Linear in place. Returns the number
    of swapped GEMMs. Refuses to run twice or on a model with FP8Linears
    already present."""
    swapped = 0
    for mod in model.modules():
        # RefitAttention / LlamaAttention expose q/k/v/o; LlamaMLP gate/up/down.
        for attr in ("q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"):
            lin = getattr(mod, attr, None)
            if lin is None:
                continue
            if isinstance(lin, FP8Linear):
                raise ValueError("apply_fp8 called twice")
            if not isinstance(lin, nn.Linear):
                continue
            new = FP8Linear(lin.in_features, lin.out_features, bias=lin.bias is not None)
            new.weight = lin.weight  # same Parameter: state_dict/optimizer stable
            if lin.bias is not None:
                new.bias = lin.bias
            setattr(mod, attr, new)
            swapped += 1
    log(f"fp8: {swapped} GEMMs switched to FP8Linear (spec §4 scope: "
        f"attention + FFN only)", print_console=True)
    return swapped
