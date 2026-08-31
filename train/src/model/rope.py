"""Rotary position embeddings — full RoPE (donor/SWA) and partial RoPE (p-RoPE).

`RotaryEmbedding` reproduces the DONOR's own RoPE configuration verbatim,
including the Llama-3.1 frequency scaling ("llama3" rope type), because M1
acceptance is bit-matching the HF reference. It is also used bare (no
scaling) for the refit's SWA layers (spec §3.2: theta=10k, window-capped
offsets make aliasing structurally impossible).

`PartialRotaryEmbedding` is the v2.0 positional scheme for GLOBAL and GATHER
layers (spec §3.3): rotate only the first `fraction` of head dims
(NeoX-style partial rotary — a contiguous 32-dim slice at fraction 0.25,
head_dim 128), theta=1M bare; the remaining dims are never rotated. The
slice's frequencies are computed over the rotary slice's own dim (the
NeoX/HF partial-rotary convention), so the slowest rotated pair keeps a
long-range position channel (wavelength ~2.6M tokens at theta=1M).

No YaRN/NTK/position-interpolation anywhere (standing rule); no NoPE
(retired in spec v2.0).

Math mirrors HF `transformers` `_compute_llama3_parameters` so the parity
test compares like for like.
"""
from __future__ import annotations

import math
from typing import Any, Optional

import torch
from torch import nn


def compute_inv_freq(
    head_dim: int,
    theta: float,
    rope_scaling: Optional[dict[str, Any]] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Base frequencies, optionally with the donor's Llama-3.1 scaling."""
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    if rope_scaling is None:
        return inv_freq
    rope_type = rope_scaling.get("rope_type", rope_scaling.get("type"))
    if rope_type in (None, "default"):
        return inv_freq  # transformers normalizes "no scaling" to {"rope_type": "default"}
    if rope_type != "llama3":
        raise ValueError(
            f"unsupported rope_scaling type {rope_type!r}; the donor uses 'llama3'. "
            "YaRN/NTK/position-interpolation are forbidden here (standing rule 2)."
        )
    factor = rope_scaling["factor"]
    low_freq_factor = rope_scaling["low_freq_factor"]
    high_freq_factor = rope_scaling["high_freq_factor"]
    old_context_len = rope_scaling["original_max_position_embeddings"]

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq
    # wavelen < high_freq_wavelen: do nothing
    # wavelen > low_freq_wavelen: divide by factor
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    # otherwise: smooth interpolation between the two
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    return torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)


class RotaryEmbedding(nn.Module):
    """Cos/sin cache, rotate_half (GPT-NeoX-style) convention, as HF Llama."""

    def __init__(
        self,
        head_dim: int,
        theta: float = 500000.0,
        rope_scaling: Optional[dict[str, Any]] = None,
        max_position_embeddings: int = 131072,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        inv_freq = compute_inv_freq(head_dim, theta, rope_scaling, device=device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_position_embeddings, device=device)

    def _build_cache(self, seq_len: int, device: Optional[torch.device] = None) -> None:
        t = torch.arange(seq_len, device=device or self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)  # [T, head_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [T, head_dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def update_inv_freq(self, inv_freq: torch.Tensor) -> None:
        """Swap base frequencies and rebuild the cache (used by the refit's
        SWA theta anneal ablation path — spec §7.2)."""
        self.inv_freq = inv_freq.to(device=self.inv_freq.device, dtype=torch.float32)
        self._build_cache(self.cos_cached.shape[0])

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (cos, sin) each [1, 1, seq_len, head_dim] for broadcasting
        against [batch, heads, seq_len, head_dim]."""
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        cos = self.cos_cached[:seq_len][None, None, :, :]
        sin = self.sin_cached[:seq_len][None, None, :, :]
        return cos, sin


class PartialRotaryEmbedding(nn.Module):
    """p-RoPE (spec §3.3): rotate only the first `rotary_dim` of each head's
    dims; dims >= rotary_dim carry no positional signal.

    `rotary_dim` is `int(head_dim * fraction)` (0.25 * 128 = 32 in the default
    build), a contiguous slice, rotate_half within the slice — the NeoX/HF
    partial-rotary convention, including computing inv_freq over the slice's
    own dim. Bare theta (1M in the default build), no llama3 scaling.
    """

    def __init__(
        self,
        head_dim: int,
        fraction: float,
        theta: float,
        max_position_embeddings: int = 131072,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        rotary_dim = int(head_dim * fraction)
        if rotary_dim < 2 or rotary_dim % 2 != 0:
            raise ValueError(
                f"p-RoPE rotary slice must be a positive even number of dims; "
                f"got int({head_dim} * {fraction}) = {rotary_dim}"
            )
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.rotary = RotaryEmbedding(
            rotary_dim, theta=theta, rope_scaling=None,
            max_position_embeddings=max_position_embeddings, device=device,
        )

    @property
    def inv_freq(self) -> torch.Tensor:
        return self.rotary.inv_freq

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(cos, sin), each [1, 1, seq_len, rotary_dim]."""
        return self.rotary(seq_len)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """x: [batch, heads, seq, head_dim]; cos/sin broadcastable to x."""
    return (x * cos) + (rotate_half(x) * sin)


def apply_partial_rotary_pos_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int
) -> torch.Tensor:
    """p-RoPE apply (spec §3.3): rotate x[..., :rotary_dim], pass the rest
    through bitwise. cos/sin: [..., rotary_dim] broadcastable to the slice."""
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    return torch.cat(((x_rot * cos) + (rotate_half(x_rot) * sin), x_pass), dim=-1)
