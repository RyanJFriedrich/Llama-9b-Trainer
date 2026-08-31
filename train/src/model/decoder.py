"""Plain dense Llama decoder — M1 (no refit features).

A clean re-implementation of the Llama 3 decoder whose only job is to
bit-match the HF reference implementation when loaded with donor weights.
Module names mirror HF exactly (`model.layers.N.self_attn.q_proj`, ...) so a
donor HF state dict loads with an identity key mapping and any unaccounted
tensor is visible in the loader report.

Refit features (SWA masks, sink logits, p-RoPE globals, AttnRes, gather) do
NOT live here — they arrive behind config flags on top of this substrate.
This file must remain "the donor, exactly".
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from torch import nn

from train.src.model.rope import RotaryEmbedding, apply_rotary_pos_emb


@dataclass
class LlamaBaseConfig:
    """Subset of HF LlamaConfig the plain decoder needs. Field names match HF."""

    vocab_size: int = 128256
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: Optional[int] = None  # defaults to hidden_size // num_attention_heads
    rms_norm_eps: float = 1e-05
    rope_theta: float = 500000.0
    rope_scaling: Optional[dict[str, Any]] = None
    max_position_embeddings: int = 131072
    tie_word_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

    @classmethod
    def from_hf(cls, hf_config: Any) -> "LlamaBaseConfig":
        """Build from a transformers LlamaConfig (or any object with these attrs)."""
        names = {f for f in cls.__dataclass_fields__} - {"head_dim"}  # type: ignore[attr-defined]
        kwargs = {n: getattr(hf_config, n) for n in names if hasattr(hf_config, n)}
        head_dim = getattr(hf_config, "head_dim", None)
        if head_dim is not None:
            kwargs["head_dim"] = head_dim
        return cls(**kwargs)


class LlamaRMSNorm(nn.Module):
    """HF-equivalent RMSNorm: statistics in fp32, scale by weight after."""

    def __init__(self, hidden_size: int, eps: float = 1e-05) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[batch, kv_heads, seq, head_dim] -> [batch, kv_heads*n_rep, seq, head_dim] (HF semantics)."""
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class LlamaAttention(nn.Module):
    """HF-equivalent eager attention (GQA via repeat_kv, fp32 softmax)."""

    def __init__(self, config: LlamaBaseConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        attn_weights = attn_weights + causal_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(attn_output)


class LlamaMLP(nn.Module):
    """SwiGLU MLP, HF naming."""

    def __init__(self, config: LlamaBaseConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.act_fn = nn.functional.silu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaBaseConfig) -> None:
        super().__init__()
        self.self_attn = LlamaAttention(config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cos, sin, causal_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class LlamaBaseModel(nn.Module):
    """Full causal LM. Mirrors HF LlamaForCausalLM module tree so donor state
    dicts load with strict=True and identity key mapping."""

    def __init__(self, config: LlamaBaseConfig) -> None:
        super().__init__()
        self.config = config
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.model.layers = nn.ModuleList(
            [LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.model.norm = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.rotary = RotaryEmbedding(
            config.head_dim,
            theta=config.rope_theta,
            rope_scaling=config.rope_scaling,
            max_position_embeddings=config.max_position_embeddings,
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def _causal_mask(self, seq_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Additive [1, 1, seq, seq] causal mask (0 / -inf), as HF eager."""
        mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask[None, None, :, :].to(dtype)

    def forward(self, input_ids: torch.Tensor, return_hidden: bool = False) -> torch.Tensor:
        """input_ids: [batch, seq] -> logits [batch, seq, vocab]; with
        return_hidden=True, final hidden states [batch, seq, hidden] instead
        (training/scoring path — lm_head is applied chunk-wise downstream so
        full logits are never materialized)."""
        hidden = self.model.embed_tokens(input_ids)
        seq_len = input_ids.shape[1]
        cos, sin = self.rotary(seq_len)
        cos = cos.to(hidden.dtype)
        sin = sin.to(hidden.dtype)
        causal_mask = self._causal_mask(seq_len, hidden.dtype, hidden.device)
        for layer in self.model.layers:
            hidden = layer(hidden, cos, sin, causal_mask)
        hidden = self.model.norm(hidden)
        if return_hidden:
            return hidden
        return self.lm_head(hidden)
