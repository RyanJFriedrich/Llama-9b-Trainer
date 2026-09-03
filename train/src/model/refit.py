"""Refit model — the v2.0 hybrid-attention architecture (spec §3).

Built on the M1 substrate (decoder.py stays "the donor, exactly"); every
architectural feature lives here behind the spec §3.1 config:

- Layer-type dispatch: "swa" | "global" | "gather" per config.layer_types.
- SWA layers: sliding-window causal mask (final window 4096; offsets never
  exceed window-1, invariant I7), learned per-head sink logit in the softmax
  denominator (init -10, the gpt-oss pattern — no position, no KV footprint),
  bare RoPE theta=10k (spec §3.2; the optional warm-start theta anneal
  interpolates donor 500k+llama3 -> final in log-inv_freq space, ablation
  tooling only).
- GLOBAL layers: full-span causal attention, p-RoPE — rotate the first 25%
  of head dims at theta=1M, the rest carry no positional signal (spec §3.3;
  NoPE is retired).
- GATHER layer: same p-RoPE family as the globals (uniform positional
  scheme); identity init (zeroed o_proj/down_proj) makes it an exact no-op
  at init regardless (spec §3.4).
- Block AttnRes (attn_res.py) before every sublayer (scope "all_layers") or
  only before global/gather attention ("globals_only"), with per-block
  delta-sum bookkeeping: blocks[0] = embedding output; a block completes
  after each GLOBAL layer (its deltas' sum becomes a new source); the gather
  layer feeds the final running partial.

Cold-start epistemology (spec v2.0 §1/§3.4): training starts at FINAL
topology. The anneal machinery (window/theta schedules via
set_anneal_state) is retained, default final-state, as ablation tooling.
There is no donor-equivalence gate; step-0 acceptance is the behavior-shaped
sanity band of spec §3.8 plus invariants I1-I9.
"""
from __future__ import annotations

import math
from typing import Any, Optional

import torch
import torch.utils.checkpoint
from torch import nn

from train.src.config import ModelConfig
from train.src.model.attn_res import BlockAttnRes
from train.src.model.decoder import LlamaMLP, LlamaRMSNorm, repeat_kv
from train.src.model.rope import (
    PartialRotaryEmbedding,
    RotaryEmbedding,
    apply_partial_rotary_pos_emb,
    apply_rotary_pos_emb,
    compute_inv_freq,
)

# The donor is fixed (Llama-3.1-8B-Instruct): its RoPE is theta=500k with
# llama3 frequency scaling. Donor facts, used only by the SWA theta-anneal
# ablation path (its "from" endpoint), not by the default build.
DONOR_ROPE_THETA = 500000.0
DONOR_ROPE_SCALING: dict[str, Any] = {
    "rope_type": "llama3",
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}

NEG_INF = float("-inf")


def swa_inv_freq_endpoints(cfg: ModelConfig, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """(start, final) inv_freq for the SWA theta-anneal ablation. Start is the
    donor (500k, llama3 scaling); final is bare cfg.swa.rope_theta (10k, spec
    §3.2: slowest wavelength ~55k ~ 13x the 4096 window — no scaling)."""
    assert cfg.swa is not None
    start = compute_inv_freq(cfg.head_dim, DONOR_ROPE_THETA, DONOR_ROPE_SCALING, device)
    final = compute_inv_freq(cfg.head_dim, cfg.swa.rope_theta, None, device)
    return start, final


class RefitAttention(nn.Module):
    """Attention for one refit layer. Parameter names mirror HF
    (q_proj/k_proj/v_proj/o_proj) so the donor warm start is an identity map.
    """

    def __init__(self, cfg: ModelConfig, layer_type: str) -> None:
        super().__init__()
        assert cfg.swa is not None and cfg.global_ is not None and cfg.gather is not None
        self.layer_type = layer_type
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=False)

        self.rotary: Optional[nn.Module] = None  # RotaryEmbedding | PartialRotaryEmbedding
        self.sink_logit: Optional[nn.Parameter] = None
        # Query-block size for the score matrix (ModelConfig.attn_query_chunk;
        # 0 = unchunked). Bounds the fp32 softmax working set at long T.
        self.attn_query_chunk = cfg.attn_query_chunk
        # Eval-time probe hook (spec §8 item 6): when set, called per query
        # chunk with the post-softmax attention probs [B, H, Tc, S]; must
        # aggregate and discard (never retain the tensor). See
        # eval/attn_probes.py.
        self.probe: Optional[Any] = None

        if layer_type == "swa":
            # Bare final theta from construction (spec §3.2); the warm-start
            # anneal ablation re-interpolates via set_anneal_state.
            self.rotary = RotaryEmbedding(cfg.head_dim, theta=cfg.swa.rope_theta)
            # Learned per-head sink logit (spec §3.2). Init -10: e^-10 ~ 4.5e-5
            # relative on the softmax denominator — a near-no-op at init (I4).
            self.sink_logit = nn.Parameter(torch.full((self.num_heads,), cfg.swa.sink.init))
            self.window: Optional[int] = cfg.swa.window  # None = full sequence (ablation)
        elif layer_type == "global":
            self.rotary = PartialRotaryEmbedding(
                cfg.head_dim, cfg.global_.rope_fraction, cfg.global_.rope_theta
            )
        elif layer_type == "gather":
            # Same p-RoPE family as the globals (spec §3.3: uniform
            # positional scheme); identity init makes the layer an exact
            # no-op at init regardless (spec §3.4).
            self.rotary = PartialRotaryEmbedding(
                cfg.head_dim, cfg.gather.rope_fraction, cfg.gather.rope_theta
            )
        else:
            raise ValueError(f"unknown layer_type {layer_type!r}")

    def set_theta_progress(self, s: float, inv_start: torch.Tensor, inv_final: torch.Tensor) -> None:
        """SWA theta anneal (ablation tooling): log-space interpolation
        between endpoint inv_freq vectors (exact at s=0 donor and s=1 final)."""
        assert self.layer_type == "swa" and isinstance(self.rotary, RotaryEmbedding)
        log_inv = (1.0 - s) * torch.log(inv_start) + s * torch.log(inv_final)
        self.rotary.update_inv_freq(torch.exp(log_inv))

    def forward(self, hidden_states: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if isinstance(self.rotary, PartialRotaryEmbedding):
            cos, sin = self.rotary(seq_len)
            cos = cos.to(q.dtype)
            sin = sin.to(q.dtype)
            rd = self.rotary.rotary_dim
            q = apply_partial_rotary_pos_emb(q, cos, sin, rd)
            k = apply_partial_rotary_pos_emb(k, cos, sin, rd)
        elif self.rotary is not None:
            cos, sin = self.rotary(seq_len)
            cos = cos.to(q.dtype)
            sin = sin.to(q.dtype)
            q = apply_rotary_pos_emb(q, cos, sin)
            k = apply_rotary_pos_emb(k, cos, sin)

        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        # Query-chunked attention (ModelConfig.attn_query_chunk). The math is
        # unchanged — softmax denominators are per query row over ALL keys —
        # but the [H, T, S] fp32 score/softmax working set is bounded to one
        # chunk at a time. Under autograd each chunk's core runs inside its
        # own gradient checkpoint, so backward also recomputes a single
        # chunk's scores instead of holding a whole layer's worth (the 96 GB
        # box OOM without this at T=8192).
        chunk = self.attn_query_chunk
        if not (0 < chunk < seq_len):
            chunk = seq_len  # unchunked: one block, the original path
        ckpt_chunks = (
            chunk < seq_len and torch.is_grad_enabled() and hidden_states.requires_grad
        )
        outs = []
        for s in range(0, seq_len, chunk):
            e = min(s + chunk, seq_len)
            qc = q[:, :, s:e]
            mask_c = causal_mask[:, :, s:e, :]
            if ckpt_chunks:
                oc, probs = torch.utils.checkpoint.checkpoint(
                    self._attend_core, qc, k, v, mask_c, use_reentrant=False
                )
            else:
                oc, probs = self._attend_core(qc, k, v, mask_c)
            outs.append(oc)
            if self.probe is not None:
                # Eval-time stats hook (spec §8 item 6): called per query
                # chunk with post-softmax probs [B, H, Tc, S]; the hook must
                # aggregate and discard (never retain the tensor).
                self.probe(probs)
        attn_output = torch.cat(outs, dim=2)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(attn_output)

    def _attend_core(
        self,
        qc: torch.Tensor,  # [B, H, Tc, D] query block
        k: torch.Tensor,   # [B, H, S, D]
        v: torch.Tensor,   # [B, H, S, D]
        mask_c: torch.Tensor,  # [1, 1, Tc, S] additive
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scores -> (sink) softmax -> values for one query block.
        Returns (output [B, H, Tc, D], probs [B, H, Tc, S])."""
        attn_weights = torch.matmul(qc, k.transpose(2, 3)) * self.scaling
        attn_weights = attn_weights + mask_c

        if self.sink_logit is None:
            # Match the M1 donor math exactly: fp32 softmax.
            attn_probs = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(qc.dtype)
        else:
            # Softmax with a learned sink in the denominator, fp32, max-stable.
            w = attn_weights.to(torch.float32)
            sink = self.sink_logit.to(torch.float32)[None, :, None, None]
            m = torch.maximum(w.amax(dim=-1, keepdim=True), sink)
            p = torch.exp(w - m)
            denom = p.sum(dim=-1, keepdim=True) + torch.exp(sink - m)
            attn_probs = (p / denom).to(qc.dtype)

        return torch.matmul(attn_probs, v), attn_probs


class RefitDecoderLayer(nn.Module):
    """Same shapes as the donor layer (spec §3.1: layers differ only in
    attention masking and position encoding, never projection shapes)."""

    def __init__(self, cfg: ModelConfig, layer_type: str) -> None:
        super().__init__()
        self.layer_type = layer_type
        self.self_attn = RefitAttention(cfg, layer_type)
        self.mlp = LlamaMLP(cfg)  # type: ignore[arg-type]  # LlamaBaseConfig-duck-typed
        self.input_layernorm = LlamaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)


class RefitModel(nn.Module):
    """The 33-layer (or dev_tiny 9-layer) hybrid model. Module tree mirrors
    the donor (model.embed_tokens / model.layers.N / model.norm / lm_head)
    for layers 0..N-2 so warm start is an identity map; the gather layer is
    the extra layer at config.gather.position."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        assert cfg.layer_types is not None and cfg.attn_res is not None
        assert cfg.global_ is not None and cfg.gather is not None and cfg.engram is not None
        self.refit_config = cfg
        self.layer_types = list(cfg.layer_types)
        self.attn_res_scope = cfg.attn_res.scope if cfg.attn_res.enabled else "off"
        self.grad_checkpointing = False  # set by the trainer

        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.model.layers = nn.ModuleList(
            [RefitDecoderLayer(cfg, t) for t in self.layer_types]
        )
        self.model.norm = LlamaRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        # AttnRes application points. all_layers: 2 per layer (pre-attn,
        # pre-mlp). globals_only: pre-attn of global/gather layers only.
        self.attn_res_map: dict[tuple[int, str], int] = {}
        points: list[BlockAttnRes] = []
        for i, t in enumerate(self.layer_types):
            if self.attn_res_scope == "all_layers":
                for sub in ("pre_attn", "pre_mlp"):
                    self.attn_res_map[(i, sub)] = len(points)
                    points.append(BlockAttnRes(cfg.hidden_size, cfg.rms_norm_eps, cfg.attn_res.gate))
            elif self.attn_res_scope == "globals_only" and t in ("global", "gather"):
                self.attn_res_map[(i, "pre_attn")] = len(points)
                points.append(BlockAttnRes(cfg.hidden_size, cfg.rms_norm_eps, cfg.attn_res.gate))
        self.model.attn_res = nn.ModuleList(points)

        # A block completes after each GLOBAL layer (spec §3.5: boundaries
        # align with the [3xSWA + 1xG] groups; generalizes to irregular
        # layer_types since they remain config-driven).
        self.block_ends = {i for i, t in enumerate(self.layer_types) if t == "global"}

        # Engram sidecar (spec §3.6, annex A1): host-resident tables + a small
        # device-side readout, injected once at the output of layer
        # engram.injection_point. The tables are NOT an nn.Module attribute —
        # they live in host RAM and are checkpointed by the trainer separately.
        self.engram: Optional[nn.Module] = None
        self.engram_tables = None
        if cfg.engram.enabled:
            from train.src.engram.readout import EngramReadout
            from train.src.engram.tables import EngramTables

            self.engram = EngramReadout(cfg.engram, cfg.hidden_size, cfg.rms_norm_eps)
            self.engram_tables = EngramTables(cfg.engram, cfg.vocab_size)

        # SWA theta-anneal ablation endpoints (spec §7.2 tooling; unused in
        # the default build, which constructs at the final bare theta).
        assert cfg.swa is not None
        self._swa_inv_start, self._swa_inv_final = swa_inv_freq_endpoints(cfg)
        self._swa_theta_annealed = cfg.swa.rope_theta_warmstart_anneal is not None

        # Default state = FINAL topology (spec v2.0 §3.4: no anneal required;
        # training starts at final topology). Ablation arms that configure an
        # anneal start at its "from" state instead.
        self.anneal_state = {
            "window": None if cfg.swa.window_anneal else cfg.swa.window,
            "theta_progress": 0.0 if self._swa_theta_annealed else 1.0,
        }
        self.set_anneal_state(**self.anneal_state)

    def set_anneal_state(
        self,
        window: Optional[int],
        theta_progress: float,
    ) -> None:
        """Set the topology-anneal knobs (spec §7.2 ablation tooling). The
        default run holds the final state: window=cfg.swa.window,
        theta_progress=1.0. window=None means full-sequence attention on SWA
        layers (the window anneal's "from"); theta_progress=0 is the donor's
        RoPE (the theta anneal's "from")."""
        self.anneal_state = {
            "window": window,
            "theta_progress": theta_progress,
        }
        for layer in self.model.layers:
            attn = layer.self_attn
            if attn.layer_type == "swa":
                attn.window = window
                if self._swa_theta_annealed:
                    attn.set_theta_progress(theta_progress, self._swa_inv_start, self._swa_inv_final)

    def _mask_for(self, layer_type: str, window: Optional[int], seq_len: int,
                  dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Additive [1, 1, T, T] mask. Causal always; SWA additionally masks
        j < i - window + 1, so max token-token relative offset = window - 1
        (< window, satisfying the §3.2 invariant)."""
        mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device)
        mask = torch.triu(mask, diagonal=1)  # causal: mask future
        if layer_type == "swa" and window is not None and window < seq_len:
            # Additionally mask j <= i - window, leaving i - j <= window - 1.
            too_far = torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device)
            too_far = torch.tril(too_far, diagonal=-window)
            mask = mask + too_far
        return mask[None, None, :, :].to(dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        capture: Optional[dict[str, Any]] = None,
        return_hidden: bool = False,
        engram: Optional[Any] = None,
    ) -> torch.Tensor:
        """input_ids: [batch, seq] -> logits [batch, seq, vocab], or the final
        hidden states [batch, seq, hidden] when return_hidden=True.

        return_hidden is the TRAINING path: the fused KD loss (distill/
        kd_loss.py) computes logits chunk-by-chunk from the hidden states, so
        the full [B, T, V] logit tensor is never materialized (standing
        rule 3). The logits path exists for eval/parity only.

        engram: a pre-gathered GatherBatch (engram/tables.py) for this batch.
        The trainer passes one per micro-batch (prefetched); when omitted the
        tables gather synchronously from input_ids (eval / non-training path).

        capture (tests only): pass a dict to record internals —
        capture["embedding"], capture["hiddens"][i] (stream after layer i),
        capture["attn_res_sources"][(i, sub)] = list of source tensors.
        """
        h = self.model.embed_tokens(input_ids)
        seq_len = input_ids.shape[1]
        dtype, device = h.dtype, h.device

        # Engram gather: trainer passes a prefetched GatherBatch; otherwise
        # gather synchronously (eval / non-training path).
        if self.engram is not None and engram is None:
            engram = self.engram_tables.gather(
                input_ids, device, requires_grad=torch.is_grad_enabled()
            )

        if capture is not None:
            capture["embedding"] = h.detach().clone()
            capture.setdefault("hiddens", {})
            capture.setdefault("attn_res_sources", {})

        # Per-layer-type masks (computed once per forward).
        masks: dict[tuple[str, Optional[int]], torch.Tensor] = {}
        for t in ("swa", "global", "gather"):
            w = self.anneal_state["window"] if t == "swa" else None
            masks[(t, w)] = self._mask_for(t, w, seq_len, dtype, device)

        # AttnRes delta-sum bookkeeping (spec §3.5).
        blocks: list[torch.Tensor] = [h]  # blocks[0] = embedding output e
        partial = torch.zeros_like(h)     # current block's running partial b_n

        def apply_attn_res(i: int, sub: str, use_ckpt: bool) -> torch.Tensor:
            key = (i, sub)
            if key not in self.attn_res_map:
                return h
            sources = blocks + [partial]
            if capture is not None:
                capture["attn_res_sources"][key] = [s.detach().clone() for s in sources]
            mod = self.model.attn_res[self.attn_res_map[key]]
            if use_ckpt:
                # BlockAttnRes stacks the sources ([N, B, T, D]) and autograd
                # saves the stack AND its keys-norm per application — ~2N x
                # 64 MB at 8k, growing with N; over 66 applications that is
                # ~60 GB and OOMs the 96 GB box. The sources themselves stay
                # materialized either way (the spec'd O(Nd) bookkeeping), so
                # checkpoint the computation: recompute is a stack + norm +
                # N-way softmax, and the inputs are already live.
                return torch.utils.checkpoint.checkpoint(mod, sources, h, use_reentrant=False)
            return mod(sources, h)

        # Checkpointed sublayer = RMSNorm + attn/MLP as one region. Defined
        # with explicit args (NOT a loop-closure lambda — a lambda would bind
        # the LAST iteration's modules at backward-recompute time and torch
        # raises CheckpointError on the mismatched saves).
        def _normed_sublayer(norm, sublayer, h_in, mask=None):
            if mask is None:
                return sublayer(norm(h_in))
            return sublayer(norm(h_in), mask)

        for i, layer in enumerate(self.model.layers):
            attn = layer.self_attn
            mask = masks[(attn.layer_type, attn.window if attn.layer_type == "swa" else None)]

            # Gradient checkpointing (M4): recompute the attention/MLP
            # sublayers in backward instead of holding their activations.
            # The checkpointed region includes the sublayer's input RMSNorm —
            # its fp32 upcast intermediates (~0.3 GB at 8k) would otherwise be
            # saved at all 66 application points (~20 GB; the #2 bring-up OOM
            # after the AttnRes stack fix). The AttnRes sources
            # (blocks/partial) stay materialized — they are the spec'd O(Nd)
            # bookkeeping, N <= 10.
            ckpt = (
                self.grad_checkpointing and self.training and h.requires_grad
            )

            x = apply_attn_res(i, "pre_attn", ckpt)
            if ckpt:
                d1 = torch.utils.checkpoint.checkpoint(
                    _normed_sublayer, layer.input_layernorm, attn, x, mask,
                    use_reentrant=False)
            else:
                d1 = attn(layer.input_layernorm(x), mask)
            h = h + d1
            partial = partial + d1

            x = apply_attn_res(i, "pre_mlp", ckpt)
            if ckpt:
                d2 = torch.utils.checkpoint.checkpoint(
                    _normed_sublayer, layer.post_attention_layernorm, layer.mlp,
                    x, use_reentrant=False)
            else:
                d2 = layer.mlp(layer.post_attention_layernorm(x))
            h = h + d2
            partial = partial + d2

            # Engram injection (spec §3.6): single delta at the output of
            # layer injection_point, registered into the running partial
            # BEFORE the block completes (I2/I8 — every residual-stream
            # addition is delta-sum registered, so block-1's AttnRes source
            # includes the injection exactly).
            if self.engram is not None and i == self.refit_config.engram.injection_point:
                delta = self.engram(engram)
                h = h + delta
                partial = partial + delta

            if i in self.block_ends:
                blocks.append(partial)
                partial = torch.zeros_like(h)

            if capture is not None:
                capture["hiddens"][i] = h.detach().clone()

        h = self.model.norm(h)
        if return_hidden:
            return h
        return self.lm_head(h)
