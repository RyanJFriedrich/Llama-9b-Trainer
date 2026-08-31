"""Config system for the Llama-9B harness (spec v2.0 §3.1 skeleton).

The spec is config-driven: every ablation arm and every run is a config
diff, never a code branch. `ModelConfig` implements the §3.1 reference
skeleton exactly — the acceptance test round-trips the spec's JSON through
`from_dict`/`to_dict` unchanged. Unknown keys are rejected on load so
misspelled knobs fail loudly instead of being silently ignored.

Key-name notes: `global` and `from` are Python keywords, so they map to the
`global_` / `start` fields. Optional anneal knobs (spec §7.2 ablation
tooling) are omitted from `to_dict()` when unset, which keeps the
round-trip exact.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import yaml

VALID_LAYER_TYPES = ("swa", "global", "gather")
VALID_ATTN_RES_SCOPES = ("all_layers", "globals_only")
VALID_GATHER_INITS = ("identity_from_layer_31", "random")
VALID_SINK_TYPES = ("learned_logit",)


def _reject_unknown(d: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(d) - allowed)
    if unknown:
        raise ValueError(f"{where}: unknown config keys {unknown}")


@dataclass
class AnnealConfig:
    """A start-state -> final-state anneal over a phase (spec §7.1).

    Serialized as {"from": <start>, "schedule": <schedule>} per §3.7, with an
    optional explicit "to". `start` may be the string "full" for the SWA
    window anneal (windows start at full sequence length, spec §5 item 4).
    """

    start: Union[float, str]
    schedule: str = "log_linear"
    to: Optional[float] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any], where: str = "anneal") -> "AnnealConfig":
        _reject_unknown(d, {"from", "schedule", "to"}, where)
        if "from" not in d:
            raise ValueError(f"{where}: anneal requires a 'from' key")
        return cls(start=d["from"], schedule=d.get("schedule", "log_linear"), to=d.get("to"))

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"from": self.start, "schedule": self.schedule}
        if self.to is not None:
            d["to"] = self.to
        return d


@dataclass
class SinkConfig:
    """Learned per-head sink logit added to the softmax denominator (spec §3.2)."""

    type: str = "learned_logit"
    init: float = -10.0  # exact no-op at step 0 — do not change without owner sign-off

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SinkConfig":
        _reject_unknown(d, {"type", "init"}, "swa.sink")
        return cls(type=d.get("type", "learned_logit"), init=d.get("init", -10.0))

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "init": self.init}


@dataclass
class SWAConfig:
    """Sliding-window (local) layers (spec §3.2): window 4096, bare RoPE
    theta=10k (window-capped offsets make aliasing structurally impossible).

    The anneal knobs are ablation tooling (spec §7.2), default unset = final
    topology from step 0: `rope_theta_warmstart_anneal` interpolates
    donor 500k(llama3-scaled) -> `rope_theta` bare in log-inv_freq space;
    `window_anneal` runs full-sequence -> `window`."""

    window: int = 4096
    rope_theta: float = 10000.0
    rope_theta_warmstart_anneal: Optional[AnnealConfig] = None  # 500k -> 10k, log-space
    sink: SinkConfig = None  # type: ignore[assignment]
    window_anneal: Optional[AnnealConfig] = None

    def __post_init__(self) -> None:
        if self.sink is None:
            self.sink = SinkConfig()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SWAConfig":
        _reject_unknown(
            d, {"window", "rope_theta", "rope_theta_warmstart_anneal", "sink", "window_anneal"}, "swa"
        )
        anneal = d.get("rope_theta_warmstart_anneal")
        win_anneal = d.get("window_anneal")
        return cls(
            window=d.get("window", 4096),
            rope_theta=d.get("rope_theta", 10000.0),
            rope_theta_warmstart_anneal=(
                AnnealConfig.from_dict(anneal, "swa.rope_theta_warmstart_anneal") if anneal else None
            ),
            sink=SinkConfig.from_dict(d["sink"]) if "sink" in d else SinkConfig(),
            window_anneal=(
                AnnealConfig.from_dict(win_anneal, "swa.window_anneal") if win_anneal else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"window": self.window, "rope_theta": self.rope_theta}
        if self.rope_theta_warmstart_anneal is not None:
            d["rope_theta_warmstart_anneal"] = self.rope_theta_warmstart_anneal.to_dict()
        d["sink"] = self.sink.to_dict()
        if self.window_anneal is not None:
            d["window_anneal"] = self.window_anneal.to_dict()
        return d


@dataclass
class GlobalConfig:
    """Full-attention p-RoPE global layers (spec §3.3): rotate only the first
    `rope_fraction` of head dims (the 32-dim slice at 0.25/head_dim 128) at
    bare theta `rope_theta` (1M); the remaining dims carry no positional
    signal. NoPE is retired (spec v2.0); there is no full-RoPE default —
    fraction 1.0 is the ablation arm (spec §10 sweep)."""

    rope_type: str = "prope"
    rope_fraction: float = 0.25
    rope_theta: float = 1000000.0
    qk_norm: bool = False  # LOCKED off for GLOBAL/GATHER (spec §3.2)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GlobalConfig":
        _reject_unknown(d, {"rope_type", "rope_fraction", "rope_theta", "qk_norm"}, "global")
        return cls(
            rope_type=d.get("rope_type", "prope"),
            rope_fraction=d.get("rope_fraction", 0.25),
            rope_theta=d.get("rope_theta", 1000000.0),
            qk_norm=d.get("qk_norm", False),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rope_type": self.rope_type,
            "rope_fraction": self.rope_fraction,
            "rope_theta": self.rope_theta,
            "qk_norm": self.qk_norm,
        }


@dataclass
class GatherConfig:
    """Final gather layer (spec §3.3/§3.4): same p-RoPE family as the GLOBAL
    layers (uniform positional scheme). Identity init is the LOCKED default."""

    rope_type: str = "prope"
    rope_fraction: float = 0.25
    rope_theta: float = 1000000.0
    qk_norm: bool = False
    init: str = "identity_from_layer_31"  # zeroed o_proj/down_proj from layer-31 copy
    position: int = 32  # [EXPERIMENTAL] flag gather_position: 30 kept as ablation

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GatherConfig":
        _reject_unknown(
            d, {"rope_type", "rope_fraction", "rope_theta", "qk_norm", "init", "position"},
            "gather",
        )
        return cls(
            rope_type=d.get("rope_type", "prope"),
            rope_fraction=d.get("rope_fraction", 0.25),
            rope_theta=d.get("rope_theta", 1000000.0),
            qk_norm=d.get("qk_norm", False),
            init=d.get("init", "identity_from_layer_31"),
            position=d.get("position", 32),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rope_type": self.rope_type,
            "rope_fraction": self.rope_fraction,
            "rope_theta": self.rope_theta,
            "qk_norm": self.qk_norm,
            "init": self.init,
            "position": self.position,
        }


@dataclass
class AttnResConfig:
    """Block Attention Residuals (spec §3.5). Sources are delta-sums, never
    residual-stream snapshots; pseudo-queries are zero-initialized so the
    mechanism is an exact no-op at step 0."""

    enabled: bool = True
    scope: str = "all_layers"  # FLEXIBLE: "all_layers" (66 points) | "globals_only" (9)
    sources: str = "embedding_plus_block_delta_sums_plus_partial"
    keys: str = "rmsnorm"
    values: str = "raw"
    pseudo_query_init: str = "zeros"
    gate: bool = False  # [EXPERIMENTAL] scalar gate per application point, default OFF

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AttnResConfig":
        _reject_unknown(
            d,
            {"enabled", "scope", "sources", "keys", "values", "pseudo_query_init", "gate"},
            "attn_res",
        )
        return cls(
            enabled=d.get("enabled", True),
            scope=d.get("scope", "all_layers"),
            sources=d.get("sources", "embedding_plus_block_delta_sums_plus_partial"),
            keys=d.get("keys", "rmsnorm"),
            values=d.get("values", "raw"),
            pseudo_query_init=d.get("pseudo_query_init", "zeros"),
            gate=d.get("gate", False),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "scope": self.scope,
            "sources": self.sources,
            "keys": self.keys,
            "values": self.values,
            "pseudo_query_init": self.pseudo_query_init,
            "gate": self.gate,
        }


def _is_prime(n: int) -> bool:
    """Trial division; table sizes are <= ~2^21, so this is trivially cheap."""
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    r = int(n**0.5)
    f = 3
    while f <= r:
        if n % f == 0:
            return False
        f += 2
    return True


@dataclass
class EngramConfig:
    """Engram sidecar tables (spec §3.6 + annex A1): n-gram content-addressable
    memory, co-trained from step 0, host-RAM resident, single injection at the
    output of layer `injection_point` (the block-1 GLOBAL), readout
    g·U(RMSNorm(concat(head rows))) with zero-init U.

    Annex A1 pins: prime row counts (distinct per head for orders >= 2;
    the unigram modulus is DERIVED — smallest prime >= |V'| from the canon
    artifact, or >= vocab_size when compression is off), NFKC+casefold
    canonical-id compression (default ON), row init Uniform(-0.01, 0.01),
    rows in their own optimizer group (LR x lr_mult, WD 0).
    """

    enabled: bool = False
    orders: list[int] = None  # type: ignore[assignment]  # default [1, 2, 3]
    heads_per_order: int = 2
    row_dim: int = 256
    # Per-order list of per-head PRIME row counts (annex A1.4). Order 1 is
    # derived from the canon artifact; pinning it here is an override.
    rows_per_head: dict[int, list[int]] = None  # type: ignore[assignment]
    injection_point: int = 3  # output of layer 3 = block-1 GLOBAL
    canonical_compression: bool = True  # annex A1.2 [FLEXIBLE, default ON]
    canon_path: str = "train/src/engram/assets/canon_llama31_v1.npy"
    canon_sha256: str = ""  # pinned after the canon build; verified at table build
    lr_mult: float = 5.0  # annex A1.7: rows LR = lr_mult x base LR, WD 0

    def __post_init__(self) -> None:
        if self.orders is None:
            self.orders = [1, 2, 3]
        if self.rows_per_head is None:
            self.rows_per_head = {2: [1048573, 1048571], 3: [1048559, 1048549]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EngramConfig":
        _reject_unknown(
            d,
            {"enabled", "orders", "heads_per_order", "row_dim", "rows_per_head",
             "injection_point", "canonical_compression", "canon_path",
             "canon_sha256", "lr_mult"},
            "engram",
        )
        rph = d.get("rows_per_head")
        parsed_rph = None
        if rph:
            # JSON/YAML object keys are strings; values may be int or [int].
            parsed_rph = {}
            for k, v in rph.items():
                vv = v if isinstance(v, list) else [v]
                parsed_rph[int(k)] = [int(x) for x in vv]
        return cls(
            enabled=d.get("enabled", False),
            orders=[int(o) for o in d["orders"]] if "orders" in d else None,
            heads_per_order=d.get("heads_per_order", 2),
            row_dim=d.get("row_dim", 256),
            rows_per_head=parsed_rph,
            injection_point=d.get("injection_point", 3),
            canonical_compression=d.get("canonical_compression", True),
            canon_path=d.get("canon_path", cls.canon_path),
            canon_sha256=d.get("canon_sha256", ""),
            lr_mult=d.get("lr_mult", 5.0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "orders": list(self.orders),
            "heads_per_order": self.heads_per_order,
            "row_dim": self.row_dim,
            "rows_per_head": {str(k): list(v) for k, v in self.rows_per_head.items()},
            "injection_point": self.injection_point,
            "canonical_compression": self.canonical_compression,
            "canon_path": self.canon_path,
            "canon_sha256": self.canon_sha256,
            "lr_mult": self.lr_mult,
        }

    def validate(self) -> None:
        if not self.orders or sorted(set(self.orders)) != sorted(self.orders):
            raise ValueError(f"engram.orders must be unique, got {self.orders}")
        if any(o < 1 for o in self.orders):
            raise ValueError(f"engram.orders must be >= 1, got {self.orders}")
        if self.heads_per_order < 1:
            raise ValueError("engram.heads_per_order must be >= 1")
        if self.row_dim < 1:
            raise ValueError("engram.row_dim must be positive")
        for order, rows in self.rows_per_head.items():
            if order not in self.orders:
                raise ValueError(
                    f"engram.rows_per_head has order {order} not in orders {self.orders}"
                )
            if len(rows) != self.heads_per_order:
                raise ValueError(
                    f"engram.rows_per_head[{order}] has {len(rows)} entries, "
                    f"expected heads_per_order={self.heads_per_order}"
                )
            bad = [m for m in rows if not _is_prime(m)]
            if bad:
                raise ValueError(
                    f"engram.rows_per_head[{order}] must be prime (annex A1.4); "
                    f"non-prime: {bad}"
                )
        if len({m for rows in self.rows_per_head.values() for m in rows}) != sum(
            len(rows) for rows in self.rows_per_head.values()
        ):
            raise ValueError("engram row counts must be distinct primes per (order, head)")
        if self.lr_mult <= 0:
            raise ValueError("engram.lr_mult must be positive")


@dataclass
class ModelConfig:
    """Top-level model config implementing spec v2.0 §3.1 exactly."""

    vocab_size: int = 128256
    hidden_size: int = 4096
    num_hidden_layers: int = 33
    layer_types: Optional[list[str]] = None  # default: 8x [swa,swa,swa,global] + [gather]
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 14336
    rms_norm_eps: float = 1e-05
    tie_word_embeddings: bool = False
    swa: Optional[SWAConfig] = None
    global_: Optional[GlobalConfig] = None  # serialized as "global"
    gather: Optional[GatherConfig] = None
    attn_res: Optional[AttnResConfig] = None
    engram: Optional[EngramConfig] = None

    def __post_init__(self) -> None:
        if self.layer_types is None:
            self.layer_types = ["swa", "swa", "swa", "global"] * 8 + ["gather"]
        if self.swa is None:
            self.swa = SWAConfig()
        if self.global_ is None:
            self.global_ = GlobalConfig()
        if self.gather is None:
            self.gather = GatherConfig()
        if self.attn_res is None:
            self.attn_res = AttnResConfig()
        if self.engram is None:
            self.engram = EngramConfig()
        self.validate()

    def validate(self) -> None:
        """Structural invariants derivable from the spec. Raises ValueError."""
        assert self.layer_types is not None
        assert self.swa is not None and self.gather is not None and self.attn_res is not None

        bad = [t for t in self.layer_types if t not in VALID_LAYER_TYPES]
        if bad:
            raise ValueError(f"layer_types: invalid entries {sorted(set(bad))}; "
                             f"must be one of {VALID_LAYER_TYPES}")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"num_hidden_layers={self.num_hidden_layers} but "
                f"layer_types has {len(self.layer_types)} entries"
            )
        n_gather = self.layer_types.count("gather")
        if n_gather != 1:
            raise ValueError(f"expected exactly 1 gather layer, got {n_gather}")
        if self.layer_types.index("gather") != self.gather.position:
            raise ValueError(
                f"gather.position={self.gather.position} does not match the 'gather' "
                f"entry at index {self.layer_types.index('gather')} in layer_types"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.head_dim * self.num_attention_heads != self.hidden_size:
            raise ValueError("head_dim * num_attention_heads must equal hidden_size")
        if self.swa.window <= 0:
            raise ValueError("swa.window must be positive")
        if self.swa.sink.type not in VALID_SINK_TYPES:
            raise ValueError(f"swa.sink.type must be one of {VALID_SINK_TYPES}")
        if self.attn_res.scope not in VALID_ATTN_RES_SCOPES:
            raise ValueError(f"attn_res.scope must be one of {VALID_ATTN_RES_SCOPES}")
        if self.gather.init not in VALID_GATHER_INITS:
            raise ValueError(f"gather.init must be one of {VALID_GATHER_INITS}")
        if self.gather.qk_norm or self.global_.qk_norm:  # type: ignore[union-attr]
            raise ValueError("QK-norm is LOCKED off for GLOBAL/GATHER layers (spec §3.2)")
        for name, sub in (("global", self.global_), ("gather", self.gather)):
            assert sub is not None
            if sub.rope_type != "prope":
                raise ValueError(f"{name}.rope_type must be 'prope' (NoPE is retired, spec v2.0)")
            rd = self.head_dim * sub.rope_fraction
            if not (0.0 < sub.rope_fraction <= 1.0) or rd != int(rd) or int(rd) % 2 or rd < 2:
                raise ValueError(
                    f"{name}.rope_fraction={sub.rope_fraction} gives rotary slice {rd} "
                    f"at head_dim={self.head_dim}; must be a positive even integer"
                )
            if sub.rope_theta <= 0:
                raise ValueError(f"{name}.rope_theta must be positive")
        assert self.engram is not None
        self.engram.validate()
        if self.engram.enabled:
            missing = [o for o in self.engram.orders if o >= 2 and o not in self.engram.rows_per_head]
            if missing:
                raise ValueError(
                    f"engram.orders {missing} (>= 2) need explicit prime rows_per_head"
                )
            if not (0 <= self.engram.injection_point < self.num_hidden_layers):
                raise ValueError(
                    f"engram.injection_point {self.engram.injection_point} outside "
                    f"[0, {self.num_hidden_layers})"
                )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModelConfig":
        _reject_unknown(
            d,
            {
                "vocab_size", "hidden_size", "num_hidden_layers", "layer_types",
                "num_attention_heads", "num_key_value_heads", "head_dim",
                "intermediate_size", "rms_norm_eps", "tie_word_embeddings",
                "swa", "global", "gather", "attn_res", "engram",
            },
            "model",
        )
        return cls(
            vocab_size=d.get("vocab_size", 128256),
            hidden_size=d.get("hidden_size", 4096),
            num_hidden_layers=d.get("num_hidden_layers", 33),
            layer_types=list(d["layer_types"]) if "layer_types" in d else None,
            num_attention_heads=d.get("num_attention_heads", 32),
            num_key_value_heads=d.get("num_key_value_heads", 8),
            head_dim=d.get("head_dim", 128),
            intermediate_size=d.get("intermediate_size", 14336),
            rms_norm_eps=d.get("rms_norm_eps", 1e-05),
            tie_word_embeddings=d.get("tie_word_embeddings", False),
            swa=SWAConfig.from_dict(d["swa"]) if "swa" in d else None,
            global_=GlobalConfig.from_dict(d["global"]) if "global" in d else None,
            gather=GatherConfig.from_dict(d["gather"]) if "gather" in d else None,
            attn_res=AttnResConfig.from_dict(d["attn_res"]) if "attn_res" in d else None,
            engram=EngramConfig.from_dict(d["engram"]) if "engram" in d else None,
        )

    def to_dict(self) -> dict[str, Any]:
        assert self.layer_types is not None
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "layer_types": list(self.layer_types),
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "intermediate_size": self.intermediate_size,
            "rms_norm_eps": self.rms_norm_eps,
            "tie_word_embeddings": self.tie_word_embeddings,
            "swa": self.swa.to_dict(),  # type: ignore[union-attr]
            "global": self.global_.to_dict(),  # type: ignore[union-attr]
            "gather": self.gather.to_dict(),  # type: ignore[union-attr]
            "attn_res": self.attn_res.to_dict(),  # type: ignore[union-attr]
            "engram": self.engram.to_dict(),  # type: ignore[union-attr]
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


def load_config(path: Union[str, Path]) -> ModelConfig:
    """Load a ModelConfig from a YAML or JSON file."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        data = json.loads(text)
    elif path.suffix in (".yaml", ".yml"):
        data = yaml.safe_load(text)
    else:
        raise ValueError(f"unsupported config extension: {path.suffix} (use .yaml/.yml/.json)")
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level config must be a mapping")
    return ModelConfig.from_dict(data)


# ---------------------------------------------------------------------------
# Training-phase config (M4). Phases are config diffs like everything else.
# ---------------------------------------------------------------------------


@dataclass
class KnobSchedule:
    """One anneal knob's schedule over a run (spec §7.2 — ablation tooling,
    default final-state). `schedule`:
    - window: "linear" (default, linear-in-steps from full seq len to final)
    - theta:  progress 0 -> 1 (the log-space interpolation between donor and
      final inv_freq happens inside the model; this is just step progress)
    """

    start_step: int = 0
    end_step: int = 0  # 0/None-ish guard handled by validation
    schedule: str = "linear"

    @classmethod
    def from_dict(cls, d: dict[str, Any], where: str) -> "KnobSchedule":
        _reject_unknown(d, {"start_step", "end_step", "schedule"}, where)
        ks = cls(
            start_step=d.get("start_step", 0),
            end_step=d.get("end_step", 0),
            schedule=d.get("schedule", "linear"),
        )
        if ks.end_step <= ks.start_step:
            raise ValueError(f"{where}: end_step must be > start_step")
        return ks

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_step": self.start_step,
            "end_step": self.end_step,
            "schedule": self.schedule,
        }


@dataclass
class TrainConfig:
    """Phase configuration (trainer + anneal driver + data + KD loss knobs)."""

    model: str = "train/configs/model/llama_8bpp_v1.yaml"  # path to model config
    data_shards: Optional[list[str]] = None
    seq_len: int = 4096
    shuffle: bool = True
    data_seed: int = 0

    steps: int = 1000
    batch_size: int = 1  # sequences per optimizer step (x grad_accum)
    grad_accum: int = 1
    lr: float = 2e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95

    # KD loss (spec §6.2): L = alpha * KL_lumped-topk + (1-alpha) * CE(gold).
    # alpha is the run default; per-slice overrides come from shard sidecars.
    alpha: float = 1.0
    temperature: float = 1.0
    loss_chunk_size: int = 512

    bf16: bool = True  # bf16 compute / fp32 masters
    optimizer: str = "adamw8bit"  # "adamw8bit" (spec §5.1 default) | "adamw" (fp32 states, dev fallback)
    grad_checkpointing: bool = False
    seed: int = 0

    init: str = "warm"  # "warm" (donor furniture init, spec §3.4) | "scratch" (ablation)
    donor_path: str = "OriginalModel"

    # Anneal knobs (spec §7.2 ablation tooling); absent = final topology for
    # the whole run.
    anneal_window: Optional[KnobSchedule] = None
    anneal_theta: Optional[KnobSchedule] = None

    out_dir: str = "train/runs/phase0"
    checkpoint_every: int = 500
    log_every: int = 10
    log_filename: str = ""  # default: <out_dir>/train.log

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainConfig":
        _reject_unknown(
            d,
            {
                "model", "data_shards", "seq_len", "shuffle", "data_seed",
                "steps", "batch_size", "grad_accum", "lr", "min_lr_ratio",
                "warmup_steps", "weight_decay", "grad_clip", "beta1", "beta2",
                "alpha", "temperature", "loss_chunk_size", "bf16", "optimizer",
                "grad_checkpointing", "seed", "init", "donor_path",
                "anneal_window", "anneal_theta",
                "out_dir", "checkpoint_every", "log_every", "log_filename",
            },
            "train",
        )
        cfg = cls(
            model=d.get("model", "train/configs/model/llama_8bpp_v1.yaml"),
            data_shards=list(d["data_shards"]) if "data_shards" in d else None,
            seq_len=d.get("seq_len", 4096),
            shuffle=d.get("shuffle", True),
            data_seed=d.get("data_seed", 0),
            steps=d.get("steps", 1000),
            batch_size=d.get("batch_size", 1),
            grad_accum=d.get("grad_accum", 1),
            lr=d.get("lr", 2e-4),
            min_lr_ratio=d.get("min_lr_ratio", 0.1),
            warmup_steps=d.get("warmup_steps", 100),
            weight_decay=d.get("weight_decay", 0.1),
            grad_clip=d.get("grad_clip", 1.0),
            beta1=d.get("beta1", 0.9),
            beta2=d.get("beta2", 0.95),
            alpha=d.get("alpha", 1.0),
            temperature=d.get("temperature", 1.0),
            loss_chunk_size=d.get("loss_chunk_size", 512),
            bf16=d.get("bf16", True),
            optimizer=d.get("optimizer", "adamw8bit"),
            grad_checkpointing=d.get("grad_checkpointing", False),
            seed=d.get("seed", 0),
            init=d.get("init", "warm"),
            donor_path=d.get("donor_path", "OriginalModel"),
            anneal_window=KnobSchedule.from_dict(d["anneal_window"], "anneal_window")
            if "anneal_window" in d else None,
            anneal_theta=KnobSchedule.from_dict(d["anneal_theta"], "anneal_theta")
            if "anneal_theta" in d else None,
            out_dir=d.get("out_dir", "train/runs/phase0"),
            checkpoint_every=d.get("checkpoint_every", 500),
            log_every=d.get("log_every", 10),
            log_filename=d.get("log_filename", ""),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.steps <= 0 or self.batch_size <= 0 or self.grad_accum <= 0:
            raise ValueError("steps/batch_size/grad_accum must be positive")
        if self.lr <= 0 or self.warmup_steps < 0 or self.warmup_steps >= self.steps:
            raise ValueError("need lr > 0 and 0 <= warmup_steps < steps")
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1]")
        if not (1.0 <= self.temperature <= 4.0):
            raise ValueError("temperature out of sane range (spec: 1-2)")
        if self.optimizer not in ("adamw", "adamw8bit"):
            raise ValueError("optimizer must be 'adamw' or 'adamw8bit'")
        if self.init not in ("warm", "scratch"):
            raise ValueError("init must be 'warm' or 'scratch'")
        if self.seq_len < 2:
            raise ValueError("seq_len must be >= 2")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "data_shards": self.data_shards,
            "seq_len": self.seq_len,
            "shuffle": self.shuffle,
            "data_seed": self.data_seed,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "grad_accum": self.grad_accum,
            "lr": self.lr,
            "min_lr_ratio": self.min_lr_ratio,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "grad_clip": self.grad_clip,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "alpha": self.alpha,
            "temperature": self.temperature,
            "loss_chunk_size": self.loss_chunk_size,
            "bf16": self.bf16,
            "optimizer": self.optimizer,
            "grad_checkpointing": self.grad_checkpointing,
            "seed": self.seed,
            "init": self.init,
            "donor_path": self.donor_path,
            "anneal_window": self.anneal_window.to_dict() if self.anneal_window else None,
            "anneal_theta": self.anneal_theta.to_dict() if self.anneal_theta else None,
            "out_dir": self.out_dir,
            "checkpoint_every": self.checkpoint_every,
            "log_every": self.log_every,
            "log_filename": self.log_filename,
        }


def load_train_config(path: Union[str, Path]) -> TrainConfig:
    """Load a TrainConfig from a YAML or JSON file."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        data = json.loads(text)
    elif path.suffix in (".yaml", ".yml"):
        data = yaml.safe_load(text)
    else:
        raise ValueError(f"unsupported config extension: {path.suffix}")
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level config must be a mapping")
    return TrainConfig.from_dict(data)
