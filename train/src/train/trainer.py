"""Trainer loop (M4) — Phase 0 machinery.

bf16 autocast compute over fp32 master weights, AdamW, cosine LR with linear
warmup, grad clip, gradient accumulation, anneal driver, resume-safe
checkpoints (model + optimizer + step + RNG + anneal state), and full
run metadata logging (config + seed + code hash — standing rule 6).

Memory note (spec §0.5 v1.2): fp32 AdamW states cost 16 B/param ≈ 132 GB at
8.25B and do NOT fit a 96 GB card. `optimizer: adamw8bit` (optim8bit.py)
brings states to 2 B/param (≈82.5 GB profile). fp32 AdamW remains the dev
default; the 8B run must use the 8-bit (or offloaded) path. Logits are the
activation hotspot — the fused KD loss handles that (distill/kd_loss.py).

Data contract: TopKLoader windows (tokens, topk_idx, topk_w, tail_w,
loss_mask). Gold targets are tokens shifted left by one; the loader already
masks doc boundaries and window tails.
"""
from __future__ import annotations

import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

import torch

from train.src.config import TrainConfig, load_config
from train.src.data.topk_loader import TopKLoader
from train.src.distill.kd_loss import kd_loss
from train.src.model.refit import RefitModel
from train.src.train.anneal import AnnealDriver
from train.utils.log import log


def code_hash() -> str:
    """git HEAD of the harness, for run metadata (standing rule 6)."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "nogit"


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup -> cosine decay to min_lr_ratio * lr over cosine_steps
    (default: the whole run), then flat at the floor — the owner-preferred
    "anneal epoch 1, steady state after" shape when cosine_steps is set."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    span = cfg.cosine_steps or cfg.steps
    p = (step - cfg.warmup_steps) / max(1, span - cfg.warmup_steps)
    cos = 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * min(p, 1.0)))).item()
    return cfg.lr * (cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cos)


def _format_eta(seconds: float) -> str:
    if seconds <= 0 or math.isinf(seconds) or math.isnan(seconds):
        return "--"
    sec = int(round(seconds))
    d, sec = divmod(sec, 86400)
    h, sec = divmod(sec, 3600)
    m, s = divmod(sec, 60)
    if d > 0:
        return f"{d}d {h:02d}h"
    if h > 0:
        return f"{h}h {m:02d}m"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _format_tokens(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1e9:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}K"
    return str(n)


def _batch_tensors(seqs: list[dict[str, torch.Tensor]], device) -> dict[str, torch.Tensor]:
    non_blocking = device != "cpu" and torch.cuda.is_available()
    res = {}
    for k in seqs[0]:
        t = torch.stack([s[k] for s in seqs])
        if non_blocking:
            t = t.pin_memory()
        res[k] = t.to(device, non_blocking=non_blocking)
    return res


class Trainer:
    def __init__(self, cfg: TrainConfig, device: Optional[str] = None) -> None:
        cfg.validate()
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.out_dir = Path(cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = cfg.log_filename or str(self.out_dir / "train.log")

        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")

        model_cfg = load_config(cfg.model)
        self.model = RefitModel(model_cfg)
        # Warm start (cfg.init == "warm") is applied by the caller/script via
        # tools/warm_start.py — it owns the donor path and the accounting
        # report. See scripts/train_phase0.py.
        self.model.to(self.device)
        if cfg.master_dtype == "bf16":
            self.model.to(dtype=torch.bfloat16)
            log("master_dtype: bf16 master weights (saving ~16.5 GB VRAM)",
                filename=self.log_file, print_console=True)
        self.model.grad_checkpointing = cfg.grad_checkpointing
        if cfg.precision == "fp8":
            # spec §4: attention/FFN GEMMs only; weight Parameter objects are
            # shared, so optimizer groups / state dicts / warm start are all
            # unaffected by the swap.
            from train.src.train.fp8 import apply_fp8
            apply_fp8(self.model)

        # Optional torch.compile (perf knob, config-driven): compile a
        # FORWARD CALLABLE used by the training loop only — self.model stays
        # the raw module, so state_dict/checkpoint/warm-start formats are
        # untouched and eval/probe paths stay eager (no recompiles from their
        # different shapes). dynamic=False: training batches are a fixed
        # [B, seq_len], so each run builds one static graph. (Anneal ablation
        # runs mutate attn.window between steps → recompiles; not a target
        # combo.)
        if cfg.torch_compile:
            try:
                import torch._inductor.config as inductor_config
                inductor_config.max_autotune = True
            except Exception:
                pass
            self._fwd = torch.compile(self.model, dynamic=False)
        else:
            self._fwd = self.model

        # Frozen donor furniture (owner decision 2026-09-03, TrainConfig
        # freeze_embeddings): embed_tokens and lm_head keep the donor's fixed
        # input/output representations — the new interior learns to bridge
        # them. requires_grad=False keeps them out of the optimizer groups
        # below AND skips their fp32 grads entirely (~4.2 GB at this scale).
        if cfg.freeze_embeddings:
            self.model.model.embed_tokens.weight.requires_grad_(False)
            self.model.lm_head.weight.requires_grad_(False)
            log("freeze_embeddings: embed_tokens + lm_head frozen "
                "(requires_grad=False)", filename=self.log_file, print_console=True)

        # bf16 gradients (owner decision 2026-09-03, TrainConfig grad_dtype):
        # grad_dtype makes autograd cast each param's grad to bf16 as it
        # lands at the leaf AND accumulate successive micro-batch grads in
        # bf16 — so the fp32 gradient set (~27 GB post-freeze) never exists
        # in memory. This is what reopens grad_accum > 1 on the 96 GB box
        # (bring-up OOM #4). Precision argument on record: the 8-bit AdamW
        # states quantize m/sqrt(v) to int8, so fp32's 24-bit grad mantissa
        # is discarded by the very next op anyway.
        # torch.optim.AdamW (dev fallback) rejects bf16 grads, so at step
        # time its params get a transient fp32 copy (toy scale only).
        self._grad_cast_fp32 = cfg.grad_dtype == "bf16" and cfg.optimizer != "adamw8bit"
        if cfg.grad_dtype == "bf16":
            n_bf16 = 0
            for p in self.model.parameters():
                if p.requires_grad:
                    p.grad_dtype = torch.bfloat16
                    n_bf16 += 1
            log(f"grad_dtype: bf16 grads on {n_bf16} trainable params",
                filename=self.log_file, print_console=True)

        # Engram (spec §3.6, annex A1.7): device params split into two groups
        # — the per-order gate scalars get WD 0, everything else the run WD.
        # The TABLES are not torch params at all: they are host-resident and
        # updated by a sparse row optimizer (LR x lr_mult, WD 0).
        self.engram_cfg = model_cfg.engram if model_cfg.engram.enabled else None
        params: Any = [p for p in self.model.parameters() if p.requires_grad]
        if self.engram_cfg is not None:
            gate_params = list(self.model.engram.gates.parameters())
            gate_ids = {id(p) for p in gate_params}
            params = [
                {"params": [p for p in params if id(p) not in gate_ids],
                 "weight_decay": cfg.weight_decay},
                {"params": gate_params, "weight_decay": 0.0},
            ]
            from train.src.engram.sparse_opt import SparseRowAdamW8bit
            self.row_optimizer = SparseRowAdamW8bit(
                self.model.engram_tables, betas=(cfg.beta1, cfg.beta2)
            )
            from concurrent.futures import ThreadPoolExecutor
            self._eng_pool = ThreadPoolExecutor(max_workers=1)  # address prefetch

        if cfg.optimizer == "adamw8bit":
            # spec §0.5 v1.2: 8-bit states (2 B/param) — required to fit the
            # 96 GB deploy box; fp32 AdamW (16 B/param ≈ 132 GB) does not.
            from train.src.train.optim8bit import AdamW8bit
            self.optimizer = AdamW8bit(
                params, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                weight_decay=cfg.weight_decay,
            )
        else:
            self.optimizer = torch.optim.AdamW(
                params, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                weight_decay=cfg.weight_decay,
            )
        self.anneal = AnnealDriver(cfg, self.model)

        if not cfg.data_shards:
            raise ValueError("train config needs data_shards")
        self.loader = TopKLoader(cfg.data_shards, seq_len=cfg.seq_len,
                                 shuffle=cfg.shuffle, seed=cfg.data_seed)
        # spec §6.4 (v1.2): folded-v1 shards already embed an ad-hoc CE blend
        # — warn if the explicit CE mix would double-count it.
        for shard in self.loader.shards:
            if shard.fold_version == "v1" and cfg.alpha < 1.0:
                log(f"WARNING: shard {shard.dir} is fold_version v1 (tail folded "
                    f"into gold) but alpha={cfg.alpha} < 1 — the explicit CE term "
                    f"double-counts the fold; consider alpha=1.0 for v1-only runs",
                    filename=self.log_file, print_console=True)
        self.step = 0
        self.loss_ema: Optional[float] = None
        self.tok_s_ema: Optional[float] = None
        self.step_time_ema: Optional[float] = None
        self.total_hw_tokens: int = 0
        self.total_loss_tokens: int = 0

        log(f"run metadata: config={json.dumps(cfg.to_dict())} seed={cfg.seed} "
            f"code_hash={code_hash()}", filename=self.log_file)

    def _seqs_to_loss_inputs(self, batch: dict[str, torch.Tensor]):
        tokens = batch["tokens"]
        gold = torch.cat([tokens[:, 1:], tokens[:, :1]], dim=1)  # shifted; last masked
        return tokens, gold

    def _clip_grads(self, max_norm: float) -> float:
        """Global-norm clip for the bf16-grad regime — same semantics as
        clip_grad_norm_ (norm over ALL params, scale applied only when over)
        but computed with a per-param fp32 transient, because a bf16
        vector_norm accumulated over tens of millions of elements would
        mis-measure the total."""
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        total_sq = 0.0
        for g in grads:
            total_sq += g.float().pow(2).sum().item()
        total = math.sqrt(total_sq)
        if total > max_norm:
            scale = max_norm / (total + 1e-6)
            for g in grads:
                g.mul_(scale)
        return total

    def train(self, max_steps: Optional[int] = None) -> None:
        cfg = self.cfg
        steps = max_steps or cfg.steps
        self.model.train()
        accum = cfg.grad_accum
        batch_seqs: list[dict[str, torch.Tensor]] = []
        t0 = time.time()
        last_log_time = t0
        last_log_hw_tokens = self.total_hw_tokens

        it = iter(self.loader.iter_sequences())
        # Resume-safe data cursor: the window order is deterministic (seeded),
        # so skipping the sequences consumed before the checkpoint restores
        # position. (Valid within a single epoch pass; an epoch wrap mid-run
        # shifts alignment slightly — documented, harmless for long corpora.)
        seqs_per_step = cfg.batch_size * accum
        for _ in range(self.step * seqs_per_step):
            try:
                next(it)
            except StopIteration:
                it = iter(self.loader.iter_sequences())
        while self.step < steps:
            try:
                seq = next(it)
            except StopIteration:
                it = iter(self.loader.iter_sequences())
                continue
            batch_seqs.append(seq)
            if len(batch_seqs) < cfg.batch_size * accum:
                continue

            step_t0 = time.time()
            state = self.anneal.apply(self.step)
            for group in self.optimizer.param_groups:
                group["lr"] = lr_at(self.step, cfg)

            self.optimizer.zero_grad(set_to_none=True)
            loss_val = 0.0

            # Engram prefetch (annex A1.6): addressing is pure host-side
            # integer work — run it off-thread for ALL micro-batches while
            # compute proceeds; the device staging happens on the main thread
            # just-in-time per micro-batch.
            eng_addr = None
            if self.engram_cfg is not None:
                eng_addr = [
                    self._eng_pool.submit(
                        self.model.engram_tables.address,
                        torch.stack([s["tokens"] for s in batch_seqs[a * cfg.batch_size:(a + 1) * cfg.batch_size]]),
                    )
                    for a in range(accum)
                ]

            if cfg.precision == "fp8":
                from train.src.train.fp8 import cache_model_fp8_weights, clear_model_fp8_weights
                cache_model_fp8_weights(self.model)

            for a in range(accum):
                chunk = batch_seqs[a * cfg.batch_size:(a + 1) * cfg.batch_size]
                batch = _batch_tensors(chunk, self.device)
                tokens, gold = self._seqs_to_loss_inputs(batch)
                gb = None
                if self.engram_cfg is not None:
                    idx, valid = eng_addr[a].result()
                    gb = self.model.engram_tables.stage(
                        idx, valid, self.device, requires_grad=True
                    )
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=(cfg.bf16 and self.device == "cuda"),
                                    cache_enabled=cfg.autocast_cache):
                    hidden = self._fwd(tokens, return_hidden=True, engram=gb)
                loss = kd_loss(
                    hidden.float(), self.model.lm_head.weight,
                    batch["topk_idx"], batch["topk_w"], batch["tail_w"], gold,
                    batch["loss_mask"], alpha=cfg.alpha, temperature=cfg.temperature,
                    chunk_size=cfg.loss_chunk_size,
                ) / accum
                loss.backward()
                if self.engram_cfg is not None:
                    # Annex v1.1 cadence: accumulate the staged row grads
                    # host-side; the sparse Adam step runs once per dense-
                    # equivalent step (below), never per micro-batch.
                    self.row_optimizer.accumulate(gb)
                loss_val += loss.item()
                self.total_hw_tokens += batch["tokens"].numel()
                self.total_loss_tokens += int(batch["loss_mask"].sum())

            if cfg.precision == "fp8":
                clear_model_fp8_weights(self.model)

            if self.cfg.grad_dtype == "bf16":
                # bf16 grads: clip via an fp32-transient norm (a bf16
                # vector_norm over 59M elements would mis-clip), then step.
                # AdamW8bit casts each grad to fp32 per param as it reads it;
                # torch.AdamW rejects bf16 grads, so the dev fallback gets a
                # transient fp32 copy of the whole set (toy scale only).
                grad_norm = self._clip_grads(cfg.grad_clip)
                if self._grad_cast_fp32:
                    for p in self.model.parameters():
                        if p.grad is not None:
                            g = p.grad.float()
                            p.grad = None
                            p.grad_dtype = torch.float32
                            p.grad = g
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.optimizer.step()
            if self._grad_cast_fp32:
                # restore bf16 leaf-grad casting for the next backward
                for p in self.model.parameters():
                    if p.requires_grad:
                        p.grad = None
                        p.grad_dtype = torch.bfloat16
            if self.engram_cfg is not None:
                self.row_optimizer.step(
                    lr=lr_at(self.step, cfg) * self.engram_cfg.lr_mult
                )
            self.step += 1
            step_time = max(time.time() - step_t0, 1e-6)
            hw_tok_step = cfg.batch_size * cfg.seq_len * accum
            inst_tok_s = hw_tok_step / step_time

            # Telemetry updates: EMA smoothing for loss and throughput
            if self.loss_ema is None:
                self.loss_ema = loss_val
            else:
                self.loss_ema = 0.9 * self.loss_ema + 0.1 * loss_val

            # Avoid anchoring EMA on Step 1 compile time when torch.compile is active
            if self.step_time_ema is None or (self.step == 2 and cfg.torch_compile):
                self.step_time_ema = step_time
                self.tok_s_ema = inst_tok_s
            else:
                self.step_time_ema = 0.9 * self.step_time_ema + 0.1 * step_time
                self.tok_s_ema = 0.9 * self.tok_s_ema + 0.1 * inst_tok_s

            batch_seqs = []

            if self.step % cfg.log_every == 0 or self.step == 1:
                now = time.time()
                interval_dt = max(now - last_log_time, 1e-6)
                interval_hw_tok = self.total_hw_tokens - last_log_hw_tokens
                interval_tok_s = interval_hw_tok / interval_dt
                last_log_time = now
                last_log_hw_tokens = self.total_hw_tokens

                rem_steps = max(steps - self.step, 0)
                eta = _format_eta(rem_steps * (self.step_time_ema or step_time))
                pct = (self.step / steps) * 100.0

                tok_s_display = interval_tok_s if self.step > 1 else inst_tok_s
                tok_s_ema_val = self.tok_s_ema if self.tok_s_ema is not None else tok_s_display

                eng = ""
                if self.engram_cfg is not None:
                    tel = self.row_optimizer.pop_telemetry()
                    eng = f" eng_round {tel['engram_bf16_rounding_loss']:.3f}"
                mem = ""
                if cfg.mem_debug and self.device == "cuda":
                    mem = (f" mem_alloc {torch.cuda.memory_allocated() / 2**30:.2f}GiB"
                           f" reserved {torch.cuda.memory_reserved() / 2**30:.2f}GiB"
                           f" peak {torch.cuda.max_memory_allocated() / 2**30:.2f}GiB")
                log(f"step {self.step}/{steps} ({pct:.1f}%) loss {loss_val:.4f} (ema {self.loss_ema:.4f}) "
                    f"lr {lr_at(self.step - 1, cfg):.2e} gnorm {grad_norm:.3f} "
                    f"tok/s {tok_s_display:.0f} (ema {tok_s_ema_val:.0f}) "
                    f"eta {eta} tok {_format_tokens(self.total_hw_tokens)} "
                    f"window {state['window']} "
                    f"theta {state['theta_progress']:.3f}{eng}{mem}",
                    filename=self.log_file, print_console=True)

            if self.step % cfg.checkpoint_every == 0 or self.step == steps:
                self.save_checkpoint()

    def save_checkpoint(self, name: Optional[str] = None) -> Path:
        path = self.out_dir / (name or f"ckpt_step{self.step}.pt")
        ckpt: dict[str, Any] = {
            "step": self.step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "anneal_state": self.model.anneal_state,
            "rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "config": self.cfg.to_dict(),
            "code_hash": code_hash(),
            "telemetry": {
                "loss_ema": self.loss_ema,
                "tok_s_ema": self.tok_s_ema,
                "step_time_ema": self.step_time_ema,
                "total_hw_tokens": self.total_hw_tokens,
                "total_loss_tokens": self.total_loss_tokens,
            },
        }
        if self.engram_cfg is not None:
            # I9: host tables + sparse row optimizer are part of the
            # resume-safe state, alongside model/optimizer/RNG/cursor.
            ckpt["engram_tables"] = self.model.engram_tables.state_dict()
            ckpt["engram_row_opt"] = self.row_optimizer.state_dict()
        torch.save(ckpt, path)
        log(f"checkpoint saved: {path} (step {self.step})", filename=self.log_file,
            print_console=True)
        return path

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.step = ckpt["step"]
        self.model.set_anneal_state(**ckpt["anneal_state"])
        rng = ckpt["rng"]
        if isinstance(rng, torch.Tensor):
            rng = rng.cpu()
        torch.set_rng_state(rng)
        if ckpt.get("cuda_rng") and torch.cuda.is_available():
            cuda_rng = [s.cpu() if isinstance(s, torch.Tensor) else s for s in ckpt["cuda_rng"]]
            torch.cuda.set_rng_state_all(cuda_rng)
        if self.engram_cfg is not None:
            tables_sd = {k: (v.cpu() if isinstance(v, torch.Tensor) else v)
                         for k, v in ckpt["engram_tables"].items()}
            self.model.engram_tables.load_state_dict(tables_sd)
            self.row_optimizer.load_state_dict(ckpt["engram_row_opt"])
        if "telemetry" in ckpt:
            telem = ckpt["telemetry"]
            self.loss_ema = telem.get("loss_ema")
            self.tok_s_ema = telem.get("tok_s_ema")
            self.step_time_ema = telem.get("step_time_ema")
            self.total_hw_tokens = telem.get("total_hw_tokens", 0)
            self.total_loss_tokens = telem.get("total_loss_tokens", 0)
        log(f"checkpoint loaded: {path} (resuming at step {self.step})",
            filename=self.log_file, print_console=True)
