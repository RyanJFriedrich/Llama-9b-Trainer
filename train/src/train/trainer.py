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
    """Linear warmup -> cosine decay to min_lr_ratio * lr."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    p = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    cos = 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * min(p, 1.0)))).item()
    return cfg.lr * (cfg.min_lr_ratio + (1.0 - cfg.min_lr_ratio) * cos)


def _batch_tensors(seqs: list[dict[str, torch.Tensor]], device) -> dict[str, torch.Tensor]:
    return {k: torch.stack([s[k] for s in seqs]).to(device) for k in seqs[0]}


class Trainer:
    def __init__(self, cfg: TrainConfig, device: Optional[str] = None) -> None:
        cfg.validate()
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.out_dir = Path(cfg.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = cfg.log_filename or str(self.out_dir / "train.log")

        torch.manual_seed(cfg.seed)

        model_cfg = load_config(cfg.model)
        self.model = RefitModel(model_cfg)
        # Warm start (cfg.init == "warm") is applied by the caller/script via
        # tools/warm_start.py — it owns the donor path and the accounting
        # report. See scripts/train_phase0.py.
        self.model.to(self.device)  # fp32 masters; bf16 via autocast
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
        self._fwd = (
            torch.compile(self.model, dynamic=False) if cfg.torch_compile else self.model
        )

        # Engram (spec §3.6, annex A1.7): device params split into two groups
        # — the per-order gate scalars get WD 0, everything else the run WD.
        # The TABLES are not torch params at all: they are host-resident and
        # updated by a sparse row optimizer (LR x lr_mult, WD 0).
        self.engram_cfg = model_cfg.engram if model_cfg.engram.enabled else None
        params: Any = self.model.parameters()
        if self.engram_cfg is not None:
            gate_params = list(self.model.engram.gates.parameters())
            gate_ids = {id(p) for p in gate_params}
            params = [
                {"params": [p for p in self.model.parameters() if id(p) not in gate_ids],
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

        log(f"run metadata: config={json.dumps(cfg.to_dict())} seed={cfg.seed} "
            f"code_hash={code_hash()}", filename=self.log_file)

    def _seqs_to_loss_inputs(self, batch: dict[str, torch.Tensor]):
        tokens = batch["tokens"]
        gold = torch.cat([tokens[:, 1:], tokens[:, :1]], dim=1)  # shifted; last masked
        return tokens, gold

    def train(self, max_steps: Optional[int] = None) -> None:
        cfg = self.cfg
        steps = max_steps or cfg.steps
        self.model.train()
        accum = cfg.grad_accum
        batch_seqs: list[dict[str, torch.Tensor]] = []
        t0 = time.time()
        n_tokens = 0

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
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(cfg.bf16 and self.device == "cuda")):
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
                n_tokens += int(batch["loss_mask"].sum())

            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.optimizer.step()
            if self.engram_cfg is not None:
                self.row_optimizer.step(
                    lr=lr_at(self.step, cfg) * self.engram_cfg.lr_mult
                )
            self.step += 1
            batch_seqs = []

            if self.step % cfg.log_every == 0 or self.step == 1:
                tok_s = n_tokens / max(time.time() - t0, 1e-9)
                eng = ""
                if self.engram_cfg is not None:
                    tel = self.row_optimizer.pop_telemetry()
                    eng = f" eng_round {tel['engram_bf16_rounding_loss']:.3f}"
                log(f"step {self.step}/{steps} loss {loss_val:.4f} "
                    f"lr {lr_at(self.step - 1, cfg):.2e} gnorm {grad_norm:.3f} "
                    f"window {state['window']} "
                    f"theta {state['theta_progress']:.3f} tok/s {tok_s:.0f}{eng}",
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
        torch.set_rng_state(ckpt["rng"])
        if ckpt.get("cuda_rng") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
        if self.engram_cfg is not None:
            self.model.engram_tables.load_state_dict(ckpt["engram_tables"])
            self.row_optimizer.load_state_dict(ckpt["engram_row_opt"])
        log(f"checkpoint loaded: {path} (resuming at step {self.step})",
            filename=self.log_file, print_console=True)
