"""Engram trainer-integration tests (dev_tiny, engram enabled): loss
decreases, row LR = lr_mult x base with WD 0, gates in the no-WD device
group, only touched rows change, and resume is bitwise including the host
tables + sparse row optimizer (I9)."""
from pathlib import Path

import numpy as np
import torch
import yaml

from train.src.config import TrainConfig, load_config
from train.src.data.topk_writer import ShardWriter
from train.src.train.trainer import Trainer, lr_at

DEV_TINY = "train/configs/model/dev_tiny.yaml"
VOCAB = 1024


def _engram_model_yaml(tmp_path: Path) -> str:
    d = load_config(DEV_TINY).to_dict()
    d["engram"]["enabled"] = True
    p = tmp_path / "dev_tiny_engram.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.dump(d), encoding="utf-8")
    return str(p)


def _learnable_shard(root: Path, name: str, n_docs: int = 8, t: int = 128,
                     k: int = 4, seed: int = 3) -> Path:
    """Same learnable synthetic shard as the M4 trainer tests (top-1 = true
    next token with 0.8 mass)."""
    rng = np.random.default_rng(seed)
    w = ShardWriter(root / name, k=k, teacher_id="synthetic", vocab_size=VOCAB)
    for _ in range(n_docs):
        tokens = rng.integers(0, VOCAB, size=t).astype(np.uint32)
        ids = np.stack([rng.permutation(VOCAB)[:k] for _ in range(t)]).astype(np.uint32)
        ids[:-1, 0] = tokens[1:]
        probs = np.full((t, k), 0.15 / (k - 1), dtype=np.float32)
        probs[:, 0] = 0.8
        w.add_document(tokens, ids, probs, loss_mask=np.ones(t, dtype=np.uint8))
    w.finalize()
    return root / name


def _train_cfg(tmp_path: Path, shards: list[str], steps: int) -> dict:
    return {
        "model": _engram_model_yaml(tmp_path), "data_shards": shards,
        "seq_len": 32, "steps": steps, "batch_size": 2, "lr": 3e-3,
        "warmup_steps": min(5, steps - 1), "alpha": 1.0, "loss_chunk_size": 16,
        "bf16": False, "seed": 0, "init": "scratch",
        "out_dir": str(tmp_path / "run"), "checkpoint_every": steps,
        "log_every": 5, "log_filename": str(tmp_path / "run.log"),
    }


def _losses(log_file: str) -> list[float]:
    out = []
    for line in Path(log_file).read_text().splitlines():
        if "loss " in line:
            out.append(float(line.split("loss ")[1].split()[0]))
    return out


def test_engram_smoke_loss_decreases(tmp_path):
    shard = _learnable_shard(tmp_path, "s")
    cfg_d = _train_cfg(tmp_path, [str(shard)], steps=30)
    Trainer(TrainConfig.from_dict(cfg_d), device="cpu").train()
    losses = _losses(cfg_d["log_filename"])
    assert len(losses) >= 5
    early = sum(losses[:2]) / 2
    late = sum(losses[-2:]) / 2
    assert late < early * 0.98, f"loss did not decrease: {early:.4f} -> {late:.4f}"


def test_engram_row_lr_and_gate_groups(tmp_path):
    """Annex A1.7: rows get lr_mult x base LR (WD 0, sparse optimizer); the
    per-order gate scalars sit in a WD-0 device param group."""
    shard = _learnable_shard(tmp_path, "s")
    cfg_d = _train_cfg(tmp_path, [str(shard)], steps=2)
    cfg_d["grad_accum"] = 2  # exercise the annex v1.1 cadence path
    cfg = TrainConfig.from_dict(cfg_d)
    trainer = Trainer(cfg, device="cpu")

    # Device groups: exactly one WD-0 group, holding exactly the gate scalars.
    wd0 = [g for g in trainer.optimizer.param_groups if g["weight_decay"] == 0.0]
    assert len(wd0) == 1
    gate_ids = {id(p) for p in trainer.model.engram.gates.parameters()}
    assert {id(p) for p in wd0[0]["params"]} == gate_ids

    # Row LR seen by the sparse optimizer = lr_mult x schedule LR, stepped
    # once per dense-equivalent step (annex v1.1 cadence), so with
    # grad_accum=2 there are exactly `steps` calls, not 2x.
    seen = []
    orig_step = trainer.row_optimizer.step
    def spy(lr):
        seen.append(lr)
        return orig_step(lr)
    trainer.row_optimizer.step = spy
    trainer.train()
    assert seen and abs(seen[0] - lr_at(0, cfg) * 5.0) < 1e-12
    assert len(seen) == cfg.steps


def test_engram_only_touched_rows_change(tmp_path):
    shard = _learnable_shard(tmp_path, "s")
    cfg_d = _train_cfg(tmp_path, [str(shard)], steps=2)
    trainer = Trainer(TrainConfig.from_dict(cfg_d), device="cpu")
    tables = trainer.model.engram_tables
    before = {k: v.clone() for k, v in tables.rows.items()}
    trainer.train()
    for key in tables.table_keys:
        changed = (tables.rows[key] != before[key]).any(dim=1).numpy()
        touched = tables.touch[key] > 0
        # Only ever-touched rows may change (some touched rows land at
        # boundary-invalid positions only -> zero grad -> unchanged).
        assert (changed <= touched).all(), key
        assert changed.any(), key


def test_engram_resume_bitwise(tmp_path):
    """I9 with the sidecar: interrupt at 10, resume to 15 — losses AND host
    tables bitwise-match the uninterrupted 15-step run."""
    shard = _learnable_shard(tmp_path, "s")

    full = _train_cfg(tmp_path / "full", [str(shard)], steps=15)
    trainer_full = Trainer(TrainConfig.from_dict(full), device="cpu")
    trainer_full.train()

    part = _train_cfg(tmp_path / "part", [str(shard)], steps=15)
    trainer = Trainer(TrainConfig.from_dict(part), device="cpu")
    trainer.train(max_steps=10)
    ckpt = trainer.save_checkpoint("ckpt.pt")

    res = _train_cfg(tmp_path / "res", [str(shard)], steps=15)
    trainer2 = Trainer(TrainConfig.from_dict(res), device="cpu")
    trainer2.load_checkpoint(ckpt)
    trainer2.train()

    assert trainer2.step == 15
    assert _losses(full["log_filename"])[-1] == _losses(res["log_filename"])[-1]
    for key in trainer_full.model.engram_tables.table_keys:
        assert torch.equal(
            trainer_full.model.engram_tables.rows[key],
            trainer2.model.engram_tables.rows[key],
        ), key
