"""Base-checkpoint export/import tests (dev_tiny scale): bitwise round-trip,
config-mismatch refusal, engram table restore, and the trainer's
init="prebuilt" path."""
from pathlib import Path

import pytest
import torch

from train.src.config import ModelConfig, TrainConfig, load_config
from train.src.model.refit import RefitModel
from train.src.tools.base_ckpt import load_base_checkpoint, save_base_checkpoint

DEV_TINY = "train/configs/model/dev_tiny.yaml"


def _enabled_cfg() -> ModelConfig:
    d = load_config(DEV_TINY).to_dict()
    d["engram"]["enabled"] = True
    return ModelConfig.from_dict(d)


def _bf16_model(seed: int = 0) -> RefitModel:
    torch.set_default_dtype(torch.bfloat16)
    try:
        torch.manual_seed(seed)
        return RefitModel(_enabled_cfg())
    finally:
        torch.set_default_dtype(torch.float32)


def test_base_ckpt_bitwise_roundtrip(tmp_path):
    model = _bf16_model()
    out = save_base_checkpoint(model, tmp_path / "base", dtype=torch.bfloat16)
    assert (out / "model.safetensors.index.json").exists()
    assert (out / "config.json").exists()
    assert (out / "engram.safetensors").exists()

    fresh = _bf16_model(seed=99)  # different init
    load_base_checkpoint(fresh, out)
    for (n1, p1), (n2, p2) in zip(model.state_dict().items(), fresh.state_dict().items()):
        assert n1 == n2
        assert torch.equal(p1, p2), n1
    for key in model.engram_tables.table_keys:
        assert torch.equal(
            model.engram_tables.rows[key], fresh.engram_tables.rows[key]
        ), key


def test_base_ckpt_loads_into_fp32_masters(tmp_path):
    """The trainer builds fp32 masters; bf16 export values must land exactly
    (bf16 -> fp32 is a widening cast)."""
    model = _bf16_model()
    out = save_base_checkpoint(model, tmp_path / "base", dtype=torch.bfloat16)
    torch.manual_seed(5)
    fresh = RefitModel(_enabled_cfg())  # fp32 default
    load_base_checkpoint(fresh, out)
    p = next(iter(model.state_dict().values()))
    f = fresh.state_dict()[next(iter(model.state_dict()))]
    assert f.dtype == torch.float32
    assert torch.equal(f, p.float())


def test_base_ckpt_config_mismatch_refused(tmp_path):
    """A base built from a different model config must not load — silently
    loading wrong-geometry weights is worse than a loud error."""
    import json

    model = _bf16_model()
    out = save_base_checkpoint(model, tmp_path / "base", dtype=torch.bfloat16)
    meta = json.loads((out / "config.json").read_text(encoding="utf-8"))
    meta["hidden_size"] = 512  # tamper: not the config this export contains
    (out / "config.json").write_text(json.dumps(meta), encoding="utf-8")
    fresh = _bf16_model(seed=1)
    with pytest.raises(ValueError, match="config mismatch"):
        load_base_checkpoint(fresh, out)


def test_trainer_prebuilt_init(tmp_path):
    """init: prebuilt loads the export before training (train_phase0 path)."""
    from train.src.data.topk_writer import ShardWriter
    from train.src.train.trainer import Trainer
    import numpy as np

    model_yaml = tmp_path / "m.yaml"
    import yaml
    model_yaml.write_text(yaml.dump(_enabled_cfg().to_dict()), encoding="utf-8")
    out = save_base_checkpoint(_bf16_model(), tmp_path / "base")

    rng = np.random.default_rng(3)
    w = ShardWriter(tmp_path / "s", k=4, teacher_id="synthetic", vocab_size=1024)
    for _ in range(4):
        tokens = rng.integers(0, 1024, size=64).astype(np.uint32)
        ids = np.stack([rng.permutation(1024)[:4] for _ in range(64)]).astype(np.uint32)
        ids[:-1, 0] = tokens[1:]
        probs = np.full((64, 4), 0.05, dtype=np.float32)
        probs[:, 0] = 0.85
        w.add_document(tokens, ids, probs, loss_mask=np.ones(64, dtype=np.uint8))
    w.finalize()

    cfg = TrainConfig.from_dict({
        "model": str(model_yaml), "data_shards": [str(tmp_path / "s")],
        "seq_len": 32, "steps": 2, "batch_size": 2, "lr": 3e-3,
        "warmup_steps": 1, "alpha": 1.0, "loss_chunk_size": 16,
        "bf16": False, "seed": 0, "init": "prebuilt",
        "prebuilt_path": str(out), "out_dir": str(tmp_path / "run"),
        "log_filename": str(tmp_path / "run.log"),
    })
    trainer = Trainer(cfg, device="cpu")
    load_base_checkpoint(trainer.model, cfg.prebuilt_path)  # train_phase0's call
    # Zero-init U survived the round trip: engram still inert at load time.
    assert all(int(p.abs().sum()) == 0 for p in trainer.model.engram.proj.parameters())
    trainer.train()
    assert "loss " in Path(cfg.log_filename).read_text()
