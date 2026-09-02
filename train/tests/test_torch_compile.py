"""torch.compile knob tests (TrainConfig.torch_compile): the training loop
runs through a compiled forward while self.model stays the raw module —
state_dict/checkpoint formats are untouched, and the full production stack
(engram dataclass input, fp8 custom-autograd GEMMs, grad checkpointing)
survives dynamo at dev_tiny scale on GPU."""
import os
from pathlib import Path

# Single-threaded inductor: the async compile worker pool spawns one
# subprocess per worker (each re-importing torch) — heavy on Windows and
# pointless for these tiny graphs.
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import numpy as np
import pytest
import torch
import yaml

from train.src.config import TrainConfig, load_config
from train.src.data.topk_writer import ShardWriter
from train.src.train.trainer import Trainer

DEV_TINY = "train/configs/model/dev_tiny.yaml"
VOCAB = 1024


def _model_yaml(tmp_path: Path) -> str:
    d = load_config(DEV_TINY).to_dict()
    d["engram"]["enabled"] = True
    p = tmp_path / "dev_tiny_engram.yaml"
    p.write_text(yaml.dump(d), encoding="utf-8")
    return str(p)


def _learnable_shard(root: Path, name: str, n_docs: int = 8, t: int = 128,
                     k: int = 4, seed: int = 3) -> Path:
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


def test_compile_config_roundtrip():
    d = {"model": DEV_TINY, "data_shards": ["x"], "seq_len": 32, "steps": 2,
         "batch_size": 1, "warmup_steps": 0, "torch_compile": True}
    cfg = TrainConfig.from_dict(d)
    assert cfg.torch_compile is True
    assert TrainConfig.from_dict(cfg.to_dict()).torch_compile is True
    d2 = dict(d)
    del d2["torch_compile"]
    assert TrainConfig.from_dict(d2).torch_compile is False  # default off


def test_compile_wiring_preserves_raw_module(tmp_path, monkeypatch):
    """torch_compile wraps a forward CALLABLE, never the module: state_dict
    keys and checkpoint format are identical to eager, eval paths stay
    eager."""
    calls = []
    real_compile = torch.compile

    def recording_compile(mod, **kw):
        calls.append(kw)
        return mod  # identity — CPU has no inductor guarantee; wiring only

    monkeypatch.setattr(torch, "compile", recording_compile)
    shard = _learnable_shard(tmp_path, "s")
    cfg_d = {
        "model": _model_yaml(tmp_path), "data_shards": [str(shard)],
        "seq_len": 32, "steps": 2, "batch_size": 2, "lr": 3e-3,
        "warmup_steps": 1, "alpha": 1.0, "loss_chunk_size": 16,
        "bf16": False, "seed": 0, "init": "scratch", "torch_compile": True,
        "out_dir": str(tmp_path / "run"), "checkpoint_every": 2,
        "log_every": 1, "log_filename": str(tmp_path / "run.log"),
    }
    trainer = Trainer(TrainConfig.from_dict(cfg_d), device="cpu")
    assert calls == [{"dynamic": False}]
    trainer.train()
    assert "_orig_mod" not in next(iter(trainer.model.state_dict().keys()))
    ckpt = torch.load(Path(cfg_d["out_dir"]) / "ckpt_step2.pt",
                      map_location="cpu", weights_only=False)
    assert not any(k.startswith("_orig_mod") for k in ckpt["model"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_compile_full_stack_gpu(tmp_path):
    """The production combo at dev_tiny scale on a real inductor backend:
    torch_compile + fp8 GEMMs + engram + grad checkpointing + bf16 autocast.
    Loss must be finite and decrease over a handful of steps."""
    # Suite-position hygiene: earlier GPU tests (e.g. the 8B parity leg) leave
    # their peak in the CUDA caching allocator for the life of the pytest
    # process. Reclaim it before compiling or this test stacks on top of it.
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    shard = _learnable_shard(tmp_path, "s")
    cfg_d = {
        "model": _model_yaml(tmp_path), "data_shards": [str(shard)],
        "seq_len": 32, "steps": 20, "batch_size": 2, "lr": 3e-3,
        "warmup_steps": 2, "alpha": 1.0, "loss_chunk_size": 16,
        "bf16": True, "precision": "fp8", "grad_checkpointing": True,
        "seed": 0, "init": "scratch", "torch_compile": True,
        "out_dir": str(tmp_path / "run"), "checkpoint_every": 20,
        "log_every": 1, "log_filename": str(tmp_path / "run.log"),
    }
    trainer = Trainer(TrainConfig.from_dict(cfg_d), device="cuda")
    assert trainer._fwd is not trainer.model
    trainer.train()
    losses = [
        float(line.split("loss ")[1].split()[0])
        for line in Path(cfg_d["log_filename"]).read_text().splitlines()
        if "loss " in line
    ]
    assert len(losses) == 20 and all(np.isfinite(losses))
    # Same bar as the fp8 overlay smoke at this scale: toy per-tensor fp8 is
    # worst-case, so "learns, does not diverge", not a tight curve.
    early, late = sum(losses[:2]) / 2, sum(losses[-2:]) / 2
    assert late < early, f"compiled loss did not decrease: {early:.4f} -> {late:.4f}"
