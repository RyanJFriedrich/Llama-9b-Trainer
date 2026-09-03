"""M4 tests: anneal schedules, trainer smoke + resume, checkpointing,
KL-to-donor, donor scorer. All at dev_tiny scale (AGENTS.md Hardware)."""
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from train.src.config import KnobSchedule, ModelConfig, TrainConfig, load_config
from train.src.data.topk_writer import ShardWriter
from train.src.model.refit import RefitModel
from train.src.train.anneal import AnnealDriver, theta_progress_at, window_at
from train.src.train.trainer import Trainer, lr_at

DEV_TINY = "train/configs/model/dev_tiny.yaml"


# --- anneal schedules -------------------------------------------------------

def test_anneal_endpoints_exact():
    """Spec §7.2: ablation knobs exact at both ends of their schedules."""
    sched = KnobSchedule(start_step=0, end_step=100)
    # window: full (None) -> final
    assert window_at(0, sched, seq_len=4096, final_window=2048) is None
    assert window_at(100, sched, 4096, 2048) == 2048
    assert window_at(50, sched, 4096, 2048) == 3072  # linear midpoint
    # theta progress: 0 -> 1
    assert theta_progress_at(0, sched) == 0.0
    assert theta_progress_at(100, sched) == 1.0
    assert theta_progress_at(50, sched) == 0.5


def test_final_state_when_no_schedule():
    """Spec v2.0 §3.4/§7.2: no schedule = hold the final topology."""
    assert window_at(500, None, 4096, 2048) == 2048
    assert theta_progress_at(500, None) == 1.0


def test_cosine_steps_then_steady_state():
    """cosine_steps < steps: warmup + cosine over the span, then flat at the
    min_lr_ratio floor for the rest of the run (owner pref 2026-09: anneal
    epoch 1, steady state after)."""
    d = {
        "model": DEV_TINY, "data_shards": ["x"], "seq_len": 64,
        "steps": 1000, "batch_size": 1, "lr": 2e-4, "warmup_steps": 10,
        "min_lr_ratio": 0.1, "cosine_steps": 100,
    }
    cfg = TrainConfig.from_dict(d)
    floor = cfg.lr * cfg.min_lr_ratio
    assert abs(lr_at(100, cfg) - floor) < 1e-12           # end of cosine span
    assert lr_at(500, cfg) == lr_at(1000, cfg) == lr_at(100, cfg)  # then flat
    mid = lr_at(55, cfg)
    assert floor < mid < cfg.lr                            # mid-span decays
    # Default (no cosine_steps) is unchanged: cosine spans the whole run.
    cfg2 = TrainConfig.from_dict({k: v for k, v in d.items() if k != "cosine_steps"})
    assert lr_at(1000, cfg2) == pytest.approx(floor, abs=1e-12)
    assert lr_at(500, cfg2) > floor + 1e-6


def test_anneal_driver_applies_to_model():
    cfg = load_config(DEV_TINY)
    model = RefitModel(cfg)
    tcfg = TrainConfig.from_dict({
        "model": DEV_TINY, "data_shards": ["x"], "seq_len": 64, "steps": 100,
        "warmup_steps": 10,
        "anneal_window": {"start_step": 0, "end_step": 100, "schedule": "linear"},
        "anneal_theta": {"start_step": 0, "end_step": 100},
    })
    driver = AnnealDriver(tcfg, model)
    state = driver.apply(0)
    assert state == {"window": None, "theta_progress": 0.0}
    state = driver.apply(100)
    assert state == {"window": 32, "theta_progress": 1.0}
    assert model.anneal_state == state


# --- trainer ----------------------------------------------------------------

def _learnable_shard(root: Path, name: str, n_docs: int = 8, t: int = 128,
                     k: int = 4, vocab: int = 1024, seed: int = 3) -> Path:
    """Synthetic shard where the teacher's top-1 is the true next token with
    0.8 mass — learnable, so a smoke run's loss must decrease."""
    rng = np.random.default_rng(seed)
    w = ShardWriter(root / name, k=k, teacher_id="synthetic", vocab_size=vocab)
    for _ in range(n_docs):
        tokens = rng.integers(0, vocab, size=t).astype(np.uint32)
        ids = np.stack([rng.permutation(vocab)[:k] for _ in range(t)]).astype(np.uint32)
        ids[:-1, 0] = tokens[1:]  # top-1 = true next token
        probs = np.full((t, k), 0.15 / (k - 1), dtype=np.float32)
        probs[:, 0] = 0.8
        mask = np.ones(t, dtype=np.uint8)
        w.add_document(tokens, ids, probs, loss_mask=mask)
    w.finalize()
    return root / name


def _train_cfg(tmp_path: Path, shards: list[str], steps: int = 30) -> dict:
    return {
        "model": DEV_TINY, "data_shards": shards, "seq_len": 32,
        "steps": steps, "batch_size": 2, "lr": 3e-3, "warmup_steps": min(5, steps - 1),
        "alpha": 1.0, "loss_chunk_size": 16, "bf16": False, "seed": 0,
        "init": "scratch", "out_dir": str(tmp_path / "run"),
        "checkpoint_every": steps, "log_every": 5,
        "log_filename": str(tmp_path / "run.log"),
        "anneal_window": {"start_step": 0, "end_step": steps, "schedule": "linear"},
        "anneal_theta": {"start_step": 0, "end_step": steps},
    }


def _run_losses(cfg_d: dict, from_step: int = 0) -> list[float]:
    """Run training, return the logged losses."""
    trainer = Trainer(TrainConfig.from_dict(cfg_d), device="cpu")
    trainer.train()
    losses = []
    for line in Path(cfg_d["log_filename"]).read_text().splitlines():
        if " step " in (" " + line) or "] step " in line:
            parts = line.split("loss ")
            if len(parts) > 1:
                losses.append(float(parts[1].split()[0]))
    return losses


def test_smoke_loss_decreases(tmp_path):
    shard = _learnable_shard(tmp_path, "s")
    cfg_d = _train_cfg(tmp_path, [str(shard)], steps=30)
    losses = _run_losses(cfg_d)
    assert len(losses) >= 5
    early = sum(losses[:2]) / 2
    late = sum(losses[-2:]) / 2
    assert late < early * 0.98, f"loss did not decrease: {early:.4f} -> {late:.4f}"


def test_freeze_embeddings(tmp_path):
    """Owner decision 2026-09-03 (TrainConfig.freeze_embeddings, default ON):
    embed_tokens + lm_head are frozen donor furniture — requires_grad False,
    absent from the optimizer groups, no grads after a backward — while the
    interior still trains. The False arm is the ablation escape hatch."""
    shard = _learnable_shard(tmp_path, "s")
    cfg_d = _train_cfg(tmp_path, [str(shard)], steps=6)
    trainer = Trainer(TrainConfig.from_dict(cfg_d), device="cpu")
    emb = trainer.model.model.embed_tokens.weight
    head = trainer.model.lm_head.weight
    assert not emb.requires_grad and not head.requires_grad
    opt_params = {id(p) for g in trainer.optimizer.param_groups for p in g["params"]}
    assert id(emb) not in opt_params and id(head) not in opt_params
    trainer.train()
    assert emb.grad is None and head.grad is None

    cfg_off = {**cfg_d, "freeze_embeddings": False,
               "out_dir": str(tmp_path / "run2"),
               "log_filename": str(tmp_path / "run2.log")}
    trainer2 = Trainer(TrainConfig.from_dict(cfg_off), device="cpu")
    assert trainer2.model.model.embed_tokens.weight.requires_grad
    assert trainer2.model.lm_head.weight.requires_grad


def test_bf16_grad_accumulation(tmp_path):
    """grad_dtype bf16 (owner decision 2026-09-03): autograd casts grads to
    bf16 at the leaf and accumulates micro-batches in bf16 — the fp32 grad
    set never exists. Frozen params get no grads; the smoke still learns at
    grad_accum 2."""
    shard = _learnable_shard(tmp_path, "s")
    cfg_d = {**_train_cfg(tmp_path, [str(shard)], steps=8), "grad_accum": 2}
    trainer = Trainer(TrainConfig.from_dict(cfg_d), device="cpu")
    trainer.train()
    emb = trainer.model.model.embed_tokens.weight
    assert emb.grad is None  # frozen -> no grad at all
    n_bf16 = 0
    for p in trainer.model.parameters():
        if not p.requires_grad:
            continue
        assert p.grad is not None and p.grad.dtype == torch.bfloat16
        n_bf16 += 1
    assert n_bf16 > 0
    losses = []
    for line in Path(cfg_d["log_filename"]).read_text().splitlines():
        if "] step " in line:
            losses.append(float(line.split("loss ")[1].split()[0]))
    assert len(losses) >= 2 and losses[-1] < losses[0]

    # bad values rejected
    with pytest.raises(ValueError, match="grad_dtype"):
        TrainConfig.from_dict({**cfg_d, "grad_dtype": "fp16"})


def test_bf16_grads_track_fp32(tmp_path):
    """Same seed, same shard, 4 steps: bf16-grad updates land within rounding
    distance of the fp32-grad regime (and on both optimizer paths)."""
    shard = _learnable_shard(tmp_path, "s")

    def run(grad_dtype: str, optimizer: str) -> dict[str, torch.Tensor]:
        cfg_d = {**_train_cfg(tmp_path, [str(shard)], steps=4),
                 "grad_dtype": grad_dtype, "optimizer": optimizer,
                 "out_dir": str(tmp_path / f"r_{grad_dtype}_{optimizer}"),
                 "log_filename": str(tmp_path / f"r_{grad_dtype}_{optimizer}.log")}
        trainer = Trainer(TrainConfig.from_dict(cfg_d), device="cpu")
        trainer.train()
        return {k: v for k, v in trainer.model.state_dict().items()}

    ref = run("fp32", "adamw8bit")
    for opt in ("adamw8bit", "adamw"):
        got = run("bf16", opt)
        for k, v in ref.items():
            # bf16 grad rounding (~4e-3 relative) amplifies through the
            # chaotic loss surface over steps: measured <= 7e-3 after 4 steps
            # at lr 3e-3 (worst: AttnRes key_norm weights), with step-1
            # loss/gnorm bitwise-identical. 1e-2 is a regression net, not an
            # equivalence claim.
            assert torch.allclose(got[k], v, atol=1e-2), f"{k} diverged ({opt})"


def test_checkpoint_resume_bitwise(tmp_path):
    """Resume-safe: same 15-step phase config; interrupt at 10 via max_steps,
    resume to 15 == uninterrupted 15 (anneal/LR schedules are functions of
    the phase config, so both runs must share it)."""
    shard = _learnable_shard(tmp_path, "s")
    full = _train_cfg(tmp_path / "full", [str(shard)], steps=15)
    losses_full = _run_losses(full)

    part = _train_cfg(tmp_path / "part", [str(shard)], steps=15)
    trainer = Trainer(TrainConfig.from_dict(part), device="cpu")
    trainer.train(max_steps=10)
    ckpt = trainer.save_checkpoint("ckpt.pt")

    res = _train_cfg(tmp_path / "res", [str(shard)], steps=15)
    trainer2 = Trainer(TrainConfig.from_dict(res), device="cpu")
    trainer2.load_checkpoint(ckpt)
    trainer2.train()
    losses_res = []
    for line in Path(res["log_filename"]).read_text().splitlines():
        if "loss " in line:
            losses_res.append(float(line.split("loss ")[1].split()[0]))

    assert trainer2.step == 15
    # The post-resume losses must bitwise-match the uninterrupted run's.
    assert losses_full[-1] == losses_res[-1]


def test_grad_checkpointing_matches(tmp_path):
    """Grad checkpointing changes memory, not math."""
    shard = _learnable_shard(tmp_path, "s")
    grads = {}
    for ckpt in (False, True):
        d = _train_cfg(tmp_path / f"ck{ckpt}", [str(shard)], steps=2)
        d["grad_checkpointing"] = ckpt
        trainer = Trainer(TrainConfig.from_dict(d), device="cpu")
        trainer.train()
        grads[ckpt] = {n: p.grad.clone() if p.grad is not None else None
                       for n, p in trainer.model.named_parameters()}
    for n in grads[False]:
        g0, g1 = grads[False][n], grads[True][n]
        assert (g0 is None) == (g1 is None), n
        if g0 is not None:
            assert torch.allclose(g0, g1, atol=1e-6), n


def test_run_metadata_logged(tmp_path):
    """Standing rule 6: full config + seed + code hash in the run log."""
    shard = _learnable_shard(tmp_path, "s")
    cfg_d = _train_cfg(tmp_path, [str(shard)], steps=2)
    _run_losses(cfg_d)
    text = Path(cfg_d["log_filename"]).read_text()
    assert '"seed": 0' in text and "code_hash=" in text and '"lr": 0.003' in text


# --- KL to donor + scorer ---------------------------------------------------

def test_kl_donor_zero_for_identical_models():
    from train.src.eval.kl_donor import kl_to_donor
    cfg = load_config(DEV_TINY)
    torch.manual_seed(5)
    model = RefitModel(cfg)
    docs = [list(range(1, 40)), [7, 3, 99, 12] * 8]
    kl = kl_to_donor(model, model, docs, device="cpu")
    assert kl < 1e-6


def test_kl_donor_positive_for_perturbed():
    from train.src.eval.kl_donor import kl_to_donor
    cfg = load_config(DEV_TINY)
    torch.manual_seed(5)
    a = RefitModel(cfg)
    b = RefitModel(cfg)
    with torch.no_grad():
        b.lm_head.weight.add_(torch.randn_like(b.lm_head.weight) * 0.01)
    docs = [list(range(1, 40))]
    assert kl_to_donor(a, b, docs, device="cpu") > 1e-4


def test_donor_scorer_shard(tmp_path):
    """Scorer emits a valid spec shard: mass invariant, sorted top-k, last
    position masked, top-1 matches a manual argmax."""
    from train.src.tools.donor_scorer import score_documents
    cfg = load_config(DEV_TINY)
    torch.manual_seed(5)
    model = RefitModel(cfg).eval()
    docs = [list(range(10, 42))]
    sidecar = score_documents(model, docs, tmp_path / "shard", k=4,
                              device="cpu", log_filename=str(tmp_path / "s.log"))
    assert sidecar["total_tokens"] == 32

    from train.src.data.topk_loader import TopKShard
    sh = TopKShard(tmp_path / "shard")
    w = np.asarray(sh.topk_w, dtype=np.float32)
    tail = np.asarray(sh.tail_w)
    assert np.allclose(w.sum(1) + tail, 1.0, atol=1e-3)
    assert (np.diff(w, axis=1) <= 1e-6).all()  # sorted by prob desc
    assert np.asarray(sh.loss_mask)[-1] == 0

    # Position 0's top-1 must equal the model's own argmax at position 0.
    with torch.no_grad():
        hidden = model(torch.tensor([docs[0]]), return_hidden=True)
        ref = (hidden[0, 0].float() @ model.lm_head.weight.float().T).argmax().item()
    assert int(sh.topk_idx[0, 0]) == ref
