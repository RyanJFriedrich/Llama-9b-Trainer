"""Bring-up tooling tests: preflight checks (dev-box thresholds) and the
§3.8 step-0 sanity loss math at dev_tiny scale."""
import math
from pathlib import Path

import torch

from train.src.config import load_config
from train.src.model.refit import RefitModel

DEV_TINY = "train/configs/model/dev_tiny.yaml"


def test_preflight_checks_structure_and_dev_box():
    """run_checks returns PASS/FAIL tuples; at dev-box thresholds everything
    that can pass here does (donor weights, canon sha, torch build)."""
    from train.scripts.preflight import run_checks

    results = run_checks(
        Path("OriginalModel"),
        Path("train/configs/model/llama_8bpp_v1.yaml"),
        min_vram_gb=1.0, min_disk_gb=1.0, min_ram_gb=1.0,
    )
    names = [r[0] for r in results]
    assert names == ["donor weights", "GPU", "disk", "system RAM", "torch build",
                     "engram canon"]
    for name, ok, detail in results:
        assert ok, f"{name}: {detail}"


def test_preflight_weights_check_catches_missing():
    from train.scripts.preflight import check_weights

    ok, detail = check_weights(Path("no/such/dir"))
    assert not ok and "config.json" in detail


def test_step0_gold_ce_uniform_for_random_model():
    """A scratch dev_tiny model on random tokens must sit AT uniform — the
    §3.8 band logic's null case (a donor-initialized model on real text is
    the deploy-box run, not a unit test)."""
    from train.scripts.step0_sanity import gold_ce

    torch.manual_seed(0)
    model = RefitModel(load_config(DEV_TINY)).eval()
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 1024, (1, 256), generator=g)
    ce = gold_ce(model, ids)
    uniform = math.log(1024)
    assert abs(ce - uniform) < 0.5, f"scratch CE {ce:.3f} vs uniform {uniform:.3f}"
