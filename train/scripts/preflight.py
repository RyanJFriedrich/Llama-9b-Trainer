"""Deploy-box preflight (first-steps §3.3) — PASS/FAIL environment checks.

Run from the repo root on the deploy box before the bf16 reference smoke:

    python -m train.scripts.preflight

Checks: donor weights present and complete, CUDA GPU with enough VRAM,
free disk (1–2 TB for shards + runs), system RAM (>= 64 GB — the Engram
tables and their optimizer states are host-resident), torch build with
working CUDA + bf16, and the Engram canon artifact matching the prod
config's pinned sha256. Exit code 1 on any FAIL.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from train.utils.log import log

REPO_ROOT = Path(__file__).resolve().parents[2]


def check_weights(weights_path: Path) -> tuple[bool, str]:
    """Donor checkpoint: config + tokenizer + the full safetensors set per
    the index (missing shards are the classic partial-download failure)."""
    cfg = weights_path / "config.json"
    if not cfg.exists():
        return False, f"no config.json at {weights_path}"
    index = weights_path / "model.safetensors.index.json"
    if not index.exists():
        return False, f"no model.safetensors.index.json at {weights_path}"
    shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    missing = [s for s in shards if not (weights_path / s).exists()]
    if missing:
        return False, f"missing weight shards: {missing}"
    tok = weights_path / "tokenizer.json"
    if not tok.exists():
        return False, "tokenizer.json missing"
    total_gb = sum((weights_path / s).stat().st_size for s in shards) / 2**30
    return True, f"{len(shards)} shards, {total_gb:.1f} GiB, tokenizer present"


def check_gpu(min_vram_gb: float) -> tuple[bool, str]:
    import torch

    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is False"
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 2**30
    ok = vram >= min_vram_gb
    return ok, f"{name}, {vram:.1f} GiB VRAM (need >= {min_vram_gb:.0f})"


def check_disk(path: Path, min_free_gb: float) -> tuple[bool, str]:
    free = shutil.disk_usage(path).free / 2**30
    return free >= min_free_gb, f"{free:.0f} GiB free at {path} (need >= {min_free_gb:.0f})"


def check_ram(min_ram_gb: float) -> tuple[bool, str]:
    try:
        import psutil

        total = psutil.virtual_memory().total / 2**30
    except ImportError:
        # psutil optional: fall back to os.sysconf (Linux deploy box).
        import os

        try:
            total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
        except (ValueError, OSError, AttributeError):
            return True, "could not determine RAM (skipped)"
    return total >= min_ram_gb, f"{total:.1f} GiB RAM (need >= {min_ram_gb:.0f})"


def check_torch() -> tuple[bool, str]:
    import torch

    detail = f"torch {torch.__version__}, cuda {torch.version.cuda}"
    if not torch.cuda.is_available():
        return True, detail + " (CUDA kernel test skipped)"
    try:
        x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
        y = (x @ x).float().mean().item()
        ok = abs(y) < 10.0  # any sane value; we're testing the kernel path runs
        return ok, detail + f", bf16 GEMM ok (probe {y:.3f})"
    except Exception as e:  # pragma: no cover - hardware dependent
        return False, detail + f", bf16 GEMM FAILED: {e}"


def check_canon(model_yaml: Path) -> tuple[bool, str]:
    """The prod config's pinned canon sha256 must match the artifact on disk —
    a wrong map silently re-addresses every Engram row."""
    from train.src.config import load_config
    from train.src.engram.canon import canon_sha256, load_canon

    cfg = load_config(str(model_yaml))
    e = cfg.engram
    if not e.canonical_compression:
        return True, "canonical_compression off (identity fallback)"
    try:
        P = load_canon(REPO_ROOT / e.canon_path, e.canon_sha256)
    except (FileNotFoundError, ValueError) as exc:
        return False, str(exc)
    return True, f"canon verified: {len(P)} ids, sha256 {canon_sha256(P)[:16]}..."


def run_checks(
    weights_path: Path,
    model_yaml: Path,
    min_vram_gb: float = 90.0,
    min_disk_gb: float = 1000.0,
    min_ram_gb: float = 64.0,
) -> list[tuple[str, bool, str]]:
    return [
        ("donor weights", *check_weights(weights_path)),
        ("GPU", *check_gpu(min_vram_gb)),
        ("disk", *check_disk(REPO_ROOT, min_disk_gb)),
        ("system RAM", *check_ram(min_ram_gb)),
        ("torch build", *check_torch()),
        ("engram canon", *check_canon(model_yaml)),
    ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=str(REPO_ROOT / "OriginalModel"))
    p.add_argument("--model-config", default=str(REPO_ROOT / "train/configs/model/llama_8bpp_v1.yaml"))
    p.add_argument("--min-vram-gb", type=float, default=90.0)
    p.add_argument("--min-disk-gb", type=float, default=1000.0)
    p.add_argument("--min-ram-gb", type=float, default=64.0)
    args = p.parse_args()

    results = run_checks(
        Path(args.weights), Path(args.model_config),
        min_vram_gb=args.min_vram_gb, min_disk_gb=args.min_disk_gb,
        min_ram_gb=args.min_ram_gb,
    )
    all_ok = True
    for name, ok, detail in results:
        all_ok &= ok
        log(f"preflight [{'PASS' if ok else 'FAIL'}] {name}: {detail}", print_console=True)
    log(f"preflight {'PASS' if all_ok else 'FAIL'} overall", print_console=True)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
