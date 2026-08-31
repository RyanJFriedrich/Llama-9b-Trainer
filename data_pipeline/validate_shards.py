"""Validate one or more spec §6.2 TopK shard directories.

Opens each shard with the training repo's TopKShard loader and checks:
  - all required arrays exist with the locked dtypes
  - k matches the sidecar
  - Σ topk_w + tail_w = 1 ± 1e-3 per position
  - loss_mask is 0/1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.src.data.topk_loader import TopKShard
from train.src.data.topk_writer import ARRAY_DTYPES


def validate_shard(shard_dir: Path) -> bool:
    print(f"Validating {shard_dir} ...")
    try:
        shard = TopKShard(shard_dir)
    except Exception as exc:
        print(f"  FAIL: could not open shard: {exc}")
        return False

    ok = True
    for name, dtype in ARRAY_DTYPES.items():
        arr = getattr(shard, name)
        if arr.dtype != dtype:
            print(f"  FAIL: {name} dtype {arr.dtype} != expected {dtype}")
            ok = False

    w = np.asarray(shard.topk_w, dtype=np.float32)
    tail = np.asarray(shard.tail_w)
    mass = w.sum(axis=1) + tail
    if not np.allclose(mass, 1.0, atol=1e-3):
        bad = np.where(~np.isclose(mass, 1.0, atol=1e-3))[0]
        print(f"  FAIL: mass invariant violated at {len(bad)} position(s); "
              f"first bad mass={mass[bad[0]]} at index {bad[0]}")
        ok = False

    lm = np.asarray(shard.loss_mask)
    if not np.isin(lm, (0, 1)).all():
        print("  FAIL: loss_mask contains values other than 0/1")
        ok = False

    if ok:
        print(f"  PASS: k={shard.k}, tokens={shard.total_tokens}, "
              f"docs={shard.sidecar.get('num_documents', '?')}, "
              f"teacher={shard.teacher_id!r}, fold={shard.fold_version}")
    else:
        print("  shard has failures")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate spec TopK shards")
    parser.add_argument("shard_dirs", nargs="+", help="One or more shard directories")
    args = parser.parse_args()

    all_ok = True
    for d in args.shard_dirs:
        all_ok &= validate_shard(Path(d))

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
