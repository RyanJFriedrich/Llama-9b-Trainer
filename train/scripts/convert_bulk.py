"""Batch-convert a directory of production bulk NPZ files to spec §6.1 shards.

One shard dir per NPZ (named after the file), under the output dir. Already-
converted NPZs (shard dir exists with a sidecar) are skipped, so the command
is safe to re-run after an interruption or after new NPZs land.

Usage (from repo root):
    python -m train.scripts.convert_bulk
    python -m train.scripts.convert_bulk --in data_pipeline/bulk_out --out train/data/shards
"""
import argparse
import glob
import os
import time

from train.src.tools.npz_converter import convert_npz
from train.utils.log import log


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--in", dest="in_dir", default="data_pipeline/bulk_out",
                   help="directory of bulk_*.npz files")
    p.add_argument("--out", dest="out_dir", default="train/data/shards",
                   help="parent dir for converted shard dirs")
    p.add_argument("--teacher-id", default="meta-llama-3.1-8b-instruct")
    p.add_argument("--data-class", default="a")
    p.add_argument("--max-chunks", type=int, default=None,
                   help="cap chunks per NPZ (smoke-testing only)")
    args = p.parse_args()

    npzs = sorted(glob.glob(os.path.join(args.in_dir, "*.npz")))
    if not npzs:
        raise SystemExit(f"no .npz files found in {args.in_dir}")
    os.makedirs(args.out_dir, exist_ok=True)

    done = skipped = 0
    t0 = time.time()
    for i, npz in enumerate(npzs, 1):
        name = os.path.splitext(os.path.basename(npz))[0]
        shard_dir = os.path.join(args.out_dir, name)
        if os.path.exists(os.path.join(shard_dir, "sidecar.json")):
            skipped += 1
            continue
        log(f"convert_bulk [{i}/{len(npzs)}]: {npz} -> {shard_dir}",
            print_console=True)
        convert_npz(npz, shard_dir, teacher_id=args.teacher_id,
                    data_class=args.data_class, max_chunks=args.max_chunks)
        done += 1
    mins = (time.time() - t0) / 60
    log(f"convert_bulk: {done} converted, {skipped} already done, "
        f"{len(npzs)} total in {mins:.1f} min", print_console=True)


if __name__ == "__main__":
    main()
