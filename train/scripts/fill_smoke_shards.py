"""Fill `data_shards` in run configs from the converted shard dirs.

Points the bring-up smoke configs at every shard dir under
train/data/shards/ (sorted, dirs with a sidecar.json only). Safe to re-run;
it just rewrites the data_shards list. Do NOT run this against a config whose
run is in flight — filling configs is a pre-run setup step.

Usage (from repo root):
    python -m train.scripts.fill_smoke_shards
    python -m train.scripts.fill_smoke_shards --shards-dir train/data/shards \
        --configs train/configs/bringup_bf16_smoke.yaml train/configs/bringup_fp8_smoke.yaml
"""
import argparse
import os

import yaml

from train.utils.log import log

DEFAULT_CONFIGS = [
    "train/configs/bringup_bf16_smoke.yaml",
    "train/configs/bringup_fp8_smoke.yaml",
]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--shards-dir", default="train/data/shards")
    p.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    args = p.parse_args()

    shards = sorted(
        f"{args.shards_dir}/{name}"
        for name in os.listdir(args.shards_dir)
        if os.path.isdir(os.path.join(args.shards_dir, name))
        and os.path.exists(os.path.join(args.shards_dir, name, "sidecar.json"))
    )
    if not shards:
        raise SystemExit(
            f"no converted shard dirs in {args.shards_dir} — run "
            "python -m train.scripts.convert_bulk first"
        )

    for cfg_path in args.configs:
        with open(cfg_path, encoding="utf-8") as f:
            d = yaml.safe_load(f)
        d["data_shards"] = shards
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(d, f, sort_keys=False)
        log(f"fill_smoke_shards: {cfg_path} -> {len(shards)} shard dirs",
            print_console=True)


if __name__ == "__main__":
    main()
