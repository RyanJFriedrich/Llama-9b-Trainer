# train/data/bulk_npz/ — bulk NPZ drop zone

Drop (or symlink) the data_pipeline's bulk NPZ shards here
(`bulk_XXXXX.npz`, format: `docs/NPZFormat.md`). Contents are gitignored —
this is a landing spot, not a store of record.

Current production source: `data_pipeline/bulk_out/` (K=32, Q8_0 8B anchor,
8192-token chunks). Either copy/symlink the files in here, or point the
converter at that directory directly — it takes explicit paths.

Convert to spec §6.1 shards (one shard dir per NPZ, into `../shards/`):

```bash
for f in train/data/bulk_npz/bulk_*.npz; do
  python -m train.src.tools.npz_converter "$f" \
    "train/data/shards/$(basename "$f" .npz)" \
    --teacher-id meta-llama-3.1-8b-instruct --quantization Q8_0 \
    --text-source wiki20231101+fineweb-edu-10BT --data-class a
done
```

Then reference the shard dirs in a run config's `data_shards:` list.
