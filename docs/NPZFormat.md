# NPZ Format — Bulk Corpus Distillation Targets

The `.npz` shards produced by `data_pipeline/run_bulk_score.py`
(default output: `data_pipeline/bulk_out/bulk_XXXXX.npz` + `manifest.json`).

This is the **legacy-style** format for the training agent, NOT the spec
§6.2 shard format consumed by this repo's own trainer (`train/`). It is a
deliberately lossless rewrite of the old Gemma-pipeline NPZ
(`ReferenceCode/LogitExtraction/targets_to_npz.py`), which was lossy. The
differences are decisions, documented below.

## Provenance

- **Teacher model**: `QuantizedModel/meta-llama-3.1-8b-instruct.Q8_0.gguf`
  (Llama-3.1-8B-Instruct-abliterated, Q8_0).
- **Scorer**: fork llama.cpp server, build `1 (88a0aaa)` in
  `LlamaCPPBinaries/`. Its `/completion` endpoint supports
  `prompt_logprobs: true` + `n_probs` → per-prompt-position top-k logprobs.
  **Mainline llama.cpp does NOT have this feature** (verified b10425:
  flag silently ignored). This binary is load-bearing — do not "upgrade" it
  without re-verifying `prompt_probabilities` in the response.
- **Tokenizer**: Llama-3.1 BPE (128,256 vocab), from `OriginalModel/`.
- **Corpus**: streaming 50/50 seeded interleave (seed 1234) of
  `wikimedia/wikipedia` 20231101.en and `HuggingFaceFW/fineweb-edu`
  sample-10BT; docs < 200 chars skipped; docs joined with token 128001
  (`<|end_of_text|>`); stream cut into 8192-token chunks; each chunk scored
  in ONE prefill (`n_predict: 0`).

## Arrays (per shard file)

| array | dtype / shape | meaning |
|---|---|---|
| `tokens` | u32 `[N]` | raw token ids, chunks concatenated |
| `teacher_ids` | i32 `[N, K]` | see row semantics below (K = 32 in current runs) |
| `teacher_probs` | f32 `[N, K]` | true softmax mass, **NOT renormalized** |
| `loss_mask` | u8 `[N]` | 1 = scored position; 0 = chunk position 0 (no distribution exists for it) |
| `chunk_start` | i64 `[C]` | start index of each chunk within the arrays |
| `chunk_length` | i64 `[C]` | length of each chunk (8192 unless partial) |

### Row semantics — the important part

Row `t` holds the teacher distribution **predicting `tokens[t]`**
(i.e. the distribution *at* position t−1, in causal-LM terms).

- **Slot 0 = GT**: the ground-truth token (`teacher_ids[t,0] == tokens[t]`
  for every row with `loss_mask[t] == 1`) carrying its **true teacher
  probability** — the mass the teacher actually assigned to the realized
  token. It is a *label*, not a distortion: the teacher's numbers are
  untouched.
- **Slots 1..K−1** = the teacher's top-K tokens *excluding GT*, shuffled
  **as intact (id, prob) pairs** (per-chunk shuffle seeded `seed+1+ordinal`,
  so it is reproducible). Rank order is re-derivable by sorting probs
  descending; the shuffle exists so no consumer can accidentally rely on
  slot position as rank.
- **Row mass ≤ 1**: rows sum to the *captured* mass. The lumped tail is
  `1 − teacher_probs[t].sum()` — explicit and exact, per position.
- **Masked rows**: `loss_mask == 0` rows carry `teacher_ids[·,0] = -1` and
  zeros. Skip them.

Edge case: when GT is outside the teacher's top-K, GT still takes slot 0
(with its true, small prob) and the teacher's rank-K token is displaced
(keeps the array rectangular). These are exactly the high-entropy positions;
their GT prob is tiny (e.g. 1.6e-4 observed), so treat a very-low-prob slot 0
as "teacher didn't have this token in its top-K".

## manifest.json

```json
{
  "config": { "k": 32, "seq_len": 8192, "seed": 1234, "wiki_frac": 0.5, ... },
  "shards": [ {"file": "bulk_00000.npz", "chunks": 128, "tokens": 1048576}, ... ],
  "total_tokens": ..., "total_chunks": ..., "stream_cursor": ...
}
```

`stream_cursor` = number of stream ordinals consumed (scored **or** dropped).
Resume = regenerate the deterministic stream (same seed/config ⇒ same
chunks) and fast-forward `stream_cursor` chunks. Resuming with a different
seed/mix silently lands in a different stream — don't.

## Design decisions (vs the old lossy pipeline) and why

1. **Id↔prob pairing is intact.** The old script inserted GT at slot 0 by
   *shifting ids while probs stayed put* — every probability ended up
   attached to the wrong token, giving GT the teacher's top-1 mass at every
   position. That systematically converts KD into soft CE toward gold and
   corrupts all alternative levels. Here, pairs never separate; slot 0 is
   purely a marker for "this was the realized token".
2. **No renormalization.** The old script softmax-renormalized top-k to
   sum 1, destroying tail mass and erasing per-position entropy (a flat
   position and a peaked one look identical afterward). For this teacher the
   top-32 captures only ~0.90 mean mass — renormalizing would inflate every
   prob by ~1.12× on average, and much more at flat positions. Consumers can
   always renormalize on load; the reverse is impossible.
3. **Dense positions.** The old pipeline stored assistant-token positions
   only (conversation data). Bulk text has no roles; every position is
   stored — the prefill is paid for anyway.
4. **k = 32, not 12.** Measured captured-mass curve (see
   `Llama8bGeometry.md`): k=12 → 0.834 mean, k=32 → 0.895, k=64 → 0.926.
   Extraction throughput is flat in k (full-vocab softmax dominates);
   storage is the only cost (~256 B/token at k=32).

## What a consumer must decide (we deliberately did not)

- **Loss mix**: pure KD over the rows as-is, or an α-blend toward gold:
  `p' = α·onehot(GT) + (1−α)·renormalize(row)`. Slot 0 makes GT trivially
  addressable; α is a training-config knob, not a data property.
- **Tail handling**: lump `1 − row_mass` as one bucket (recommended; matches
  the harness's lumped-tail KD), or ignore it (≈10% of mass, concentrated at
  flat positions).
- **Geometry shaping**: if imposing a synthetic tail shape, use the measured
  α(p₁) family in `docs/Llama8bGeometry.md` rather than a guessed exponent.

## Minimal reader

```python
import numpy as np
d = np.load('data_pipeline/bulk_out/bulk_00000.npz')
tok, tids, tprobs, mask = d['tokens'], d['teacher_ids'], d['teacher_probs'], d['loss_mask']
rows = mask == 1
gt_ids   = tids[rows, 0]          # == tok[rows] by construction
gt_probs = tprobs[rows, 0]        # teacher's true prob of the realized token
tail     = 1.0 - tprobs[rows].sum(axis=1)   # lumped tail mass per position
# recover rank order: np.argsort(-tprobs[rows], axis=1)
```
