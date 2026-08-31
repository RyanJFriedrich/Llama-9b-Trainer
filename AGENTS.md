# AGENTS.md — Llama-9B (spec v2.0)

## What this project is

A from-scratch ~9.4B dense decoder (Llama-3.1 geometry/tokenizer) with a hybrid
architecture and an Engram sidecar table, trained on top-k distillation shards
(text gold + frozen 8B-donor logit anchors) on a single 96 GB Blackwell.
**Spec v2.0 premise: cold start with donor-furniture init.** Weights initialize
from Llama-3.1-8B-Instruct layers (gather = donor layer 31 with zeroed
`o_proj`/`down_proj`), but there is no recovery phase, no donor-equivalence
gate, no required anneal. The teacher is the data: gold tokens (corpus +
70B-authored) plus frozen top-k distributions from the quantized 8B donor.

The authoritative design docs are in `docs/`:

- `docs/llama-9b-refit-spec.md` (**v2.0**) — the *what/why*. [LOCKED]/[FLEXIBLE]/
  [EXPERIMENTAL] tags and the normative/informative split (§0.1) are binding.
  Filename keeps the legacy "refit" name; the project is no longer a refit.
- `docs/llama-9b-refit-first-steps.md` (v1.3) — run order, owner deliverables
  (D1–D6), standing rules, box bring-up (§3).
- `docs/llama-9b-agent-addendum.md` — the *why* behind each mechanism, the
  deliberate-exclusions list (§7), and the **escalation protocol (§1): when
  docs don't pin something down, STOP and ask the owner — do not improvise.**
- `docs/llama-9b-refit-data-pipeline.md` (v0.2) — data-production work order
  (separate agent owns `data_pipeline/`).
- `docs/NPZFormat.md`, `docs/Llama8bGeometry.md` — the production NPZ contract
  and the measured teacher rank geometry (tail is a power law α≈2; k=32
  captures ~0.90 mean mass — the lumped tail is material, keep it explicit).

Where code/docs and the spec disagree, **the spec wins — flag the conflict,
don't improvise.**

## Layout

```
docs/                     # owner-placed design docs (do not edit without being asked)
OriginalModel/            # Llama-3.1-8B-Instruct HF checkpoint (donor: tokenizer,
                          # furniture init, KD anchor scorer, M1 parity reference)
QuantizedModel/           # GGUFs for llama.cpp scoring (8B Q8_0 anchor; 70B Q4_K_M)
LlamaCPPBinaries/         # fork llama.cpp build with prompt_logprobs (load-bearing;
                          # mainline silently ignores it — do not "upgrade")
data_pipeline/            # owner's data production (bulk NPZ scoring; separate agent)
ReferenceCode/            # owner-placed Gemma-4 refit pipeline (gitignored)
train/                    # everything the agent builds lives here
  configs/                # YAML: model (spec §3.1), run configs, ablation arms
  src/
    config.py             # config dataclasses + YAML/JSON loader
    model/                # decoder (M1 substrate), refit (dispatch/SWA/sink/p-RoPE/
                          # AttnRes/gather), rope (full + partial rotary), attn_res
    data/                 # TopK shard writer/loader (mmap, spec §6.1)
    distill/              # fused/chunked lumped-tail KD loss (§6.2)
    engram/               # Engram sidecar (§3.6 + annex A1): canon (NFKC+casefold
                          # map + artifact), addressing (pinned SHA-256 salt stream,
                          # injective unigram + multiplicative-XOR n-gram mix),
                          # tables (host-resident bf16 rows, gather/stage),
                          # readout (zero-init U, I1), sparse_opt (host 8-bit
                          # AdamW over touched rows only, LR x5 WD 0)
    train/                # trainer loop, anneal schedules (§7.2 ablation tooling),
                          # 8-bit AdamW (√v), checkpointing
    eval/                 # perplexity, KL-to-donor (diagnostic), NIAH, attn probes
    tools/                # warm_start (donor furniture init), shard_converter
                          # (JSONL), npz_converter (production NPZ -> spec shards),
                          # donor scorer
  tests/                  # invariant + sanity tests (spec §3.7 I1–I9, §3.8)
  scripts/                # entry points (train_phase0.py, score_donor.py,
                          # parity_probe.py, build_engram_canon.py, preflight.py,
                          # step0_sanity.py)
  runs/                   # outputs, checkpoints, logs (gitignored)
  utils/log.py            # owner-provided logging helper (see Conventions)
```

Run everything from the repo root so `train` imports as a package
(`python -m train.scripts.<entry>`, `python -m pytest train/tests`).

## Commands

- Tests: `python -m pytest train/tests -v`
- Env: Python 3.13, torch 2.10+cu130, transformers 5.7 — pinned in `train/requirements.txt`.

## Hardware (D1 — delivered)

- **Local dev box: RTX 4090, 24 GB VRAM; 68.5 GB RAM.** Builds and unit-tests
  only. bf16 *inference* of the full 8B fits (the M1 parity probe ran here);
  training-scale does not. Do not treat a full-scale OOM here as a code bug;
  reproduce at toy scale instead.
- **Deploy box: rented RTX 6000 Pro Blackwell, 96 GB**, Linux, NGC PyTorch
  container (pinned), non-spot for bring-up; owner runs scripts. Bring-up
  sequence: first-steps §3 (preflight → bf16 reference smoke → FP8 overlay
  check → first real run).
- **Test strategy:** all unit/invariant tests run against
  `configs/model/dev_tiny.yaml` (9-layer, 256-hidden chopped-down variant with
  the same 3:1 + gather topology and p-RoPE scheme). Donor-scale step-0 sanity
  (spec §3.8) runs on the deploy box.

## Conventions

- **Logging — always use `log()`, never bare `print`.** `log()` from
  `train/utils/log.py` is a drop-in `print` replacement; every call appends a
  timestamped line to a log file (default `common.log`), console output is
  opt-in via `print_console=True`. Check what any run is doing with
  `tail common.log`. Run artifacts belong under `train/runs/<run_name>/`.
- **Config-driven everything.** Ablations and runs are config diffs, never code
  branches. Unknown config keys are rejected on load — extend
  `train/src/config.py` deliberately.
- Match surrounding code style; keep changes minimal and scoped.

## Standing rules (first-steps §5 + spec — binding)

1. Invariants I1–I9 (spec §3.7) are enforced by tests, not by care. The locked
   near-no-op inits: AttnRes zero-init pseudo-queries (I3), sink logits at −10
   (I4), Engram zero-init U (I1), gather identity init. Training starts at
   FINAL topology (§3.4) — anneal machinery is ablation tooling, default off.
2. No YaRN/NTK/position-interpolation code anywhere (long-context extension is
   a deferred, documented surgery on the rotated quarter only). No NoPE layers
   (retired v2.0). No QK-norm (QK-clip is the standing remedy, owner decision).
3. Never materialize full-vocab logits in the loss path (fused/chunked KD only).
4. Config-driven; no architecture constants hardcoded outside the config.
5. Never edit a run's config mid-flight; `max_steps` exists for clean
   interrupts. Schedules are functions of the run config.
6. Every run logs its full config + seed + code revision + data manifest;
   checkpoints are resume-safe bitwise (I9: model + optimizer + RNG + cursor).
7. One experimental variable at a time (optimizer, precision, architecture).
8. Spec §0.1: process parameters (tokens, epochs, cadence) are advisory; owner
   deviation is not a spec violation. Do not argue; log and proceed.
9. AttnRes sources are delta-sums, never stream snapshots; nothing may be added
   to the residual stream without delta-sum registration (I8 — the Engram
   injection depends on this).
10. Deliberate exclusions (addendum §7) are not to be re-added without an owner
    decision: contextual Engram gates, NoPE, donor-recovery phases, full-RoPE
    globals as default, FP8 weight storage, Muon (specified candidate,
    owner-triggered), fp32 AdamW on the deploy box.

## Status (2026-08-31, v2.0 rework landed)

- **M1 — HF parity vs Llama-3.1-8B-Instruct: DONE** (bit-exact; permanent unit
  test + `scripts/parity_probe.py`). Code validation only — not a training gate.
- **Model at v2.0 final topology: DONE at dev-tiny scale.** `src/model/refit.py`:
  SWA window 4096 @ bare θ=10k, learned sink logits (−10), p-RoPE 0.25 @ θ=1M on
  GLOBAL **and** GATHER (`rope.PartialRotaryEmbedding`, NeoX-style 32-dim slice,
  frequencies computed over the slice dim), NoPE removed, Block AttnRes
  (delta-sum sources, keys-only RMSNorm, zero-init queries), gather identity
  init. Anneal machinery (window/θ) retained as §7.2 ablation tooling, default
  final-state. Config schema gains the §3.1 `engram:` block and per-slice
  sidecar metadata (text_source/logit_source/data_class/alpha_override).
- **M3 — data path: DONE** + v2.0 ingest: `tools/npz_converter.py` converts the
  production bulk NPZ (docs/NPZFormat.md) to spec §6.1 shards, fold_version v2.
  The one-row shift between the formats (NPZ row t predicts tokens[t]; shard
  row t is the distribution at position t) is the load-bearing detail — gold
  lands at slot 0 with its true teacher prob. Validated on a real 1M-token
  bulk shard (I6 mass err ≤ 4e-4, gold@slot0 on all unmasked rows).
- **M4 — trainer: DONE at dev scale.** fp32 masters + bf16 autocast, cosine +
  warmup, grad accum/checkpointing, bitwise resume-safe checkpoints (I9),
  anneal driver, run metadata logging. Default optimizer: `adamw8bit`
  (harness's √v int8 AdamW, spec §5.1; ≈82.5 GB static at 8.25B — fits the
  96 GB box; fp32 AdamW ≈132 GB is the dev fallback only).
- **Tests: 88/88 on dev_tiny** (step-0 sanity §3.8 toy shape, invariants
  I1/I2/I3/I4/I7/I8, p-RoPE slice + relative-position property, gather
  identity, AttnRes delta-sum bookkeeping, warm-start accounting, converter
  round-trips, trainer smoke/resume/checkpointing, Engram addressing
  acceptance A1.8.1–4 + module + trainer integration (incl. the v1.1 cadence
  rule), preflight + step-0 sanity tooling).
- **Engram module (spec §3.6 + annex A1): DONE at dev-tiny scale.**
  `src/engram/`: canon (NFKC+casefold, min-id representative; artifact
  `assets/canon_llama31_v1.npy`, |V|=128256 → |V'|=101747, sha256 pinned in
  the prod config), addressing (SHA-256 salt stream `engram9b.v1:n=:k=:field`;
  unigram injective by construction (A·c+B mod prime); n-gram uint64
  multiplicative-XOR mix, oldest-first, distinct primes per (order, head)),
  host-resident bf16 tables (Uniform(−0.01, 0.01) init from dedicated
  per-table generators — global RNG untouched), gather/stage with np.unique
  collapse, readout g·U(RMSNorm(concat(head rows))) with zero-init U (I1),
  injection after layer 3's MLP registered into the block-1 partial (I2/I8).
  Sparse host 8-bit AdamW updates touched rows only (LR × lr_mult, WD 0,
  per-table step counter); gates in a WD-0 device group. Cadence (annex
  v1.1): table grads accumulate host-side fp32 across the grad-accum window;
  the sparse Adam step runs once per dense-equivalent step — ×5 never
  compounds to ×5N. Trainer prefetch:
  addressing off-thread (ThreadPoolExecutor(1)), staging on the main thread.
  Checkpoints carry tables + row-optimizer state (I9); canon sha256 mismatch
  on resume is a hard error. Recorded decisions: (a) spec §3.1's row counts
  (262144/1048576) are non-prime — annex A1.4 primes win (newer, normative
  for addressing); (b) M_uni = smallest prime ≥ max(canonical_id)+1 (the
  annex's "|V'|" sizing conflicts with its own min-id representatives, which
  are not dense; identity fallback reproduces the annex's stated fallback
  exactly); (c) boundary rule: positions t < n−1 contribute nothing for
  order n; (d) Adam bias correction uses a per-table global step counter.
- **Next owner action: box bring-up** (first-steps §3): `scripts/preflight.py`
  (weights/GPU/disk/RAM/torch/canon-sha, PASS/FAIL) → bf16 reference smoke
  (loss curve + tok/s, discard checkpoint) → FP8 overlay check → first real
  run. Donor-scale §3.8 step-0 sanity (`scripts/step0_sanity.py`: fixed probe
  batch, gold-CE band vs uniform, KL-to-donor diagnostic) is part of
  bring-up.
- **Data:** class-(a) bulk shards in production by the owner (K=32, Q8_0 8B
  anchor, ~86M tokens/day); the data-pipeline doc (v0.2) governs classes
  b–e (70B SFT, thinking traces with labeled uncertainty, AP multilingual,
  news slices). ≥70% rehearsal in any delivered batch.
- Stop and report before the first real 9B run.
