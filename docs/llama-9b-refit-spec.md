# Llama-9B — Final Specification (v2.0)

> **Filename note:** this file keeps the legacy `llama-9b-refit-spec.md` name for repo compatibility. **The project is no longer a refit.** v2.0 is a premise-level rewrite: the model is a from-scratch ~9.4B-param training run whose weight init happens to reuse donor layers, and whose "warmth" lives entirely in pre-computed distillation targets. All v1.x "recovery / donor-preservation" vocabulary is retired.

---

## §0. Read this first — how to interpret this document (NORMATIVE)

This section governs every coding agent that works from this document. Read it before anything else.

### 0.1 Normative vs. informative

This spec distinguishes **the machine** from **the run**:

- **Normative (binding):** architecture constants, the invariants in §3.7, and the data contracts in §6. These define *what the model is* and *what the data is*. Do not deviate without an explicit owner decision.
- **Informative (advisory):** token counts, epochs, cadences, schedules, hardware notes, eval timings, worked examples. These are guidance for runs, never gates.

**Agent contract:** process parameters are owner-discretionary at runtime. If the owner runs more tokens, fewer tokens, more epochs, or a different budget than any example in this document shows, **that is not a spec violation and requires no objection**. Do not argue. Numbers in informative sections illustrate; they do not bind.

### 0.2 Requirement tags

- **[LOCKED]** — implement exactly as written.
- **[FLEXIBLE]** — default given; expose as a config knob; owner may vary freely.
- **[EXPERIMENTAL]** — implement behind a flag, default off.
- **[INFORMATIVE]** — context, rationale, or guidance. No implementation requirement.

Anything unspecified defaults to **[FLEXIBLE]**. When in doubt, prefer the simpler implementation and record the choice in the changelog.

### 0.3 Provenance discipline

Design choices cite their source (paper, shipped model, or this project's own 1B Engram program). Where a choice rests on owner measurement rather than literature, it says so. Do not "simplify away" a choice whose rationale is recorded here without flagging it to the owner.

---

## §0.5 TL;DR + headline numbers (INFORMATIVE)

A from-scratch ~9.4B dense decoder: Llama-3.1 geometry and tokenizer, 8 blocks of [3×SWA + 1×Global] + 1 gather layer, Moonshot-style Attention Residuals, an Engram-derived n-gram sidecar table, trained on top-k distillation shards (text gold + frozen 8B-donor logit anchors) on a single 96GB Blackwell.

| Quantity | Value |
|---|---|
| Body params | ~8.25B (33 layers × ~218M + 1.05B untied embed/head) |
| Engram table params | ~1.2B (latent rows, host-RAM resident) |
| d_model / heads | 4096; 32 Q / 8 KV (GQA 4:1), head_dim 128 |
| FFN | SwiGLU, hidden 14336 |
| Locals | SWA window 4096, RoPE θ=10k, learned sink logits |
| Globals + gather | full-span, p-RoPE p=0.25, θ=1M |
| Tokenizer | Llama 3.1, vocab 128,256 |
| Static train memory (8-bit AdamW) | ≈ 82.5 GB + activations (fits 96 GB) |
| Data | packed 8k sequences; top-k KD shards (current K=32) + CE |

---

## §1. Premise (NORMATIVE in spirit, INFORMATIVE in mechanics)

1. **Cold start with donor furniture.** Weights initialize from Llama-3.1-8B-Instruct layers (§3.4), but no claim about "warmth" is load-bearing. There is no recovery target, no donor-equivalence gate, no required anneal. The donor checkpoint's remaining load-bearing roles are: (a) the tokenizer, (b) the logit anchor that scores the KD shards, (c) the M1 code-parity reference.
2. **The teacher is the data.** Training signal = gold tokens (from corpora and 70B-authored text) plus frozen top-k logit distributions from the quantized 8B donor. The anchor distribution is regularization-first; capability transfer flows through the gold tokens.
3. **Depth over width; computation over storage.** The architecture spends parameters on depth and routing (AttnRes), pushes static pattern storage into an explicit table (Engram), and buys supervision density with distillation. Every component serves that one thesis.
4. **Artifact identity.** A generalist ~9.4B local-inference model + sidecar memory, built as an amalgamation of published, converged findings (Cohere 3:1, Moonshot residuals, Google/Alibaba positional scheme, DeepSeek tables) plus one self-measured component (the {1,2,3} mixed-order table). Not a research paper; a working artifact.

---

## §2. Tokenizer & embeddings [LOCKED]

- Tokenizer: **Llama 3.1 exactly** (vocab 128,256, BPE, tiktoken-style). Use the Llama-3.1-8B-Instruct tokenizer files verbatim.
- Embeddings **untied** (separate embed and lm_head), init from the donor.
- The 256 reserved special-token slots are available for thinking/uncertainty markers (§7.3) and future use.
- Accepted trade (recorded): Llama's BPE is weaker on code/math/CJK than a Qwen-class 151k vocab; taken in exchange for tokenizer-matched donor embeddings, donor-scored KD shards, and ecosystem tooling.

---

## §3. Architecture

### 3.1 Skeleton [LOCKED]

```yaml
model:
  d_model: 4096
  n_layers: 33            # 8 blocks x [SWA,SWA,SWA,GLOBAL] + 1 GATHER
  pattern: [SWA, SWA, SWA, GLOBAL] x 8 + [GATHER]
  heads: { q: 32, kv: 8, head_dim: 128 }   # GQA 4:1
  ffn: { type: swiglu, hidden: 14336 }
  norm: rmsnorm_prenorm
  vocab: 128256
  embeddings: untied
  biases: false
  gather: { layer_index: 32, type: global }
attn_res:
  scope: all_layers        # [FLEXIBLE] ablation arm: globals_only
  block_granularity: 4     # blocks align to the [3+1] groups; gather = own block
positional:
  swa:  { type: rope,  theta: 10000, window: 4096 }
  global: { type: prope, fraction: 0.25, theta: 1000000 }
  gather: { type: prope, fraction: 0.25, theta: 1000000 }
engram:
  enabled: true
  orders: [1, 2, 3]
  heads_per_order: 2
  row_dim: 256
  rows_per_head: { 2: [1048573, 1048571], 3: [1048559, 1048549] }   # [FLEXIBLE] annex A1.4 primes (v2.1: re-tagged from non-prime powers of two); order 1 derived: smallest prime >= max(canonical_id)+1 = 128257
  injection_point: layer 3 output    # block-1 GLOBAL
optimizer: { backend: adamw8bit }    # [FLEXIBLE] candidate: muon (§5.2)
precision: bf16                       # [FLEXIBLE] bf16 | fp8 (§4)
```

### 3.2 Local layers (SWA) [LOCKED]

- Sliding-window causal attention, **window 4096**. Offsets never exceed the window (invariant I7).
- **RoPE, θ = 10,000.** Rationale (recorded so it is not relitigated): with window-capped offsets, aliasing is structurally impossible (slowest wavelength ≈ 63k ≈ 15× the window); ~45 of 64 frequency pairs work inside the window; matches the shipped constants of both reference hybrids (RNoPE: 10k @ 4096; Gemma locals: 10k).
- **Sink:** learned per-head additive bias logit (gpt-oss style), init ≈ −10 (near-no-op at init). *Not* token-anchored attention — a sink token at long range would violate the offset invariant.
- No QK-norm anywhere (two independent sources: retrieval-spike flattening; worse convergence).

### 3.3 Global layers and the gather [LOCKED]

- Full-span causal attention over the whole sequence.
- **p-RoPE: rotate the first 25% of head dims (the high-frequency quarter), θ = 1,000,000; the remaining 75% carry no positional signal.** Convention pinned (v2.1): HF/NeoX-style contiguous slice with frequencies computed over the slice's own dim count (`inv_freq_i = θ^(−2i/32)` for a 32-dim slice of a 128-dim head) — the rotated quarter spans the full clock spectrum (Qwen3-Next's shipped convention), NOT the 16 fastest pairs of the full 64-pair spectrum (which would leave globals positionally blind beyond ~160 tokens). Applies to all 8 GLOBAL layers **and** the final GATHER layer (uniform positional family; Gemma 4's shipped pattern, incl. "final layer always global"). At 8k training lengths only ~9 of the 16 rotated pairs do meaningful in-window work — the slow pairs are near-static BY DESIGN (they are the YaRN extension reserve).
- Consequence: beyond-window order information exists but whispers; long-context extension is YaRN-on-the-rotated-quarter only (16 dim-pairs per head) — small, documented surgery when the time comes.
- NoPE was evaluated and retired (v2.0 changelog).

### 3.4 Weight init [LOCKED]

| Component | Init |
|---|---|
| Layers 0–31 | Llama-3.1-8B-Instruct layers 0–31, 1:1 |
| Embeddings / lm_head | donor |
| Gather (layer 32) | copy of donor layer 31 with `o_proj` and `down_proj` zeroed (exact no-op at init) |
| AttnRes pseudo-queries | zero (→ uniform source mix; invariant I3) |
| Sink logits | ≈ −10 (invariant I4) |
| Engram U projections | zero (invariant I1); row tables fresh random; head scalars g = 1.0 |

Epistemology: cold start. The donor weights are furniture — a meaningful geometry to start from — not a function to preserve. No anneal is required; training starts at final topology. The anneal machinery (window/θ/positional schedules) is retained, default final-state, as ablation tooling.

### 3.5 Attention Residuals (AttnRes) [LOCKED]

Mechanics (verified against Moonshot's official pseudocode; the v0.1 snapshot formulation was a bug — do not regress):

- **Sources** at every application point: `{token embedding} ∪ {delta-sum of each completed block} ∪ {current block's partial}`. Block delta-sums accumulate sublayer outputs (`attn_out`, `mlp_out`) within the block; the embedding is source 0. With 4-layer blocks + gather, the deepest points mix ~10 sources.
- **Application points:** before *every* sublayer (attention and MLP) of every layer — `scope: all_layers`. (`globals_only` exists as an [EXPERIMENTAL] ablation flag.)
- Per application point: one pseudo-query vector (zero-init) scores the sources; **keys RMSNorm'd, values raw**; output = softmax-weighted sum of sources, used as that sublayer's input.
- At init, zero queries → uniform weights → each source weighted 1/(N+1); by RMSNorm scale-invariance this is *exactly* the PreNorm residual input. Step-0 equivalence is therefore exact by construction, not approximate (invariant I3).
- Parameter cost: one query vector + RMSNorm gains per application point — negligible.
- Note (I8, §3.7): any content added to the residual stream that is not registered in the block's delta-sum accumulation is **silently dropped** at the next AttnRes recomputation. The Engram injection (§3.6) depends on this rule.

### 3.6 Engram sidecar tables [LOCKED component, FLEXIBLE geometry]

An n-gram content-addressable memory, co-trained from step 0. This component is justified by this project's own 1B program (research log v3) plus two external production replications (DeepSeek; Qwen3.8-Flash-Next). Nobody has published a {1,2,3} mixed-order table; ours is the first, and it is included on cost asymmetry (the unigram head is nearly free; excluding it forfeits the option permanently).

- **Addressing [LOCKED]:** deterministic hash of the n-gram ending at the current position, per-order, per-head salts. Unigram keyspace (128,256) maps collision-free at the default row count — assert this at init.
- **Rows [FLEXIBLE]:** latent, `row_dim = 256`. Default allocation by keyspace: unigram 2×2^18, bigram 2×2^20, trigram 2×2^20 (~1.2B params, bf16 ≈ 2.4 GB, host-RAM resident with prefetch; sparse touched-row updates only).
- **Readout [LOCKED]:** per head, contribution = `g · U(RMSNorm(row))`, U: 256→4096 **zero-init**, g a learned scalar (init 1.0). No contextual per-position gating (measured to collapse in the 1B program; admission is the residual economy's job — every later block's AttnRes source weights are trained, co-adapted gates on table content).
- **Injection [LOCKED]:** single point, at the output of layer 3 (the block-1 GLOBAL). The injected vector **must be accumulated into the block's delta-sum partial** (invariant I2), so it persists through all later AttnRes recomputations and rides the source mix to every downstream sublayer.
- **Optimizer routing:** rows and U receive gradients only for touched rows (sparse update path); see §5.
- **Telemetry (must-support):** touched-row counts, cross-table top-row overlap (duplication vs. specialization), top-rows audits, fraction of bf16 row-updates rounding to zero.

### 3.7 Invariants (NORMATIVE — enforced by harness self-tests)

- **I1** Engram inert at init: zero-init U ⇒ contribution exactly 0.
- **I2** Delta-registration: with random nonzero U, injected content persists across subsequent AttnRes application points (persistence self-test).
- **I3** Zero-init AttnRes ⇒ sublayer inputs at init exactly equal PreNorm residual inputs.
- **I4** Sink logits init ≈ −10 ⇒ near-no-op at init.
- **I5** Token-id identity: re-tokenizing shard samples reproduces stored token ids exactly.
- **I6** Per-position KD weights normalize: stored top-k weights + `tail_w` account for total mass 1.
- **I7** SWA mass beyond window = 0 (runtime probe, not just config).
- **I8** Nothing may be added to the residual stream without delta-sum registration.
- **I9** Checkpoint resume is bitwise: model + optimizer + RNG + data cursor.

### 3.8 Step-0 sanity (NORMATIVE, behavior-shaped)

With donor furniture init and all invariants holding, a fixed probe batch must yield loss well below uniform (ln 128,256 ≈ 11.76 nats) and in a sane band for a windowed/perturbed donor. KL-to-donor on the probe set is logged as a *diagnostic* of topology-shift magnitude — **not a gate**. (This replaces the v1.x M2 "exact donor equivalence" keystone, which died with the warm-start premise.)

---

## §4. Precision & compute [FLEXIBLE]

- Config flag `precision: bf16 | fp8`. Master weights always fp32.
- **FP8 scope:** attention/FFN GEMMs only. Embeddings, lm_head, norms, AttnRes source-mixing, and the loss stay bf16/fp32. FP8 *weight storage* is rejected (unneeded at 96 GB; divergence risk).
- **Bring-up protocol [LOCKED process, owner-budgeted]:** bf16 reference smoke first — short run, record loss curve + tokens/sec, discard checkpoint — then the FP8 run must overlay the reference within tolerance before FP8 is adopted for the run. Rationale: attribution. Every validated reference in this project is bf16/fp32; FP8-first would leave four suspects (box, container, scaling recipe, harness) if the curve is wrong. Expected FP8 gain: ~1.25–1.4× end-to-end (GEMM-bound parts only).
- torch.compile + CUDA graphs: encouraged at fixed 8k shapes; graph-with-eager-breaks is the expected pattern; keep an eager fallback flag for debugging.
- Activation budget: per-sublayer gradient checkpointing on; small micro-batch; flash-style attention kernels that never materialize score matrices (SWA banded; globals full 8k).

---

## §5. Optimizer

### 5.1 Default backend: 8-bit AdamW [LOCKED for v1 bring-up]

The harness's hand-rolled block-wise int8 AdamW (stores **√v** — pointwise `sqrt(EMA(g²)) ≥ |m|` by Jensen, so zero-rounded √v blocks imply near-zero m and the ratio stays bounded). Cross-validated against bitsandbytes 0.50.1 AdamW8bit on the toy benchmark (final losses 0.0688 vs 0.0722; the two 8-bit paths agreeing is the trustworthiness signal; both sit ~19% above fp32 on the toy — expected to shrink at scale; verify on the first run's smoke if it matters). fp32 AdamW remains the dev fallback; note its memory (≈132 GB static) does not fit the deploy box.

Memory ledger (8.25B body): bf16 weights 16.5 + fp32 masters 33 + bf16 grads 16.5 + int8 m,√v 16.5 ≈ **82.5 GB static** + activations. Engram rows/optimizer state live in host RAM (sparse touched-row updates), so the table does not move this number.

### 5.2 Candidate backend: Muon [EXPERIMENTAL — pending owner decision]

Not committed. The evidence (Moonlight arXiv:2502.16982; K2/GLM-4.5 production use; systematic comparisons) supports Muon as the better default for cold-start 9B pretraining (~10–15% token savings conservative), and optimizer consistency across stages (pretrain→SFT) favors deciding before the long-horizon run. **Revisit trigger:** before the long-horizon run, after a toy-scale validation of the implementation.

If adopted, the recipe is fixed here so no re-derivation is needed:

- **Routing predicate:** `param.ndim == 2 and not embedding/lm_head` → Muon. Everything else (embeddings, lm_head, norms, AttnRes pseudo-queries, sink logits, head scalars, **Engram rows** — lookup vectors, no spectrum to orthogonalize) → 8-bit AdamW. Engram **U projections** (2D) may go either way; default Muon. Hybrid is the canonical Muon form, not a deviation.
- **Recipe (Moonlight):** momentum μ=0.95; Newton-Schulz, 5 iterations, coefficients (3.4445, −4.7750, 2.0315); update RMS-matched to AdamW via `0.2 · sqrt(max(fan_out, fan_in))` scaling so AdamW-tuned LR/WD transfer; decoupled weight decay.
- **Memory:** bf16 momentum on ~7.2B eligible params ≈ 14.5 GB + 8-bit AdamW on the rest ≈ 2.5 GB — parity with the default backend. Do not use fp32 momentum on the 96 GB box (~95 GB static, too tight).
- **Rules:** single GPU → Newton-Schulz runs locally, no distributed complications; keep Muon through SFT if adopted; never A/B optimizers at intermediate checkpoints (rankings flip during LR decay — final-loss-at-budget only); QK-clip exists as a standby stability flag (K2's MuonClip lineage), default off, and it is *not* QK-norm (which remains banned, §3.2).

---

## §6. Data contract

### 6.1 Shard format (NORMATIVE)

- Memmap-able flat shards + sidecar JSON. Per token position: `token_id` (uint32), top-k `indices` (uint32 — vocab 128,256 exceeds uint16), top-k `weights` (fp16), and `tail_w` (fp32) for v2 shards. **K is flexible** — the loader must accept any K (current production K=32; ~200 bytes/token).
- **Dense per-position coverage** over the full token stream is the only non-negotiable. Positions genuinely excluded from loss must carry an explicit `loss_mask`; silent absence is a contract violation. User turns in conversation data are loss-masked but still scored.
- Sidecar metadata per shard: `text_source`, `logit_source`, numerics path (e.g., fp8 8B, Q4_K_M 70B), chat-template hash, provider/version, `fold_version`, data class (§6.3), α override (optional).
- Legacy NPZ shards: ingested via the converter, marked `fold_version: v1` (tail folded into gold), usable as-is. New extraction is `fold_version: v2` (unfolded, `tail_w` separate). Owner measurement settled this: at K≥10 the tail mass is ~1e-6 and folding is harmless (it amounts to an ad-hoc CE blend, redundant with the explicit CE term — consider α=1.0 for v1-only runs).
- Invariants I5 (token-id identity) and I6 (mass accounting) are asserted on every shard sample.

### 6.2 Loss (NORMATIVE formula, FLEXIBLE weights)

Per position: `L = α · KL_lumped-topk(anchor ‖ student) + (1−α) · CE(gold)`, where the lumped-topk KL uses the coarsened anchor distribution (top-k entries + one tail lump). α is per-slice, from the sidecar. Chunked/fused KD loss is a hard requirement (full [T, 128,256] logits never materialize). Defaults:

| Data class | text_source | logit_source (anchor) | α default |
|---|---|---|---|
| (a) bulk natural text | corpus (Wiki/OpenWeb/FineWeb-Edu) | fp8/q8 8B donor | 0.9 |
| (b) 70B-authored SFT | 70B API | 8B donor | ≤ 0.1 (CE-dominant) |
| (c) thinking traces | reasoning-prompted 3.3-70B + verifier | none (γ=0) | 0 (pure CE) |
| (d) multilingual task data (AP summarize→translate→nuance) | 70B or off-policy Gemma text | none or 8B | 0–0.5 (CE-heavy) |
| (e) news/knowledge slices | AP corpus pipeline | none or 8B | 0–0.5 (CE-heavy) |

Rationale (recorded): the 8B anchor is a *frozen, informed, cheap* regularizer — good where the 8B is competent (bulk text), confidently wrong where it is not (thinking, post-cutoff facts). For post-cutoff knowledge, the text is the only teacher in the stack.

### 6.3 Data classes and mixing rules [FLEXIBLE]

- **Rehearsal rule:** ≥ 70% broad old-distribution text in every phase/run. Never a news-only run.
- **Multi-epoch:** permitted on KD slices (2–4 epochs reasonable — soft targets suppress verbatim memorization pressure). Keep epochs low on CE-dominant knowledge slices; lean on paraphrase views (the AP pipeline's summary/N-translations/N-nuance structure) for repetition instead.
- Knowledge-cutoff push: loss spikes on unknown entities are expected and harmless; forgetting is monitored via quick-suite drift, not loss shape; factual-uptake probes built from the AP corpus are the injection metric.
- KD dynamics guidance [FLEXIBLE]: soft dense targets tolerate higher LR than CE folklore suggests — re-tune LR upward cautiously *after* early training stabilizes; per-token comparisons against CE-pretraining lore do not apply; teacher-scoring FLOPs are amortized since shards are reusable.
- Expectation note: on pure-KD slices the student's ceiling is the anchor's quality, so table/capacity upside concentrates on CE-bearing slices (news, thinking, off-policy text). This is by design — facts to the table, reasoning to the backbone.

---

## §7. Training

### 7.1 Cadence (INFORMATIVE — owner-discretionary by §0.1)

Run → inspect → decide. Inspections at owner-chosen increments (the instrument list in §8). No token budgets, epoch counts, or phase gates are specified anywhere in this document; the harness must support clean resume/extend (`max_steps` interrupts; never edit a run config mid-flight — schedules are functions of the config) precisely so the owner can keep a good box running or walk away. "We trained what was packed while the box was up" is a legitimate run record.

### 7.2 Schedules [FLEXIBLE]

- LR: cosine + warmup (harness default). Optimizer state resets (e.g., a future Muon switch) cost roughly one re-warm transient (measured on the 1B program: ~1 increment).
- Anneal machinery (window/θ/positional schedules): present, default final-state, ablation tooling only.

### 7.3 Thinking data [LOCKED process]

- Teacher: reasoning-prompted Llama-3.3-70B via the owner's two-question protocol (structured work-through scaffold → final answer), with an external verifier (strict JSON).
- **Uncertainty is a first-class, labeled behavior.** Verified / unverified / could-not-verify outcomes are kept and distinctly marked — reserved special-token slots exist in the vocab for this. Teaching "I checked and I'm not sure" is a design goal, not noise.
- Thinking spans are pure CE (α=0); generation-time length compliance preferred over post-hoc filtering; thinking should fit within the 4096 SWA window with margin for prompt + response (≤ ~3k tokens guidance) so response tokens attend back over their own reasoning inside the local window.

### 7.4 Knowledge injection rules (NORMATIVE principles, FLEXIBLE execution)

Text is the only teacher for post-cutoff facts; CE-heavier slices (§6.2); ≥ 70% rehearsal; no LR bump for knowledge slices; uptake probes from the AP corpus run at the normal inspection cadence. Multi-view paraphrase repetition (the AP pipeline's structure) is the uptake mechanism; single-exposure tail facts will be weak at this scale — expected, not a failure.

---

## §8. Instrumentation & inspection (must-support measurements)

The harness must be able to produce, at any checkpoint, with fixed held-out probe data:

1. Loss curve + KL-to-anchor (per data class).
2. AttnRes source weights vs. uniform (drift, NaN, collapse-onto-single-source alarms).
3. Engram telemetry: touched-row counts, cross-table overlap (duplication vs. specialization), top-rows audits, bf16 rounding-loss fraction.
4. Entropy-bucketed loss (where is the table doing work?).
5. NIAH-lite at 8k (global-layer retrieval behavior).
6. Quick suite (frozen subsets: MMLU ~1.5k, ARC-C ~500, HellaSwag ~1k, GSM8k ~300; loglikelihood, fixed seed).
7. Sink behavior (logit drift, sink mass).
8. SWA mass-beyond-window = 0 (invariant I7 runtime probe).
9. Optimizer health (zero-rounded √v fraction, update RMS) and throughput (tok/s).
10. When thinking data exists: thinking-length compliance (% over guidance) and verified/unverified rates.

**Alarm shapes (behavioral, not numeric gates):** loss spikes that survive a few increments; AttnRes source collapse; duplication persisting across tables deep into training; KL-to-anchor diverging; quick-suite regression (forgetting); rounding-loss fraction climbing (bf16 row saturation).

---

## §9. Hardware environment (D1 — delivered)

- Deploy box: rented **RTX 6000 Pro Blackwell, 96 GB**, Linux, NGC PyTorch container (pinned), non-spot for bring-up. Owner runs scripts; the repo ships a `RUNBOOK.md`, `scripts/preflight.py`, and one-command entry points with unambiguous PASS/FAIL output. No interactive anything.
- Requirements: ≥ 64 GB system RAM (host-RAM Engram tables + prefetch), 1–2 TB disk (each resume-safe checkpoint of the body with optimizer state runs 50–80 GB), decent CPU for prefetch.
- Local 4090 (24 GB): data scoring (8B anchor, thousands of tok/s prefill via llama.cpp), dev tests, optional full-width toy smoke (see `llama-9b-refit-local-smoke.md`, owner-undecided — if run, it must include a table and the final positional scheme to remain representative).
- Measured: Q4_K_M 70B local prefill ≈ 400 tok/s ≈ 35M tokens/day — the middle scoring tier for high-value slices (measure-and-record; not the bulk path).

---

## §10. Parked items & open questions (INFORMATIVE)

| Item | State | Trigger |
|---|---|---|
| Muon backend (§5.2) | specified, pending | pre-long-horizon, after toy validation |
| p-RoPE fraction sweep (0/0.125/0.25/0.5/1.0) on RULER at/beyond training length | confirming experiment, optional | post-first-run curiosity |
| Multi-window locals (1k/2k/4k) | parked ablation | never required |
| Long-context extension (YaRN on rotated quarter) | deferred | when extension is wanted |
| Engram row budget growth (retrofit regime measured on the 1B program) | upgrade path open | owner appetite |
| Inference-stack ports (llama.cpp/vLLM need AttnRes + p-RoPE + table support) | real work, unscheduled | before public release of weights |
| 1B-program arms (C3 sticky-note, U-frozen, gate anatomy) | complete record in research log v3 | background only |

---

## §11. Changelog (v0.1 → v2.1)

| Version | Change |
|---|---|
| v0.1 | Initial spec (warm-start refit premise). |
| v1.0 | External critique adjudicated (13 accepted / 6 rejected): AttnRes delta-sum mechanics fixed, learned sink logits, lumped-tail KD, memmap shards, control arms. |
| v1.1 | SWA θ=20k locked (owner belief, math-verified); AttnRes scope default all_layers. |
| v1.2 | Donor → Instruct checkpoint everywhere; 1B-token ceiling; fold settlement (v1/v2). |
| v1.3–1.4 | Self-KL anchor encoded then withdrawn by owner (instantaneous-detach ≡ LR rescale); KD dynamics notes. |
| v1.5–1.7 | Quantized-teacher preflight; two-tier pipeline (70B writes text, 8B anchors); measured Q4_K_M tiers. |
| v1.8 | `teacher_id` split into `text_source` + `logit_source`. |
| **v2.0** | **Premise rewrite:** cold start with donor-furniture init (warm-start framing retired); positional spec final — SWA 4096 @ θ=10k, p-RoPE 0.25 @ θ=1M on globals **and** gather (NoPE retired); optimizer becomes backend-pluggable (default 8-bit AdamW; Muon specified, pending); phases retired → owner-discretionary inspect cadence; normative/informative preamble added (§0.1 agent contract); Engram {1,2,3} table added as a locked component (§3.6, invariants I1/I2/I8); thinking teacher decided (reasoning-prompted 3.3-70B + verifier, uncertainty labeled first-class); K=32 production shards; D1 delivered (96 GB Blackwell); M2 demoted to step-0 sanity (§3.8). |
| **v2.1** | **Post-implementation corrections (external review + agent flags, all verified):** §3.1 Engram rows re-tagged to the annex A1.4 primes (the 262144/1048576 powers of two were non-prime); §3.3 p-RoPE convention pinned to the NeoX-style slice (frequencies over the slice's own dim — Qwen3-Next convention), closing a 4-orders-of-magnitude ambiguity; annex A1 amended to v1.1 (M_uni erratum + table-update cadence rule). No architecture change; the code already implemented exactly this. |

---

## §12. Provenance (INFORMATIVE)

Cohere RNoPE (arXiv:2501.18795) — 3:1 hybrid, θ=10k @ 4096 · Moonshot Attention Residuals (Kimi Linear tech report) — delta-sum sources, zero-init queries · Gemma 4 TR (arXiv:2607.02770) — p-RoPE 0.25 globals, θ=1M global/10k local, final-layer-global · Qwen3-Next / Qwen3.8-Flash-Next — 0.25 rotary fraction, production Engram table · Barbero et al. 2025 ("Round and Round We Go") — high-frequency pairs carry position · "Fractional Rotation, Full Potential?" (arXiv:2603.11611) — partial-RoPE convergence ablation · Moonlight (arXiv:2502.16982) — Muon at scale · DeepSeek Engram paper — sidecar memory · This project's 1B Engram program (research log v3) — unigram strength, gate collapse, ownership, point ablations · Llama 3.1 (Meta) — tokenizer, geometry, donor.
