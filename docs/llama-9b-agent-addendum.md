# Llama-9B — Coding-Agent Addendum (v1.0)

**Audience:** the coding agent(s) building and operating the training harness and data pipeline.

**What this file is:** an orientation briefing. Several components of this project are 2025–2026 research that may be absent from or misrepresented in your prior knowledge. This file explains the *why* behind the *what*, lists deliberate exclusions, and defines when you must stop and ask the owner.

**Precedence:** the spec (`llama-9b-refit-spec.md` v2.0) is the normative source of truth. This addendum **clarifies, never overrides**. If anything here appears to conflict with the spec, the spec wins — and the apparent conflict is itself an escalation trigger (§1).

---

## §1. The escalation protocol — ask the owner (NORMATIVE behavior)

Your prior knowledge ends before some of the research this project is built on. The owner can fetch current, highly specific research on demand; you cannot. Therefore:

**When you hit any of the following, STOP and ask the owner. Do not improvise, do not average two conflicting sources, do not "pick the reasonable one," do not spend cycles re-deriving.**

Escalation triggers:

1. Any apparent contradiction between normative statements (within one doc or across docs).
2. Any invariant test (spec §3.7) that fails and you cannot explain why from the docs.
3. Any constant, shape, formula, or mechanism that the docs reference but do not pin down (example: the exact hash function used for Engram addressing in the owner's 1B harness).
4. Any urge to add a mechanism "for completeness" or because your priors say it belongs (see §7 — most of those were considered and deliberately excluded).
5. Any uncertainty about whether a number is normative (binds) or informative (advisory). Default assumption: numbers describing the architecture and data format bind; numbers describing runs (tokens, epochs, cadence) are advisory.
6. Before modifying, refactoring, or "optimizing" any code path covered by an invariant test.
7. When your training-data knowledge of a technique conflicts with these docs. The docs reflect decisions made *after* reviewing the current literature; treat the docs as newer and correct, and ask if you believe they contain an error.

**How to ask — format:**

```
WHAT I'M BUILDING: <the component>
WHAT'S UNCLEAR: <the specific ambiguity, with doc references>
MY DEFAULT WOULD BE: <what you'd implement if forced>
WHAT I NEED: <the decision, or the specific research question the owner could fetch>
```

The owner will respond with a decision or with fetched research; treat fetched research as an extension of this addendum. A short, pointed question costs the owner seconds. A silent wrong guess costs a training run. Asking is always the cheaper path.

**One behavioral rule that overrides your instincts:** the owner will routinely run more tokens, more epochs, or different budgets than any example in these docs shows. Per spec §0.1, process parameters are advisory and owner deviation is not a spec violation. Do not argue, do not warn, do not block. Log what was actually run (standing rule: run cards) and proceed.

---

## §2. AttnRes (Attention Residuals) — the most-misread component

Source: Moonshot's Kimi Linear production architecture (2025–26). If you have never seen this mechanism, read this section twice before touching the model module.

**What it is.** Before *every* sublayer (attention and MLP), the sublayer's input is **recomputed** as a learned, attention-weighted combination over depth-history sources, instead of taken as the raw residual stream. One pseudo-query vector per application point scores the sources; softmax weights mix them; the mix becomes the input.

**The sources are delta-sums, not stream snapshots — this is the single most important detail.** Each [SWA,SWA,SWA,GLOBAL] group is one block. Within a block, every sublayer's *output delta* (`attn_out`, `mlp_out`) is accumulated into a running `partial_block` sum; at block completion that sum is appended to the source list. Source 0 is the token embedding. So the sources are: embedding + one delta-sum per completed block + the current block's partial.

Why not snapshots of the residual stream at block boundaries? Because a snapshot at boundary k already contains all earlier deltas — mixing snapshots weights early contributions ~N times more than late ones, which silently breaks the architecture's step-0 behavior and skews everything after. Delta-sums partition the stream without overlap. (This was the v0.1 spec bug; it was caught precisely because it would have been invisible in casual testing. Do not regress it.)

**Why keys are RMSNorm'd but values are raw.** The key RMSNorm makes source scores comparable across depths (block outputs grow in magnitude with depth); normalizing the *values* would destroy the very scale information the residual stream needs. Only the scoring path is normalized.

**Why zero-init pseudo-queries are exact, not approximate.** Zero query → all source scores equal → uniform weights → the mix is (sum of sources)/(N+1). The sum of all delta-sums plus embedding plus partial *is* the ordinary residual stream; dividing by (N+1) rescales it uniformly; the sublayer's pre-norm RMSNorm is scale-invariant, so it sees *exactly* the standard PreNorm input. Step-0 equivalence holds by construction, for any weights, donor or random. Spec invariant I3 tests this.

**The erasure hazard (spec I8) — read this before adding anything to the stream.** Because sublayer inputs are *recomputed from sources*, anything added directly to the residual stream that was not registered in the block's delta-sum accumulation is silently discarded at the next application point. No error, no warning — the contribution just vanishes. The Engram injection (§3) must be accumulated into the partial block sum. Any future stream modification faces the same rule. There is a persistence self-test (random nonzero U, verify the contribution survives subsequent application points); if you are unsure whether some path registers correctly, run that test pattern on it.

**What AttnRes is not:** not DenseFormer (fixed learned scalar depth-weights — ablated by the authors, lost, excluded); not a skip-connection variant; not optional at SWA layers (scope `all_layers` is the default because that is the configuration all published measurements used; `globals_only` exists only as an owner-flagged ablation).

---

## §3. Engram sidecar tables — address vs. payload, and the two invariants

Sources: DeepSeek's Engram paper; Qwen3.8-Flash-Next (production, Aug 2026); the owner's 1B experimental program (the research log in `/docs/` is the detailed record). This component is the least documented externally; the owner's measurements are the primary evidence, and they were made deliberately.

**The mechanism.** At each position, form the n-grams *ending at that position* (orders 1, 2, 3 — e.g., the unigram key is the current token, the bigram key is the previous token + current, the trigram key spans three). Each key is deterministically hashed (per-order, per-head salts) to a row in a table of learned vectors (dim 256). The row is read out through `g · U(RMSNorm(row))` where U (256→4096) projects into residual width. Contributions are summed and injected once, at the output of layer 3 (the first GLOBAL layer).

**Address vs. payload — the distinction everyone gets wrong.** The *address* looks backward at most 2 tokens. The *payload* is a single vector added at the current position only. A layer does not "see" neighboring positions' rows through the table. Nothing about the table is a window or an attention pattern. Each row's teacher is the next-token loss at the position where the row was read: rows learn local "what tends to follow this exact short context" statistics. Do not expect rows to store facts attached to entities; there is no global "Japan" memory, only drawers selected by short local keys.

**Why the two invariants exist.**
- **I1 (inert at init):** U is zero-init, so the table contributes exactly 0 at step 0 regardless of row contents or gates. The model starts as a plain transformer; the table earns its way in through training. Never "warm" U randomly.
- **I2 (delta-registration):** the injected vector must be added into the block's AttnRes delta-sum accumulation (see §2 erasure hazard). This is what lets every later block's AttnRes source weights act as learned admission gates on table content — the architecture's intended gating mechanism.

**Why there is deliberately no contextual gate on the table.** The 1B program measured it: per-position contextual gates collapse to near-zero scalars in training (three independent arms, same result), while the scalar×projection product trains through anyway. The residual economy provides the gating. Do not add a per-position gate "for completeness" — it is a measured dead end, and its presence would confuse the telemetry.

**Why rows live in host RAM with prefetch.** Addressing is deterministic, so the rows a batch will touch are known before the forward pass — fetch-ahead over PCIe hides the latency. Only touched rows receive gradients (sparse update path); untouched rows' optimizer state never moves. This is why a ~1.2B-param table costs the GPU nothing in the memory ledger.

**Sizing rationale (so the defaults don't look arbitrary).** The unigram tables were the *strongest measured artifact* in the 1B program (a collision-free unigram head beat bigram/trigram heads at equal budget — the n-gram keyspace smears across shared slots at small row counts, while unigram keys map cleanly). The {2,3}-gram heads are the upside bet at longer budgets; the unigram head is nearly free (128k keys). Rows are allocated by keyspace for exactly this reason. The {1,2,3} mixed-order design has no published precedent — it is a measured owner decision, not an omission.

**If anything here is underspecified for implementation** (exact hash function, salt scheme, row init distribution, prefetch granularity): check the owner's 1B harness code if present in the repo; otherwise escalate (§1 trigger 3). Do not substitute a hash of your choosing — collision behavior was measured and matters.

---

## §4. The positional scheme — RoPE locals, p-RoPE globals, sink logits

- **Locals (SWA, window 4096): full RoPE, θ = 10,000.** This constant is deliberate and math-checked: with offsets window-capped at 4096 forever, aliasing is structurally impossible (slowest wavelength ≈ 63k ≈ 15× the window), and ~45 of 64 frequency pairs work inside the window. It matches the shipped constants of both reference hybrids. Do not "round it up."
- **Globals and the gather: partial RoPE — rotate only the first 25% of head dims (the 16 highest-frequency pairs), θ = 1,000,000; the remaining 75% of dims are never rotated.** This is the Gemma 4 / Qwen3-Next pattern. Mechanistic basis (Barbero et al. 2025): high-frequency pairs carry positional signal while low-frequency pairs get co-opted for semantics — rotating only the fast quarter keeps a position channel without taxing the content channel. Kernel note: partial rotary is standard in NeoX-style implementations; verify your kernel rotates *only* the 32-dim slice.
- **NoPE (no positional encoding) was evaluated and retired** — do not introduce NoPE layers.
- **Sink logits:** a learned per-head *additive bias* in the attention logits (init ≈ −10), giving heads a learned "no-op" bucket. It is **not** "attend to token 0." Token-anchored sinks would violate the SWA offset invariant (a sink at absolute position 1 seen from position 6000 is an offset of 5999). This distinction is easy to get wrong and was a critique-round fix.
- **No QK-norm anywhere.** Two independent findings: it flattens retrieval-relevant logit spikes and converges to higher loss. If attention-logit instability ever appears (watch under any future optimizer change), the standing remedy is QK-**clip** (dynamic clamp, spec §5.2), not norm — and deploying it is an owner decision.

---

## §5. The data scheme — two sources, and why the "teacher" is weaker than the "author"

This is the piece most likely to look like a bug. It is not.

**`text_source` ≠ `logit_source`, deliberately.** The gold tokens come from a corpus or from the 70B API; the per-position top-k distribution (the "anchor") comes from the *quantized 8B donor* — a much weaker model. Standard KD uses one strong teacher for both. Here, the 8B anchor is a **frozen, cheap, informed regularizer**: where the 8B is competent (bulk natural text), its distribution is genuine anti-forgetting signal; where it is not (thinking traces, post-cutoff knowledge), the anchor weight α is reduced or zeroed per data class (spec §6.2 table) so it cannot fight the gold tokens. Capability transfer flows through the gold text; the anchor shapes *how* the model is allowed to move. This is why a 70B-authored shard scored by an 8B is correct behavior, and why the sidecar records both sources separately.

**The loss:** per position, `α · KL_lumped-topk + (1−α) · CE(gold)`. The lumped-topk KL coarsens both distributions to k+1 symbols: the stored top-k entries as-is, plus one lump for all remaining mass (`tail_w` on the anchor side; summed student mass over non-top-k tokens on the student side). The separate CE term guarantees the gold token is taught even when it falls outside the anchor's top-k. Compute it chunked/fused — full [T, 128,256] logits must never materialize.

**Format landmines, all deliberate:** uint32 indices because vocab 128,256 exceeds uint16; fp16 weights + fp32 `tail_w`; dense per-position coverage with an explicit `loss_mask` for excluded positions (silent absence is a contract violation); K is loader-flexible (current production K=32); `fold_version` v1 (legacy, tail folded into gold — fine, do not regenerate) vs v2 (new, unfolded). Owner measurements settled the fold question: at K≥10 the tail mass is ~1e-6 and folding is harmless.

**Thinking traces (class c):** generated by the reasoning-prompted 70B via a fixed two-question protocol, then externally verified; verified / unverified / could-not-verify are **three kept, distinctly labeled outcomes** — teaching calibrated uncertainty is a design goal. Markers come from the vocab's 256 reserved special-token slots. Length guidance ≤ ~3k thinking tokens is derived from the 4096 SWA window (response tokens must be able to attend back over the reasoning within the local window); compliance is enforced at generation time, not by post-hoc filtering.

---

## §6. The premise — cold start with donor furniture (do not "help" here)

The model's weights initialize from Llama-3.1-8B-Instruct layers (gather layer = donor layer 31 with zeroed `o_proj`/`down_proj` — an exact no-op at init). **That is the entire role of the donor weights.** There is no recovery phase, no donor-equivalence gate, no required anneal, no KL-to-donor target curve. Older versions of these ideas existed and were deliberately retired when the premise changed (spec §11 changelog). If your priors say a warm-started model "should" be recovered toward its donor — that is the retired plan. Do not reintroduce it.

What remains donor-flavored, and why:
- **M1 parity test** (bit-exact vs. HF's Llama-3.1-8B-Instruct): pure code validation of the layer math. Keep it green; it is not a training gate.
- **Step-0 sanity (spec §3.8):** a behavior-shaped check (loss well below uniform ≈ 11.76 nats, invariants green). KL-to-donor is logged as a diagnostic, not a target.
- **Anneal machinery:** present in the harness, default final-state, ablation tooling only. Do not schedule anneals for the main run.
- **The 8B donor as scorer:** it produces the anchor logits for the KD shards (data pipeline). That is "the warmth": pre-computed teaching targets, not preserved behavior.

**Optimizer:** default backend is the harness's 8-bit AdamW, which stores **√v** rather than v. This is deliberate: v's dynamic range is quadratic in the gradient's, so naive block-int8 quantization of v zeroes small-v entries and the ratio m/(v+ε) explodes; √v shares m's dynamic range, and pointwise √EMA(g²) ≥ |m| (Jensen), so zero-rounded √v implies near-zero m and the ratio stays bounded. It was cross-validated against bitsandbytes 0.50.1. Muon is fully specified in spec §5.2 as a *candidate* — **do not implement it unless the owner asks.**

**Precision:** `bf16 | fp8` flag; bf16 first on any new box to establish the reference loss curve and throughput (attribution — every validated reference in the project is bf16/fp32); FP8 (GEMMs only — embeddings, lm_head, norms, AttnRes mixing, and the loss stay high-precision) is adopted only if its curve overlays the reference. FP8 *weight storage* is rejected.

---

## §7. Deliberate exclusions — do not add these

Each was considered and rejected, mostly with evidence. If you believe one is needed, that is an escalation trigger (§1.4), not an implementation task.

- Contextual/per-position gates on the Engram table (measured collapse).
- QK-norm (two independent negative findings).
- NoPE layers anywhere (retired design).
- Donor-recovery phases, donor-equivalence training gates, mandatory anneal schedules (retired premise).
- Exact token budgets or epoch counts as requirements (owner-discretionary by spec §0.1).
- Full-RoPE globals (legacy default; Gemma's own team moved away).
- FP8 weight storage; fp32 Muon momentum on the 96GB box (memory).
- Muon implementation (specified candidate; owner-triggered).
- fp32 AdamW on the deploy box (≈132 GB static — does not fit; dev/local fallback only).
- Attention to a sink *token* (learned sink *logit* only — §4).

## §8. Where each answer lives (document map)

| Question | Owner document |
|---|---|
| What is the model, exactly? | spec §3 (+ invariants §3.7) |
| What is binding vs. advisory? | spec §0.1 |
| Shard bytes, loss formula, data classes | spec §6; production side in the pipeline doc |
| Run order, box bring-up, milestones | first-steps doc |
| Data generation (interrogation, scoring tiers, fences) | pipeline doc |
| Why a mechanism exists / common misreadings | this addendum |
| Engram evidence base | the 1B research log in `/docs/` |
| Anything unclear after all of the above | **ask the owner (§1)** |
