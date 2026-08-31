# Annex A1 — Engram Addressing: Canonical Specification

**Status:** normative addendum to `llama-9b-refit-spec.md` §3.6 and `llama-9b-agent-addendum.md` §3
**Version:** 1.1
**Purpose:** pins the four addressing details the coding agent escalated: (1) exact hash function, (2) per-order/per-head salt scheme, (3) row init distribution, (4) prefetch granularity. Self-contained: implementing this annex requires no access to the 1B harness or any research log.

> **Authority note for the coding agent:** this annex is the single source of truth for Engram addressing. The 1B harness's addressing code is **not required** and must not be requested. Bit-level parity with the 1B harness is explicitly **not** a goal — the 9B tables are new, sized differently, and trained from scratch. Pattern-consistency with the published DeepSeek Engram scheme (arXiv:2601.07372, §2.1) is the target. As with the main spec, owner runtime deviation is not a violation; do not argue.

---

## A1.0 The four answers at a glance

| Escalated question | Answer |
|---|---|
| Exact hash function | **n=1 (unigram):** affine map `idx = (A·c + B) mod M` with `M` prime ≥ vocab — injective by construction, collision-free guaranteed. **n≥2:** 64-bit multiplicative-XOR mix over canonical token ids, `idx = state mod M_{n,k}`, `M_{n,k}` prime. |
| Salt scheme | Per-(order, head) 64-bit constants derived from a pinned SHA-256 stream (§A1.3). No hand-picked magic numbers; constants are reproducible from a label string. |
| Row init distribution | `row ~ Uniform(−0.01, +0.01)`, initialized in fp32, stored bf16. (Harness parity: this is the exact distribution the 1B program's 262k-row table used, measured healthy.) |
| Prefetch granularity | **Per-batch, deduplicated, one batch ahead.** Indices are computed on CPU from token ids alone (addressing has no GPU dependency); the dataloader emits them alongside tokens; touched rows are gathered through a pinned staging buffer with a non-blocking copy while the previous batch computes. Single injection point (layer 3 output), so there is no per-layer dimension to the question. |

---

## A1.1 Why unigrams are special (the measurement this design explains)

The 1B harness measured **zero** unigram collisions for two independent salts. A random hash mapping ~128k keys into 2^18–2^21 slots must collide thousands of times (birthday math), so the harness's unigram mapping was necessarily *structured*, not a plain hash. Rather than guess that structure, this annex pins a construction with the same guaranteed property:

> **Unigram addressing is injective by construction** (§A1.4). The zero-collision measurement is reproduced as a theorem, not as luck.

For n≥2, the key space is ~10¹⁰–10¹⁵ and collisions are unavoidable and harmless — this is why there are two heads per order: distinct per-head primes and constants decorrelate the collision sets, so a token pair that collides in head 1 essentially never collides in head 2.

---

## A1.2 Stage 0 — canonical token ids (compression) `[FLEXIBLE, default ON]`

Following DeepSeek §2.1, hashing operates on **canonical ids**, not raw token ids. This merges surface variants ("Apple", "apple", "Ａｐｐｌｅ") into one memory row — the right semantics for a lexical memory.

**Construction (pinned):**

```
canon_string(token) = casefold(NFKC(text(token)))
class(token)        = canon_string  (equivalence = identical canon_string)
canonical_id(token) = min { token_id | token_id in class(token) }
```

- Normalization is exactly **NFKC then casefold** — nothing else. No whitespace stripping, no punctuation rules, no stemmer. (DeepSeek lists "etc." beyond these; we deliberately pin the minimal two-step version.)
- The canonical representative is the **minimum raw token id** in each equivalence class: deterministic, reproducible, requires no stored table beyond the map itself.
- Build script computes `P: V → V′` once from the Llama 3.1 tokenizer and saves it as a `uint32` array of length 128,256 (`P[t] = canonical_id(t)`), checksummed into the run config. This is a data artifact; recompute only if the tokenizer changes.
- `|V′|` is an output of the build script. Expect roughly 95k–100k (DeepSeek measured a 23% reduction on their 128k vocab; ours will differ).

**Fallback (owner vetoes compression):** `P = identity`, `|V′| = 128,256`, unigram modulus `M = 128257` (smallest prime ≥ 128256). Everything else in this annex is unchanged.

---

## A1.3 Stage 1 — per-(order, head) constant derivation

All salt constants are derived from a pinned SHA-256 stream so the scheme extends to any number of heads without new magic numbers:

```
def const64(n: int, k: int, field: str) -> int:
    label = f"engram9b.v1:n={n}:k={k}:{field}".encode()
    return int.from_bytes(hashlib.sha256(label).digest()[:8], "little")

MUL(n, k)  = const64(n, k, "mul")  | 1        # forced odd
SEED(n, k) = const64(n, k, "seed")
AFFA(n, k) = const64(n, k, "affA") % (M(n, k) - 1) + 1   # in [1, M-1], nonzero mod M
AFFB(n, k) = const64(n, k, "affB")
```

Fields per (n, k): `mul`, `seed` for n≥2; `affA`, `affB` for n=1. The label string `engram9b.v1` is the scheme version; bump it only with a deliberate re-addressing decision (invalidates all trained tables).

---

## A1.4 Stage 2 — addressing functions

**Table sizes (rows = modulus = prime), per head:**

| (order n, head k) | rows M_{n,k} | bf16 host memory (row_dim 256) |
|---|---|---|
| unigram (1, 1) and (1, 2) | smallest prime ≥ **max(canonical_id) + 1** (build-script computed; compression OFF fallback: 128257) | ~66 MB each |
| bigram (2, 1) | 1048573 | 536.9 MB |
| bigram (2, 2) | 1048571 | 536.9 MB |
| trigram (3, 1) | 1048559 | 536.9 MB |
| trigram (3, 2) | 1048549 | 536.9 MB |

Distinct primes per (n, k) — including across orders — decorrelate collisions (DeepSeek §2.1). Total host footprint ≈ 2.2 GB bf16 rows (+ ~2.4 GB host-side 8-bit optimizer state), comfortable in the ≥64 GB system RAM gate.

**Unigram (n=1), head k:**

```
c   = P[token]
idx = (AFFA(1, k) * c + AFFB(1, k)) % M_uni
```

Since `M_uni` is prime, `AFFA ≢ 0 (mod M_uni)`, and `c ≤ max(canonical_id) < M_uni`, this map is a bijection on `{0, …, M_uni−1}` restricted to the image of `P`: **provably collision-free**, reproducing the 1B measurement by construction. (v1.1 erratum: the modulus must clear the SPARSE min-id representatives, not the class count |V′| — see changelog. Built artifact: |V′| = 101,747, max(canonical_id) = 128,255 → M_uni = 128,257, landing exactly on the compression-OFF fallback. Cost: ~26.5k unreachable rows per unigram head, ~13.5 MB bf16 — accepted, no dense re-indexing layer.)

**n-gram (n≥2), head k — 64-bit multiplicative-XOR:**

```
GOLD = 0x9E3779B97F4A7C15          # 2^64 / golden ratio
state = SEED(n, k)
for j in 0 .. n-1:                 # c_0 is the OLDEST token in the window
    c     = P[token_{t - (n-1) + j}]
    state = ((state ^ (c + GOLD)) * MUL(n, k)) mod 2^64
idx = state % M[n, k]
```

Properties this pins: order-sensitive (the loop is positional; "dog bites man" ≠ "man bites dog"); no self-cancellation (multiply after xor, odd multiplier); per-head decorrelation (distinct `SEED`, `MUL`, and prime `M`); 64-bit mixing before the prime-mod (never mod-then-mix).

**Readout (unchanged from spec §3.6):** per order, concatenate the 2 head rows → `g · U(RMSNorm(concat))`, `U` zero-init (invariant I1), `g` init 1.0, no contextual gate. Contribution is added at layer 3 output and registered into the block delta-sum (invariant I2).

---

## A1.5 Row initialization

```
row ~ Uniform(−0.01, +0.01)   # sampled in fp32
stored: bf16, host RAM
```

Rationale: exact parity with the 1B program's 262k-row table init, which was measured healthy (no saturation, clean gate-gradient behavior). The agent's proposed `N(0, 0.02)` is rejected — not because it would fail, but because the uniform init has in-house evidence behind it and there is no reason to re-open a measured choice.

---

## A1.6 Prefetch and update path (host-RAM tables)

Injection happens exactly once per forward pass (layer 3 output), so prefetch granularity has one sensible answer:

1. **Indices come from token ids alone.** Addressing (§A1.2–A1.4) is pure CPU integer work on the token-id tensor. No GPU tensor is an input.
2. **Dataloader emits indices with the batch.** Per token: 6 `uint32` values (3 orders × 2 heads). If an NPZ shard carries a precomputed index sidecar from an older addressing scheme, **ignore it** — recompute from token ids using this annex. Single source of truth, no stale-index risk.
3. **Prefetch one batch ahead, deduplicated.** While batch *i* is on the GPU, the CPU computes batch *i+1*'s index tensor, takes `unique()` per table, and stages the touched rows into a pinned host buffer, then a non-blocking H2D copy. Expected touched-set per 4096-token batch: ≤ 24,576 row-slots, typically far fewer after dedup — a few MB over PCIe, trivially hidden.
4. **Backward is sparse.** Gradients exist only for gathered rows. Scatter-add per unique row (weight by occurrence count if the gather deduplicated), then apply the optimizer update to exactly those rows host-side. Untouched rows are not read, not decayed, not moved.

This mirrors the production pattern (vLLM PLE host-RAM offload with async prefetch, as shipped for Qwen3.8-Next-class models) scaled down to a single-injection-point training loop.

---

## A1.7 Optimizer group for table rows `[FLEXIBLE, default adopted from DeepSeek Appendix A]`

Within the locked 8-bit AdamW backend (spec §5), the Engram rows form their own parameter group:

- **LR multiplier ×5** relative to base LR (DeepSeek Appendix A: `engram_lr_mult: 5`). Compensation for sparse exposure: each row is touched in a small fraction of steps.
- **Weight decay 0.0** (DeepSeek Appendix A: `engram_wd: 0.0`). Rows are embeddings; decoupled WD on sparsely-touched rows would be biased by touch frequency.
- 8-bit state (`m`, `√v`) lives host-side, allocated densely (1.2 B params × 2 B ≈ 2.4 GB — trivial), updated only on touched rows.
- **Update cadence (v1.1, normative):** the ×5 multiplier is defined per dense-equivalent step. Table gradients accumulate host-side in fp32 across the gradient-accumulation window (scatter-add per unique row), and the sparse Adam step runs ONCE per dense step — a row touched by N micro-batches gets one update with the summed grad, never N compounded ×5 updates. Zero-accumulated-grad rows (boundary-invalid only) are skipped: no state decay, no drift.

`g` and `U` are ordinary device-side parameters in the base group (no multiplier, standard WD on `U` per its matrix class; `g` is a scalar gate — no WD, matching norms/gains convention).

---

## A1.8 Acceptance tests (the agent's definition of done for this annex)

1. **Injectivity test:** for both unigram heads, `len({addr(c) for c in range(|V′|)}) == |V′|`. Must pass exactly. This is the theorem of §A1.4 expressed as a test.
2. **Decorrelation test:** on a 1M-token corpus sample, bigram head-1 and head-2 collision sets intersect at chance level (≈ |collisions₁|·|collisions₂| / M). Loose tolerance; failure means a constant-derivation bug.
3. **Order-sensitivity test:** `addr_3gram(a,b,c) != addr_3gram(c,b,a)` for a probe set (except palindromes).
4. **Determinism test:** same token-id sequence → identical index tensor across two processes/machines. (Guards the SHA-256 constant stream and the compression artifact.)
5. **Inert-at-init test:** with `U` zeroed, a forward pass with tables present matches a tables-disabled forward to bf16 tolerance (restates invariant I1 at integration level).
6. **Prefetch overlap smoke:** index computation + row staging for batch *i+1* completes within batch *i*'s step time at the target batch size. If not, widen the lookahead to two batches before touching anything else.

---

## A1.9 What this annex deliberately does NOT pin

- The 1B harness's exact salts/init: unnecessary (new tables, new scale, new model — there is no continuity to preserve). The measured collision-free property is reproduced by construction (§A1.4), which was the only thing worth carrying over.
- DeepSeek's literal hash constants: their repo ships an implementation, but this project is pattern-consistent, not weight-compatible; our constants (§A1.3) serve identically.
- Touch-frequency instrumentation cadence: covered by spec §8 instrumentation; not an addressing concern.

## Changelog

| Ver | Change |
|---|---|
| 1.0 | Initial issue. Resolves escalation #1 (hash, salts, row init, prefetch). Adopts DeepSeek §2.1 addressing pattern (canonical-id compression, prime tables, multiplicative-XOR, per-head decorrelation) and Appendix A training group (LR ×5, WD 0.0); pins injective unigram construction explaining the 1B zero-collision measurement. |
| 1.1 | **Erratum (caught by the coding agent, verified at prod scale):** M_uni = smallest prime ≥ max(canonical_id)+1, not ≥ |V′| — min-id representatives are sparse (max 128,255 > |V′| = 101,747), so a |V′|-sized modulus would wrap high ids and quietly break injectivity. Fix lands exactly on the compression-OFF fallback (128,257); acceptance test A1.8.1 iterates over the image of P, not range(|V′|). **Cadence rule (A1.7):** table grads accumulate host-side fp32 across the grad-accum window; sparse Adam steps once per dense-equivalent step (×5 never compounds to ×5N). |

*Provenance: DeepSeek Engram paper arXiv:2601.07372 §2.1 + Appendix A (fetched 2026-08-31); 1B harness measurements as reported in the project transcript; vLLM PLE offload pattern for prefetch.*
