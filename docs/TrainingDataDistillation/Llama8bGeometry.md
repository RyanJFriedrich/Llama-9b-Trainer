# Llama-8B Teacher: Rank Geometry of the Predictive Distribution

Measured 2026-08-31 over **39,841,024 scored positions** (38 shards) of the
bulk corpus produced by `data_pipeline/run_bulk_score.py` (see `NPZFormat.md`
for the data contract).

- **Teacher**: `QuantizedModel/meta-llama-3.1-8b-instruct.Q8_0.gguf`
  (Llama-3.1-8B-Instruct-abliterated, Q8_0) via the fork llama-server
  (`prompt_logprobs` prefill extraction, bf16 KV, flash attn).
- **Corpus**: 50/50 seeded interleave of `wikimedia/wikipedia` (20231101.en)
  and `HuggingFaceFW/fineweb-edu` (sample-10BT), packed into 8192-token
  chunks, dense scoring of every position.
- **k = 32** per position; rank order reconstructed by sorting each row's
  stored probs descending (rows are shuffled at write time by design).
- Reproduce: `python data_pipeline/rank_geometry.py` →
  `data_pipeline/rank_geometry.json`.

## Headline: the tail is a power law, not an exponential

Fit to the geometric-mean mass per rank (ranks 1–32):

| shape | parameters | R² |
|---|---|---|
| **power law** `mass ∝ r^(−α)` | **α = 1.99** | **0.993** |
| stretched exponential `e^(−β·r^γ)` | γ = 0.2 (grid edge) | 0.969 |
| exponential `e^(−β·r)` | β = 0.163 | 0.794 |

The exponential is decisively rejected. The stretched exponential's best γ
hit the search-grid boundary in the direction of becoming a power law.

Note the exponent: **α ≈ 2, not 1**. Corpus token *frequencies* follow
Zipf's α ≈ 1; the per-position *predictive* distribution is steeper —
conditioning on context collapses the candidate set.

## The geometry is a one-parameter family: α depends on head mass p₁

Power-law fit per bucket of p₁ (the rank-1 mass at that position):

| p₁ bucket | n positions | α | R² | mean mass captured in top-32 |
|---|---|---|---|---|
| < 0.1     | 1,354,799   | 0.86 | 0.916 | 0.516 |
| 0.1–0.3   | 9,368,533   | 1.46 | 0.979 | 0.745 |
| 0.3–0.6   | 12,247,278  | 1.97 | 0.998 | 0.837 |
| 0.6–0.9   | 8,111,885   | 2.24 | 0.989 | 0.910 |
| > 0.9     | 8,758,529   | 2.54 | 0.899 | 0.980 |

- **α rises monotonically with p₁** (0.86 → 2.54): the more confident the
  head, the steeper the tail. One scalar (p₁) parameterizes the geometry.
- **The head overshoots the law.** A pure zeta distribution with α = 2.54
  puts 1/ζ(2.54) ≈ 0.75 on rank 1, but the >0.9 bucket sits at 0.9+. The
  true shape is a hybrid: **sharp head + Zipfian tail from rank 2 down.**
  Mid-entropy buckets fit best (R² 0.998); the extremes fit worst.
- The overall α = 1.99 curve is a *mixture* over these buckets; use the
  conditional α(p₁), not the global fit, when shaping targets.

## Per-rank mass table (all 39.8M positions)

| rank | mean | geomean | p25 | median | p75 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.55204 | 0.449383 | 0.284667 | 0.508864 | 0.85894 |
| 2 | 0.12248 | 0.056457 | 0.042832 | 0.106709 | 0.18181 |
| 3 | 0.05920 | 0.023150 | 0.013596 | 0.051500 | 0.09006 |
| 4 | 0.03600 | 0.012904 | 0.006303 | 0.031063 | 0.05570 |
| 5 | 0.02461 | 0.008322 | 0.003621 | 0.020704 | 0.03883 |
| 6 | 0.01813 | 0.005866 | 0.002351 | 0.014691 | 0.02909 |
| 7 | 0.01406 | 0.004383 | 0.001649 | 0.010909 | 0.02290 |
| 8 | 0.01131 | 0.003417 | 0.001223 | 0.008397 | 0.01864 |
| 9 | 0.00935 | 0.002750 | 0.000947 | 0.006656 | 0.01556 |
| 10 | 0.00788 | 0.002265 | 0.000755 | 0.005395 | 0.01323 |
| 11 | 0.00666 | 0.001831 | 0.000572 | 0.004374 | 0.01134 |
| 12 | 0.00581 | 0.001558 | 0.000478 | 0.003684 | 0.00992 |
| 13 | 0.00512 | 0.001347 | 0.000405 | 0.003144 | 0.00876 |
| 14 | 0.00455 | 0.001178 | 0.000349 | 0.002716 | 0.00781 |
| 15 | 0.00409 | 0.001040 | 0.000304 | 0.002371 | 0.00701 |
| 16 | 0.00370 | 0.000927 | 0.000268 | 0.002089 | 0.00633 |
| 17 | 0.00337 | 0.000833 | 0.000238 | 0.001856 | 0.00575 |
| 18 | 0.00309 | 0.000753 | 0.000213 | 0.001662 | 0.00525 |
| 19 | 0.00284 | 0.000684 | 0.000192 | 0.001498 | 0.00482 |
| 20 | 0.00263 | 0.000626 | 0.000174 | 0.001355 | 0.00444 |
| 21 | 0.00244 | 0.000574 | 0.000158 | 0.001234 | 0.00411 |
| 22 | 0.00227 | 0.000530 | 0.000145 | 0.001129 | 0.00381 |
| 23 | 0.00213 | 0.000490 | 0.000133 | 0.001036 | 0.00355 |
| 24 | 0.00199 | 0.000455 | 0.000123 | 0.000955 | 0.00332 |
| 25 | 0.00187 | 0.000424 | 0.000114 | 0.000884 | 0.00310 |
| 26 | 0.00177 | 0.000396 | 0.000106 | 0.000820 | 0.00291 |
| 27 | 0.00167 | 0.000371 | 0.000099 | 0.000764 | 0.00274 |
| 28 | 0.00158 | 0.000349 | 0.000092 | 0.000713 | 0.00259 |
| 29 | 0.00150 | 0.000328 | 0.000086 | 0.000667 | 0.00244 |
| 30 | 0.00142 | 0.000310 | 0.000081 | 0.000626 | 0.00231 |
| 31 | 0.00135 | 0.000293 | 0.000076 | 0.000588 | 0.00219 |
| 32 | 0.00108 | 0.000222 | 0.000057 | 0.000421 | 0.00161 |

The mean exceeds the median at every rank — right-skewed; the mixture of
peaked and flat positions inflates the average. The median/geomean curves
are the typical-position geometry.

## Coverage context (k choice)

Mean captured mass at k=32 is 0.895 (median 0.950). Measured curve over k
(4k-token probe, same teacher/corpus): k=12 → 0.834, k=16 → 0.855,
k=25 → 0.882, k=32 → 0.895, k=48 → 0.914, k=64 → 0.926. Even at k=64,
~7.5% of mean mass lies beyond the cutoff — the 8B has a genuinely long
tail at flat positions. The lumped tail (`1 − row mass`) is therefore
material and must stay explicit in any loss.

## Recipe for geometry-shaped soft targets

Given only an approximate rank order for a position:

1. Set head mass p₁ (measured, or imposed by the shaping).
2. α(p₁): interpolate the measured table (0.86 @ p₁<0.1 … 2.54 @ p₁>0.9).
3. Tail mass at rank r ≥ 2: `(1 − p₁) · r^(−α(p₁)) / Σ_{r=2..k} r^(−α(p₁))`.
4. Beyond k: lump into a single tail bucket; keep it explicit in the loss.

## Caveats

- Teacher is Q8_0-quantized; Q8-vs-bf16 fidelity of this scorer was never
  measured (the spec §6 pre-flight gate). The geometry could be modestly
  sharper or flatter at bf16.
- Rank-32 entries are slightly polluted: when GT lies outside the teacher's
  top-32, it is stored at slot 0 with its true (small) prob and displaces
  the true rank-32 token (see NPZFormat.md). Rare; ignore rank 32 in fits
  if it matters.
- Geometry was measured on wiki/web bulk text. Other domains (code,
  dialogue) will have their own α(p₁); re-measure before shaping those.
