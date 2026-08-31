# Llama-9B — Data Generation Pipeline (v0.2)

Work order for the **data-production agent**. You own data production end-to-end; the training agent owns the harness. Your acceptance contract is the *training loader's own tests*: **if the loader rejects a shard, the shard is wrong, not the loader.** The shard contract itself is normative in the spec (§6.1) — this document is the production side of it.

Spec §0.1 applies here too: counts, batch sizes, and rates in this document are advisory; the owner may vary runs at will.

---

## §1. The four stages

```
interrogate  →  raw JSONL transcripts  →  format/mask  →  score (top-k logits)  →  shards + sidecars
```

1. **Interrogate.** A small local model (Gemma 4-class) plays a curious student and interrogates the Llama-3.3-70B API over several turns per subject. The purposeful assistant/user mismatch (Gemma asks, Llama answers) is the design. Old code for this loop exists in the project repo — find it before writing anything new.
2. **Write raw.** Raw transcripts (exact API request/response pairs, template state, timestamps, provider/version) are the **resume point**. Never re-call the API to recover; re-derive everything downstream from raw JSONL.
3. **Format/mask.** Emit the training text with the chat template that will be used at training time; user turns are **loss-masked but still scored** (keeps the choice reversible). Record the chat-template hash.
4. **Score.** Compute per-position top-k logits with the local scorer and pack shards per the contract (§3). Existing llama.cpp binaries are already compiled for this machine; the logit-extraction script already exists in the repo — locate it first.

## §2. The interrogation loop — hard anti-collapse guards (required)

A small model interrogating for hours *will* degrade. All three guards are mandatory:

- **Question dedup:** track asked-question embeddings/hashes per subject; reject near-duplicate questions.
- **Answer-loop detection:** if the 70B's responses show verbatim near-repetition across turns, terminate the conversation.
- **Assistant-token floor:** a conversation is only written if total assistant content clears a minimum (thin conversations are discarded, not padded).

Owner signs off on the interrogator model + persona prompt (D-p3) before bulk production.

## §3. Shard production contract (normative per spec §6.1)

- Dense per-position top-k over the full token stream; `uint32` indices (vocab 128,256 > uint16 range), fp16 weights, fp32 `tail_w`. Loader is K-flexible; current production K=32.
- New shards are **`fold_version: v2`** (unfolded, tail separate). Legacy folded shards are v1 and remain usable via the converter — do not regenerate them.
- Sidecar metadata per shard: `text_source`, `logit_source`, numerics path, chat-template hash, provider/version, `fold_version`, data class, α override if any.
- Assert on every shard sample: token-id identity (re-tokenize and compare) and mass accounting (top-k + tail sums to 1).

## §4. Data classes (what to produce)

| Class | Text source | Scorer | α (loss mix) | Status |
|---|---|---|---|---|
| (a) bulk natural text | Wiki/OpenWeb (+ optional FineWeb-Edu) | fp8 8B donor, llama.cpp | 0.9 | **In production by owner** — do not duplicate |
| (b) 70B-authored SFT | 3.3-70B API (this pipeline) | fp8 8B donor | ≤ 0.1 | This pipeline's main job |
| (c) thinking traces | reasoning-prompted 3.3-70B + verifier (§5) | none (pure CE) | 0 | **Decided** — build per §5 |
| (d) multilingual task data | AP corpus: summarize → translate → explain nuance, × N languages | 8B or none | 0–0.5 | Owner's prior pipeline exists (Gemma-era outputs too, §6) |
| (e) news/knowledge slices | AP-derived | 8B or none | 0–0.5 | Same pipeline as (d) |

Mixing rules that constrain *you*: ≥ 70% bulk/rehearsal in any delivered batch; never ship a news-only batch; label every shard's class in the sidecar.

## §5. Thinking-trace production (class c — decided design)

1. **Generation protocol (two questions, verbatim from the owner):**
   - **Q1** instructs the 70B to work through the problem in a fixed order: restate what's asked → classify problem type/domain → state the standard method → solve step by step → verify against the original problem (re-derive load-bearing steps, check constraints/units/definitions, confirm the answer addresses what was asked; on failure, identify the broken step, correct, continue; if the check cannot pass, say exactly where and why).
   - **Q2** asks for the final response: concise if verified; otherwise state findings, where the uncertainty is, and that the answer could not be confirmed — **do not guess**.
2. **Verification:** Q1+Q2 output goes to the verification model (Google free-tier API), which returns strict JSON. The verified / unverified / could-not-verify label is **first-class data**: keep all three outcomes; mark spans distinctly (reserved special-token slots exist in the 128,256 vocab for this). Teaching calibrated uncertainty is an explicit goal.
3. **Length compliance at generation time:** thinking spans should fit the 4096 SWA window with margin for prompt + response (guidance ≤ ~3k tokens). Prompt for concise reasoning rather than filtering long traces post-hoc.
4. Class-(c) shards are pure CE (α=0, no anchor logits needed at those positions); still emit dense scores if cheap, masked accordingly.

## §6. Off-policy text (Gemma outputs)

The owner's prior Gemma translation/nuance outputs are usable as **CE-only text** (α=0): re-tokenize with the Llama BPE, assert token-id identity, tag `text_source` honestly, ship as class (d)/(e). No scoring required. (Sequence-level use of strong off-policy text is standard; Gemma's translation strength is content our Llama-vocab teachers can't supply.)

## §7. Scoring tiers (measured)

| Tier | Model | Throughput (measured on owner's hardware) | Use for |
|---|---|---|---|
| Bulk | fp8/q8 8B donor, llama.cpp | thousands tok/s prefill (~86M tokens/day observed) | class (a),(b) |
| Middle | Q4_K_M 70B local | ~400 tok/s prefill ≈ 35M tokens/day | highest-value slices (~100M-token scale = days) |
| Upgrade | Q8/fp8 70B on deploy box | profile on first batch | bulk full-KD if ever wanted |

The middle tier is **measure-and-record**, not pass/fail: record realized top-1 agreement / set overlap vs. the 8B anchor path in the dataset card; owner accepts the risk explicitly. Generation speed is irrelevant to scoring — KD scoring is prefill-only.

## §8. Fences (what you must NOT do)

- Do not modify anything in `/train/` or the training docs. The shard contract is owned by the spec; conflicts are raised to the owner, not worked around.
- Do not regenerate or "fix" legacy v1 shards.
- Do not build class-(a) bulk scoring — the owner is running it.
- Do not invent new data classes without an owner decision.

## §9. Owner deliverables to you

| ID | Item | State |
|---|---|---|
| D-p1 | API address / model / key names for the 70B endpoint | pending |
| D-p2 | Seed subject list | pending |
| D-p3 | Interrogator model + persona sign-off | pending |
| D-p4 | First-batch size suggestion | owner's call (inspection-granularity scale) |

## §10. Changelog

- **v0.1:** initial work order (four stages, loader-as-acceptance, fences).
- **v0.2:** class (c) decided (Q1/Q2 protocol + verifier, uncertainty labeled first-class, generation-time length compliance); K=32 production noted; two-tier scoring reality folded in (`text_source`/`logit_source` metadata, spec v1.8+); AP multilingual pipeline and Gemma off-policy text added as classes (d)/(e); fences clarified (owner owns class (a) production).
