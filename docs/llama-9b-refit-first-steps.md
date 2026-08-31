# Llama-9B — First Steps / Bootstrap Plan (v1.3)

Companion to the spec (v2.0). If they conflict, the spec wins. Spec §0.1 applies here: **process parameters are advisory; the owner may vary runs without spec violation.**

Repo layout: `/docs/` (this and the other project docs) · `/train/` (harness) · `/data_pipeline/` (data production, separate agent) · public release gets only the minimal training code — the Engram workbench, research logs, and internal notes stay private.

---

## §1. Where the project stands

- Harness complete through the Phase-0-era machinery: model module (config-driven), AttnRes with verified delta-sum mechanics, SWA + sink logits, anneal schedules (now default final-state, ablation tooling), TopK shard loader + fused chunked KD loss, fp32 masters + bf16 autocast trainer, resume-safe bitwise checkpoints, donor scorer, KL metric, eval tooling (perplexity, KL, NIAH, attention probes). **62/62 tests.**
- M1 re-validated on the correct donor: bit-exact HF parity vs **Llama-3.1-8B-Instruct**.
- 8-bit AdamW (sqrt(v) int8 moments) built and cross-validated against bitsandbytes 0.50.1.
- Deploy box (D1) **delivered**: rented RTX 6000 Pro Blackwell 96 GB (§3).
- Data production running: Wiki/OpenWeb bulk shards scored by the fp8 8B donor anchor, K=32, growing ~1k tok/s (~86M tokens/day).

### Premise change since this plan was written (v1.2 → v1.3)

The project is no longer a warm-start refit. Cold-start epistemology: donor weights are furniture init, the teacher is the data. Consequences below: M2 demoted, phase gates retired, the cadence is run → inspect → decide.

---

## §2. Owner deliverables (what only the owner can provide)

| ID | Item | State |
|---|---|---|
| D1 | Deploy box spec + access | **Delivered** — RTX 6000 Pro 96 GB Blackwell, owner runs scripts |
| D2 | Llama-3.1-8B-Instruct weights (init furniture + anchor + parity reference) | Delivered (in `OriginalModel/`) |
| D3 | Sample of legacy TopK shards for the converter | Delivered (converter built) |
| D4 | Corpus mix | Delivered — Wiki/OpenWeb bulk (in production); FineWeb-Edu optional breadth; AP-news slice 5–15% when its pipeline lands |
| D5 | Token budget | **Retired** — owner-discretionary per spec §0.1; runs are budgeted at runtime, not in docs |
| D6 | Box bring-up (§3) | Next owner action |

## §3. Box bring-up (the next thing that happens)

1. Pull the newest NGC PyTorch container for the RTX 6000 Pro (Blackwell); pin versions. Non-spot instance for bring-up.
2. Mount the data drive; pull repo + shards from GitHub/HuggingFace.
3. `scripts/preflight.py` — checks weights path, GPU, disk (need 1–2 TB free), system RAM (≥ 64 GB for host-RAM tables), torch build. PASS/FAIL output.
4. **bf16 reference smoke** (spec §4): short run at bf16, record the loss curve + tokens/sec as raw data, discard the checkpoint. This is the reference and the throughput calibration in one.
5. **FP8 adoption run:** same short slice at FP8; loss curve must overlay the reference within tolerance. Adopt FP8 for the real run if it does; stay bf16 if not. One variable at a time.
6. First real run: owner-set budget (the KD-shard pile is the natural unit — "what's packed" grows daily). Multi-epoch on KD slices is permitted.

## §4. Milestones, revised

- **M1 (done):** HF bit-parity on the instruct donor. Permanent unit test.
- **M2 (demoted):** was the step-0 donor-equivalence keystone. Now: step-0 sanity band (spec §3.8) + invariant self-tests I1–I4 (zero-init inertness of AttnRes/tables/sinks, delta-registration). Still required; no longer a gate with a pass metric.
- **M3–M4 (done):** config/knob schedules, trainer, scorer, KL eval, 62/62 suite.
- **M4.5 (optional, owner-undecided):** full-width 5-layer toy smoke on the 4090 — see `llama-9b-refit-local-smoke.md`. If run, it must carry the final positional scheme (SWA 4096 @ θ=10k, p-RoPE 0.25 globals) and a table, or it no longer represents production.
- **M5 (demoted to optional):** the 1B pre-flight arm stack priced *warm-start recovery cost* — that question died with the premise. Remaining value: `attn_res_scope` (all_layers vs globals_only) and optimizer A/Bs at toy scale, both owner-discretionary.
- **M6:** the first real 9B run on the D1 box. Acceptance is behavior-shaped (spec §8): instruments healthy, no alarm shapes, loss/KL curves sane. Budget is the owner's.

## §5. Standing rules for the coding agent

1. The spec's invariants I1–I9 are enforced by tests, not by care.
2. Config-driven everything; no architecture constants hardcoded outside the config.
3. Never edit a run's config mid-flight; `max_steps` exists for clean interrupts. Schedules are functions of the run config.
4. Every run logs its full config + code revision + data manifest.
5. One experimental variable at a time (optimizer, precision, architecture — never two at once).
6. Informative numbers in these docs (tokens, epochs, cadences) are advisory. Owner deviation is not a spec violation. Do not argue; log and proceed.
7. Dataset cards / run cards record what was actually done (text_source, logit_source, numerics path, fold_version, counts).

## §6. Progress log

- Harness M1–M4 complete, 62/62; M1 re-validated on instruct donor (bit-exact).
- sqrt(v) 8-bit AdamW built; bnb 0.50.1 cross-check passed (0.0688 vs 0.0722 toy finals; two independent 8-bit paths agreeing = trustworthiness signal).
- Spec v2.0 landed: cold-start premise, positional spec final (4096/10k locals, p-RoPE 0.25 globals+gather), optimizer backend-pluggable (default 8-bit AdamW; Muon specified/pending), Engram {1,2,3} locked component, normative/informative split.
- D1 delivered; bring-up sequence in §3 is the immediate next action.
- Data: class-(a) bulk shards in production (K=32, fp8 8B anchor); pipeline doc v0.2 governs the rest.
