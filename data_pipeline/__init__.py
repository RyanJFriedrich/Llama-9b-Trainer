"""Llama-8B++ Phase-1 class-(b) data production pipeline.

Four-stage pipeline (spec companion `docs/llama-9b-refit-data-pipeline.md`):
  1. run_interrogate.py  – Gemma student ↔ 70B teacher API → raw JSONL
  2. (raw JSONL is the resume point)
  3. run_format.py       – Llama-3.1 chat template + tokenizer + loss_mask
  4. run_score.py        – local quantized 8B donor → spec §6.2 TopK shards

Run all entry points from the repo root so `train` and `data_pipeline`
import as packages (`python -m data_pipeline.run_interrogate ...`).
"""
