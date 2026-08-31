"""Stage 3: Format & mask raw conversations for the Llama-3.1 student template.

Reads the raw JSONL from Stage 1, renders each conversation with the exact
Llama-3.1 chat template, tokenizes with the Llama 128,256 tokenizer, builds a
per-position loss_mask (1 = assistant content, 0 = user/system/scaffolding),
and writes a formatted JSONL. A token-id identity check verifies that
re-tokenizing the rendered stream reproduces the stored ids.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm
from transformers import AutoTokenizer

from data_pipeline.common import (
    load_config,
    log,
    now_iso,
    save_manifest,
    set_log_file,
    sha256_hash,
)


def build_loss_mask(
    tokenizer,
    full_text: str,
    messages: list[dict[str, str]],
) -> tuple[list[int], list[int]]:
    """Return (token_ids, loss_mask) for the formatted conversation.

    loss_mask[i] == 1 iff token i+1 is part of an assistant content string.
    The last token of the document is masked by the scorer, not here.
    """
    encoding = tokenizer(
        full_text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        return_attention_mask=False,
    )
    token_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]
    n = len(token_ids)

    # Build character intervals covering assistant content.
    assistant_intervals: list[tuple[int, int]] = []
    search_from = 0
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = (msg.get("text") or "").strip()
        if not content:
            continue
        idx = full_text.find(content, search_from)
        if idx == -1:
            # Fallback: try without leading whitespace in case template stripped it.
            content = (msg.get("text") or "").lstrip()
            idx = full_text.find(content, search_from)
        if idx == -1:
            log(f"[WARN] Could not locate assistant content in formatted text: {content[:80]!r}...", level="WARNING")
            continue
        assistant_intervals.append((idx, idx + len(content)))
        search_from = idx + len(content)

    # Mark tokens whose char interval overlaps any assistant content interval.
    is_assistant_token = [0] * n
    for j, (c0, c1) in enumerate(offsets):
        if c0 == c1:
            continue
        for a0, a1 in assistant_intervals:
            if c0 < a1 and c1 > a0:
                is_assistant_token[j] = 1
                break

    # Position i predicts token i+1, so loss applies when the target token
    # (i+1) is assistant content. The final position has no target -> masked.
    loss_mask = [0] * n
    for i in range(n - 1):
        loss_mask[i] = is_assistant_token[i + 1]

    return token_ids, loss_mask


def verify_token_identity(tokenizer, full_text: str, stored_ids: list[int]) -> bool:
    """Re-tokenize the rendered stream and assert it matches stored ids."""
    re_ids = tokenizer.encode(full_text, add_special_tokens=False)
    if re_ids != stored_ids:
        log(
            f"[ERROR] Token-id identity check failed: stored {len(stored_ids)} tokens, "
            f"re-tokenized {len(re_ids)} tokens. First diff at index {next((i for i, (a, b) in enumerate(zip(re_ids, stored_ids)) if a != b), 'none')}",
            level="ERROR",
        )
        return False
    return True


def process_conversation(
    conv: dict[str, Any],
    tokenizer,
    cfg: dict[str, Any],
) -> dict[str, Any] | None:
    """Format one conversation and produce the masked record."""
    messages = conv.get("turns", [])
    if len(messages) < 3:
        log(f"[SKIP] {conv.get('id')}: too few messages ({len(messages)})", level="WARNING")
        return None

    # Build OpenAI-style messages for apply_chat_template.
    chat_messages = [{"role": m["role"], "content": m["text"]} for m in messages]

    # Render without a trailing generation prompt: we are scoring existing turns.
    full_text = tokenizer.apply_chat_template(
        chat_messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    token_ids, loss_mask = build_loss_mask(tokenizer, full_text, messages)
    if not token_ids:
        log(f"[SKIP] {conv.get('id')}: empty tokenization", level="WARNING")
        return None

    if cfg.get("verify_identity", True):
        if not verify_token_identity(tokenizer, full_text, token_ids):
            log(f"[SKIP] {conv.get('id')}: token-id identity check failed", level="ERROR")
            return None

    assistant_tokens = int(sum(loss_mask))
    if assistant_tokens == 0:
        log(f"[SKIP] {conv.get('id')}: no assistant tokens found", level="WARNING")
        return None

    # Chat-template hash from the tokenizer's actual template string.
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template is None:
        chat_template = ""
    chat_template_hash = sha256_hash(str(chat_template))

    return {
        "id": conv.get("id"),
        "seed": conv.get("seed"),
        "turns": messages,
        "tokens": token_ids,
        "loss_mask": loss_mask,
        "formatted_text": full_text,
        "generator_meta": conv.get("generator_meta", {}),
        "interrogator_meta": conv.get("interrogator_meta", {}),
        "chat_template_hash": chat_template_hash,
        "token_count": len(token_ids),
        "assistant_token_count": assistant_tokens,
        "formatted_at": now_iso(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3: format & mask conversations")
    parser.add_argument("--config", default="data_pipeline/config.yaml", help="Pipeline config YAML")
    parser.add_argument("--input", default=None, help="Override raw JSONL input")
    parser.add_argument("--output", default=None, help="Override formatted JSONL output")
    args = parser.parse_args()

    full_cfg = load_config(args.config)
    cfg = full_cfg["format"]
    set_log_file(full_cfg.get("log_file", "data_pipeline/common.log"))

    input_path = Path(args.input) if args.input else Path(cfg["input"])
    output_path = Path(args.output) if args.output else Path(cfg["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log(f"[START] Stage 3: format & mask -> {output_path}", print_console=True)
    log(f"Loading tokenizer from {cfg['tokenizer']}...", print_console=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"], use_fast=True)

    total = sum(1 for _ in open(input_path, "r", encoding="utf-8") if _.strip())
    from data_pipeline.common import load_or_create_manifest
    completed, failed = load_or_create_manifest(output_path)

    with (
        open(input_path, "r", encoding="utf-8") as in_f,
        open(output_path, "a", encoding="utf-8") as out_f,
    ):
        pbar = tqdm(in_f, total=total, desc="Formatting", unit="conv")
        for line in pbar:
            line = line.strip()
            if not line:
                continue
            conv = json.loads(line)
            conv_id = conv.get("id", "unknown")
            if conv_id in completed:
                continue

            try:
                formatted = process_conversation(conv, tokenizer, cfg)
            except Exception as exc:
                log(f"[ERROR] {conv_id}: {exc}", level="ERROR")
                failed[conv_id] = {"id": conv_id, "reason": str(exc), "timestamp": now_iso()}
                save_manifest(output_path, completed, failed)
                continue

            if formatted is None:
                failed[conv_id] = {"id": conv_id, "reason": "filtered", "timestamp": now_iso()}
            else:
                out_f.write(json.dumps(formatted, ensure_ascii=False) + "\n")
                out_f.flush()
                completed.add(conv_id)
                failed.pop(conv_id, None)
                log(f"[OK] {conv_id}: {formatted['assistant_token_count']} assistant tokens")
            save_manifest(output_path, completed, failed)

    log(f"[DONE] Formatted {len(completed)} conversation(s), failed {len(failed)}", print_console=True)


if __name__ == "__main__":
    main()
