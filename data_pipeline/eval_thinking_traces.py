#!/usr/bin/env python3
"""Run a 10-question test of the thinking-trace protocol.

Protocol (no tags in generation):
  1. Send the problem + thinking-stage prompt. Model reasons freely, no answer.
  2. Send the final-response prompt. Model gives the formal answer.
  3. Post-hoc wrap the first turn in <thinking>...</thinking> tags.

Outputs the combined record for each question and a short quality report.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from data_pipeline.common import ChatClient, load_config, log, set_log_file


DEFAULT_SYSTEM = "You are an expert science and reasoning assistant."


def parse_thinking_traces_md(path: Path) -> tuple[str, str]:
    """Parse docs/thinking-traces.md into (thinking_prompt, final_prompt)."""
    text = path.read_text(encoding="utf-8")
    # Split on the second question marker.
    m = re.search(r"Question\s*2\s*:", text, re.IGNORECASE)
    if not m:
        raise SystemExit(f"Could not find Question 2 in {path}")

    first = text[: m.start()]
    second = text[m.end() :]

    # Strip the "Question 1:" header and surrounding dividers/dashes.
    first = re.sub(r"^\s*Question\s*1\s*:\s*", "", first, flags=re.IGNORECASE)
    first = re.sub(r"^\s*-+\s*", "", first)
    first = re.sub(r"\s*-+\s*$", "", first)

    second = re.sub(r"^\s*-+\s*", "", second)
    second = re.sub(r"\s*-+\s*$", "", second)

    return first.strip(), second.strip()


def load_seeds(path: Path, n: int = 10) -> list[dict[str, Any]]:
    seeds: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            seeds.append(json.loads(line))
            if 0 < n <= len(seeds):
                break
    return seeds


def run_thinking_trace(
    client: ChatClient,
    question: str,
    thinking_prompt: str,
    final_prompt: str,
) -> dict[str, str]:
    """Run the two-turn protocol and return raw + post-wrapped outputs."""
    client.system_message = DEFAULT_SYSTEM

    # Turn 1: think, do not answer.
    turn1_input = f"{question}\n\n{thinking_prompt}"
    thinking_raw = client.chat([{"role": "user", "content": turn1_input}])

    # Turn 2: ask for the final answer.
    final_raw = client.chat([
        {"role": "user", "content": turn1_input},
        {"role": "assistant", "content": thinking_raw},
        {"role": "user", "content": final_prompt},
    ])

    # Post-hoc wrap thinking. Normalize whitespace inside tags.
    thinking_trimmed = thinking_raw.strip()
    wrapped = f"<thinking>\n{thinking_trimmed}\n</thinking>\n\n{final_raw.strip()}"

    return {
        "thinking_raw": thinking_raw,
        "final_raw": final_raw,
        "wrapped": wrapped,
    }


def has_tags(text: str) -> bool:
    return "<thinking>" in text or "</thinking>" in text


def has_final_answer_tags(text: str) -> bool:
    return "<final_answer>" in text.lower() or "</final_answer>" in text.lower()


def looks_like_reasoning(text: str) -> bool:
    text = text.lower()
    markers = [
        "first,", "second,", "third,", "next,", "then,", "finally,",
        "step 1", "step 2", "step 3",
        "let's think", "to solve", "to answer", "reasoning:", "therefore,",
        "one approach", "another approach", "consider", "given that",
    ]
    return any(m in text for m in markers)


def main() -> None:
    parser = argparse.ArgumentParser(description="10-question thinking-trace API test")
    parser.add_argument("--config", default="data_pipeline/config.local.yaml", help="Pipeline config YAML")
    parser.add_argument("--traces", default="docs/thinking-traces.md", help="Thinking trace template markdown")
    parser.add_argument("--seeds", default="data_pipeline/seeds_vendor_10_clean.jsonl", help="Seed JSONL path")
    parser.add_argument("--output", default="data_pipeline/thinking_trace_results.jsonl", help="Output JSONL path")
    parser.add_argument("--n", type=int, default=10, help="Number of seeds to test")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens per generation")
    args = parser.parse_args()

    full_cfg = load_config(args.config)
    cfg = full_cfg["interrogate"]["teacher"]
    set_log_file(full_cfg.get("log_file", "data_pipeline/common.log"))

    thinking_prompt, final_prompt = parse_thinking_traces_md(Path(args.traces))
    seeds = load_seeds(Path(args.seeds), args.n)
    if len(seeds) < args.n:
        log(f"[WARN] Only {len(seeds)} seed(s) available; requested {args.n}", level="WARNING", print_console=True)

    client = ChatClient(
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        model=cfg["model"],
        temperature=cfg.get("temperature", 0.7),
        max_tokens=args.max_tokens,
        top_p=cfg.get("top_p", None),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")

    log(f"[START] Thinking-trace test: {len(seeds)} question(s) -> {output_path}", print_console=True)

    stats = {
        "total": 0,
        "thinking_has_tags": 0,
        "final_has_tags": 0,
        "final_has_final_answer_tags": 0,
        "thinking_looks_like_reasoning": 0,
        "wrapped_total_chars": 0,
    }

    for seed in seeds:
        seed_id = seed.get("seed_id", f"seed_{stats['total']}")
        question = seed["question"]
        log(f"[EVAL] {seed_id}: running two-turn protocol...", print_console=True)

        result = run_thinking_trace(client, question, thinking_prompt, final_prompt)

        record = {
            "seed_id": seed_id,
            "source_config": seed.get("source_config"),
            "question": question,
            "thinking_prompt": thinking_prompt,
            "final_prompt": final_prompt,
            "thinking_raw": result["thinking_raw"],
            "final_raw": result["final_raw"],
            "wrapped": result["wrapped"],
            "metrics": {
                "thinking_has_tags": has_tags(result["thinking_raw"]),
                "final_has_tags": has_tags(result["final_raw"]),
                "final_has_final_answer_tags": has_final_answer_tags(result["final_raw"]),
                "thinking_looks_like_reasoning": looks_like_reasoning(result["thinking_raw"]),
                "thinking_chars": len(result["thinking_raw"]),
                "final_chars": len(result["final_raw"]),
                "wrapped_chars": len(result["wrapped"]),
            },
        }

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        stats["total"] += 1
        if record["metrics"]["thinking_has_tags"]:
            stats["thinking_has_tags"] += 1
        if record["metrics"]["final_has_tags"]:
            stats["final_has_tags"] += 1
        if record["metrics"]["final_has_final_answer_tags"]:
            stats["final_has_final_answer_tags"] += 1
        if record["metrics"]["thinking_looks_like_reasoning"]:
            stats["thinking_looks_like_reasoning"] += 1
        stats["wrapped_total_chars"] += record["metrics"]["wrapped_chars"]

        print(f"\n=== {seed_id} ===")
        print(f"  thinking <thinking> tags: {record['metrics']['thinking_has_tags']}")
        print(f"  final    <thinking> tags: {record['metrics']['final_has_tags']}")
        print(f"  final <final_answer> tags: {record['metrics']['final_has_final_answer_tags']}")
        print(f"  thinking looks like reasoning: {record['metrics']['thinking_looks_like_reasoning']}")
        print(f"  thinking chars: {record['metrics']['thinking_chars']}, final chars: {record['metrics']['final_chars']}")

    log("[DONE] Thinking-trace test complete.", print_console=True)
    print("\n=== Summary ===")
    print(f"  Questions: {stats['total']}")
    print(f"  Thinking turns with <thinking> tags: {stats['thinking_has_tags']}/{stats['total']}")
    print(f"  Final answers with <thinking> tags: {stats['final_has_tags']}/{stats['total']}")
    print(f"  Final answers with <final_answer> tags: {stats['final_has_final_answer_tags']}/{stats['total']}")
    print(f"  Thinking turns that look like reasoning: {stats['thinking_looks_like_reasoning']}/{stats['total']}")
    print(f"  Total wrapped chars: {stats['wrapped_total_chars']:,}")


if __name__ == "__main__":
    main()
