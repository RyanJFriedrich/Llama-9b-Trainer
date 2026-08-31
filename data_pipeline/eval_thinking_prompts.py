#!/usr/bin/env python3
"""Evaluate forced-thinking prompting on a non-thinking teacher model.

For each seed question, runs five variants against the configured 70B teacher:

  A. Direct: answer the question normally (baseline).
  B. Two-step tagged: first ask the model to think inside
     <thinking>...</thinking> tags without solving; then ask for the formal
     answer in a follow-up turn.
  C. Single-shot tagged: ask the model to first think inside
     <thinking>...</thinking> tags and then provide the formal answer.
  D. Two-step free: ask the model to think/reason step by step without
     mentioning any tags; then ask for the formal answer.
  E. Single-shot free: ask the model to reason step by step and then answer,
     without mentioning any tags.

Raw outputs are appended incrementally to JSONL for inspection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from data_pipeline.common import ChatClient, load_config, log, set_log_file


DIRECT_SYSTEM = "You are an expert science teacher. Answer clearly, accurately, and thoroughly."

TAGGED_SYSTEM = (
    "You are an expert science teacher. When asked to think about a problem, "
    "reason through it step by step inside <thinking>...</thinking> tags. "
    "When asked for a formal answer, provide only the final solution or answer."
)

FREE_SYSTEM = (
    "You are an expert science teacher. When asked to think about a problem, "
    "reason through it step by step. When asked for a formal answer, provide "
    "only the final solution or answer."
)


def load_seeds(path: Path, n: int = 0) -> list[dict[str, Any]]:
    seeds: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            seeds.append(json.loads(line))
            if n > 0 and len(seeds) >= n:
                break
    return seeds


def run_direct(client: ChatClient, question: str) -> str:
    client.system_message = DIRECT_SYSTEM
    return client.chat([{"role": "user", "content": question}])


def run_two_step_tagged(client: ChatClient, question: str) -> dict[str, str]:
    client.system_message = TAGGED_SYSTEM
    think_prompt = (
        f"{question}\n\n"
        "Before answering, think about this problem carefully. "
        "Write your reasoning inside <thinking>...</thinking> tags. "
        "Do NOT provide a final answer yet."
    )
    thinking = client.chat([{"role": "user", "content": think_prompt}])
    answer = client.chat([
        {"role": "user", "content": think_prompt},
        {"role": "assistant", "content": thinking},
        {"role": "user", "content": "Now provide the formal answer or solution."},
    ])
    return {"thinking": thinking, "answer": answer}


def run_single_shot_tagged(client: ChatClient, question: str) -> str:
    client.system_message = TAGGED_SYSTEM
    prompt = (
        f"{question}\n\n"
        "First, think about this problem carefully inside <thinking>...</thinking> tags. "
        "Then, after the closing </thinking> tag, provide the formal answer or solution."
    )
    return client.chat([{"role": "user", "content": prompt}])


def run_two_step_free(client: ChatClient, question: str) -> dict[str, str]:
    client.system_message = FREE_SYSTEM
    think_prompt = (
        f"{question}\n\n"
        "Before answering, think about this problem carefully and reason through it step by step. "
        "Do NOT provide a final answer yet."
    )
    thinking = client.chat([{"role": "user", "content": think_prompt}])
    answer = client.chat([
        {"role": "user", "content": think_prompt},
        {"role": "assistant", "content": thinking},
        {"role": "user", "content": "Now provide the formal answer or solution."},
    ])
    return {"thinking": thinking, "answer": answer}


def run_single_shot_free(client: ChatClient, question: str) -> str:
    client.system_message = FREE_SYSTEM
    prompt = (
        f"{question}\n\n"
        "First, think about this problem carefully and reason through it step by step. "
        "Then provide the formal answer or solution."
    )
    return client.chat([{"role": "user", "content": prompt}])


def has_thinking_block(text: str) -> bool:
    return "<thinking>" in text and "</thinking>" in text


def looks_like_reasoning(text: str) -> bool:
    """Heuristic: does the text contain structural reasoning markers?"""
    text = text.lower()
    markers = [
        "first,", "second,", "third,", "next,", "then,", "finally,",
        "step 1", "step 2", "step 3",
        "let's think", "to solve", "to answer", "reasoning:", "therefore,",
        "one approach", "another approach", "consider", "given that",
    ]
    return any(m in text for m in markers)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate forced-thinking prompting")
    parser.add_argument("--config", default="data_pipeline/config.local.yaml", help="Pipeline config YAML")
    parser.add_argument("--seeds", required=True, help="Seed JSONL path")
    parser.add_argument("--output", default="data_pipeline/thinking_eval.jsonl", help="Output JSONL path")
    parser.add_argument("--n", type=int, default=3, help="Number of seeds to evaluate")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens per generation")
    args = parser.parse_args()

    full_cfg = load_config(args.config)
    cfg = full_cfg["interrogate"]["teacher"]
    set_log_file(full_cfg.get("log_file", "data_pipeline/common.log"))

    client = ChatClient(
        base_url=cfg["base_url"],
        api_key=cfg["api_key"],
        model=cfg["model"],
        temperature=cfg.get("temperature", 0.7),
        max_tokens=args.max_tokens,
        top_p=cfg.get("top_p", None),
    )

    seeds = load_seeds(Path(args.seeds), args.n)
    if not seeds:
        raise SystemExit(f"No seeds found in {args.seeds}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Fresh output file for this run.
    output_path.write_text("", encoding="utf-8")

    log(f"[START] Thinking eval: {len(seeds)} seed(s), output -> {output_path}", print_console=True)

    for seed in seeds:
        seed_id = seed.get("seed_id", "unknown")
        question = seed["question"]
        log(f"[EVAL] {seed_id}: evaluating...", print_console=True)

        direct = run_direct(client, question)
        two_step_tagged = run_two_step_tagged(client, question)
        single_shot_tagged = run_single_shot_tagged(client, question)
        two_step_free = run_two_step_free(client, question)
        single_shot_free = run_single_shot_free(client, question)

        result = {
            "seed_id": seed_id,
            "source_config": seed.get("source_config"),
            "question": question,
            "direct": direct,
            "two_step_tagged": two_step_tagged,
            "single_shot_tagged": single_shot_tagged,
            "two_step_free": two_step_free,
            "single_shot_free": single_shot_free,
            "has_thinking": {
                "direct": has_thinking_block(direct),
                "two_step_tagged_thinking": has_thinking_block(two_step_tagged["thinking"]),
                "two_step_tagged_answer": has_thinking_block(two_step_tagged["answer"]),
                "single_shot_tagged": has_thinking_block(single_shot_tagged),
                "two_step_free_thinking": has_thinking_block(two_step_free["thinking"]),
                "two_step_free_answer": has_thinking_block(two_step_free["answer"]),
                "single_shot_free": has_thinking_block(single_shot_free),
            },
            "looks_like_reasoning": {
                "direct": looks_like_reasoning(direct),
                "two_step_tagged_thinking": looks_like_reasoning(two_step_tagged["thinking"]),
                "two_step_tagged_answer": looks_like_reasoning(two_step_tagged["answer"]),
                "single_shot_tagged": looks_like_reasoning(single_shot_tagged),
                "two_step_free_thinking": looks_like_reasoning(two_step_free["thinking"]),
                "two_step_free_answer": looks_like_reasoning(two_step_free["answer"]),
                "single_shot_free": looks_like_reasoning(single_shot_free),
            },
        }

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"\n=== {seed_id} ===")
        for key, val in result["has_thinking"].items():
            print(f"  <thinking> {key}: {val}")
        for key, val in result["looks_like_reasoning"].items():
            print(f"  reasoning {key}: {val}")

    log(f"[DONE] Wrote results to {output_path}", print_console=True)


if __name__ == "__main__":
    main()
