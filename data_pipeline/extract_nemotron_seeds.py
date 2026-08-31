#!/usr/bin/env python3
"""Extract seed questions from nvidia/Nemotron-SFT-Science-v2.

Streams the dataset from HuggingFace so the full ~52 GB download is not
required. Pulls the first user message from each record, strips formatting
instructions and multiple-choice options, and writes a JSONL seed file
compatible with `run_interrogate.py` (requires `seed_id` and `question`).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any

# Make UTF-8 the default on Windows terminals.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from datasets import load_dataset  # type: ignore[import-untyped]

from data_pipeline.common import log, set_log_file


DATASET_REPO = "nvidia/Nemotron-SFT-Science-v2"
CONFIGS = ["so", "rqa", "syn_mcq", "vendor"]


# Lines that are formatting instructions or answer formatting.
INSTRUCTION_LINE_PATTERNS = [
    r"^\s*(Place|Put|Provide)\s+(the|your)\s+final\s+answer",
    r"^\s*Your\s+final\s+answer",
    r"^\s*Conclude\s+with",
    r"^\s*End\s+your\s+response\s+with",
    r"^\s*End\s+with",
    r"^\s*At\s+the\s+end\s+of\s+your\s+response",
    r"^\s*At\s+the\s+end,\s+select\s+one\s+option",
    r"^\s*provide\s+the\s+selected\s+option\s+in\s+this\s+exact\s+format",
    r"^\s*Selected\s+Option\s+->",
    r"^\s*Answer\s+is\s+\[",
    r"^\s*Final\s+Answer:",
    r"^\s*Answer:\s*$",
    r"^\s*Apply\s+your\s+comprehensive\s+knowledge",
    r"^\s*Leveraging\s+your\s+extensive\s+knowledge",
    r"^\s*What\s+is\s+the\s+correct\s+answer\s+to\s+the",
    r"^\s*What\s+is\s+the\s+solution",
    r"^\s*Provide\s+a\s+precise\s+and\s+accurate\s+response",
    r"^\s*Solve\s+the\s+problem\s+and\s+provide",
    r"^\s*Answer\s+the\s+question\.?\s*$",
    r"^\s*Solve\s+the\s+following\s+problem\.?\s*$",
    r"^\s*Determine\s+the\s+correct\s+answer.*?\s*$",
    r"^\s*Apply\s+your\s+knowledge\s+to\s+answer",
    r"^\s*Provide\s+a\s+solution\s+with\s+reasoning",
    r"^\s*Your\s+task\s+is\s+to\s+solve",
    r"^\s*Answer\s+the\s+following\s+question\s+thoroughly",
    r"^\s*Which\s+option\s+is\s+correct\?",
    r"^\s*Choose\s+the\s+most\s+appropriate\s+option",
    r"^\s*Answer\s+the\s+following\s+multiple\s+choice\s+question",
    r"^\s*The\s+final\s+answer\s+must\s+be\s+placed",
    r"^\s*Solve\s+the\s+problem\s+as\s+presented",
    r"^\s*Provide\s+the\s+correct\s+answer\s+after\s+solving",
    r"^\s*Present\s+the\s+final\s+answer",
    r"^\s*Provide\s+a\s+clear\s+and\s+accurate\s+response",
    r"^\s*Select\s+the\s+correct\s+option",
    r"^\s*Select\s+the\s+most\s+appropriate\s+answer",
    r"^\s*Select\s+exactly\s+one\s+letter",
    r"^\s*putting\s+the\s+final\s+answer\s+alone\s+inside",
    r"^\s*Answer\s+the\s+question,\s+putting",
    r"^\s*Provide\s+your\s+reasoning\s+to\s+arrive\s+at\s+the\s+answer",
    r"^\s*Remember\s+to\s+end\s+with",
    r"^\s*End\s+your\s+response\s+with:",
    r"^\s*state\s+the\s+final\s+answer",
    r"^\s*and\s+then\s+state\s+the\s+final\s+answer",
]
INSTRUCTION_RE = re.compile("|".join(INSTRUCTION_LINE_PATTERNS), re.IGNORECASE)
OPTION_RE = re.compile(r"^[ \t]*[A-J][\):\.][ \t]")


def clean_question(text: str) -> str | None:
    """Strip formatting instructions and multiple-choice options from the seed."""
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        if INSTRUCTION_RE.match(line):
            continue
        if OPTION_RE.match(line):
            break
        cleaned.append(line)

    text = "\n".join(cleaned).strip()

    # Drop any short trailing fragment after the last question mark (usually formatting).
    last_q = text.rfind("?")
    if last_q != -1 and last_q < len(text) - 1:
        tail = text[last_q + 1 :].strip()
        if len(tail) < 300:
            text = text[: last_q + 1]

    text = re.sub(r"\\+\s*$", "", text).strip()

    # Strip XML-style answer tags and other formatting artifacts that leak
    # into model output if left in the seed.
    text = re.sub(r"<final_answer>.*?</final_answer>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    text = re.sub(r"</?final_answer>", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"\(\([^)]*\)\)", "", text).strip()  # double-parentheses answers
    text = re.sub(r"\\boxed\{[^}]*\}", "", text).strip()
    text = re.sub(r"Correct Answer >> [A-Z]", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"ANSWER IS [A-Z]", "", text, flags=re.IGNORECASE).strip()

    return text if len(text) > 10 else None


def extract_question(record: dict[str, Any]) -> str | None:
    """Pull the first user message content from a Nemotron record."""
    messages = record.get("messages")
    if not isinstance(messages, list):
        return None
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                cleaned = clean_question(content.strip())
                if cleaned:
                    return cleaned
    return None


def process_config(config: str, max_per_config: int, max_scan: int, exclude_formats: set[str] | None) -> list[dict[str, Any]]:
    log(f"Streaming config: {config} (max_scan={max_scan:,}, max_per_config={max_per_config:,})")
    ds = load_dataset(DATASET_REPO, config, split="train", streaming=True)
    candidates: list[dict[str, Any]] = []
    scanned = 0

    for i, record in enumerate(ds):
        if i >= max_scan:
            break
        scanned = i + 1
        metadata = record.get("metadata") or {}
        qfmt = metadata.get("question_format")
        if exclude_formats and qfmt in exclude_formats:
            continue
        question = extract_question(record)
        if question is None:
            continue

        if len(candidates) < max_per_config:
            candidates.append({"record": record, "question": question})
        else:
            j = random.randint(0, i)
            if j < max_per_config:
                candidates[j] = {"record": record, "question": question}

    log(f"  {config}: scanned {scanned:,}, kept {len(candidates)}")
    return candidates


def build_seed(record: dict[str, Any], question: str, config: str) -> dict[str, Any]:
    metadata = record.get("metadata") or {}
    return {
        "seed_id": record.get("uuid") or record.get("id") or f"{config}_{id(record)}",
        "source_config": config,
        "question": question,
        "license": record.get("license", "unknown"),
        "used_in": record.get("used_in", []),
        "topic": metadata.get("topic"),
        "subtopic": metadata.get("subtopic"),
        "question_format": metadata.get("question_format"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract seed questions from nvidia/Nemotron-SFT-Science-v2")
    parser.add_argument("--repo-id", default=DATASET_REPO, help="HuggingFace dataset repo")
    parser.add_argument("--configs", nargs="+", default=CONFIGS, help="Dataset configs to stream")
    parser.add_argument("--max-per-config", type=int, default=250, help="Questions to keep per config")
    parser.add_argument("--max-scan", type=int, default=50000, help="Records to scan per config")
    parser.add_argument("--exclude-formats", nargs="+", default=None, help="Skip records with these question_format values")
    parser.add_argument("--skip", type=int, default=0, help="Skip this many records before collecting")
    parser.add_argument("--output", default="data_pipeline/seeds.jsonl", help="Output JSONL path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-file", default="data_pipeline/common.log", help="Log file path")
    args = parser.parse_args()

    set_log_file(args.log_file)
    random.seed(args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exclude_formats = set(args.exclude_formats) if args.exclude_formats else None

    log(f"[START] Extracting seeds from {args.repo_id} -> {output_path}", print_console=True)

    total = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for config in args.configs:
            sampled = process_config(config, args.max_per_config, args.max_scan, exclude_formats)
            for item in sampled:
                seed = build_seed(item["record"], item["question"], config)
                f.write(json.dumps(seed, ensure_ascii=False) + "\n")
                total += 1

    log(f"[DONE] Wrote {total:,} seed questions to {output_path}", print_console=True)


if __name__ == "__main__":
    main()
