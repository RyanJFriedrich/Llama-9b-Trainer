#!/usr/bin/env python3
"""Dump the first N raw Q/A pairs from each Nemotron-SFT-Science-v2 config.

Writes plain text files so the owner can inspect formatting artifacts in VSCode
and decide how aggressively to clean seeds.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from datasets import load_dataset  # type: ignore[import-untyped]

from data_pipeline.common import log, set_log_file


DATASET_REPO = "nvidia/Nemotron-SFT-Science-v2"
CONFIGS = ["so", "rqa", "syn_mcq", "vendor"]


def extract_first_qa(record: dict[str, Any]) -> tuple[str, str] | None:
    """Return (question, answer) from the first user/assistant pair."""
    messages = record.get("messages")
    if not isinstance(messages, list):
        return None
    question = ""
    answer = ""
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user" and not question:
            question = content.strip() if isinstance(content, str) else ""
        elif role == "assistant" and not answer:
            answer = content.strip() if isinstance(content, str) else ""
        if question and answer:
            break
    return (question, answer) if question else None


def dump_config(config: str, n: int, output_dir: Path) -> Path:
    log(f"Streaming first {n} records from {config}...")
    ds = load_dataset(DATASET_REPO, config, split="train", streaming=True)
    output_path = output_dir / f"nemotron_raw_{config}_{n}.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# Raw Q/A dump from {DATASET_REPO} config='{config}'\n")
        f.write(f"# First {n} records\n\n")
        for i, record in enumerate(ds):
            if i >= n:
                break
            qa = extract_first_qa(record)
            if qa is None:
                f.write(f"--- Record {i + 1} ---\n[NO USER/ASSISTANT PAIR]\n\n")
                continue
            question, answer = qa
            f.write(f"--- Record {i + 1} ---\n")
            f.write("QUESTION:\n")
            f.write(question)
            f.write("\n\nANSWER:\n")
            f.write(answer)
            f.write("\n\n")

    log(f"[DONE] Wrote {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump raw Nemotron Q/A for inspection")
    parser.add_argument("--configs", nargs="+", default=CONFIGS, help="Configs to dump")
    parser.add_argument("--n", type=int, default=100, help="Records per config")
    parser.add_argument("--output-dir", default="data_pipeline/raw_inspect", help="Output directory")
    parser.add_argument("--log-file", default="data_pipeline/common.log", help="Log file path")
    args = parser.parse_args()

    set_log_file(args.log_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"[START] Dumping raw Q/A from {DATASET_REPO}", print_console=True)
    for config in args.configs:
        dump_config(config, args.n, output_dir)
    log("[DONE] All dumps complete.", print_console=True)


if __name__ == "__main__":
    main()
