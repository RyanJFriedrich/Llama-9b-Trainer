"""Stage 1: Gemma-student ↔ 70B-teacher interrogation loop.

Loads seed questions, spawns a curious Gemma student against a 70B teacher API,
and writes raw transcripts to `data/raw/*.jsonl`. Each conversation is saved
immediately; a manifest tracks progress so the stage is resume-safe.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from data_pipeline.common import (
    ChatClient,
    NativeChatClient,
    install_sigint_handler,
    jaccard_similarity,
    load_manifest,
    log,
    ngram_overlap,
    now_iso,
    safe_filename,
    save_manifest,
    set_log_file,
    STOP_EVENT,
)


DEFAULT_STUDENT_SYSTEM = """You are a curious student reading a reply from a teacher.
You will be shown a topic and a short conversation so far.
You interact conversationally.
Your job is to generate exactly ONE concise follow-up question.
Do NOT answer the question. Do NOT explain. Only output the follow-up question."""

DEFAULT_TEACHER_SYSTEM = """You are an expert teacher. Answer clearly, accurately, and thoroughly.
Use examples where helpful. If you are uncertain, say so."""


def load_seeds(path: Path) -> list[dict[str, Any]]:
    seeds: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seeds.append(json.loads(line))
    if not seeds:
        raise SystemExit(f"No seeds found in {path}")
    return seeds


def reconcile_output_file(output_path: Path, completed: set[str]) -> None:
    """Rewrite the JSONL so every line's id is in `completed` and ids are unique.

    A crash between appending a line and saving the manifest can leave orphan
    or duplicate conversations; this removes them so the file matches the
    resume state.
    """
    if not output_path.exists():
        return
    seen: set[str] = set()
    kept: list[str] = []
    dropped = 0
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            dropped += 1
            continue
        seed_id = obj.get("id")
        if seed_id is None or seed_id not in completed or seed_id in seen:
            dropped += 1
            continue
        seen.add(seed_id)
        kept.append(line)
    if dropped:
        output_path.write_text("\n".join(kept) + "\n", encoding="utf-8")
        log(f"[RESUME] Reconciled {output_path}: dropped {dropped} orphan/duplicate line(s), kept {len(kept)}", level="WARNING", print_console=True)


def build_student_client(cfg: dict[str, Any]) -> ChatClient | NativeChatClient:
    student = cfg["student"]
    provider = student.get("provider", "openai")
    if provider == "native_chat":
        return NativeChatClient(
            base_url=student["base_url"],
            api_key=student["api_key"],
            model=student["model"],
            api_path=student.get("api_path", "api/v1/chat"),
            temperature=student.get("temperature", 0.9),
            system_message=DEFAULT_STUDENT_SYSTEM,
        )
    return ChatClient(
        base_url=student["base_url"],
        api_key=student["api_key"],
        model=student["model"],
        temperature=student.get("temperature", 0.9),
        max_tokens=student.get("max_tokens", 128),
        top_p=student.get("top_p", 0.95),
        system_message=DEFAULT_STUDENT_SYSTEM,
    )


def build_teacher_client(cfg: dict[str, Any]) -> ChatClient:
    teacher = cfg["teacher"]
    return ChatClient(
        base_url=teacher["base_url"],
        api_key=teacher["api_key"],
        model=teacher["model"],
        temperature=teacher.get("temperature", 0.7),
        max_tokens=teacher.get("max_tokens", 1024),
        top_p=teacher.get("top_p", 0.95),
        system_message=DEFAULT_TEACHER_SYSTEM,
    )


def student_follow_up(
    client: ChatClient | NativeChatClient,
    seed_question: str,
    history: list[dict[str, str]],
    turn_number: int,
) -> str:
    """Ask the Gemma student to produce exactly one follow-up question."""
    # Native endpoint takes a single structured input string.
    if isinstance(client, NativeChatClient):
        lines = [f"Original topic: {seed_question}"]
        for msg in history[-6:]:
            role_label = "Teacher" if msg["role"] == "assistant" else "Student"
            lines.append(f"{role_label}: {msg['content']}")
        lines.append("Generate exactly one concise follow-up question.")
        prompt_input = "\n".join(lines)

        if turn_number > 1 and turn_number % 3 == 0:
            early_teacher = [m for m in history if m["role"] == "assistant"][:2]
            if early_teacher:
                target = early_teacher[0]["content"][:200]
                prompt_input += (
                    f"\n\nDEEP RECALL: Your next question must explicitly reference "
                    f"this earlier point: '{target}...' Ask how it relates to or "
                    f"contradicts what was just discussed."
                )
        return client.chat(prompt_input)

    # OpenAI-compatible endpoint.
    messages: list[dict[str, str]] = [
        {"role": "user", "content": f"Original topic: {seed_question}"},
    ]
    messages.extend(history[-6:])
    messages.append({"role": "user", "content": "Generate exactly one concise follow-up question."})

    if turn_number > 1 and turn_number % 3 == 0:
        early_teacher = [m for m in history if m["role"] == "assistant"][:2]
        if early_teacher:
            target = early_teacher[0]["content"][:200]
            recall = (
                f"DEEP RECALL: Your next question must explicitly reference "
                f"this earlier point: '{target}...' Ask how it relates to or "
                f"contradicts what was just discussed."
            )
            messages[0]["content"] += "\n\n" + recall

    return client.chat(messages)


def teacher_answer(
    client: ChatClient,
    history: list[dict[str, str]],
    question: str,
) -> str:
    """Ask the 70B teacher to answer the student's question."""
    messages = list(history)
    messages.append({"role": "user", "content": question})
    return client.chat(messages)


def is_duplicate_question(new_q: str, prior: list[str], cfg: dict[str, Any]) -> bool:
    dedup = cfg.get("dedup", {})
    if not dedup.get("enabled", True):
        return False
    method = dedup.get("method", "ngram_overlap")
    threshold = dedup.get("overlap_threshold", 0.6)
    n = dedup.get("ngram_size", 8)
    for old in prior:
        if method == "ngram_overlap":
            if ngram_overlap(new_q, old, n) >= threshold:
                return True
        else:
            if jaccard_similarity(new_q, old) >= threshold:
                return True
    return False


def is_loop(new_answer: str, prior_answers: list[str], threshold: float = 0.7) -> bool:
    if not prior_answers:
        return False
    return jaccard_similarity(new_answer, prior_answers[-1]) >= threshold


def generate_conversation(
    seed: dict[str, Any],
    student_client: ChatClient,
    teacher_client: ChatClient,
    cfg: dict[str, Any],
) -> dict[str, Any] | None:
    """Generate one multi-turn conversation with anti-collapse guards."""
    seed_id = seed.get("seed_id") or seed.get("id") or safe_filename(str(id(seed)))
    seed_question = seed.get("question", "").strip()
    if not seed_question:
        log(f"[SKIP] {seed_id}: empty seed question", level="WARNING")
        return None

    max_turns = cfg.get("max_turns", 6)
    max_total_tokens = cfg.get("max_total_tokens", 3500)
    sleep_seconds = cfg.get("sleep_seconds", 0.5)
    loop_cfg = cfg.get("loop_detection", {})
    len_cfg = cfg.get("length_floor", {})
    min_assistant_chars = len_cfg.get("min_assistant_chars", 300) if len_cfg.get("enabled", True) else 0

    messages: list[dict[str, str]] = [
        {"role": "system", "content": DEFAULT_TEACHER_SYSTEM},
        {"role": "user", "content": seed_question},
    ]
    # Lightweight history used by the student/teacher clients (no system prefix duplication).
    chat_history: list[dict[str, str]] = []
    prior_questions: list[str] = [seed_question]
    prior_answers: list[str] = []

    try:
        first_answer = teacher_answer(teacher_client, chat_history, seed_question)
    except Exception as exc:
        log(f"[ERROR] {seed_id}: teacher seed failure: {exc}", level="ERROR")
        return None

    messages.append({"role": "assistant", "content": first_answer})
    chat_history.append({"role": "user", "content": seed_question})
    chat_history.append({"role": "assistant", "content": first_answer})
    prior_answers.append(first_answer)

    for turn_num in range(1, max_turns + 1):
        if STOP_EVENT:
            break

        # Token-budget guard (cheap approximate check before each new turn).
        if sum(len(m["content"]) for m in messages) // 4 > max_total_tokens:
            log(f"[INFO] {seed_id}: token budget reached at turn {turn_num}")
            break

        try:
            student_q = student_follow_up(student_client, seed_question, chat_history, turn_num)
        except Exception as exc:
            log(f"[WARN] {seed_id}: student failed at turn {turn_num}: {exc}", level="WARNING")
            break

        if not student_q:
            break

        if is_duplicate_question(student_q, prior_questions, cfg):
            log(f"[INFO] {seed_id}: duplicate question at turn {turn_num}; ending conversation")
            break

        try:
            teacher_a = teacher_answer(teacher_client, chat_history, student_q)
        except Exception as exc:
            log(f"[WARN] {seed_id}: teacher failed at turn {turn_num}: {exc}", level="WARNING")
            break

        if loop_cfg.get("enabled", True) and is_loop(teacher_a, prior_answers, loop_cfg.get("overlap_threshold", 0.7)):
            log(f"[INFO] {seed_id}: loop detected at turn {turn_num}; ending conversation")
            break

        messages.append({"role": "user", "content": student_q})
        messages.append({"role": "assistant", "content": teacher_a})
        chat_history.append({"role": "user", "content": student_q})
        chat_history.append({"role": "assistant", "content": teacher_a})
        prior_questions.append(student_q)
        prior_answers.append(teacher_a)

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    assistant_text = "\n\n".join(m["content"] for m in messages if m["role"] == "assistant")
    if len(assistant_text) < min_assistant_chars:
        log(f"[SKIP] {seed_id}: assistant length {len(assistant_text)} < floor {min_assistant_chars}", level="WARNING")
        return None

    completed_turns = (len(messages) - 1) // 2
    return {
        "id": seed_id,
        "seed": seed,
        "turns": [{"role": m["role"], "text": m["content"]} for m in messages],
        "generator_meta": {
            "student_model": student_client.model,
            "student_provider": cfg["student"].get("provider", "openai"),
            "student_base_url": cfg["student"]["base_url"],
            "teacher_model": teacher_client.model,
            "teacher_provider": cfg["teacher"].get("provider", "openai"),
            "teacher_base_url": cfg["teacher"]["base_url"],
            "persona_prompt_hash": None,  # set below
            "date": now_iso(),
        },
        "interrogator_meta": {
            "persona": "curious_student",
            "system_prompt": DEFAULT_STUDENT_SYSTEM,
            "max_turns": max_turns,
            "completed_turns": completed_turns,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1: Gemma-student ↔ 70B-teacher interrogation")
    parser.add_argument("--config", default="data_pipeline/config.yaml", help="Pipeline config YAML")
    parser.add_argument("--output", default=None, help="Override raw JSONL output path")
    args = parser.parse_args()

    from data_pipeline.common import load_config, get_config_path
    cfg_path = get_config_path()
    full_cfg = load_config(cfg_path)
    cfg = full_cfg["interrogate"]

    log_file = full_cfg.get("log_file", "data_pipeline/common.log")
    set_log_file(log_file)
    install_sigint_handler("Interrupt received; finishing current conversation before exiting...")

    seeds_path = Path(cfg["seeds"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else output_dir / "conversations.jsonl"

    log(f"[START] Stage 1: interrogation loop -> {output_path}", print_console=True)

    seeds = load_seeds(seeds_path)
    random.seed(cfg.get("seed", 42))
    random.shuffle(seeds)

    completed, failed = load_manifest(output_path)
    reconcile_output_file(output_path, completed)
    remaining = [s for s in seeds if (s.get("seed_id") or s.get("id")) not in completed]
    max_conversations = cfg.get("max_conversations", 0)
    if max_conversations > 0:
        remaining = remaining[:max_conversations]

    if not remaining:
        log("No conversations to generate.", print_console=True)
        return

    student_client = build_student_client(cfg)
    teacher_client = build_teacher_client(cfg)

    # Record the hash of the persona prompt once.
    from data_pipeline.common import sha256_hash
    persona_hash = sha256_hash(DEFAULT_STUDENT_SYSTEM)

    pbar = tqdm(total=len(remaining), desc="Conversations", unit="conv")
    try:
        for seed in remaining:
            if STOP_EVENT:
                log("[INTERRUPT] Stopping before next conversation.", print_console=True)
                break

            seed_id = seed.get("seed_id") or seed.get("id") or safe_filename(str(id(seed)))
            pbar.set_postfix(seed=seed_id[:16])

            conv = generate_conversation(seed, student_client, teacher_client, cfg)
            if conv is None:
                failed[seed_id] = {
                    "id": seed_id,
                    "reason": "generation_failed",
                    "timestamp": now_iso(),
                }
            else:
                conv["generator_meta"]["persona_prompt_hash"] = persona_hash
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(conv, ensure_ascii=False) + "\n")
                    f.flush()
                completed.add(seed_id)
                failed.pop(seed_id, None)
                log(f"[OK] {seed_id}: {conv['interrogator_meta']['completed_turns']} turns")

            save_manifest(output_path, completed, failed)
            pbar.update(1)
    finally:
        pbar.close()

    log(
        f"[DONE] Completed {len(completed)} conversation(s), failed {len(failed)}",
        print_console=True,
    )


if __name__ == "__main__":
    main()
