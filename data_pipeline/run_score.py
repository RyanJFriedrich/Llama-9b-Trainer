"""Stage 4: Anchor scoring with the local 8B donor.

Reads the formatted JSONL from Stage 3, scores every token position with either
LlamaCPPBinaries/llama-server.exe (Q8 GGUF) or the PyTorch HF donor, and writes
spec §6.2 TopK shards using the training repo's ShardWriter. Every shard sidecar
carries the required metadata.

Note: the llama-server backend requires a binary that returns prompt logprobs
via the /completion endpoint. The current LlamaCPPBinaries build returns empty
prompt_probabilities; use backend=pytorch as the working fallback.
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from data_pipeline.common import (
    find_free_port,
    install_sigint_handler,
    kill_existing_llama_servers,
    list_llama_server_pids,
    load_config,
    log,
    now_iso,
    save_manifest,
    set_log_file,
    start_llama_server,
    stop_llama_server,
    wait_for_server,
)
from data_pipeline.scorer_backends import (
    build_pytorch_scorer,
    iter_llama_server_scores,
)

# Ensure repo-root imports work.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from train.src.data.topk_loader import TopKShard
from train.src.data.topk_writer import ShardWriter


_INTERRUPTED = False


def make_sigint_handler(proc):
    def _handler(signum, frame):  # pragma: no cover
        global _INTERRUPTED
        if _INTERRUPTED:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            import os as _os
            _os.kill(_os.getpid(), signal.SIGINT)
            return
        _INTERRUPTED = True
        log("[INTERRUPT] Ctrl+C received; stopping server and cleaning up...", print_console=True)
        try:
            proc.kill()
        except Exception:
            pass
    return _handler


def build_sidecar_metadata(record: dict[str, Any], cfg: dict[str, Any], backend: str) -> dict[str, Any]:
    """Merge pipeline metadata with the formatted record's provenance."""
    meta = cfg.get("metadata", {})
    gen = record.get("generator_meta", {})

    text_source = meta.get("text_source")
    if text_source is None:
        text_source = {
            "generator_model": gen.get("teacher_model", "unknown"),
            "generator_provider": gen.get("teacher_provider", "unknown"),
            "student_model": gen.get("student_model", "unknown"),
            "date": gen.get("date", now_iso()),
        }

    logit_source = meta.get("logit_source", "llama-3.1-8b-instruct")
    if isinstance(logit_source, str):
        quant = "Q8_0" if backend == "llama_server" else cfg.get("dtype", "bf16")
        logit_source = {"scorer": logit_source, "quantization": quant, "date": now_iso()}

    return {
        "text_source": text_source,
        "logit_source": logit_source,
        "chat_template_hash": record.get("chat_template_hash", ""),
        "fold_version": meta.get("fold_version", "v2"),
        "data_class": meta.get("data_class", "b"),
        "anchor_gamma_recommended": float(meta.get("anchor_gamma_recommended", 0.1)),
    }


def validate_shard(shard_dir: Path) -> None:
    """Open the shard and assert the loader invariants."""
    shard = TopKShard(shard_dir)
    w = np.asarray(shard.topk_w, dtype=np.float32)
    tail = np.asarray(shard.tail_w)
    mass = w.sum(axis=1) + tail
    ok = np.allclose(mass, 1.0, atol=1e-3)
    log(
        f"[VALIDATE] {shard_dir}: k={shard.k}, tokens={shard.total_tokens}, "
        f"mass invariant {'PASS' if ok else 'FAIL'}",
        print_console=True,
    )
    if not ok:
        raise RuntimeError(f"Shard {shard_dir} failed mass invariant")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 4: anchor scoring with local 8B donor")
    parser.add_argument("--config", default="data_pipeline/config.yaml", help="Pipeline config YAML")
    parser.add_argument("--input", default=None, help="Override formatted JSONL input")
    parser.add_argument("--shard-dir", default=None, help="Override output shard directory")
    args = parser.parse_args()

    full_cfg = load_config(args.config)
    cfg = full_cfg["score"]
    set_log_file(full_cfg.get("log_file", "data_pipeline/common.log"))
    install_sigint_handler("Interrupt received; stopping server before exiting...")

    input_path = Path(args.input) if args.input else Path(cfg["input"])
    shard_dir = Path(args.shard_dir) if args.shard_dir else Path(cfg["shard_dir"])
    shard_dir.mkdir(parents=True, exist_ok=True)

    k = cfg.get("k", 10)
    max_conversations = cfg.get("max_conversations", 0)
    backend = cfg.get("backend", "llama_server")

    log(f"[START] Stage 4: anchor scoring -> {shard_dir} (backend={backend})", print_console=True)

    # -----------------------------------------------------------------------
    # Resume / idempotency for a single shard.
    # -----------------------------------------------------------------------
    sidecar_path = shard_dir / "sidecar.json"
    if sidecar_path.exists():
        log(f"[RESUME] {shard_dir} already finalized; validating.", print_console=True)
        validate_shard(shard_dir)
        return

    for partial in shard_dir.glob("*.npy"):
        partial.unlink()
    records_jsonl = shard_dir / "records.jsonl"
    if records_jsonl.exists():
        records_jsonl.unlink()
    manifest_path = records_jsonl.with_suffix(".manifest.json")
    if manifest_path.exists():
        manifest_path.unlink()

    # Load records.
    with open(input_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if max_conversations > 0:
        records = records[:max_conversations]
    if not records:
        log("No records to score.", print_console=True)
        return

    # -----------------------------------------------------------------------
    # Initialize the chosen backend.
    # -----------------------------------------------------------------------
    server_proc = None
    pytorch_scorer = None
    if backend == "llama_server":
        attach_port = cfg.get("attach_port")
        if attach_port is not None:
            port = int(attach_port)
            log(f"[SERVER] Attaching to existing server on port {port}", print_console=True)
        else:
            if cfg.get("kill_existing_server", False):
                kill_existing_llama_servers()
            else:
                pids = list_llama_server_pids()
                if pids:
                    log(
                        f"[WARN] Existing llama-server.exe PIDs detected: {pids}. "
                        "Set kill_existing_server: true to terminate them.",
                        level="WARNING", print_console=True,
                    )
            port = find_free_port()
            server_proc = start_llama_server(
                server_exe=Path(cfg["server_exe"]),
                model_path=Path(cfg["gguf"]),
                port=port,
                ctx_size=cfg.get("ctx_size", 8192),
                gpu_layers=cfg.get("gpu_layers", 999),
                kv_type=cfg.get("kv_type", "bf16"),
                batch_size=cfg.get("batch_size", 2048),
                ubatch_size=cfg.get("ubatch_size", 512),
                flash_attn=cfg.get("flash_attn", True),
            )
            signal.signal(signal.SIGINT, make_sigint_handler(server_proc))
        try:
            log("Waiting for server to become healthy...", print_console=True)
            if not wait_for_server(port, timeout=120):
                raise SystemExit("Server failed to start within timeout")
            log("Server healthy.", print_console=True)
        except Exception:
            stop_llama_server(server_proc)
            raise
    elif backend == "pytorch":
        log(f"Loading PyTorch donor from {cfg.get('donor', 'OriginalModel')}...", print_console=True)
        pytorch_scorer = build_pytorch_scorer(
            checkpoint_dir=cfg.get("donor", "OriginalModel"),
            device=cfg.get("device", "cuda"),
            dtype_str=cfg.get("dtype", "bf16"),
        )
    else:
        raise SystemExit(f"Unknown backend: {backend}")

    try:
        completed: set[str] = set()
        failed: dict[str, dict] = {}

        writer = ShardWriter(
            shard_dir,
            k=k,
            teacher_id="llama-3.1-8b-instruct",
            quantization="Q8_0" if backend == "llama_server" else "bf16",
            chat_template_version=records[0].get("chat_template_hash", ""),
            fold_version="v2",
        )

        pbar = tqdm(total=len(records), desc="Scoring", unit="conv")
        try:
            if backend == "llama_server":
                scorer_iter = iter_llama_server_scores(port, records, k)
            else:
                scorer_iter = pytorch_scorer(records, k)

            for record, arrays in scorer_iter:
                if _INTERRUPTED:
                    log("[INTERRUPT] Stopping before next document.", print_console=True)
                    break
                if backend == "llama_server" and server_proc is not None and server_proc.poll() is not None:
                    log(
                        "[ERROR] llama-server exited unexpectedly; aborting. Re-run to resume.",
                        level="ERROR", print_console=True,
                    )
                    break

                doc_id = record.get("id", "unknown")
                pbar.set_postfix(doc=doc_id[:16])

                if arrays is None:
                    failed[doc_id] = {"id": doc_id, "reason": "no_logits", "timestamp": now_iso()}
                    save_manifest(records_jsonl, completed, failed)
                    pbar.update(1)
                    continue

                _commit(arrays, writer, record, completed, failed, records_jsonl, pbar)
        finally:
            pbar.close()

        if writer._doc_lengths:
            sidecar = writer.finalize()
            sidecar.update(build_sidecar_metadata(records[0], full_cfg, backend))
            (shard_dir / "sidecar.json").write_text(
                json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            log(f"[DONE] Wrote shard: {sidecar['total_tokens']} tokens, {sidecar['num_documents']} docs", print_console=True)
            validate_shard(shard_dir)
        else:
            log("[DONE] No documents scored.", print_console=True)

    finally:
        if server_proc is not None:
            log("Stopping server...", print_console=True)
            stop_llama_server(server_proc)
            log("Server stopped.", print_console=True)


def _commit(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None,
    writer: ShardWriter,
    record: dict[str, Any],
    completed: set[str],
    failed: dict[str, dict],
    records_jsonl: Path,
    pbar: Any,
) -> None:
    doc_id = record.get("id", "unknown")
    if arrays is None:
        failed[doc_id] = {"id": doc_id, "reason": "no_logits", "timestamp": now_iso()}
    else:
        tokens, topk_idx, probs, loss_mask = arrays
        writer.add_document(tokens, topk_idx, probs, loss_mask=loss_mask)
        completed.add(doc_id)
        failed.pop(doc_id, None)
        log(f"[OK] {doc_id}: {len(tokens)} tokens scored")
        with open(records_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": doc_id, "tokens": len(tokens)}, ensure_ascii=False) + "\n")
    save_manifest(records_jsonl, completed, failed)
    pbar.update(1)


if __name__ == "__main__":
    main()
