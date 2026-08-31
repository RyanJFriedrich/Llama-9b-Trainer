"""Shared helpers for the Llama-8B++ data production pipeline.

Implements config loading, logging, retry logic, OpenAI-compatible clients,
llama-server lifecycle, and small utilities used by all stages.
"""
from __future__ import annotations

import json
import os
import random
import re
import signal
import socket
import string
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import requests
import yaml

# Ensure repo-root imports work when scripts are run as `python -m ...`
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.utils.log import log as _log

LOG_FILE: Path | None = None


def set_log_file(path: str | Path) -> None:
    global LOG_FILE
    LOG_FILE = Path(path)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def log(*args: Any, level: str = "INFO", print_console: bool = False, **kwargs: Any) -> None:
    """Pipeline logging wrapper; lands in the configured log file."""
    filename = str(LOG_FILE) if LOG_FILE is not None else "data_pipeline/common.log"
    _log(*args, level=level, filename=filename, print_console=print_console, **kwargs)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _substitute_env(value: Any) -> Any:
    if isinstance(value, str):

        def repl(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in os.environ:
                raise KeyError(f"Config references unset environment variable ${key}")
            return os.environ[key]

        return _ENV_RE.sub(repl, value)
    if isinstance(value, dict):
        return {k: _substitute_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute_env(v) for v in value]
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    """Load pipeline config with ${ENV_VAR} substitution."""
    text = Path(path).read_text(encoding="utf-8")
    raw = yaml.safe_load(text)
    return _substitute_env(raw)


def get_config_path(argv: list[str] | None = None) -> Path:
    """Parse --config from argv; default to data_pipeline/config.yaml."""
    argv = argv if argv is not None else sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg == "--config" and i + 1 < len(argv):
            return Path(argv[i + 1])
        if arg.startswith("--config="):
            return Path(arg.split("=", 1)[1])
    return Path("data_pipeline/config.yaml")


# ---------------------------------------------------------------------------
# Retry / API clients
# ---------------------------------------------------------------------------
STOP_EVENT = False


def install_sigint_handler(message: str = "Interrupt received; finishing current unit then exiting...") -> None:
    def _handler(signum: Any, frame: Any) -> None:  # pragma: no cover
        global STOP_EVENT
        if STOP_EVENT:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            os.kill(os.getpid(), signal.SIGINT)
            return
        STOP_EVENT = True
        log(message, level="WARNING", print_console=True)
    signal.signal(signal.SIGINT, _handler)


def is_transient(exc: Exception) -> bool:
    text = str(exc).lower()
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    return (
        (code == 429)
        or (code is not None and 500 <= int(code) < 600)
        or ("rate limit" in text)
        or ("resource exhausted" in text)
        or ("too many requests" in text)
        or ("timeout" in text)
        or ("deadline exceeded" in text)
        or ("connection" in text and "reset" in text)
        or ("service unavailable" in text)
        or ("internal server error" in text)
    )


def retry_with_backoff(
    func,
    max_retries: int = 5,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    rate_limit_delay: float = 60.0,
) -> Any:
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as exc:
            last_exc = exc
            if attempt == max_retries:
                raise last_exc
            if STOP_EVENT:
                raise last_exc
            if is_transient(exc):
                # Rate-limit gets a fixed visible cooldown.
                text = str(exc).lower()
                if "rate limit" in text or "resource exhausted" in text or "too many requests" in text:
                    log(f"Rate limit observed, timeout {rate_limit_delay:.0f}s", level="WARNING")
                    time.sleep(rate_limit_delay)
                else:
                    delay = min(base_delay * (2 ** attempt) + random.random(), max_delay)
                    log(f"Transient API failure ({exc}); retry {attempt + 1}/{max_retries} after {delay:.1f}s", level="WARNING")
                    time.sleep(delay)
            else:
                raise last_exc
    raise last_exc  # pragma: no cover


class ChatClient:
    """Thin OpenAI-compatible chat client with retry/backoff."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = 1024,
        top_p: float | None = None,
        system_message: str | None = None,
    ) -> None:
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.system_message = system_message

    def chat(self, messages: list[dict[str, str]]) -> str:
        full_messages = list(messages)
        if self.system_message is not None:
            full_messages = [{"role": "system", "content": self.system_message}] + full_messages

        def call() -> Any:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": full_messages,
                "temperature": self.temperature,
            }
            if self.max_tokens is not None:
                kwargs["max_tokens"] = self.max_tokens
            if self.top_p is not None:
                kwargs["top_p"] = self.top_p
            return self.client.chat.completions.create(**kwargs)

        response = retry_with_backoff(call)
        return (response.choices[0].message.content or "").strip()


class NativeChatClient:
    """Custom non-OpenAI chat endpoint used by some local servers (e.g. LM Studio's
    `/api/v1/chat` style interface). Sends `system_prompt` + `input` and parses
    the returned `output` array for the message payload.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        api_path: str = "api/v1/chat",
        temperature: float = 0.7,
        system_message: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.api_path = api_path.lstrip("/")
        self.temperature = temperature
        self.system_message = system_message

    def chat(self, prompt_input: str) -> str:
        """Send a single-turn prompt_input; return the message content."""
        url = f"{self.base_url}/{self.api_path}"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: dict[str, Any] = {
            "model": self.model,
            "input": prompt_input,
            "temperature": self.temperature,
        }
        if self.system_message is not None:
            payload["system_prompt"] = self.system_message

        def call() -> Any:
            response = requests.post(url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            return response.json()

        data = retry_with_backoff(call)
        for item in data.get("output", []):
            if item.get("type") == "message":
                return (item.get("content") or "").strip()
        return ""


# ---------------------------------------------------------------------------
# llama-server lifecycle
# ---------------------------------------------------------------------------
def find_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def list_llama_server_pids() -> list[int]:
    """Return PIDs of any running llama-server.exe processes (Windows)."""
    try:
        result = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq llama-server.exe", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, check=False,
        )
        pids = []
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            pid_str = parts[1].strip().strip('"')
            if pid_str.isdigit():
                pids.append(int(pid_str))
        return pids
    except Exception:
        return []


def kill_existing_llama_servers() -> None:
    for pid in list_llama_server_pids():
        log(f"[SERVER] Killing existing llama-server.exe PID {pid}", print_console=True)
        try:
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], check=False, capture_output=True)
        except Exception as exc:
            log(f"[WARN] Failed to kill PID {pid}: {exc}", level="WARNING", print_console=True)
    time.sleep(1)


def start_llama_server(
    server_exe: Path,
    model_path: Path,
    port: int,
    ctx_size: int,
    gpu_layers: int = 999,
    kv_type: str = "bf16",
    batch_size: int = 2048,
    ubatch_size: int = 512,
    flash_attn: bool = True,
    extra_args: list[str] | None = None,
) -> subprocess.Popen:
    """Spawn LlamaCPPBinaries/llama-server.exe with settings tuned for scoring."""
    if gpu_layers < 0:
        gpu_layers = 999
    cmd = [
        str(server_exe),
        "--model", str(model_path.resolve()),
        "--ctx-size", str(ctx_size),
        "--fit", "off",
        "--gpu-layers", str(gpu_layers),
        "-ctk", kv_type,
        "-ctv", kv_type,
        "--cache-ram", "0",
        "--ctx-checkpoints", "0",
        "--batch-size", str(batch_size),
        "--ubatch-size", str(ubatch_size),
        "--parallel", "1",
        "--no-webui",
        "--host", "127.0.0.1",
        "--port", str(port),
    ]
    if flash_attn:
        cmd.extend(["--flash-attn", "on"])
    if extra_args:
        cmd.extend(extra_args)

    env = dict(os.environ)
    env["PATH"] = str(server_exe.parent) + os.pathsep + env.get("PATH", "")

    log(f"[SERVER] Starting {server_exe.name} on port {port}", print_console=True)
    stderr_path = Path(os.environ.get("TEMP", "/tmp")) / f"llama_server_{port}.log"
    stderr_fh = open(stderr_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=server_exe.parent,
        stdout=subprocess.DEVNULL,
        stderr=stderr_fh,
        env=env,
    )
    proc._stderr_fh = stderr_fh  # type: ignore[attr-defined]
    proc._stderr_path = stderr_path  # type: ignore[attr-defined]
    return proc


def stop_llama_server(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
    except Exception:
        pass
    finally:
        fh = getattr(proc, "_stderr_fh", None)
        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass
            path = getattr(proc, "_stderr_path", None)
            if path and path.exists():
                log(f"[SERVER] Server log: {path}")


def wait_for_server(port: int, timeout: float = 120.0) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(0.5)
    return False


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------
def load_manifest(output_path: Path) -> tuple[set[str], dict[str, dict]]:
    manifest_path = output_path.with_suffix(".manifest.json")
    if not manifest_path.exists():
        return set(), {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    completed = set(data.get("completed_ids", []))
    failed: dict[str, dict] = {}
    for entry in data.get("failed", []):
        key = entry.get("id") or entry.get("seed_id")
        if not key:
            continue
        prev = failed.get(key, {})
        if entry.get("timestamp", "") >= prev.get("timestamp", ""):
            failed[key] = entry
    return completed, failed


def load_or_create_manifest(output_path: Path) -> tuple[set[str], dict[str, dict]]:
    """Load manifest, creating the output file first if it does not exist."""
    if not output_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")
    return load_manifest(output_path)


def save_manifest(output_path: Path, completed: set[str], failed: dict[str, dict]) -> None:
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    failed_list = sorted(
        failed.values(),
        key=lambda e: (e.get("timestamp", ""), e.get("id", e.get("seed_id", ""))),
    )
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "completed_ids": sorted(completed),
                "failed": failed_list,
                "updated_at": now_iso(),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def safe_filename(name: str, max_len: int = 100) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    return safe[:max_len]


def sha256_hash(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ngram_set(text: str, n: int = 8) -> set[str]:
    """Case-insensitive word n-gram set."""
    words = re.findall(r"\w+", text.lower())
    if len(words) < n:
        return set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def ngram_overlap(a: str, b: str, n: int = 8) -> float:
    """Fraction of n-grams in `a` that also appear in `b`."""
    ga, gb = ngram_set(a, n), ngram_set(b, n)
    if not ga:
        return 0.0
    return len(ga & gb) / len(ga)


def jaccard_similarity(a: str, b: str) -> float:
    """Simple word-overlap similarity for loop detection."""
    ta = set(re.findall(r"\w+", a.lower()))
    tb = set(re.findall(r"\w+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)
