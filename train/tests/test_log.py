"""Smoke test for the owner-provided logging helper (train/utils/log.py)."""
from pathlib import Path

from train.utils.log import log


def test_log_writes_file_and_header(tmp_path: Path):
    f = tmp_path / "common.log"
    log("hello", "world", filename=str(f))
    log("second line", level="DEBUG", filename=str(f))
    lines = f.read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("=== Log started at")
    assert "[INFO] hello world" in lines[1]
    assert "[DEBUG] second line" in lines[2]


def test_log_console_opt_in(tmp_path: Path, capsys):
    f = tmp_path / "quiet.log"
    log("file only", filename=str(f))
    assert capsys.readouterr().out == ""
    log("and console", filename=str(f), print_console=True)
    assert "and console" in capsys.readouterr().out
