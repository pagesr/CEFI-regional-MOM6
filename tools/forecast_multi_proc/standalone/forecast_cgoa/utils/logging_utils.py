"""Logging and command execution helpers."""

from __future__ import annotations

import os
import subprocess
import sys
from shutil import which
from pathlib import Path
from typing import Sequence


from .helpers import ensure_dir, now_stamp




def _absolutize_config_args(command: Sequence[str]) -> list[str]:
    cmd = list(command)
    for i, token in enumerate(cmd[:-1]):
        if token in {"--config", "--config_file"}:
            cmd[i + 1] = str(Path(cmd[i + 1]).expanduser().resolve())
    return cmd


def _resolve_python_executable(command: Sequence[str]) -> list[str]:
    cmd = list(command)
    if not cmd:
        return cmd
    if cmd[0] == "python" and which("python") is None and sys.executable:
        cmd[0] = sys.executable
    return cmd


def _force_unbuffered_python(command: Sequence[str]) -> list[str]:
    """
    Ensure python child processes flush output promptly into stage logs.
    """
    cmd = list(command)
    if not cmd:
        return cmd
    exe = Path(cmd[0]).name
    if "python" in exe and "-u" not in cmd[1:3]:
        cmd.insert(1, "-u")
    return cmd


def run_command(command: Sequence[str], cwd: Path, log_file: Path) -> None:
    command = _force_unbuffered_python(_resolve_python_executable(_absolutize_config_args(command)))
    ensure_dir(log_file.parent)
    with log_file.open("a", encoding="utf-8") as log:
        log.write(f"\n[{now_stamp()}] START\n")
        log.write(f"cwd: {cwd}\n")
        log.write(f"command: {' '.join(command)}\n")
        log.flush()
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        result = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log.write(f"[{now_stamp()}] END return_code={result.returncode}\n")

    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}")
