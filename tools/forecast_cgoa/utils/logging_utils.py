"""Logging and command execution helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Sequence

from .helpers import ensure_dir, now_stamp


def run_command(command: Sequence[str], cwd: Path, log_file: Path) -> None:
    ensure_dir(log_file.parent)
    with log_file.open("a", encoding="utf-8") as log:
        log.write(f"\n[{now_stamp()}] START\n")
        log.write(f"cwd: {cwd}\n")
        log.write(f"command: {' '.join(command)}\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=str(cwd),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log.write(f"[{now_stamp()}] END return_code={result.returncode}\n")

    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}")
