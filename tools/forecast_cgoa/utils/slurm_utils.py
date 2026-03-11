"""Slurm helpers for local scripting."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional


def submit_slurm(script: Path, extra_args: Optional[list[str]] = None) -> str:
    cmd = ["sbatch"]
    if extra_args:
        cmd.extend(extra_args)
    cmd.append(str(script))
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip()
