"""Generic helper functions for the workflow."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

YEARS = [str(y) for y in range(2012, 2020)]
RESTART_MONTHS = ["01", "04", "07", "10"]
ENSEMBLES = ["01", "02", "03", "04", "05"]


def ensure_dir(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def now_stamp() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def expected_marker_file(step: str, out_dir: Path) -> Path:
    """Simple completion marker used for restart/skip support."""
    return out_dir / f".{step}.done"


def write_marker(step: str, out_dir: Path) -> Path:
    marker = expected_marker_file(step, out_dir)
    marker.write_text(f"completed_at_utc: {now_stamp()}\n", encoding="utf-8")
    return marker
