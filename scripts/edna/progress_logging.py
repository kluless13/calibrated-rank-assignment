#!/usr/bin/env python3
"""Small timestamped progress logger for long-running research scripts."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


class ProgressLogger:
    """Write progress messages to stdout and an optional log file."""

    def __init__(self, log_file: Path | str | None = None) -> None:
        self.log_file = Path(log_file) if log_file else None
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def utc_now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def log(self, message: str) -> None:
        line = f"[{self.utc_now()}] {message}"
        print(line, flush=True)
        if self.log_file:
            with self.log_file.open("a") as handle:
                handle.write(line + "\n")

    def start(self, script_name: str) -> None:
        self.log(f"START {script_name}")

    def done(self, script_name: str) -> None:
        self.log(f"DONE {script_name}")


def default_log_path(root: Path, script_name: str) -> Path:
    return root / "results" / "paper1_phylo_calibrated_assignment" / "logs" / f"{script_name}.log"
