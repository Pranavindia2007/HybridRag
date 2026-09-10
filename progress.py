"""
progress.py

Small terminal progress helpers for ingestion. Kept dependency-free so progress
works in a plain Python environment.
"""

from __future__ import annotations

import sys
import time


class ProgressBar:
    def __init__(self, label: str, total: int, enabled: bool = True, width: int = 28):
        self.label = label
        self.total = max(0, total)
        self.enabled = enabled
        self.width = width
        self.current = 0
        self.started_at = time.monotonic()
        self._finished = False
        self._last_render_len = 0

    def advance(self, amount: int = 1, suffix: str = "") -> None:
        if self._finished:
            return
        self.current = min(self.total, self.current + amount)
        self._render(suffix)

    def finish(self, suffix: str = "done") -> None:
        if self._finished:
            return
        self.current = self.total
        self._render(suffix)
        if self.enabled and self.total:
            sys.stdout.write("\n")
            sys.stdout.flush()
        self._finished = True

    def _render(self, suffix: str = "") -> None:
        if not self.enabled or not self.total:
            return
        pct = self.current / self.total
        filled = int(self.width * pct)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.monotonic() - self.started_at
        suffix_text = f" | {suffix}" if suffix else ""
        line = (
            f"\r{self.label}: [{bar}] {self.current}/{self.total} "
            f"({pct:>5.1%}) {elapsed:>5.1f}s{suffix_text}"
        )
        padding = " " * max(0, self._last_render_len - len(line) + 1)
        sys.stdout.write(line + padding)
        self._last_render_len = len(line)
        sys.stdout.flush()


def progress_log(message: str, enabled: bool = True) -> None:
    if enabled:
        print(message, flush=True)
