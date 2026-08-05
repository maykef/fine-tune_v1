"""Tiny progress bar + JSON snapshot, copied from the corpus tooling.

`Bar` renders a one-line bar to stdout (throttled, log-friendly) AND writes a
machine-readable snapshot to a JSON file, so a detached run's progress can be
reported on demand.
"""
from __future__ import annotations

import json
import sys
import time


def fmt_dur(s: float) -> str:
    s = int(max(0, s))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


class Bar:
    def __init__(self, label: str, total: int, json_path, width: int = 34,
                 min_interval: float = 2.0):
        self.label = label
        self.total = max(1, int(total))
        self.path = str(json_path)
        self.width = width
        self.min_interval = min_interval
        self.start = time.time()
        self.last = 0.0
        self.istty = sys.stdout.isatty()

    def update(self, done: int, extra: str = "") -> None:
        now = time.time()
        done_final = done >= self.total
        if not done_final and (now - self.last) < self.min_interval:
            return
        self.last = now
        elapsed = max(1e-6, now - self.start)
        rate = done / elapsed
        pct = 100.0 * done / self.total
        eta = (self.total - done) / rate if rate > 0 else 0
        filled = int(self.width * done / self.total)
        bar = "█" * filled + "░" * (self.width - filled)
        line = (f"{self.label} [{bar}] {pct:5.1f}%  {done:,}/{self.total:,}  "
                f"{rate:,.0f}/s  ETA {fmt_dur(eta)}")
        if extra:
            line += f"  {extra}"
        if self.istty:
            sys.stdout.write("\r" + line)
            sys.stdout.flush()
        else:
            print(line, flush=True)
        try:
            with open(self.path, "w") as fh:
                json.dump({"label": self.label, "done": int(done), "total": self.total,
                           "pct": pct, "rate": rate, "eta_s": eta, "elapsed_s": elapsed,
                           "extra": extra, "updated": now}, fh)
        except Exception:
            pass

    def close(self, done: int | None = None) -> None:
        self.update(self.total if done is None else done)
        if self.istty:
            sys.stdout.write("\n")
            sys.stdout.flush()
