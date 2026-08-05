"""Evaluation for trained checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def evaluate(config_path: str | Path) -> dict[str, float]:
    """Evaluate a checkpoint and return metric name to value."""
    raise NotImplementedError


def perplexity(config: dict[str, Any]) -> float:
    """Held-out perplexity for a DAPT checkpoint."""
    raise NotImplementedError


def sft_eval(config: dict[str, Any]) -> dict[str, float]:
    """Held-out loss and QA accuracy for an SFT checkpoint."""
    raise NotImplementedError
