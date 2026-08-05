"""Training entry points for the DAPT and SFT stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def train(stage: str, config_path: str | Path) -> Path:
    """Run a stage ("dapt" or "sft") and return the output checkpoint directory."""
    raise NotImplementedError


def run_dapt(config: dict[str, Any]) -> Path:
    """Full-parameter continued pretraining with replay interleaving."""
    raise NotImplementedError


def run_sft(config: dict[str, Any]) -> Path:
    """Supervised fine-tuning on grounded QA pairs via TRL SFTTrainer."""
    raise NotImplementedError
