"""Dataset loading and packing for the DAPT and SFT stages."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datasets import Dataset


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a stage YAML config into a dict."""
    raise NotImplementedError


def load_dapt_corpus(config: dict[str, Any]) -> Dataset:
    """Interleave the domain corpus with the replay mix for DAPT."""
    raise NotImplementedError


def load_sft_pairs(config: dict[str, Any]) -> Dataset:
    """Load the grounded QA chat JSONL for SFT."""
    raise NotImplementedError


def pack_sequences(dataset: Dataset, sequence_length: int) -> Dataset:
    """Concatenate and chunk tokenized text into fixed-length training sequences."""
    raise NotImplementedError


def prepare(stage: str, config_path: str | Path) -> Path:
    """Build processed training shards for a stage and return their directory."""
    raise NotImplementedError
