"""Dataset loading and packing for the DAPT and SFT stages.

The DAPT corpus is the gzipped-JSONL PMC shards (one paper per line, raw text in
`data.text_field`); DAPT replay is a general-text source (a HF hub id or a local
dir of same-format shards) interleaved at `data.replay_fraction`. The SFT set is
the teacher export chat JSONL ({"messages": [...]}); it carries no paper
provenance, so its held-out split is by record.

Tokenisation (text -> input_ids) happens in the load/prepare helpers where the
model tokenizer is available; `pack_sequences` only groups already-tokenised
`input_ids` into fixed blocks. That split keeps `pack_sequences` a pure,
CPU-testable function with the two-argument signature the scripts expect.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict, IterableDataset

_REQUIRED_SECTIONS = ("model", "data", "train", "output")
_TOKENIZE_PROC = min(16, os.cpu_count() or 1)
_PACK_SCHEMA = "1"  # bump when the packed-block dtypes/columns change (invalidates caches)


# ── config ────────────────────────────────────────────────────────────────────
def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a stage YAML config into a dict and validate its required sections."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"config {path}: top level must be a mapping")
    missing = [s for s in _REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ValueError(
            f"config {path}: missing required section(s): {', '.join(missing)}"
        )
    for section in _REQUIRED_SECTIONS:
        if not isinstance(cfg[section], dict):
            raise ValueError(f"config {path}: section '{section}' must be a mapping")
    return cfg


# ── tokenisation / packing ────────────────────────────────────────────────────
def _load_tokenizer(config: dict[str, Any]):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        config["model"]["name"], cache_dir=config["model"].get("cache_dir")
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _load_causal_lm(source: str, config: dict[str, Any], dtype: Any = None):
    """Load the text CausalLM backbone of the (multimodal) base model.

    Qwen3.5-9B-Base is a vision-language model; DAPT/SFT train only its text
    backbone, so we load via the text sub-config to get a flat Qwen3_5ForCausalLM.
    `source` is the base model name or a checkpoint dir. Passing a checkpoint dir
    here (used when resuming) reloads the trained weights through from_pretrained,
    which applies the key remapping the raw Trainer resume-load cannot — so the
    weights are correct before Trainer's resume load runs as a no-op over them.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    cache_dir = config["model"].get("cache_dir")
    text_cfg = AutoConfig.from_pretrained(source, cache_dir=cache_dir).get_text_config()
    return AutoModelForCausalLM.from_pretrained(
        source,
        config=text_cfg,
        cache_dir=cache_dir,
        dtype=torch.bfloat16 if dtype is None else dtype,
    )


def _verify_resume(model, checkpoint: str) -> None:
    """Confirm a resumed model actually holds the checkpoint's weights.

    from_pretrained remaps keys, so a silent fallback to base weights would still
    load cleanly but be wrong. Compare one tensor shared between the loaded model
    and the checkpoint shards; raise if it does not match so a bad resume fails
    loudly instead of wasting a run.
    """
    import glob

    import torch
    from safetensors import safe_open

    state = model.state_dict()
    for shard in sorted(glob.glob(f"{checkpoint}/*.safetensors")):
        with safe_open(shard, "pt") as handle:
            # Undo the save-time key nesting (model.language_model.* -> model.*).
            shared = [k for k in handle.keys() if k.replace("model.language_model.", "model.") in state]
            if not shared:
                continue
            # Prefer a substantive trained weight so the check would catch a
            # silent revert to base, not just an unchanged tied tensor.
            key = next((k for k in shared if "mlp.down_proj.weight" in k), shared[0])
            ref = handle.get_tensor(key)
            mkey = key.replace("model.language_model.", "model.")
            if not torch.equal(state[mkey].cpu().to(ref.dtype), ref):
                raise RuntimeError(
                    f"resume weight check failed for {mkey!r}: loaded model does "
                    f"not match {checkpoint} — weights were not restored"
                )
            return
    raise RuntimeError(f"resume weight check: no comparable tensor found under {checkpoint}")


def _tokenize(dataset, tokenizer, text_field: str):
    """Map raw-text rows to a single `input_ids` column, one EOS per document."""
    eos = tokenizer.eos_token_id

    def _tok(batch):
        ids = tokenizer(batch[text_field], add_special_tokens=False)["input_ids"]
        return {"input_ids": [seq + [eos] for seq in ids]}

    kwargs: dict[str, Any] = {"batched": True, "remove_columns": dataset.column_names}
    if hasattr(dataset, "num_rows"):  # map-style only; num_proc is invalid on iterables
        kwargs["num_proc"] = _TOKENIZE_PROC
    return dataset.map(_tok, **kwargs)


def pack_sequences(dataset: Dataset, sequence_length: int) -> Dataset:
    """Concatenate `input_ids` and chunk into fixed `sequence_length` blocks.

    Standard causal-LM packing: labels are a copy of input_ids and the trailing
    remainder shorter than one block is dropped. The input rows must already carry
    an `input_ids` column (see module docstring); works on map-style and iterable
    datasets alike.
    """
    from datasets import Features, Sequence, Value

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")

    def _group(batch):
        flat = list(chain.from_iterable(batch["input_ids"]))
        n = (len(flat) // sequence_length) * sequence_length
        blocks = [flat[i : i + sequence_length] for i in range(0, n, sequence_length)]
        return {
            "input_ids": blocks,
            "labels": [b[:] for b in blocks],
            "attention_mask": [[1] * sequence_length for _ in blocks],
        }

    # Pin dtypes so the map-style corpus cache and the streamed replay align
    # under interleave_datasets (which rejects int64-vs-int32 feature mismatch).
    features = Features({
        "input_ids": Sequence(Value("int32")),
        "labels": Sequence(Value("int32")),
        "attention_mask": Sequence(Value("int8")),
    })
    return dataset.map(
        _group, batched=True, remove_columns=dataset.column_names, features=features
    )


# ── DAPT corpus ───────────────────────────────────────────────────────────────
def _corpus_files(corpus_dir: str | Path) -> list[str]:
    files = sorted(glob.glob(str(Path(corpus_dir) / "*.jsonl.gz")))
    files += sorted(glob.glob(str(Path(corpus_dir) / "*.jsonl")))
    if not files:
        raise FileNotFoundError(f"no *.jsonl(.gz) shards under {corpus_dir}")
    return files


def _dapt_fingerprint(config: dict[str, Any], files: list[str]) -> str:
    data = config["data"]
    parts = [_PACK_SCHEMA, config["model"]["name"], data.get("text_field", "text")]
    parts += [str(data["sequence_length"]), str(data.get("heldout_fraction", 0.01))]
    parts += [str(config["train"].get("seed", 0)), str(data.get("max_docs") or "")]
    for f in files:
        st = os.stat(f)
        parts.append(f"{os.path.basename(f)}:{st.st_size}:{int(st.st_mtime)}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def _packed_domain(config: dict[str, Any], tokenizer):
    """Tokenise+pack the PMC corpus, cache it, and return (train, heldout) splits.

    The cache lives under `data.processed_dir`; it is rebuilt only when the
    fingerprint (corpus files, tokenizer, packing params) changes.
    """
    from datasets import load_dataset, load_from_disk

    data = config["data"]
    proc = Path(data["processed_dir"])
    train_dir, held_dir = proc / "packed_train", proc / "packed_heldout"
    meta_path = proc / "meta.json"
    files = _corpus_files(data["corpus_dir"])
    fp = _dapt_fingerprint(config, files)

    if train_dir.exists() and held_dir.exists() and meta_path.exists():
        if json.loads(meta_path.read_text()).get("fingerprint") == fp:
            return load_from_disk(str(train_dir)), load_from_disk(str(held_dir))

    text_field = data.get("text_field", "text")
    raw = load_dataset("json", data_files=files, split="train")
    if data.get("max_docs"):
        raw = raw.select(range(min(int(data["max_docs"]), raw.num_rows)))
    keep = [c for c in raw.column_names if c != text_field]
    if keep:
        raw = raw.remove_columns(keep)
    packed = pack_sequences(
        _tokenize(raw, tokenizer, text_field), int(data["sequence_length"])
    )
    split = packed.train_test_split(
        test_size=float(data.get("heldout_fraction", 0.01)),
        seed=int(config["train"].get("seed", 0)),
    )
    proc.mkdir(parents=True, exist_ok=True)
    split["train"].save_to_disk(str(train_dir))
    split["test"].save_to_disk(str(held_dir))
    meta_path.write_text(json.dumps({"fingerprint": fp, "blocks": packed.num_rows}))
    return split["train"], split["test"]


def _load_replay(config: dict[str, Any], tokenizer, sequence_length: int):
    """Load, tokenise and pack the DAPT replay source as an iterable dataset."""
    from datasets import Dataset as _Ds
    from datasets import load_dataset

    data = config["data"]
    src = data["replay_dir"]
    text_field = data.get("replay_text_field", "text")
    local = Path(str(src)).exists()
    if local:
        raw = load_dataset("json", data_files=_corpus_files(src), split="train")
    else:
        raw = load_dataset(
            str(src), name=data.get("replay_name") or None,
            split="train", streaming=True,
        )
    packed = pack_sequences(_tokenize(raw, tokenizer, text_field), sequence_length)
    if isinstance(packed, _Ds):  # local dir loads map-style; make it iterable to interleave
        packed = packed.to_iterable_dataset()
    return packed


def load_dapt_corpus(config: dict[str, Any]) -> IterableDataset:
    """Interleave the packed PMC corpus with the replay mix at `replay_fraction`.

    Returns an iterable dataset of packed blocks; the domain side is the cached
    train split (held-out slice excluded), the replay side is streamed.
    """
    from datasets import interleave_datasets

    data = config["data"]
    seq_len = int(data["sequence_length"])
    seed = int(config["train"].get("seed", 0))
    tokenizer = _load_tokenizer(config)
    domain_train, _ = _packed_domain(config, tokenizer)
    domain = domain_train.to_iterable_dataset()

    frac = float(data.get("replay_fraction", 0.0))
    if not data.get("replay_dir") or frac <= 0.0:
        return domain
    replay = _load_replay(config, tokenizer, seq_len)
    return interleave_datasets(
        [domain, replay],
        probabilities=[1.0 - frac, frac],
        seed=seed,
        stopping_strategy="first_exhausted",
    )


# ── SFT pairs ─────────────────────────────────────────────────────────────────
def _first_content(messages: list[dict], role: str) -> str | None:
    for m in messages:
        if isinstance(m, dict) and m.get("role") == role:
            c = m.get("content")
            if isinstance(c, str) and c.strip():
                return c
    return None


def load_sft_pairs(config: dict[str, Any]) -> DatasetDict:
    """Load the teacher export chat JSONL into deduplicated train/validation splits.

    Malformed lines (unparseable, or missing a non-empty user or assistant turn)
    are dropped and counted; exact-duplicate (user, assistant) pairs are removed.
    The split is by record (the export carries no paper provenance) and both
    splits are written under `data.processed_dir` as JSONL.
    """
    from datasets import Dataset, DatasetDict

    data = config["data"]
    train_file = Path(data["train_file"])
    if not train_file.exists():
        raise FileNotFoundError(f"SFT train_file not found: {train_file}")

    records: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    malformed = duplicates = 0
    with open(train_file) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                messages = rec["messages"]
            except (json.JSONDecodeError, KeyError, TypeError):
                malformed += 1
                continue
            user = _first_content(messages, "user")
            assistant = _first_content(messages, "assistant")
            if user is None or assistant is None:
                malformed += 1
                continue
            key = (user, assistant)
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            records.append(
                {"messages": [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": assistant},
                ]}
            )

    if not records:
        raise ValueError(f"no valid SFT pairs in {train_file}")

    ds = Dataset.from_list(records)
    val_fraction = float(data.get("val_fraction", 0.02))
    seed = int(config["train"].get("seed", 0))
    if val_fraction > 0.0 and ds.num_rows > 1:
        split = ds.train_test_split(test_size=val_fraction, seed=seed)
        splits = DatasetDict(train=split["train"], validation=split["test"])
    else:
        splits = DatasetDict(train=ds, validation=ds.select(range(0)))

    proc = Path(data["processed_dir"])
    proc.mkdir(parents=True, exist_ok=True)
    splits["train"].to_json(str(proc / "train.jsonl"), force_ascii=False)
    splits["validation"].to_json(str(proc / "validation.jsonl"), force_ascii=False)
    print(
        f"[sft] kept {ds.num_rows:,} pairs "
        f"(train {splits['train'].num_rows:,}, val {splits['validation'].num_rows:,}); "
        f"dropped {malformed:,} malformed, {duplicates:,} duplicate"
    )
    return splits


# ── prepare ───────────────────────────────────────────────────────────────────
def prepare(stage: str, config_path: str | Path) -> Path:
    """Build and cache the processed shards for a stage; return their directory."""
    config = load_config(config_path)
    proc = Path(config["data"]["processed_dir"])
    if stage == "dapt":
        _packed_domain(config, _load_tokenizer(config))
    elif stage == "sft":
        load_sft_pairs(config)
    else:
        raise ValueError(f"unknown stage: {stage!r} (expected 'dapt' or 'sft')")
    return proc
