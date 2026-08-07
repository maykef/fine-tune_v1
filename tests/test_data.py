"""CPU-only tests for the pure data path: config validation, SFT JSONL
validation/dedup/split, and packing block shapes. No GPU, network, or model
downloads. Run: pytest tests/test_data.py"""
import json

import pytest
import yaml
from datasets import Dataset

from finetune.data import load_config, load_sft_pairs, pack_sequences


# ── load_config ───────────────────────────────────────────────────────────────
def _write(path, cfg):
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


def _valid_cfg():
    return {"model": {}, "data": {}, "train": {}, "output": {}}


def test_load_config_ok(tmp_path):
    cfg = load_config(_write(tmp_path / "c.yaml", _valid_cfg()))
    assert set(cfg) == {"model", "data", "train", "output"}


def test_load_config_missing_section_names_it(tmp_path):
    cfg = _valid_cfg()
    del cfg["output"]
    with pytest.raises(ValueError, match="output"):
        load_config(_write(tmp_path / "c.yaml", cfg))


def test_load_config_section_not_mapping(tmp_path):
    cfg = _valid_cfg()
    cfg["train"] = [1, 2]
    with pytest.raises(ValueError, match="train"):
        load_config(_write(tmp_path / "c.yaml", cfg))


def test_load_config_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


# ── load_sft_pairs ────────────────────────────────────────────────────────────
def _pair(user, assistant):
    return {"messages": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]}


def _sft_config(tmp_path, lines, val_fraction=0.5):
    tmp_path.mkdir(parents=True, exist_ok=True)
    train_file = tmp_path / "export.jsonl"
    with open(train_file, "w") as fh:
        for line in lines:
            fh.write(line + "\n")
    return {
        "model": {}, "output": {}, "train": {"seed": 0},
        "data": {
            "train_file": str(train_file),
            "processed_dir": str(tmp_path / "proc"),
            "val_fraction": val_fraction,
        },
    }


def test_load_sft_pairs_dedup_and_malformed(tmp_path):
    lines = [
        json.dumps(_pair("q1", "a1")),
        json.dumps(_pair("q1", "a1")),          # exact duplicate -> dropped
        json.dumps(_pair("q2", "a2")),
        "{not json",                             # malformed -> dropped
        json.dumps({"messages": [{"role": "user", "content": "q3"}]}),  # no assistant
        json.dumps(_pair("q4", "   ")),          # empty assistant -> dropped
        json.dumps(_pair("q5", "a5")),
    ]
    splits = load_sft_pairs(_sft_config(tmp_path, lines))
    total = splits["train"].num_rows + splits["validation"].num_rows
    assert total == 3  # q1, q2, q5 kept
    users = set(splits["train"]["messages"][i][0]["content"]
                for i in range(splits["train"].num_rows))
    users |= set(splits["validation"]["messages"][i][0]["content"]
                 for i in range(splits["validation"].num_rows))
    assert users == {"q1", "q2", "q5"}


def test_load_sft_pairs_split_deterministic(tmp_path):
    lines = [json.dumps(_pair(f"q{i}", f"a{i}")) for i in range(20)]
    a = load_sft_pairs(_sft_config(tmp_path / "a", lines))
    b = load_sft_pairs(_sft_config(tmp_path / "b", lines))
    assert a["train"]["messages"] == b["train"]["messages"]
    assert a["validation"]["messages"] == b["validation"]["messages"]
    assert a["train"].num_rows + a["validation"].num_rows == 20


def test_load_sft_pairs_writes_splits(tmp_path):
    lines = [json.dumps(_pair(f"q{i}", f"a{i}")) for i in range(10)]
    cfg = _sft_config(tmp_path, lines)
    load_sft_pairs(cfg)
    proc = tmp_path / "proc"
    assert (proc / "train.jsonl").exists()
    assert (proc / "validation.jsonl").exists()


# ── pack_sequences ────────────────────────────────────────────────────────────
def test_pack_sequences_block_shapes_and_labels():
    # 25 tokens, block 4 -> 6 full blocks (24 tokens), remainder dropped.
    ds = Dataset.from_dict({"input_ids": [list(range(10)), list(range(10, 25))]})
    packed = pack_sequences(ds, sequence_length=4)
    assert packed.num_rows == 6
    for row in packed:
        assert len(row["input_ids"]) == 4
        assert row["labels"] == row["input_ids"]
        assert row["attention_mask"] == [1, 1, 1, 1]
    # concatenation order preserved, remainder (tokens 24) dropped
    flat = [t for row in packed for t in row["input_ids"]]
    assert flat == list(range(24))


def test_pack_sequences_rejects_bad_length():
    ds = Dataset.from_dict({"input_ids": [[1, 2, 3]]})
    with pytest.raises(ValueError):
        pack_sequences(ds, sequence_length=0)
