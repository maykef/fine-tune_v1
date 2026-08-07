"""Training entry points for the DAPT and SFT stages.

DAPT is full-parameter continued pretraining on the interleaved (corpus + replay)
packed stream via the HF `Trainer`; SFT is TRL `SFTTrainer` on the teacher chat
pairs with assistant-only loss. Both are single-GPU, bf16, 8-bit AdamW, and resume
from the latest checkpoint under `output.dir` if one is present. Every
hyperparameter comes from the stage YAML.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from finetune.data import (
    _load_causal_lm,
    _load_tokenizer,
    _packed_domain,
    _verify_resume,
    load_dapt_corpus,
    load_sft_pairs,
)


def _last_checkpoint(output_dir: Path) -> str | None:
    from transformers.trainer_utils import get_last_checkpoint

    if output_dir.exists():
        return get_last_checkpoint(str(output_dir))
    return None


def train(stage: str, config_path: str | Path) -> Path:
    """Run a stage ("dapt" or "sft") and return the output checkpoint directory."""
    from finetune.data import load_config

    config = load_config(config_path)
    if stage == "dapt":
        out = run_dapt(config)
    elif stage == "sft":
        out = run_sft(config)
    else:
        raise ValueError(f"unknown stage: {stage!r} (expected 'dapt' or 'sft')")
    print(f"[{stage}] output dir: {out}")
    return out


def run_dapt(config: dict[str, Any]) -> Path:
    """Full-parameter continued pretraining with replay interleaving."""
    import transformers
    from transformers import Trainer, TrainingArguments, default_data_collator

    train_cfg, data_cfg = config["train"], config["data"]
    seed = int(train_cfg.get("seed", 0))
    transformers.set_seed(seed)

    tokenizer = _load_tokenizer(config)
    domain_train, _ = _packed_domain(config, tokenizer)  # builds/loads the cache
    dataset = load_dapt_corpus(config)

    # An interleaved stream is iterable, so Trainer needs an explicit step budget.
    # Base it on the finite domain block count, inflated for the replay share.
    per_device = int(train_cfg.get("per_device_batch_size", 1))
    grad_accum = int(train_cfg.get("gradient_accumulation_steps", 1))
    global_batch = per_device * grad_accum
    replay_fraction = float(data_cfg.get("replay_fraction", 0.0))
    if not data_cfg.get("replay_dir"):
        replay_fraction = 0.0
    epochs = float(train_cfg.get("epochs", 1))
    domain_steps = math.ceil(domain_train.num_rows / global_batch)
    total_steps = math.ceil(domain_steps * epochs / max(1e-9, 1.0 - replay_fraction))
    max_steps = int(train_cfg["max_steps"]) if train_cfg.get("max_steps") else total_steps

    output_dir = Path(config["output"]["dir"])
    resume = _last_checkpoint(output_dir)
    # On resume, load from the checkpoint so trained weights are restored via
    # from_pretrained's key remapping (Trainer's later raw load is then a no-op).
    model = _load_causal_lm(resume or config["model"]["name"], config)
    if resume:
        _verify_resume(model, resume)
    model.config.use_cache = False

    args = TrainingArguments(
        output_dir=str(output_dir),
        max_steps=max_steps,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=grad_accum,
        learning_rate=float(train_cfg["learning_rate"]),
        lr_scheduler_type=train_cfg.get("lr_scheduler", "cosine"),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.0)),
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        bf16=True,
        optim=_optim(train_cfg.get("optimizer", "adamw_bnb_8bit")),
        logging_steps=int(train_cfg.get("logging_steps", 10)),
        save_steps=int(train_cfg.get("save_steps", 500)),
        save_strategy="steps",
        seed=seed,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=default_data_collator,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return output_dir


def run_sft(config: dict[str, Any]) -> Path:
    """Supervised fine-tuning on grounded QA pairs via TRL SFTTrainer."""
    import transformers
    from trl import SFTConfig, SFTTrainer

    train_cfg, data_cfg = config["train"], config["data"]
    seed = int(train_cfg.get("seed", 0))
    transformers.set_seed(seed)

    splits = load_sft_pairs(config)
    train_ds = _mix_instruction_replay(config, splits["train"])
    eval_ds = splits["validation"] if splits["validation"].num_rows else None

    output_dir = Path(config["output"]["dir"])
    resume = _last_checkpoint(output_dir)
    tokenizer = _load_tokenizer(config)
    model = _load_causal_lm(resume or config["model"]["name"], config)
    if resume:
        _verify_resume(model, resume)
    max_steps = int(train_cfg["max_steps"]) if train_cfg.get("max_steps") else -1
    args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=float(train_cfg.get("epochs", 3)),
        max_steps=max_steps,
        per_device_train_batch_size=int(train_cfg.get("per_device_batch_size", 1)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 1)),
        learning_rate=float(train_cfg["learning_rate"]),
        lr_scheduler_type=train_cfg.get("lr_scheduler", "cosine"),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.0)),
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        bf16=True,
        optim=_optim(train_cfg.get("optimizer", "adamw_bnb_8bit")),
        max_length=int(data_cfg["sequence_length"]),
        packing=False,
        assistant_only_loss=True,  # completion-only loss on assistant tokens
        logging_steps=int(train_cfg.get("logging_steps", 10)),
        save_steps=int(train_cfg.get("save_steps", 500)),
        save_strategy="steps",
        eval_strategy="epoch" if eval_ds is not None else "no",
        per_device_eval_batch_size=int(train_cfg.get("per_device_batch_size", 1)),
        seed=seed,
        report_to=[],
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(output_dir))
    return output_dir


def _optim(name: str) -> str:
    """Map the config optimizer alias to a transformers `optim` value."""
    return {"adamw_8bit": "adamw_bnb_8bit"}.get(name, name)


def _mix_instruction_replay(config: dict[str, Any], train_ds):
    """Interleave a general instruction dataset into the SFT train split.

    Controlled by `data.instruction_replay_dir` (a HF hub id or a local chat
    JSONL) and `data.instruction_replay_fraction`. An unset dir or a
    non-positive fraction means no replay, and the train split is returned as is.
    """
    from datasets import interleave_datasets, load_dataset

    data = config["data"]
    src = data.get("instruction_replay_dir")
    fraction = float(data.get("instruction_replay_fraction", 0.0))
    if not src or fraction <= 0.0:
        return train_ds

    if Path(str(src)).exists():
        replay = load_dataset("json", data_files=str(src), split="train")
    else:
        replay = load_dataset(
            str(src), name=data.get("instruction_replay_name") or None, split="train"
        )
    replay = replay.select_columns(["messages"])
    return interleave_datasets(
        [train_ds, replay],
        probabilities=[1.0 - fraction, fraction],
        seed=int(config["train"].get("seed", 0)),
        stopping_strategy="first_exhausted",
    )
