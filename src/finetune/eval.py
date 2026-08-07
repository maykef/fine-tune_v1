"""Evaluation for trained checkpoints.

DAPT checkpoints are scored by held-out perplexity on the reserved corpus slice;
SFT checkpoints by held-out loss on the validation split plus a greedy grounded-QA
probe (exact-match and token-F1). The stage is inferred from the config: a
`data.corpus_dir` selects perplexity, a `data.train_file` selects the SFT probe.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from finetune.data import _load_tokenizer, load_config


def _resolve_checkpoint(config: dict[str, Any]) -> str:
    from transformers.trainer_utils import get_last_checkpoint

    output_dir = Path(config["output"]["dir"])
    if output_dir.exists():
        last = get_last_checkpoint(str(output_dir))
        if last:
            return last
        if any(output_dir.glob("*.safetensors")) or (output_dir / "config.json").exists():
            return str(output_dir)
    return config["model"]["name"]


def _load_model(config: dict[str, Any], checkpoint: str):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint, cache_dir=config["model"].get("cache_dir"), dtype=torch.bfloat16
    )
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    return model


def perplexity(config: dict[str, Any]) -> float:
    """Held-out perplexity of a DAPT checkpoint on the reserved corpus slice."""
    import torch
    from datasets import load_from_disk
    from torch.utils.data import DataLoader
    from transformers import default_data_collator

    held_dir = Path(config["data"]["processed_dir"]) / "packed_heldout"
    if not held_dir.exists():
        raise FileNotFoundError(
            f"held-out slice not found at {held_dir}; run prepare/train for DAPT first"
        )
    dataset = load_from_disk(str(held_dir))
    model = _load_model(config, _resolve_checkpoint(config))
    device = next(model.parameters()).device
    batch_size = int(config.get("eval", {}).get("batch_size", 4))
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=default_data_collator)

    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = input_ids.clone()
            out = model(input_ids=input_ids, labels=labels)
            n = int((labels[..., 1:] != -100).sum())  # shifted target tokens
            total_loss += float(out.loss) * n
            total_tokens += n
    return math.exp(total_loss / max(1, total_tokens))


def _normalize(text: str) -> list[str]:
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def _token_f1(pred: str, ref: str) -> float:
    p, r = _normalize(pred), _normalize(ref)
    if not p or not r:
        return float(p == r)
    common = Counter(p) & Counter(r)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(r)
    return 2 * precision * recall / (precision + recall)


def sft_eval(config: dict[str, Any]) -> dict[str, float]:
    """Held-out loss on the SFT validation split plus a greedy QA accuracy probe."""
    import torch
    from datasets import load_dataset

    val_file = Path(config["data"]["processed_dir"]) / "validation.jsonl"
    if not val_file.exists():
        raise FileNotFoundError(
            f"validation split not found at {val_file}; run prepare/train for SFT first"
        )
    dataset = load_dataset("json", data_files=str(val_file), split="train")
    tokenizer = _load_tokenizer(config)
    model = _load_model(config, _resolve_checkpoint(config))
    device = next(model.parameters()).device

    total_loss, total_tokens, exact, f1, n = 0.0, 0, 0, 0.0, 0
    eval_cfg = config.get("eval", {})
    max_probe = int(eval_cfg.get("num_qa_samples", 200))
    max_new = int(eval_cfg.get("max_new_tokens", 256))

    with torch.no_grad():
        for i, rec in enumerate(dataset):
            messages = rec["messages"]
            user = next(m["content"] for m in messages if m["role"] == "user")
            ref = next(m["content"] for m in messages if m["role"] == "assistant")

            # Held-out loss with the prompt tokens masked (assistant-only, as in training).
            full = tokenizer.apply_chat_template(
                messages, tokenize=True, return_tensors="pt"
            ).to(device)
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user}],
                tokenize=True, add_generation_prompt=True, return_tensors="pt",
            ).to(device)
            labels = full.clone()
            labels[:, : prompt.shape[1]] = -100
            out = model(input_ids=full, labels=labels)
            tokens = int((labels[..., 1:] != -100).sum())
            if tokens:
                total_loss += float(out.loss) * tokens
                total_tokens += tokens

            if i < max_probe:  # deterministic greedy generation probe
                gen = model.generate(
                    prompt, max_new_tokens=max_new, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
                answer = tokenizer.decode(
                    gen[0, prompt.shape[1] :], skip_special_tokens=True
                ).strip()
                exact += int(_normalize(answer) == _normalize(ref))
                f1 += _token_f1(answer, ref)
                n += 1

    metrics = {
        "sft_val_loss": total_loss / max(1, total_tokens),
        "sft_val_ppl": math.exp(total_loss / max(1, total_tokens)),
        "qa_exact_match": exact / max(1, n),
        "qa_token_f1": f1 / max(1, n),
        "qa_n": float(n),
    }
    return metrics


def evaluate(config_path: str | Path) -> dict[str, float]:
    """Evaluate the configured checkpoint and return metric name to value."""
    config = load_config(config_path)
    data = config["data"]
    if "train_file" in data:
        metrics = sft_eval(config)
    elif "corpus_dir" in data:
        metrics = {"perplexity": perplexity(config)}
    else:
        raise ValueError(
            "cannot infer stage: config.data has neither 'train_file' nor 'corpus_dir'"
        )
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    return metrics
