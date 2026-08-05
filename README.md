# microscopy-finetune

Fine-tunes Qwen3.5-9B-Base on a corpus of ~324k open-access optical-microscopy papers (~3.4B tokens) using Transformers + TRL. Two stages: domain-adaptive pretraining (DAPT) then supervised fine-tuning (SFT) on grounded QA pairs.

## Setup

    pip install -e .

## Usage

    python scripts/prepare_data.py --stage dapt --config configs/dapt.yaml
    python scripts/run_train.py --stage dapt --config configs/dapt.yaml
    python scripts/run_train.py --stage sft --config configs/sft.yaml
    python scripts/run_eval.py --config configs/sft.yaml

## Structure

    configs/  one YAML per stage (dapt, sft)
    data/     local staging (raw, processed); gitignored
    src/      the finetune package (data, train, eval)
    scripts/  thin CLI wrappers
    tests/    tests
