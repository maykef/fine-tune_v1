# microscopy-finetune

Fine-tunes Qwen3.5-9B-Base on a corpus of ~324k open-access optical-microscopy papers (~3.4B tokens) using Transformers + TRL. Two stages: domain-adaptive pretraining (DAPT) then supervised fine-tuning (SFT) on grounded QA pairs.

## Setup

    mamba env create -f environment.yml
    mamba activate finetune

Non-mamba users: install the matching torch wheel first, then the package:

    pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
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

## Tested on

| Component | Value |
|---|---|
| CPU | AMD Ryzen Threadripper 7970X (32 cores, 64 threads) |
| GPU | NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 96 GB (97887 MiB) |
| RAM | 123 GiB |
| OS | Ubuntu 24.04.4 LTS |
| Driver | 595.71.05 |
| CUDA | 13.2 (driver) · torch wheel cu128 |

Reproduced by environment.yml (`mamba env create -f environment.yml`). Package versions:

    python 3.12.13
    torch 2.8.0+cu128
    transformers 5.5.0
    trl 0.24.0
    datasets 4.3.0
    accelerate 1.14.0
    bitsandbytes 0.49.2
