# microscopy-finetune

Fine-tunes Qwen3.5-9B-Base on a corpus of ~324k open-access optical-microscopy papers (~3.4B tokens) using Transformers + TRL. Two stages: domain-adaptive pretraining (DAPT) then supervised fine-tuning (SFT) on grounded QA pairs generated in-repo by a teacher pipeline (generate → guard → judge → export).

## Setup

    mamba env create -f environment.yml
    mamba activate finetune
    cp configs/teacher.example.yaml configs/teacher.yaml   # then set your paths (same for dapt, sft)

## Usage

    python scripts/prepare_data.py --stage dapt --config configs/dapt.yaml
    python scripts/run_train.py --stage dapt --config configs/dapt.yaml
    python scripts/run_teacher.py --stage all --config configs/teacher.yaml
    python scripts/run_train.py --stage sft --config configs/sft.yaml
    python scripts/run_eval.py --config configs/sft.yaml

Teacher-gen stages (select, sections, generate, guard, judge, export) are resumable and can be run individually with --stage.

## Structure

    configs/  one YAML per stage (dapt, sft, teacher)
    data/     local staging (raw, processed); gitignored
    src/      the finetune package (data, train, eval)
    scripts/  thin CLI wrappers, incl. run_teacher.py and run_daily.sh
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
