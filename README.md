# microscopy-finetune

Fine-tunes Qwen3.5-9B-Base on ~324k open-access optical-microscopy papers (~3.4B tokens) using Transformers + TRL. Two training stages: domain-adaptive pretraining (DAPT) on the corpus interleaved with a general-text replay set (FineWeb-Edu) to limit forgetting, then supervised fine-tuning (SFT) on grounded QA pairs. The SFT pairs are produced in-repo by a teacher pipeline: a teacher model reads each paper and writes QA pairs, and any pair whose supporting span or numbers are not verbatim in the source is dropped, so fabricated facts do not reach the student.

## Setup

    mamba env create -f environment.yml
    mamba activate finetune
    cp configs/teacher.example.yaml configs/teacher.yaml   # then set your paths (same for dapt, sft)

## Usage

DAPT, then SFT on the teacher-generated pairs:

    python scripts/prepare_data.py --stage dapt --config configs/dapt.yaml
    python scripts/run_train.py --stage dapt --config configs/dapt.yaml
    python scripts/run_train.py --stage sft --config configs/sft.yaml
    python scripts/run_eval.py --config configs/sft.yaml

Teacher pipeline — six resumable stages over one state DB (the corpus DB is opened read-only). `export` writes the SFT `train_file` that `configs/sft.yaml` reads:

    select    stratified sample of papers           -> sft_paper
    sections  parse Methods+Results into a context  -> paper_ctx
    generate  model writes grounded QA pairs         -> gen     (needs the model)
    guard     drop pairs not grounded in the source  -> pair
    judge     second adversarial model pass          -> pair    (needs the model)
    export    kept+judged pairs -> chat JSONL train_file

Run the whole pipeline, a single stage, or a status report:

    python scripts/run_teacher.py --stage all --config configs/teacher.yaml
    python scripts/run_teacher.py --stage generate --config configs/teacher.yaml
    python scripts/run_teacher.py --stage stats --config configs/teacher.yaml

`--limit N` bounds how many papers a run processes (accumulates into the state DB); paths and the model endpoint can be overridden on the CLI:

    python scripts/run_teacher.py --stage generate --config configs/teacher.yaml --limit 100
    python scripts/run_teacher.py --stage generate --config configs/teacher.yaml \
      --source-db DB --state-db DB --export-file FILE --base-url URL --model NAME --limit 100

Paced runs — one timed GPU window per invocation, auto-picking the next stage and resuming:

    ./scripts/run_daily.sh          # HOURS=6 ./scripts/run_daily.sh for a longer window

`generate` and `judge` need the teacher model serving at `model.base_url`.

## Structure

    configs/  one YAML per stage (dapt, sft, teacher); *.example.yaml tracked, real configs gitignored
    data/     local staging (raw, processed); gitignored
    src/      the finetune package (data, train, eval, teacher/)
    scripts/  thin CLI wrappers (prepare_data, run_train, run_eval, run_teacher) + run_daily.sh
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
    flash-linear-attention 0.5.2
