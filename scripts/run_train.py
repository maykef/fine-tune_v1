"""CLI: run a training stage (DAPT or SFT)."""

import argparse

from finetune.train import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["dapt", "sft"], required=True)
    parser.add_argument("--config", required=True, help="Path to the stage YAML config.")
    args = parser.parse_args()
    train(args.stage, args.config)


if __name__ == "__main__":
    main()
