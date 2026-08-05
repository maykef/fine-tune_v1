"""CLI: build processed training shards for a stage."""

import argparse

from finetune.data import prepare


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["dapt", "sft"], required=True)
    parser.add_argument("--config", required=True, help="Path to the stage YAML config.")
    args = parser.parse_args()
    prepare(args.stage, args.config)


if __name__ == "__main__":
    main()
