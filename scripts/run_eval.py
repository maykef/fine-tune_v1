"""CLI: evaluate a trained checkpoint."""

import argparse

from finetune.eval import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the stage YAML config.")
    args = parser.parse_args()
    evaluate(args.config)


if __name__ == "__main__":
    main()
