"""CLI: run the teacher-generation pipeline (grounded SFT data)."""

import argparse

from finetune.teacher import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["select", "sections", "generate", "guard", "judge", "export", "all"],
        required=True)
    parser.add_argument("--config", required=True, help="Path to the teacher YAML config.")
    args = parser.parse_args()
    run(args.stage, args.config)


if __name__ == "__main__":
    main()
