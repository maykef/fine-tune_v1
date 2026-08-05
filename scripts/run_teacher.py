"""CLI: run the teacher-generation pipeline (grounded SFT data).

--config supplies all values; the path/model flags below override individual
config entries when given (so any path can be set on the command line)."""

import argparse

from finetune.teacher import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--stage",
        choices=["select", "sections", "generate", "guard", "judge", "export", "all"],
        required=True)
    parser.add_argument("--config", required=True, help="Path to the teacher YAML config.")
    # Path / endpoint overrides — each replaces the matching config value when given.
    parser.add_argument("--store", help="Corpus tank root (source of truth).")
    parser.add_argument("--source-db", dest="source_db", help="Read-only corpus DB.")
    parser.add_argument("--state-db", dest="state_db", help="Pipeline state DB.")
    parser.add_argument("--export-file", dest="export_file", help="SFT export JSONL (the train_file).")
    parser.add_argument("--base-url", dest="base_url", help="Model endpoint base URL.")
    parser.add_argument("--model", help="Served model name.")
    parser.add_argument("--concurrency", type=int, help="Max in-flight requests to the model.")
    args = parser.parse_args()
    run(args.stage, args.config, overrides=vars(args))


if __name__ == "__main__":
    main()
