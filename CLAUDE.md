# Project rules

## Structure — non-negotiable
- All reusable logic lives in `src/finetune/`. Scripts in `scripts/` are thin CLI wrappers only (argparse + one function call). Never put logic in scripts.
- Before writing any new function, grep `src/finetune/` for an existing one. Extend or refactor existing code; do not duplicate.
- Never create new top-level directories or new scripts without asking first.
- One script per task. If a script needs a variant, add a CLI flag; do not create `train_v2.py`, `train_final.py`, `train_fixed.py`, etc.
- All hyperparameters and paths go in `configs/*.yaml` (one per stage), never hardcoded.
- Do not create "experiment", "utils2", "helpers", "misc", "temp" or "old" files or folders. Delete dead code instead of keeping copies.

## Workflow
- Prefer editing existing files over creating new ones. Creating a file is the exception and needs a stated reason.
- After any change, run `ruff check` and `pytest`.
- Do not add dependencies without asking.
- Do not commit unless asked.

## Documentation style
- README and docstrings: plain, factual, terse. Describe what things do and how to run them. Nothing else.
- Banned: marketing language, adjectives about the code ("robust", "powerful", "honest", "load-bearing", "battle-tested"), emoji, badges, "Why this project?" sections, roadmaps, contribution guides, licences not asked for.
- README sections are limited to: what it is, setup, usage, structure. Never add others.
- Do not write summaries of your work into the repo (no CHANGES.md, SUMMARY.md, NOTES.md).
