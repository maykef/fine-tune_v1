"""Teacher-generation pipeline: grounded synthetic QA for the SFT stage.

Six resumable stages (select, sections, generate, guard, judge, export) over an own
SQLite state DB; the source corpus DB is opened read-only. The guard and judge
stages are the validated anti-fabrication core. All parameters come from
configs/teacher.yaml.
"""
from .pipeline import (
    Config,
    check_pair,
    gen_prompt,
    judge_prompt,
    load_config,
    numbers_grounded,
    run,
    select_types,
    span_grounded,
    stage_export,
    stage_generate,
    stage_guard,
    stage_judge,
    stage_sections,
    stage_select,
    stage_stats,
)

__all__ = [
    "Config", "load_config", "run",
    "span_grounded", "numbers_grounded", "check_pair", "select_types",
    "gen_prompt", "judge_prompt",
    "stage_select", "stage_sections", "stage_generate",
    "stage_guard", "stage_judge", "stage_export", "stage_stats",
]
