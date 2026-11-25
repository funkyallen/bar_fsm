"""Deterministic 5-class failure taxonomy classifier for failed episodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

FAILURE_TAGS = (
    "SCHEMA_HALLUCINATION",
    "PRECONDITION_VIOLATION",
    "NO_PROGRESS_LOOP",
    "LONG_HORIZON_FORGETTING",
    "STEP_OR_BUDGET_EXHAUSTED",
)


@dataclass
class EpisodeTraceStats:
    """Observable execution statistics used for failure classification."""

    passed: bool
    steps_taken: int
    max_steps: int
    budget_exhausted: bool
    loop_aborted: bool
    schema_errors: int
    precondition_blocks: int
    duplicate_successful_actions: int
    total_tool_calls: int
    write_calls: int
    repeated_final_state_streak: int
    truncation_abort: bool = False


def classify_failure(stats: EpisodeTraceStats, verify_violations: list[str] | None = None) -> str | None:
    """Classify a failed episode into one of five mutually exclusive categories."""
    if stats.passed:
        return None

    if stats.truncation_abort:
        return "STEP_OR_BUDGET_EXHAUSTED"

    if stats.loop_aborted or stats.repeated_final_state_streak >= 3:
        return "NO_PROGRESS_LOOP"

    if stats.schema_errors >= 2 or (stats.total_tool_calls > 0 and stats.schema_errors / stats.total_tool_calls >= 0.4):
        return "SCHEMA_HALLUCINATION"

    if stats.precondition_blocks >= 2:
        return "PRECONDITION_VIOLATION"

    if stats.budget_exhausted or stats.steps_taken >= stats.max_steps:
        if stats.duplicate_successful_actions >= 2 or stats.steps_taken >= 10:
            return "LONG_HORIZON_FORGETTING"
        return "STEP_OR_BUDGET_EXHAUSTED"

    # Model stopped early with missing assertions or repeated forgotten steps
    if stats.duplicate_successful_actions >= 1 or (verify_violations and len(verify_violations) >= 2 and stats.steps_taken >= 6):
        return "LONG_HORIZON_FORGETTING"

    if stats.precondition_blocks >= 1:
        return "PRECONDITION_VIOLATION"

    return "LONG_HORIZON_FORGETTING" if stats.steps_taken >= 6 else "STEP_OR_BUDGET_EXHAUSTED"
