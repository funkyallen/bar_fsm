"""Research telemetry using the existing model pool and immutable billing schema."""

from __future__ import annotations

import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any

from .config import TierConfig


def request_settings(
    tier: TierConfig, *, simulated: bool = False, max_tokens: int | None = None,
) -> dict[str, Any]:
    """Record requested settings. Null means the setting was left to the provider."""
    extra = tier.extra_body
    return {
        "model_requested": tier.model_id if simulated else tier.resolve_model_id(),
        "model_version_resolution": "offline_simulated" if simulated else "rolling_alias_or_provider_id",
        "catalog_verified_snapshot_id": tier.verified_snapshot_id,
        "temperature": extra.get("temperature", tier.temperature),
        "top_p": extra.get("top_p"),
        "max_tokens": extra.get("max_tokens", max_tokens if max_tokens is not None else tier.max_tokens),
        "thinking": extra.get("thinking", extra.get("enable_thinking")),
        "thinking_budget": extra.get("thinking_budget"),
        "reasoning_effort": extra.get("reasoning_effort", tier.reasoning_effort),
    }


def episode_timing(db_path: Path, episode_id: str, started: float) -> dict[str, int]:
    """Use a monotonic wall clock and sum all recorded API attempts, including failed retries."""
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    with closing(sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)) as conn:
        api_ms = conn.execute(
            "SELECT COALESCE(SUM(latency_ms), 0) FROM api_attempts WHERE episode_id = ?", (episode_id,),
        ).fetchone()[0]
    return {"episode_wall_clock_ms": elapsed_ms, "sum_api_latency_ms": api_ms}
