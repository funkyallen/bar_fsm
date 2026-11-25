"""SQLite billing ledger, peak/off-peak pricing calculator, and 3-layer budget circuit-breaker."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from .config import ProviderConfig, TierConfig

CN_TZ = timezone(timedelta(hours=8))


class BudgetExceededError(RuntimeError):
    """Raised when a round, stage, or project budget limit would be exceeded."""


def now_cn() -> datetime:
    """Return current datetime in UTC+8 (Asia/Shanghai)."""
    return datetime.now(tz=CN_TZ)


def iso_now_cn() -> str:
    """Return ISO-8601 timestamp string in UTC+8."""
    return now_cn().isoformat(timespec="seconds")


def is_deepseek_peak(dt_cn: datetime, holidays: set[str] | None = None) -> bool:
    """Determine if timestamp falls in DeepSeek CN peak hours.

    Peak: Mon-Fri 09:00-12:00 and 14:00-18:00 (UTC+8), excluding public holidays.
    """
    local = dt_cn.astimezone(CN_TZ)
    if local.weekday() >= 5:
        return False
    if holidays and local.strftime("%Y-%m-%d") in holidays:
        return False
    minutes = local.hour * 60 + local.minute
    return (9 * 60 <= minutes < 12 * 60) or (14 * 60 <= minutes < 18 * 60)


def get_active_tariff_rates(
    provider_cfg: ProviderConfig,
    tier_cfg: TierConfig,
    *,
    dt_cn: datetime | None = None,
    holidays: set[str] | None = None,
    regime_override: str | None = None,
) -> tuple[float, float, float, str]:
    """Return active per-1M CNY tariff rates (input_hit, input_miss, output, period_label)."""
    if regime_override is not None:
        if regime_override not in ("peak", "offpeak"):
            raise ValueError(f"Unsupported regime_override '{regime_override}', expected 'peak' or 'offpeak'")
        peak = regime_override == "peak"
    elif provider_cfg.peak_schedule == "deepseek_cn_workday":
        peak = is_deepseek_peak(dt_cn or now_cn(), holidays)
    else:
        peak = False

    period = "peak" if peak else "offpeak"
    p = tier_cfg.pricing
    if peak:
        return p.peak_input_hit, p.peak_input_miss, p.peak_output, period
    return p.offpeak_input_hit, p.offpeak_input_miss, p.offpeak_output, period


def compute_cost_cny(
    provider_cfg: ProviderConfig,
    tier_cfg: TierConfig,
    prompt_tokens: int,
    cache_hit_tokens: int,
    completion_tokens: int,
    dt_cn: datetime | None = None,
    holidays: set[str] | None = None,
    regime_override: str | None = None,
) -> tuple[float, str]:
    """Compute exact request cost in CNY and return (cost_cny, pricing_period)."""
    r_hit, r_miss, r_out, period = get_active_tariff_rates(
        provider_cfg,
        tier_cfg,
        dt_cn=dt_cn,
        holidays=holidays,
        regime_override=regime_override,
    )
    hit = max(0, min(cache_hit_tokens, prompt_tokens))
    miss = max(0, prompt_tokens - hit)
    out = max(0, completion_tokens)
    cost = (hit * r_hit + miss * r_miss + out * r_out) / 1_000_000.0
    return round(cost, 8), period


def estimate_worst_case_cost_cny(
    tier_cfg: TierConfig,
    est_prompt_tokens: int,
    max_output_tokens: int,
) -> float:
    """Conservative worst-case cost estimate assuming peak uncached pricing."""
    p = tier_cfg.pricing
    cost = (est_prompt_tokens * p.peak_input_miss + max_output_tokens * p.peak_output) / 1_000_000.0
    return round(cost, 8)


class BillingMeter:
    """Transaction-safe SQLite ledger with 3-layer hard budget circuit-breakers."""

    def __init__(
        self,
        db_path: Path | str,
        *,
        project_cap_cny: float = 350.0,
        holidays: list[str] | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.project_cap_cny = float(project_cap_cny)
        self.holidays = set(holidays or [])
        self._budget_lock = threading.Lock()
        self._init_schema()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=5000;")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._tx() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    stage TEXT NOT NULL,
                    split TEXT NOT NULL,
                    policy TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    round_cap_cny REAL NOT NULL,
                    stage_cap_cny REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS episodes (
                    episode_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    difficulty TEXT NOT NULL,
                    status TEXT NOT NULL,
                    passed INTEGER,
                    failure_tag TEXT,
                    steps_taken INTEGER NOT NULL DEFAULT 0,
                    total_cost_cny REAL NOT NULL DEFAULT 0.0,
                    loop_abort INTEGER NOT NULL DEFAULT 0,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    summary_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_episodes_run_task ON episodes(run_id, task_id);

                CREATE TABLE IF NOT EXISTS api_attempts (
                    attempt_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    episode_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    step_idx INTEGER NOT NULL,
                    provider TEXT NOT NULL,
                    tier TEXT NOT NULL,
                    model_requested TEXT NOT NULL,
                    model_actual TEXT,
                    pricing_period TEXT,
                    status TEXT NOT NULL,
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    cache_hit_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
                    estimated_max_cost_cny REAL NOT NULL DEFAULT 0.0,
                    actual_cost_cny REAL NOT NULL DEFAULT 0.0,
                    latency_ms INTEGER NOT NULL DEFAULT 0,
                    request_hash TEXT NOT NULL,
                    response_json TEXT,
                    error_msg TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_attempts_run ON api_attempts(run_id);

                CREATE TABLE IF NOT EXISTS step_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    episode_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    step_idx INTEGER NOT NULL,
                    tier TEXT NOT NULL,
                    tool_name TEXT,
                    tool_args_json TEXT,
                    fsm_blocked INTEGER NOT NULL DEFAULT 0,
                    fsm_reason TEXT,
                    tool_ok INTEGER,
                    error_code TEXT,
                    state_changed INTEGER NOT NULL DEFAULT 0,
                    ledger_summary TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_events_episode ON step_events(episode_id, step_idx);
                """
            )

    def create_or_resume_run(
        self,
        *,
        run_id: str,
        stage: str,
        split: str,
        policy: str,
        provider: str,
        round_cap_cny: float,
        stage_cap_cny: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        with self._tx() as conn:
            row = conn.execute("SELECT run_id FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if row is None:
                conn.execute(
                    """
                    INSERT INTO runs (
                        run_id, stage, split, policy, provider,
                        round_cap_cny, stage_cap_cny, created_at, status, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'RUNNING', ?)
                    """,
                    (
                        run_id,
                        stage,
                        split,
                        policy,
                        provider,
                        float(round_cap_cny),
                        float(stage_cap_cny),
                        iso_now_cn(),
                        json.dumps(metadata or {}, ensure_ascii=False),
                    ),
                )

    def finish_run(self, run_id: str, status: str = "COMPLETED") -> None:
        with self._tx() as conn:
            conn.execute("UPDATE runs SET status = ? WHERE run_id = ?", (status, run_id))

    def get_completed_task_ids(self, run_id: str) -> set[str]:
        with self._tx() as conn:
            rows = conn.execute(
                "SELECT task_id FROM episodes WHERE run_id = ? AND status = 'COMPLETED'",
                (run_id,),
            ).fetchall()
            return {r["task_id"] for r in rows}

    def get_spent_and_reserved(
        self,
        conn: sqlite3.Connection,
        *,
        run_id: str,
        stage: str,
    ) -> tuple[float, float, float]:
        """Return (run_total, stage_total, project_total) including PENDING reserves."""
        expr = """
            COALESCE(SUM(
                CASE
                    WHEN a.status IN ('PENDING', 'UNKNOWN_BILLING') THEN a.estimated_max_cost_cny
                    ELSE a.actual_cost_cny
                END
            ), 0.0)
        """
        run_total = float(
            conn.execute(f"SELECT {expr} FROM api_attempts a WHERE a.run_id = ?", (run_id,)).fetchone()[0]
        )
        stage_total = float(
            conn.execute(
                f"""
                SELECT {expr}
                FROM api_attempts a
                JOIN runs r ON a.run_id = r.run_id
                WHERE r.stage = ?
                """,
                (stage,),
            ).fetchone()[0]
        )
        project_total = float(conn.execute(f"SELECT {expr} FROM api_attempts a").fetchone()[0])
        return run_total, stage_total, project_total

    def current_spend_summary(self, run_id: str, stage: str) -> dict[str, float]:
        with self._tx() as conn:
            run_total, stage_total, project_total = self.get_spent_and_reserved(
                conn, run_id=run_id, stage=stage
            )
        return {
            "run_cny": round(run_total, 6),
            "stage_cny": round(stage_total, 6),
            "project_cny": round(project_total, 6),
        }

    def begin_attempt(
        self,
        *,
        run_id: str,
        episode_id: str,
        task_id: str,
        step_idx: int,
        provider: str,
        tier: str,
        model_requested: str,
        estimated_max_cost_cny: float,
        request_hash: str,
    ) -> str:
        """Check 3-layer caps and write PENDING attempt in a short isolated transaction."""
        attempt_id = str(uuid.uuid4())
        now_str = iso_now_cn()
        with self._budget_lock:
            with self._tx() as conn:
                run_row = conn.execute(
                    "SELECT stage, round_cap_cny, stage_cap_cny FROM runs WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
                if run_row is None:
                    raise ValueError(f"Unknown run_id: {run_id}")

                stage = str(run_row["stage"])
                round_cap = float(run_row["round_cap_cny"])
                stage_cap = float(run_row["stage_cap_cny"])
                run_used, stage_used, project_used = self.get_spent_and_reserved(
                    conn, run_id=run_id, stage=stage
                )

                if run_used + estimated_max_cost_cny > round_cap + 1e-12:
                    raise BudgetExceededError(
                        f"Round budget cap exceeded: used={run_used:.4f}, est={estimated_max_cost_cny:.4f}, cap={round_cap:.4f}"
                    )
                if stage_used + estimated_max_cost_cny > stage_cap + 1e-12:
                    raise BudgetExceededError(
                        f"Stage '{stage}' budget cap exceeded: used={stage_used:.4f}, est={estimated_max_cost_cny:.4f}, cap={stage_cap:.4f}"
                    )
                if project_used + estimated_max_cost_cny > self.project_cap_cny + 1e-12:
                    raise BudgetExceededError(
                        f"Project budget cap exceeded: used={project_used:.4f}, est={estimated_max_cost_cny:.4f}, cap={self.project_cap_cny:.4f}"
                    )

                conn.execute(
                    """
                    INSERT INTO api_attempts (
                        attempt_id, run_id, episode_id, task_id, step_idx,
                        provider, tier, model_requested, status,
                        estimated_max_cost_cny, actual_cost_cny, request_hash,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?, 0.0, ?, ?, ?)
                    """,
                    (
                        attempt_id,
                        run_id,
                        episode_id,
                        task_id,
                        step_idx,
                        provider,
                        tier,
                        model_requested,
                        float(estimated_max_cost_cny),
                        request_hash,
                        now_str,
                        now_str,
                    ),
                )
        return attempt_id

    def finish_attempt_ok(
        self,
        *,
        attempt_id: str,
        model_actual: str,
        pricing_period: str,
        prompt_tokens: int,
        cache_hit_tokens: int,
        completion_tokens: int,
        reasoning_tokens: int,
        actual_cost_cny: float,
        latency_ms: int,
        response_payload: dict[str, Any],
    ) -> None:
        with self._tx() as conn:
            conn.execute(
                """
                UPDATE api_attempts
                SET status = 'OK',
                    model_actual = ?,
                    pricing_period = ?,
                    prompt_tokens = ?,
                    cache_hit_tokens = ?,
                    completion_tokens = ?,
                    reasoning_tokens = ?,
                    actual_cost_cny = ?,
                    latency_ms = ?,
                    response_json = ?,
                    updated_at = ?
                WHERE attempt_id = ?
                """,
                (
                    model_actual,
                    pricing_period,
                    int(prompt_tokens),
                    int(cache_hit_tokens),
                    int(completion_tokens),
                    int(reasoning_tokens),
                    float(actual_cost_cny),
                    int(latency_ms),
                    json.dumps(response_payload, ensure_ascii=False),
                    iso_now_cn(),
                    attempt_id,
                ),
            )

    def finish_attempt_error(
        self,
        *,
        attempt_id: str,
        error_msg: str,
        latency_ms: int,
        billing_unknown: bool = False,
    ) -> None:
        status = "UNKNOWN_BILLING" if billing_unknown else "ERROR"
        with self._tx() as conn:
            conn.execute(
                """
                UPDATE api_attempts
                SET status = ?,
                    error_msg = ?,
                    latency_ms = ?,
                    actual_cost_cny = 0.0,
                    updated_at = ?
                WHERE attempt_id = ?
                """,
                (status, error_msg[:2000], int(latency_ms), iso_now_cn(), attempt_id),
            )

    def start_episode(
        self,
        *,
        episode_id: str,
        run_id: str,
        task_id: str,
        domain: str,
        difficulty: str,
    ) -> None:
        with self._tx() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO episodes (
                    episode_id, run_id, task_id, domain, difficulty,
                    status, started_at, summary_json
                ) VALUES (?, ?, ?, ?, ?, 'RUNNING', ?, '{}')
                """,
                (episode_id, run_id, task_id, domain, difficulty, iso_now_cn()),
            )

    def finish_episode(
        self,
        *,
        episode_id: str,
        passed: bool,
        failure_tag: str | None,
        steps_taken: int,
        total_cost_cny: float,
        loop_abort: bool,
        summary: dict[str, Any],
    ) -> None:
        with self._tx() as conn:
            conn.execute(
                """
                UPDATE episodes
                SET status = 'COMPLETED',
                    passed = ?,
                    failure_tag = ?,
                    steps_taken = ?,
                    total_cost_cny = ?,
                    loop_abort = ?,
                    finished_at = ?,
                    summary_json = ?
                WHERE episode_id = ?
                """,
                (
                    1 if passed else 0,
                    failure_tag,
                    int(steps_taken),
                    float(total_cost_cny),
                    1 if loop_abort else 0,
                    iso_now_cn(),
                    json.dumps(summary, ensure_ascii=False),
                    episode_id,
                ),
            )

    def record_step_event(
        self,
        *,
        run_id: str,
        episode_id: str,
        task_id: str,
        step_idx: int,
        tier: str,
        tool_name: str | None,
        tool_args: dict[str, Any] | None,
        fsm_blocked: bool,
        fsm_reason: str | None,
        tool_ok: bool | None,
        error_code: str | None,
        state_changed: bool,
        ledger_summary: str,
    ) -> None:
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO step_events (
                    run_id, episode_id, task_id, step_idx, tier,
                    tool_name, tool_args_json, fsm_blocked, fsm_reason,
                    tool_ok, error_code, state_changed, ledger_summary, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    episode_id,
                    task_id,
                    int(step_idx),
                    tier,
                    tool_name,
                    json.dumps(tool_args, ensure_ascii=False) if tool_args is not None else None,
                    1 if fsm_blocked else 0,
                    fsm_reason,
                    None if tool_ok is None else (1 if tool_ok else 0),
                    error_code,
                    1 if state_changed else 0,
                    ledger_summary,
                    iso_now_cn(),
                ),
            )

    def summarize_run(self, run_id: str) -> dict[str, Any]:
        with self._tx() as conn:
            eps = conn.execute(
                "SELECT * FROM episodes WHERE run_id = ? AND status = 'COMPLETED'",
                (run_id,),
            ).fetchall()
            attempts = conn.execute(
                "SELECT status, tier, prompt_tokens, cache_hit_tokens, completion_tokens, reasoning_tokens, actual_cost_cny FROM api_attempts WHERE run_id = ?",
                (run_id,),
            ).fetchall()

        n = len(eps)
        passed = sum(1 for r in eps if r["passed"] == 1)
        loop_aborts = sum(1 for r in eps if r["loop_abort"] == 1)
        total_cost = sum(float(r["actual_cost_cny"]) for r in attempts if r["status"] == "OK")
        avg_cost = (total_cost / n) if n else 0.0
        avg_steps = (sum(int(r["steps_taken"]) for r in eps) / n) if n else 0.0
        think_calls = sum(1 for a in attempts if a["status"] == "OK" and a["tier"] in ("think", "pro"))
        ok_calls = sum(1 for a in attempts if a["status"] == "OK")

        failure_counts: dict[str, int] = {}
        for r in eps:
            if r["passed"] == 0 and r["failure_tag"]:
                tag = str(r["failure_tag"])
                failure_counts[tag] = failure_counts.get(tag, 0) + 1

        return {
            "run_id": run_id,
            "completed_episodes": n,
            "pass_rate": round(passed / n, 6) if n else 0.0,
            "passed_count": passed,
            "loop_abort_rate": round(loop_aborts / n, 6) if n else 0.0,
            "avg_steps": round(avg_steps, 3),
            "total_cost_cny": round(total_cost, 6),
            "avg_cost_cny": round(avg_cost, 6),
            "ok_api_calls": ok_calls,
            "think_call_ratio": round(think_calls / ok_calls, 4) if ok_calls else 0.0,
            "failure_counts": failure_counts,
        }
