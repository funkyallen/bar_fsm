"""Plan, smoke-test, run, and inspect the matched BAR-FSM pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import socket
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from contextlib import closing, contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import getproxies, proxy_bypass, urlopen

from openai import APIConnectionError, APIStatusError, AuthenticationError, RateLimitError

from aroute.core.billing_meter import BillingMeter, BudgetExceededError
from aroute.core.config import PROJECT_ROOT, get_api_key, get_base_url, load_models_catalog
from aroute.core.immutable import audit_orchestrator_ast, sha256_file, verify_manifest
from aroute.core.model_pool import FakeModelPool, LiveModelPool
from aroute.core.research_telemetry import request_settings
from aroute.core.runner import run_episode
from aroute.orchestrator.router import load_router_params


ARMS = ("p0", "p1", "p2")
STATES = ("relaxed", "tight")
PILOT_DATA = PROJECT_ROOT / "data" / "research_pilot"
PILOT_RUNS = PROJECT_ROOT / "runs" / "research_pilot"
MANIFEST = "protocol.json"
DATABASE = "meter.sqlite"


def _json_digest(value: Any) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_pairs(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    tasks = _read_json(path)
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Pilot tasks must be a nonempty JSON list.")

    by_id = {}
    families: dict[str, dict[str, Any]] = defaultdict(dict)
    for task in tasks:
        task_id = task["task_id"]
        family = task["family_id"]
        state = task["state_label"]
        if state not in STATES or state in families[family] or task_id in by_id:
            raise ValueError(f"Invalid or duplicate state/task in family {family}: {task_id}")
        by_id[task_id] = task
        families[family][state] = task

    for family, states in families.items():
        if set(states) != set(STATES):
            raise ValueError(f"Family {family} must have both relaxed and tight tasks.")
        relaxed, tight = (states[state] for state in STATES)
        for field in ("prompt", "assertions", "domain", "difficulty", "template_id", "max_steps"):
            if relaxed[field] != tight[field]:
                raise ValueError(f"Family {family} changed the matched field {field}.")
        if relaxed["initial_state"] == tight["initial_state"]:
            raise ValueError(f"Family {family} must change the initial state.")
    return by_id, dict(families)


def _make_plan(
    families: dict[str, dict[str, Any]], *, seed: int, repeats: int, families_per_domain: int | None,
    arms: tuple[str, ...] = ARMS,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    domains: dict[str, list[str]] = defaultdict(list)
    for family, states in sorted(families.items()):
        domains[states["relaxed"]["domain"]].append(family)

    for group in domains.values():
        rng.shuffle(group)
        if families_per_domain is not None:
            del group[families_per_domain:]

    jobs: list[dict[str, Any]] = []
    for repeat in range(1, repeats + 1):
        slots = [group.copy() for _, group in sorted(domains.items())]
        for group in slots:
            rng.shuffle(group)
        while any(slots):
            rng.shuffle(slots)
            for group in slots:
                if not group:
                    continue
                family = group.pop()
                conditions = [(state, arm) for state in STATES for arm in arms]
                rng.shuffle(conditions)
                for state, arm in conditions:
                    task = families[family][state]
                    jobs.append({
                        "slot": len(jobs) + 1,
                        "family_id": family,
                        "template_id": task["template_id"],
                        "domain": task["domain"],
                        "task_id": task["task_id"],
                        "state": state,
                        "arm": arm,
                        "repeat": repeat,
                    })
    return jobs


def _read_database(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _read_runs(path: Path, fingerprint: str) -> dict[int, list[dict[str, Any]]]:
    if not path.exists():
        return {}
    runs: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with closing(_read_database(path)) as conn:
        rows = conn.execute(
            """
            SELECT r.run_id, r.metadata_json, e.status, e.passed, e.failure_tag, e.summary_json,
                   e.started_at, e.finished_at
            FROM runs r LEFT JOIN episodes e USING(run_id)
            ORDER BY r.run_id
            """
        )
        for row in rows:
            data = dict(row)
            metadata = json.loads(data.pop("metadata_json"))
            if metadata["protocol_fingerprint"] != fingerprint:
                raise ValueError("Pilot ledger contains an unrelated run.")
            runs[int(metadata["slot"])].append(data)
    return dict(runs)


@contextmanager
def _lock_directory(run_dir: Path) -> Iterator[None]:
    with (run_dir / "execution.lock").open("a+b") as lock:
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError(f"Pilot directory is already in use: {run_dir}") from exc
        yield


def _save_protocol(run_dir: Path, protocol: dict[str, Any], resume: bool) -> None:
    manifest = run_dir / MANIFEST
    if manifest.exists():
        if not resume:
            raise ValueError(f"{run_dir} already has a protocol. Use --resume to continue it.")
        if _read_json(manifest) != protocol:
            raise ValueError("Resume would change the frozen protocol. Use the original parameters and code.")
        return
    if resume:
        raise ValueError(f"No existing protocol to resume in {run_dir}.")
    if any(path.name != "execution.lock" for path in run_dir.iterdir()):
        raise ValueError(f"Refusing to reuse a non-pilot directory: {run_dir}")
    temporary = manifest.with_suffix(".tmp")
    temporary.write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(manifest)


def _make_protocol(args: argparse.Namespace, jobs: list[dict[str, Any]]) -> dict[str, Any]:
    budget = args.budget_cny
    if args.action == "run" and (not args.live or budget is None or not math.isfinite(budget) or budget <= 0):
        raise ValueError("Live execution requires --live and a positive finite --budget-cny.")
    if args.action == "smoke" and (args.live or budget is not None):
        raise ValueError("Offline smoke execution does not accept live credentials or budget flags.")

    router_params = load_router_params().model_copy(update={"max_proactive_think_steps": 1})
    catalog = load_models_catalog()
    provider = catalog.providers[args.provider]
    tiers = {tier: provider.tiers[tier].resolve_model_id() for tier in ("fast", "think")}
    if args.action == "run":
        errors = verify_manifest() + audit_orchestrator_ast()
        if errors:
            raise ValueError("Integrity checks failed:\n" + "\n".join(errors))
        if get_api_key(provider) is None:
            raise ValueError(f"No API key is configured for {args.provider}.")
        if budget > catalog.project_budget_cap_cny:
            raise ValueError("Pilot budget exceeds the configured project budget.")

    source_paths = sorted((PROJECT_ROOT / "src" / "aroute").rglob("*.py"))
    source_digest = _json_digest({
        source.relative_to(PROJECT_ROOT).as_posix(): sha256_file(source)
        for source in [*source_paths, *(PROJECT_ROOT / "scripts" / name for name in (
            "research_run_pilot.py", "research_analyze_paired.py", "research_prepare_decision_probes.py",
        ))]
    })
    experiment = {
        "schema": 2,
        "mode": "live" if args.action == "run" else "offline_smoke",
        "tasks_sha256": sha256_file(args.tasks),
        "models_sha256": sha256_file(PROJECT_ROOT / "configs" / "models.yaml"),
        "source_sha256": source_digest,
        "oracle_sha256": sha256_file(args.oracle) if args.action == "smoke" else None,
        "provider": args.provider,
        "endpoint_sha256": _json_digest(get_base_url(provider)) if args.action == "run" else None,
        "resolved_model_ids": tiers,
        "router_params": router_params.model_dump(),
        "orchestrator_guards": not args.without_guards,
        "seed": args.seed,
        "repeats": args.repeats,
        "live_budget_cny": budget,
        "jobs": jobs,
    }
    return {"fingerprint": _json_digest(experiment), **experiment}


def _connectivity_report(provider: str, base_url: str) -> dict[str, Any]:
    """Report an optional unauthenticated check without blocking SDK requests."""
    endpoint = urlsplit(base_url)
    proxy_configured = bool(getproxies().get(endpoint.scheme)) and not proxy_bypass(endpoint.hostname or "")
    report = {"provider": provider, "proxy_configured": proxy_configured}
    try:
        with urlopen(base_url.rstrip("/") + "/models", timeout=8) as response:
            return {**report, "status": "http_response", "http_status": response.status}
    except HTTPError as exc:
        status = "proxy_authentication_required" if exc.code == 407 else "http_response"
        return {**report, "status": status, "http_status": exc.code}
    except (URLError, OSError) as exc:
        reason = exc.reason if isinstance(exc, URLError) else exc
        codes = (getattr(reason, "errno", None), getattr(reason, "winerror", None))
        if isinstance(reason, PermissionError) or any(code in (1, 13, 10013) for code in codes):
            status = "local_permission_denied"
        elif isinstance(reason, socket.gaierror):
            status = "proxy_dns_failure" if proxy_configured else "endpoint_dns_failure"
        else:
            status = "proxy_or_upstream_transport_failure" if proxy_configured else "upstream_transport_failure"
        return {**report, "status": status, "error_type": type(reason).__name__, "errno": codes[0], "winerror": codes[1]}


def _canary(args: argparse.Namespace) -> dict[str, Any]:
    """One authenticated, metered Fast request, with no tool or retry."""
    catalog = load_models_catalog()
    if get_api_key(catalog.providers[args.provider]) is None:
        raise ValueError(f"No API key is configured for {args.provider}.")
    if not args.live or args.budget_cny is None or not 0 < args.budget_cny <= catalog.project_budget_cap_cny:
        raise ValueError("Canary requires --live and a positive --budget-cny within the project budget.")
    run_dir = args.run_dir or PILOT_RUNS / f"canary_{args.provider}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with _lock_directory(run_dir):
        path = run_dir / DATABASE
        if path.exists() or (run_dir / MANIFEST).exists():
            raise ValueError("Canary requires a fresh run directory to keep its billing separate from the pilot.")
        meter = BillingMeter(path, project_cap_cny=args.budget_cny, holidays=catalog.holidays_2026)
        meter.create_or_resume_run(
            run_id="canary", stage="research_canary", split="canary", policy="fast", provider=args.provider,
            round_cap_cny=args.budget_cny, stage_cap_cny=args.budget_cny,
            metadata={"request_settings": request_settings(catalog.providers[args.provider].tiers["fast"], max_tokens=1)},
        )
        pool = LiveModelPool(catalog, meter, timeout_sec=15, max_retries=0)
        try:
            response = pool.call_with_tools(
                run_id="canary", episode_id="canary", task_id="canary", step_idx=0,
                provider=args.provider, tier="fast", messages=[{"role": "user", "content": "Reply OK."}],
                max_tokens_override=1,
            )
        except Exception:
            meter.finish_run("canary", status="INTERRUPTED")
            raise
        meter.finish_run("canary")
        return {"mode": "live_canary", "provider": args.provider, "ledger": str(path),
                "model_requested": response.model_requested, "model_actual": response.model_actual,
                "completion_tokens": response.completion_tokens, "cost_cny": response.cost_cny}


def _execute(args: argparse.Namespace, tasks: dict[str, Any], jobs: list[dict[str, Any]]) -> dict[str, Any]:
    from aroute.orchestrator.research_router import ResearchStepRouter
    from aroute.orchestrator.router import RouterParams

    protocol = _make_protocol(args, jobs)
    catalog = load_models_catalog()
    run_dir = args.run_dir or PILOT_RUNS / protocol["mode"]
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / DATABASE
    cap = protocol["live_budget_cny"] if args.action == "run" else 1_000.0
    oracle: dict[str, Any] = {}
    if args.action == "smoke":
        oracle = _read_json(args.oracle)
        missing = {job["task_id"] for job in jobs} - oracle.keys()
        if missing:
            raise ValueError(f"Offline fixture is missing {len(missing)} planned task traces.")

    with _lock_directory(run_dir):
        _save_protocol(run_dir, protocol, args.resume)
        previous = _read_runs(path, protocol["fingerprint"])
        meter = BillingMeter(path, project_cap_cny=cap, holidays=catalog.holidays_2026)
        if args.action == "smoke":
            def replay(task_id: str, tier: str, step: int, messages: list, tools: list | None) -> dict:
                actions = oracle[task_id]
                if step < len(actions):
                    return {"tool_calls": [actions[step]]}
                return {"content": "Scripted offline trace complete."}

            pool = FakeModelPool(catalog, meter, replay)
        else:
            pool = LiveModelPool(catalog, meter, timeout_sec=60, max_retries=2)
        base_params = RouterParams.model_validate(protocol["router_params"])

        completed_now = 0
        for job in jobs:
            slot = job["slot"]
            prior = previous.get(slot, [])
            if any(row["status"] == "COMPLETED" for row in prior):
                continue

            run_id = f"pilot_{slot:04d}_{len(prior) + 1:03d}"
            meter.create_or_resume_run(
                run_id=run_id, stage="research_pilot", split="pilot", policy=job["arm"],
                provider=args.provider, round_cap_cny=cap, stage_cap_cny=cap,
                metadata={"protocol_fingerprint": protocol["fingerprint"], "slot": slot},
            )
            try:
                run_episode(
                    task=tasks[job["task_id"]], run_id=run_id, provider=args.provider,
                    router=ResearchStepRouter(job["arm"], base_params, use_guards=protocol["orchestrator_guards"]),
                    pool=pool, meter=meter,
                    pause_on_transport_error=args.action == "run",
                )
            except BaseException:
                meter.finish_run(run_id, status="INTERRUPTED")
                raise
            meter.finish_run(run_id)
            completed_now += 1
            if completed_now == args.max_new_episodes:
                break
        return _analyze(run_dir)


def _paired_inputs(completed: list[dict[str, Any]], state: str, repeats: int) -> dict[str, Any]:
    outcomes = {
        (row["family_id"], row["repeat"], row["arm"]): row["passed"]
        for row in completed
        if row["state"] == state and row["arm"] in ("p0", "p1") and row["failure"] != "API_ERROR"
    }
    by_family: dict[str, list[int]] = defaultdict(list)
    templates = {row["family_id"]: row["template_id"] for row in completed}
    for (family, repeat, arm), passed in outcomes.items():
        if arm == "p1" and (family, repeat, "p0") in outcomes:
            by_family[family].append(passed - outcomes[family, repeat, "p0"])

    full_families = {family: diffs for family, diffs in by_family.items() if len(diffs) == repeats}
    matched = Counter(diff for diffs in full_families.values() for diff in diffs)
    by_template: dict[str, list[float]] = defaultdict(list)
    for family, diffs in full_families.items():
        by_template[templates[family]].append(statistics.mean(diffs))
    n = sum(matched.values())
    return {
        "complete_families": len(full_families),
        "complete_template_clusters": len(by_template),
        "matched_repeats": n,
        "p1_only_success": matched[1],
        "p0_only_success": matched[-1],
        "paired_disagreement_rate": (matched[1] + matched[-1]) / n if n else None,
        "success_difference_p1_minus_p0": (matched[1] - matched[-1]) / n if n else None,
        "sd_family_success_difference": (
            statistics.stdev(statistics.mean(diffs) for diffs in full_families.values())
            if len(full_families) > 1 else None
        ),
        "sd_template_cluster_success_difference": (
            statistics.stdev(statistics.mean(diffs) for diffs in by_template.values())
            if len(by_template) > 1 else None
        ),
    }


def _analyze(run_dir: Path) -> dict[str, Any]:
    protocol = _read_json(run_dir / MANIFEST)
    if protocol["fingerprint"] != _json_digest({k: v for k, v in protocol.items() if k != "fingerprint"}):
        raise ValueError("Pilot protocol fingerprint verification failed.")
    path = run_dir / DATABASE
    runs = _read_runs(path, protocol["fingerprint"])
    completed = []
    charge_cny = 0.0
    reserved_cny = 0.0
    attempt_status: dict[str, int] = {}
    models: list[str] = []
    if path.exists():
        with closing(_read_database(path)) as conn:
            conn.execute("BEGIN")
            charge_cny, reserved_cny = conn.execute(
                """
                SELECT COALESCE(SUM(actual_cost_cny), 0),
                       COALESCE(SUM(CASE WHEN status IN ('PENDING', 'UNKNOWN_BILLING')
                                         THEN estimated_max_cost_cny ELSE 0 END), 0)
                FROM api_attempts
                """
            ).fetchone()
            attempt_status = dict(conn.execute("SELECT status, COUNT(*) FROM api_attempts GROUP BY status"))
            models = [
                row[0] for row in conn.execute(
                    "SELECT DISTINCT model_actual FROM api_attempts WHERE status = 'OK' ORDER BY model_actual"
                )
            ]
            for job in protocol["jobs"]:
                result = next((row for row in runs.get(job["slot"], []) if row["status"] == "COMPLETED"), None)
                if result is None:
                    continue
                cost, think_calls, api_latency_ms = conn.execute(
                    """
                    SELECT COALESCE(SUM(actual_cost_cny), 0),
                           COALESCE(SUM(CASE WHEN tier = 'think' AND status = 'OK' THEN 1 ELSE 0 END), 0),
                           COALESCE(SUM(latency_ms), 0)
                    FROM api_attempts WHERE run_id = ?
                    """,
                    (result["run_id"],),
                ).fetchone()
                routes = [
                    (row["step_idx"], json.loads(row["tool_args_json"]))
                    for row in conn.execute(
                        """
                        SELECT step_idx, tool_args_json FROM step_events
                        WHERE run_id = ? AND tool_name = '__ROUTE__' ORDER BY step_idx, event_id
                        """,
                        (result["run_id"],),
                    )
                ]
                reasons = Counter(route["reason"] for _, route in routes)
                first_proactive = next(
                    (step for step, route in routes if route["reason"] in ("proactive_state", "proactive_fixed")),
                    None,
                )
                elapsed = datetime.fromisoformat(result["finished_at"]) - datetime.fromisoformat(result["started_at"])
                summary = json.loads(result["summary_json"])
                completed.append({
                    **job, "passed": result["passed"], "failure": result["failure_tag"],
                    "cost_cny": cost, "think_calls": think_calls,
                    "elapsed_s": summary.get("episode_wall_clock_ms", elapsed.total_seconds() * 1_000) / 1_000,
                    "api_latency_s": api_latency_ms / 1_000, "route_reasons": reasons,
                    "first_proactive_step": first_proactive,
                    "episode_summary": summary,
                })

    planned = Counter((job["arm"], job["state"]) for job in protocol["jobs"])
    groups = {}
    for (arm, state), target in sorted(planned.items()):
        rows = [row for row in completed if row["arm"] == arm and row["state"] == state]
        n = len(rows)
        success = sum(row["passed"] for row in rows)
        cost = sum(row["cost_cny"] for row in rows)
        reasons: Counter[str] = Counter()
        write_failures = 0
        write_blocks = 0
        proactive_before_write = 0
        episodes_with_first_write = 0
        episodes_with_constrained_write = 0
        constrained_write_failures = 0
        proactive_steps_to_constrained_write: Counter[int] = Counter()
        for row in rows:
            reasons.update(row["route_reasons"])
            summary = row["episode_summary"]
            write_failures += summary.get("failed_write_attempts", 0)
            write_blocks += summary.get("blocked_write_attempts", 0)
            constrained_write_failures += summary.get("failed_constrained_write_attempts", 0)
            first_write = summary.get("first_write_step")
            first_proactive = row["first_proactive_step"]
            if first_write is not None:
                episodes_with_first_write += 1
                if first_proactive is not None and first_proactive <= first_write:
                    proactive_before_write += 1
            first_constrained_write = summary.get("first_constrained_write_step")
            if first_constrained_write is not None:
                episodes_with_constrained_write += 1
                if first_proactive is not None:
                    gap = first_constrained_write - first_proactive
                    proactive_steps_to_constrained_write[gap] += 1
        elapsed = sorted(row["elapsed_s"] for row in rows)
        api_elapsed = sorted(row["api_latency_s"] for row in rows)
        infrastructure_errors = sum(row["failure"] == "API_ERROR" for row in rows)
        evaluated = n - infrastructure_errors
        groups[f"{arm}:{state}"] = {
            "planned": target, "completed": n, "passed": success,
            "terminal_outcomes": evaluated, "infrastructure_api_errors": infrastructure_errors,
            "terminal_success_rate": success / evaluated if evaluated else None,
            "assignment_success_rate_completed": success / n if n else None,
            "mean_completed_episode_cost_cny": cost / n if n else None,
            "completed_episode_cost_cny_per_success": cost / success if success else None,
            "mean_think_calls": statistics.mean(row["think_calls"] for row in rows) if n else None,
            "mean_elapsed_s": statistics.mean(elapsed) if n else None,
            "median_elapsed_s": statistics.median(elapsed) if n else None,
            "p95_elapsed_s": elapsed[math.ceil(0.95 * n) - 1] if n else None,
            "p99_elapsed_s": elapsed[math.ceil(0.99 * n) - 1] if n else None,
            "max_elapsed_s": elapsed[-1] if n else None,
            "mean_api_latency_s": statistics.mean(api_elapsed) if n else None,
            "median_api_latency_s": statistics.median(api_elapsed) if n else None,
            "p95_api_latency_s": api_elapsed[math.ceil(0.95 * n) - 1] if n else None,
            "p99_api_latency_s": api_elapsed[math.ceil(0.99 * n) - 1] if n else None,
            "max_api_latency_s": api_elapsed[-1] if n else None,
            "route_reasons": dict(reasons),
            "episodes_with_first_write": episodes_with_first_write,
            "episodes_proactive_by_first_write": proactive_before_write,
            "episodes_with_first_constrained_write": episodes_with_constrained_write,
            "episodes_proactive_by_first_constrained_write": sum(
                count for gap, count in proactive_steps_to_constrained_write.items() if gap >= 0
            ),
            "proactive_steps_to_first_constrained_write": dict(sorted(proactive_steps_to_constrained_write.items())),
            "failed_constrained_write_attempts": constrained_write_failures,
            "failed_write_attempts": write_failures,
            "blocked_write_attempts": write_blocks,
            "terminal_failures": dict(Counter(
                row["failure"] or "unclassified" for row in rows
                if not row["passed"] and row["failure"] != "API_ERROR"
            )),
        }
    return {
        "mode": protocol["mode"], "provider": protocol["provider"],
        "protocol_fingerprint": protocol["fingerprint"], "ledger": str(path),
        "planned_episodes": len(protocol["jobs"]), "completed_episodes": len(completed),
        "ledger_cost_cny": round(charge_cny, 8),
        "recorded_live_api_cost_cny": round(charge_cny, 8) if protocol["mode"] == "live" else 0.0,
        "unresolved_reserved_cny": round(reserved_cny, 8),
        "api_attempts_by_status": attempt_status, "returned_model_ids": models,
        "cost_scope": (
            "Total ledger cost includes interrupted attempts; arm/state costs include each completed "
            "assignment's terminal run, including infrastructure errors. Unresolved billing reserves are separate."
        ),
        "planning_input_scope": "Complete families only. API_ERROR assignments excluded, repeats are correlated.",
        "restarted_or_incomplete_attempts": sum(len(attempts) for attempts in runs.values()) - len(completed),
        "by_arm_state": groups,
        "paired_planning_inputs": {
            state: _paired_inputs(completed, state, protocol["repeats"]) for state in STATES
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("plan", "smoke", "run", "analyze", "preflight", "canary"), default="plan")
    parser.add_argument("--tasks", type=Path, default=PILOT_DATA / "tasks.json")
    parser.add_argument("--oracle", type=Path, default=PILOT_DATA / "oracle_offline_only.json")
    parser.add_argument("--provider", choices=("deepseek", "dashscope"), default="deepseek")
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--families-per-domain", type=int)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--without-guards", action="store_true", help="Use only backend validation for matched P0/P1 runs.")
    parser.add_argument("--budget-cny", type=float)
    parser.add_argument("--max-new-episodes", type=int, help="Pause after this many new episodes; resume the same plan.")
    args = parser.parse_args()

    try:
        if args.max_new_episodes is not None and (
            args.action not in ("run", "smoke") or args.max_new_episodes < 1
        ):
            raise ValueError("--max-new-episodes requires run or smoke and a positive count.")
        if args.action == "analyze":
            result = _analyze(args.run_dir or PILOT_RUNS / "live")
        elif args.action == "preflight":
            result = _connectivity_report(args.provider, get_base_url(load_models_catalog().providers[args.provider]))
        elif args.action == "canary":
            result = _canary(args)
        else:
            if args.repeats < 1 or (args.families_per_domain is not None and args.families_per_domain < 1):
                raise ValueError("Repeats and families-per-domain must be positive.")
            tasks, families = _load_pairs(args.tasks)
            jobs = _make_plan(
                families, seed=args.seed, repeats=args.repeats, families_per_domain=args.families_per_domain,
                arms=("p0", "p1") if args.without_guards else ARMS,
            )
            if args.action == "plan":
                result = {
                    "planned_episodes": len(jobs),
                    "families": len({job["family_id"] for job in jobs}),
                    "templates": len({job["template_id"] for job in jobs}),
                    "tasks_sha256": sha256_file(args.tasks),
                    "schedule_sha256": _json_digest(jobs), "first_block": jobs[:6],
                }
            else:
                result = _execute(args, tasks, jobs)
        print(json.dumps(result, ensure_ascii=True, indent=2, allow_nan=False))
    except AuthenticationError:
        print("Pilot stopped: provider authentication failed.", file=sys.stderr)
        return 1
    except (APIConnectionError, RateLimitError, APIStatusError) as exc:
        print(f"Pilot paused at the current slot: {type(exc).__name__}. Resume retains earlier billing.", file=sys.stderr)
        return 1
    except (BudgetExceededError, OSError, ValueError, KeyError, sqlite3.Error) as exc:
        print(f"Pilot stopped: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
