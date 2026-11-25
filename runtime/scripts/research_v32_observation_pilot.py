"""Freeze, execute and audit the paired v3.2 observation pilot."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from decimal import Decimal
from pathlib import Path

from openai import APIError

from aroute.core.billing_meter import BillingMeter, BudgetExceededError
from aroute.core.config import PROJECT_ROOT, get_api_key, get_base_url, load_models_catalog
from aroute.core.immutable import audit_orchestrator_ast, sha256_file, verify_manifest
from aroute.core.model_pool import FakeModelPool, LiveModelPool
from aroute.core.runner import run_episode
from aroute.envs import create_env
from aroute.orchestrator.research_router import ResearchStepRouter
from aroute.orchestrator.router import load_router_params

from .research_run_pilot import DATABASE, _json_digest, _read_database, _read_json
from .research_v31_rule_timing_pilot import PROVIDER, public_rule
from .research_v32_candidates import prepare
from .research_v32_replay import RESTRICTED, control_for, read_events, replay_actions


ARMS = {"natural_R": False, "supplement_R": True}
CAP, MAX_STEPS, TIMEOUT = 1.2, 30, 120


def _write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _sources():
    paths = sorted((PROJECT_ROOT / "src/aroute").rglob("*.py"))
    paths += [PROJECT_ROOT / "configs/models.yaml"]
    paths += [PROJECT_ROOT / "scripts" / f"{s}.py" for s in (
        "research_run_pilot", "research_v31_timing_candidates", "research_v31_control_checks",
        "research_v31_accounting_fork", "research_v31_rule_fork", "research_v31_rule_timing_pilot",
        "research_v32_replay", "research_v32_candidates", "research_v32_observation_pilot")]
    return {p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p) for p in paths}


def freeze(output):
    tasks, witnesses = prepare(MAX_STEPS)
    order = ("healthy_upstream_fanout-tight", "discount_fee_rounding-relaxed",
             "single_service_local_policy-relaxed", "separate_close_and_void-tight",
             "healthy_upstream_fanout-relaxed", "discount_fee_rounding-tight",
             "single_service_local_policy-tight", "separate_close_and_void-relaxed")
    schedule = []
    for i, tid in enumerate(order):
        arms = tuple(ARMS) if i in (0, 2, 5, 7) else tuple(reversed(ARMS))
        for arm in arms:
            schedule.append({"slot": len(schedule) + 1, "wave": i // 2 + 1, "task_id": tid, "arm": arm})
    provider = load_models_catalog().providers[PROVIDER]
    protocol = {
        "scheme": "v32_observation_R_development_1", "task_sha256": _json_digest(tasks),
        "witness_sha256": _json_digest(witnesses), "sources": _sources(),
        "selection": "Four prespecified structures, one family each, both quota states regardless of reachability in model runs.",
        "arms": ARMS, "schedule": schedule, "max_paid_episodes": 16, "max_steps": MAX_STEPS,
        "provider": PROVIDER, "model_ids": {tier: provider.tiers[tier].resolve_model_id() for tier in ("fast", "think")},
        "router": "research_p0_reactive_only", "router_params": load_router_params().model_dump(),
        "transport": {"timeout_seconds": TIMEOUT, "retries": 0},
        "budget_ceiling_cny": CAP, "meaningful_success_gain_pp": 10,
        "max_incremental_mean_cost_cny_per_start": 0.02,
        "endpoint": "official synthetic sandbox terminal verifier",
        "missing": "Unstarted, interrupted or budget-exhausted episodes remain unknown, with all 8 pairs in bounds.",
        "shadow": "In a rule-permitted dispatch, Fast and complete observations; C first remaining=1, E first remaining>=1.",
        "followup": "No template substitution or paid retry. Four families are for feasibility; expand on independent structures."
    }
    assert set(order) == tasks.keys()
    output.mkdir(parents=True)
    _write_json(output / "protocol.json", protocol)
    _write_json(output / "tasks.json", tasks)
    _write_json(output / "offline_witnesses.json", witnesses)
    return protocol


def _scripted_pool(task, catalog, meter, actions):
    index = 0
    last_id = None

    def reply(_task, _tier, _step, messages, _tools):
        nonlocal index, last_id
        if last_id is not None and any(m.get("tool_call_id") == last_id for m in messages):
            index += 1
        if index >= len(actions):
            return {"content": "The requested work is complete."}
        last_id = f"v32_witness_{index}"
        return {"tool_calls": [{**actions[index], "id": last_id}]}

    return FakeModelPool(catalog, meter, reply)


def _diagnostics(events, proposals, writes, domain):
    tool = RESTRICTED[domain]
    attempted = [c for e in events if e["payload"].get("event") == "batch" for c in e["payload"]["proposals"]
                 if c["tool"] == tool]
    shadow = {"C": None, "E": None}
    permitted = []
    for item in proposals:
        dispatch = item.get("dispatch")
        assessment = item.get("rule_assessment", item["assessment"])
        risk = assessment["risk"]
        if not dispatch or not dispatch["facts_complete"] or risk not in ("tight", "relaxed"):
            continue
        candidate = {key: dispatch[key] for key in ("id", "tool", "target", "pools", "ok")}
        candidate.update(step=item["step"], tier=item["tier"], remaining=assessment["before"]["remaining"])
        permitted.append(candidate)
        if item["tier"] == "fast":
            shadow["E"] = shadow["E"] or candidate
            if risk == "tight":
                shadow["C"] = shadow["C"] or candidate
    c, e = shadow.values()
    if c and e:
        choice = "same_candidate" if (c["step"], c["id"]) == (e["step"], e["id"]) else (
            "different_object" if c["target"] != e["target"] else "same_object_different_candidate")
    else:
        choice = "E_only" if e else "neither"
    supplement_events = [e for e in events if e["payload"].get("event") == "supplement"]
    request_summary = []
    for item in proposals:
        if "defer_for_facts" not in item:
            continue
        step, target = item["step"], item["assessment"]["target"]
        candidate = next((p for p in permitted if p["target"] == target and p["step"] > step and p["ok"]), None)
        following = min((e["step_idx"] for e in events if e["step_idx"] > step
                         and e["payload"].get("event") == "batch"), default=None)
        request_summary.append({"step": step, "target": target, "n_reads": len(item["supplements"]),
                                "all_facts_filled_at_deferral": not item["defer_for_facts"]["unresolved"],
                                "first_accepted_same_target": candidate,
                                "accepted_in_next_proposal_batch": bool(candidate and candidate["step"] == following)})
    rules = [{"step": e["step_idx"], **e["payload"]} for e in events
             if e["payload"].get("event") == "rule_before_review"]
    for rule in rules:
        if rule["decision"] == "replace" and domain == "db_transaction":
            applied = next((w for w in writes if w["step"] == rule["step"] and w["arguments"] == rule["effective"]), None)
            if applied and applied["required_delta_after_mandatory_voids"] is not None:
                correct = applied["required_delta_after_mandatory_voids"]
                rule["correction_evaluation"] = ("fixed_delta" if rule["effective"]["delta_cents"] == correct
                                                 else "introduced_error" if rule["original"]["delta_cents"] == correct
                                                 else "replacement_still_incorrect")
    return {"restricted_proposals": len(attempted), "supplement_reads": len(supplement_events),
            "rule_decisions": rules, "eligible_dispatches": permitted, "restricted_writes": writes,
            "supplemented_candidates": request_summary, "shadow": {**shadow, "comparison": choice}}


def _summary(rows):
    pairs = {}
    by_arm = {arm: Counter() for arm in ARMS}
    for r in rows:
        y, arm = r["outcome"], r["arm"]
        pairs.setdefault(r["task_id"], {})[arm] = y
        counts = by_arm[arm]
        counts["assigned"] += 1
        counts["success"] += y is True
        counts["failure"] += y is False
        counts["unresolved"] += y is None
        d = r["diagnostics"]
        counts["end_state_failure_reaching_C"] += y is False and d["shadow"]["C"] is not None
        counts["episode_reaching_C"] += d["shadow"]["C"] is not None
        counts["episodes_no_restricted_dispatch"] += not d["restricted_writes"]
        counts["episodes_supplementing_restricted_candidate"] += any(s["n_reads"] for s in d["supplemented_candidates"])
        counts["episodes_accepted_after_restricted_supplement"] += any(
            s["n_reads"] and s["first_accepted_same_target"] for s in d["supplemented_candidates"])
        counts["supplement_reads"] += d["supplement_reads"]
        counts["actual_debits"] += sum(w["actual_debit"] for w in d["restricted_writes"])
        counts["certified_quota_losses"] += sum(w["certified_new_loss"] is True for w in d["restricted_writes"])
        counts["uncertified_new_quota_deficits"] += sum(w["certified_new_loss"] is None for w in d["restricted_writes"])
        counts["reactive_think_calls"] += r["terminal_summary"].get("reactive_think_calls", 0)
    paired = Counter()
    for p in pairs.values():
        a, b = (p[arm] for arm in ARMS)
        paired["unresolved" if None in (a, b) else "B_win" if b > a else "A_win" if a > b
               else "both_success" if a else "both_fail"] += 1
    limits = []
    for pair in pairs.values():
        a, b = (pair[arm] for arm in ARMS)
        limits.append(((b if b is not None else 0) - (a if a is not None else 1),
                       (b if b is not None else 1) - (a if a is not None else 0)))
    b = by_arm["supplement_R"]
    return {"by_arm": by_arm, "pairs": paired, "pair_effect_bounds_pp": [100 * sum(k) / len(pairs) for k in zip(*limits)],
            "C_possible_gain_upper_bound_numerator": [b["end_state_failure_reaching_C"],
                                                       b["end_state_failure_reaching_C"] + b["unresolved"]],
            "C_possible_gain_upper_bound_denominator": b["assigned"],
            "shadow_choices_in_supplement_R": Counter(r["diagnostics"]["shadow"]["comparison"]
                                                      for r in rows if r["arm"] == "supplement_R")}


def analyze(output):
    protocol = _read_json(output / "protocol.json")
    tasks = _read_json(output / "tasks.json")
    rows = []
    with closing(_read_database(output / DATABASE)) as db:
        run_rows = {r["run_id"]: r for r in db.execute(
            "SELECT r.run_id,r.status AS run_status,e.status,e.passed,e.failure_tag,e.steps_taken,e.summary_json "
            "FROM runs r LEFT JOIN episodes e USING(run_id)")}
        fees = {rid: Decimal() for rid in run_rows}
        statuses = Counter()
        reserve = Decimal()
        for r in db.execute("SELECT run_id,status,actual_cost_cny,estimated_max_cost_cny FROM api_attempts"):
            fees[r["run_id"]] += Decimal(str(r["actual_cost_cny"]))
            statuses[r["status"]] += 1
            if r["status"] in ("PENDING", "UNKNOWN_BILLING"):
                reserve += Decimal(str(r["estimated_max_cost_cny"]))
        for job in protocol["schedule"]:
            rid = f"v32_observation_{job['slot']:02d}"
            task = tasks[job["task_id"]]
            episode = run_rows.get(rid)
            summary = json.loads(episode["summary_json"] or "{}") if episode else {}
            outcome = bool(episode["passed"]) if (episode and episode["status"] == "COMPLETED"
                                                  and episode["failure_tag"] != "API_ERROR"
                                                  and not summary.get("budget_exhausted")) else None
            events = list(read_events(db, rid))
            if any(e["payload"].get("event") == "review_started" for e in events):
                raise ValueError("An active review would invalidate the v3.2 observation pilot.")
            with closing(create_env(task["domain"])) as env:
                env.reset(task)
                proposals, writes = replay_actions(env, task, events, supplement=ARMS[job["arm"]])
                if outcome is not None and outcome != env.verify(task).passed:
                    raise ValueError(f"Terminal replay disagrees for {rid}")
            rows.append({**job, "run_id": rid, "family": task["family"], "domain": task["domain"], "state": task["state"],
                         "outcome": outcome, "paid_cny": str(fees.get(rid, 0)) if protocol["mode"] == "live" else "0",
                         "run_status": episode["run_status"] if episode else "NOT_STARTED",
                         "failure_tag": episode["failure_tag"] if episode else None, "terminal_summary": summary,
                         "diagnostics": _diagnostics(events, proposals, writes, task["domain"])})
        models = [r[0] for r in db.execute("SELECT DISTINCT model_actual FROM api_attempts WHERE status='OK' ORDER BY 1")]
    return {"protocol_sha256": _json_digest(protocol), "scope": protocol["mode"], "summary": _summary(rows),
            "assignments": rows, "models_actual": models, "attempt_statuses": dict(statuses),
            "charge_cny": str(sum(fees.values())) if protocol["mode"] == "live" else "0",
            "reserve_cny": str(reserve) if protocol["mode"] == "live" else "0"}


def execute(frozen, output, *, budget=None):
    protocol = _read_json(frozen / "protocol.json")
    tasks = _read_json(frozen / "tasks.json")
    witnesses = _read_json(frozen / "offline_witnesses.json")
    if _sources() != protocol["sources"] or _json_digest(tasks) != protocol["task_sha256"] or (
            _json_digest(witnesses) != protocol["witness_sha256"]):
        raise ValueError("Frozen protocol no longer matches the inputs.")
    errors = verify_manifest() + audit_orchestrator_ast()
    if errors:
        raise ValueError("Immutable-input verification failed: " + "; ".join(errors))
    catalog = load_models_catalog()
    provider = catalog.providers[PROVIDER]
    params = load_router_params()
    if params.model_dump() != protocol["router_params"] or any(
            provider.tiers[tier].resolve_model_id() != name for tier, name in protocol["model_ids"].items()):
        raise ValueError("Configured router or models no longer match the frozen protocol.")
    if budget is not None and not (0 < budget <= CAP and get_api_key(provider)
                                  and get_base_url(provider).rstrip("/") == "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        raise ValueError("Use the authorized cap and configured DashScope host.")
    if budget is not None:
        qa = _read_json(frozen / "offline" / "result.json")
        offline_protocol = {**protocol, "mode": "offline_scripted", "actual_cap_cny": 0}
        if qa["protocol_sha256"] != _json_digest(offline_protocol) or not all(
                row["outcome"] is True for row in qa["assignments"]) or len(qa["assignments"]) != 16:
            raise ValueError("The matching frozen offline acceptance is required.")
    cap = budget or CAP
    output.mkdir(parents=True)
    mode = "live" if budget is not None else "offline_scripted"
    _write_json(output / "protocol.json", {**protocol, "mode": mode, "actual_cap_cny": budget or 0})
    _write_json(output / "tasks.json", tasks)
    meter = BillingMeter(output / DATABASE, project_cap_cny=cap)
    live_pool = LiveModelPool(catalog, meter, timeout_sec=TIMEOUT, max_retries=0) if budget is not None else None

    def run(job):
        task, arm = tasks[job["task_id"]], job["arm"]
        domain = task["domain"]
        rid = f"v32_observation_{job['slot']:02d}"
        meter.create_or_resume_run(run_id=rid, stage="v32_observation_pilot", split="development", policy=arm,
                                   provider=PROVIDER, round_cap_cny=cap, stage_cap_cny=cap, metadata=job)
        try:
            run_episode(task=task, run_id=rid, provider=PROVIDER,
                        router=ResearchStepRouter("p0", params, use_guards=False),
                        pool=live_pool or _scripted_pool(task, catalog, meter, witnesses[job["task_id"]]),
                        meter=meter, prewrite=control_for(domain, supplement=ARMS[arm]), proposal_rule=public_rule)
        except (APIError, BudgetExceededError):
            meter.finish_run(rid, status="INTERRUPTED")
        else:
            meter.finish_run(rid)

    try:
        with ThreadPoolExecutor(max_workers=4) as workers:
            for i in range(0, len(protocol["schedule"]), 4):
                list(workers.map(run, protocol["schedule"][i:i + 4]))
    finally:
        report = analyze(output)
        _write_json(output / "result.json", report)
    if budget is None:
        for row in report["assignments"]:
            assert row["outcome"] is True
            diag = row["diagnostics"]
            if row["family"] != "separate_close_and_void":
                assert bool(diag["supplement_reads"]) is ARMS[row["arm"]]
                assert bool(diag["eligible_dispatches"]) is ARMS[row["arm"]]
                assert bool(diag["supplemented_candidates"]) is ARMS[row["arm"]]
            else:
                assert not diag["restricted_writes"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--offline", action="store_true")
    group.add_argument("--live-out", type=Path)
    parser.add_argument("--live-budget-cny", type=float)
    args = parser.parse_args()
    if args.freeze:
        data = freeze(args.frozen)
        print({"planned": len(data["schedule"]), "protocol_sha256": _json_digest(data)})
    else:
        if bool(args.live_out) != (args.live_budget_cny is not None):
            parser.error("A live output and an explicit budget are required together.")
        data = execute(args.frozen, args.live_out or args.frozen / "offline", budget=args.live_budget_cny)
        print({k: v for k, v in data.items() if k != "assignments"}, flush=True)


if __name__ == "__main__":
    main()
