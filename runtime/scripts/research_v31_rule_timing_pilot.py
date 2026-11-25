"""Run the prespecified 12-slot B+R, C+R and E+R development comparison."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
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
from aroute.orchestrator.prewrite import PrewriteControl, Read
from aroute.orchestrator.research_router import ResearchStepRouter
from aroute.orchestrator.router import load_router_params

from .research_run_pilot import DATABASE, PILOT_RUNS, _json_digest, _read_database, _read_json
from .research_v31_rule_fork import apply_invoice_rule
from .research_v31_timing_candidates import CurrentClusterControl, accounting, restart_rule, service


PROVIDER = "dashscope"
ARMS = {"B+R": "off", "C+R": "targeted", "E+R": "first"}
TIMEOUT = 120
MAX_STEPS = 30


def selected_tasks():
    draft_file = PILOT_RUNS / "v31_timing_drafts_20261123" / "candidate_tasks.json"
    available = _read_json(draft_file)
    selected = {task["task_id"]: {**task, "difficulty": "development", "max_steps": MAX_STEPS}
                for task in available if task["family"] == task["template_id"] + "-1"}
    if set(Counter(task["domain"] for task in selected.values()).values()) != {2} or len(selected) != 4:
        raise ValueError("Expected exactly family 1 and two starting states from each archived template.")
    order = [("db_transaction", "tight"), ("ops_troubleshoot", "relaxed"),
             ("db_transaction", "relaxed"), ("ops_troubleshoot", "tight")]
    index = {(task["domain"], task["state"]): task for task in selected.values()}
    schedule = [{"slot": wave*4 + offset + 1, "wave": wave+1, "task_id": index[block]["task_id"],
                 "arm": tuple(ARMS)[(wave + offset) % len(ARMS)]}
                for wave in range(3) for offset, block in enumerate(order)]
    return selected, schedule


def public_rule(control, candidate):
    if candidate.name == "post_adjustment":
        revised, decision = apply_invoice_rule(control, candidate)
        if decision is not None and decision["decision"] == "defer":
            decision = {**decision, "feedback": "The observed invoice is balanced. Check the other invoices and remaining work."}
        return revised, decision
    if candidate.name != "restart_service" or candidate.arguments_dict is None or candidate.parse_error:
        return candidate, None
    assessment = control.assess(candidate.name, candidate.arguments_dict)
    decision = restart_rule(assessment, control, candidate.arguments_dict)
    if decision["decision"] != "defer":
        return candidate, decision
    receipt = control.receipts[Read("inspect_service", "svc_id", candidate.arguments_dict["svc_id"])].data
    if decision["reason"] == "visible_cluster_memory":
        feedback = f"Current public cluster memory {receipt['current_total_mem_mb']} MB exceeds its {receipt['cluster_mem_cap_mb']} MB limit. Resolve the visible memory conflict before proposing a restart."
    else:
        feedback = "The observed active configuration violates the target service's published startup policy. Recheck the configuration before restarting."
    return None, {**decision, "feedback": feedback}


def offline_pool(task, catalog, meter):
    domain = task["domain"]
    _, _, legal, wrong, _ = (accounting if domain == "db_transaction" else service)(1)
    actions = ([wrong[0], wrong[1], *legal[2:]] if domain == "db_transaction" else wrong)
    index = 0
    last_reviewable = None

    def reply(_task, _tier, _step, messages, tools):
        nonlocal index, last_reviewable
        if tools is None:
            if domain == "db_transaction":
                proposed = json.loads(messages[-1]["content"].split("\n", 1)[1])["candidate"]
                return {"content": json.dumps({"decision": "replace", "name": proposed["name"],
                                               "arguments": {**proposed["arguments"], "delta_cents": -9}})}
            return {"content": '{"decision":"keep"}'}
        if last_reviewable is not None:
            dispatched = any(message.get("tool_call_id") == last_reviewable for message in messages)
            known_reject = (domain == "ops_troubleshoot" and index == 6
                            and messages[-1]["role"] == "user" and "public cluster memory" in messages[-1]["content"])
            if dispatched or known_reject:
                index += 1
        if index >= len(actions):
            return {"content": "The scripted operation path is complete."}
        last_reviewable = f"v31-timing-{index}"
        return {"tool_calls": [{"id": last_reviewable, **actions[index]}]}

    return FakeModelPool(catalog, meter, reply)


def analyze(output):
    protocol = _read_json(output / "protocol.json")
    assignments = {job["slot"]: job for job in protocol["schedule"]}
    result = []
    billed = defaultdict(Decimal)
    statuses = Counter()
    reserve = Decimal()
    with closing(_read_database(output / DATABASE)) as db:
        for row in db.execute("SELECT run_id,status,actual_cost_cny,estimated_max_cost_cny FROM api_attempts"):
            statuses[row["status"]] += 1
            billed[row["run_id"]] += Decimal(str(row["actual_cost_cny"]))
            if row["status"] in ("UNKNOWN_BILLING", "PENDING"):
                reserve += Decimal(str(row["estimated_max_cost_cny"]))
        for run in db.execute("SELECT r.run_id,r.metadata_json,r.status AS run_status,e.status,e.passed,e.failure_tag,e.summary_json "
                              "FROM runs r LEFT JOIN episodes e USING(run_id) ORDER BY r.run_id"):
            rid = run["run_id"]
            meta = json.loads(run["metadata_json"])
            events, tool_events = [], []
            for row in db.execute("SELECT step_idx,tier,tool_name,tool_args_json,tool_ok,error_code FROM step_events "
                                  "WHERE run_id=? ORDER BY event_id", (rid,)):
                args = json.loads(row["tool_args_json"] or "{}")
                stamp = {"step": row["step_idx"], "tier": row["tier"]}
                if row["tool_name"] == "__V3_PREWRITE__":
                    events.append({**stamp, **args})
                elif row["tool_name"] and not row["tool_name"].startswith("__"):
                    tool_events.append({**stamp, "tool": row["tool_name"], "arguments": args,
                                        "ok": bool(row["tool_ok"]), "error": row["error_code"]})
            valid = run["status"] == "COMPLETED" and run["failure_tag"] != "API_ERROR"
            result.append({**assignments[meta["slot"]], "run_id": rid, "domain": meta["domain"], "state": meta["state"],
                           "outcome": bool(run["passed"]) if valid else None, "failure_tag": run["failure_tag"],
                           "run_status": run["run_status"], "paid_cny": str(billed[rid]),
                           "terminal_summary": json.loads(run["summary_json"] or "{}"),
                           "rules": [e for e in events if e["event"].startswith("rule_") and "decision" in e],
                           "reviews": [e for e in events if e["event"] in ("review_started", "review_result")],
                           "write_assessments": [e for e in events if e["event"] in ("assessment", "rule_assessment", "dispatch")],
                           "supplements": sum(e["event"] == "supplement" for e in events), "tool_events": tool_events})
        models = [r[0] for r in db.execute("SELECT DISTINCT model_actual FROM api_attempts WHERE status='OK' ORDER BY 1")]
    return {"scope": protocol["mode"], "protocol_sha256": _json_digest(protocol), "planned": len(assignments),
            "successful_or_failed_business_outcomes": sum(row["outcome"] is not None for row in result),
            "actual_model_ids": models, "attempt_statuses": dict(statuses), "assignments": result,
            "charge_cny": str(sum(billed.values())) if protocol["mode"] == "live" else "0",
            "unresolved_reserve_cny": str(reserve) if protocol["mode"] == "live" else "0"}


def execute(output, *, budget=None):
    catalog = load_models_catalog()
    provider = catalog.providers[PROVIDER]
    if budget is not None and (not 0 < budget <= 2.4 or not get_api_key(provider)
                               or get_base_url(provider).rstrip("/") != "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        raise ValueError("Live internal development requires the configured endpoint and a budget at most 2.4 CNY.")
    if errors := verify_manifest() + audit_orchestrator_ast():
        raise ValueError("Immutable input verification failed: " + "; ".join(errors))
    params = load_router_params()
    tasks, schedule = selected_tasks()
    paths = sorted((PROJECT_ROOT / "src/aroute").rglob("*.py"))
    paths += [PROJECT_ROOT / "scripts" / name for name in ("research_v31_rule_fork.py", "research_v31_control_checks.py",
                                                           "research_v31_timing_candidates.py", Path(__file__).name)]
    output.mkdir(parents=True)
    protocol = {"mode": "live" if budget is not None else "offline_scripted", "task_sha256": _json_digest(tasks),
                "family_selection": "variant 1 from each archived timing draft, prespecified",
                "schedule": schedule, "arms": ARMS, "budget_cny": budget or 0, "max_steps": MAX_STEPS,
                "transport": {"timeout_seconds": TIMEOUT, "retries": 0}, "controller_order": "supplement > R > optional review > R on replacement",
                "proactive_only_on_fast": True, "router_params": params.model_dump(), "provider": PROVIDER,
                "model_ids": {k: provider.tiers[k].resolve_model_id() for k in ("fast", "think")},
                "catalog_sha256": sha256_file(PROJECT_ROOT / "configs/models.yaml"),
                "sources": {p.relative_to(PROJECT_ROOT).as_posix(): sha256_file(p) for p in paths}}
    (output / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "tasks.json").write_text(json.dumps(tasks, indent=2) + "\n", encoding="utf-8")
    cap = budget or 20.0
    meter = BillingMeter(output / DATABASE, project_cap_cny=cap)
    pool = LiveModelPool(catalog, meter, timeout_sec=TIMEOUT, max_retries=0) if budget is not None else None

    def run(job):
        task = tasks[job["task_id"]]
        domain, arm = task["domain"], job["arm"]
        cls = CurrentClusterControl if domain == "ops_troubleshoot" else PrewriteControl
        control = cls(domain, supplement=True, review=ARMS[arm])
        rid = f"v31_timing_{job['slot']:02d}"
        meter.create_or_resume_run(run_id=rid, stage="v31_rule_timing", split="development", policy=arm, provider=PROVIDER,
                                   round_cap_cny=cap, stage_cap_cny=cap,
                                   metadata={"slot": job["slot"], "domain": domain, "state": task["state"]})
        try:
            run_episode(task=task, run_id=rid, provider=PROVIDER, router=ResearchStepRouter("p0", params, use_guards=False),
                        pool=pool or offline_pool(task, catalog, meter), meter=meter, prewrite=control, proposal_rule=public_rule)
        except (APIError, BudgetExceededError):
            meter.finish_run(rid, status="INTERRUPTED")
        else:
            meter.finish_run(rid)

    try:
        with ThreadPoolExecutor(max_workers=4) as workers:
            for start in range(0, len(schedule), 4):
                list(workers.map(run, schedule[start:start+4]))
    finally:
        report = analyze(output)
        (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if budget is None:
        for row in report["assignments"]:
            assert row["outcome"] is True
            assert any(rule["decision"] in ("replace", "defer") for rule in row["rules"])
            starts = [event for event in row["reviews"] if event["event"] == "review_started"]
            assert len(starts) == int(row["arm"] != "B+R" and (row["arm"] == "E+R" or row["state"] == "tight"))
            if starts:
                assert starts[0]["remaining"] == (1 if row["arm"] == "C+R" else (3 if row["domain"] == "db_transaction" and row["state"] == "relaxed" else 2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--live-budget-cny", type=float)
    args = parser.parse_args()
    report = execute(args.out_dir, budget=args.live_budget_cny)
    print({key: value for key, value in report.items() if key != "assignments"}, flush=True)


if __name__ == "__main__":
    main()
