"""Branch a logged pre-dispatch accounting checkpoint and continue all branches with P0."""

from __future__ import annotations

import argparse
import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from openai import APIError

from aroute.core.billing_meter import BillingMeter, BudgetExceededError
from aroute.core.config import get_api_key, get_base_url, load_models_catalog
from aroute.core.immutable import audit_orchestrator_ast, sha256_file, verify_manifest
from aroute.core.model_pool import FakeModelPool, LiveModelPool, ToolCallSpec
from aroute.core.prewrite_execution import assistant_action, resolve_proposal, review_decision, review_instruction
from aroute.core.runner import SYSTEM_PROMPT
from aroute.core.verifier import verify_episode
from aroute.envs import create_env
from aroute.envs.base import ToolResult
from aroute.orchestrator.memory_ledger import StateLedger
from aroute.orchestrator.prewrite import PrewriteControl
from aroute.orchestrator.research_router import ResearchStepRouter
from aroute.orchestrator.router import RouterParams

from .research_run_pilot import DATABASE, PILOT_RUNS, _json_digest, _read_database, _read_json
from .research_v31_accounting_audit import matched_terminal, ticket_facts


PARENT = PILOT_RUNS / "prewrite_v3_live_20261123"
PROVIDER = "dashscope"
SLOT = 79
MODEL_TIMEOUT = 120
REPLAN = "Pending proposal and remaining batch were not dispatched. Replan using current public observations and propose the next action."
TRUNCATED = "Previous response hit max_tokens before emitting a tool call. Emit a concise direct tool call now."


@dataclass
class Checkpoint:
    task: dict
    step: int
    candidate: ToolCallSpec
    database: bytes
    ledger: StateLedger
    control: PrewriteControl
    params: RouterParams
    prefix_cost: float


def _tool_call(call_id, name, arguments):
    return ToolCallSpec(call_id, name, json.dumps(arguments, ensure_ascii=False), arguments)


def _record_exchange(ledger, call, result):
    ledger.record_assistant_message(assistant_action(call))
    ledger.record_tool_message(call.id, result.to_tool_message())


def capture(run_dir: Path) -> Checkpoint:
    protocol = _read_json(run_dir / "protocol.json")
    tasks = _read_json(run_dir / "tasks.json")
    if (_json_digest(tasks) != protocol["task_sha256"]
            or _json_digest({k: v for k, v in protocol.items() if k != "fingerprint"}) != protocol["fingerprint"]):
        raise ValueError("The parent checkpoint source does not match its frozen digest.")
    params = RouterParams.model_validate(protocol["router_params"])
    with closing(_read_database(run_dir / DATABASE)) as db, closing(create_env("db_transaction")) as env:
        run = matched_terminal(db, protocol, SLOT)
        task = tasks[run["task_id"]]
        env.reset(task)
        ledger = StateLedger(window_k=params.ledger_window_k)
        ledger.init_raw_conversation(SYSTEM_PROMPT, task["prompt"])
        control = PrewriteControl(task["domain"], supplement=True)
        pending = deque()
        for row in db.execute("SELECT * FROM step_events WHERE run_id=? ORDER BY event_id", (run["run_id"],)):
            step, name, tier = row["step_idx"], row["tool_name"], row["tier"]
            arguments = json.loads(row["tool_args_json"] or "{}")
            if name == "__V3_PREWRITE__":
                event = arguments["event"]
                if event == "batch":
                    pending = deque(arguments["proposals"])
                elif event == "supplement":
                    call = _tool_call(arguments["id"], arguments["tool"], arguments["arguments"])
                    result = env.execute_tool(call.name, call.arguments_dict)
                    if json.loads(result.to_tool_message()) != arguments["observation"]:
                        raise ValueError("Reconstructed application receipt differs from the archived receipt.")
                    _record_exchange(ledger, call, result)
                    control.observe(call.name, call.arguments_dict, result)
                    control.reads_started += 1
                elif event == "defer_for_facts":
                    ledger.record_user_feedback(REPLAN)
                elif event == "review_started":
                    original = next(proposal for proposal in pending if proposal["id"] == arguments["id"])
                    candidate = _tool_call(original["id"], original["tool"], original["arguments"])
                    facts = control.assess(candidate.name, candidate.arguments_dict)
                    if len(pending) != 1 or facts is None or facts.missing or facts.risk != "tight":
                        raise ValueError("Expected a single eligible sticky candidate in the archived checkpoint.")
                    prefix = db.execute("SELECT SUM(actual_cost_cny) FROM api_attempts WHERE run_id=? "
                                        "AND (step_idx < ? OR (step_idx=? AND tier='fast'))",
                                        (run["run_id"], step, step)).fetchone()[0]
                    return Checkpoint(task, step, candidate, env.conn.serialize(), ledger, control, params, prefix)
                continue
            if name.startswith("__"):
                if name != "__ROUTE__":
                    raise ValueError(f"Unsupported archived prefix event: {name}")
                continue
            archived = pending.popleft()
            if archived["tool"] != name or archived["arguments"] != arguments or row["fsm_blocked"]:
                raise ValueError("Archived proposal and actual dispatch differ before the selected review.")
            call = _tool_call(archived["id"], name, arguments)
            result = env.execute_tool(name, arguments)
            if (result.ok, result.error_code) != (bool(row["tool_ok"]), row["error_code"]):
                raise ValueError("Archived outcome disagrees with the reconstructed backend.")
            _record_exchange(ledger, call, result)
            control.observe(name, arguments, result)
            ledger.update(step_idx=step, tier=tier, tool_name=name, args=arguments, result=result, fsm_blocked=False)
    raise ValueError("Archived slot has no qualifying checkpoint.")


def run_branch(arm, checkpoint, pool, meter, cap, *, proposal_rule=None):
    task = checkpoint.task
    run_id = f"v31_{arm}"
    episode_id = f"{run_id}_{task['task_id']}"
    meter.create_or_resume_run(run_id=run_id, stage="v31_accounting_checkpoint", split="development",
                               policy=arm, provider=PROVIDER, round_cap_cny=cap, stage_cap_cny=cap,
                               metadata={"source_slot": SLOT, "branch": arm})
    meter.start_episode(episode_id=episode_id, run_id=run_id, task_id=task["task_id"], domain=task["domain"],
                        difficulty=task["difficulty"])
    env = create_env(task["domain"])
    env.reset(task)
    env.conn.deserialize(checkpoint.database)
    ledger, control = deepcopy(checkpoint.ledger), deepcopy(checkpoint.control)
    router = ResearchStepRouter("p0", checkpoint.params, use_guards=False)
    schemas = env.tool_schemas()
    trace, route = [], []
    rule_events = []
    additional_cost = 0.0
    first_effect = None
    start, calls = checkpoint.step, 0
    step, tier = start, "fast"

    def api(tier, messages, tools):
        nonlocal additional_cost, calls
        answer = pool.call_with_tools(run_id=run_id, episode_id=episode_id, task_id=task["task_id"], step_idx=step,
                                      provider=PROVIDER, tier=tier, messages=messages, tools=tools)
        additional_cost += answer.cost_cny
        calls += 1
        return answer

    def dispatch(call, application_read=False):
        args = call.arguments_dict or {}
        if application_read and env.is_write_tool(call.name):
            raise ValueError("Application read cannot dispatch a mutation.")
        ticket = args.get("ticket_id")
        before = ticket_facts(env, ticket) if call.name == "post_adjustment" and ticket in task["assertions"]["closed_tickets"] else None
        if call.parse_error or call.arguments_dict is None:
            result = ToolResult(ok=False, error_code="SCHEMA_PARSE_ERROR", error_msg=call.parse_error)
        else:
            result = env.execute_tool(call.name, args)
            control.observe(call.name, args, result)
        _record_exchange(ledger, call, result)
        if not application_read:
            ledger.update(step_idx=step, tier=tier, tool_name=call.name, args=args, result=result, fsm_blocked=False)
        trace.append({"step": step, "tier": tier, "tool": call.name, "arguments": args, "application_read": application_read,
                      "ok": result.ok, "error": result.error_code, "ticket_before": before,
                      "ticket_after": ticket_facts(env, args["ticket_id"]) if before else None})
        return result

    def no_review(_):
        raise AssertionError("P0 continuation cannot invoke proactive review.")

    def apply(call):
        if call is None:
            ledger.record_user_feedback(REPLAN)
            return False
        resolved = resolve_proposal(control, call, tier=tier, schemas=schemas,
                                    read=lambda request: dispatch(request, application_read=True),
                                    review=no_review, can_review=False, log=lambda *_: None)
        if resolved.call is None:
            ledger.record_user_feedback(REPLAN)
            return False
        selected = resolved.call
        if proposal_rule is not None:
            selected, decision = proposal_rule(control, selected)
            if decision is not None:
                rule_events.append({"step": step, "proposal": resolved.call.arguments_dict, **decision})
            if selected is None:
                ledger.record_user_feedback("The invoice is already balanced. Recheck which task actions are still pending.")
                return False
        dispatch(selected)
        return True

    initial = checkpoint.candidate
    disposition, reviewer_model = "direct", None
    reason = "step_limit"
    terminal = None
    shared_input = [*ledger.raw_messages, {"role": "user", "content": review_instruction(initial, schemas)}]
    review_hash = _json_digest(shared_input)
    try:
        if arm in ("F", "T"):
            reviewer = api("fast" if arm == "F" else "think", shared_input, None)
            reviewer_model = reviewer.model_actual
            disposition, initial = review_decision(reviewer.content, initial, schemas)
            if reviewer.tool_calls or reviewer.finish_reason == "length":
                disposition, initial = "invalid_review", None
        replan_fast = not apply(initial)
        if trace and not trace[-1]["application_read"]:
            first_effect = trace[-1]

        truncations = 0
        for step in range(start + 1, task["max_steps"]):
            budget = checkpoint.params.episode_budget_cap_cny
            remaining = max(0.0, (budget - checkpoint.prefix_cost - additional_cost) / budget)
            if not remaining:
                reason = "episode_budget"
                break
            selected = router.select_tier(ledger=ledger, remaining_budget_ratio=remaining, step_idx=step, domain=task["domain"])
            tier = "fast" if replan_fast else selected
            route.append({"step": step, "tier": tier, "reason": "prewrite_reproposal" if replan_fast else router.last_route_event["reason"]})
            replan_fast = False
            reply = api(tier, ledger.build_messages(system_prompt=SYSTEM_PROMPT, task_prompt=task["prompt"],
                                                   active_tier=tier, use_state_ledger=False, provider=PROVIDER), schemas)
            ledger.last_prompt_tokens = reply.prompt_tokens
            if not reply.tool_calls:
                if reply.finish_reason != "length":
                    reason = "agent_stop"
                    break
                truncations += 1
                ledger.open_errors = [f"s{step} OUTPUT_TRUNCATION: {TRUNCATED}"]
                ledger.error_streak += 1
                ledger.record_user_feedback(TRUNCATED)
                if truncations == 2:
                    reason = "truncation_limit"
                    break
                continue
            truncations = 0
            for call in reply.tool_calls:
                if not apply(call):
                    replan_fast = True
                    break
        terminal = verify_episode(env, task)
    except (APIError, BudgetExceededError) as exc:
        reason = type(exc).__name__
    finally:
        final = {ticket: ticket_facts(env, ticket) for ticket in task["assertions"]["closed_tickets"]}
        env.close()
        meter.finish_episode(episode_id=episode_id, passed=bool(terminal and terminal.passed),
                             failure_tag=None if terminal and terminal.passed else reason,
                             steps_taken=calls, total_cost_cny=round(additional_cost, 8), loop_abort=False,
                             summary={"verified": terminal is not None, "review_disposition": disposition})
        meter.finish_run(run_id, status="COMPLETED" if terminal is not None else "INTERRUPTED")
    return {"arm": arm, "review_message_hash": review_hash if arm in ("F", "T") else None, "review_disposition": disposition,
            "reviewer_model": reviewer_model, "first_effect": first_effect, "trace": trace, "P0_routes": route,
            "passed": terminal.passed if terminal is not None else None, "stop": reason, "final_tickets": final,
            "successful_model_calls": calls, "charged_cny": round(additional_cost, 8),
            **({"rule_events": rule_events} if proposal_rule is not None else {})}


def _fake_pool(arm, catalog, meter, checkpoint):
    adjustment = checkpoint.candidate.arguments_dict
    ticket, invoice = adjustment["ticket_id"], adjustment["inv_id"]
    other, = (t for t in checkpoint.task["assertions"]["closed_tickets"] if t != ticket)
    installment = f"INSTALLMENTS-{ticket.split('-')[1]}"
    other_inv = invoice[:-1] + "2"
    script = deque([("run_audit_check", {"ticket_id": ticket}), ("reconcile_invoice", {"inv_id": installment}),
                    ("reconcile_invoice", {"inv_id": invoice}), ("close_ticket", {"ticket_id": ticket}),
                    ("get_invoice_bundle", {"ticket_id": other}), ("query_account", {"acct_id": adjustment["acct_id"]}),
                    ("post_adjustment", {**adjustment, "ticket_id": other, "inv_id": other_inv, "delta_cents": 247}),
                    ("run_audit_check", {"ticket_id": other}), ("reconcile_invoice", {"inv_id": other_inv}),
                    ("close_ticket", {"ticket_id": other})])

    def reply(task_id, tier, step, messages, tools):
        if tools is None:
            decision = {"decision": "keep"} if arm == "F" else {"decision": "replace", "name": "post_adjustment",
                                                                 "arguments": {**adjustment, "delta_cents": 242}}
            return {"content": json.dumps(decision)}
        if not script:
            return {"content": "Finished scripted continuation."}
        name, arguments = script.popleft()
        return {"tool_calls": [{"name": name, "arguments": arguments}]}

    return FakeModelPool(catalog, meter, reply)


def execute(output, live, cap):
    checkpoint = capture(PARENT)
    catalog = load_models_catalog()
    if live and (not 0 < cap <= 2 or not get_api_key(catalog.providers[PROVIDER])
                 or get_base_url(catalog.providers[PROVIDER]).rstrip("/") != "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        raise ValueError("Live execution requires the approved DashScope endpoint and a positive cap of at most 2 CNY.")
    if errors := verify_manifest() + audit_orchestrator_ast():
        raise ValueError("Integrity checks failed: " + "; ".join(errors))
    output.mkdir(parents=True)
    parent = _read_json(PARENT / "protocol.json")
    protocol = {"mode": "live" if live else "fake", "parent_protocol": parent["fingerprint"], "source_slot": SLOT,
                "archived_candidate": checkpoint.candidate.arguments_dict, "archived_step": checkpoint.step,
                "snapshot_sha256": sha256(checkpoint.database).hexdigest(),
                "public_prefix_sha256": _json_digest(checkpoint.ledger.raw_messages), "paid_cap_cny": cap if live else 0,
                "timeout_seconds": MODEL_TIMEOUT, "retries": 0, "suffix_policy": "same O1/P0, guards off",
                "archive_match": "reconstructed public messages and original candidate; no archived raw wire bytes",
                "executor_sha256": sha256_file(Path(__file__))}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    meter = BillingMeter(output / DATABASE, project_cap_cny=cap)
    live_pool = LiveModelPool(catalog, meter, timeout_sec=MODEL_TIMEOUT, max_retries=0) if live else None

    def branch(arm):
        return run_branch(arm, checkpoint, live_pool or _fake_pool(arm, catalog, meter, checkpoint), meter, cap)

    with ThreadPoolExecutor(max_workers=3) as threads:
        branches = list(threads.map(branch, ("D", "F", "T")))
    if branches[1]["review_message_hash"] != branches[2]["review_message_hash"]:
        raise ValueError("Fast and Think checkpoint inputs differ.")
    with closing(_read_database(output / DATABASE)) as db:
        attempts = {r["status"]: {"count": r["n"], "charged": round(r["cost"], 8), "estimated": round(r["estimate"], 8)}
                    for r in db.execute("SELECT status,COUNT(*) n,SUM(actual_cost_cny) cost, "
                                        "SUM(estimated_max_cost_cny) estimate FROM api_attempts GROUP BY status")}
    result = {**protocol, "branches": branches, "attempts": attempts,
              "external_charge_cny": round(sum(r["charged"] for r in attempts.values()), 8) if live else 0,
              "external_reserve_cny": round(sum(r["estimated"] for tag, r in attempts.items()
                                                  if tag in ("PENDING", "UNKNOWN_BILLING")), 8) if live else 0}
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not live and [row["passed"] for row in branches] != [False, False, True]:
        raise ValueError("Offline checkpoint continuation did not pass its expected three-branch fixture.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--budget-cny", type=float)
    args = parser.parse_args()
    if args.live != (args.budget_cny is not None):
        parser.error("Provide --live and --budget-cny together.")
    report = execute(args.out_dir, args.live, args.budget_cny if args.live else 2.0)
    print({row["arm"]: [row["passed"], row["stop"]] for row in report["branches"]}, flush=True)


if __name__ == "__main__":
    main()
