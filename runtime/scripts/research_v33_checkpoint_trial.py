"""Freeze and run the v3.3 same-prefix development trials in SQLite sandboxes."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from contextlib import closing
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path

from openai import APIError

from aroute.core.billing_meter import BillingMeter, BudgetExceededError
from aroute.core.config import PROJECT_ROOT, get_api_key, get_base_url, load_models_catalog
from aroute.core.immutable import audit_orchestrator_ast, sha256_file, verify_manifest
from aroute.core.model_pool import FakeModelPool, LiveModelPool, ToolCallSpec
from aroute.core.prewrite_execution import assistant_action, contract_debit, resolve_proposal
from aroute.envs import create_env
from aroute.envs.ops_troubleshoot_env import check_upstream_compat
from aroute.orchestrator.prewrite import Read

from .research_run_pilot import DATABASE, _json_digest, _read_database, _read_json
from .research_v31_rule_timing_pilot import PROVIDER, public_rule
from .research_v31_timing_candidates import call
from .research_v32_observation_pilot import _sources, _write_json
from .research_v32_replay import ACCOUNTING, SERVICE, control_for, remaining, service_witness
from .research_v33_cases import accounting_cases, service_cases


CAPS = {"accounting": 0.60, "service": 0.50}
ARMS = {"accounting": ("replan", "revalidate"), "service": ("local_R", "dependency_R")}
SYSTEM = ("Use the public conversation and sandbox tools to complete the requested transaction. "
          "Current tool observations supersede earlier observations. Follow current user authorization. "
          "Propose the next individual tool action or stop when finished.")


def _spec(item, cid):
    args = item["arguments"]
    return ToolCallSpec(cid, item["name"], json.dumps(args, ensure_ascii=False), args)


def _record(messages, spec, result):
    messages.extend([assistant_action(spec), {"role": "tool", "tool_call_id": spec.id, "content": result.to_tool_message()}])


def _read_set(case, current):
    target = case["candidate"]["arguments"]
    if case["task"]["domain"] == ACCOUNTING:
        return [call("get_invoice_bundle", ticket_id=target["ticket_id"]), call("query_account", acct_id=target["acct_id"])]
    return [call("inspect_service", svc_id=target["svc_id"]),
            call("list_config_revisions", svc_id=target["svc_id"]), call("inspect_service", svc_id=case["upstream"])]


def _checkpoint(case):
    """Capture two public observations with a frozen proposal and the current sandbox state."""
    task, candidate = case["task"], case["candidate"]
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task["prompt"]}]
    domain = task["domain"]
    control = control_for(domain, supplement=True)
    reads = _read_set(case, None)
    previous = case.get("prior")
    if previous is not None:
        messages.append({"role": "user", "content": "These are the earlier records. Initial posting grant was version 1."})
        with closing(create_env(domain)) as old:
            old.reset(previous)
            for i, observation in enumerate(reads):
                spec = _spec(observation, f"prior_{i}")
                _record(messages, spec, old.execute_tool(spec.name, spec.arguments_dict))
    messages.append({"role": "user", "content": "A concrete pending proposal was recorded before the current checks. "
                     + json.dumps(candidate, ensure_ascii=False, sort_keys=True)})
    messages.append({"role": "user", "content": "The following checks are current."})
    env = create_env(domain)
    env.reset(task)
    for i, observation in enumerate(reads):
        spec = _spec(observation, f"current_{i}")
        result = env.execute_tool(spec.name, spec.arguments_dict)
        if not result.ok or result.state_changed or result.is_write:
            env.close()
            raise ValueError("A checkpoint did not provide the frozen public reads.")
        control.observe(spec.name, spec.arguments_dict, result)
        _record(messages, spec, result)
    approval = case.get("approval")
    if approval:
        messages.append({"role": "user", "content": "Latest posting authorization for this transaction: "
                         + json.dumps(approval, sort_keys=True)
                         + ". Use current facts. An earlier grant does not override the latest record."})
    return env, control, messages


def _authorized(approval, args, step):
    return (approval["allowed"] and step <= approval["valid_through_decision"]
            and all(args.get(key) == approval[key] for key in ("ticket_id", "inv_id", "acct_id"))
            and type(args.get("delta_cents")) is int and 0 < abs(args["delta_cents"]) <= approval["max_abs_delta_cents"])


def retain_original(control, candidate, approval, step):
    args = candidate.arguments_dict
    if not _authorized(approval, args, step):
        return False, "authorization_or_binding"
    assessment = control.assess(candidate.name, args)
    if not assessment or assessment.missing or assessment.risk not in ("tight", "relaxed"):
        return False, "current_receipts_or_quota"
    bundle = control.receipts[Read("get_invoice_bundle", "ticket_id", args["ticket_id"])].data
    invoice = next((row for row in bundle["invoices"] if row["inv_id"] == args["inv_id"]), None)
    if invoice is None or invoice["acct_id"] != args["acct_id"] or bundle["ticket"]["status"] != "OPEN":
        return False, "invoice_or_ticket_binding"
    revised, decision = public_rule(control, candidate)
    if revised != candidate or decision is None or decision["decision"] != "keep":
        return False, "rule_requires_change_or_cannot_determine"
    return True, "unchanged_and_currently_authorized"


def check_dependency(control, candidate):
    sid = candidate.arguments_dict["svc_id"]
    local = control.receipts[Read("inspect_service", "svc_id", sid)].data["service"]
    params = {}
    for dep in local["deps"]:
        observed = control.receipts.get(Read("inspect_service", "svc_id", dep))
        if observed is None:
            return False, "upstream_receipt_missing"
        service = observed.data["service"]
        if service["status"] != "HEALTHY":
            return False, "upstream_not_healthy"
        params[dep] = service["active_params"]
    violation = check_upstream_compat(local["policy"], params)
    return (False, "upstream_incompatible") if violation else (True, "upstream_confirmed")


def _sources_v33():
    return {**_sources(), **{f"scripts/{name}": sha256_file(PROJECT_ROOT / "scripts" / name)
                             for name in ("research_v33_cases.py", Path(__file__).name)}}


def freeze(out):
    groups = {"accounting": accounting_cases(), "service": service_cases()}
    model = load_models_catalog().providers[PROVIDER].tiers["fast"].resolve_model_id()
    hashes = {}
    for cases in groups.values():
        for case in cases:
            env, _, messages = _checkpoint(case)
            with closing(env):
                hashes[case["id"]] = _json_digest({"messages": messages, "tools": env.tool_schemas()})
                for witness in case["witness"]:
                    if not env.execute_tool(witness["name"], witness["arguments"]).ok:
                        raise ValueError(f"{case['id']} has no registered valid completion witness.")
                if not env.verify(case["task"]).passed:
                    raise ValueError(f"{case['id']} has no verified terminal witness.")
    out.mkdir(parents=True)
    protocol = {"provider": PROVIDER, "fast_model": model, "case_hash": _json_digest(groups),
                "paired_prefix_sha256": hashes, "sources": _sources_v33(), "group_arms": ARMS, "maximum_caps_cny": CAPS,
                "model_decision_limit": 12, "tools_per_model_decision": 1, "transport": {"timeout_seconds": 120, "retries": 0},
                "continuation": "same Fast, public tools, common R and authorization gate; fixed seeded snapshots; no proactive review",
                "checkpoint_order": {k: [x["id"] for x in v] for k, v in groups.items()},
                "schedule": {k: [{"case": case["id"], "arm": arm} for i, case in enumerate(group)
                                 for arm in (ARMS[k] if i % 2 == 0 else tuple(reversed(ARMS[k])))]
                             for k, group in groups.items()}}
    _write_json(out / "protocol.json", protocol)
    _write_json(out / "cases.json", groups)
    return protocol


def _scripted(case, catalog, meter):
    index = 0

    def handler(_task, _tier, _step, messages, _tools):
        nonlocal index
        while index < len(case["witness"]):
            action = case["witness"][index]
            for i, message in enumerate(messages[:-1]):
                for tool_call in message.get("tool_calls", ()):
                    function = tool_call["function"]
                    following = messages[i + 1]
                    if (function["name"] == action["name"] and json.loads(function["arguments"]) == action["arguments"]
                            and following.get("tool_call_id") == tool_call["id"]
                            and json.loads(following["content"]).get("ok")):
                        break
                else:
                    continue
                break
            else:
                return {"tool_calls": [{**action, "id": f"scripted_{index}_{_step}"}]}
            index += 1
        return {"content": "Done"}

    return FakeModelPool(catalog, meter, handler)


def run_case(case, group, arm, pool, meter, cap, prefix_expected):
    env, control, messages = _checkpoint(case)
    task, case_id = case["task"], case["id"]
    schemas = env.tool_schemas()
    prefix_hash = _json_digest({"messages": messages, "tools": schemas})
    if prefix_hash != prefix_expected:
        env.close()
        raise ValueError("The public checkpoint changed after registration.")
    run_id = f"{case_id}__{arm}"
    meter.create_or_resume_run(run_id=run_id, stage=f"v33_{group}", split="development", policy=arm, provider=PROVIDER,
                               round_cap_cny=cap, stage_cap_cny=cap, metadata={"case": case_id, "prefix_hash": prefix_hash})
    meter.start_episode(episode_id=run_id, run_id=run_id, task_id=case_id, domain=task["domain"], difficulty="development")
    candidate = _spec(case["candidate"], "pending_checkpoint_candidate")
    steps, attempts = -1, 0
    trace, branch_decision, cost = [], None, Decimal()
    start = time.perf_counter()

    def dispatch(proposal):
        args = proposal.arguments_dict
        count_before = remaining(env, proposal.name, args)
        path_before = service_witness(env, task) if group == "service" and steps == -1 else None
        result = env.execute_tool(proposal.name, args)
        control.observe(proposal.name, args, result)
        _record(messages, proposal, result)
        count_after = remaining(env, proposal.name, args)
        evidence = {"step": steps, "event": "dispatch", "id": proposal.id, "tool": proposal.name, "arguments": args,
                    "reply": json.loads(result.to_tool_message()), "remaining_before": count_before,
                    "remaining_after": count_after, "contract_debit": contract_debit(proposal.name, result)}
        if count_before is not None and count_after is not None:
            evidence["actual_debit"] = count_before - count_after
        if path_before is not None:
            evidence["reachable_before"] = True
            evidence["reachable_after"] = service_witness(env, task) is not None
        trace.append(evidence)
        return result

    def no_review(_):
        raise AssertionError("No active model review in these comparisons.")

    if arm == "revalidate":
        allowed, branch_decision = retain_original(control, candidate, case["approval"], steps)
        if allowed:
            dispatch(candidate)
        else:
            messages.append({"role": "user", "content": "The pending proposal did not pass current validation. Replan using the current public evidence."})
    elif group == "service":
        revised, result = public_rule(control, candidate)
        branch_decision = result["decision"]
        allowed = revised is not None
        if allowed and arm == "dependency_R":
            allowed, branch_decision = check_dependency(control, candidate)
        if allowed:
            dispatch(revised)
        else:
            messages.append({"role": "user", "content": f"The pending restart was deferred: {branch_decision}. Continue from the public evidence."})
    else:
        branch_decision = "free_reproposal"
        messages.append({"role": "user", "content": "Replan using the current public evidence. Propose the next action."})
    verdict, reason = None, "decision_limit"
    try:
        for steps in range(task["max_steps"]):
            if env.verify(task).passed:
                reason = "business_goal_reached"
                break
            reply = pool.call_with_tools(run_id=run_id, episode_id=run_id, task_id=case_id, step_idx=steps,
                                         provider=PROVIDER, tier="fast", messages=messages, tools=schemas)
            attempts += 1
            cost += Decimal(str(reply.cost_cny))
            if not reply.tool_calls:
                reason = "model_stop" if reply.finish_reason != "length" else "output_limit"
                break
            proposed = reply.tool_calls[0]
            args = proposed.arguments_dict
            trace.append({"step": steps, "event": "proposal", "id": proposed.id, "tool": proposed.name, "arguments": args})
            schema = next((s["function"]["parameters"] for s in schemas if s["function"]["name"] == proposed.name), None)
            if args is None or schema is None or not set(schema.get("required", [])) <= args.keys() <= schema["properties"].keys():
                messages.append({"role": "user", "content": "The last tool proposal did not satisfy the public schema. Try a valid next action."})
                continue
            if any(type(value) is not {"integer": int, "string": str}[schema["properties"][key]["type"]] for key, value in args.items()):
                messages.append({"role": "user", "content": "A tool argument did not match the public type. Replan."})
                continue
            outcome = resolve_proposal(control, proposed, tier="fast", schemas=schemas, read=dispatch,
                                       review=no_review, log=lambda name, data: trace.append(
                                           {"step": steps, "event": name, "details": data}),
                                       can_review=False, rule=public_rule)
            selected = outcome.call
            if selected is None:
                messages.append({"role": "user", "content": outcome.feedback or "The pending action was deferred. Replan from current facts."})
            elif selected.name == "post_adjustment" and not _authorized(case["approval"], selected.arguments_dict, steps):
                trace.append({"step": steps, "event": "authorization_block", "arguments": selected.arguments_dict})
                messages.append({"role": "user", "content": "The current authorization does not cover that posting. Review the task and replan."})
            else:
                dispatch(selected)
        verdict = env.verify(task)
    except (APIError, BudgetExceededError) as exc:
        reason = type(exc).__name__
    finally:
        terminal = env.snapshot_state()
        env.close()
        passed = verdict.passed if verdict is not None else None
        elapsed = round(time.perf_counter() - start, 6)
        meter.finish_episode(episode_id=run_id, passed=bool(passed), failure_tag=None if passed else reason,
                             steps_taken=attempts, total_cost_cny=float(cost), loop_abort=False, summary={"elapsed_s": elapsed})
        meter.finish_run(run_id, status="COMPLETED" if passed is not None else "INTERRUPTED")
    original_posts = [x for x in trace if x["event"] == "dispatch" and x["tool"] == candidate.name
                      and x["arguments"] == candidate.arguments_dict and x["reply"]["ok"]]
    return {"case": case_id, "family": case["family"], "conditions": case["conditions"], "arm": arm,
            "public_prefix_sha256": prefix_hash, "checkpoint_disposition": branch_decision,
            "original_candidate_accepted": bool(original_posts), "trace": trace, "terminal": terminal,
            "verdict": asdict(verdict) if verdict is not None else None, "passed": passed, "stop": reason,
            "model_calls": attempts, "elapsed_s": elapsed}


def execute(frozen, group, out, *, budget=None):
    protocol, groups = (_read_json(frozen / name) for name in ("protocol.json", "cases.json"))
    if protocol["sources"] != _sources_v33() or protocol["case_hash"] != _json_digest(groups):
        raise ValueError("Frozen trial inputs differ from their registration.")
    if verify_manifest() or audit_orchestrator_ast():
        raise ValueError("Immutable environment validation failed.")
    catalog = load_models_catalog()
    provider = catalog.providers[PROVIDER]
    if protocol["fast_model"] != provider.tiers["fast"].resolve_model_id():
        raise ValueError("The live model is not the frozen configuration.")
    if budget is not None:
        qa = _read_json(frozen / f"{group}_offline" / "result.json")
        if not (0 < budget <= CAPS[group] and qa["scope"] == "scripted" and qa["protocol_sha256"] == _json_digest(protocol)
                and len(qa["branches"]) == 8 and get_api_key(provider)
                and get_base_url(provider).rstrip("/") == "https://dashscope.aliyuncs.com/compatible-mode/v1"):
            raise ValueError("A matching offline acceptance and separate authorized budget are required.")
    out.mkdir(parents=True)
    cap = budget or CAPS[group]
    scope = "live" if budget is not None else "scripted"
    _write_json(out / "protocol.json", {**protocol, "scope": scope, "group": group, "cap_cny": budget or 0})
    meter = BillingMeter(out / DATABASE, project_cap_cny=cap)
    live = LiveModelPool(catalog, meter, timeout_sec=120, max_retries=0) if budget is not None else None
    by_id = {c["id"]: c for c in groups[group]}
    branches = []
    try:
        for slot in protocol["schedule"][group]:
            case = by_id[slot["case"]]
            report = run_case(case, group, slot["arm"], live or _scripted(case, catalog, meter), meter, cap,
                              protocol["paired_prefix_sha256"][case["id"]])
            branches.append(report)
    finally:
        with closing(_read_database(out / DATABASE)) as db:
            attempts = list(db.execute("SELECT run_id,status,actual_cost_cny,estimated_max_cost_cny,model_actual FROM api_attempts"))
        fees = {branch["case"] + "__" + branch["arm"]: Decimal() for branch in branches}
        reserve = Decimal()
        for item in attempts:
            if item["run_id"] in fees:
                fees[item["run_id"]] += Decimal(str(item["actual_cost_cny"]))
            if item["status"] in ("PENDING", "UNKNOWN_BILLING"):
                reserve += Decimal(str(item["estimated_max_cost_cny"]))
        for b in branches:
            b["charge_cny"] = str(fees[b["case"]+"__"+b["arm"]]) if live else "0"
        report = {"scope": scope, "group": group, "protocol_sha256": _json_digest(protocol), "branches": branches,
                  "attempts": Counter(x["status"] for x in attempts), "models": sorted({x["model_actual"] for x in attempts if x["model_actual"]}),
                  "charge_cny": str(sum(Decimal(str(item["actual_cost_cny"])) for item in attempts)) if live else "0",
                  "reserve_cny": str(reserve) if live else "0"}
        _write_json(out / "result.json", report)
    if scope == "scripted":
        for b in branches:
            expected = not (group == "service" and b["arm"] == "local_R" and b["conditions"] == "blocked")
            if b["passed"] != expected:
                raise ValueError(f"The scripted acceptance disagrees at {b['case']} {b['arm']}.")
            if group == "accounting" and (b["original_candidate_accepted"] != (b["conditions"] == "stable")):
                raise ValueError("The scripted freshness control failed.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", required=True, type=Path)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--group", choices=sorted(ARMS))
    parser.add_argument("--live-out", type=Path)
    parser.add_argument("--budget-cny", type=float)
    args = parser.parse_args()
    if args.freeze:
        state = freeze(args.frozen)
        print(_json_digest(state))
        return
    if not args.group or bool(args.live_out) != (args.budget_cny is not None):
        parser.error("Supply a group and both live arguments for a paid run.")
    r = execute(args.frozen, args.group, args.live_out or args.frozen / f"{args.group}_offline", budget=args.budget_cny)
    print({**{k: v for k, v in r.items() if k != "branches"}, "outcomes": [
        (b["case"], b["arm"], b["passed"], b["checkpoint_disposition"]) for b in r["branches"]]})


if __name__ == "__main__":
    main()
