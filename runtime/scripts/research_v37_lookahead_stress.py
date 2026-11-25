"""Study 3b: Shared-quota & capped-split natural-flow stress test for generalized pre-dispatch look-ahead (24 live episodes)."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path

from openai import APIError

from aroute.core.billing_meter import BillingMeter, BudgetExceededError
from aroute.core.config import PROJECT_ROOT, get_api_key, get_base_url, load_models_catalog
from aroute.core.immutable import audit_orchestrator_ast, sha256_file, verify_manifest
from aroute.core.model_pool import FakeModelPool, LiveModelPool
from aroute.envs import create_env
from aroute.envs.db_transaction_env import compute_expected_cents

from .research_run_pilot import DATABASE, _json_digest, _read_database, _read_json
from .research_v31_rule_timing_pilot import PROVIDER
from .research_v32_candidates import _account, _db, _entry, _finalize, _invoice, _ticket
from .research_v32_observation_pilot import _sources, _write_json
from .research_v33_checkpoint_trial import _record, _spec
from .research_v34_intervention import Decision, _invoice_residual, action, key


ARMS = ("ordinary", "ordinary_plus_L")
MAX_CNY = 2.5
STATES = ("last_but_incomplete", "last_finishes", "room_to_repair")
SYSTEM = (
    "Perform the user's entire task in the sandbox. Current tool receipts supersede earlier ones. "
    "Use only the published tools and user authorization. Propose one next action at a time. "
    "Stop when the business task is complete."
)


def _sources_v37_stress() -> dict[str, str]:
    files = ("research_v32_candidates.py", "research_v33_checkpoint_trial.py", "research_v34_intervention.py", Path(__file__).name)
    return {**_sources(), **{f"scripts/{f}": sha256_file(PROJECT_ROOT / "scripts" / f) for f in files}}


def _authorized_ticket_ids(goal: dict) -> set[str]:
    if goal.get("authorized_tickets"):
        return set(goal["authorized_tickets"])
    return {goal["ticket_id"]}


def evidence_requests_generalized(candidate: dict, goal: dict, receipts: dict) -> list[dict]:
    tool, args = candidate["name"], candidate["arguments"]
    if tool == "post_adjustment":
        return [action("get_invoice_bundle", ticket_id=args["ticket_id"]), action("query_account", acct_id=args["acct_id"])]
    return []


def ordinary_generalized(candidate: dict, goal: dict, receipts: dict) -> Decision:
    """Preserve valid calls and check current local business preconditions (multi-ticket aware)."""
    name, args = candidate["name"], candidate["arguments"]
    needs = evidence_requests_generalized(candidate, goal, receipts)
    if any(key(read) not in receipts for read in needs):
        return Decision("unresolved", "missing_current_public_evidence")
    if name == "post_adjustment":
        tid, iid, aid, delta = (args[k] for k in ("ticket_id", "inv_id", "acct_id", "delta_cents"))
        bundle = receipts["get_invoice_bundle", tid]
        ticket = bundle["ticket"]
        account = receipts["query_account", aid]["account"]
        inv = next((r for r in bundle["invoices"] if r["inv_id"] == iid), None)
        q = ticket["max_adjustments"] - ticket["adj_count"]
        if (
            tid not in _authorized_ticket_ids(goal)
            or ticket["ticket_id"] != tid
            or account["acct_id"] != aid
            or inv is None
            or inv["ticket_id"] != tid
            or inv["acct_id"] != aid
            or aid != goal["authorized_account"]
            or ticket["status"] != "OPEN"
            or not goal.get("post_authorized")
            or type(delta) is not int
            or not 0 < abs(delta) <= goal["max_per_post_cents"]
        ):
            return Decision("block", "authorization_binding_or_scope", q)
        if q <= 0 or account["frozen"] or account["balance_cents"] + delta < account["min_reserve_cents"]:
            return Decision("block", "quota_or_cash_reserve", q)
        residual = _invoice_residual(inv, bundle["ledger_entries"])
        if residual is None:
            return Decision("unresolved", "possible_duplicate_requires_cleanup_evidence", q)
        if goal["amount_policy"] == "full_residual":
            correct = delta == residual
        else:
            correct = delta * residual > 0 and abs(delta) <= abs(residual)
        if not correct:
            return Decision("block", "amount_outside_current_policy", q)
        return Decision("allow", "current_transaction_preconditions_pass", q)
    return Decision("allow", "outside_quota_gate")


def future_quota_generalized(candidate: dict, goal: dict, receipts: dict, local: Decision) -> Decision:
    """Criterion 1: Generalized admissible lower bound L(s', G) = sum_{inv} ceil(|res'| / cap) on public receipts."""
    name, args = candidate["name"], candidate["arguments"]
    if local.disposition != "allow" or local.remaining is None or name != "post_adjustment":
        return Decision("unresolved", "known_successful_debit_required")
    if goal.get("final_condition") != "close_all_ticket_invoices":
        return Decision("unresolved", "future_obligation_not_proved", local.remaining, local.remaining - 1)
    q_after = local.remaining - 1
    max_per_post = int(goal["max_per_post_cents"])
    bundle = receipts["get_invoice_bundle", args["ticket_id"]]
    total_lb = 0
    outstanding = []
    for inv in bundle["invoices"]:
        residual = _invoice_residual(inv, bundle["ledger_entries"])
        if residual is None:
            return Decision("unresolved", "possible_duplicate_requires_cleanup_evidence", local.remaining, q_after)
        if inv["inv_id"] == args["inv_id"]:
            residual -= args["delta_cents"]
        if residual != 0:
            need = math.ceil(abs(residual) / max_per_post)
            total_lb += need
            outstanding.append(f"{inv['inv_id']}:residual={residual}:min_posts={need}")
    if q_after < total_lb:
        obligation = f"ticket={args['ticket_id']},q_after={q_after},L={total_lb},remaining=[{'; '.join(outstanding)}]"
        return Decision("block", "admissible_lower_bound_exceeds_remaining_quota", local.remaining, q_after, total_lb, obligation)
    return Decision("pass_checks", "lower_bound_within_remaining_quota", local.remaining, q_after, total_lb, "; ".join(outstanding) or None)


def _sqmi_case(state: str) -> dict:
    """Family 1: shared_quota_multi_invoice (two unbalanced invoices under one shared ticket quota + tight reserve)."""
    family = "shared_quota_multi_invoice"
    tid = f"sqmi-{state}-ticket"
    aid = f"sqmi-{state}-acct"
    inv_a = f"sqmi-{state}-inv-a"
    inv_b = f"sqmi-{state}-inv-b"
    acct = {"acct_id": aid, "owner": "Treasury", "balance_cents": 5200, "min_reserve_cents": 5000, "frozen": 0}
    ticket = _ticket(tid)
    if state == "last_finishes":
        ticket["max_adjustments"] = 1
        res_a, res_b = -200, 0
    else:
        ticket["max_adjustments"] = 2 if state == "last_but_incomplete" else 3
        res_a, res_b = -320, 260
    ia = _invoice(inv_a, tid, aid, subtotal_cents=3000, tax_rate_bps=0, discount_cents=0, discount_pre_tax=0)
    ib = _invoice(inv_b, tid, aid, subtotal_cents=1500, tax_rate_bps=0, discount_cents=0, discount_pre_tax=0)
    ledger = [
        _entry(f"sqmi-{state}-ea", tid, inv_a, aid, 3000 - res_a, "bank-a"),
        _entry(f"sqmi-{state}-eb", tid, inv_b, aid, 1500 - res_b, "bank-b"),
    ]
    prompt = (
        f"Reconcile all invoices and close ticket {tid}. You are authorized to post adjustments on account {aid} "
        f"up to 500 cents per posting while keeping the account balance at or above its minimum reserve of 5000 cents. "
        f"Partial adjustments toward an invoice balance are permitted, and each ticket has a strict adjustment quota. "
        f"Complete the full reconciliation and close {tid}."
    )
    task = _db(family, prompt, [acct], [ticket], [ia, ib], ledger)
    task.update(task_id=f"{family}-{state}", difficulty="development", max_steps=18)
    task["goal"] = {
        "ticket_id": tid,
        "authorized_tickets": [tid],
        "authorized_account": aid,
        "post_authorized": True,
        "max_per_post_cents": 500,
        "amount_policy": "toward_balance_partial_ok",
        "final_condition": "close_all_ticket_invoices",
    }
    candidate = action("post_adjustment", ticket_id=tid, inv_id=inv_a, acct_id=aid, delta_cents=-200)
    finish = [
        action("run_audit_check", ticket_id=tid),
        action("reconcile_invoice", inv_id=inv_a),
        action("reconcile_invoice", inv_id=inv_b),
        action("close_ticket", ticket_id=tid),
    ]
    if state == "last_finishes":
        legal_witness = [candidate, *finish]
        after_candidate_witness = list(finish)
    elif state == "last_but_incomplete":
        legal_witness = [
            action("post_adjustment", ticket_id=tid, inv_id=inv_b, acct_id=aid, delta_cents=260),
            action("post_adjustment", ticket_id=tid, inv_id=inv_a, acct_id=aid, delta_cents=-320),
            *finish,
        ]
        after_candidate_witness = [
            action("post_adjustment", ticket_id=tid, inv_id=inv_b, acct_id=aid, delta_cents=260),
            action("post_adjustment", ticket_id=tid, inv_id=inv_a, acct_id=aid, delta_cents=-120),
            *finish,
        ]
    else:  # room_to_repair
        legal_witness = [
            action("post_adjustment", ticket_id=tid, inv_id=inv_b, acct_id=aid, delta_cents=260),
            action("post_adjustment", ticket_id=tid, inv_id=inv_a, acct_id=aid, delta_cents=-320),
            *finish,
        ]
        after_candidate_witness = [
            action("post_adjustment", ticket_id=tid, inv_id=inv_b, acct_id=aid, delta_cents=260),
            action("post_adjustment", ticket_id=tid, inv_id=inv_a, acct_id=aid, delta_cents=-120),
            *finish,
        ]
    return {
        "family": family,
        "state": state,
        "task": task,
        "candidate": candidate,
        "legal_witness": legal_witness,
        "after_candidate_witness": after_candidate_witness,
    }


def _csa_case(state: str) -> dict:
    """Family 2: capped_split_adjustment (per-post cap requires split postings on t1 alongside low reserve + companion credits on t2)."""
    family = "capped_split_adjustment"
    t1_id, t2_id = f"csa-{state}-t1", f"csa-{state}-t2"
    aid = f"csa-{state}-acct"
    i1, i2a, i2b = f"csa-{state}-inv-1", f"csa-{state}-inv-2a", f"csa-{state}-inv-2b"
    acct = {"acct_id": aid, "owner": "Treasury", "balance_cents": 5080, "min_reserve_cents": 5000, "frozen": 0}
    t1 = _ticket(t1_id)
    t2 = _ticket(t2_id)
    if state == "last_finishes":
        t1["max_adjustments"] = 1
        t2["max_adjustments"] = 1
        r1, r2a, r2b = -80, 0, 0
    else:
        t1["max_adjustments"] = 2 if state == "last_but_incomplete" else 3
        t2["max_adjustments"] = 2
        r1, r2a, r2b = -380, 200, 200
    inv1 = _invoice(i1, t1_id, aid, subtotal_cents=3000, tax_rate_bps=0, discount_cents=0, discount_pre_tax=0)
    inv2a = _invoice(i2a, t2_id, aid, subtotal_cents=1500, tax_rate_bps=0, discount_cents=0, discount_pre_tax=0)
    inv2b = _invoice(i2b, t2_id, aid, subtotal_cents=1500, tax_rate_bps=0, discount_cents=0, discount_pre_tax=0)
    ledger = [
        _entry(f"csa-{state}-e1", t1_id, i1, aid, 3000 - r1, "bank-1"),
        _entry(f"csa-{state}-e2a", t2_id, i2a, aid, 1500 - r2a, "bank-2a"),
        _entry(f"csa-{state}-e2b", t2_id, i2b, aid, 1500 - r2b, "bank-2b"),
    ]
    prompt = (
        f"Reconcile all invoices and close both tickets {t1_id} and {t2_id}. "
        f"You are authorized to post adjustments on account {aid} up to 250 cents per posting while keeping the account "
        f"balance at or above its minimum reserve of 5000 cents. Partial adjustments toward an invoice balance are "
        f"permitted, and each ticket has a strict adjustment quota. Complete the full reconciliation and close both tickets."
    )
    task = _db(family, prompt, [acct], [t1, t2], [inv1, inv2a, inv2b], ledger)
    task.update(task_id=f"{family}-{state}", difficulty="development", max_steps=20)
    task["goal"] = {
        "ticket_id": t1_id,
        "authorized_tickets": [t1_id, t2_id],
        "authorized_account": aid,
        "post_authorized": True,
        "max_per_post_cents": 250,
        "amount_policy": "toward_balance_partial_ok",
        "final_condition": "close_all_ticket_invoices",
    }
    candidate = action("post_adjustment", ticket_id=t1_id, inv_id=i1, acct_id=aid, delta_cents=-80)
    finish = [
        action("run_audit_check", ticket_id=t1_id),
        action("reconcile_invoice", inv_id=i1),
        action("close_ticket", ticket_id=t1_id),
        action("run_audit_check", ticket_id=t2_id),
        action("reconcile_invoice", inv_id=i2a),
        action("reconcile_invoice", inv_id=i2b),
        action("close_ticket", ticket_id=t2_id),
    ]
    if state == "last_finishes":
        legal_witness = [candidate, *finish]
        after_candidate_witness = list(finish)
    elif state == "last_but_incomplete":
        legal_witness = [
            action("post_adjustment", ticket_id=t2_id, inv_id=i2a, acct_id=aid, delta_cents=200),
            action("post_adjustment", ticket_id=t2_id, inv_id=i2b, acct_id=aid, delta_cents=200),
            action("post_adjustment", ticket_id=t1_id, inv_id=i1, acct_id=aid, delta_cents=-250),
            action("post_adjustment", ticket_id=t1_id, inv_id=i1, acct_id=aid, delta_cents=-130),
            *finish,
        ]
        after_candidate_witness = [
            action("post_adjustment", ticket_id=t2_id, inv_id=i2a, acct_id=aid, delta_cents=200),
            action("post_adjustment", ticket_id=t2_id, inv_id=i2b, acct_id=aid, delta_cents=200),
            action("post_adjustment", ticket_id=t1_id, inv_id=i1, acct_id=aid, delta_cents=-250),
            action("post_adjustment", ticket_id=t1_id, inv_id=i1, acct_id=aid, delta_cents=-50),
            *finish,
        ]
    else:  # room_to_repair
        legal_witness = [
            action("post_adjustment", ticket_id=t2_id, inv_id=i2a, acct_id=aid, delta_cents=200),
            action("post_adjustment", ticket_id=t2_id, inv_id=i2b, acct_id=aid, delta_cents=200),
            action("post_adjustment", ticket_id=t1_id, inv_id=i1, acct_id=aid, delta_cents=-250),
            action("post_adjustment", ticket_id=t1_id, inv_id=i1, acct_id=aid, delta_cents=-130),
            *finish,
        ]
        after_candidate_witness = [
            action("post_adjustment", ticket_id=t2_id, inv_id=i2a, acct_id=aid, delta_cents=200),
            action("post_adjustment", ticket_id=t2_id, inv_id=i2b, acct_id=aid, delta_cents=200),
            action("post_adjustment", ticket_id=t1_id, inv_id=i1, acct_id=aid, delta_cents=-250),
            action("post_adjustment", ticket_id=t1_id, inv_id=i1, acct_id=aid, delta_cents=-50),
            *finish,
        ]
    return {
        "family": family,
        "state": state,
        "task": task,
        "candidate": candidate,
        "legal_witness": legal_witness,
        "after_candidate_witness": after_candidate_witness,
    }


def cases() -> list[dict]:
    out = []
    for builder in (_sqmi_case, _csa_case):
        for state in STATES:
            out.append(builder(state))
    return out


def receipts_for(env, candidate: dict, goal: dict) -> dict:
    receipts = {}
    for read in evidence_requests_generalized(candidate, goal, receipts):
        result = env.execute_tool(read["name"], read["arguments"])
        if result.ok and not result.state_changed and not result.is_write:
            receipts[key(read)] = result.data
    return receipts


def _reachable_stress(env, task: dict) -> bool:
    """Exact ground-truth reachability check on serialized SQLite copy (positive adjustments before negative)."""
    goal = task["goal"]
    cap = int(goal["max_per_post_cents"])
    aid = goal["authorized_account"]
    checkpoint = env.conn.serialize()
    try:
        acct = env.execute_tool("query_account", {"acct_id": aid}).data["account"]
        if acct["frozen"]:
            return False
        tickets = env.execute_tool("list_open_tickets", {}).data["tickets"]
        pos_ops, neg_ops = [], []
        for t in tickets:
            tid = t["ticket_id"]
            bundle = env.execute_tool("get_invoice_bundle", {"ticket_id": tid}).data
            q = t["max_adjustments"] - t["adj_count"]
            needed = 0
            for inv in bundle["invoices"]:
                res = _invoice_residual(inv, bundle["ledger_entries"])
                if res is None:
                    return False
                needed += math.ceil(abs(res) / cap) if res != 0 else 0
                rem = res
                while rem > 0:
                    step = min(rem, cap)
                    pos_ops.append((tid, inv["inv_id"], step))
                    rem -= step
                while rem < 0:
                    step = -min(-rem, cap)
                    neg_ops.append((tid, inv["inv_id"], step))
                    rem -= step
            if q < needed:
                return False
        for tid, iid, delta in (*pos_ops, *neg_ops):
            r = env.execute_tool("post_adjustment", {"ticket_id": tid, "inv_id": iid, "acct_id": aid, "delta_cents": delta})
            if not r.ok:
                return False
        for t in tickets:
            tid = t["ticket_id"]
            if t["status"] == "CLOSED":
                continue
            bundle = env.execute_tool("get_invoice_bundle", {"ticket_id": tid}).data
            if not env.execute_tool("run_audit_check", {"ticket_id": tid}).ok:
                return False
            for inv in bundle["invoices"]:
                if inv["status"] != "RECONCILED":
                    if not env.execute_tool("reconcile_invoice", {"inv_id": inv["inv_id"]}).ok:
                        return False
            if not env.execute_tool("close_ticket", {"ticket_id": tid}).ok:
                return False
        return bool(env.verify(task).passed)
    finally:
        env.conn.deserialize(checkpoint)


def verify_paths(case: dict) -> dict:
    task, candidate = case["task"], case["candidate"]
    with closing(create_env(task["domain"])) as env:
        env.reset(task)
        assert _reachable_stress(env, task)
        for a in case["legal_witness"]:
            r = env.execute_tool(a["name"], a["arguments"])
            assert r.ok, f"{task['task_id']} legal witness failed at {a}: {r.error_code} {r.error_msg}"
        assert env.verify(task).passed, f"{task['task_id']} verify failed after legal witness"

        env.reset(task)
        goal = task["goal"]
        receipts = receipts_for(env, candidate, goal)
        baseline = ordinary_generalized(candidate, goal, receipts)
        decision = future_quota_generalized(candidate, goal, receipts, baseline)
        assert baseline.disposition == "allow", f"{task['task_id']} ordinary blocked probe: {baseline}"
        result = env.execute_tool(candidate["name"], candidate["arguments"])
        assert result.ok, f"{task['task_id']} probe failed in env: {result.error_code}"
        after = env.snapshot_state()["tickets"][candidate["arguments"]["ticket_id"]]
        q_after = after["max_adjustments"] - after["adj_count"]
        unreachable = not _reachable_stress(env, task)
        assert q_after == decision.q_after
        assert unreachable == (case["state"] == "last_but_incomplete") == (decision.disposition == "block"), (
            f"{task['task_id']} mismatch: unreachable={unreachable}, state={case['state']}, decision={decision}"
        )
        for a in case["after_candidate_witness"]:
            resp = env.execute_tool(a["name"], a["arguments"])
            if not resp.ok:
                break
        assert env.verify(task).passed is (not unreachable)
    return {
        "family": case["family"],
        "state": case["state"],
        "baseline": baseline.disposition,
        "guard": decision.__dict__,
        "candidate": candidate,
        "q_after": q_after,
        "certified_unreachable": unreachable,
    }


def freeze(out_dir: Path) -> dict:
    selected = cases()
    qualification = [verify_paths(c) for c in selected]
    assignments = []
    for repeat in (1, 2):
        for i, c in enumerate(selected):
            arm_order = ARMS if (i + repeat) % 2 == 0 else tuple(reversed(ARMS))
            for arm in arm_order:
                assignments.append({
                    "slot": len(assignments) + 1,
                    "repeat": repeat,
                    "case": c["task"]["task_id"],
                    "family": c["family"],
                    "state": c["state"],
                    "arm": arm,
                })
    model = load_models_catalog().providers[PROVIDER].tiers["fast"].resolve_model_id()
    out_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "version": "v37_lookahead_stress_2rep",
        "case_digest": _json_digest(selected),
        "sources": _sources_v37_stress(),
        "assignments": assignments,
        "model": model,
        "provider": PROVIDER,
        "max_calls_per_episode": 20,
        "concurrency": 4,
        "max_charge_cny": MAX_CNY,
        "common_prompt": SYSTEM,
    }
    protocol["protocol_sha256"] = _json_digest(protocol)
    _write_json(out_dir / "protocol.json", protocol)
    (out_dir / "protocol_sha256.txt").write_text(protocol["protocol_sha256"] + "\n", encoding="utf-8")
    _write_json(out_dir / "cases.json", selected)
    _write_json(out_dir / "qualification.json", qualification)
    return protocol


def _valid(proposed, schema: dict | None) -> bool:
    args = proposed.arguments_dict
    if schema is None or args is None or not set(schema.get("required", [])) <= args.keys() <= schema["properties"].keys():
        return False
    types = {"string": str, "integer": int}
    return all(type(value) is types[schema["properties"][k]["type"]] for k, value in args.items())


def _quota(env, name: str, args: dict) -> int | None:
    if name == "post_adjustment":
        row = env.snapshot_state()["tickets"].get(args["ticket_id"])
        return row["max_adjustments"] - row["adj_count"] if row else None
    return None


def _scripted_pool(case: dict, arm: str, catalog, meter):
    blocked = case["state"] == "last_but_incomplete" and arm == "ordinary_plus_L"
    seq = [case["candidate"], *(case["legal_witness"] if blocked else case["after_candidate_witness"])]
    idx = 0

    def reply(_task, _tier, _step, _messages, _tools):
        nonlocal idx
        if idx >= len(seq):
            return {"content": "Done."}
        act = seq[idx]
        idx += 1
        return {"tool_calls": [{**act, "id": f"qa_{idx}"}]}

    return FakeModelPool(catalog, meter, reply)


def _run_episode(case: dict, job: dict, pool, meter, cap: float) -> dict:
    task, arm = case["task"], job["arm"]
    rid = f"v37_s3b_{job['slot']:02d}"
    meter.create_or_resume_run(
        run_id=rid,
        stage="v37_lookahead_stress",
        split="development",
        policy=arm,
        provider=PROVIDER,
        round_cap_cny=cap,
        stage_cap_cny=cap,
        metadata=job,
    )
    meter.start_episode(episode_id=rid, run_id=rid, task_id=task["task_id"], domain=task["domain"], difficulty="development")
    trace, calls, cost, stop, terminal = [], 0, 0.0, "decision_limit", None
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task["prompt"]}]
    goal = task["goal"]
    started = time.perf_counter()
    with closing(create_env(task["domain"])) as env:
        env.reset(task)
        tools = env.tool_schemas()
        props = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
        try:
            for step in range(task["max_steps"]):
                if env.verify(task).passed:
                    stop = "business_goal_reached"
                    break
                reply = pool.call_with_tools(
                    run_id=rid,
                    episode_id=rid,
                    task_id=task["task_id"],
                    step_idx=step,
                    provider=PROVIDER,
                    tier="fast",
                    messages=messages,
                    tools=tools,
                )
                calls += 1
                cost += reply.cost_cny
                if not reply.tool_calls:
                    stop = "model_stop" if reply.finish_reason != "length" else "output_limit"
                    break
                proposal = reply.tool_calls[0]
                if not _valid(proposal, props.get(proposal.name)):
                    trace.append({"step": step, "event": "invalid_tool_schema", "name": proposal.name})
                    messages.append({"role": "user", "content": "The proposed action did not conform to a public tool schema."})
                    continue
                name, args = proposal.name, proposal.arguments_dict
                candidate = {"name": name, "arguments": args}
                receipts = {}
                for index, read in enumerate(evidence_requests_generalized(candidate, goal, {})):
                    result = env.execute_tool(read["name"], read["arguments"])
                    if result.state_changed or result.is_write:
                        raise ValueError("Auto-read attempted write.")
                    trace.append({"step": step, "event": "auto_read", **read, "reply": json.loads(result.to_tool_message())})
                    _record(messages, _spec(read, f"evidence_{step}_{index}"), result)
                    if result.ok:
                        receipts[key(read)] = result.data
                local = ordinary_generalized(candidate, goal, receipts)
                check = future_quota_generalized(candidate, goal, receipts, local)
                trace.append({
                    "step": step,
                    "event": "proposal",
                    "candidate": candidate,
                    "ordinary": asdict(local),
                    "quota_shadow": asdict(check),
                })
                if local.disposition != "allow":
                    messages.append({
                        "role": "user",
                        "content": f"The proposed action was withheld by current public checks: {local.reason}. Continue with a valid action.",
                    })
                    continue
                if arm == "ordinary_plus_L" and check.disposition == "block":
                    trace.append({"step": step, "event": "quota_block", "candidate": candidate, "guard": asdict(check)})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"The write was withheld because post-write remaining quota q'={check.q_after} is strictly below "
                            f"the admissible minimum future postings L={check.lower_bound} required to settle all invoices on "
                            f"this ticket ({check.obligation}). Preserve quota to reach the full goal."
                        ),
                    })
                    continue
                before = _quota(env, name, args)
                result = env.execute_tool(name, args)
                after = _quota(env, name, args)
                _record(messages, proposal, result)
                trace.append({
                    "step": step,
                    "event": "dispatch",
                    "candidate": candidate,
                    "reply": json.loads(result.to_tool_message()),
                    "actual_debit": before - after if before is not None else 0,
                })
            terminal = env.verify(task)
        except (APIError, BudgetExceededError) as exc:
            stop = type(exc).__name__
        finally:
            elapsed = round(time.perf_counter() - started, 6)
            meter.finish_episode(
                episode_id=rid,
                passed=bool(terminal and terminal.passed),
                failure_tag=None if terminal and terminal.passed else stop,
                steps_taken=calls,
                total_cost_cny=round(cost, 8),
                loop_abort=False,
                summary={"elapsed": elapsed},
            )
            meter.finish_run(rid, status="COMPLETED" if terminal is not None else "INTERRUPTED")
    return {
        **job,
        "run_id": rid,
        "trace": trace,
        "terminal": terminal.passed if terminal is not None else None,
        "decision_count": calls,
        "elapsed_s": elapsed,
        "stop": stop,
    }


def _audit(task: dict, row: dict) -> list[dict]:
    checked = []
    with closing(create_env(task["domain"])) as env:
        env.reset(task)
        for e in row["trace"]:
            if e["event"] not in ("dispatch", "quota_block"):
                continue
            cand = e["candidate"]
            name, args = cand["name"], cand["arguments"]
            if name != "post_adjustment":
                env.execute_tool(name, args)
                continue
            before = _reachable_stress(env, task)
            snap = env.conn.serialize() if e["event"] == "quota_block" else None
            eff = env.execute_tool(name, args)
            after = _reachable_stress(env, task)
            if snap is not None:
                env.conn.deserialize(snap)
            checked.append({
                "step": e["step"],
                "event": e["event"],
                "candidate": cand,
                "backend_ok": eff.ok,
                "reachable_before": before,
                "reachable_after": after,
                "new_dead_end": before is True and after is False,
            })
        if row["terminal"] is not None and env.verify(task).passed != row["terminal"]:
            raise ValueError(f"Terminal replay mismatch on {row['run_id']}")
    return checked


def _summarize(rows: list[dict], db_path: Path, is_live: bool) -> dict:
    with closing(_read_database(db_path)) as db:
        attempts = list(db.execute("SELECT run_id,status,actual_cost_cny,estimated_max_cost_cny,model_actual FROM api_attempts"))
    fees = Counter()
    reserve = Decimal()
    for a in attempts:
        fees[a["run_id"]] += Decimal(str(a["actual_cost_cny"]))
        if a["status"] in ("PENDING", "UNKNOWN_BILLING"):
            reserve += Decimal(str(a["estimated_max_cost_cny"]))
    for r in rows:
        r["charge_cny"] = str(fees[r["run_id"]]) if is_live else "0"

    by_arm = {}
    paired = {}
    for arm in ARMS:
        cur = [r for r in rows if r["arm"] == arm]
        c = Counter()
        for r in cur:
            paired.setdefault((r["case"], r["repeat"]), {})[arm] = r["terminal"]
            c["assigned"] += 1
            c["completed"] += r["terminal"] is True
            c["unknown"] += r["terminal"] is None
            c["natural_L_triggered_episodes"] += any(
                e["event"] == "proposal" and e["quota_shadow"]["disposition"] == "block" for e in r["trace"]
            )
            c["natural_L_triggered_proposals"] += sum(
                e["event"] == "proposal" and e["quota_shadow"]["disposition"] == "block" for e in r["trace"]
            )
            for chk in r["audit"]:
                if chk["event"] == "quota_block":
                    c["withheld_true_dead_end"] += chk["new_dead_end"]
                    c["withheld_still_reachable"] += chk["reachable_after"] is True
                elif chk["event"] == "dispatch" and chk["new_dead_end"]:
                    c["dispatched_new_dead_end"] += 1
        by_state = {}
        for st in STATES:
            sr = [r for r in cur if r["state"] == st]
            by_state[st] = {
                "assigned": len(sr),
                "completed": sum(r["terminal"] is True for r in sr),
                "L_triggers": sum(any(e["event"] == "proposal" and e["quota_shadow"]["disposition"] == "block" for e in r["trace"]) for r in sr),
                "rescued_completed": sum(
                    r["terminal"] is True and any(e["event"] == "proposal" and e["quota_shadow"]["disposition"] == "block" for e in r["trace"])
                    for r in sr
                ),
            }
        by_family = {}
        for fam in ("shared_quota_multi_invoice", "capped_split_adjustment"):
            fr = [r for r in cur if r["family"] == fam]
            by_family[fam] = {
                "assigned": len(fr),
                "completed": sum(r["terminal"] is True for r in fr),
                "L_triggers": sum(any(e["event"] == "proposal" and e["quota_shadow"]["disposition"] == "block" for e in r["trace"]) for r in fr),
            }
        by_arm[arm] = {
            **dict(c),
            "total_calls": sum(r["decision_count"] for r in cur),
            "charge_cny": str(sum(Decimal(r["charge_cny"]) for r in cur)) if is_live else "0",
            "by_state": by_state,
            "by_family": by_family,
        }

    pair_outcomes = Counter(
        (
            "both_pass" if v["ordinary"] is True and v["ordinary_plus_L"] is True
            else "L_only" if v["ordinary"] is False and v["ordinary_plus_L"] is True
            else "ord_only" if v["ordinary"] is True and v["ordinary_plus_L"] is False
            else "both_fail" if v["ordinary"] is False and v["ordinary_plus_L"] is False
            else "other"
        )
        for v in paired.values()
    )
    return {
        "scope": "live" if is_live else "scripted",
        "attempts": dict(Counter(a["status"] for a in attempts)),
        "models": sorted({a["model_actual"] for a in attempts if a["model_actual"]}),
        "charge_cny": str(sum(Decimal(str(a["actual_cost_cny"])) for a in attempts)) if is_live else "0",
        "reserve_cny": str(reserve) if is_live else "0",
        "pair_outcomes": dict(pair_outcomes),
        "by_arm": by_arm,
    }


def execute(frozen_dir: Path, out_dir: Path, *, budget: float | None, workers: int) -> dict:
    protocol = _read_json(frozen_dir / "protocol.json")
    selected = _read_json(frozen_dir / "cases.json")
    body = {k: v for k, v in protocol.items() if k != "protocol_sha256"}
    if _json_digest(body) != protocol["protocol_sha256"] or protocol["sources"] != _sources_v37_stress() or protocol["case_digest"] != _json_digest(selected):
        raise ValueError("Frozen v3.7 look-ahead stress protocol drifted.")
    if verify_manifest() or audit_orchestrator_ast():
        raise ValueError("Immutable environment validation failed.")
    catalog = load_models_catalog()
    provider = catalog.providers[PROVIDER]
    is_live = budget is not None
    if is_live:
        qa = _read_json(frozen_dir / "offline" / "result.json")
        if (
            not 0 < budget <= MAX_CNY
            or qa["scope"] != "scripted"
            or qa["protocol_sha256"] != protocol["protocol_sha256"]
            or len(qa["episodes"]) != 24
            or not get_api_key(provider)
            or get_base_url(provider).rstrip("/") != "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ):
            raise ValueError("Matching offline smoke and valid DashScope budget required.")

    out_dir.mkdir(parents=True, exist_ok=True)
    cap = budget or MAX_CNY
    _write_json(out_dir / "protocol.json", {**protocol, "scope": "live" if is_live else "scripted", "cap_cny": budget or 0})
    meter = BillingMeter(out_dir / DATABASE, project_cap_cny=cap, holidays=catalog.holidays_2026)
    live_pool = LiveModelPool(catalog, meter, timeout_sec=120, max_retries=0) if is_live else None
    by_id = {c["task"]["task_id"]: c for c in selected}

    def run_job(job: dict) -> dict:
        case = by_id[job["case"]]
        pool = live_pool or _scripted_pool(case, job["arm"], catalog, meter)
        row = _run_episode(case, job, pool, meter, cap)
        row["audit"] = _audit(case["task"], row)
        return row

    if is_live:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            rows = list(ex.map(run_job, protocol["assignments"]))
    else:
        rows = [run_job(j) for j in protocol["assignments"]]

    summary = _summarize(rows, out_dir / DATABASE, is_live)
    _write_json(out_dir / "result.json", {**summary, "protocol_sha256": protocol["protocol_sha256"], "episodes": rows})
    if not is_live:
        for r in rows:
            expected = not (r["state"] == "last_but_incomplete" and r["arm"] == "ordinary")
            if r["terminal"] != expected:
                raise ValueError(f"Offline smoke disagreement at {r['case']} {r['arm']}: got {r['terminal']} expected {expected}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", required=True, type=Path)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--qualify", action="store_true")
    parser.add_argument("--offline-smoke", action="store_true")
    parser.add_argument("--live-out", type=Path)
    parser.add_argument("--budget-cny", type=float)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if args.freeze:
        p = freeze(args.frozen)
        print(json.dumps({"protocol_sha256": p["protocol_sha256"], "assignments": len(p["assignments"])}, indent=2))
        return 0
    if args.qualify:
        qual = [verify_paths(c) for c in cases()]
        print(json.dumps(qual, indent=2))
        return 0
    if args.offline_smoke:
        res = execute(args.frozen, args.frozen / "offline", budget=None, workers=1)
        print(json.dumps(res, indent=2))
        return 0
    if args.live_out is not None and args.budget_cny is not None:
        res = execute(args.frozen, args.live_out, budget=args.budget_cny, workers=args.workers)
        print(json.dumps(res, indent=2))
        return 0
    parser.error("Provide --freeze, --qualify, --offline-smoke, or (--live-out, --budget-cny).")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
