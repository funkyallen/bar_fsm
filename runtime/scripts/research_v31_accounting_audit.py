"""Read-only v3.1 replay of the prespecified accounting cases and billing ledger."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from contextlib import closing
from decimal import Decimal
from pathlib import Path

from aroute.envs import create_env
from aroute.envs.db_transaction_env import compute_expected_cents

from .research_run_pilot import DATABASE, PILOT_RUNS, _json_digest, _read_database, _read_json


PAIRS = ((43, 44), (69, 70), (79, 73))


def ticket_facts(env, ticket_id):
    bundle = env.execute_tool("get_invoice_bundle", {"ticket_id": ticket_id})
    if not bundle.ok:
        raise ValueError(f"Unknown ticket in replay: {ticket_id}")
    ticket = bundle.data["ticket"]
    differences = {}
    for invoice in bundle.data["invoices"]:
        inv_id = invoice["inv_id"]
        expected = compute_expected_cents(**{field: invoice[field] for field in (
            "subtotal_cents", "tax_rate_bps", "discount_cents", "discount_pre_tax", "late_fee_bps", "days_late",
        )})
        observed = sum(row["amount_cents"] for row in bundle.data["ledger_entries"]
                       if row["inv_id"] == inv_id and not row["voided"])
        differences[inv_id] = expected - observed
    return {"status": ticket["status"], "audit_passed": ticket["audit_passed"], "used": ticket["adj_count"],
            "remaining": ticket["max_adjustments"] - ticket["adj_count"], "invoice_gap_cents": differences}


def matched_terminal(db, protocol, slot):
    job = protocol["jobs"][slot - 1]
    assert job["slot"] == slot and job["domain"] == "db_transaction"
    rows = db.execute("SELECT r.run_id,e.passed,e.failure_tag FROM runs r JOIN episodes e USING(run_id) "
                      "WHERE json_extract(r.metadata_json,'$.slot')=? AND e.status='COMPLETED'", (slot,)).fetchall()
    row, = rows
    return {**job, **dict(row)}


def replay(db, run, task):
    tickets = task["assertions"]["closed_tickets"]
    env = create_env(task["domain"])
    env.reset(task)
    pending_calls, confirmed, resolutions, generated = {}, {}, [], {}
    tool_trace, proposed, adjustments = [], [], []
    last_bundle = None
    with closing(env):
        for row in db.execute("SELECT * FROM step_events WHERE run_id=? ORDER BY event_id", (run["run_id"],)):
            tool = row["tool_name"]
            step = row["step_idx"]
            args = json.loads(row["tool_args_json"] or "{}")
            if tool == "__V3_PREWRITE__":
                label = args["event"]
                if label == "batch":
                    pending_calls.update((call["id"], call) for call in args["proposals"])
                elif label == "assessment":
                    proposed.append({"step": step, **pending_calls[args["id"]], "missing": args["missing"],
                                     "sources": args["sources"], "remaining": args["before"].get("remaining")})
                elif label == "review_started":
                    proposal = pending_calls[args["id"]]
                    target = proposal["arguments"]["ticket_id"]
                    with closing(create_env(task["domain"])) as direct:
                        direct.reset(task)
                        env.conn.backup(direct.conn)
                        result = direct.execute_tool(proposal["tool"], proposal["arguments"])
                        direct_evidence = {"ok": result.ok, "error": result.error_code, "ticket_after": ticket_facts(direct, target)}
                    confirmed[args["id"]] = {"step": step, "proposal": proposal, "direct_branch": direct_evidence}
                elif label == "review_result":
                    resolutions.append({**confirmed[args["id"]], **args})
                elif label == "supplement":
                    last_bundle = env.execute_tool(args["tool"], args["arguments"])
                    if last_bundle.ok != args["observation"]["ok"]:
                        raise ValueError("Replay diverged on an application read.")
                    tool_trace.append({"event_id": row["event_id"], "step": step, "tool": args["tool"],
                                       "arguments": args["arguments"], "application_read": True, "ok": last_bundle.ok})
                if label == "receipt" and args["tool"] == "get_invoice_bundle":
                    last_bundle = env.execute_tool(args["tool"], args["arguments"])
                if label in ("receipt", "supplement") and args["tool"] == "get_invoice_bundle" and last_bundle and last_bundle.ok:
                    data = args["observation"] if label == "receipt" else args["observation"]["data"]
                    current = last_bundle.data["ledger_entries"]
                    keys = ("ticket_id", "acct_id", "inv_id", "amount_cents", "memo", "voided")
                    for actual in data["ledger_entries"]:
                        old_id = actual["entry_id"]
                        if old_id.startswith("ADJ-") and old_id not in generated:
                            matches = [entry["entry_id"] for entry in current if entry["entry_id"].startswith("ADJ-")
                                       and all(entry[k] == actual[k] for k in keys) and entry["entry_id"] not in generated.values()]
                            if len(matches) == 1:
                                generated[old_id] = matches[0]
                continue
            if tool.startswith("__"):
                continue
            logged = {"event_id": row["event_id"], "step": step, "tool": tool, "arguments": args,
                      "ok": bool(row["tool_ok"]), "error": row["error_code"], "blocked": bool(row["fsm_blocked"])}
            tool_trace.append(logged)
            if row["fsm_blocked"] or row["error_code"] == "SCHEMA_PARSE_ERROR":
                continue
            dispatch_args = dict(args)
            if dispatch_args.get("entry_id") in generated:
                dispatch_args["entry_id"] = generated[dispatch_args["entry_id"]]
            if tool == "post_adjustment":
                before = ticket_facts(env, args["ticket_id"])
            outcome = env.execute_tool(tool, dispatch_args)
            if (outcome.ok, outcome.error_code) != (logged["ok"], logged["error"]):
                raise ValueError(f"Replay mismatch in {run['run_id']} step {step}: {tool} {outcome.error_code} vs {logged['error']}")
            if tool == "post_adjustment":
                adjustments.append({**logged, "before": before, "after": ticket_facts(env, args["ticket_id"])})

        final = {ticket: ticket_facts(env, ticket) for ticket in tickets}
        verification = env.verify(task)
        if verification.passed != bool(run["passed"]):
            raise ValueError(f"Replay terminal differs for {run['run_id']}")
    for resolution in resolutions:
        proposal_id = resolution["proposal"]["id"]
        resolution["before_dispatch"] = next(item for item in proposed if item["id"] == proposal_id)
        actual = next(item for item in adjustments if item["step"] == resolution["step"])
        resolution["replacement_dispatch"] = actual
        original = resolution["proposal"]["arguments"]
        if (resolution["proposal"]["tool"] == "post_adjustment" and original["ticket_id"] == actual["arguments"]["ticket_id"]
                and original["inv_id"] == actual["arguments"]["inv_id"]):
            gap = actual["before"]["invoice_gap_cents"][original["inv_id"]]
            resolution["original_delta_cents"] = original["delta_cents"]
            resolution["gap_if_original_posts_cents"] = gap - original["delta_cents"]
    return {**run, "adjustments": adjustments, "reviews": resolutions, "proposals": proposed,
            "terminal_tickets": final, "verifier_violations": verification.violations, "dispatch_trace": tool_trace}


def billing(db, protocol):
    status_by_slot = defaultdict(set)
    runs = {row["run_id"]: row for row in db.execute("SELECT run_id,status,metadata_json FROM runs")}
    for row in runs.values():
        status_by_slot[json.loads(row["metadata_json"])["slot"]].add(row["status"])
    attempted = set(status_by_slot)
    completed = {slot for slot, states in status_by_slot.items() if "COMPLETED" in states}
    missing = set(job["slot"] for job in protocol["jobs"]) - completed
    unknown, fees, repeated = [], Decimal(), Counter()
    for row in db.execute("SELECT attempt_id,run_id,step_idx,tier,request_hash,status,estimated_max_cost_cny,actual_cost_cny "
                          "FROM api_attempts ORDER BY created_at,attempt_id"):
        slot = json.loads(runs[row["run_id"]]["metadata_json"])["slot"]
        fees += Decimal(str(row["actual_cost_cny"]))
        repeated[row["run_id"], row["step_idx"], row["request_hash"]] += 1
        if row["status"] in ("PENDING", "UNKNOWN_BILLING"):
            unknown.append({"slot": slot, **dict(row)})
    return {"complete": len(completed), "attempted_incomplete": sorted(missing & attempted),
            "unstarted": sorted(missing - attempted), "charged_cny": str(fees),
            "repeat_request_groups_within_run_step": sum(n > 1 for n in repeated.values()),
            "reserved_cny": str(sum(Decimal(str(row["estimated_max_cost_cny"])) for row in unknown)),
            "unreconciled_attempts": unknown, "vendor_reconciliation": "awaiting provider bill; local request hashes only"}


def audit(run_dir):
    protocol = _read_json(run_dir / "protocol.json")
    tasks = _read_json(run_dir / "tasks.json")
    if (_json_digest(tasks) != protocol["task_sha256"]
            or _json_digest({k: v for k, v in protocol.items() if k != "fingerprint"}) != protocol["fingerprint"]):
        raise ValueError("Input does not match the frozen experiment.")
    with closing(_read_database(run_dir / DATABASE)) as db:
        db.execute("BEGIN")
        pairs = []
        for left, right in PAIRS:
            runs = [matched_terminal(db, protocol, slot) for slot in (left, right)]
            if runs[0]["task_id"] != runs[1]["task_id"]:
                raise ValueError("Paired assignments do not share an initial state.")
            pairs.append([replay(db, run, tasks[run["task_id"]]) for run in runs])
        costs = billing(db, protocol)
    return {"protocol_fingerprint": protocol["fingerprint"], "pairs": pairs, "billing": costs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=PILOT_RUNS / "prewrite_v3_live_20261123")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.run_dir)
    with args.out.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    main()
