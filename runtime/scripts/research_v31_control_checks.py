"""Verify C/E allocation and deterministic checks on the same public development receipts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from contextlib import closing
from copy import deepcopy
from pathlib import Path

from aroute.envs import create_env
from aroute.envs.db_transaction_env import compute_expected_cents
from aroute.orchestrator.prewrite import Assessment, PrewriteControl, Read

from .research_run_pilot import DATABASE, PILOT_RUNS, _read_database, _read_json
from .research_v31_accounting_audit import matched_terminal


def deterministic_invoice_delta(arguments, bundle, account):
    """Use a complete receipt for one invoice. Leave ambiguity and reserve checks explicit."""
    ticket = bundle["ticket"]
    invoice_id = arguments["inv_id"]
    if ticket["ticket_id"] != arguments["ticket_id"] or account["acct_id"] != arguments["acct_id"]:
        return {"decision": "unresolved", "reason": "receipt_binding"}
    invoice = next((inv for inv in bundle["invoices"] if inv["inv_id"] == invoice_id), None)
    if invoice is None:
        return {"decision": "unresolved", "reason": "missing_invoice"}
    active = [entry for entry in bundle["ledger_entries"] if not entry["voided"] and entry["inv_id"] == invoice_id]
    signatures = Counter((entry["acct_id"], entry["amount_cents"], entry["ref_code"]) for entry in active)
    if any(n > 1 for n in signatures.values()):
        return {"decision": "unresolved", "reason": "possible_duplicate"}
    expected = compute_expected_cents(**{k: invoice[k] for k in ("subtotal_cents", "tax_rate_bps", "discount_cents",
                                                                 "discount_pre_tax", "late_fee_bps", "days_late")})
    delta = expected - sum(entry["amount_cents"] for entry in active)
    if not delta:
        return {"decision": "defer", "reason": "balanced_invoice", "delta_cents": 0}
    enough = account["balance_cents"] + delta >= account["min_reserve_cents"]
    if account["frozen"] or not enough or ticket["adj_count"] >= ticket["max_adjustments"]:
        return {"decision": "unresolved", "reason": "insufficient_public_capacity", "delta_cents": delta}
    return {"decision": "keep" if delta == arguments["delta_cents"] else "replace", "delta_cents": delta}


def _events(db, run):
    return [dict(row) | {"payload": json.loads(row["tool_args_json"])} for row in db.execute(
        "SELECT event_id,step_idx,tier,tool_args_json FROM step_events WHERE run_id=? AND tool_name='__V3_PREWRITE__' "
        "ORDER BY event_id", (run["run_id"],))]


def followup_on_live_think(fork_result):
    from .research_v31_accounting_fork import PARENT, capture

    result = _read_json(fork_result)
    if result["mode"] != "live" or result["source_slot"] != 79:
        raise ValueError("This audit requires the real slot 79 continuation.")
    checkpoint = capture(PARENT)
    first_ticket = checkpoint.candidate.arguments_dict["ticket_id"]
    branch = next(row for row in result["branches"] if row["arm"] == "T")
    control = deepcopy(checkpoint.control)
    with closing(create_env("db_transaction")) as env:
        env.reset(checkpoint.task)
        env.conn.deserialize(checkpoint.database)
        for event in branch["trace"]:
            tool, arguments = event["tool"], event["arguments"]
            if tool == "post_adjustment" and arguments["ticket_id"] != first_ticket:
                break
            actual = env.execute_tool(tool, arguments)
            if (actual.ok, actual.error_code) != (event["ok"], event["error"]):
                raise ValueError("The live Think branch diverges before the second ticket.")
            control.observe(tool, arguments, actual)
        else:
            raise ValueError("The live Think branch has no second-ticket proposal.")
        ticket, invoice = arguments["ticket_id"], arguments["inv_id"]
        receipt = control.receipts[Read("get_invoice_bundle", "ticket_id", ticket)]
        account = control.receipts[Read("query_account", "acct_id", arguments["acct_id"])]
        decision = deterministic_invoice_delta(arguments, receipt.data, account.data["account"])
        if decision["decision"] != "replace":
            raise ValueError("Expected a determinate alternative using the archived public receipts.")
        alternative = {**arguments, "delta_cents": decision["delta_cents"]}
        for action, args in (("post_adjustment", alternative), ("run_audit_check", {"ticket_id": ticket}),
                             ("reconcile_invoice", {"inv_id": invoice}), ("close_ticket", {"ticket_id": ticket})):
            assert env.execute_tool(action, args).ok
        passed = env.verify(checkpoint.task).passed
    return {"source": "real Think branch prefix, deterministic alternative and scripted tool suffix offline",
            "step": event["step"], "ticket": ticket, "actual_delta_cents": arguments["delta_cents"],
            "public_receipt_sequences": [receipt.sequence, account.sequence], "candidate": decision,
            "both_tickets_verified": passed}


def checks():
    root = PILOT_RUNS / "prewrite_v3_live_20261123"
    protocol = _read_json(root / "protocol.json")
    deterministic = []
    with closing(_read_database(root / DATABASE)) as db:
        for slot in (43, 69, 79):
            rows = _events(db, matched_terminal(db, protocol, slot))
            review = next(row for row in rows if row["payload"]["event"] == "review_started")
            call_id = review["payload"]["id"]
            call = next(call for row in rows if row["payload"]["event"] == "batch"
                        for call in row["payload"]["proposals"] if call["id"] == call_id)
            assessment = next(row["payload"] for row in rows if row["payload"]["event"] == "assessment"
                              and row["payload"]["id"] == call_id)
            receipts = {}
            for row in rows:
                e = row["payload"]
                if row["event_id"] < review["event_id"] and e.get("sequence") in assessment["sources"]:
                    data = e["observation"]
                    receipts[e["tool"]] = data["data"] if e["event"] == "supplement" else data
            decision = deterministic_invoice_delta(call["arguments"], receipts["get_invoice_bundle"], receipts["query_account"]["account"])
            replacement = next(row["payload"]["replacement_args"]["delta_cents"] for row in rows
                               if row["payload"]["event"] == "review_result" and row["payload"]["id"] == call_id)
            deterministic.append({"slot": slot, "receipt_sequences": assessment["sources"], "original_delta_cents": call["arguments"]["delta_cents"],
                                  "deterministic": decision, "archived_replacement_delta_cents": replacement})
            if slot == 69:
                timing_rows = rows

    def timing_choice(assume_fast):
        choice = {}
        for arm, policy in (("C", "targeted"), ("E", "first")):
            ctl = PrewriteControl("db_transaction", supplement=True, review=policy)
            for row in timing_rows:
                facts = row["payload"]
                if facts["event"] != "assessment" or facts["missing"]:
                    continue
                known = Assessment(tuple(facts["target"]), tuple(facts["pools"]), (), tuple(facts["sources"]),
                                   facts["risk"], facts["before"])
                if ctl.wants_review(known, "fast" if assume_fast else row["tier"]):
                    choice[arm] = {"step": row["step_idx"], "target": facts["target"], "remaining": facts["before"]["remaining"]}
                    break
        return choice

    control_choice = timing_choice(False)
    fixed_fast = timing_choice(True)
    assert fixed_fast["E"]["step"] < fixed_fast["C"]["step"]

    bridge = PILOT_RUNS / "observation_bridge_live"
    task_id = _read_json(bridge / "protocol.json")["jobs"][42]["task_id"]
    task = _read_json(bridge / "tasks.json")[task_id]
    with closing(create_env("ops_troubleshoot")) as env:
        env.reset(task)
        services = env.execute_tool("get_topology", {}).data["services"]
        service = next(row["svc_id"] for row in services if row["max_restarts"] == 1 and not row["maint_approved"])
        args = {"svc_id": service}
        before = env.execute_tool("inspect_service", args).data["service"]["restart_count"]
        result = env.execute_tool("restart_service", args)
        after = env.execute_tool("inspect_service", args).data["service"]["restart_count"]
        assert result.error_code == "MAINT_REQUIRED" and before == after
    return {"evidence_scope": "offline, fixed development traces", "same_receipt_deterministic": deterministic,
            "C_vs_E_first_eligible_on_logged_slot_69": control_choice,
            "C_vs_E_fixed_fast_candidate_replay": fixed_fast,
            "refused_without_debit": {"source_slot": 43, "target": service, "error": result.error_code, "before": before, "after": after}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--fork-result", type=Path, help="Also replay the completed real Think branch's second ticket.")
    args = parser.parse_args()
    result = checks()
    if args.fork_result is not None:
        result["second_ticket_after_real_think"] = followup_on_live_think(args.fork_result)
    with args.out.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write("\n")


if __name__ == "__main__":
    main()
