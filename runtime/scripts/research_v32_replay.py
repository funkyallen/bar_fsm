"""Replay synthetic transactions on independent copies of the frozen sandbox."""

from __future__ import annotations

import argparse
import json
from collections import deque
from contextlib import closing
from pathlib import Path

from aroute.envs import create_env
from aroute.orchestrator.prewrite import PrewriteControl, Read

from .research_run_pilot import DATABASE, PILOT_RUNS, _read_database, _read_json
from .research_v31_timing_candidates import CurrentClusterControl, call


SERVICE = "ops_troubleshoot"
ACCOUNTING = "db_transaction"
RESTRICTED = {SERVICE: "restart_service", ACCOUNTING: "post_adjustment"}
PARENT = PILOT_RUNS / "v31_rule_timing12_live_20261123"


def control_for(domain, *, supplement):
    kind = CurrentClusterControl if domain == SERVICE else PrewriteControl
    return kind(domain, supplement=supplement, review="off")


def read_events(db, run_id):
    for row in db.execute("SELECT * FROM step_events WHERE run_id=? ORDER BY event_id", (run_id,)):
        yield dict(row) | {"payload": json.loads(row["tool_args_json"] or "{}")}


def _service_successors(task, state):
    goals = set(task["assertions"]["healthy_and_probed"])
    revisions = task["initial_state"]["config_revisions"]
    for sid, svc in state["services"].items():
        if not svc["maint"]:
            yield call("request_maintenance_window", svc_id=sid)
            continue
        for revision in revisions:
            if revision["svc_id"] == sid and revision["rev_id"] != svc["active_rev"]:
                yield call("rollback_config", svc_id=sid, rev_id=revision["rev_id"])
        if svc["status"] != "HEALTHY" and svc["restarts"] < svc["max_restarts"]:
            yield call("restart_service", svc_id=sid)
        if sid in goals and svc["status"] == "HEALTHY" and not svc["probed"]:
            yield call("run_health_probe", svc_id=sid)


def service_witness(env, task):
    """Find a shortest successful tool sequence, without imposing an action order."""
    original = env.conn.serialize()
    start = env.snapshot_state()
    queue = deque([(original, start, [])])
    visited = {json.dumps(start, sort_keys=True)}
    stats = env.precondition_violations, env.write_count, env.read_count
    try:
        while queue:
            state_blob, state, path = queue.popleft()
            env.conn.deserialize(state_blob)
            if env.verify(task).passed:
                return path
            for action in _service_successors(task, state):
                env.conn.deserialize(state_blob)
                result = env.execute_tool(action["name"], action["arguments"])
                if not result.ok:
                    continue
                successor = env.snapshot_state()
                key = json.dumps(successor, sort_keys=True)
                if key not in visited:
                    visited.add(key)
                    queue.append((env.conn.serialize(), successor, [*path, action]))
        return None
    finally:
        env.conn.deserialize(original)
        env.precondition_violations, env.write_count, env.read_count = stats


def accounting_deficits(env, task):
    """Certify exhausted quotas using the amounts left after all mandatory voids."""
    query = """
        SELECT i.inv_id, i.expected_cents - COALESCE(SUM(l.amount_cents), 0) AS delta
        FROM invoices i LEFT JOIN ledger_entries l ON l.inv_id=i.inv_id
            AND l.voided=0 AND l.is_duplicate=0
        WHERE i.ticket_id=? GROUP BY i.inv_id, i.expected_cents
    """
    deficits = {}
    for ticket in task["assertions"]["closed_tickets"]:
        row = env.conn.execute("SELECT max_adjustments-adj_count FROM reconciliation_tickets WHERE ticket_id=?",
                               (ticket,)).fetchone()
        needed = sum(r["delta"] != 0 for r in env.conn.execute(query, (ticket,)))
        if needed > row[0]:
            deficits[ticket] = {"required_adjustments_lower_bound": needed, "remaining": row[0]}
    return deficits


def _accounting_residual(env, invoice):
    row = env.conn.execute("""
        SELECT i.expected_cents - COALESCE(SUM(l.amount_cents), 0)
        FROM invoices i LEFT JOIN ledger_entries l ON l.inv_id=i.inv_id
            AND l.voided=0 AND l.is_duplicate=0
        WHERE i.inv_id=? GROUP BY i.inv_id
    """, (invoice,)).fetchone()
    return None if row is None else row[0]


def accounting_witness(env, task):
    """Try one constructive completion; failure leaves reachability unresolved."""
    original = env.conn.serialize()
    stats = env.precondition_violations, env.write_count, env.read_count
    goals = set(task["assertions"]["closed_tickets"])
    try:
        for row in env.conn.execute("SELECT entry_id,ticket_id FROM ledger_entries WHERE is_duplicate=1 AND voided=0").fetchall():
            if row["ticket_id"] in goals and not env.execute_tool("void_duplicate_entry", {"entry_id": row["entry_id"]}).ok:
                return False
        invoices = [row for row in env.conn.execute("SELECT * FROM invoices") if row["ticket_id"] in goals]
        deltas = sorted(((_accounting_residual(env, inv["inv_id"]), inv) for inv in invoices),
                        key=lambda pair: -pair[0])
        for delta, inv in deltas:
            if delta and not env.execute_tool("post_adjustment", {"ticket_id": inv["ticket_id"], "inv_id": inv["inv_id"],
                                                                  "acct_id": inv["acct_id"], "delta_cents": delta}).ok:
                return False
        for tid in sorted(goals):
            env.execute_tool("run_audit_check", {"ticket_id": tid})
            for inv in invoices:
                if inv["ticket_id"] == tid and inv["status"] != "RECONCILED":
                    env.execute_tool("reconcile_invoice", {"inv_id": inv["inv_id"]})
            env.execute_tool("close_ticket", {"ticket_id": tid})
        return env.verify(task).passed
    finally:
        env.conn.deserialize(original)
        env.precondition_violations, env.write_count, env.read_count = stats


def remaining(env, tool, args):
    if tool == RESTRICTED[SERVICE]:
        svc = env.snapshot_state()["services"].get(args.get("svc_id"))
        return None if svc is None else svc["max_restarts"] - svc["restarts"]
    if tool == RESTRICTED[ACCOUNTING]:
        ticket = env.snapshot_state()["tickets"].get(args.get("ticket_id"))
        return None if ticket is None else ticket["max_adjustments"] - ticket["adj_count"]
    return None


def replay_actions(env, task, rows, *, supplement):
    """Reproduce dispatches and frozen public evidence. Hidden state stays in this audit."""
    control = control_for(task["domain"], supplement=supplement)
    restricted = RESTRICTED[task["domain"]]
    last_invalidation = {}
    proposals, writes = {}, []
    for row in rows:
        payload = row["payload"]
        step = row["step_idx"]
        tool = row["tool_name"] or ""
        event = payload.get("event") if tool == "__V3_PREWRITE__" else None
        cid = payload.get("id")
        key = step, cid
        if event == "assessment" and payload["tool"] == restricted:
            gaps = [Read(**gap) for gap in payload["missing"]]
            proposals[key] = {"step": step, "tier": row["tier"], "assessment": payload,
                              "invalidations": [last_invalidation[g] for g in gaps if g in last_invalidation],
                              "supplements": [], "remaining_model_steps": task["max_steps"] - step - 1}
            if task["domain"] == SERVICE:
                proposals[key].update({"state": env.snapshot_state(), "witness": service_witness(env, task)})
            continue
        if event in ("defer_for_facts", "rule_before_review", "rule_assessment", "dispatch", "review_started"):
            if key in proposals:
                proposals[key][event] = payload
            continue
        if event == "supplement":
            tool, args = payload["tool"], payload["arguments"]
        elif tool and not tool.startswith("__") and not row["fsm_blocked"] and row["error_code"] != "SCHEMA_PARSE_ERROR":
            args = payload
            if env.is_write_tool(tool):
                for proposal in proposals.values():
                    if proposal["step"] < step and "next_write" not in proposal:
                        proposal["next_write"] = {"step": step, "tool": tool, "arguments": args,
                                                  "ok": bool(row["tool_ok"]), "error": row["error_code"]}
        else:
            continue
        before = remaining(env, tool, args)
        checkpoint = env.conn.serialize() if before is not None else None
        residual = _accounting_residual(env, args.get("inv_id")) if checkpoint and task["domain"] == ACCOUNTING else None
        result = env.execute_tool(tool, args)
        after = remaining(env, tool, args)
        if event == "supplement":
            observed = json.loads(result.to_tool_message())
            if observed != payload["observation"]:
                raise ValueError(f"Supplemental observation diverged at step {step}: {tool}")
            parent = step, cid.rsplit("_prewrite_", 1)[0]
            if parent in proposals:
                proposals[parent]["supplements"].append(payload)
        elif (result.ok, result.error_code) != (bool(row["tool_ok"]), row["error_code"]):
            raise ValueError(f"Tool dispatch diverged at step {step}: {tool}")
        if checkpoint is not None:
            debit = before - after
            loss, evidence = False, None
            if debit:
                if task["domain"] == SERVICE:
                    after_path = service_witness(env, task)
                else:
                    after_deficits = accounting_deficits(env, task)
                after_blob = env.conn.serialize()
                env.conn.deserialize(checkpoint)
                try:
                    if task["domain"] == SERVICE:
                        before_path = service_witness(env, task)
                        loss = before_path is not None and after_path is None
                        evidence = {"reachable_before": before_path is not None, "reachable_after": after_path is not None}
                    else:
                        before_deficits = accounting_deficits(env, task)
                        evidence = {"quota_deficits_before": before_deficits, "quota_deficits_after": after_deficits}
                        if after_deficits and not before_deficits:
                            has_witness = accounting_witness(env, task)
                            loss = True if has_witness else None
                            evidence["reachable_before_witness"] = has_witness
                finally:
                    env.conn.deserialize(after_blob)
            writes.append({"step": step, "tool": tool, "arguments": args, "ok": result.ok,
                           "error": result.error_code, "remaining_before": before, "remaining_after": after,
                           "actual_debit": debit, "certified_new_loss": loss, "reachability": evidence,
                           "required_delta_after_mandatory_voids": residual})
        before_receipts = set(control.receipts)
        control.observe(tool, args, result)
        for gone in before_receipts - control.receipts.keys():
            last_invalidation[gone] = {"step": step, "tool": tool, "arguments": args, "read": gone.tool}
    return list(proposals.values()), writes


def audit_service(parent):
    saved = _read_json(parent / "result.json")
    tasks = _read_json(parent / "tasks.json")
    cases = []
    with closing(_read_database(parent / DATABASE)) as db:
        for row in saved["assignments"]:
            if row["domain"] != SERVICE:
                continue
            task = tasks[row["task_id"]]
            events = list(read_events(db, row["run_id"]))
            with closing(create_env(SERVICE)) as env:
                env.reset(task)
                proposals, writes = replay_actions(env, task, events, supplement=True)
                passed = env.verify(task).passed
                if passed != row["outcome"]:
                    raise ValueError(f"Frozen terminal outcome diverged for slot {row['slot']}")
                terminal_path = service_witness(env, task)
            evaluated = {(item["step"], item["assessment"]["id"]) for item in proposals}
            unvisited = [{"step": e["step_idx"], **c} for e in events if e["payload"].get("event") == "batch"
                         for c in e["payload"]["proposals"] if c["tool"] == RESTRICTED[SERVICE]
                         and (e["step_idx"], c["id"]) not in evaluated]
            next_batches = [e for e in events if e["payload"].get("event") == "batch"]
            for item in proposals:
                item["next_proposed_batch"] = next(({"step": e["step_idx"], "proposals": e["payload"]["proposals"]}
                                                   for e in next_batches if e["step_idx"] > item["step"]), None)
            cases.append({"slot": row["slot"], "arm": row["arm"], "state": row["state"], "passed": passed,
                          "proposals": proposals, "undispatched_batch_suffix_proposals": unvisited,
                          "restricted_writes": writes, "terminal_legal_tool_witness": terminal_path})
    return {"method": "SQLite replay and shortest successful tool path on a replay copy",
            "step_caveat": "Witness length counts tool invocations. This is not a simulated model continuation.",
            "source_protocol": saved["protocol_sha256"], "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = audit_service(PARENT)
    with args.out.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print([(row["slot"], row["passed"], len(row["proposals"])) for row in report["cases"]])


if __name__ == "__main__":
    main()
