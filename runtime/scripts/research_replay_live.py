"""Check archived v4.0 public receipts, quota debits and terminal scores offline."""

import argparse
from collections import Counter
from contextlib import closing
import json
from pathlib import Path

from .research_v40_cases import env_for
from .research_v40_trial import _independent_state


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def normalize(name, reply, recorded, ids):
    if not reply["ok"] or not recorded["ok"]:
        return reply
    data = reply["data"]
    if name == "post_adjustment":
        entry_id = data["entry_id"]
        ids[entry_id] = recorded["data"]["entry_id"]
        data["entry_id"] = ids[entry_id]
    if name == "request_maintenance_window":
        data["req_id"] = recorded["data"]["req_id"]
    if name == "get_invoice_bundle":
        for entry in data["ledger_entries"]:
            entry_id = entry["entry_id"]
            if entry_id in ids:
                entry["entry_id"] = ids[entry_id]
                if entry["ref_code"] == f"ADJ-{entry_id}":
                    entry["ref_code"] = f"ADJ-{ids[entry_id]}"
        data["ledger_entries"].sort(key=lambda entry: entry["entry_id"])
    return reply


def allowances(env, candidate):
    if candidate["name"] == "restart_service":
        args = candidate["arguments"]
        record = env.snapshot_state()["services"][args["svc_id"]]
        return record["max_restarts"] - record["restarts"]
    args = candidate["arguments"]
    return env.conn.execute("SELECT max_adjustments-adj_count FROM reconciliation_tickets WHERE ticket_id=?",
                            (args["ticket_id"],)).fetchone()[0]


def replay(rows, cases):
    diagnostics = []
    receipts = Counter()
    for row in rows:
        case = cases[row["task"]]
        task = case["task"]
        ids = {}
        events = []
        with closing(env_for(task)) as env:
            for event in row["trace"]:
                kind = event["event"]
                flagged = event.get("oracle", {}).get("new_loss") or kind == "future_block"
                if flagged:
                    candidate = event["candidate"]
                    start = allowances(env, candidate)
                    before, after = _independent_state(env, case, candidate)
                    expected = {"before": before, "after": after, "new_loss": before is True and after is False}
                    if expected != event["oracle"]:
                        raise ValueError(f"Reachability mismatch in {row['run_id']} at {event['step']}.")
                if kind in ("dispatch", "auto_read"):
                    candidate = event["candidate"]
                    name = candidate["name"]
                    args = dict(candidate["arguments"])
                    if "entry_id" in args:
                        reverse_ids = {saved: current for current, saved in ids.items()}
                        args["entry_id"] = reverse_ids.get(args["entry_id"], args["entry_id"])
                    result = env.execute_tool(name, args)
                    recorded = event["reply"]
                    reply = normalize(name, json.loads(result.to_tool_message()), recorded, ids)
                    if reply != recorded:
                        raise ValueError(f"Public receipt mismatch in {row['run_id']} at {event['step']} ({name}).")
                    if kind == "auto_read" and (result.is_write or result.state_changed):
                        raise ValueError(f"Automatic observation mutated state in {row['run_id']}.")
                    receipts[kind] += 1
                if flagged:
                    remaining = allowances(env, candidate)
                    expected_remaining = start if kind == "future_block" else event["assessment"]["q_after"]
                    if remaining != expected_remaining:
                        raise ValueError(f"Allowance mismatch in {row['run_id']} at {event['step']}.")
                    events.append({**event, "replayed_quota_before": start, "replayed_quota_after": remaining})
            outcome = env.verify(task)
            if row["terminal"] is not None and row["terminal"] != outcome.passed:
                raise ValueError(f"Terminal mismatch in {row['run_id']}.")
            if row["terminal"] is not True or events:
                keys = ("slot", "task", "provider", "repeat", "arm", "family", "quota_group", "terminal", "stop", "meter_calls")
                diagnostics.append({**{key: row[key] for key in keys}, "domain": task["domain"],
                                    "replayed_terminal": outcome.passed, "unmet_conditions": outcome.violations,
                                    "quota_events": events})
    return {"assignments": len(rows), "verified_receipts": receipts, "diagnostic_rows": diagnostics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cases = {case["task"]["task_id"]: case for case in load(args.cases)}
    report = replay(load(args.result)["rows"], cases)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("assignments", "verified_receipts")}))


if __name__ == "__main__":
    main()
