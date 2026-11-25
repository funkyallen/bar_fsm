"""Recompute recorded decisions and check physical quota effects on the frozen v4.0 replay."""

import argparse
from collections import Counter
from contextlib import closing
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3

from .research_replay_live import allowances, normalize
from .research_v40_cases import env_for
from .research_v40_control import assess
from .research_v40_trial import _independent_state, _json_digest, _valid
from .research_v34_intervention import key
from .research_v40_validation import qualify
from aroute.envs.db_transaction_env import DB_WRITE_TOOLS
from aroute.envs.ops_troubleshoot_env import OPS_WRITE_TOOLS
from aroute.core.model_pool import _parse_tool_call_fields


QUOTA_WRITES = {"post_adjustment", "restart_service"}


def same(value, recorded):
    return _json_digest(value) == _json_digest(recorded)


def call(env, event, ids):
    name = event["candidate"]["name"]
    args = dict(event["candidate"]["arguments"])
    if "entry_id" in args:
        reverse = {old: new for new, old in ids.items()}
        args["entry_id"] = reverse.get(args["entry_id"], args["entry_id"])
    response = env.execute_tool(name, args)
    wire = normalize(name, json.loads(response.to_tool_message()), event["reply"], ids)
    if not same(wire, event["reply"]):
        raise ValueError(f"Replayed receipt differs for {name}.")
    return response, wire


def replay(rows, case_list):
    counts = Counter()
    forks = []
    cases = {c["task"]["task_id"]: c for c in case_list}
    for row in rows:
        case = cases[row["task"]]
        task = case["task"]
        ids = {}
        receipts = {}
        prior_step, decided = -1, False
        with closing(env_for(task)) as env:
            for event in row["trace"]:
                step, kind = event["step"], event["event"]
                if not prior_step <= step < task["max_steps"] or step < 0 or (step == prior_step and decided):
                    raise ValueError(f"Incorrect event order or duplicate decision at {row['run_id']}:{step}.")
                if step != prior_step:
                    receipts = {}
                    prior_step = step
                    decided = False
                decided = kind != "auto_read"
                if kind == "schema_hold":
                    counts[kind] += 1
                    continue
                candidate = event["candidate"]
                if kind == "auto_read":
                    reply, wire = call(env, event, ids)
                    if reply.is_write or reply.state_changed:
                        raise ValueError("An automatic read mutated business state.")
                    if wire["ok"]:
                        receipts[key(candidate)] = wire["data"]
                    counts[kind] += 1
                    continue
                decision = assess(candidate, task["goal"], receipts)
                expected = ("local_hold" if decision.local != "pass" else
                            "future_block" if row["arm"] != "M0" and decision.result == "block" else "dispatch")
                if kind != expected or not same(asdict(decision), event["assessment"]):
                    raise ValueError(f"Guard proof or disposition differs at {row['run_id']}:{step}.")
                counts[kind] += 1
                quota_candidate = kind != "local_hold" and candidate["name"] in QUOTA_WRITES
                if quota_candidate:
                    before, after = _independent_state(env, case, candidate)
                    label = {"before": before, "after": after, "new_loss": before is True and after is False}
                    if label != event.get("oracle"):
                        raise ValueError(f"Missing or changed reachability label at {row['run_id']}:{step}.")
                    counts["validated_quota_labels"] += 1
                    quota_before = allowances(env, candidate)
                    if candidate["name"] == "restart_service":
                        sid = candidate["arguments"]["svc_id"]
                        svc = receipts["inspect_service", sid]["service"]
                        target = task["goal"]["running_revisions"][sid]
                        simple = quota_before == 1 and svc["staged_rev"] != target
                        counts["service_rule_comparisons"] += 1
                        counts["service_rule_disagreements"] += simple != (decision.result == "block")
                if kind == "dispatch":
                    reply, _ = call(env, event, ids)
                    if quota_candidate:
                        quota_after = allowances(env, candidate)
                        if reply.ok and quota_before - quota_after != 1:
                            raise ValueError("A successful quota write did not spend exactly one unit.")
                        counts["observed_new_losses"] += int(label["new_loss"])
                if kind == "future_block":
                    with closing(env_for(task)) as fork:
                        env.conn.backup(fork.conn)
                        observed = fork.execute_tool(candidate["name"], candidate["arguments"])
                        successor_quota = allowances(fork, candidate)
                    if (not observed.ok or successor_quota != decision.q_after or
                            allowances(env, candidate) != quota_before):
                        raise ValueError(f"Incorrect withheld action consequence at {row['run_id']}:{step}.")
                    forks.append({"slot": row["slot"], "step_zero_based": step, "arm": row["arm"],
                                  "operator": candidate["name"], "actual_allowance_preserved": quota_before,
                                  "forked_allowance_after_dispatch": successor_quota,
                                  "forked_response": observed.data, "new_loss_under_oracle": label["new_loss"]})
            score = env.verify(task).passed
            if row["terminal"] is not None and score != row["terminal"]:
                raise ValueError(f"Terminal score differs at {row['run_id']}.")
    return {"assignments": len(rows), "verified": dict(counts), "withheld_dispatch_forks": forks}


def ledger_proposals(rows, ledger, cases):
    schemas = {}
    for case in cases:
        task = case["task"]
        with closing(env_for(task)) as env:
            schemas[task["task_id"]] = {tool["function"]["name"]: tool["function"]["parameters"]
                                        for tool in env.tool_schemas()}
    by_run = {}
    for row in rows:
        decisions = [event for event in row["trace"] if event["event"] != "auto_read"]
        by_step = {event["step"]: event for event in decisions}
        if len(by_step) != len(decisions):
            raise ValueError(f"Multiple controller decisions at one step in {row['run_id']}.")
        by_run[row["run_id"]] = (schemas[row["task"]], by_step)
    provider_counts = {}
    validated = 0
    sql = """SELECT run_id,step_idx,provider,response_json FROM api_attempts WHERE status='OK'"""
    with closing(sqlite3.connect(ledger.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        for run_id, step, provider, response in db.execute(sql):
            calls = json.loads(response)["tool_calls"]
            schema, decisions = by_run[run_id]
            first = decisions.pop(step, None)
            if not calls:
                if first is not None:
                    raise ValueError(f"Proposal absent from the ledger at {run_id}:{step}.")
                continue
            if first is None:
                raise ValueError(f"Proposal missing from the controller trace at {run_id}:{step}.")
            call = calls[0]
            parsed = _parse_tool_call_fields("archived", call["name"], call["arguments"])
            valid = _valid(parsed, schema.get(parsed.name))
            if parsed.parse_error != call["parse_error"] or valid == (first["event"] == "schema_hold"):
                raise ValueError(f"Schema decision differs from the ledger at {run_id}:{step}.")
            expected = {"name": parsed.name, "arguments": parsed.arguments_dict}
            if valid and not same(expected, first["candidate"]):
                raise ValueError(f"Controller proposal differs from the ledger at {run_id}:{step}.")
            validated += 1
            if len(calls) > 1:
                stats = provider_counts.setdefault(provider, Counter())
                stats["replies_with_multiple_calls"] += 1
                stats["later_calls_dropped"] += len(calls) - 1
                stats["later_state_change_operations_dropped"] += sum(
                    call["name"] in DB_WRITE_TOOLS | OPS_WRITE_TOOLS or call["name"] == "run_audit_check"
                    for call in calls[1:])
    if any(decisions for _, decisions in by_run.values()):
        raise ValueError("Some trace decisions have no corresponding API attempt.")
    return {"first_proposals_verified": validated, "multi_call_by_provider": provider_counts}


def tamper_checks(rows, cases, ledger):
    checks = {}
    mutations = (("missing_loss_label", 11, "dispatch"), ("changed_certificate", 10, "future_block"),
                 ("duplicated_withheld_decision", 10, "future_block"))
    for name, slot, kind in mutations:
        row = deepcopy(next(row for row in rows if row["slot"] == slot))
        event = next(e for e in row["trace"] if e["event"] == kind and e.get("oracle", {}).get("new_loss"))
        if name == "missing_loss_label":
            del event["oracle"]
        elif name == "changed_certificate":
            event["assessment"].update(q_after=999, lower=0)
        else:
            row["trace"].insert(row["trace"].index(event), deepcopy(event))
        try:
            replay([row], cases)
        except ValueError as failure:
            checks[name] = str(failure)
        else:
            raise ValueError(f"Corrupted trace was accepted: {name}.")
    index = next(i for i, row in enumerate(rows) if row["slot"] == 11)
    fake = list(rows)
    fake[index] = deepcopy(rows[index])
    step = next(i for i, event in enumerate(fake[index]["trace"]) if event["step"] == 0)
    fake[index]["trace"][step] = {"step": 0, "event": "schema_hold"}
    try:
        ledger_proposals(fake, ledger, cases)
    except ValueError as failure:
        checks["accepted_proposal_relabelled_as_invalid"] = str(failure)
    else:
        raise ValueError("The ledger accepted a corrupted schema decision.")
    return checks


def qualify_frozen_cases(archived):
    reproduced = qualify()
    comparable = ("task", "domain", "evidence", "candidate", "truth_before", "truth_after",
                  "backend_ok_if_authorized", "controller", "label", "fixture")
    for old, new in zip(archived["cases"], reproduced["cases"], strict=True):
        if not same([new[key] for key in comparable], [old[key] for key in comparable]):
            raise ValueError(f"Reconstruction of the frozen fixture differs at {old['task']}.")
    return {
        "reclassified_fixtures": len(reproduced["cases"]),
        "initially_unreachable_with_certificate": sum(
            r["label"] == "initially_unreachable" and r["controller"]["result"] == "block"
            for r in reproduced["cases"]),
        "summary": reproduced["summary"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--offline-oracle", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    results = json.loads(args.results.read_text(encoding="utf-8"))
    report = replay(results["rows"], cases)
    report["ledger_proposals"] = ledger_proposals(results["rows"], args.ledger, cases)
    report["rejected_tamper_checks"] = tamper_checks(results["rows"], cases, args.ledger)
    if args.offline_oracle:
        report["reproduced_offline"] = qualify_frozen_cases(json.loads(args.offline_oracle.read_text(encoding="utf-8")))
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["verified"]))


if __name__ == "__main__":
    main()
