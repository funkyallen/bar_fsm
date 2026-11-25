"""Audit exposure, omitted tool calls, and stage provenance using archived records."""

import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ARMS = ("M0", "M1", "M2")
STAGES = {"original": "formal", "completion": "formal_completion"}
CONTRASTS = (("M1", "M0"), ("M2", "M1"), ("M2", "M0"))


def read_json(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def read_meter(directory):
    path = ROOT / "live" / directory / "meter.sqlite"
    return sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)


def audit_stages(rows, tasks):
    stages = {}
    groups = defaultdict(dict)
    for row in rows:
        key = (row["task"], row["provider"], row["repeat"])
        if row["arm"] in groups[key] or row["origin"] not in STAGES:
            raise ValueError(f"Unexpected stage or duplicate scored row: {row['slot']}.")
        groups[key][row["arm"]] = row
    if len(groups) != 96 or any(set(group) != set(ARMS) for group in groups.values()):
        raise ValueError("The terminal allocation is incomplete.")
    for origin, directory in STAGES.items():
        with closing(read_meter(directory)) as connection:
            start, end = connection.execute(
                "SELECT MIN(created_at), MAX(updated_at) FROM api_attempts"
            ).fetchone()
            models = dict(connection.execute(
                "SELECT provider, GROUP_CONCAT(DISTINCT model_actual) FROM api_attempts "
                "WHERE status='OK' GROUP BY provider ORDER BY provider"
            ))
        stage_rows = [r for r in rows if r["origin"] == origin]
        stages[origin] = {
            "first_request_timestamp": start,
            "last_attempt_update_timestamp": end,
            "reported_model_ids": models,
            "completed_and_scored_by_arm": {
                arm: {
                    "completed": sum(r["terminal"] for r in stage_rows if r["arm"] == arm),
                    "scored": sum(r["arm"] == arm for r in stage_rows),
                }
                for arm in ARMS
            },
            "composition": [
                {"provider": provider, "domain": domain, "quota": quota, "scored": count}
                for (provider, domain, quota), count in sorted(Counter(
                    (r["provider"], tasks[r["task"]]["domain"], r["quota_group"]) for r in stage_rows
                ).items())
            ],
        }

    contrasts = {}
    for treatment, control in CONTRASTS:
        counts = {origin: Counter() for origin in (*STAGES, "cross_stage")}
        for group in groups.values():
            treated, baseline = group[treatment], group[control]
            origin = treated["origin"] if treated["origin"] == baseline["origin"] else "cross_stage"
            counts[origin][int(treated["terminal"]) - int(baseline["terminal"])] += 1
        same_stage = counts["original"] + counts["completion"]
        counts["same_stage_combined"] = same_stage
        contrasts[f"{treatment}_minus_{control}"] = {
            origin: {
                "pairs": sum(outcomes.values()), "treatment_only": outcomes[1],
                "control_only": outcomes[-1], "tied": outcomes[0],
            }
            for origin, outcomes in counts.items()
        }
    gap = datetime.fromisoformat(stages["completion"]["first_request_timestamp"]) - datetime.fromisoformat(
        stages["original"]["last_attempt_update_timestamp"]
    )
    return {"stages": stages, "between_stage_seconds": int(gap.total_seconds()), "contrasts": contrasts}


def audit_tight_service(rows, tasks):
    episodes = []
    for row in rows:
        task = tasks[row["task"]]
        if (row["provider"], row["quota_group"], task["domain"]) != (
            "deepseek", "tight", "ops_troubleshoot"
        ):
            continue
        targets = task["goal"]["running_revisions"]
        staged_at = {}
        activated = set()
        probed = set()
        restarts = []
        withheld = 0
        for event in row["trace"]:
            candidate = event.get("candidate", {})
            name = candidate.get("name")
            assessment = event.get("assessment", {})
            if (name == "restart_service" and event["event"] != "dispatch"
                    and assessment.get("local") == "pass" and assessment.get("result") == "block"):
                withheld += 1
            if event["event"] != "dispatch" or name not in {
                "rollback_config", "restart_service", "run_health_probe"
            }:
                continue
            args = candidate["arguments"]
            svc = args["svc_id"]
            reply = event["reply"]
            if name == "rollback_config" and reply["ok"]:
                staged_at[svc] = (args["rev_id"], event["step"])
            if name == "restart_service":
                staged_rev, staged_step = staged_at.get(svc, (None, None))
                running_rev = reply.get("data", {}).get("running_rev")
                if reply["ok"] and running_rev == targets[svc]:
                    activated.add(svc)
                probed.discard(svc)
                restarts.append({
                    "step": event["step"], "service": svc, "staged_step": staged_step,
                    "staged_target_before_restart": staged_rev == targets[svc],
                    "restart_ok": reply["ok"], "running_target_after_restart": running_rev == targets[svc],
                    "new_reachability_loss": event["oracle"]["new_loss"],
                })
            if name == "run_health_probe" and reply["ok"] and svc in activated:
                probed.add(svc)
        episodes.append({
            "slot": row["slot"], "arm": row["arm"], "completed": row["terminal"], "stop": row["stop"],
            "certificate_holds": withheld, "restarts": restarts,
            "targets_never_activated": sorted(set(targets) - activated),
            "required_probes_outstanding": sorted(set(task["assertions"]["healthy_and_probed"]) - probed),
        })
    if len(episodes) != 36 or Counter(e["arm"] for e in episodes) != dict.fromkeys(ARMS, 12):
        raise ValueError("The tight DeepSeek service allocation changed.")
    restarts = [restart for e in episodes for restart in e["restarts"]]
    return {
        "episodes": len(episodes), "completed": sum(e["completed"] for e in episodes),
        "reached_restart": sum(bool(e["restarts"]) for e in episodes),
        "dispatched_restarts": len(restarts),
        "target_staged_and_activated": sum(
            r["staged_target_before_restart"] and r["restart_ok"] and r["running_target_after_restart"]
            for r in restarts
        ),
        "new_reachability_losses": sum(r["new_reachability_loss"] for r in restarts),
        "certificate_holds": sum(e["certificate_holds"] for e in episodes),
        "episodes_detail": sorted(episodes, key=lambda e: e["slot"]),
    }


def audit_accounting_calls():
    sys.path[:0] = [str(ROOT / "runtime"), str(ROOT / "runtime" / "src")]
    from scripts.research_voidability_qualification import VoidabilityReceiptEnv

    path = "shared_allowance/live"
    rows = {r["run_id"]: r for r in read_json(f"live/{path}/result.json")["rows"]}
    tasks = {c["task"]["task_id"]: c["task"] for c in read_json("live/shared_allowance/cases.json")}
    calls = defaultdict(list)
    with closing(read_meter(path)) as connection:
        for run_id, step, response in connection.execute(
            "SELECT run_id, step_idx, response_json FROM api_attempts WHERE status='OK' "
            "ORDER BY run_id, step_idx"
        ):
            tool_calls = json.loads(response)["tool_calls"]
            calls[run_id].extend((step, tool_calls[0], call) for call in tool_calls[1:])
    cohorts = {"completed": Counter(), "unfinished": Counter()}
    omitted_audits = []
    for run_id, row in sorted(rows.items()):
        dropped = calls[run_id]
        if len(dropped) != row["later_calls_dropped"]:
            raise ValueError(f"Meter and execution log disagree on dropped calls: {run_id}.")
        counts = cohorts["completed" if row["terminal"] else "unfinished"]
        counts.update(call["name"] for _, _, call in dropped)
        if row["terminal"]:
            continue
        for step, first, call in dropped:
            if call["name"] != "run_audit_check":
                continue
            if call["parse_error"] or first["parse_error"]:
                raise ValueError(f"Invalid arguments in a replayed call: {run_id} at {step}.")
            selected = next(event for event in row["trace"] if event["step"] == step)
            if selected["disposition"] != "dispatch" or selected["candidate"] != {
                "name": first["name"], "arguments": json.loads(first["arguments"])
            }:
                raise ValueError(f"First call does not match the dispatched candidate: {run_id} at {step}.")
            env = VoidabilityReceiptEnv()
            env.reset(tasks[row["task"]])
            with closing(env.conn):
                for event in row["trace"]:
                    if event["step"] > step:
                        break
                    if event["disposition"] != "dispatch":
                        continue
                    action = event["candidate"]
                    actual = env.execute_tool(action["name"], action["arguments"])
                    reply = event["reply"]
                    if actual.ok != reply["ok"] or (actual.ok and actual.data != reply["data"]):
                        raise ValueError(f"Recorded prefix does not match backend replay: {run_id}.")
                audit = env.execute_tool(call["name"], json.loads(call["arguments"]))
            omitted_audits.append({
                "slot": row["slot"], "step": step, "selected": first["name"],
                "audit_executed": audit.ok, "audit_passed": audit.data.get("audit_passed"),
                "unbalanced_invoices": audit.data.get("unbalanced_invoices"),
                "has_active_duplicates": audit.data.get("has_active_duplicates"),
            })
    return {
        "episode_counts": {"completed": sum(r["terminal"] for r in rows.values()),
                           "unfinished": sum(not r["terminal"] for r in rows.values())},
        "omitted_call_counts": {cohort: dict(sorted(counts.items())) for cohort, counts in cohorts.items()},
        "unfinished_omitted_audits_replayed": omitted_audits,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Directory for the trace audit JSON.")
    output = parser.parse_args().out
    output.mkdir(parents=True, exist_ok=True)
    rows = read_json("live/formal_completion/result.json")["rows"]
    tasks = {c["task"]["task_id"]: c["task"] for c in read_json("formal_fixtures/cases.json")}
    report = {"formal_collection": audit_stages(rows, tasks),
              "tight_deepseek_service": audit_tight_service(rows, tasks),
              "accounting_first_call": audit_accounting_calls()}
    (output / "trace_forensics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
