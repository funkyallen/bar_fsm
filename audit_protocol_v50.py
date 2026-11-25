"""Reconstruct historical request and tool-call boundaries without a model API."""

import argparse
from collections import Counter, defaultdict
from contextlib import closing
from hashlib import sha256
import json
from pathlib import Path
import sqlite3

import yaml

from audit_trace_forensics import audit_accounting_calls
from reproduce_frozen_hold import reproduce

from aroute.core.model_pool import _request_hash
from scripts.research_feedback_and_transfer import replace_obligations
from scripts.research_v33_checkpoint_trial import _spec
from scripts.research_v40_cases import env_for
from scripts.research_v40_control import Assessment, feedback, reads
from scripts.research_v40_trial import SYSTEM, TOKENS, _json_digest, key


ROOT = Path(__file__).resolve().parent
COHORTS = (
    ("development", "dev", "dev_cases.json"),
    ("formal_original", "formal", "formal_fixtures/cases.json"),
    ("formal_completion", "formal_completion", "formal_fixtures/cases.json"),
    ("accounting_comparison", "shared_allowance/live", "live/shared_allowance/cases.json"),
    ("frozen_continuation", "frozen_hold", "formal_fixtures/cases.json"),
)
FORMAL_SCHEMA_HOLD = "The tool or its arguments did not match the published schema."
ACCOUNTING_SCHEMA_HOLD = "Propose one action matching the published schema."
ACCOUNTING_COMPARISON_HOLD = ("The proposed posting was held. "
                              "Use the current public facts and propose a different operation.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def records(directory):
    db = ROOT / "live" / directory / "meter.sqlite"
    with closing(sqlite3.connect(db.as_uri() + "?mode=ro&immutable=1", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        return list(connection.execute("""
            SELECT attempt_id, run_id, task_id, step_idx, provider, tier,
                   model_requested, model_actual, request_hash, status, response_json
            FROM api_attempts ORDER BY run_id, step_idx, attempt_id
        """))


def as_message_pair(name, raw_arguments, call_id, reply):
    # Same wire shape as assistant_action() and _record() in the archived runner.
    declaration = {"id": call_id, "type": "function",
                   "function": {"name": name, "arguments": raw_arguments}}
    return [{"role": "assistant", "content": "", "tool_calls": [declaration]},
            {"role": "tool", "tool_call_id": call_id, "content": json.dumps(reply, ensure_ascii=False)}]


def automatic_observations(automatic, decision, goal, step, accounting):
    if accounting:
        return [
            (entry["tool"], entry["receipt"], f"receipt_{step}_{i}")
            for i, entry in enumerate(decision.get("receipts", []))
        ]
    if not decision or decision["event"] == "schema_hold":
        require(not automatic, "Receipt without a valid selected call.")
        return []
    observed = iter(automatic)
    receipts, result = {}, []
    for turn in range(2):
        pending = [r for r in reads(decision["candidate"], goal, receipts) if key(r) not in receipts]
        for i, request in enumerate(pending):
            entry = next(observed, None)
            require(entry and entry["candidate"] == request, f"Automatic observation order mismatch at step {step}.")
            payload = entry["reply"]
            if payload["ok"]:
                receipts[key(request)] = payload["data"]
            result.append((request, payload, f"r{step}_{turn}_{i}"))
    require(next(observed, None) is None, f"Unmatched automatic observation at step {step}.")
    return result


def decision_messages(decision, selected, arm, accounting, replayed, step):
    kind = decision.get("disposition") if accounting else decision.get("event")
    if kind == "schema_hold":
        content = ACCOUNTING_SCHEMA_HOLD if accounting else FORMAL_SCHEMA_HOLD
    else:
        require(selected is not None and selected.get("parse_error") is None,
                f"The archived decision at step {step} lacks a parseable initial proposal.")
        actual = {"name": selected["name"], "arguments": json.loads(selected["arguments"])}
        require(decision["candidate"] == actual, f"Initial proposal mismatch at step {step}.")
        if kind == "dispatch":
            call_id = f"checkpoint_{step}" if replayed else f"{{provider-call-id-not-archived:{step}}}"
            return kind, as_message_pair(selected["name"], selected["arguments"], call_id, decision["reply"]), not replayed
        if accounting:
            require(kind in ("comparison_hold", "local_hold"), f"Unexpected accounting disposition {kind}.")
            content = (ACCOUNTING_COMPARISON_HOLD if kind == "comparison_hold"
                       else f"The operation was held: {decision['guard']['reason']}.")
        else:
            require(kind in ("future_block", "local_hold"), f"Unexpected formal disposition {kind}.")
            content = feedback(Assessment(**replace_obligations(decision["assessment"])), arm)
    return kind, [{"role": "user", "content": content}], False


def maps_for_run(cohort, task, row, requests, checkpoint):
    """Yield additive transcript records rather than duplicating entire prefixes."""
    accounting = cohort == "accounting_comparison"
    steps = defaultdict(list)
    for entry in row["trace"]:
        steps[entry["step"]].append(entry)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task["prompt"]}]
    last_reads = {}
    unknown_ids = 0
    prefix_end = checkpoint["trace"][-1]["step"] + 1 if checkpoint else 0
    context_id = f"{cohort}/{row['run_id']}"

    def append_step(step, calls, replayed=False):
        nonlocal unknown_ids
        events = steps.pop(step, [])
        automatic = [e for e in events if e.get("event") == "auto_read"]
        decisions = [e for e in events if e.get("event") != "auto_read"]
        require(len(decisions) == bool(calls), f"Decision/calls cardinality mismatch at {context_id} step {step}.")
        decision = decisions[0] if calls else {}
        delta = []
        observations = automatic_observations(automatic, decision, task["goal"], step, accounting)
        for action, payload, call_id in observations:
            content = json.dumps(payload, ensure_ascii=False)
            if last_reads.get(key(action)) != content:
                raw_arguments = _spec(action, call_id).arguments_raw
                delta.extend(as_message_pair(action["name"], raw_arguments, call_id, payload))
                last_reads[key(action)] = content
        disposition, opaque = "model_returned_no_calls", False
        if calls:
            disposition, new_messages, opaque = decision_messages(decision, calls[0], row["arm"], accounting,
                                                                  replayed, step)
            delta.extend(new_messages)
        messages.extend(delta)
        unknown_ids += opaque
        dispatch = decision["candidate"] if disposition == "dispatch" else None
        return delta, dispatch, disposition, len(observations)

    for step in range(prefix_end):
        decision = next(e for e in steps[step] if e.get("event") != "auto_read")
        candidate = decision["candidate"]
        prefix_call = {"name": candidate["name"], "arguments": _spec(candidate, "").arguments_raw,
                       "parse_error": None}
        append_step(step, [prefix_call], replayed=True)
    base = list(messages)
    require([r["step_idx"] for r in requests] == list(range(prefix_end, prefix_end + len(requests))),
            f"Request step ordinals changed at {context_id}.")
    for i, request in enumerate(requests):
        payload = json.loads(request["response_json"])
        calls = payload["tool_calls"]
        require(all("id" not in call for call in calls), "Adjust reconstruction to use available provider call IDs.")
        exact = not unknown_ids
        if exact:
            calculated = _request_hash(request["provider"], request["tier"], request["model_requested"], messages)
            require(calculated == request["request_hash"], f"Reconstructed request hash differs at {context_id} step {i}.")
        input_unknown_ids = unknown_ids
        delta, dispatched, disposition, auto_reads = append_step(request["step_idx"], calls)
        following = requests[i + 1] if i + 1 < len(requests) else None
        yield {
            "run": context_id, "slot": row["slot"], "arm": row["arm"], "terminal": row["terminal"],
            "request_id": request["attempt_id"], "step": request["step_idx"], "source_task": request["task_id"],
            "provider": request["provider"], "tier": request["tier"], "model": request["model_requested"],
            "request_hash": request["request_hash"], "request_hash_verified": exact,
            "input_unarchived_provider_call_ids": input_unknown_ids,
            **({"initial_context": base} if i == 0 else {}),
            "next_request_id": following["attempt_id"] if following else None,
            "next_request_hash": following["request_hash"] if following else None,
            "stop_reason_in_reply": payload["finish_reason"],
            "proposed_calls": calls, "selected_index": 0 if calls else None,
            "disposition": disposition, "executed_candidate": dispatched,
            "omitted_indices": list(range(1, len(calls))), "guard_auto_reads": auto_reads,
            "context_messages_appended": delta,
        }
    require(not steps, f"Trace steps without a matching request: {context_id}.")


def audit_requests(output):
    registration = read("live/frozen_hold/registration.json")
    checkpoints = {p["id"]: p for p in registration["checkpoints"]}
    summaries = {}
    with (output / "request_call_map_v50.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for name, directory, cases_path in COHORTS:
            report = read(f"live/{directory}/result.json")
            rows = {r["run_id"]: r for r in report["rows"]
                    if name != "formal_completion" or r["origin"] == "completion"}
            tasks = {c["task"]["task_id"]: c["task"] for c in read(cases_path)}
            requests_by_run = defaultdict(list)
            for r in records(directory):
                require(r["status"] == "OK" and r["run_id"] in rows, f"Unmatched model request in {name}.")
                requests_by_run[r["run_id"]].append(r)
            for run_id in rows.keys() - requests_by_run.keys():
                require(not rows[run_id]["trace"], f"{name}/{run_id} has a trace without model requests.")
            counts, omitted_names, omitted_by_provider = Counter(), Counter(), Counter()
            omitted_by_domain_and_outcome = defaultdict(Counter)
            for run, requests in requests_by_run.items():
                row = rows[run]
                require(all(r["task_id"] == row["task"] for r in requests),
                        f"Incorrect task linkage in {name}/{run}.")
                counts["observed_runs"] += 1
                omissions = 0
                task = tasks[row["task"]]
                outcome = {True: "completed", False: "unfinished", None: "unscored_attempt"}[row["terminal"]]
                for entry in maps_for_run(name, task, row, requests, checkpoints.get(row.get("prefix"))):
                    stream.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")
                    dropped = [entry["proposed_calls"][i]["name"] for i in entry["omitted_indices"]]
                    omissions += len(dropped)
                    omitted_names.update(dropped)
                    omitted_by_provider[entry["provider"]] += len(dropped)
                    omitted_by_domain_and_outcome[(task["domain"], outcome)].update(dropped)
                    counts["requests"] += 1
                    counts["byte_exact_message_hash_verified"] += entry["request_hash_verified"]
                    counts["multiple_call_replies"] += bool(dropped)
                    counts["indexed_omitted_calls"] += len(dropped)
                    counts[entry["disposition"]] += 1
                if name == "accounting_comparison":
                    require(omissions == row["later_calls_dropped"], f"Accounting omission total differs in {run}.")
            summaries[name] = {
                "row_count": len(rows), **counts, "omitted_tools": dict(sorted(omitted_names.items())),
                "omitted_calls_by_provider": dict(sorted(omitted_by_provider.items())),
                "omitted_tools_by_domain_and_outcome": [
                    {"domain": domain, "terminal_group": outcome, "counts": dict(sorted(values.items()))}
                    for (domain, outcome), values in sorted(omitted_by_domain_and_outcome.items()) if values
                ],
            }
    return summaries


def audit_feedback():
    repeated = reproduce()
    report = read("live/frozen_hold/result.json")
    protocol = read("live/frozen_hold/registration.json")
    parent = read("live/formal/protocol.json")
    source = {r["slot"]: r for r in read("live/formal_completion/result.json")["rows"]}
    attempts = defaultdict(list)
    for r in records("frozen_hold"):
        attempts[r["run_id"]].append(r)
    require(protocol["max_reply_tokens"] == parent["max_output_tokens"] == TOKENS
            and protocol["max_decisions_including_prefix"] == parent["max_decisions"]
            and protocol["transport"] == parent["transport"], "Frozen continuation settings changed.")
    catalog_path = ROOT / "source_data/archived_models_v50.yaml"
    catalog_bytes = catalog_path.read_bytes()
    config_hash = sha256(catalog_bytes).hexdigest()
    require(config_hash == parent["source_hashes"]["configs/models.yaml"],
            "Model configuration differs from the archived source hash.")
    catalog = yaml.safe_load(catalog_bytes)
    config = catalog["providers"]["dashscope"]["tiers"]["fast"]
    require(config["model_id"] == protocol["models"]["dashscope"], "Registered model and catalog differ.")
    checks = []
    for prefix, outcome in zip(protocol["checkpoints"], repeated["checkpoints"], strict=True):
        expected = prefix["trace"]
        step = expected[-1]["step"]
        require(outcome["id"] == prefix["id"] and expected[-1]["event"] == "future_block",
                "Invalid freeze point or unmatched outcome.")
        for slot in prefix["source_slots"]:
            original = source[slot]
            original_prefix = [e for e in original["trace"] if e["step"] <= step]
            require(original_prefix == expected, f"Historical prefix differs at slot {slot}.")
            require(_json_digest([original["task"], original["provider"], expected]) == prefix["id"],
                    f"Historical prefix key differs at slot {slot}.")
        frozen = [r for r in report["rows"] if r["prefix"] == prefix["id"]]
        hashes = {r["shared_prompt_sha256"] for r in frozen}
        settings = {tuple(attempt[field] for field in ("provider", "tier", "model_requested", "model_actual"))
                    for row in frozen for attempt in attempts[row["run_id"]]}
        require(len(hashes) == 1 and settings == {("dashscope", "fast", protocol["models"]["dashscope"],
                                                  protocol["models"]["dashscope"])}, "Mismatched feedback settings.")
        by_arm = {arm: [r for r in frozen if r["arm"] == arm] for arm in ("M1", "M2")}
        limits = {r["remaining_decisions"] for r in frozen}
        require(limits == {protocol["max_decisions_including_prefix"] - step - 1}
                and {r["checkpoint_state_sha256"] for r in frozen} == {prefix["state_sha256"]},
                "Mismatched feedback state or remaining steps.")
        task = next(c["task"] for c in protocol["cases"] if c["task"]["task_id"] == prefix["task"])
        with closing(env_for(task)) as env:
            tool_hash = _json_digest(env.tool_schemas())
        treatments = {}
        for arm, rows in by_arm.items():
            require(len(rows) == protocol["repeats_per_arm_per_checkpoint"]
                    and len({r["feedback_text"] for r in rows}) == 1, "Incomplete or inconsistent treatment cell.")
            treatments[arm] = {"assigned": len(rows), "completed": sum(r["terminal"] is True for r in rows),
                               "feedback": rows[0]["feedback_text"]}
        checks.append({
            "task": prefix["task"], "historical_slots": prefix["source_slots"], "prefix": prefix["id"],
            "prompt_sha256": hashes.pop(), "state_sha256": prefix["state_sha256"], "tool_schema_sha256": tool_hash,
            "remaining_model_decisions": limits.pop(), "first_request_hashes_verified": len(frozen),
            "settings": dict(zip(("provider", "tier", "requested_model", "reported_model"), settings.pop(), strict=True)),
            "treatments": treatments,
        })
    return {
        "checkpoint_count": len(checks), "checkpoints": checks, "verified_replay": repeated,
        "max_reply_tokens": TOKENS, "transport": protocol["transport"],
        "model_configuration": {"config_hash": config_hash, "temperature": config["temperature"],
                                "enable_thinking": config["extra_body"]["enable_thinking"],
                                "reasoning_effort": config.get("reasoning_effort"), "tool_choice": "auto"},
        "contrast": {
            "M1": ["hold announcement", "insufficient repair allowance"],
            "M2": ["hold announcement", "post-action allowance", "minimum remaining uses",
                   "target service and required revision", "reconsider the prerequisites and shared allowance"],
            "M2_prescribes_specific_call": False,
            "either_directly_guarantees_task_continuation": False,
            "interpretation": "One bundled comparison of evidence content, action cue, and presentation.",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    output = parser.parse_args().out
    output.mkdir(parents=True, exist_ok=True)
    request = audit_requests(output)
    accounting = audit_accounting_calls()
    feedback_report = audit_feedback()
    report = {
        "first_call": {
            "instruction": SYSTEM,
            "omission_notice": False,
            "transcript_orphan_calls_under_runner": False,
            "context_representation": {
                "format": "Within each run, start at initial_context and accumulate context_messages_appended.",
                "evidence": "Archived request hash, raw call names and argument strings, replayed frozen prefix, and trace replies.",
                "unavailable": "Full outbound messages and original provider tool-call IDs are absent from the ledgers.",
                "placeholder": "{provider-call-id-not-archived:STEP}",
                "verification": "Compare the exact archived request hash only until any original provider call ID is needed.",
            },
            "cohorts": request, "accounting_omission_replay": accounting,
            "endpoint_scope": "First-call runner only. Omitted observations may alter future model behavior.",
        },
        "feedback": feedback_report,
    }
    (output / "protocol_v50.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
