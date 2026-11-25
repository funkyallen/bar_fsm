"""Audit accounting decisions across receipt and controller versions offline."""

import argparse
import ast
from collections import Counter
from contextlib import closing
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "runtime"), str(ROOT / "runtime" / "src")]

from aroute.envs.db_transaction_env import DbTransactionEnv
from reproduce_paper import audit_accounting_contract
from scripts.research_accounting_void_contract import assess_authoritative_voidability, qualify
from scripts.research_complete_frozen_trial import combined_rows
from scripts.research_replay_live import allowances, normalize
from scripts.research_strict_quota_replay import call, ledger_proposals, same
from scripts.research_v34_intervention import key
from scripts.research_v40_cases import _account_case, env_for, evidence
from scripts.research_v40_control import assess
from scripts.research_voidability_qualification import VoidabilityReceiptEnv, assess_with_receipt


VERSION_FILES = (
    "runtime/src/aroute/envs/db_transaction_env.py",
    "runtime/scripts/research_v40_cases.py",
    "runtime/scripts/research_v40_control.py",
    "runtime/scripts/research_v40_trial.py",
    "runtime/scripts/research_v40_oracle.py",
    "runtime/scripts/research_accounting_void_contract.py",
    "runtime/scripts/research_voidability_qualification.py",
    "runtime/scripts/research_shared_allowance.py",
    "runtime/scripts/research_audit_shared_allowance.py",
    "runtime/scripts/research_v40_validation.py",
    "runtime/scripts/research_v37_lookahead_stress.py",
)


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def digest(data):
    return sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def compare_public_extension():
    path = "runtime/scripts/research_voidability_qualification.py"
    module = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "VoidabilityReceiptEnv")
    if [n.name for n in cls.body if isinstance(n, ast.FunctionDef)] != ["execute_tool"]:
        raise ValueError("The authoritative receipt environment changed its inheritance contract.")
    database_statements = [n.args[0].value.lstrip() for n in ast.walk(cls)
                           if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                           and n.func.attr == "execute"]
    if len(database_statements) != 1 or not database_statements[0].startswith("SELECT "):
        raise ValueError("The extended receipt reads must issue exactly one SELECT.")
    for name in ("reset", "tool_schemas", "snapshot_state", "verify"):
        if getattr(VoidabilityReceiptEnv, name) is not getattr(DbTransactionEnv, name):
            raise ValueError(f"The extended receipt changes {name}.")
    result = audit_accounting_contract()
    return {
        "extension_source": path,
        "additional_sql": database_statements[0],
        "inherited_methods": ["reset", "tool_schemas", "snapshot_state", "verify"],
        "native_contract": result,
        "checked_behavior": "Only successful get_invoice_bundle replies add entry_voidability. No mutation SQL is added.",
    }


def replay_augmented(env, event, ids):
    """Compare the real extension with the archived wire without its one extra field."""
    name = event["candidate"]["name"]
    args = dict(event["candidate"]["arguments"])
    if "entry_id" in args:
        reverse = {old: new for new, old in ids.items()}
        args["entry_id"] = reverse.get(args["entry_id"], args["entry_id"])
    reply = env.execute_tool(name, args)
    wire = json.loads(reply.to_tool_message())
    is_bundle = name == "get_invoice_bundle" and wire["ok"]
    flags = wire["data"].pop("entry_voidability") if is_bundle else None
    wire = normalize(name, wire, event["reply"], ids)
    if not same(wire, event["reply"]):
        raise ValueError(f"Exercise of the observation extension changed the archived reply for {name}.")
    if is_bundle:
        mapped = {ids.get(entry, entry): value for entry, value in flags.items()}
        if mapped.keys() != {e["entry_id"] for e in wire["data"]["ledger_entries"]}:
            raise ValueError("Authoritative flags do not bind exactly to the returned entry set.")
        wire["data"]["entry_voidability"] = mapped
    return reply, wire


def describe(candidate, assessment, arm):
    if assessment.local != "pass":
        disposition = "HOLD_LOCAL"
    elif assessment.result == "block" and arm != "M0":
        disposition = "HOLD_CERTIFICATE"
    else:
        disposition = "DISPATCH"
    if candidate["name"] != "post_adjustment" or assessment.local != "pass":
        resource = "not_evaluated"
    else:
        resource = {"pass_checks": "no_deficit", "unknown": "unresolved", "block": "certified_deficit"}[assessment.result]
    return {**asdict(assessment), "resource": resource, "disposition": disposition}


def replay_accounting_row(stage, row, task):
    archived_ids, augmented_ids = {}, {}
    results, counts = [], Counter()
    with closing(env_for(task)) as native, closing(VoidabilityReceiptEnv()) as extended:
        extended.reset(task)
        if task.get("prefix"):
            raise ValueError("The formal receipt comparison has no prefixed writes.")
        if not same(native.tool_schemas(), extended.tool_schemas()):
            raise ValueError("The observation extension changed the accounting tool list.")
        previous_step, seen_decision = -1, False
        for event in row["trace"]:
            step, kind = event["step"], event["event"]
            if not previous_step <= step < task["max_steps"] or (step == previous_step and seen_decision):
                raise ValueError(f"Out-of-order or duplicate trace decision in {row['run_id']} at {step}.")
            if step != previous_step:
                original, augmented = {}, {}
                previous_step, seen_decision = step, False
            counts[kind] += 1
            seen_decision = kind != "auto_read"
            if kind == "schema_hold":
                continue
            candidate = event["candidate"]
            if kind != "auto_read":
                a = assess(candidate, task["goal"], original)
                b = assess_authoritative_voidability(candidate, task["goal"], original, {})
                c = assess_with_receipt(candidate, task["goal"], augmented)
                da, db, dc = [describe(candidate, decision, row["arm"]) for decision in (a, b, c)]
                saved = {"dispatch": "DISPATCH", "local_hold": "HOLD_LOCAL", "future_block": "HOLD_CERTIFICATE"}
                if not same(asdict(a), event["assessment"]) or da["disposition"] != saved[kind]:
                    raise ValueError(f"Archived decision does not reproduce in {row['run_id']} at {step}.")
                results.append(
                    {"stage": stage, "run_id": row["run_id"], "slot": row["slot"], "step": step,
                     "task": row["task"], "cohort": "scored" if row["terminal"] is not None else "interrupted",
                     "arm": row["arm"], "provider": row["provider"], "candidate": candidate,
                     "original_receipts_sha256": digest(sorted(original.items())),
                     "augmented_receipts_sha256": digest(sorted(augmented.items())),
                     "A": da, "B": db, "C": dc}
                )
            if kind in ("auto_read", "dispatch"):
                before_native = native.conn.total_changes
                before_extended = extended.conn.total_changes
                raw, wire = call(native, event, archived_ids)
                extra, extended_wire = replay_augmented(extended, event, augmented_ids)
                if (raw.is_write, raw.state_changed) != (extra.is_write, extra.state_changed):
                    raise ValueError("The observation extension changed backend state-change semantics.")
                if native.conn.total_changes - before_native != extended.conn.total_changes - before_extended:
                    raise ValueError("The observation extension changed the native mutation count.")
                if kind == "auto_read":
                    if raw.state_changed or raw.is_write or native.conn.total_changes != before_native:
                        raise ValueError("An archived controller read mutated state.")
                    if wire["ok"]:
                        original[key(candidate)] = wire["data"]
                        augmented[key(candidate)] = extended_wire["data"]
                if candidate["name"] == "get_invoice_bundle" and wire["ok"]:
                    counts["extended_bundle_reads"] += 1
        if not same(native.snapshot_state(), extended.snapshot_state()):
            raise ValueError("Final ticket and invoice states differ across backend receipt versions.")
        score = native.verify(task)
        if not same(asdict(score), asdict(extended.verify(task))):
            raise ValueError("Final scores differ across backend versions for the fixed archived action sequence.")
        if row["terminal"] is not None and row["terminal"] != score.passed:
            raise ValueError(f"The frozen terminal score does not reproduce for {row['run_id']}.")
    return results, counts


def decision_matrix(rows):
    fields = ("local", "resource", "disposition")
    result = {"rows": len(rows)}
    for field in fields:
        result[field] = {
            "counts": {version: dict(sorted(Counter(r[version][field] for r in rows).items()))
                       for version in ("A", "B", "C")},
            "transitions": {a + "_to_" + b: dict(sorted(Counter(
                r[a][field] + " -> " + r[b][field] for r in rows
            ).items())) for a, b in (("A", "B"), ("A", "C"), ("B", "C"))},
        }
    return result


def confirm_boundary():
    recorded = read("void_contract.json")
    old_report = qualify()
    if not same(old_report, recorded):
        raise ValueError("The archived free-void experiment could not be reproduced.")
    adverse = _account_case("void-eligibility", (2, -2), (1,), 8, (0, 2), (0, 0), 2, 0, 2)
    for entry in adverse["task"]["initial_state"]["ledger_entries"]:
        if entry["is_duplicate"]:
            entry["ref_code"] = "migrated-ref"
    positive = _account_case("void-proof", (2, 2), (2,), 8, (0, 0), (0, 0), 2, 0, 1)
    cases = {}
    for case, label in ((adverse, "hidden_voidability_counterexample"), (positive, "known_no_void_positive_control")):
        task, candidate = case["task"], case["candidate"]
        with closing(env_for(task)) as native, closing(VoidabilityReceiptEnv()) as extended:
            extended.reset(task)
            receipts = evidence(native, candidate, task["goal"])
            a = assess(candidate, task["goal"], receipts)
            b = assess_authoritative_voidability(candidate, task["goal"], receipts, {})
            c = assess_with_receipt(candidate, task["goal"], evidence(extended, candidate, task["goal"]))
            if not same([asdict(d) for d in (a, b, c)], [recorded[label][field] for field in (
                "old_guard", "missing_authoritative_receipt", "authoritative_receipt"
            )]):
                raise ValueError("The genuine augmented receipt does not match the historical boundary qualification.")
            before = allowances(extended, candidate)
            response = extended.execute_tool(candidate["name"], candidate["arguments"])
            after = allowances(extended, candidate)
            if not response.ok or before - after != 1 or a.q_after != after:
                raise ValueError("The boundary action did not cause its expected native debit.")
            path = []
            if case is adverse:
                for operation in recorded[label]["native_continuation"]:
                    call_result = extended.execute_tool(operation["name"], operation["arguments"])
                    if not call_result.ok or allowances(extended, candidate) != after:
                        raise ValueError("A free correction and its close-out did not reproduce at zero allowance.")
                    path.append({"tool": operation, "reply": call_result.data})
                if not extended.verify(task).passed:
                    raise ValueError("The free-void counterexample failed native terminal validation.")
            if case is positive and (c.result, c.q_after, c.lower) != ("block", 1, 2):
                raise ValueError("The known-no-void positive control does not certify its deficit.")
            cases[label] = {
                "candidate": candidate, "quota_before": before, "quota_after": after,
                "A": describe(candidate, a, "M1"), "B": describe(candidate, b, "M1"), "C": describe(candidate, c, "M1"),
                "completion_witness_executed": path, "native_completion": extended.verify(task).passed,
            }
    return {"archived_experiment_recomputed": True, **cases}


def run():
    manifest = read("baseline_sha256.json")
    sources = {path: sha256((ROOT / path).read_bytes()).hexdigest() for path in VERSION_FILES}
    if any(manifest[path] != value for path, value in sources.items()):
        raise ValueError("An implementation path differs from the frozen v4.9 archive.")
    cases = read("formal_fixtures/cases.json")
    task_by_name = {c["task"]["task_id"]: c["task"] for c in cases}
    previous = read("live/formal/result.json")
    original = previous["rows"]
    later = [read(path.relative_to(ROOT)) for path in sorted((ROOT / "live/formal_completion/rows").glob("*.json"))]
    if {r["slot"] for r in later} != {r["slot"] for r in original if r["terminal"] is None}:
        raise ValueError("Stage two does not match the incomplete stage-one allocations.")
    if not same(combined_rows(previous, {r["slot"]: r for r in later}),
                read("live/formal_completion/result.json")["rows"]):
        raise ValueError("A scored source row does not match the merged terminal archive.")
    rows, stage_counts = [], {}
    for relative, stage, records in (("formal", "original", original), ("formal_completion", "completion", later)):
        ledger = ROOT / "live" / relative / "meter.sqlite"
        summary = ledger_proposals(records, ledger, cases)
        accounting = [r for r in records if task_by_name[r["task"]]["domain"] == "db_transaction"]
        counts = Counter()
        for row in accounting:
            audited, validated = replay_accounting_row(stage, row, task_by_name[row["task"]])
            rows.extend(audited)
            counts.update(validated)
        stage_counts[stage] = {
            "all_domain_ledger_proposals_verified": summary["first_proposals_verified"],
            "accounting_assignments_in_stage_archive": len(accounting),
            "scored": sum(r["terminal"] is not None for r in accounting),
            "unscored_with_trace": sum(r["terminal"] is None and bool(r["trace"]) for r in accounting),
            "unscored_without_trace": sum(r["terminal"] is None and not r["trace"] for r in accounting),
            "accounting_event_counts": dict(sorted(counts.items())),
        }
    scopes = {
        "all_schema_valid_accounting_candidates": rows,
        "scored_schema_valid_accounting_candidates": [r for r in rows if r["cohort"] == "scored"],
        "interrupted_schema_valid_accounting_candidates": [r for r in rows if r["cohort"] == "interrupted"],
        "all_posting_candidates": [r for r in rows if r["candidate"]["name"] == "post_adjustment"],
        "scored_posting_candidates": [r for r in rows if r["candidate"]["name"] == "post_adjustment" and r["cohort"] == "scored"],
        "interrupted_posting_candidates": [r for r in rows if r["candidate"]["name"] == "post_adjustment" and r["cohort"] == "interrupted"],
    }
    return {
        "scope": "Archived proposed actions under fixed historical prefixes; all B and C results are offline re-evaluation.",
        "versions": {
            "A": "Historical controller with archived fresh receipts",
            "B": "Conservative controller with the same archived fresh receipts and no authoritative fields",
            "C": "Conservative controller with the same read schedule and reconstructed authoritative entry flags"
        },
        "source_file_sha256": sources, "stage_counts": stage_counts,
        "observation_only_check": compare_public_extension(),
        "boundary": confirm_boundary(),
        "decision_matrices": {name: decision_matrix(scope) for name, scope in scopes.items()},
        "candidates": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run()
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / "control_versions_v50.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
