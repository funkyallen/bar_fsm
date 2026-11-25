"""Build the study evidence inventory from archived records and verified audits."""

import argparse
from collections import Counter
from contextlib import closing
import csv
from pathlib import Path
import json
import sqlite3


ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def source(name):
    return read(ROOT / name)


def ledger_counts(path):
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True)) as db:
        return {
            "request_attempts_by_status": dict(db.execute("SELECT status, COUNT(*) FROM api_attempts GROUP BY status")),
            "distinct_attempts_with_requests": db.execute("SELECT COUNT(DISTINCT run_id) FROM api_attempts").fetchone()[0],
        }


def formal_evidence(audits):
    slots = source("formal_design.json")["allocation"]
    rows = source("live/formal_completion/result.json")["rows"]
    original = source("live/formal/result.json")["rows"]
    if len(slots) != 288 or len(rows) != len(slots) or not all(isinstance(r["terminal"], bool) for r in rows):
        raise ValueError("The terminal allocation is incomplete.")
    for planned, observed in zip(slots, rows, strict=True):
        if any(observed[k] != v for k, v in planned.items()):
            raise ValueError(f"Allocation mismatch at slot {planned['slot']}.")
    completion = read(audits / "formal_completion_audit.json")
    forks = completion["second_stage"]["replayed"]["withheld_dispatch_forks"]
    fork_slots = {
        (r["slot"], r["step_zero_based"])
        for r in [*read(audits / "strict_replay.json")["withheld_dispatch_forks"], *forks]
        if r["new_loss_under_oracle"] and r["forked_allowance_after_dispatch"] == 0
    }
    design = {r["task"]["task_id"]: r["task"]["domain"] for r in source("formal_fixtures/cases.json")}

    def bucket(row):
        return design[row["task"]], row["provider"], row["quota_group"], row["arm"]

    terminal_groups, candidate_groups = {}, {}
    # Interrupted attempts can contribute candidates, but each slot has one scored episode.
    older = [r for r in original if r["terminal"] is None and r["trace"]]
    observed = [(r, False) for r in rows] + [(r, True) for r in older]
    for row, interrupted in observed:
        group = bucket(row)
        domain = group[0]
        target = "restart_service" if domain == "ops_troubleshoot" else "post_adjustment"
        candidates = [
            e for e in row["trace"] if e["event"] in {"dispatch", "local_hold", "future_block"}
            and e.get("candidate", {}).get("name") == target
        ]
        eligible = [e for e in candidates if e.get("assessment", {}).get("local") == "pass"
                    and e.get("oracle", {}).get("before") is True and e["oracle"].get("after") is not None]
        losses = [e for e in eligible if e["oracle"]["new_loss"]]
        held = [e for e in losses if e["event"] == "future_block" and (row["slot"], e["step"]) in fork_slots]
        cohort = "interrupted_attempt" if interrupted else "scored_episode"
        candidate_counts = candidate_groups.setdefault((*group, cohort), Counter())
        candidate_counts.update({"proposals": len(candidates), "locally_cleared": sum(
            e.get("assessment", {}).get("local") == "pass" for e in candidates),
            "eligible_known_pre_and_post": len(eligible), "new_losses": len(losses),
            "actual_losses": sum(e["event"] == "dispatch" for e in losses), "replay_confirmed_holds": len(held)})
        if interrupted:
            continue
        episode_counts = terminal_groups.setdefault(group, Counter())
        episode_counts.update({
            "scored": 1, "reached_candidate": bool(candidates), "reached_eligible_candidate": bool(eligible),
            "any_harmful_candidate": bool(losses), "any_actual_loss": any(e["event"] == "dispatch" for e in losses),
            "any_replay_confirmed_hold": bool(held), "completed": row["terminal"],
            "completed_after_replay_confirmed_hold": bool(held) and row["terminal"],
        })

    index = ("domain", "provider", "quota", "arm")
    live = ROOT / "live"
    return {
        "assigned": len(slots), "scored": len(rows),
        "scored_by_collection": dict(Counter(r["origin"] for r in rows)),
        "request_ledgers": {d: ledger_counts(live / d / "meter.sqlite") for d in ("formal", "formal_completion")},
        "both_no_certificate": completion["pooled"]["pairs_with_no_certificate_on_either_trajectory"],
        "workflow": completion["pooled"]["summary"]["all"],
        "service_case_slots": [10, 11, 12],
        "replay_confirmed_candidate_keys": sorted(fork_slots),
        "episodes": [{**dict(zip(index, k)), **v} for k, v in sorted(terminal_groups.items())],
        "candidates": [{**dict(zip((*index, "cohort"), k)), **v} for k, v in sorted(candidate_groups.items())],
    }


def evidence(audits):
    fixed = read(audits / "voidability_720.json")
    editorial = read(audits / "editorial_checks.json")
    account = read(audits / "shared_allowance_audit.json")
    retail = source("retail_quota_boundary.json")
    recovery = read(audits / "frozen_hold_audit.json")
    versions = read(audits / "control_versions_v50.json")
    oracle = read(audits / "oracle_v50.json")
    protocol = read(audits / "protocol_v50.json")
    rows = fixed["rows"]
    natural = {}
    for row in source("live/shared_allowance/live/result.json")["rows"]:
        group = natural.setdefault((row["quota_group"], row["arm"]), Counter())
        postings = [e for e in row["trace"] if e["candidate"]["name"] == "post_adjustment"]
        cleared = [e for e in postings if e["guard"]["local"] == "pass"]
        if any(any(e["shadow_blocks"].values()) for e in cleared):
            raise ValueError("The common-candidate billing audit reported zero disagreements.")
        group.update({"episodes": 1, "reached_posting": bool(postings), "reached_local_pass": bool(cleared),
                      "locally_cleared_candidates": len(cleared), "any_loss": 0, "actual_losses": 0,
                      "replay_confirmed_holds": 0, "shadow_disagreements": 0, "completed": row["terminal"]})
    if sum(v["locally_cleared_candidates"] for v in natural.values()) != \
            editorial["common_candidate_comparison"]["scope"]["preserving"]:
        raise ValueError("Natural accounting decisions and replayed scope do not match.")
    formal = formal_evidence(audits)
    current_losses = [r for r in formal["candidates"] if r["cohort"] == "scored_episode"]
    if oracle["formal_service_losses"]["candidate_counts"] != {
        "isolated_dispatch_of_hold": sum(r["replay_confirmed_holds"] for r in current_losses),
        "observed_dispatch": sum(r["actual_losses"] for r in current_losses),
    }:
        raise ValueError("The new-loss search and logged candidate audit disagree.")
    return {
        "constructed": {"records": len(rows), "by_contract": fixed["summary"],
                        "oracle_qualification": editorial["constructed_sample_classification"],
                        "boundary_controls": fixed["boundary_controls"]},
        "formal": formal,
        "accounting": {"fixed_candidate_counts": account["fixed_candidate_counts"],
                       "natural_comparisons": editorial["common_candidate_comparison"],
                       "natural_by_quota": account["by_quota"], "failure_flags": account["failure_flags"],
                       "natural_exposure": [{"domain": "db_transaction", "provider": "deepseek",
                                             "quota": quota, "arm": arm, **v}
                                            for (quota, arm), v in sorted(natural.items())]},
        "frozen_hold": recovery,
        "retail": {"contract_counts": retail["counts"], "archived": retail["archived_v5"],
                   "scoring_controls": read(audits / "retail_scoring_replay.json")},
        "verification": {
            "accounting_version_matrix": versions["decision_matrices"],
            "oracle_scope": oracle["fixture_checks"],
            "oracle_formal_loss_candidates": oracle["formal_service_losses"]["candidate_counts"],
            "oracle_cutoff": oracle["computational_cutoff_control"],
            "protocol_cohorts": protocol["first_call"]["cohorts"],
            "feedback_checkpoint_count": protocol["feedback"]["checkpoint_count"],
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    result = evidence(args.audit_dir)
    (args.out / "evidence_v50.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for name, groups in (
        ("formal_episode_exposure", result["formal"]["episodes"]),
        ("formal_candidate_exposure", result["formal"]["candidates"]),
        ("natural_accounting_exposure", result["accounting"]["natural_exposure"]),
    ):
        with (args.out / (name + ".csv")).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(groups[0]))
            writer.writeheader()
            writer.writerows(groups)
