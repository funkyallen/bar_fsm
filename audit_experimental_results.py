"""Recompute the experimental tables and sensitivity checks from frozen records."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from decimal import Decimal
import json
from math import isclose
from pathlib import Path
from random import Random
import sys

ROOT = Path(__file__).resolve().parent
ARMS = ("M0", "M1", "M2")
CONTRASTS = (("M1", "M0"), ("M2", "M1"), ("M2", "M0"))
CONDITIONS = ("archived", "conservative_without_authority", "conservative_with_public_receipt")
STAGES = ("certificate_hold", "no_certified_deficit", "local_evidence_hold",
          "resource_unresolved", "local_rule_hold")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def qualification(report):
    def truth(row):
        if row["truth_after"] is None:
            return "inadmissible"
        if not row["truth_before"]:
            return "already_unreachable"
        return "preserved" if row["truth_after"] else "new_loss"

    def disposition(decision):
        return {("unknown", "unknown"): "local_evidence_hold", ("block", "block"): "local_rule_hold",
                ("pass", "unknown"): "resource_unresolved", ("pass", "block"): "certificate_hold",
                ("pass", "pass_checks"): "no_certified_deficit"}[decision["local"], decision["result"]]

    classes = Counter()
    matrix = {condition: defaultdict(Counter) for condition in CONDITIONS}
    transitions = defaultdict(Counter)
    for row in report["rows"]:
        cohort = truth(row)
        classes[cohort] += 1
        stages = [disposition(row["decisions"][condition]) for condition in CONDITIONS]
        for condition, stage in zip(CONDITIONS, stages, strict=True):
            matrix[condition][cohort][stage] += 1
        transitions[cohort][f"{stages[1]} -> {stages[2]}"] += 1
    assert classes == {"inadmissible": 366, "already_unreachable": 62, "new_loss": 36, "preserved": 256}
    matrix = {name: {cohort: {stage: rows[cohort][stage] for stage in STAGES} for cohort in classes}
              for name, rows in matrix.items()}
    return {"oracle_classes": classes, "dispositions": matrix, "same_logic_receipt_transitions": transitions}


def formal(allocation, fixtures, episodes, recorded, oracle):
    cases = {case["task"]["task_id"]: case for case in fixtures}
    assert len(cases) == 24 and len(episodes) == len(allocation) == 288
    losses = {(e["slot"], e["step_zero_based"]): e for e in oracle}
    assert len(losses) == 15
    strata, matched, arms = defaultdict(Counter), defaultdict(dict), defaultdict(Counter)
    unfinished = defaultdict(Counter)
    for expected, row in zip(allocation, sorted(episodes, key=lambda e: e["slot"]), strict=True):
        assert all(row[key] == value for key, value in expected.items())
        task, family, provider, arm = (row[key] for key in ("task", "family", "provider", "arm"))
        domain = "service" if "running" in cases[task]["truth"] else "accounting"
        group = matched[(task, provider, row["repeat"])]
        assert arm not in group
        group[arm] = int(row["terminal"])
        target = "restart_service" if domain == "service" else "post_adjustment"
        relevant = [e for e in row["trace"] if e["candidate"]["name"] == target
                    and e["event"] in {"dispatch", "local_hold", "future_block"}]
        eligible = [e for e in relevant if e.get("assessment", {}).get("local") == "pass"
                    and e.get("oracle", {}).get("before") is True
                    and e["oracle"]["after"] is not None]
        harmful = [e for e in eligible if e["oracle"]["new_loss"]]
        assert all((row["slot"], e["step"]) in losses for e in harmful)
        actual = sum(e["event"] == "dispatch" for e in harmful)
        held = sum(e["event"] == "future_block" for e in harmful)
        counts = {
            "episodes": 1, "episodes_with_relevant_proposal": int(bool(relevant)),
            "eligible_candidates": len(eligible), "episodes_with_harmful_proposal": int(bool(harmful)),
            "harmful_candidates": len(harmful), "actual_losses": actual,
            "episodes_with_actual_loss": int(bool(actual)), "replay_confirmed_holds": held,
            "episodes_with_confirmed_hold": int(bool(held)), "completed_after_hold": int(bool(held) and row["terminal"]),
            "completed": int(row["terminal"]),
        }
        strata[(family, domain, row["quota_group"], provider, arm)].update(counts)
        arms[arm].update(counts)
        if not row["terminal"]:
            unfinished[arm][row["stop"]] += 1
            unfinished[arm]["with_actual_loss"] += bool(actual)
    assert len(matched) == 96
    for arm in ARMS:
        summary = recorded["summary"]["all"][arm]
        assert arms[arm]["completed"] == summary["success"]
        assert arms[arm]["actual_losses"] == summary["actual_reachability_losses"]
        assert arms[arm]["replay_confirmed_holds"] == summary["true_future_blocks"]
        total = Decimal(summary["cost_cny"])
        arms[arm]["total_cny"] = str(total)
        arms[arm]["cny_per_assignment"] = f"{total / arms[arm]['episodes']:.8f}"
        arms[arm]["cny_per_completion"] = f"{total / arms[arm]['completed']:.8f}"
    contrasts = {}
    for treatment, baseline in CONTRASTS:
        name = f"{treatment}-{baseline}"
        differences = defaultdict(list)
        cells = Counter({key: 0 for key in ("both_success", "treatment_only", "baseline_only", "both_fail")})
        for (task, _, _), scores in matched.items():
            assert set(scores) == set(ARMS)
            t, b = scores[treatment], scores[baseline]
            differences[cases[task]["family"]].append(t-b)
            cells[("both_fail", "treatment_only", "baseline_only", "both_success")[t + 2*b]] += 1
        means = {family: sum(values) / len(values) for family, values in differences.items()}
        assert len(means) == 8 and all(len(values) == 12 for values in differences.values())
        rng = Random(20261124)
        draws = sorted(sum(rng.choices(list(means.values()), k=8)) * 100 / 8 for _ in range(5000))
        interval = [draws[125], draws[4874]]
        archived = recorded["paired"][name]
        assert cells == archived["pair_counts"]
        assert all(isclose(a, b, abs_tol=1e-12) for a, b in zip(interval, archived["family_equal_bootstrap_95_pp"], strict=True))
        contrasts[name] = {
            "paired_counts": cells, "pp": sum(means.values()) * 100 / 8,
            "interval_95_pp": interval,
            "leave_one_family_out_pp": {f: (sum(means.values()) - rate) * 100 / 7 for f, rate in sorted(means.items())},
        }
    families = [{
        "family": case["family"], "task": task, "quota_group": case["quota_group"],
        "domain": "service" if "running" in case["truth"] else "accounting",
        "initial_allowance": case["truth"]["quotas"], "fixture_parameters": case["truth"],
    } for task, case in cases.items()]
    return {"starts": families, "policies": arms, "paired": contrasts,
            "unfinished_by_policy": unfinished,
            "strata": [dict(zip(("family", "domain", "quota", "provider", "policy"), group, strict=True), **counts)
                       for group, counts in sorted(strata.items())]}


def accounting(root, report):
    runtime = root / "runtime"
    sys.path[:0] = [str(runtime), str(runtime / "src")]
    from scripts.research_shared_allowance import cases, fixed_candidates

    fixtures = read(root / "live/shared_allowance/cases.json")
    expected = read(root / "live/shared_allowance/fixed_candidates.json")
    assert json.loads(json.dumps(cases())) == fixtures
    assert json.loads(json.dumps(fixed_candidates(fixtures))) == expected
    groups = defaultdict(Counter)
    for case in fixtures:
        task = case["task"]["task_id"]
        truth = case["truth"]
        groups[task]["enumerated"] = sum(min(abs(r), truth["cap"]) for r in truth["residual"])
    for row in expected:
        c = groups[row["task"]]
        c["retained"] += 1
        c["preserved" if row["successor_reachable"] else "new_loss"] += 1
    for counts in groups.values():
        counts["excluded_locally"] = counts["enumerated"] - counts["retained"]
    rows = read(root / "live/shared_allowance/live/result.json")["rows"]
    failures = report["terminal_failures"]
    assert sorted(r["slot"] for r in rows if not r["terminal"]) == sorted(r["slot"] for r in failures)
    assert len(rows) == 64
    return {"fixed_starts": groups, "baseline_counts": report["fixed_candidate_counts"],
            "model_episodes": len(rows), "completed": sum(r["terminal"] for r in rows),
            "unfinished_stop": Counter(r["stop"] for r in rows if not r["terminal"]),
            "unfinished_scorer_flags": report["failure_flags"]}


def feedback(report):
    totals = report["successes"]
    n = report["assigned_pairs"]
    checkpoints = report["checkpoints"]
    per_checkpoint = n // len(checkpoints)
    return {"pairs": report["pairs"], "leave_one_checkpoint_out": {
        row["task"]: {"pairs": n-per_checkpoint, **{
            arm: totals[arm] - row["successes"][arm] for arm in ("M1", "M2")
        }} for row in checkpoints
    }}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root, audits = args.root, args.audit_dir
    episodes = read(root / "live/formal_completion/result.json")["rows"]
    report = {
        "qualification": qualification(read(audits / "voidability_720.json")),
        "formal": formal(read(root / "formal_design.json")["allocation"],
                         read(root / "formal_fixtures/cases.json"), episodes,
                         read(audits / "formal_completion_audit.json")["pooled"],
                         read(audits / "oracle_v50.json")["formal_service_losses"]["cases"]),
        "accounting": accounting(root, read(audits / "shared_allowance_audit.json")),
        "feedback": feedback(read(audits / "frozen_hold_audit.json")),
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
