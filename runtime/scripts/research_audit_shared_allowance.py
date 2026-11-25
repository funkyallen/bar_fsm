"""Independently replay the four-arm study and report matched development outcomes."""

import argparse
from collections import Counter
from contextlib import closing
from dataclasses import asdict
import json
from pathlib import Path

from .research_complete_frozen_trial import bill, write
from .research_replay_live import allowances, normalize
from .research_shared_allowance import ARMS, flags, summarize
from .research_strict_quota_replay import ledger_proposals
from .research_v34_intervention import key
from .research_v40_oracle import accounting_reachable, action_accounting
from .research_v40_trial import _independent_state, _json_digest, _read_database
from .research_voidability_qualification import VoidabilityReceiptEnv, assess_with_receipt


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def replay(rows, fixtures):
    cases = {case["task"]["task_id"]: case for case in fixtures}
    counts = Counter()
    failures = []
    for row in rows:
        case = cases[row["task"]]
        task = case["task"]
        ids = {}
        with closing(VoidabilityReceiptEnv()) as env:
            env.reset(task)
            for item in row["trace"]:
                counts[item["disposition"]] += 1
                if item["disposition"] == "schema_hold":
                    continue
                observed = {}
                for reading in item["receipts"]:
                    request = reading["tool"]
                    receipt = env.execute_tool(request["name"], request["arguments"])
                    wire = json.loads(receipt.to_tool_message())
                    expected = reading["receipt"]
                    normalize(request["name"], wire, expected, ids)
                    evidence = wire.get("data", {})
                    if "entry_voidability" in evidence:
                        evidence["entry_voidability"] = {ids.get(eid, eid): value
                                                        for eid, value in evidence["entry_voidability"].items()}
                    if wire != expected or receipt.state_changed or receipt.is_write:
                        raise ValueError(f"Unexpected public receipt in slot {row['slot']} step {item['step']}.")
                    if receipt.ok:
                        observed[key(request)] = evidence
                    counts["replayed_receipts"] += 1
                candidate = item["candidate"]
                proof = assess_with_receipt(candidate, task["goal"], observed)
                shadow = flags(candidate, task["goal"], observed, proof)
                expected = ("local_hold" if proof.local != "pass" else
                            "comparison_hold" if shadow[row["arm"]] else "dispatch")
                if _json_digest(asdict(proof)) != _json_digest(item["guard"]) or shadow != item["shadow_blocks"] or expected != item["disposition"]:
                    raise ValueError(f"Wrong disposition in slot {row['slot']} step {item['step']}.")
                posting = candidate["name"] == "post_adjustment" and proof.local == "pass"
                if posting:
                    before, after = _independent_state(env, case, candidate)
                    if item["oracle"] != {"before": before, "after_if_dispatched": after}:
                        raise ValueError(f"Wrong reachability label in slot {row['slot']} step {item['step']}.")
                    counts["validated_postings"] += 1
                if expected == "dispatch":
                    name, args = candidate["name"], dict(candidate["arguments"])
                    if "entry_id" in args:
                        reverse = {saved: actual for actual, saved in ids.items()}
                        args["entry_id"] = reverse.get(args["entry_id"], args["entry_id"])
                    result = env.execute_tool(name, args)
                    saved = item["reply"]
                    wire = normalize(name, json.loads(result.to_tool_message()), saved, ids)
                    data = wire.get("data", {})
                    if "entry_voidability" in data:
                        data["entry_voidability"] = {ids.get(eid, eid): value for eid, value in data["entry_voidability"].items()}
                    if wire != saved or (posting and result.ok and allowances(env, candidate) != proof.q_after):
                        raise ValueError(f"Wrong dispatch receipt or debit in slot {row['slot']} step {item['step']}.")
                    if name == "void_duplicate_entry" and result.ok:
                        counts["successful_public_voids"] += 1
            score = env.verify(task)
            if row["terminal"] is not None and score.passed != row["terminal"]:
                raise ValueError(f"Wrong terminal state in slot {row['slot']}.")
            if row["terminal"] is False:
                failures.append({"slot": row["slot"], "family": row["family"], "arm": row["arm"],
                                 "stop": row["stop"], "violations": score.violations})
    return dict(counts), failures


def pairs(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["task"], row["repeat"]), {})[row["arm"]] = row["terminal"]
    matrix = {}
    for baseline in ("B0", "B1", "B2"):
        count = Counter()
        lo = hi = 0
        for block in groups.values():
            base, method = block[baseline], block["certificate"]
            label = "missing" if base is None or method is None else (
                "both_success" if base and method else "both_fail" if not base and not method else
                "certificate_only" if method else "baseline_only")
            count[label] += 1
            lo += int(method is True) - int(base is not False)
            hi += int(method is not False) - int(base is True)
        matrix[baseline] = {"groups": len(groups), "counts": dict(count),
                            "missing_bounds_pp": [100 * lo / len(groups), 100 * hi / len(groups)]}
    return matrix


def relaxed_demands(fixed, fixtures):
    cases = {case["task"]["task_id"]: case for case in fixtures}
    checks = []
    for row in fixed:
        case = cases[row["task"]]
        truth = case["truth"]
        proposal = row["candidate"]
        index = case["names"]["invoices"].index(proposal["arguments"]["inv_id"])
        residual, _, cash, duplicates = action_accounting(
            truth["residual"], truth["quotas"], truth["headroom"], truth["duplicates"], index,
            proposal["arguments"]["delta_cents"], 0)
        owners = tuple(truth["ticket_for"])
        cap = truth["cap"]
        # Unit-size monotone corrections bound the search in this single-ticket fixture.
        search_limit = sum(abs(amount) + abs(void) for amount, void in zip(residual, duplicates, strict=True))
        demand = next((q for q in range(search_limit + 1)
                       if accounting_reachable(residual, (q,), cash, duplicates, owners, cap)), None)
        estimate = row["guard"]["lower"]
        if demand is not None and estimate > demand:
            raise ValueError("The certificate lower bound exceeded the independently enumerated relaxed demand.")
        if row["successor_reachable"] != (demand is not None and row["guard"]["q_after"] >= demand):
            raise ValueError("Reinstating the actual quota changed the archived truth label.")
        checks.append({"task": row["task"], "candidate": proposal, "proved_lower_bound": estimate,
                       "exact_relaxed_demand": demand, "actual_successor_quota": row["guard"]["q_after"]})
    return checks


def audit_study(folder):
    fixtures = load(folder / "cases.json")
    fixed = load(folder / "fixed_candidates.json")
    raw = load(folder / "live/result.json")
    ledger_path = folder / "live/meter.sqlite"
    rows = raw["rows"]
    with closing(_read_database(ledger_path)) as db:
        ledger = bill(list(db.execute("SELECT * FROM api_attempts")))
    if raw["billing"] != ledger or raw["summary"] != summarize(rows):
        raise ValueError("Ledger or arm summary disagrees with the archive.")
    proposal_rows = [{**row, "trace": [{**item, "event": item["disposition"]} for item in row["trace"]]} for row in rows]
    replayed, failures = replay(rows, fixtures)
    return {"replay": replayed, "terminal_failures": failures, "relaxed_demand_checks": relaxed_demands(fixed, fixtures),
            "failure_flags": {label: sum(any(fragment in item for item in row["violations"]) for row in failures)
                              for label, fragment in (("incorrect_balance", "balance mismatch"),
                                                     ("duplicate_present", "unvoided duplicate"),
                                                     ("invoice_unreconciled", "is not RECONCILED"),
                                                     ("ticket_open", "is not CLOSED"))},
            "ledger_proposals": ledger_proposals(proposal_rows, ledger_path, fixtures),
            "summary": raw["summary"], "billing": ledger, "pairs_certificate_vs": pairs(rows),
            "by_quota": {state: summarize([r for r in rows if r["quota_group"] == state]) for state in ("tight", "wide")},
            "fixed_candidate_counts": {arm: {
                "true_blocks": sum(not row["successor_reachable"] and row["blocked"][arm] for row in fixed),
                "preserving_blocks": sum(row["successor_reachable"] and row["blocked"][arm] for row in fixed),
            } for arm in ARMS}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report = audit_study(args.experiment)
    write(args.out, report)
    print(report)


if __name__ == "__main__":
    main()
