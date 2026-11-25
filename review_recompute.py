"""Recompute v4.1 review diagnostics without model access."""

import argparse
from collections import Counter
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def tally(rows, policies):
    counts = {}
    for name, reject in policies.items():
        labels = Counter((row["reachable_after"], reject(row)) for row in rows)
        counts[name] = {
            "blocked_new_losses": labels[False, True],
            "missed_new_losses": labels[False, False],
            "blocked_preserving": labels[True, True],
            "allowed_preserving": labels[True, False],
        }
    return counts


def service_policies(rows):
    return tally(rows, {
        "stage_target_first": lambda row: row["stage_mismatch"],
        "block_all_last_restarts": lambda row: row["last"],
        "quota_certificate": lambda row: row["certificate"],
    })


def offline_diagnostics(archive):
    selected = [row for row in archive["cases"] if row["evidence"] == "complete"
                and row["truth_before"] and row["truth_after"] is not None and row["controller"]["local"] == "pass"]
    service, accounting = [], []
    for row in selected:
        decision = row["controller"]
        values = {"reachable_after": row["truth_after"], "certificate": decision["result"] == "block"}
        fixture = row["fixture"]
        index = fixture["candidate_index"]
        if row["domain"] == "ops_troubleshoot":
            service.append({**values, "last": fixture["quotas"][index] == 1,
                            "stage_mismatch": fixture["staged"][index] != fixture["required"][index]})
        else:
            delta = row["candidate"]["arguments"]["delta_cents"]
            cap = fixture["cap"]
            limit = min(abs(fixture["residual"][index]), cap, fixture["headroom"] if delta < 0 else cap)
            ticket = fixture["ticket_for"][index]
            remaining_invoices = sum(bool(residual - (delta if i == index else 0))
                                     for i, residual in enumerate(fixture["residual"])
                                     if fixture["ticket_for"][i] == ticket)
            accounting.append({**values, "short_installment": abs(delta) < limit,
                               "invoice_count_deficit": remaining_invoices > fixture["quotas"][ticket] - 1})
    return {
        "scope": "complete-receipt, locally cleared candidates from oracle-feasible starts",
        "service_n": len(service),
        "service": service_policies(service),
        "accounting_n": len(accounting),
        "accounting": tally(accounting, {
            "require_largest_currently_legal_installment": lambda row: row["short_installment"],
            "one_post_per_residual_invoice": lambda row: row["invoice_count_deficit"],
            "quota_certificate": lambda row: row["certificate"],
        }),
    }


def match(rows):
    groups = {}
    for row in rows:
        unit = groups.setdefault((row["task"], row["provider"], row["repeat"]), {})
        unit[row["arm"]] = row
    return list(groups.values())


def pair_bounds(groups):
    bounds = {}
    for old, new in (("M0", "M1"), ("M1", "M2"), ("M0", "M2")):
        counts = Counter()
        lower = upper = 0
        for unit in groups:
            a, b = unit[old]["terminal"], unit[new]["terminal"]
            lower += int(b is True) - int(a is not False)
            upper += int(b is not False) - int(a is True)
            if a is None or b is None:
                counts["at_least_one_missing"] += 1
            elif a == b:
                counts["both_success" if a else "both_fail"] += 1
            else:
                counts["treatment_only" if b else "baseline_only"] += 1
        complete = len(groups) - counts["at_least_one_missing"]
        bounds[f"{new}-{old}"] = {
            "pairs_planned": len(groups), "counts": dict(counts),
            "complete_pair_difference_pp": (counts["treatment_only"] - counts["baseline_only"]) * 100 / complete
            if complete else None,
            "sharp_bounds_pp": [lower * 100 / len(groups), upper * 100 / len(groups)],
        }
    return bounds


def coverage_and_pairs(rows):
    groups = match(rows)
    triplets = [unit for unit in groups if all(row["terminal"] is not None for row in unit.values())]
    never_certified = [unit for unit in groups if all(unit[arm]["terminal"] is not None
                       and not unit[arm]["certified_candidates"] for arm in ("M1", "M2"))]
    strata_pairs = {}
    for unit in groups:
        row = unit["M0"]
        stratum = "/".join(row[key] for key in ("domain", "provider", "quota_group"))
        strata_pairs.setdefault(stratum, []).append(unit)
    exposed_stratum = strata_pairs["ops_troubleshoot/dashscope/tight"]
    families = {unit["M0"]["family"] for unit in exposed_stratum}
    strata = {}
    for column in ("provider", "family", "quota_group"):
        values = {}
        for row in rows:
            result = values.setdefault(row[column], Counter())
            result["assigned"] += 1
            result["assessed"] += row["terminal"] is not None
        strata[column] = {key: dict(value) for key, value in sorted(values.items())}
    return {
        "matched_triplets_planned": len(groups),
        "matched_triplets_assessed": len(triplets),
        "complete_triplets_contrasts": pair_bounds(triplets),
        "M2_minus_M1_pairs_with_no_certification": pair_bounds(never_certified)["M2-M1"],
        "posthoc_stratified_pairs": {key: pair_bounds(value) for key, value in sorted(strata_pairs.items())},
        "tight_qwen_service_leave_one_family_out": {
            family: pair_bounds([unit for unit in exposed_stratum if unit["M0"]["family"] != family])
            for family in sorted(families)
        },
        "by_enrollment_position": [
            {"first": rows[i]["slot"], "last": rows[i+23]["slot"],
             "assessed": sum(row["terminal"] is not None for row in rows[i:i+24])}
            for i in range(0, len(rows), 24)
        ],
        "last_slot_assessed": max(row["slot"] for row in rows if row["terminal"] is not None),
        "strata": strata, "bounds": pair_bounds(groups),
    }


def replay_diagnostics(raw):
    eligible = []
    already_unreachable_certified = 0
    for row in raw["rows"]:
        for event in row["trace"]:
            truth = event.get("oracle", {})
            if (event["event"] not in ("dispatch", "future_block") or event["candidate"]["name"] != "restart_service"
                    or type(truth.get("after")) is not bool):
                continue
            decision = event["assessment"]
            if truth["before"] is not True:
                already_unreachable_certified += decision["result"] == "block"
                continue
            eligible.append({
                "reachable_after": truth["after"], "stage_mismatch": decision["lower"] == 1,
                "last": decision["q_after"] == 0, "certificate": decision["result"] == "block",
                "slot": row["slot"], "step_zero_based": event["step"], "terminal": row["terminal"],
                "candidate": event["candidate"], "candidate_successor_allowance": decision["q_after"],
            })
    return {
        "scope": "observed restart candidates at oracle-feasible prefixes, including withheld proposals",
        "n": len(eligible), "already_unreachable_certified_candidates": already_unreachable_certified,
        "shadow": service_policies(eligible),
        "stage_mismatch_preserving": [row for row in eligible if row["stage_mismatch"] and row["reachable_after"]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-formal", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "offline_comparators": offline_diagnostics(read(HERE / "oracle_results.json")),
        "formal_coverage": coverage_and_pairs(read(HERE / "formal_live_summary.json")["assignments"]),
        "formal_shadow_comparators": replay_diagnostics(read(args.raw_formal)),
    }
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "formal_coverage"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
