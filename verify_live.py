"""Recompute the live trial accounting and paired outcomes from preserved inputs."""

import argparse
from collections import Counter, defaultdict
from contextlib import closing
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
from random import Random
import sqlite3


ROOT = Path(__file__).resolve().parent
ARMS = ("M0", "M1", "M2")


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(value):
    wire = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256(wire.encode("utf-8")).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def metrics(rows):
    successful = sum(row["terminal"] is True for row in rows)
    assessed = sum(row["terminal"] is not None for row in rows)
    cost = sum((Decimal(row["cost_cny"]) for row in rows), Decimal(0))
    return {
        "assigned": len(rows),
        "assessed": assessed,
        "success": successful,
        "failure": assessed - successful,
        "attempted_unresolved": sum(row["terminal"] is None and row["requests"] > 0 for row in rows),
        "unstarted_unresolved": sum(row["terminal"] is None and row["requests"] == 0 for row in rows),
        "stops": Counter(row["stop"] for row in rows),
        "certified_episodes": sum(row["certified_candidates"] > 0 for row in rows),
        "intervened": sum(row["blocks"] > 0 for row in rows),
        "completed_after_intervention": sum(row["blocks"] > 0 and row["terminal"] is True for row in rows),
        "true_blocks": sum(row["true_blocks"] for row in rows),
        "false_blocks": sum(row["false_blocks"] for row in rows),
        "loss_dispatches": sum(row["loss_dispatches"] for row in rows),
        "requests": sum(row["requests"] for row in rows),
        "tokens": sum(row["tokens"] for row in rows),
        "cost_cny": str(cost),
        "cost_per_success_cny": str(cost / successful) if successful else None,
    }


def pairs(rows, stratifier=None):
    blocks = defaultdict(dict)
    for row in rows:
        strata = row[stratifier] if stratifier else "all"
        blocks[(strata, row["task"], row["provider"], row["repeat"])][row["arm"]] = row
    report = {}
    for strata in sorted({key[0] for key in blocks}):
        selected = [group for key, group in blocks.items() if key[0] == strata]
        report[strata] = {}
        for old, new in (("M0", "M1"), ("M1", "M2"), ("M0", "M2")):
            counts = Counter()
            differences = defaultdict(list)
            low, high = 0, 0
            for group in selected:
                a, b = group[old]["terminal"], group[new]["terminal"]
                low += int(b is True) - int(a is not False)
                high += int(b is not False) - int(a is True)
                if a is None or b is None:
                    counts["unresolved"] += 1
                    continue
                counts[{(True, True): "both_success", (False, False): "both_fail",
                        (False, True): "treatment_only", (True, False): "baseline_only"}[(a, b)]] += 1
                differences[group[old]["family"]].append(int(b) - int(a))
            interval = None
            if not counts["unresolved"]:
                rng = Random(20261124)
                means = [sum(values) / len(values) for values in differences.values()]
                sampled = sorted(sum(rng.choices(means, k=len(means))) * 100 / len(means) for _ in range(5000))
                interval = [sampled[125], sampled[4874]]
            report[strata][f"{new}-{old}"] = {
                "pairs": len(selected),
                "pair_counts": counts,
                "assignment_effect_bounds_pp": [low * 100 / len(selected), high * 100 / len(selected)],
                "family_equal_bootstrap_95_pp": interval,
                "families_with_complete_pairs": len(differences),
            }
    return report


def verify(directory, phase):
    base = "dev" if phase == "dev" else "formal_design"
    protocol_file = "dev_protocol.json" if phase == "dev" else "formal_design.json"
    source = load(ROOT / protocol_file)
    cases = load(ROOT / f"{base}_cases.json")
    live = load(directory / "protocol.json")
    result = load(directory / "result.json")
    cap = Decimal(str(live.pop("approved_cap")))
    require(live.pop("mode") == result["scope"] == "live", "Expected real model execution.")
    require(live == source and digest(source) == result["protocol_sha256"], "Frozen protocol mismatch.")
    require(digest(cases) == source["task_digest"], "Frozen case mismatch.")
    domains = {case["task"]["task_id"]: case["task"]["domain"] for case in cases}
    rows = []
    output = defaultdict(list)
    finish = defaultdict(Counter)
    status = Counter()
    model_ids = defaultdict(set)
    reserve = Decimal(0)
    with closing(sqlite3.connect((directory / "meter.sqlite").resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        stored = {entry["run_id"]: entry for entry in db.execute(
            "SELECT r.run_id, r.metadata_json, r.status, e.passed, e.task_id, e.failure_tag "
            "FROM runs AS r JOIN episodes AS e ON r.run_id = e.run_id"
        )}
        attempts = defaultdict(list)
        for attempt in db.execute("SELECT * FROM api_attempts ORDER BY run_id, step_idx"):
            attempts[attempt["run_id"]].append(attempt)
        require(set(attempts) <= set(stored), "Orphan API request.")
        require(len(source["allocation"]) == len(stored) == len(result["rows"]), "Missing assigned row.")
        for assigned, row in zip(source["allocation"], result["rows"], strict=True):
            run_id = f"{phase}_{assigned['slot']:03d}"
            record = stored[run_id]
            terminal = row["terminal"]
            require(row["run_id"] == run_id and assigned == json.loads(record["metadata_json"]) ==
                    {key: row[key] for key in assigned}, f"Assignment mismatch in {run_id}.")
            require(record["task_id"] == row["task"] and
                    (record["status"] == "COMPLETED" and record["passed"] == int(terminal) if terminal is not None
                     else record["status"] == "INTERRUPTED" and record["failure_tag"] == row["stop"]),
                    f"Terminal mismatch in {run_id}.")
            provider = row["provider"]
            cost = Decimal(0)
            total = attempts.get(run_id, [])
            tokens = sum((attempt["prompt_tokens"] or 0) + (attempt["completion_tokens"] or 0)
                         for attempt in total)
            for attempt in total:
                require(attempt["provider"] == provider and attempt["model_requested"] == source["models"][provider]
                        and attempt["tier"] == "fast", f"Request configuration mismatch in {run_id}.")
                cost += Decimal(str(attempt["actual_cost_cny"]))
                status[attempt["status"]] += 1
                if attempt["status"] in ("PENDING", "UNKNOWN_BILLING"):
                    reserve += Decimal(str(attempt["estimated_max_cost_cny"]))
                if attempt["status"] == "OK":
                    output[provider].append(attempt["completion_tokens"])
                    finish[provider][json.loads(attempt["response_json"])["finish_reason"]] += 1
                    model_ids[provider].add(attempt["model_actual"])
            require(len(total) == row["meter_calls"] and tokens == row["tokens"] and cost == Decimal(row["paid_cny"]),
                    f"Request ledger mismatch in {run_id}.")
            trace = row["trace"]
            flagged = [e for e in trace if e.get("assessment", {}).get("local") == "pass"
                       and e["assessment"]["result"] == "block"]
            blocked = [e for e in trace if e["event"] == "future_block"]
            require(len(blocked) == (0 if row["arm"] == "M0" else len(flagged)),
                    f"Controller disposition mismatch in {run_id}.")
            rows.append({
                **assigned, "domain": domains[row["task"]], "family": row["family"],
                "quota_group": row["quota_group"], "terminal": terminal, "stop": row["stop"],
                "wall_seconds": row["wall_seconds"], "requests": len(total), "tokens": tokens, "cost_cny": str(cost),
                "certified_candidates": len(flagged), "blocks": len(blocked),
                "true_blocks": sum(e["oracle"]["new_loss"] for e in blocked),
                "false_blocks": sum(e["oracle"]["after"] is True for e in blocked),
                "loss_dispatches": sum(e["event"] == "dispatch" and e.get("oracle", {}).get("new_loss", False)
                                       for e in trace),
            })
    costs = sum(Decimal(row["cost_cny"]) for row in rows)
    require(costs == Decimal(result["charge_cny"]) and reserve == Decimal(result["reserve_cny"]) and
            status == result["statuses"] and costs + reserve <= cap, "Total ledger or budget mismatch.")
    groupings = {}
    for field in ("all", "provider", "domain", "quota_group", "family"):
        labels = {"all"} if field == "all" else {row[field] for row in rows}
        groupings[field] = {label: {arm: metrics([r for r in rows if r["arm"] == arm
                                                 and (field == "all" or r[field] == label)]) for arm in ARMS}
                            for label in sorted(labels)}
    compare = pairs(rows)["all"]
    for contrast, info in compare.items():
        require(all(info[key] == result["paired"][contrast][key] for key in
                    ("pair_counts", "assignment_effect_bounds_pp", "family_equal_bootstrap_95_pp")),
                f"Pair accounting mismatch for {contrast}.")
    for group, arms in (("all", groupings["all"]["all"]), *groupings["provider"].items()):
        for arm, metrics_by_arm in arms.items():
            expected = result["summary"][group][arm]
            require(all(metrics_by_arm[key] == expected[key] for key in ("assigned", "success", "intervened")) and
                    metrics_by_arm["requests"] == expected["calls"] and
                    Decimal(metrics_by_arm["cost_cny"]) == Decimal(expected["cost_cny"]),
                    f"Arm summary mismatch for {group}/{arm}.")
    return {
        "phase": phase, "protocol_sha256": digest(source), "budget_cap_cny": str(cap), "cost_cny": str(costs),
        "reserve_cny": str(reserve), "status": status, "overall": metrics(rows), "groups": groupings,
        "paired": compare, "paired_by_provider": pairs(rows, "provider"), "paired_by_quota": pairs(rows, "quota_group"),
        "output": {provider: {"model_actual": sorted(model_ids[provider]), "finish_reasons": finish[provider],
                              "max_completion_tokens": max(values),
                              "p95_completion_tokens": sorted(values)[(len(values) * 95 + 99) // 100 - 1]}
                   for provider, values in sorted(output.items())},
        "assignments": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--phase", choices=("dev", "formal"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.run_directory, args.phase)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"phase": args.phase, "cost_cny": report["cost_cny"], "overall": report["overall"]}))


if __name__ == "__main__":
    main()
