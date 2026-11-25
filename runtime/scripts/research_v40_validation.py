"""Qualify ECRC against independent transition search and reconcile v3.7 ledgers."""

import argparse
import json
import sqlite3
import time
from collections import Counter
from contextlib import closing
from decimal import Decimal
from pathlib import Path

from .research_run_pilot import PILOT_RUNS, _read_json
from .research_v32_observation_pilot import _write_json
from .research_v40_cases import env_for, evidence, offline_cases, paired_cases
from .research_v40_control import assess
from .research_v40_oracle import candidate_truth


def qualify():
    rows = []
    for channel, case in offline_cases():
        task, candidate = case["task"], case["candidate"]
        started = time.perf_counter()
        before, after = candidate_truth(case)
        oracle_ms = (time.perf_counter() - started) * 1000
        with closing(env_for(task)) as env:
            receipts = evidence(env, candidate, task["goal"], omit=channel in ("stale", "missing"),
                                ambiguous=channel == "ambiguous")
            started = time.perf_counter()
            verdict = assess(candidate, task["goal"], receipts)
            guard_ms = (time.perf_counter() - started) * 1000
            backend_ok = None
            if after is not None:
                result = env.execute_tool(candidate["name"], candidate["arguments"])
                backend_ok = result.ok
                if not backend_ok:
                    raise ValueError(f"Independent action dynamics disagree with SQLite at {task['task_id']}: {result.error_code}")
        if after is None or not before:
            label = "local_precondition_failed" if after is None else "initially_unreachable"
        elif verdict.local == "block":
            label = "local_false_block"
        elif verdict.result == "unknown":
            label = "unknown"
        elif verdict.result == "block":
            label = "false_block" if after else "true_block"
        else:
            label = "true_pass" if after else "missed_dead_end"
        rows.append({"task": task["task_id"], "domain": task["domain"], "evidence": channel, "candidate": candidate,
                     "truth_before": before, "truth_after": after, "backend_ok_if_authorized": backend_ok,
                     "controller": verdict.__dict__, "label": label, "oracle_ms": round(oracle_ms, 6),
                     "guard_ms": round(guard_ms, 6), "fixture": case["truth"]})
    breakdown = {domain: Counter(r["label"] for r in rows if r["domain"] == domain)
                 for domain in ("db_transaction", "ops_troubleshoot")}
    positives = [r for r in rows if r["truth_before"] and r["truth_after"] is False]
    negatives = [r for r in rows if r["truth_before"] and r["truth_after"] is True]
    # Avoid counting UNKNOWN as a correct classification.
    summary = {"cases": len(rows), "authorized_candidates": sum(r["truth_after"] is not None for r in rows),
               "oracle_irreversible_transitions": len(positives), "oracle_preserving_transitions": len(negatives),
               "labels": Counter(r["label"] for r in rows), "by_domain": breakdown,
               "by_evidence": {name: Counter(r["label"] for r in rows if r["evidence"] == name)
                               for name in ("complete", "missing", "stale", "ambiguous")}}
    for phase in ("dev", "formal"):
        starts = paired_cases(phase)
        assert all(candidate_truth(c)[0] for c in starts), phase
        summary[f"{phase}_reachable_starts"] = len(starts)
    return {"summary": summary, "cases": rows}


LEDGERS = (
    ("v3.2", ("v32_observation16_live_20261123", "v32_service_replan_live_20261123"), 18, "0.1622518"),
    ("v3.3", ("v33_checkpoint_frozen_20261123/accounting_live", "v33_checkpoint_frozen_20261123/service_live"), 16, "0.0509600"),
    ("v3.6 internal", ("v36_internal_frozen_20261124/accounting_live", "v36_internal_frozen_20261124/service_live"), 64, "0.4319348"),
    ("v3.4", ("v34_natural_pairs_frozen_20261123/live",), 24, "0.1204012"),
    ("Retail V4", ("v35_retail39_frozen_20261124/live",), 117, "5.947868"),
    ("Retail V5", ("v36_retail39_v5_frozen_20261124/live",), 234, "8.804558"),
    ("v3.7 DeepSeek", ("v37_crossmodel_deepseek_20261124/accounting_live", "v37_crossmodel_deepseek_20261124/service_live"), 64, "0.55292232"),
    ("v3.7 stress", ("v37_lookahead_stress_20261124/live",), 24, "0.2253072"),
    ("v3.7 joint", ("v37_joint_composition_20261124/live",), 16, "0.0793024"),
)


def historical():
    audited = []
    for name, paths, count, expected_fee in LEDGERS:
        episodes, attempts = [], []
        for part in paths:
            dbpath = PILOT_RUNS / part / "meter.sqlite"
            with closing(sqlite3.connect(dbpath.as_uri() + "?mode=ro", uri=True)) as db:
                episodes.extend(db.execute("SELECT passed,failure_tag FROM episodes"))
                attempts.extend(db.execute("SELECT status,model_actual,actual_cost_cny FROM api_attempts"))
        fee = sum(Decimal(str(x[2])) for x in attempts)
        assert fee == Decimal(expected_fee) and len(episodes) == count, name
        statuses = Counter(x[0] for x in attempts)
        assert not statuses["UNKNOWN_BILLING"] and not statuses["PENDING"]
        audited.append({"study": name, "logical_episodes": count, "success": sum(x[0] == 1 for x in episodes),
                        "api_attempt_statuses": statuses, "model_ids": sorted({x[1] for x in attempts if x[1]}),
                        "paid_cny": str(fee), "sources": paths})
    assert sum(r["logical_episodes"] for r in audited) == 577
    assert sum(Decimal(r["paid_cny"]) for r in audited) == Decimal("16.37550572")
    for name in ("v34_natural_pairs_frozen_20261123", "v37_lookahead_stress_20261124"):
        result = _read_json(PILOT_RUNS / name / "live" / "result.json")
        rows = result["rows"] if "rows" in result else result["episodes"]
        for row in rows:
            events = row["trace"]
            assert not any(e.get("event") == "quota_block" for e in events)
        if name == "v37_lookahead_stress_20261124":
            stress_stops = {arm: Counter(x["stop"] for x in rows if x["arm"] == arm)
                            for arm in ("ordinary", "ordinary_plus_L")}
    return {"nine_protocols": audited, "observed_total_cost_cny": "16.37550572", "sum_logical_episodes": 577,
            "null_lookahead_triggers_observed_in": 48, "stress_stop_reasons_by_arm": stress_stops}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-out", type=Path)
    parser.add_argument("--historical-out", type=Path)
    args = parser.parse_args()
    if args.oracle_out:
        result = qualify()
        _write_json(args.oracle_out, result)
        print(result["summary"])
    if args.historical_out:
        result = historical()
        _write_json(args.historical_out, result)
        print({"episodes": result["sum_logical_episodes"], "paid": result["observed_total_cost_cny"]})


if __name__ == "__main__":
    main()
