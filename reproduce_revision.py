"""Recompute the v4.1 audit with the bundled sources and local SQLite copies."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(ROOT / "runtime"), str(ROOT / "runtime" / "src")]
    from scripts.research_accounting_void_contract import qualify
    from scripts.research_strict_quota_replay import ledger_proposals, qualify_frozen_cases, replay, tamper_checks
    from review_recompute import coverage_and_pairs, offline_diagnostics, replay_diagnostics

    cases = load(ROOT / "formal_design_cases.json")
    raw = load(ROOT / "live" / "formal" / "result.json")
    ledger = ROOT / "live" / "formal" / "meter.sqlite"
    oracle = load(ROOT / "oracle_results.json")
    verified = replay(raw["rows"], cases)
    verified["ledger_proposals"] = ledger_proposals(raw["rows"], ledger, cases)
    verified["rejected_tamper_checks"] = tamper_checks(raw["rows"], cases, ledger)
    verified["reproduced_offline"] = qualify_frozen_cases(oracle)
    reports = {
        "strict_replay.json": verified,
        "void_contract.json": qualify(),
        "review_diagnostics.json": {
            "offline_comparators": offline_diagnostics(oracle),
            "formal_coverage": coverage_and_pairs(load(ROOT / "formal_live_summary.json")["assignments"]),
            "formal_shadow_comparators": replay_diagnostics(raw),
        },
    }
    for name, report in reports.items():
        normalized = json.loads(json.dumps(report))
        if normalized != load(ROOT / name):
            raise ValueError(f"Recomputed evidence differs from the published report: {name}.")
        (args.out / name).write_text(json.dumps(normalized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Recomputed {len(reports)} reports. No model requests were made.")


if __name__ == "__main__":
    main()
