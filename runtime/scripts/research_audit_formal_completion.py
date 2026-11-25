"""Validate a second-stage backfill without replacing the original outcomes."""

import argparse
from collections import Counter
from contextlib import closing
import json
from pathlib import Path

from .research_complete_frozen_trial import bill, combined_rows, write
from .research_strict_quota_replay import ledger_proposals, replay
from .research_v40_trial import _read_database, paired_contrasts, summarize


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit(frozen, original, completion):
    cases = load(frozen / "cases.json")
    previous = load(original / "result.json")
    later = load(completion / "result.json")
    observed = {row["slot"]: row for row in (load(p) for p in sorted((completion / "rows").glob("*.json")))}
    if observed.keys() != {row["slot"] for row in previous["rows"] if row["terminal"] is None}:
        raise ValueError("Stage two does not cover exactly the original missing assignments.")
    combined = combined_rows(previous, observed)
    if combined != later["rows"] or later["summary"] != summarize(combined) or later["paired"] != paired_contrasts(combined):
        raise ValueError("The merged result disagrees with the component archives.")
    ledger_path = completion / "meter.sqlite"
    with closing(_read_database(ledger_path)) as db:
        totals = bill(list(db.execute("SELECT * FROM api_attempts")))
    if totals != later["completion_ledger"]:
        raise ValueError("Stage two ledger disagrees with its report.")
    observed_rows = list(observed.values())
    reconstructed = replay(observed_rows, cases)
    proposals = ledger_proposals(observed_rows, ledger_path, cases)
    groups = {}
    for row in combined:
        groups.setdefault((row["task"], row["provider"], row["repeat"]), {})[row["arm"]] = row
    origins = {}
    unexposed = {}
    for control, treatment in (("M0","M1"), ("M1","M2"), ("M0","M2")):
        name = f"{treatment}-{control}"
        origins[name] = dict(Counter(
            ("across_stages" if arms[control]["origin"] != arms[treatment]["origin"] else arms[control]["origin"])
            for arms in groups.values()
        ))
        complete = Counter()
        for arms in groups.values():
            a, b = (arms[arm] for arm in (control, treatment))
            if any(row["terminal"] is None or any(event.get("assessment", {}).get("local") == "pass"
                                                 and event["assessment"]["result"] == "block"
                                                 for event in row["trace"]) for row in (a, b)):
                continue
            passed = (a["terminal"], b["terminal"])
            complete[{(True, True): "both_success", (True, False): "baseline_only",
                      (False, True): "treatment_only", (False, False): "both_fail"}[passed]] += 1
        unexposed[name] = dict(complete)
    family_outcomes = {
        family: {arm: dict(Counter("missing" if row["terminal"] is None else
                                  "success" if row["terminal"] else "failure"
                                  for row in combined if row["family"] == family and row["arm"] == arm))
                 for arm in ("M0", "M1", "M2")}
        for family in sorted({row["family"] for row in combined})
    }
    return {
        "first_stage": {"charge_cny": previous["charge_cny"], "summary": previous["summary"], "paired": previous["paired"]},
        "second_stage": {"ledger": totals, "summary": summarize(observed_rows), "replayed": reconstructed,
                         "ledger_proposals": proposals},
        "pooled": {"charge_cny": later["combined_charge_cny"], "summary": later["summary"], "paired": later["paired"],
                   "pair_stage_membership": origins, "family_terminal_counts": family_outcomes,
                   "pairs_with_no_certificate_on_either_trajectory": unexposed},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.frozen, args.original, args.completion)
    write(args.out, result)
    print({"second_stage_ledger": result["second_stage"]["ledger"], "pooled_paired": result["pooled"]["paired"]})


if __name__ == "__main__":
    main()
