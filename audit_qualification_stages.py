"""Tabulate the two guard stages for the constructed qualification cases."""
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
import json


CONDITION = "conservative_with_public_receipt"


def audit(report):
    rows = report["rows"]
    assert len(rows) == 720
    categories = ("new_loss", "preserved")
    breakdown = {key: Counter() for key in categories}
    for row in rows:
        if not row["truth_before"] or row["truth_after"] is None:
            continue
        cohort = "preserved" if row["truth_after"] else "new_loss"
        decision = row["decisions"][CONDITION]
        breakdown[cohort][(decision["local"], decision["result"])] += 1
    expected = {
        "new_loss": {("pass", "block"): 21, ("unknown", "unknown"): 15},
        "preserved": {("pass", "pass_checks"): 131, ("pass", "unknown"): 2, ("unknown", "unknown"): 123},
    }
    if breakdown != expected:
        raise ValueError(f"Qualification guard stages differ from the paper: {breakdown}")
    return {
        "scope": "Constructed candidates eligible under the native contract with reachable predecessors",
        "condition": CONDITION,
        "definition": "Disposition under the enforcing guard in Algorithm 1 on the supplied receipts",
        "rows": [
            {"oracle_class": cohort, "local_verdict": local, "resource_verdict": verdict, "candidates": count}
            for cohort in categories
            for (local, verdict), count in sorted(breakdown[cohort].items())
        ],
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(json.loads(args.qualification.read_text(encoding="utf-8")))
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
