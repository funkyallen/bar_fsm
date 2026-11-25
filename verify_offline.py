"""Validate the exported truth labels using the independent finite search."""

import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

from oracle import candidate_truth


root = Path(__file__).resolve().parent


def load(name):
    return json.loads((root / name).read_text(encoding="utf-8"))


def main():
    oracle = load("oracle_results.json")
    for row in oracle["cases"]:
        actual = candidate_truth({"truth": row["fixture"], "task": {"domain": row["domain"]},
                                  "candidate": row["candidate"]})
        if actual != (row["truth_before"], row["truth_after"]):
            raise ValueError(f"Independent state search changed for {row['task']}")
    labels = Counter(row["label"] for row in oracle["cases"])
    assert labels == oracle["summary"]["labels"]
    assert len(oracle["cases"]) == 720
    assert not labels["false_block"] and labels["unknown"] == 140
    archive = load("archive_audit.json")["nine_protocols"]
    assert sum(row["logical_episodes"] for row in archive) == 577
    assert sum(Decimal(row["paid_cny"]) for row in archive) == Decimal("16.37550572")
    assert len(load("dev_protocol.json")["allocation"]) == 24
    assert len(load("formal_design.json")["allocation"]) == 288
    print(json.dumps({"oracle_states": 720, "labels": labels, "historical_episodes": 577,
                      "development_slots": 24, "planned_formal_slots": 288}))


if __name__ == "__main__":
    main()
