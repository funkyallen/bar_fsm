"""Reproduce v4.2's qualification and live-study analyses from packaged files."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(ROOT / "runtime"), str(ROOT / "runtime" / "src")]
    from scripts.research_audit_formal_completion import audit
    from scripts.research_audit_shared_allowance import audit_study
    from scripts.research_voidability_qualification import run

    reports = {
        "voidability_720.json": run(),
        "formal_completion_audit.json": audit(ROOT / "formal_fixtures", ROOT / "live/formal", ROOT / "live/formal_completion"),
        "shared_allowance_audit.json": audit_study(ROOT / "live/shared_allowance"),
    }
    for name, report in reports.items():
        payload = json.loads(json.dumps(report))
        expected = json.loads((ROOT / name).read_text(encoding="utf-8"))
        if payload != expected:
            raise ValueError(f"The archive and recomputed report differ: {name}.")
        (args.out / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Recomputed all three v4.2 reports without model requests.")


if __name__ == "__main__":
    main()
