"""Qualify the accounting certificate under two explicit receipt contracts."""

import argparse
from collections import Counter
from contextlib import closing
from dataclasses import asdict
from pathlib import Path

from aroute.envs.db_transaction_env import DbTransactionEnv

from .research_accounting_void_contract import assess_authoritative_voidability, qualify
from .research_complete_frozen_trial import write
from .research_v40_cases import env_for, evidence, offline_cases
from .research_v40_control import assess
from .research_v40_oracle import candidate_truth


class VoidabilityReceiptEnv(DbTransactionEnv):
    """Publish per-entry void permission as part of the current invoice-bundle receipt."""

    def execute_tool(self, name, args):
        result = super().execute_tool(name, args)
        if name == "get_invoice_bundle" and result.ok:
            rows = self.conn.execute(
                "SELECT entry_id, is_duplicate, voided FROM ledger_entries WHERE ticket_id=?",
                (args["ticket_id"],),
            )
            result.data["entry_voidability"] = {
                row["entry_id"]: bool(row["is_duplicate"] and not row["voided"]) for row in rows
            }
        return result


def public_voidability(receipts):
    return {entry: allowed for (name, _), receipt in receipts.items() if name == "get_invoice_bundle"
            for entry, allowed in receipt.get("entry_voidability", {}).items()}


def assess_with_receipt(candidate, goal, receipts):
    return assess_authoritative_voidability(candidate, goal, receipts, public_voidability(receipts))


def classify(before, after, verdict):
    if after is None:
        return "ineligible"
    if not before:
        return "initially_unreachable"
    if verdict.local == "block":
        return "local_policy_hold"
    if verdict.result == "unknown":
        return "unknown_preserving" if after else "unknown_loss"
    if verdict.result == "block":
        return "false_block" if after else "true_block"
    return "true_pass" if after else "missed_loss"


def run():
    rows = []
    for channel, case in offline_cases():
        task, candidate = case["task"], case["candidate"]
        options = {"omit": channel in ("stale", "missing"), "ambiguous": channel == "ambiguous"}
        before, after = candidate_truth(case)
        with closing(env_for(task)) as env:
            original = assess(candidate, task["goal"], evidence(env, candidate, task["goal"], **options))
        if task["domain"] == "db_transaction":
            with closing(VoidabilityReceiptEnv()) as native:
                native.reset(task)
                observed = evidence(native, candidate, task["goal"], **options)
                public = assess_with_receipt(candidate, task["goal"], observed)
                missing = assess_authoritative_voidability(candidate, task["goal"], observed, {})
        else:
            public = missing = original
        policies = {"archived": original, "conservative_without_authority": missing,
                    "conservative_with_public_receipt": public}
        rows.append({"task": task["task_id"], "domain": task["domain"], "candidate": candidate,
                     "evidence": channel, "truth_before": before, "truth_after": after,
                     "decisions": {name: {**asdict(verdict), "label": classify(before, after, verdict)}
                                   for name, verdict in policies.items()}})
    summaries = {}
    for label in rows[0]["decisions"]:
        summaries[label] = {
            "all": dict(Counter(row["decisions"][label]["label"] for row in rows)),
            "accounting": dict(Counter(row["decisions"][label]["label"] for row in rows
                                       if row["domain"] == "db_transaction")),
        }
    transitions = Counter(
        (row["evidence"], row["decisions"]["archived"]["label"])
        for row in rows if row["decisions"]["conservative_without_authority"]["label"].startswith("unknown")
        and row["decisions"]["archived"]["label"] in ("true_pass", "true_block")
    )
    return {"scope": "constructed 720-case qualification; no model requests",
            "summary": summaries, "new_unknown_without_authority":
                [{"evidence": channel, "old_label": label, "count": count}
                 for (channel, label), count in sorted(transitions.items())],
            "boundary_controls": qualify(), "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write(args.out, result)
    print(result["summary"])


if __name__ == "__main__":
    main()
