"""Versioned qualification of the public evidence needed for an accounting quota proof."""

import argparse
from contextlib import closing
import json
from pathlib import Path

from .research_v34_intervention import _invoice_residual, action
from .research_v40_cases import _account_case, env_for, evidence
from .research_v40_control import Assessment, assess


def assess_authoritative_voidability(candidate, goal, receipts, voidable):
    """Keys and booleans must come from an authoritative read of the bound ticket.

    If an active entry has no explicit false value, its invoice cannot contribute
    to the lower bound. Declared voidability also governs the corresponding void.
    """
    original = assess(candidate, goal, receipts)
    args = candidate["arguments"]
    if candidate["name"] == "void_duplicate_entry" and original.local != "unknown":
        entry_id = args["entry_id"]
        authorized = any(entry["entry_id"] == entry_id and not entry["voided"]
                         and entry["acct_id"] == goal["authorized_account"]
                         for ticket in goal["authorized_tickets"]
                         for entry in receipts["get_invoice_bundle", ticket]["ledger_entries"])
        if not authorized or voidable.get(entry_id) is False:
            return Assessment("block", "block", "void_not_authorized")
        if voidable.get(entry_id) is not True:
            return Assessment("unknown", "unknown", "voidability_unresolved")
        return Assessment("pass", "pass_checks", "authoritative_void_allowed")
    if original.local != "pass" or candidate["name"] != "post_adjustment":
        return original
    bundle = receipts["get_invoice_bundle", args["ticket_id"]]
    cap = goal["max_per_post_cents"]
    lower = 0
    unresolved = False
    obligations = []
    for invoice in bundle["invoices"]:
        invoice_id = invoice["inv_id"]
        active = [entry for entry in bundle["ledger_entries"] if not entry["voided"] and entry["inv_id"] == invoice_id]
        if invoice_id == args["inv_id"] and any(voidable.get(entry["entry_id"]) is True for entry in active):
            return Assessment("block", "block", "known_voidable_entry_before_posting")
        if any(voidable.get(entry["entry_id"]) is not False for entry in active):
            unresolved = True
            continue
        residual = _invoice_residual(invoice, active)
        if residual is None:
            unresolved = True
            continue
        if invoice_id == args["inv_id"]:
            residual -= args["delta_cents"]
        if residual:
            lower += (abs(residual) + cap - 1) // cap
            obligations.append((invoice_id, residual))
    if original.q_after < lower:
        return Assessment("pass", "block", "verified_no_free_void_residual", original.q_after, lower, tuple(obligations))
    result = "unknown" if unresolved else "pass_checks"
    return Assessment("pass", result, "authoritative_voidability_required" if unresolved else "bound_within_budget",
                      original.q_after, lower, tuple(obligations))


def outcome(case, candidate, following_postings=()):
    task = case["task"]
    with closing(env_for(task)) as env:
        receipts = evidence(env, candidate, task["goal"])
        original = assess(candidate, task["goal"], receipts)
        # Only this test fixture builds a mock authoritative receipt from the backend.
        # The older public API does not supply these facts.
        rows = list(env.conn.execute("SELECT entry_id,is_duplicate FROM ledger_entries"))
        eligibility = {entry_id: bool(allowed) for entry_id, allowed in rows}
        missing = assess_authoritative_voidability(candidate, task["goal"], receipts, {})
        known = assess_authoritative_voidability(candidate, task["goal"], receipts, eligibility)
        reply = env.execute_tool(candidate["name"], candidate["arguments"])
        if not reply.ok:
            raise ValueError("The purported counterexample was not accepted by the backend.")
        continuation = [action("post_adjustment", **{**candidate["arguments"], "delta_cents": delta})
                        for delta in following_postings]
        for entry_id, can_void in eligibility.items():
            if can_void:
                continuation.append(action("void_duplicate_entry", entry_id=entry_id))
        for ticket in case["names"]["tickets"]:
            continuation += [action("run_audit_check", ticket_id=ticket),
                             *(action("reconcile_invoice", inv_id=inv) for inv in case["names"]["invoices"]),
                             action("close_ticket", ticket_id=ticket)]
        corrected_voids = []
        for call in continuation:
            if call["name"] == "void_duplicate_entry":
                fresh = evidence(env, call, task["goal"])
                corrected_voids.append(assess_authoritative_voidability(call, task["goal"], fresh, eligibility).local)
            env.execute_tool(call["name"], call["arguments"])
        return {"old_guard": original.__dict__, "missing_authoritative_receipt": missing.__dict__,
                "authoritative_receipt": known.__dict__, "native_completion_on_continuation": env.verify(task).passed,
                "native_continuation": continuation, "corrected_void_dispositions": corrected_voids}


def qualify():
    counterexample = _account_case("void-eligibility", (2, -2), (1,), 8, (0, 2), (0, 0), 2, 0, 2)
    for entry in counterexample["task"]["initial_state"]["ledger_entries"]:
        if entry["is_duplicate"]:
            entry["ref_code"] = "migrated-ref"
    example = outcome(counterexample, counterexample["candidate"])
    # Both invoices need a posting. Spending one on a partial adjustment leaves one
    # posting available and two residuals to clear.
    control_case = _account_case("void-proof", (2, 2), (2,), 8, (0, 0), (0, 0), 2, 0, 1)
    control = outcome(control_case, control_case["candidate"])
    installment = _account_case("installment-witness", (10,), (3,), 8, (0,), (0,), 4, 0, 1)
    first = installment["candidate"]
    alternative = action("post_adjustment", **{**first["arguments"], "delta_cents": 2})
    lossy = outcome(installment, first, (4, 4))
    preserving = outcome(installment, alternative, (4, 4))
    if not (example["old_guard"]["result"] == "block"
            and example["missing_authoritative_receipt"]["result"] == "unknown"
            and example["authoritative_receipt"]["result"] != "block"
            and example["corrected_void_dispositions"] == ["pass"]
            and example["native_completion_on_continuation"]
            and control["authoritative_receipt"]["result"] == "block"
            and lossy["authoritative_receipt"]["result"] == "block"
            and not lossy["native_completion_on_continuation"]
            and preserving["authoritative_receipt"]["result"] == "pass_checks"
            and preserving["native_completion_on_continuation"]):
        raise ValueError("Voidability qualification did not match the expected counterexample and control.")
    return {
        "scope": "post-hoc boundary qualification with a mock authoritative receipt, no model calls",
        "hidden_voidability_counterexample": example, "known_no_void_positive_control": control,
        "installment_witness": {"initial_residual": 10, "initial_quota": 3, "max_single_post": 4,
                                "candidate_plus_one": lossy, "candidate_plus_two": preserving},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = qualify()
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Authorized voidability boundary and positive control verified.")


if __name__ == "__main__":
    main()
