"""Run the same-receipt deterministic accounting arm on the archived development checkpoint."""

from __future__ import annotations

import argparse
import json
from contextlib import closing
from hashlib import sha256
from pathlib import Path

from aroute.core.billing_meter import BillingMeter
from aroute.core.config import get_api_key, get_base_url, load_models_catalog
from aroute.core.immutable import audit_orchestrator_ast, sha256_file, verify_manifest
from aroute.core.model_pool import LiveModelPool
from aroute.orchestrator.prewrite import Read

from . import research_v31_accounting_fork as fork
from .research_run_pilot import DATABASE, _json_digest, _read_database
from .research_v31_control_checks import deterministic_invoice_delta


def apply_invoice_rule(control, call):
    if call.name != "post_adjustment" or call.parse_error or call.arguments_dict is None:
        return call, None
    args = call.arguments_dict
    if not {"ticket_id", "inv_id", "acct_id", "delta_cents"} <= args.keys():
        return call, None
    assessment = control.assess(call.name, args)
    if assessment is None or assessment.missing:
        return call, {"decision": "unresolved", "reason": "missing_fresh_public_receipts"}
    invoice = control.receipts[Read("get_invoice_bundle", "ticket_id", args["ticket_id"])]
    account = control.receipts[Read("query_account", "acct_id", args["acct_id"])]
    decision = deterministic_invoice_delta(args, invoice.data, account.data["account"])
    evidence = {**decision, "receipt_sequences": [invoice.sequence, account.sequence]}
    if decision["decision"] == "replace":
        return fork._tool_call(call.id, call.name, {**args, "delta_cents": decision["delta_cents"]}), evidence
    if decision["decision"] == "defer":
        return None, evidence
    return call, evidence


def execute(output, *, budget=None):
    catalog = load_models_catalog()
    provider = catalog.providers[fork.PROVIDER]
    if budget is not None and (not 0 < budget <= 0.8 or not get_api_key(provider)
                               or get_base_url(provider).rstrip("/") != "https://dashscope.aliyuncs.com/compatible-mode/v1"):
        raise ValueError("Use the configured DashScope endpoint and a rule-arm cap of at most 0.8 CNY.")
    if errors := verify_manifest() + audit_orchestrator_ast():
        raise ValueError("Integrity checks failed: " + "; ".join(errors))
    checkpoint = fork.capture(fork.PARENT)
    out = output / "result.json"
    output.mkdir(parents=True)
    transport = {"timeout_seconds": fork.MODEL_TIMEOUT, "retries": 0}
    protocol = {"mode": "live" if budget is not None else "scripted", "cap_cny": budget or 0,
                "source_slot": fork.SLOT, "candidate": checkpoint.candidate.arguments_dict,
                "public_prefix_sha256": _json_digest(checkpoint.ledger.raw_messages),
                "snapshot_sha256": sha256(checkpoint.database).hexdigest(), "P0": checkpoint.params.model_dump(),
                "transport": transport, "rule_source_sha256": sha256_file(Path(__file__)),
                "executor_sha256": sha256_file(Path(fork.__file__))}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    cap = budget or 2
    meter = BillingMeter(output / DATABASE, project_cap_cny=cap)
    pool = LiveModelPool(catalog, meter, timeout_sec=fork.MODEL_TIMEOUT, max_retries=0) if budget else fork._fake_pool("D", catalog, meter, checkpoint)
    result = fork.run_branch("R", checkpoint, pool, meter, cap, proposal_rule=apply_invoice_rule)
    with closing(_read_database(output / DATABASE)) as db:
        rows = [dict(row) for row in db.execute("SELECT status,COUNT(*) n,ROUND(SUM(actual_cost_cny),8) paid,"
                    "ROUND(SUM(estimated_max_cost_cny),8) estimate FROM api_attempts GROUP BY status")]
        models = [row[0] for row in db.execute("SELECT DISTINCT model_actual FROM api_attempts WHERE status='OK' ORDER BY model_actual")]
    report = {**protocol, "branch": result, "attempts": rows, "models_actual": models,
              "external_paid_cny": round(sum(row["paid"] for row in rows), 8) if budget else 0,
              "external_reserve_cny": round(sum(row["estimate"] for row in rows if row["status"] in ("PENDING", "UNKNOWN_BILLING")), 8) if budget else 0}
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if budget is None and (not result["passed"] or result["rule_events"][0]["delta_cents"] != 242):
        raise ValueError("Rule branch did not recover the expected scripted checkpoint.")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--live-budget-cny", type=float)
    args = parser.parse_args()
    result = execute(args.out_dir, budget=args.live_budget_cny)
    print({"passed": result["branch"]["passed"], "paid": result["external_paid_cny"], "reserve": result["external_reserve_cny"]}, flush=True)


if __name__ == "__main__":
    main()
