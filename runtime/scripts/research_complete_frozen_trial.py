"""Complete only the missing assignments of the frozen v4.0 formal trial."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from decimal import Decimal
from pathlib import Path

from aroute.core.billing_meter import BillingMeter
from aroute.core.immutable import audit_orchestrator_ast, sha256_file, verify_manifest
from aroute.core.model_pool import LiveModelPool

from .research_v40_trial import (
    CAP, DATABASE, PROVIDERS, _json_digest, _read_database, _read_json, _script,
    _sources, _write_json, audit, get_api_key, get_base_url, load_models_catalog,
    paired_contrasts, run_branch, summarize,
)


def write(path, value):
    temp = path.with_suffix(".tmp")
    _write_json(temp, value)
    temp.replace(path)


def bill(attempts):
    amounts = (Decimal(str(item["actual_cost_cny"])) for item in attempts)
    reserved = (Decimal(str(item["estimated_max_cost_cny"])) for item in attempts
                if item["status"] in ("PENDING", "UNKNOWN_BILLING"))
    return {
        "paid_cny": str(sum(amounts, Decimal(0))),
        "reserve_cny": str(sum(reserved, Decimal(0))),
        "meter_calls": len(attempts),
        "tokens": sum((item["prompt_tokens"] or 0) + (item["completion_tokens"] or 0) for item in attempts),
        "actual_model_ids": sorted({item["model_actual"] for item in attempts if item["model_actual"]}),
        "statuses": Counter(item["status"] for item in attempts),
    }


def model_pool(catalog, models, meter):
    for provider, url in PROVIDERS.items():
        config = catalog.providers[provider]
        if (not get_api_key(config) or get_base_url(config).rstrip("/") != url or
                config.tiers["fast"].resolve_model_id() != models[provider]):
            raise ValueError(f"Model, endpoint or credential does not match {provider}.")
    return LiveModelPool(catalog, meter, timeout_sec=120, max_retries=0)


def load_registry(frozen, prior):
    protocol = _read_json(frozen / "protocol.json")
    cases = _read_json(frozen / "cases.json")
    previous = _read_json(prior / "result.json")
    protocol_sha = _json_digest(protocol)
    if (protocol["phase"] != "formal" or protocol["source_hashes"] != _sources()
            or protocol["task_digest"] != _json_digest(cases) or previous["scope"] != "live"
            or previous["protocol_sha256"] != protocol_sha or verify_manifest() or audit_orchestrator_ast()):
        raise ValueError("Frozen v4.0 source, fixtures, or historical results did not verify.")
    if _read_json(prior / "protocol.json") != {**protocol, "mode": "live", "approved_cap": CAP["formal"]}:
        raise ValueError("The original live protocol does not match the frozen allocation.")
    with closing(_read_database(prior / DATABASE)) as db:
        accounting = bill(list(db.execute("SELECT * FROM api_attempts")))
    if (Decimal(accounting["paid_cny"]) != Decimal(previous["charge_cny"]) or
            Decimal(accounting["reserve_cny"]) != Decimal(previous["reserve_cny"])):
        raise ValueError("The original report and ledger disagree.")
    missing = {}
    for job, row in zip(protocol["allocation"], previous["rows"], strict=True):
        if job != {field: row[field] for field in job}:
            raise ValueError("The original report has a changed assignment.")
        if row["terminal"] is None:
            missing[job["slot"]] = {"assignment": job, "previous_stop": row["stop"],
                                    "previous_requests": row["meter_calls"]}
    registration = {
        "protocol_sha256": protocol_sha,
        "original_result_sha256": sha256_file(prior / "result.json"),
        "completion_runner_sha256": sha256_file(Path(__file__)),
        "order": "Original matched blocks and arm order restricted to missing slots.",
        "partially_attempted": "Start afresh from the identical frozen initial state; retain old attempts and fees.",
        "assignments": list(missing.values()),
    }
    return protocol, {case["task"]["task_id"]: case for case in cases}, previous, missing, registration


def combined_rows(previous, completion):
    merged = []
    for original in previous["rows"]:
        new = completion.get(original["slot"])
        if new is None:
            merged.append({**original, "origin": "original"})
            continue
        if original["terminal"] is not None:
            raise ValueError(f"Cannot replace a previously evaluated slot {original['slot']}.")
        merged.append({
            **new, "origin": "completion", "previous_stop": original["stop"],
            "previous_calls": original["meter_calls"], "previous_paid_cny": original["paid_cny"],
            "meter_calls": original["meter_calls"] + new["meter_calls"],
            "tokens": original["tokens"] + new["tokens"],
            "paid_cny": str(Decimal(original["paid_cny"]) + Decimal(new["paid_cny"])),
            "actual_model_ids": sorted(set(original["actual_model_ids"]) | set(new["actual_model_ids"])),
        })
    return merged


def execute(frozen, prior, out, budget=None):
    protocol, cases, previous, missing, registration = load_registry(frozen, prior)
    registration_sha = _json_digest(registration)
    out.mkdir(parents=True, exist_ok=True)
    register = out / "registration.json"
    if register.exists() and _read_json(register) != registration:
        raise ValueError("Completion registration or source changed.")
    write(register, registration)
    offline = budget is None
    if not offline:
        acceptance = _read_json(out / "offline" / "result.json")
        if (not 0 < budget <= CAP["formal"] or acceptance["scope"] != "scripted"
                or acceptance["registration_sha256"] != registration_sha
                or acceptance["new_assessed"] != len(missing)):
            raise ValueError("Matching offline acceptance and an authorized completion cap are required.")
    session = out / ("offline" if offline else "live")
    session.mkdir(exist_ok=True)
    budget_file = session / "budget.json"
    if budget_file.exists() and _read_json(budget_file)["cap_cny"] != budget:
        raise ValueError("The completion ledger cap is fixed once execution starts.")
    write(budget_file, {"cap_cny": budget})
    row_dir = session / "rows"
    row_dir.mkdir(exist_ok=True)
    completed = {row["slot"]: row for row in (_read_json(path) for path in row_dir.glob("*.json"))}
    if not completed.keys() <= missing.keys():
        raise ValueError("The completion directory contains an unexpected slot.")
    catalog = load_models_catalog()
    meter = BillingMeter(session / DATABASE, project_cap_cny=CAP["formal"] if offline else budget)
    pool = None if offline else model_pool(catalog, protocol["models"], meter)
    with closing(_read_database(session / DATABASE)) as db:
        seen = {row["run_id"] for row in db.execute("SELECT run_id FROM runs")}
    orphaned = sorted(slot for slot in missing if f"formal_{slot:03d}" in seen and slot not in completed)

    def run(job):
        case = cases[job["task"]]
        row = run_branch(case, job, pool or _script(case, catalog, meter), meter, "formal")
        audit(case, row)
        with closing(_read_database(session / DATABASE)) as db:
            accounting = bill(list(db.execute("SELECT * FROM api_attempts WHERE run_id=?", (row["run_id"],))))
        row.update(accounting)
        write(row_dir / f"{job['slot']:03d}.json", row)
        return row

    to_run = missing.keys() - completed.keys() - set(orphaned)
    with ThreadPoolExecutor(max_workers=3) as workers:
        for start in range(0, len(protocol["allocation"]), 3):
            jobs = [job for job in protocol["allocation"][start:start + 3] if job["slot"] in to_run]
            completed.update((row["slot"], row) for row in workers.map(run, jobs))
    with closing(_read_database(session / DATABASE)) as db:
        accounting = bill(list(db.execute("SELECT * FROM api_attempts")))
    if not offline and Decimal(accounting["paid_cny"]) + Decimal(accounting["reserve_cny"]) > Decimal(str(budget)):
        raise ValueError("Completion ledger exceeded the approved cap.")
    report = {
        "scope": "scripted" if offline else "live", "registration_sha256": registration_sha,
        "assigned_to_completion": len(missing), "new_assessed": sum(row["terminal"] is not None for row in completed.values()),
        "unrecorded_slots_with_started_ledger": orphaned, "completion_ledger": accounting,
    }
    if offline and (report["new_assessed"] != len(missing) or not all(row["terminal"] for row in completed.values())):
        raise ValueError("Scripted completion did not pass every missing slot.")
    if not offline:
        merged = combined_rows(previous, completed)
        report.update(
            rows=merged, protocol_sha256=previous["protocol_sha256"],
            original_charge_cny=previous["charge_cny"],
            combined_charge_cny=str(Decimal(previous["charge_cny"]) + Decimal(accounting["paid_cny"])),
            summary=summarize(merged), paired=paired_contrasts(merged),
        )
    write(session / "result.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", required=True, type=Path)
    parser.add_argument("--prior", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--budget", type=float)
    args = parser.parse_args()
    result = execute(args.frozen, args.prior, args.out, args.budget)
    print({key: value for key, value in result.items() if key != "rows"})


if __name__ == "__main__":
    main()
