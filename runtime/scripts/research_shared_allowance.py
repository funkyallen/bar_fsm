"""Compare four accounting policies under a public, versioned receipt contract."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import asdict
from decimal import Decimal
import json
from pathlib import Path
from random import Random

from openai import APIError

from aroute.core.billing_meter import BillingMeter, BudgetExceededError
from aroute.core.config import get_api_key, get_base_url, load_models_catalog
from aroute.core.immutable import sha256_file
from aroute.core.model_pool import FakeModelPool, LiveModelPool

from .research_complete_frozen_trial import bill, write
from .research_v33_checkpoint_trial import _record, _spec
from .research_v34_intervention import _invoice_residual, action, key
from .research_v40_cases import _account_case, evidence
from .research_v40_control import reads
from .research_v40_oracle import candidate_truth
from .research_v40_trial import DATABASE, PROVIDERS, SYSTEM, TOKENS, _independent_state, _json_digest, _read_database, _valid
from .research_voidability_qualification import VoidabilityReceiptEnv, assess_with_receipt


ARMS = ("B0", "B1", "B2", "certificate")
PROVIDER = "deepseek"
CAP = 1.50


def cases():
    # The structures and their quotas are chosen before natural model proposals.
    structures = (
        ("capacity-shared", (11, 3), 4, 15, (0, 0)),
        ("reserve-dependency", (10, -6), 4, 2, (0, 0)),
        ("zero-cost-alternative", (6, -2, 3), 3, 20, (0, 2, 0)),
        ("routine-two-invoices", (3, 4), 4, 8, (0, 0)),
    )
    result = []
    for index, (family, remaining, cap, headroom, duplicates) in enumerate(structures):
        minimum = sum((abs(a + d) + cap - 1) // cap for a, d in zip(remaining, duplicates, strict=True))
        for j, slack in enumerate((0, 1)):
            tag = f"settlement-{index * 2 + j + 1:02d}"
            case = _account_case(tag, remaining, (minimum + slack,), headroom, duplicates, (0,) * len(remaining),
                                 cap, 0, 1)
            case.update(family=family, quota_group="tight" if slack == 0 else "wide")
            for entry in case["task"]["initial_state"]["ledger_entries"]:
                if entry["is_duplicate"]:
                    entry["ref_code"] = f"amended-{entry['entry_id']}"
            case["task"]["prompt"] += (" The current invoice bundle includes an authoritative entry_voidability map. "
                                       "Use it to identify records eligible for a free void.")
            if not candidate_truth(case)[0]:
                raise ValueError(f"Infeasible prespecified start: {tag}.")
            result.append(case)
    return result


def flags(candidate, goal, receipts, decision):
    """All comparators use the same locally validated candidate and public receipt."""
    blocked = dict.fromkeys(ARMS, False)
    if candidate["name"] != "post_adjustment" or decision.local != "pass":
        return blocked
    args = candidate["arguments"]
    bundle = receipts["get_invoice_bundle", args["ticket_id"]]
    account = receipts["query_account", args["acct_id"]]["account"]
    invoice = next(row for row in bundle["invoices"] if row["inv_id"] == args["inv_id"])
    residual = _invoice_residual(invoice, bundle["ledger_entries"])
    max_current = min(abs(residual), goal["max_per_post_cents"])
    if residual < 0:
        max_current = min(max_current, account["balance_cents"] - account["min_reserve_cents"])
    blocked["B1"] = decision.q_after < len(decision.obligations)
    blocked["B2"] = abs(args["delta_cents"]) < max_current
    blocked["certificate"] = decision.result == "block"
    return blocked


def fixed_candidates(fixtures):
    rows = []
    for case in fixtures:
        task, truth = case["task"], case["truth"]
        aid = task["goal"]["authorized_account"]
        ticket, = case["names"]["tickets"]
        with closing(VoidabilityReceiptEnv()) as env:
            env.reset(task)
            for i, (invoice, residual) in enumerate(zip(case["names"]["invoices"], truth["residual"], strict=True)):
                for amount in range(1, min(abs(residual), truth["cap"]) + 1):
                    candidate = action("post_adjustment", ticket_id=ticket, inv_id=invoice, acct_id=aid,
                                       delta_cents=amount if residual > 0 else -amount)
                    receipts = evidence(env, candidate, task["goal"])
                    decision = assess_with_receipt(candidate, task["goal"], receipts)
                    if decision.local != "pass":
                        continue
                    before, after = candidate_truth({**case, "candidate": candidate,
                                                     "truth": {**truth, "candidate_index": i}})
                    if not before or after is None:
                        raise ValueError("Complete-candidate audit disagrees with the local guard.")
                    rows.append({"task": task["task_id"], "family": case["family"], "quota_group": case["quota_group"],
                                 "candidate": candidate, "successor_reachable": after, "guard": asdict(decision),
                                 "blocked": flags(candidate, task["goal"], receipts, decision)})
    if any(row["successor_reachable"] and row["blocked"]["certificate"] for row in rows):
        raise ValueError("The certificate blocked a preserving write in the offline fixture audit.")
    if not any(not row["successor_reachable"] and row["blocked"]["certificate"] and not row["blocked"]["B1"] for row in rows):
        raise ValueError("The chosen starts lack a candidate that separates the certificate and invoice counting.")
    return rows


def script(case, catalog, meter):
    task, truth = case["task"], case["truth"]
    invoice_ids = case["names"]["invoices"]
    ticket, = case["names"]["tickets"]
    aid = task["goal"]["authorized_account"]
    commands = [action("void_duplicate_entry", entry_id=entry["entry_id"])
                for entry in task["initial_state"]["ledger_entries"] if entry["is_duplicate"]]
    remaining = [a + d for a, d in zip(truth["residual"], truth["duplicates"], strict=True)]
    for i in sorted(range(len(invoice_ids)), key=lambda index: -remaining[index]):
        while remaining[i]:
            delta = min(truth["cap"], abs(remaining[i])) * (1 if remaining[i] > 0 else -1)
            commands.append(action("post_adjustment", ticket_id=ticket, inv_id=invoice_ids[i],
                                   acct_id=aid, delta_cents=delta))
            remaining[i] -= delta
    commands += [action("run_audit_check", ticket_id=ticket),
                 *(action("reconcile_invoice", inv_id=invoice) for invoice in invoice_ids),
                 action("close_ticket", ticket_id=ticket)]
    pending = iter(commands)

    def respond(_task, _tier, step, _messages, _tools):
        item = next(pending, None)
        answer = {"tool_calls": [dict(item, id=f"verify_{step}")]} if item else {"content": "Finished"}
        return {**answer, "prompt_tokens": 0, "completion_tokens": 0}

    return FakeModelPool(catalog, meter, respond)


def run(case, job, pool, meter):
    task, arm = case["task"], job["arm"]
    run_id = f"settlement_{job['slot']:03d}"
    meter.create_or_resume_run(run_id=run_id, stage="shared_allowance", split="development", policy=arm,
                               provider=PROVIDER, round_cap_cny=CAP, stage_cap_cny=CAP, metadata=job)
    meter.start_episode(episode_id=run_id, run_id=run_id, task_id=task["task_id"], domain=task["domain"],
                        difficulty=case["quota_group"])
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task["prompt"]}]
    trace, last_reads = [], {}
    terminal, stop = None, "decision_limit"
    attempts, charged, later_calls = 0, 0.0, 0
    with closing(VoidabilityReceiptEnv()) as env:
        env.reset(task)
        tools = env.tool_schemas()
        schemas = {tool["function"]["name"]: tool["function"]["parameters"] for tool in tools}
        try:
            for step in range(task["max_steps"]):
                reply = pool.call_with_tools(run_id=run_id, episode_id=run_id, task_id=task["task_id"], step_idx=step,
                                             provider=PROVIDER, tier="fast", messages=messages, tools=tools,
                                             max_tokens_override=TOKENS)
                attempts += 1
                charged += reply.cost_cny
                later_calls += max(0, len(reply.tool_calls) - 1)
                if not reply.tool_calls:
                    stop = "output_limit" if reply.finish_reason == "length" else "model_stop"
                    break
                proposal = reply.tool_calls[0]
                if not _valid(proposal, schemas.get(proposal.name)):
                    trace.append({"step": step, "disposition": "schema_hold"})
                    messages.append({"role": "user", "content": "Propose one action matching the published schema."})
                    continue
                candidate = action(proposal.name, **proposal.arguments_dict)
                receipts = {}
                observations = []
                for index, request in enumerate(reads(candidate, task["goal"], receipts)):
                    result = env.execute_tool(request["name"], request["arguments"])
                    if result.is_write or result.state_changed:
                        raise ValueError("A pre-write receipt mutated business state.")
                    payload = result.to_tool_message()
                    if result.ok:
                        receipts[key(request)] = result.data
                    observations.append({"tool": request, "receipt": json.loads(payload)})
                    if last_reads.get(key(request)) != payload:
                        _record(messages, _spec(request, f"receipt_{step}_{index}"), result)
                        last_reads[key(request)] = payload
                proof = assess_with_receipt(candidate, task["goal"], receipts)
                decisions = flags(candidate, task["goal"], receipts, proof)
                record = {"step": step, "candidate": candidate, "receipts": observations, "guard": asdict(proof),
                          "shadow_blocks": decisions}
                trace.append(record)
                if proof.local != "pass":
                    record["disposition"] = "local_hold"
                    messages.append({"role": "user", "content": f"The operation was held: {proof.reason}."})
                    continue
                if candidate["name"] == "post_adjustment":
                    before, after = _independent_state(env, case, candidate)
                    record.update(oracle={"before": before, "after_if_dispatched": after})
                if decisions[arm]:
                    record["disposition"] = "comparison_hold"
                    messages.append({"role": "user", "content": "The proposed posting was held. "
                                     "Use the current public facts and propose a different operation."})
                    continue
                result = env.execute_tool(candidate["name"], candidate["arguments"])
                record.update(disposition="dispatch", reply=json.loads(result.to_tool_message()))
                _record(messages, proposal, result)
            terminal = bool(env.verify(task).passed)
        except (APIError, BudgetExceededError) as exc:
            stop = type(exc).__name__
        finally:
            meter.finish_episode(episode_id=run_id, passed=bool(terminal), failure_tag=None if terminal else stop,
                                 steps_taken=attempts, total_cost_cny=round(charged, 8), loop_abort=False,
                                 summary={"terminal": terminal, "later_calls_dropped": later_calls})
            meter.finish_run(run_id, status="COMPLETED" if terminal is not None else "INTERRUPTED")
    return {**job, "run_id": run_id, "terminal": terminal, "stop": stop, "trace": trace,
            "later_calls_dropped": later_calls}


def summarize(rows):
    arms = {}
    for arm in ARMS:
        subset = [row for row in rows if row["arm"] == arm]
        proposals = [event for row in subset for event in row["trace"] if "oracle" in event]
        newly_lost = [event for event in proposals if event["oracle"] == {"before": True, "after_if_dispatched": False}]
        preserved = [event for event in proposals if event["oracle"] == {"before": True, "after_if_dispatched": True}]
        arms[arm] = {"assigned": len(subset), "assessed": sum(row["terminal"] is not None for row in subset),
                     "success": sum(row["terminal"] is True for row in subset), "locally_cleared_postings": len(proposals),
                     "new_losses_dispatched": sum(event["disposition"] == "dispatch" and event["reply"]["ok"]
                                                  for event in newly_lost),
                     "new_losses_held": sum(event["disposition"] == "comparison_hold" for event in newly_lost),
                     "preserving_held": sum(event["disposition"] == "comparison_hold" for event in preserved),
                     "certificate_only_losses_vs_B1": sum(event["shadow_blocks"]["certificate"] and
                                                          not event["shadow_blocks"]["B1"] for event in newly_lost),
                     "preserving_allowed_by_certificate_vs_B2": sum(not event["shadow_blocks"]["certificate"] and
                                                                     event["shadow_blocks"]["B2"] for event in preserved),
                     "requests": sum(row["meter_calls"] for row in subset),
                     "cost_cny": round(sum(float(row["paid_cny"]) for row in subset), 8),
                     "later_calls_dropped": sum(row["later_calls_dropped"] for row in subset),
                     "stops": Counter(row["stop"] for row in subset)}
    return arms


def execute(out, budget=None):
    fixtures = cases()
    fixed = fixed_candidates(fixtures)
    catalog = load_models_catalog()
    model = catalog.providers[PROVIDER].tiers["fast"].resolve_model_id()
    matched = [(case["task"]["task_id"], case["family"], case["quota_group"], repeat)
               for case in fixtures for repeat in (1, 2)]
    rng = Random(20261124)
    rng.shuffle(matched)
    jobs = []
    for task, family, quota_group, repeat in matched:
        for arm in rng.sample(ARMS, len(ARMS)):
            jobs.append({"slot": len(jobs)+1, "task": task, "family": family, "quota_group": quota_group,
                         "repeat": repeat, "arm": arm})
    source_files = (Path(__file__), Path(__file__).with_name("research_voidability_qualification.py"),
                    Path(__file__).with_name("research_accounting_void_contract.py"))
    registration = {"study": "shared_allowance_public_voidability_v1", "provider": PROVIDER, "model": model,
                    "sources": {path.name: sha256_file(path) for path in source_files}, "fixture_digest": _json_digest(fixtures),
                    "shared_hold_feedback": True, "timeout_sec": 120, "retries": 0,
                    "decision_limit": 28, "output_tokens": TOKENS, "allocation": jobs}
    signature = _json_digest(registration)
    out.mkdir(parents=True, exist_ok=True)
    prior = out / "registration.json"
    if prior.exists() and json.loads(prior.read_text(encoding="utf-8")) != registration:
        raise ValueError("The existing registration has a different experiment definition.")
    write(prior, registration)
    write(out / "cases.json", fixtures)
    write(out / "fixed_candidates.json", fixed)
    offline = budget is None
    if not offline:
        smoke = json.loads((out / "offline/result.json").read_text(encoding="utf-8"))
        config = catalog.providers[PROVIDER]
        if (not 0 < budget <= CAP or smoke["signature"] != signature or smoke["mode"] != "scripted"
                or sum(value["success"] for value in smoke["summary"].values()) != len(jobs)
                or not get_api_key(config) or get_base_url(config).rstrip("/") != PROVIDERS[PROVIDER]):
            raise ValueError("The paid batch needs matching offline results and the registered provider and budget.")
    batch = out / ("offline" if offline else "live")
    batch.mkdir(exist_ok=False)
    meter = BillingMeter(batch / DATABASE, project_cap_cny=CAP if offline else budget)
    live = None if offline else LiveModelPool(catalog, meter, timeout_sec=120, max_retries=0)
    indexed = {case["task"]["task_id"]: case for case in fixtures}

    def perform(job):
        case = indexed[job["task"]]
        row = run(case, job, live or script(case, catalog, meter), meter)
        with closing(_read_database(batch / DATABASE)) as db:
            row.update(bill(list(db.execute("SELECT * FROM api_attempts WHERE run_id=?", (row["run_id"],)))))
        write(batch / f"slot_{row['slot']:03d}.json", row)
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=len(ARMS)) as workers:
        for index in range(0, len(jobs), len(ARMS)):
            rows.extend(workers.map(perform, jobs[index:index+len(ARMS)]))
    if offline and not all(row["terminal"] is True for row in rows):
        raise ValueError("Scripted paths must complete in all arms before the paid study.")
    with closing(_read_database(batch / DATABASE)) as db:
        ledger = bill(list(db.execute("SELECT * FROM api_attempts")))
    if not offline and Decimal(ledger["paid_cny"]) + Decimal(ledger["reserve_cny"]) > Decimal(str(budget)):
        raise ValueError("Combined paid and reserved cost exceeds the study cap.")
    result = {"mode": "scripted" if offline else "live", "signature": signature,
              "summary": summarize(rows), "billing": ledger, "rows": rows}
    write(batch / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--budget", type=float)
    args = parser.parse_args()
    print({k: v for k, v in execute(args.out, args.budget).items() if k != "rows"})


if __name__ == "__main__":
    main()
