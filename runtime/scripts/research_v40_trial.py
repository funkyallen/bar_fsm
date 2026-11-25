"""Frozen matched M0/M1/M2 comparisons with separate paid and scripted modes."""

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from random import Random

from openai import APIError

from aroute.core.billing_meter import BillingMeter, BudgetExceededError
from aroute.core.config import PROJECT_ROOT, get_api_key, get_base_url, load_models_catalog
from aroute.core.immutable import audit_orchestrator_ast, sha256_file, verify_manifest
from aroute.core.model_pool import FakeModelPool, LiveModelPool

from .research_run_pilot import DATABASE, _json_digest, _read_database, _read_json
from .research_v32_observation_pilot import _write_json
from .research_v33_checkpoint_trial import _record, _spec
from .research_v34_intervention import action, key
from .research_v40_cases import env_for, paired_cases
from .research_v40_control import assess, feedback, reads
from .research_v40_oracle import candidate_truth


ARMS = ("M0", "M1", "M2")
PROVIDERS = {"dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
             "deepseek": "https://api.deepseek.com/v1"}
CAP = {"dev": 0.75, "formal": 3.0}
TOKENS = 2048
SYSTEM = ("Complete the whole stated business task using only the public tool interface. Current observations "
          "supersede older observations. Follow the user authorization. Propose one tool call at a time. "
          "Stop once the task is complete or cannot be completed, and explain any unmet condition.")


def _sources():
    files = [*Path("scripts").glob("research_v40_*.py"), *Path("src/aroute").rglob("*.py"),
             Path("configs/models.yaml"), Path("scripts/research_v34_intervention.py"),
             Path("scripts/research_v37_lookahead_stress.py")]
    return {f.as_posix(): sha256_file(PROJECT_ROOT / f) for f in sorted(files)}


def freeze(root, phase):
    cases = paired_cases(phase)
    if not all(candidate_truth(c)[0] for c in cases):
        raise ValueError("Initial feasibility gate failed.")
    catalog = load_models_catalog()
    reps = [1] if phase == "dev" else [1, 2]
    blocks = [(c["task"]["task_id"], provider, repeat) for c in cases for provider in PROVIDERS for repeat in reps]
    rng = Random(20261124)
    rng.shuffle(blocks)
    jobs = []
    for tag, provider, repeat in blocks:
        for arm in rng.sample(ARMS, 3):
            jobs.append({"slot": len(jobs)+1, "task": tag, "provider": provider, "repeat": repeat, "arm": arm})
    protocol = {"phase": phase, "task_digest": _json_digest(cases), "source_hashes": _sources(),
                "models": {p: catalog.providers[p].tiers["fast"].resolve_model_id() for p in PROVIDERS},
                "prompt": SYSTEM, "models_use_thinking": False, "max_output_tokens": TOKENS,
                "transport": {"timeout": 120, "retries": 0}, "max_decisions": 28, "max_calls_per_model_decision": 1,
                "branches": {"M0": "common local guard", "M1": "M0 plus certified block, generic feedback",
                             "M2": "same block, structured evidence feedback"},
                "unknown": {"local": "hold", "future": "dispatch after local pass, log uncertified"},
                "allocation": jobs, "cap_cny": CAP[phase], "group_unit": "task family",
                "frozen_during_development": phase == "dev"}
    root.mkdir(parents=True)
    _write_json(root / "cases.json", cases)
    _write_json(root / "protocol.json", protocol)
    return protocol


def _script(case, catalog, meter):
    task, true = case["task"], case["truth"]
    steps = []
    if task["domain"] == "db_transaction":
        ids, tickets = case["names"]["invoices"], case["names"]["tickets"]
        remain = list(true["residual"])
        for i in sorted(range(len(remain)), key=lambda x: -remain[x]):
            while remain[i]:
                part = min(true["cap"], abs(remain[i])) * (1 if remain[i] > 0 else -1)
                steps.append(action("post_adjustment", ticket_id=tickets[true["ticket_for"][i]], inv_id=ids[i],
                                    acct_id=task["goal"]["authorized_account"], delta_cents=part))
                remain[i] -= part
        for j, tid in enumerate(tickets):
            steps.extend([action("run_audit_check", ticket_id=tid),
                          *[action("reconcile_invoice", inv_id=iid) for i, iid in enumerate(ids) if true["ticket_for"][i] == j],
                          action("close_ticket", ticket_id=tid)])
    else:
        for i, sid in enumerate(case["names"]["services"]):
            if true["staged"][i] != true["required"][i]:
                steps.append(action("rollback_config", svc_id=sid, rev_id=task["goal"]["running_revisions"][sid]))
            if not true["healthy"][i] or true["running"][i] != true["required"][i]:
                steps.append(action("restart_service", svc_id=sid))
            steps.append(action("run_health_probe", svc_id=sid))
    queued = iter(steps)

    def reply(_task, _tier, step, _messages, _tools):
        a = next(queued, None)
        spec = {"tool_calls": [{**a, "id": f"qa_{step}"}]} if a else {"content": "Finished"}
        return {**spec, "prompt_tokens": 0, "completion_tokens": 0}

    return FakeModelPool(catalog, meter, reply)


def _valid(call, schema):
    args = call.arguments_dict
    if args is None or schema is None or not set(schema.get("required", [])) <= args.keys() <= schema["properties"].keys():
        return False
    types = {"string": str, "integer": int}
    return all(type(value) is types[schema["properties"][k]["type"]] for k, value in args.items())


def run_branch(case, job, pool, meter, phase):
    task, arm, provider = case["task"], job["arm"], job["provider"]
    rid = f"{phase}_{job['slot']:03d}"
    meter.create_or_resume_run(run_id=rid, stage=f"v40_{phase}", split=phase, policy=arm, provider=provider,
                               round_cap_cny=CAP[phase], stage_cap_cny=CAP[phase], metadata=job)
    meter.start_episode(episode_id=rid, run_id=rid, task_id=task["task_id"], domain=task["domain"], difficulty=phase)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task["prompt"]}]
    trace = []
    last_public_reads = {}
    n_calls, total, terminal, stop = 0, 0.0, None, "decision_limit"
    started = time.perf_counter()
    with closing(env_for(task)) as env:
        tools = env.tool_schemas()
        schemas = {x["function"]["name"]: x["function"]["parameters"] for x in tools}
        try:
            for step in range(task["max_steps"]):
                reply = pool.call_with_tools(run_id=rid, episode_id=rid, task_id=task["task_id"], step_idx=step,
                                             provider=provider, tier="fast", messages=messages, tools=tools,
                                             max_tokens_override=TOKENS)
                n_calls += 1
                total += reply.cost_cny
                if not reply.tool_calls:
                    stop = "output_limit" if reply.finish_reason == "length" else "model_stop"
                    break
                proposed = reply.tool_calls[0]
                if not _valid(proposed, schemas.get(proposed.name)):
                    trace.append({"step": step, "event": "schema_hold"})
                    messages.append({"role": "user", "content": "The tool or its arguments did not match the published schema."})
                    continue
                candidate = action(proposed.name, **proposed.arguments_dict)
                receipts = {}
                for fetch_round in range(2):
                    pending = [request for request in reads(candidate, task["goal"], receipts) if key(request) not in receipts]
                    for i, request in enumerate(pending):
                        result = env.execute_tool(request["name"], request["arguments"])
                        if result.state_changed or result.is_write:
                            raise ValueError("Automatic query mutated the environment.")
                        trace.append({"step": step, "event": "auto_read", "candidate": request,
                                      "reply": json.loads(result.to_tool_message())})
                        rkey = key(request)
                        if last_public_reads.get(rkey) != result.to_tool_message():
                            _record(messages, _spec(request, f"r{step}_{fetch_round}_{i}"), result)
                            last_public_reads[rkey] = result.to_tool_message()
                        if result.ok:
                            receipts[key(request)] = result.data
                decision = assess(candidate, task["goal"], receipts)
                defer = decision.local != "pass" or arm != "M0" and decision.result == "block"
                event = "future_block" if defer and decision.local == "pass" else "local_hold" if defer else "dispatch"
                evidence = {"step": step, "event": event, "candidate": candidate, "assessment": asdict(decision)}
                trace.append(evidence)
                if defer:
                    messages.append({"role": "user", "content": feedback(decision, arm)})
                    continue
                result = env.execute_tool(candidate["name"], candidate["arguments"])
                evidence["reply"] = json.loads(result.to_tool_message())
                _record(messages, proposed, result)
            terminal = env.verify(task).passed
        except (APIError, BudgetExceededError) as exc:
            stop = type(exc).__name__
        finally:
            elapsed = round(time.perf_counter()-started, 6)
            meter.finish_episode(episode_id=rid, passed=bool(terminal), failure_tag=None if terminal else stop,
                                 steps_taken=n_calls, total_cost_cny=round(total,8), loop_abort=False,
                                 summary={"wall_seconds": elapsed})
            meter.finish_run(rid, status="COMPLETED" if terminal is not None else "INTERRUPTED")
    return {**job, "run_id": rid, "family": case.get("family", task["task_id"]), "quota_group": case.get("quota_group", "dev"),
            "terminal": terminal, "stop": stop, "wall_seconds": elapsed, "trace": trace}


def _independent_state(env, case, candidate):
    true = dict(case["truth"])
    task = case["task"]
    args = candidate["arguments"]
    if task["domain"] == "ops_troubleshoot":
        state = [env.snapshot_state()["services"][sid] for sid in case["names"]["services"]]
        true["running"] = [int(x["running_rev"] == "approved") for x in state]
        true["healthy"] = [x["status"] == "HEALTHY" for x in state]
        true["quotas"] = [x["max_restarts"]-x["restarts"] for x in state]
        true["staged"] = [int(x["active_rev"] == "approved") for x in state]
        true["candidate_index"] = case["names"]["services"].index(args["svc_id"])
    else:
        conn = env.conn
        true["residual"], true["duplicates"] = [], []
        for iid in case["names"]["invoices"]:
            expected = conn.execute("SELECT expected_cents FROM invoices WHERE inv_id=?", (iid,)).fetchone()[0]
            total, dup = conn.execute("SELECT COALESCE(SUM(amount_cents),0), COALESCE(SUM(amount_cents*is_duplicate),0) "
                                      "FROM ledger_entries WHERE inv_id=? AND voided=0", (iid,)).fetchone()
            true["residual"].append(expected-total)
            true["duplicates"].append(dup)
        true["quotas"] = [conn.execute("SELECT max_adjustments-adj_count FROM reconciliation_tickets WHERE ticket_id=?",
                                     (tid,)).fetchone()[0] for tid in case["names"]["tickets"]]
        true["headroom"] = conn.execute("SELECT balance_cents-min_reserve_cents FROM accounts WHERE acct_id=?",
                                        (args["acct_id"],)).fetchone()[0]
        true["candidate_index"] = case["names"]["invoices"].index(args["inv_id"])
    return candidate_truth({**case, "truth": true, "candidate": candidate})


def audit(case, row):
    with closing(env_for(case["task"])) as env:
        for item in row["trace"]:
            event = item["event"]
            if event not in ("dispatch", "future_block"):
                continue
            a = item["candidate"]
            if a["name"] in ("post_adjustment", "restart_service"):
                before, after = _independent_state(env, case, a)
                item["oracle"] = {"before": before, "after": after, "new_loss": before is True and after is False}
            if event == "dispatch":
                result = env.execute_tool(a["name"], a["arguments"])
                previous = item["reply"]
                if result.ok != previous["ok"] or result.error_code != previous.get("error_code"):
                    raise ValueError(f"Archived terminal replay tool mismatch at {row['run_id']}")
        if row["terminal"] is not None and env.verify(case["task"]).passed != row["terminal"]:
            raise ValueError(f"Terminal mismatch at {row['run_id']}")


def summarize(rows):
    groups = {}
    for model in ("all", *PROVIDERS):
        bucket = {}
        for arm in ARMS:
            selected = [x for x in rows if x["arm"] == arm and (model == "all" or x["provider"] == model)]
            n = len(selected)
            success = sum(r["terminal"] is True for r in selected)
            triggers = [r for r in selected if any(e.get("assessment",{}).get("local") == "pass" and
                                                    e["assessment"]["result"] == "block" for e in r["trace"])]
            intervened = [r for r in selected if any(e["event"] == "future_block" for e in r["trace"])]
            stop = Counter(r["stop"] for r in selected)
            cost = sum(Decimal(r["paid_cny"]) for r in selected)
            bucket[arm] = {"assigned": n, "success": success, "unresolved": sum(r["terminal"] is None for r in selected),
                           "incidence_episodes": len(triggers), "intervened": len(intervened),
                           "completed_after_intervention": sum(r["terminal"] is True for r in intervened),
                           "actual_reachability_losses": sum(e["event"]=="dispatch" and e.get("oracle",{}).get("new_loss",False)
                                                             for r in selected for e in r["trace"]),
                           "false_future_blocks": sum(e["event"]=="future_block" and e.get("oracle",{}).get("after") is True
                                                      for r in selected for e in r["trace"]),
                           "true_future_blocks": sum(e["event"]=="future_block" and e.get("oracle",{}).get("new_loss",False)
                                                     for r in selected for e in r["trace"]),
                           "calls": sum(r["meter_calls"] for r in selected), "tokens": sum(r["tokens"] for r in selected),
                           "cost_cny": str(cost), "cost_per_success_cny": str(cost/success) if success else None,
                           "stop": stop}
        groups[model] = bucket
    return groups


def paired_contrasts(rows):
    indexed = {}
    for row in rows:
        group = indexed.setdefault((row["task"], row["provider"], row["repeat"]), {})
        group[row["arm"]] = row
    contrasts = {}
    for base, treatment in (("M0", "M1"), ("M1", "M2"), ("M0", "M2")):
        counts, by_family = Counter(), {}
        minimum = maximum = 0
        for block in indexed.values():
            a, b = block[base]["terminal"], block[treatment]["terminal"]
            count = "unresolved" if a is None or b is None else (
                "both_success" if a and b else "both_fail" if not a and not b else "treatment_only" if b else "baseline_only")
            counts[count] += 1
            minimum += (b if b is not None else 0) - (a if a is not None else 1)
            maximum += (b if b is not None else 1) - (a if a is not None else 0)
            if a is not None and b is not None:
                by_family.setdefault(block[base]["family"], []).append(int(b)-int(a))
        bootstrap = None
        if by_family and not counts["unresolved"]:
            rng = Random(20261124)
            means = [sum(v)/len(v) for v in by_family.values()]
            samples = sorted(sum(rng.choices(means, k=len(means)))*100/len(means) for _ in range(5000))
            bootstrap = [samples[125], samples[4874]]
        contrasts[f"{treatment}-{base}"] = {"pair_counts": counts,
                                               "assignment_effect_bounds_pp": [minimum*100/len(indexed), maximum*100/len(indexed)],
                                               "family_equal_bootstrap_95_pp": bootstrap, "families": len(by_family)}
    return contrasts


def execute(frozen, out, budget=None):
    p = _read_json(frozen / "protocol.json")
    case_list = _read_json(frozen / "cases.json")
    if p["source_hashes"] != _sources() or p["task_digest"] != _json_digest(case_list):
        raise ValueError("Frozen protocol or tasks changed.")
    if verify_manifest() or audit_orchestrator_ast():
        raise ValueError("Archived workspace integrity checks failed.")
    phase = p["phase"]
    catalog = load_models_catalog()
    if budget is not None:
        smoke = _read_json(frozen / "offline" / "result.json")
        if not (0 < budget <= CAP[phase] and smoke["scope"] == "scripted" and
                smoke["protocol_sha256"] == _json_digest(p)):
            raise ValueError("Independent authorization and matching offline acceptance required.")
        for prov, url in PROVIDERS.items():
            provider = catalog.providers[prov]
            if not get_api_key(provider) or get_base_url(provider).rstrip("/") != url or (
                    p["models"][prov] != provider.tiers["fast"].resolve_model_id()):
                raise ValueError(f"Model ID, endpoint or credential does not match: {prov}.")
    out.mkdir(parents=True)
    _write_json(out / "protocol.json", {**p, "mode": "live" if budget is not None else "scripted", "approved_cap": budget})
    meter = BillingMeter(out / DATABASE, project_cap_cny=CAP[phase] if budget is None else budget)
    live = LiveModelPool(catalog, meter, timeout_sec=120, max_retries=0) if budget is not None else None
    cases = {c["task"]["task_id"]: c for c in case_list}

    def perform(job):
        c = cases[job["task"]]
        return run_branch(c, job, live or _script(c, catalog, meter), meter, phase)

    rows = []
    with ThreadPoolExecutor(max_workers=3) as workers:
        for n in range(0, len(p["allocation"]), 3):
            rows.extend(workers.map(perform, p["allocation"][n:n+3]))
    for row in rows:
        audit(cases[row["task"]], row)
    with closing(_read_database(out / DATABASE)) as db:
        charges, reserve, status = Decimal(), Decimal(), Counter()
        for row in rows:
            attempts = list(db.execute("SELECT * FROM api_attempts WHERE run_id=?", (row["run_id"],)))
            fee = sum((Decimal(str(x["actual_cost_cny"])) for x in attempts), Decimal())
            charges += fee
            reserve += sum(Decimal(str(x["estimated_max_cost_cny"])) for x in attempts
                           if x["status"] in ("PENDING", "UNKNOWN_BILLING"))
            status.update(x["status"] for x in attempts)
            row.update(paid_cny=str(fee) if live else "0", meter_calls=len(attempts),
                       tokens=sum((x["prompt_tokens"] or 0)+(x["completion_tokens"] or 0) for x in attempts),
                       actual_model_ids=sorted({x["model_actual"] for x in attempts if x["model_actual"]}))
    report = {"scope": "live" if live else "scripted", "protocol_sha256": _json_digest(p), "rows": rows,
              "summary": summarize(rows), "paired": paired_contrasts(rows),
              "reserve_cny": str(reserve) if live else "0", "charge_cny": str(charges) if live else "0",
              "statuses": status}
    if live is None and not all(r["terminal"] is True for r in rows):
        raise ValueError("Scripted acceptance failed; inspect tools and state binding.")
    _write_json(out / "result.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--freeze", choices=("dev","formal"))
    parser.add_argument("--live-out", type=Path)
    parser.add_argument("--budget", type=float)
    args = parser.parse_args()
    if args.freeze:
        print(_json_digest(freeze(args.frozen, args.freeze)))
    elif bool(args.live_out) != (args.budget is not None):
        parser.error("A live output and a budget must be supplied together.")
    else:
        result = execute(args.frozen, args.live_out or args.frozen/"offline", args.budget)
        print({k:v for k,v in result.items() if k != "rows"})


if __name__ == "__main__":
    main()
