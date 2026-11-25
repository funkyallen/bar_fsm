"""Qualify two offline templates with distinct early and late review opportunities."""

from __future__ import annotations

import argparse
import json
from contextlib import closing
from copy import deepcopy
from pathlib import Path

from aroute.core.prewrite_execution import contract_debit
from aroute.envs import create_env
from aroute.envs.ops_troubleshoot_env import check_local_policy
from aroute.orchestrator.prewrite import PrewriteControl, Read

from .research_build_pilot_pairs import canonical_hash, state_differences
from .research_run_pilot import _read_json


class CurrentClusterControl(PrewriteControl):
    """A config change can stale the cluster-wide memory observation of any service."""

    def observe(self, tool, args, result):
        receipt = super().observe(tool, args, result)
        if tool == "rollback_config" and result.ok:
            self.invalidate_tools("inspect_service", "list_config_revisions")
        return receipt


def restart_rule(assessment, control, arguments):
    if assessment is None or assessment.missing:
        return {"decision": "unresolved", "reason": "missing_fresh_receipt"}
    key = Read("inspect_service", "svc_id", arguments["svc_id"])
    receipt = control.receipts[key]
    public = receipt.data
    service = public["service"]
    local = check_local_policy(service["active_params"], service["policy"])
    if local:
        return {"decision": "defer", "reason": "visible_local_policy", "source": receipt.sequence}
    if public["cluster_mem_cap_mb"] > 0 and public["current_total_mem_mb"] > public["cluster_mem_cap_mb"]:
        return {"decision": "defer", "reason": "visible_cluster_memory", "source": receipt.sequence}
    if not service["maint_approved"] or assessment.risk not in ("tight", "relaxed") or service["deps"]:
        return {"decision": "unresolved", "reason": "eligibility_or_upstream_requires_evidence"}
    return {"decision": "keep", "source": receipt.sequence}


def call(name, **arguments):
    return {"name": name, "arguments": arguments}


def accounting(variant):
    ticket, account = f"timing-ticket-{variant}", f"timing-account-{variant}"
    early, late = f"timing-invoice-{variant}-A", f"timing-invoice-{variant}-B"
    positive, negative = 120 + 11 * variant, -(50 + 9 * variant)
    initial = {
        "accounts": [{"acct_id": account, "owner": "Settlement", "balance_cents": 3200 + variant,
                      "min_reserve_cents": 1000, "frozen": 0}],
        "tickets": [{"ticket_id": ticket, "title": "Two invoice corrections", "status": "OPEN", "max_adjustments": 2}],
        "invoices": [{"inv_id": inv, "ticket_id": ticket, "acct_id": account, "subtotal_cents": expected,
                      "tax_rate_bps": 0, "discount_cents": 0, "discount_pre_tax": 0, "status": "OPEN"}
                     for inv, expected in ((early, 2400 + variant), (late, 3100 - variant))],
        "ledger_entries": [{"entry_id": f"opening-{inv}", "ticket_id": ticket, "acct_id": account, "inv_id": inv,
                            "amount_cents": expected - delta, "ref_code": f"opening-{inv}", "is_duplicate": 0, "voided": 0}
                           for inv, expected, delta in ((early, 2400 + variant, positive), (late, 3100 - variant, negative))],
    }
    first = call("post_adjustment", ticket_id=ticket, acct_id=account, inv_id=early, delta_cents=positive)
    second = call("post_adjustment", ticket_id=ticket, acct_id=account, inv_id=late, delta_cents=negative)
    bad = {**second, "arguments": {**second["arguments"], "delta_cents": -17}}
    fix = {**second, "arguments": {**second["arguments"], "delta_cents": negative + 17}}
    finish = [call("run_audit_check", ticket_id=ticket), call("reconcile_invoice", inv_id=early),
              call("reconcile_invoice", inv_id=late), call("close_ticket", ticket_id=ticket)]
    base = {"domain": "db_transaction", "template_id": "one_ticket_two_invoice_adjustments", "initial_state": initial,
            "prompt": f"Reconcile both invoices on ticket {ticket}, applying outstanding corrections before closing it.",
            "assertions": {"closed_tickets": [ticket]}}
    return base, "/tickets/0/max_adjustments", [first, second, *finish], [first, bad, fix, *finish], 2


def service(variant):
    early, late, cache = (f"timing-{variant}-{name}" for name in ("worker", "api", "cache"))
    initial = {
        "cluster_mem_cap_mb": 160 + 6 * variant,
        "services": [{"svc_id": early, "deps": [], "status": "DOWN", "active_rev": "stable", "max_restarts": 2},
                     {"svc_id": late, "deps": [], "status": "DOWN", "active_rev": "old", "max_restarts": 1,
                      "policy": {"min_schema": 2}},
                     {"svc_id": cache, "deps": [], "status": "HEALTHY", "active_rev": "large", "max_restarts": 2}],
        "config_revisions": [{"svc_id": sid, "rev_id": key, "params": {"schema": api, "mem_mb": mem + variant}}
                             for sid, key, api, mem in ((early, "stable", 1, 34), (late, "old", 1, 26),
                                                         (late, "fixed", 2, 97), (cache, "large", 1, 75), (cache, "small", 1, 12))],
        "logs": [{"svc_id": late, "level": "WARN", "message": "Review schema compatibility and shared cluster memory."}],
    }
    prefix = [call("request_maintenance_window", svc_id=early), call("restart_service", svc_id=early),
              call("run_health_probe", svc_id=early), call("request_maintenance_window", svc_id=late),
              call("request_maintenance_window", svc_id=cache), call("rollback_config", svc_id=late, rev_id="fixed")]
    trim = call("rollback_config", svc_id=cache, rev_id="small")
    restart = call("restart_service", svc_id=late)
    probe = call("run_health_probe", svc_id=late)
    base = {"domain": "ops_troubleshoot", "template_id": "healthy_neighbor_shared_memory_release", "initial_state": initial,
            "prompt": f"Restore and probe {early} and {late}. Respect the shared memory limit and keep compatible revisions.",
            "assertions": {"healthy_and_probed": [early, late]}}
    return base, "/services/1/max_restarts", [*prefix, trim, restart, probe], [*prefix, restart, trim, restart, probe], 8


def _change(obj, path):
    keys = path.strip("/").split("/")
    node = obj
    for key in keys[:-1]:
        node = node[int(key)] if isinstance(node, list) else node[key]
    node[keys[-1]] += 1


def run_script(task, path):
    domain = task["domain"]
    cls = CurrentClusterControl if domain == "ops_troubleshoot" else PrewriteControl
    c = cls(domain, supplement=True, review="targeted")
    e = cls(domain, supplement=True, review="first")
    choices, outcomes, rules = {}, [], []
    with closing(create_env(domain)) as env:
        env.reset(task)
        for ordinal, action in enumerate(path):
            name, args = action["name"], action["arguments"]
            assessment = c.assess(name, args)
            if assessment:
                for requirement in assessment.missing:
                    request = requirement.arguments
                    reply = env.execute_tool(requirement.tool, request)
                    if not reply.ok:
                        raise ValueError(f"Fixture lacks the public receipt for {action}.")
                    c.observe(requirement.tool, request, reply)
                    e.observe(requirement.tool, request, reply)
                assessment = c.assess(name, args)
                assert not assessment.missing and not e.assess(name, args).missing
                for label, controller in (("C", c), ("E", e)):
                    if controller.wants_review(assessment, "fast"):
                        controller.review_used = True
                        choices[label] = {"ordinal": ordinal, "remaining": assessment.before["remaining"],
                                          "target": assessment.target, "pool": assessment.pools}
                if name == "restart_service":
                    rules.append({"ordinal": ordinal, **restart_rule(assessment, c, args)})
            outcome = env.execute_tool(name, args)
            c.observe(name, args, outcome)
            e.observe(name, args, outcome)
            outcomes.append({"ordinal": ordinal, "name": name, "ok": outcome.ok, "error": outcome.error_code,
                             "quota_consumed": contract_debit(name, outcome),
                             "observed_remaining": assessment.before.get("remaining") if assessment else None})
        return {"passed": env.verify(task).passed, "dispatches": outcomes, "choices": choices, "rules": rules}


def qualify(output):
    output.mkdir(parents=True)
    tasks, report = [], []
    for builder in (accounting, service):
        for variant in (1, 2, 3):
            base, changed, correct, wrong, blocked_at = builder(variant)
            relaxed = deepcopy(base)
            _change(relaxed["initial_state"], changed)
            assert state_differences(base["initial_state"], relaxed["initial_state"]) == [changed]
            for label, item in (("tight", base), ("relaxed", relaxed)):
                item.update(task_id=f"{base['template_id']}-{variant}-{label}", state=label, family=f"{base['template_id']}-{variant}")
                correct_trace, wrong_trace = run_script(item, correct), run_script(item, wrong)
                assert correct_trace["passed"] and wrong_trace["passed"] == (label == "relaxed")
                assert wrong_trace["dispatches"][blocked_at]["ok"] == (label == "relaxed")
                if label == "tight":
                    assert correct_trace["choices"]["E"]["remaining"] == 2
                    assert correct_trace["choices"]["C"]["remaining"] == 1
                    assert correct_trace["choices"]["E"]["ordinal"] < correct_trace["choices"]["C"]["ordinal"]
                if builder is service:
                    exhausted, = [event for event in wrong_trace["dispatches"] if event["error"] == "CLUSTER_MEM_EXCEEDED"]
                    assert exhausted["quota_consumed"] == 1
                    gate = next(rule for rule in wrong_trace["rules"] if rule["ordinal"] == exhausted["ordinal"])
                    assert gate["reason"] == "visible_cluster_memory"
                tasks.append(item)
                report.append({"task_id": item["task_id"], "domain": item["domain"], "state": label,
                               "planned_state_difference": changed, "correct_trace": correct_trace, "error_trace": wrong_trace})
    archived = [*Path("data").glob("tasks_*.json"), *Path("data/research_pilot").glob("tasks.json")]
    old = {canonical_hash(task["initial_state"]) for path in archived for task in _read_json(path)}
    hashes = [canonical_hash(task["initial_state"]) for task in tasks]
    if len(set(hashes)) != len(tasks) or old.intersection(hashes):
        raise ValueError("A candidate exactly duplicates an archived state.")
    (output / "candidate_tasks.json").write_text(json.dumps(tasks, indent=2) + "\n", encoding="utf-8")
    result = {"scope": "offline feasibility drafts, not formal holdout", "templates": 2, "families": 6,
              "states": len(tasks), "exact_archived_state_duplicates": 0, "results": report}
    (output / "validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    res = qualify(parser.parse_args().out_dir)
    print({key: val for key, val in res.items() if key != "results"})


if __name__ == "__main__":
    main()
