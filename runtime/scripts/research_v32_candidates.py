"""Four prespecified development structures and offline feasibility witnesses."""

from __future__ import annotations

from contextlib import closing
from copy import deepcopy

from aroute.envs import create_env
from aroute.envs.db_transaction_env import compute_expected_cents

from .research_v31_timing_candidates import call
from .research_v32_replay import ACCOUNTING, SERVICE


def _ops(suffix, title, services, revisions, goals):
    return {"family": suffix, "domain": SERVICE, "prompt": title,
            "initial_state": {"cluster_mem_cap_mb": 250, "services": services,
                              "config_revisions": revisions, "logs": []},
            "assertions": {"healthy_and_probed": goals}}


def _service(revisions, service, active, *, policy=None, deps=(), healthy=False):
    return {"svc_id": service, "active_rev": active, "status": "HEALTHY" if healthy else "DOWN",
            "deps": list(deps), "policy": policy or {}, "maint_approved": 0 if healthy else 1, "max_restarts": 1}, [
        {"svc_id": service, "rev_id": revision, "params": {"mem_mb": 30, **params}}
        for revision, params in revisions.items()
    ]


def service_fanout():
    hub, left, right = "directory", "scheduler", "notifier"
    upstream, upstream_revs = _service({"legacy": {"schema": 1}, "compatible": {"schema": 2}}, hub, "legacy", healthy=True)
    client = {"upstream_reqs": {hub: {"min_schema": 2}}}
    west, west_revs = _service({"current": {"schema": 2}}, left, "current", policy=client, deps=(hub,))
    east, east_revs = _service({"current": {"schema": 2}}, right, "current", policy=client, deps=(hub,))
    task = _ops("healthy_upstream_fanout",
                "Restore scheduler and notifier and verify both with health probes. Keep directory available.",
                [upstream, west, east], upstream_revs + west_revs + east_revs, [left, right])
    path = [call("request_maintenance_window", svc_id=hub), call("rollback_config", svc_id=hub, rev_id="compatible")]
    for name in (left, right):
        path.extend([call("restart_service", svc_id=name), call("run_health_probe", svc_id=name)])
    return task, ("services", 1, "max_restarts"), path


def service_local():
    sid = "session-worker"
    svc, revisions = _service({"current": {"pool": 18, "auth_mode": "token"},
                               "legacy": {"pool": 5, "auth_mode": "password"},
                               "stable": {"pool": 8, "auth_mode": "token"}},
                              sid, "current", policy={"min_pool": 4, "max_pool": 12, "required_auth": "token"})
    task = _ops("single_service_local_policy", "Restore session-worker and verify it with a health probe.",
                [svc], revisions, [sid])
    return task, ("services", 0, "max_restarts"), [call("rollback_config", svc_id=sid, rev_id="stable"),
                                                   call("restart_service", svc_id=sid),
                                                   call("run_health_probe", svc_id=sid)]


def _db(family, prompt, accounts, tickets, invoices, ledger):
    return {"family": family, "domain": ACCOUNTING, "prompt": prompt,
            "initial_state": {"accounts": accounts, "tickets": tickets, "invoices": invoices, "ledger_entries": ledger},
            "assertions": {"closed_tickets": [ticket["ticket_id"] for ticket in tickets]}}


def _account(acct_id):
    return {"acct_id": acct_id, "owner": "Settlement", "balance_cents": 14000, "min_reserve_cents": 1000, "frozen": 0}


def _ticket(ticket_id, audit=0):
    return {"ticket_id": ticket_id, "title": ticket_id, "status": "OPEN", "audit_passed": audit, "max_adjustments": 1}


def _invoice(invoice, ticket, account, **amount):
    return {"inv_id": invoice, "ticket_id": ticket, "acct_id": account, "status": "OPEN", **amount}


def _entry(entry, ticket, invoice, account, amount, ref, duplicate=0):
    return {"entry_id": entry, "ticket_id": ticket, "inv_id": invoice, "acct_id": account,
            "amount_cents": amount, "ref_code": ref, "is_duplicate": duplicate, "voided": 0}


def _finalize(ticket, invoice, *, audit=True):
    return ([call("run_audit_check", ticket_id=ticket)] if audit else []) + [
        call("reconcile_invoice", inv_id=invoice), call("close_ticket", ticket_id=ticket)]


def accounting_fee():
    tid, iid, aid = "settlement-fee", "invoice-fee", "account-fee"
    amounts = {"subtotal_cents": 7863, "discount_cents": 381, "discount_pre_tax": 1,
               "tax_rate_bps": 725, "late_fee_bps": 21, "days_late": 3}
    ledger = [_entry("first-payment", tid, iid, aid, 4402, "receipt-110"),
              _entry("second-payment", tid, iid, aid, 3814, "receipt-111")]
    task = _db("discount_fee_rounding", "Reconcile and close settlement-fee using the invoice and ledger.",
               [_account(aid)], [_ticket(tid)], [_invoice(iid, tid, aid, **amounts)], ledger)
    delta = compute_expected_cents(**amounts) - sum(item["amount_cents"] for item in ledger)
    return task, ("tickets", 0, "max_adjustments"), [
        call("post_adjustment", ticket_id=tid, inv_id=iid, acct_id=aid, delta_cents=delta), *_finalize(tid, iid)]


def accounting_dispositions():
    first, second = "settlement-cleared", "settlement-duplicate"
    i1, i2, a1, a2 = "invoice-cleared", "invoice-duplicate", "account-cleared", "account-duplicate"
    amounts = {"tax_rate_bps": 0, "discount_cents": 0, "discount_pre_tax": 0}
    ledger = [_entry("payment-cleared", first, i1, a1, 2165, "receipt-207"),
              _entry("payment-original", second, i2, a2, 3400, "receipt-209"),
              _entry("payment-repeat", second, i2, a2, 3400, "receipt-209", duplicate=1)]
    task = _db("separate_close_and_void", "Complete and close settlement-cleared and settlement-duplicate.",
               [_account(a1), _account(a2)], [_ticket(first, 1), _ticket(second)],
               [_invoice(i1, first, a1, subtotal_cents=2165, **amounts),
                _invoice(i2, second, a2, subtotal_cents=3400, **amounts)], ledger)
    return task, ("tickets", 1, "max_adjustments"), [
        *_finalize(first, i1, audit=False), call("void_duplicate_entry", entry_id="payment-repeat"),
        *_finalize(second, i2)]


def prepare(max_steps):
    tasks, witnesses = {}, {}
    for build in (service_fanout, accounting_fee, service_local, accounting_dispositions):
        base, (table, index, field), path = build()
        for state in ("tight", "relaxed"):
            task = deepcopy(base)
            task["initial_state"][table][index][field] += int(state == "relaxed")
            tid = task["family"] + "-" + state
            task.update(task_id=tid, template_id=task["family"], state=state, difficulty="development", max_steps=max_steps)
            with closing(create_env(task["domain"])) as env:
                env.reset(task)
                if env.verify(task).passed:
                    raise ValueError(f"{tid} already complete at reset")
                for action in path:
                    if not env.execute_tool(action["name"], action["arguments"]).ok:
                        raise ValueError(f"{tid} witness failed at {action}")
                if not env.verify(task).passed:
                    raise ValueError(f"{tid} witness did not achieve the terminal goal")
            tasks[tid], witnesses[tid] = task, path
    return tasks, witnesses
