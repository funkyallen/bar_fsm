"""Prespecified accounting and service forks for the v3.3 development comparison."""

from copy import deepcopy

from .research_v31_timing_candidates import call
from .research_v32_candidates import _account, _db, _entry, _finalize, _invoice, _ops, _service, _ticket


def accounting_cases():
    cases = []
    for family, delta in (("split_payment_fee", 200), ("cross_ticket_credit", -180)):
        tid, inv, acct = f"{family}-ticket", f"{family}-invoice", f"{family}-account"
        primary = _ticket(tid)
        primary["max_adjustments"] = 2
        tickets, invoices = [primary], [_invoice(inv, tid, acct, subtotal_cents=2600, tax_rate_bps=0,
                                                    discount_cents=0, discount_pre_tax=0)]
        ledger = [_entry(f"{family}-payment", tid, inv, acct, 2600 - delta, "bank-initial")]
        other_finish = []
        if family == "cross_ticket_credit":
            paired, paired_inv = "close-companion", "invoice-companion"
            tickets.append(_ticket(paired, audit=1))
            invoices.append(_invoice(paired_inv, paired, acct, subtotal_cents=890, tax_rate_bps=0,
                                     discount_cents=0, discount_pre_tax=0))
            ledger.append(_entry("companion-receipt", paired, paired_inv, acct, 890, "bank-companion"))
            other_finish = _finalize(paired, paired_inv, audit=False)
        prior = _db(family, f"Reconcile and close these tickets: {', '.join(t['ticket_id'] for t in tickets)}.",
                    [_account(acct)], tickets, invoices, ledger)
        candidate = call("post_adjustment", ticket_id=tid, inv_id=inv, acct_id=acct, delta_cents=delta)
        for changed in (False, True):
            case_id = family + ("-changed" if changed else "-stable")
            present = deepcopy(prior)
            present.update(task_id=case_id, difficulty="development", max_steps=12)
            settlement = (70 if delta > 0 else delta) if changed else 0
            revoke = changed and delta < 0
            if settlement:
                present["initial_state"]["ledger_entries"].append(
                    _entry(f"{family}-external", tid, inv, acct, settlement, "bank-confirmed"))
                present["initial_state"]["accounts"][0]["balance_cents"] += settlement
            approval = {"ticket_id": tid, "inv_id": inv, "acct_id": acct, "max_abs_delta_cents": 500,
                        "grant_version": 2 if revoke else 1, "allowed": not revoke,
                        "valid_through_decision": 12}
            current_delta = delta - settlement
            completion = [call("post_adjustment", **{**candidate["arguments"], "delta_cents": current_delta})] if current_delta else []
            cases.append({"id": case_id, "family": family, "conditions": "changed" if changed else "stable",
                          "task": present, "prior": prior, "candidate": candidate, "approval": approval,
                          "witness": [*completion, *_finalize(tid, inv), *other_finish]})
    return cases


def service_cases():
    cases = []
    for family in ("tls_provider", "healthy_auth_dependency"):
        tls = family == "tls_provider"
        up, target = ("cert-store", "tls-front") if tls else ("token-agent", "queue-consumer")
        req = {"min_tls": "1.3"} if tls else {"required_auth": "token"}
        revs = {"old": {"tls": "1.1"}, "supported": {"tls": "1.3"}} if tls else {"supported": {"auth_mode": "token"}}
        for satisfied in (False, True):
            state = "satisfied" if satisfied else "blocked"
            case_id = family + "-" + state
            p, p_configs = _service(revs, up, "supported" if satisfied or not tls else "old",
                                    healthy=tls or satisfied)
            p["maint_approved"] = 1
            t, t_configs = _service({"current": {}}, target, "current",
                                    policy={"upstream_reqs": {up: req}}, deps=(up,))
            goals = [target] if tls else [up, target]
            task = _ops(family, f"Restore and probe {', '.join(goals)} using safe dependency order.",
                        [p, t], p_configs + t_configs, goals)
            task.update(task_id=case_id, difficulty="development", max_steps=12)
            if satisfied:
                before = []
            elif tls:
                before = [call("rollback_config", svc_id=up, rev_id="supported")]
            else:
                before = [call("restart_service", svc_id=up)]
            after = [call("restart_service", svc_id=target), *(call("run_health_probe", svc_id=sid) for sid in goals)]
            cases.append({"id": case_id, "family": family, "conditions": state, "task": task,
                          "candidate": call("restart_service", svc_id=target), "upstream": up, "witness": before + after})
    return cases
