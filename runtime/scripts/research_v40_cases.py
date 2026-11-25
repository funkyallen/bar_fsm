"""Prespecified fixture families and deterministic public tool receipts."""

from contextlib import closing
from itertools import cycle
from random import Random

from aroute.envs import create_env

from .research_v32_candidates import _account, _db, _entry, _invoice, _ops, _service, _ticket
from .research_v34_intervention import action, key
from .research_v40_control import ActivationEnv, reads


def _account_case(tag, residual, quotas, headroom, duplicate, owners, cap, candidate_i, value):
    aid = f"{tag}-funds"
    ticket_ids = [f"{tag}-t{i+1}" for i in range(len(quotas))]
    invoice_ids = [f"{tag}-i{i+1}" for i in range(len(residual))]
    accounts = [{**_account(aid), "balance_cents": 500 + headroom, "min_reserve_cents": 500}]
    tickets = [{**_ticket(tid), "max_adjustments": q} for tid, q in zip(ticket_ids, quotas)]
    invs, entries = [], []
    for i, iid in enumerate(invoice_ids):
        t = ticket_ids[owners[i]]
        invs.append(_invoice(iid, t, aid, subtotal_cents=900, tax_rate_bps=0, discount_cents=0, discount_pre_tax=0))
        d = duplicate[i]
        entries.append(_entry(f"{iid}-main", t, iid, aid, 900-residual[i]-2*d, f"{iid}-bank"))
        if d:
            entries += [_entry(f"{iid}-base", t, iid, aid, d, f"{iid}-batch"),
                        _entry(f"{iid}-repeat", t, iid, aid, d, f"{iid}-batch", duplicate=1)]
    prompt = (f"Reconcile all invoices and close tickets {', '.join(ticket_ids)}. You may post partial or full adjustments "
              f"toward the reconciled invoice amount on account {aid}, at most {cap} cents per posting in absolute value. "
              "Respect each ticket's adjustment quota and the account's minimum reserve on every posting. "
              "Remove known duplicate postings before adjusting their invoice. Use public tools to resolve exact amounts and finish both ledger and close-out work.")
    task = _db(tag, prompt, accounts, tickets, invs, entries)
    task.update(task_id=tag, difficulty="development", max_steps=28)
    task["goal"] = {"ticket_id": ticket_ids[0], "authorized_tickets": ticket_ids, "authorized_account": aid, "post_authorized": True,
                    "max_per_post_cents": cap, "amount_policy": "toward_balance_partial_ok", "final_condition": "close_all_ticket_invoices"}
    candidate = action("post_adjustment", ticket_id=ticket_ids[owners[candidate_i]], inv_id=invoice_ids[candidate_i],
                       acct_id=aid, delta_cents=value)
    return {"task": task, "candidate": candidate, "truth": {"residual": residual, "quotas": quotas, "headroom": headroom,
            "duplicates": duplicate, "ticket_for": owners, "cap": cap, "candidate_index": candidate_i},
            "names": {"invoices": invoice_ids, "tickets": ticket_ids}}


def _service_case(tag, running, healthy, quotas, deps, required, staged, candidate_i):
    ids = [f"{tag}-s{i+1}" for i in range(len(running))]
    labels = ("legacy", "approved")
    rows, cfgs, prefix = [], [], []
    for i, sid in enumerate(ids):
        policy = {"min_schema": 1, "upstream_reqs": {ids[d]: {"min_schema": minimum + 1} for d, minimum in deps[i]}}
        row, cfg = _service({"legacy": {"schema": 1}, "approved": {"schema": 2}}, sid, labels[running[i]], deps=[ids[d] for d, _ in deps[i]],
                            policy=policy, healthy=healthy[i])
        rows.append({**row, "maint_approved": 1, "max_restarts": quotas[i]})
        cfgs.extend(cfg)
        if staged[i] != running[i]:
            prefix.append(action("rollback_config", svc_id=sid, rev_id=labels[staged[i]]))
    wanted = {sid: labels[revision] for sid, revision in zip(ids, required)}
    terms = ", ".join(f"{sid} on {rev}" for sid, rev in wanted.items())
    prompt = (f"Restore and health-probe {terms}. You are authorized to stage and activate revisions on these services. "
              "Changing a staged revision does not change what is running. Use current public dependency and quota information.")
    task = _ops(tag, prompt, rows, cfgs, ids)
    task.update(task_id=tag, difficulty="development", max_steps=28, prefix=prefix)
    task["goal"] = {"authorized_services": ids, "running_revisions": wanted}
    return {"task": task, "candidate": action("restart_service", svc_id=ids[candidate_i]),
            "truth": {"running": running, "healthy": healthy, "quotas": quotas, "deps": deps, "required": required,
                      "staged": staged, "candidate_index": candidate_i}, "names": {"services": ids}}


def env_for(task):
    env = ActivationEnv() if task["domain"] == "ops_troubleshoot" else create_env(task["domain"])
    env.reset(task)
    for a in task.get("prefix", []):
        if not env.execute_tool(a["name"], a["arguments"]).ok:
            env.close()
            raise ValueError("Staged initial state could not be constructed.")
    return env


def evidence(env, candidate, goal, omit=False, ambiguous=False):
    receipts = {}
    for _ in range(2):
        for a in reads(candidate, goal, receipts):
            if key(a) not in receipts:
                result = env.execute_tool(a["name"], a["arguments"])
                if result.ok and not result.is_write and not result.state_changed:
                    receipts[key(a)] = result.data
    if omit and receipts:
        receipts.pop(next(iter(receipts)))
    if ambiguous and candidate["name"] == "restart_service":
        receipts["inspect_service", candidate["arguments"]["svc_id"]]["service"].pop("staged_params")
    return receipts


def _graph(n, topology, strict=True):
    dep = [() for _ in range(n)]
    for i in range(1, n):
        parents = ([i-1] if topology == "chain" else list(range(i)) if topology == "fanin" and i == n-1 else
                   [0] if topology == "branch" else [])
        dep[i] = tuple((p, int(strict)) for p in parents)
    return tuple(dep)


def offline_cases():
    rng = Random(20261124)
    channels = ("complete", "complete", "complete", "missing", "stale", "ambiguous")
    for i, channel in zip(range(360), cycle(channels)):
        n = i % 4 + 1
        split_tickets = n > 1 and i % 3 == 0
        owners = (0,)*(n-1)+(1,) if split_tickets else (0,)*n
        residual = tuple(rng.choice((-4, -3, -2, -1, 1, 2, 3, 4)) if rng.random() < 0.7 else 0 for _ in range(n))
        cap = 1 + rng.randrange(4)
        dup = tuple(2 if channel == "ambiguous" and j == n-1 else 0 for j in range(n))
        quotas = (i // 4 % 5,)*(1+split_tickets)
        idx = rng.randrange(n)
        need = residual[idx]
        magnitude = min(abs(need), cap) if i % 3 == 1 else rng.randint(1, cap)
        proposed = magnitude * (-1 if (need < 0) != (i % 7 == 0) else 1)
        yield channel, _account_case(f"eval-a{i:03}", residual, quotas, rng.randrange(9), dup, owners, cap, idx, proposed)
    for i, channel in zip(range(360), cycle(channels)):
        n = i % 4 + 1
        graph = _graph(n, ("none", "chain", "fanin", "branch")[(i//4) % 4], strict=i%3 != 0)
        required = (1,)*n
        running = tuple(rng.randrange(2) for _ in range(n))
        healthy = tuple(rng.random() < 0.62 for _ in range(n))
        quotas = (i // 4 % 5,)*n
        staged = tuple(rng.randrange(2) for _ in range(n))
        yield channel, _service_case(f"eval-s{i:03}", running, healthy, quotas, graph, required, staged, rng.randrange(n))


FAMILIES = ("F01-single", "F02-shared", "F03-liquidity", "F04-post-cap", "F05-chain", "F06-fanin", "F07-branch", "F08-wide")


def paired_case(family_index, j, phase):
    tag = f"{phase}-{FAMILIES[family_index]}-{j+1}"
    slack = int(j == 2)
    if family_index == 0:
        case = _account_case(tag, (-(4+j),), (1+slack,), 6+j, (0,), (0,), 7, 0, -2)
    elif family_index == 1:
        case = _account_case(tag, (2+j, -(3+j), 1), (3+slack,), 6+j, (0,0,0), (0,0,0), 5, 0, 1)
    elif family_index == 2:
        case = _account_case(tag, (-(5+j), 5+j), (1+slack,1), 2+j, (0,0), (0,1), 8, 0, -2)
    elif family_index == 3:
        case = _account_case(tag, (5+j, 1), (3+slack,), 4, (0,0), (0,0), 3+j, 0, 1)
    elif family_index < 7:
        n = 3 if family_index in (4,5) else 4
        topology = ("chain","fanin","branch")[family_index-4]
        graph = _graph(n, topology, strict=bool(j))
        case = _service_case(tag, (0,)*n, tuple(i == 0 and bool(j) for i in range(n)), (1+slack,)*n,
                             graph, (1,)*n, (0,)*n, 0)
    elif j % 2 == 0:
        case = _account_case(tag, (0, 2), (4,), 8, (0,0), (0,0), 5, 1, 2)
    else:
        case = _service_case(tag, (1,0), (True,False), (3,3), _graph(2,"chain"), (1,1), (1,1), 1)
    case.update(family=FAMILIES[family_index], quota_group="wide" if slack or family_index==7 else "tight")
    return case


def paired_cases(phase):
    if phase == "dev":
        # Disjoint starts with different caps, amounts, and service shapes from formal cells.
        return [_account_case("dev-settlement", (-7, 8), (1,2), 2, (0,0), (0,1), 9, 0, -2),
                _account_case("dev-installments", (9, -4), (4,), 6, (0,0), (0,0), 4, 0, 1),
                _service_case("dev-activation", (0,0), (True,False), (1,2), ((),((0,1),)), (1,1), (0,0), 0),
                _service_case("dev-handoff", (1,0,1), (True,False,True), (2,1,2), ((),((0,1),(2,1)),()),
                              (1,1,1), (1,0,1), 1)]
    return [paired_case(i,j,"formal") for i in range(8) for j in range(3)]
