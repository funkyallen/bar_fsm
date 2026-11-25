"""Check reachability labels with native witnesses and exhausted offline searches."""

import argparse
import ast
from collections import Counter, deque
from contextlib import closing
from graphlib import TopologicalSorter
from pathlib import Path
import json
import sys


ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime"
sys.path[:0] = [str(RUNTIME), str(RUNTIME / "src")]

from scripts.research_v40_cases import _account_case, env_for, offline_cases
from scripts.research_strict_quota_replay import call as replay_archived_call


SERVICE_TOOLS = {
    "get_topology": "read",
    "inspect_service": "read",
    "read_service_logs": "read",
    "list_config_revisions": "read",
    "request_maintenance_window": "free approval, already present in the audited fixtures",
    "rollback_config": "free staging, folded into the next restart or final probe",
    "restart_service": "paid transition, including debit-only rejection in the relaxation",
    "run_health_probe": "free final probe, executed in each native witness",
}
ACCOUNTING_TOOLS = {
    "list_open_tickets": "read",
    "get_invoice_bundle": "read",
    "query_account": "read",
    "run_audit_check": "free close-out, executed once ledger obligations are satisfied",
    "void_duplicate_entry": "free explicit transition that consumes an existing duplicate",
    "post_adjustment": "paid integer-cent transition on the authorized invoice and account",
    "reconcile_invoice": "free close-out following a passed audit",
    "close_ticket": "free final close-out after reconciliation",
}


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def action(name, **arguments):
    return {"name": name, "arguments": arguments}


def search(start, successors, goal, max_states=None):
    """State-deduplicated breadth-first search, with explicit cutoff semantics."""
    assert max_states is None or max_states >= 0
    pending = deque([start])
    parents = {start: None}
    examined = 0
    while pending:
        if max_states is not None and examined >= max_states:
            return {"verdict": "unknown", "termination": "computation_limit",
                    "expanded_states": examined, "pending_states": len(pending)}
        state = pending.popleft()
        examined += 1
        if goal(state):
            path = []
            while parents[state] is not None:
                state, operation = parents[state]
                path.append(operation)
            return {"verdict": "reachable", "termination": "goal",
                    "expanded_states": examined, "pending_states": len(pending),
                    "path": list(reversed(path))}
        for next_state, operation in successors(state):
            if next_state not in parents:
                parents[next_state] = state, operation
                pending.append(next_state)
    return {"verdict": "unreachable", "termination": "frontier_exhausted",
            "expanded_states": examined, "pending_states": 0}


def execute(env, operation, require_success=True):
    result = env.execute_tool(operation["name"], operation["arguments"])
    if require_success and not result.ok:
        raise ValueError(f"Backend rejected {operation}: {result.error_code}")
    return result


def execute_witness(env, task, path):
    observations = []
    for operation in path:
        result = execute(env, operation)
        response = json.loads(result.to_tool_message())
        if operation["name"] == "post_adjustment":
            # Generated IDs do not carry reachability information and are nondeterministic.
            response["data"].pop("entry_id")
        if operation["name"] == "run_audit_check" and not response["data"]["audit_passed"]:
            raise ValueError("A close-out witness failed its audit.")
        observations.append({"operation": operation, "backend_reply": response})
    scored = env.verify(task)
    if not scored.passed:
        raise ValueError(f"Witness failed native scoring: {scored.violations}")
    return observations


def require_tool_coverage(env, expected):
    names = {tool["function"]["name"] for tool in env.tool_schemas()}
    if names != set(expected):
        raise ValueError(f"An environment tool lacks coverage: {names ^ set(expected)}")


def mutation_inventory(source, method):
    tree = ast.parse((RUNTIME / source).read_text(encoding="utf-8"))
    functions = [node for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef) and node.name == method]
    assert len(functions) == 1
    return sorted({
        " ".join(node.value.split()) for node in ast.walk(functions[0])
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and node.value.lstrip().startswith(("UPDATE ", "INSERT ", "DELETE ", "REPLACE ", "ALTER "))
    })


def contract_checks():
    service_backend = "src/aroute/envs/ops_troubleshoot_env.py"
    activation_wrapper = "scripts/research_v34_intervention.py"
    activation_variant = "scripts/research_v40_control.py"
    accounting_backend = "src/aroute/envs/db_transaction_env.py"
    writers = {name: mutation_inventory(name, "execute_tool")
               for name in (service_backend, activation_wrapper, activation_variant, accounting_backend)}
    assert writers[service_backend] == sorted({
        "INSERT INTO change_requests VALUES (?, ?, 'APPROVED')",
        "UPDATE services SET maint_approved = 1 WHERE svc_id = ?",
        "UPDATE services SET restart_count = restart_count + 1 WHERE svc_id = ?",
        "UPDATE services SET active_rev = ?, health_probed = 0 WHERE svc_id = ?",
        "UPDATE services SET status = 'HEALTHY', health_probed = 0 WHERE svc_id = ?",
        "UPDATE services SET health_probed = 1 WHERE svc_id = ?",
    })
    assert writers[activation_wrapper] == [
        "UPDATE runtime_activation SET running_rev=(SELECT active_rev FROM services WHERE svc_id=?) WHERE svc_id=?",
    ]
    assert writers[activation_variant] == []
    assert writers[accounting_backend] == sorted({
        "UPDATE reconciliation_tickets SET audit_passed = ? WHERE ticket_id = ?",
        "UPDATE ledger_entries SET voided = 1 WHERE entry_id = ?",
        "UPDATE accounts SET balance_cents = balance_cents - ? WHERE acct_id = ?",
        "UPDATE reconciliation_tickets SET audit_passed = 0 WHERE ticket_id = ?",
        "INSERT INTO ledger_entries VALUES (?, ?, ?, ?, ?, ?, '2026-11-20', 'adjustment', 0, 0)",
        "UPDATE accounts SET balance_cents = ? WHERE acct_id = ?",
        "UPDATE reconciliation_tickets SET audit_passed = 0, adj_count = adj_count + 1 WHERE ticket_id = ?",
        "UPDATE invoices SET status = 'RECONCILED' WHERE inv_id = ?",
        "UPDATE reconciliation_tickets SET status = 'CLOSED' WHERE ticket_id = ?",
    })
    return {"service_tools": SERVICE_TOOLS, "accounting_tools": ACCOUNTING_TOOLS,
            "dispatcher_mutations": writers}


def service_contract(task):
    state = task["initial_state"]
    expected = task["goal"]["running_revisions"]
    services = {row["svc_id"]: row for row in state["services"]}
    parents = {sid: services[sid]["deps"] for sid in services}
    order = list(TopologicalSorter(parents).static_order())
    assert set(expected) == set(services) == set(task["goal"]["authorized_services"])
    assert set(task["assertions"]["healthy_and_probed"]) == set(services)
    revisions = {sid: {} for sid in services}
    for revision in state["config_revisions"]:
        revisions[revision["svc_id"]][revision["rev_id"]] = revision["params"]
    worst_memory = sum(max(params["mem_mb"] for params in choices.values()) for choices in revisions.values())
    assert worst_memory <= state["cluster_mem_cap_mb"]
    for sid, service in services.items():
        assert service["maint_approved"] == 1
        assert expected[sid] == "approved"
        assert set(revisions[sid]) == {"legacy", "approved"}
        assert revisions[sid] == {"legacy": {"schema": 1, "mem_mb": 30},
                                  "approved": {"schema": 2, "mem_mb": 30}}
        policy = service["policy"]
        assert policy["min_schema"] == 1 and set(policy) == {"min_schema", "upstream_reqs"}
        assert set(policy["upstream_reqs"]) == set(parents[sid])
        assert all(rule in ({"min_schema": 1}, {"min_schema": 2}) for rule in policy["upstream_reqs"].values())
    return order, revisions


def service_projection(env, order):
    snapshot = env.snapshot_state()["services"]
    return tuple((snapshot[sid]["running_rev"], snapshot[sid]["status"] == "HEALTHY",
                  snapshot[sid]["max_restarts"] - snapshot[sid]["restarts"]) for sid in order)


def service_optimistic_search(env, task, contract, max_states=None):
    """Relax startup and closure tests, but preserve actual quota and running-revision writes."""
    order, revisions = contract
    wanted = task["goal"]["running_revisions"]

    def successors(state):
        for index, (current, healthy, remaining) in enumerate(state):
            if remaining <= 0:
                continue
            updates = [(revision, True) for revision in revisions[order[index]]]
            updates.append((current, healthy))  # rejected restart may still debit
            for revision, ok in updates:
                yield (state[:index] + ((revision, ok, remaining - 1),) + state[index + 1:], None)

    def goal(state):
        return all(current == wanted[sid] and healthy
                   for sid, (current, healthy, _) in zip(order, state, strict=True))

    return search(service_projection(env, order), successors, goal, max_states)


def service_verdict(env, task, contract, retain_witness=False):
    order, _ = contract
    optimistic = service_optimistic_search(env, task, contract)
    if optimistic["verdict"] == "unknown":
        return {"reachable": None, "search": optimistic}
    if optimistic["verdict"] == "unreachable":
        return {"reachable": False, "exhaustion": optimistic}

    snapshot = env.snapshot_state()["services"]
    wanted = task["goal"]["running_revisions"]
    path = []
    for sid in order:
        current = snapshot[sid]
        if current["active_rev"] != wanted[sid]:
            path.append(action("rollback_config", svc_id=sid, rev_id=wanted[sid]))
        if current["running_rev"] != wanted[sid] or current["status"] != "HEALTHY":
            path.append(action("restart_service", svc_id=sid))
    path.extend(action("run_health_probe", svc_id=sid) for sid in order)
    with closing(env_for(task)) as fork:
        env.conn.backup(fork.conn)
        observations = execute_witness(fork, task, path)
    result = {"reachable": True, "native_scorer_passed": True, "tool_actions": len(path)}
    if retain_witness:
        result["witness"] = observations
    return result


def account_context(env, task):
    goal = task["goal"]
    assert goal["amount_policy"] == "toward_balance_partial_ok" and goal["post_authorized"]
    assert set(goal["authorized_tickets"]) == set(task["assertions"]["closed_tickets"])
    db = env.conn
    tickets = tuple(db.execute("SELECT * FROM reconciliation_tickets ORDER BY ticket_id"))
    ids = tuple(row["ticket_id"] for row in tickets)
    assert set(ids) == set(goal["authorized_tickets"])
    invoice_rows = tuple(db.execute("SELECT * FROM invoices ORDER BY inv_id"))
    invoices = tuple(row["inv_id"] for row in invoice_rows)
    owners = tuple(ids.index(row["ticket_id"]) for row in invoice_rows)
    account, = db.execute("SELECT * FROM accounts")
    assert account["acct_id"] == goal["authorized_account"] and account["frozen"] == 0
    entries = tuple(db.execute("SELECT * FROM ledger_entries WHERE voided = 0 ORDER BY entry_id"))
    assert all(row["ticket_id"] == ids[owners[invoices.index(row["inv_id"])]]
               and row["acct_id"] == account["acct_id"] for row in entries)
    duplicates = tuple((row["entry_id"], invoices.index(row["inv_id"]), row["amount_cents"])
                       for row in entries if row["is_duplicate"])
    residuals = tuple(row["expected_cents"] - sum(entry["amount_cents"] for entry in entries
                      if entry["inv_id"] == row["inv_id"]) for row in invoice_rows)
    quotas = tuple(row["max_adjustments"] - row["adj_count"] for row in tickets)
    headroom = account["balance_cents"] - account["min_reserve_cents"]
    return (ids, invoices, owners, account["acct_id"], goal["max_per_post_cents"], duplicates,
            (residuals, quotas, headroom, (1 << len(duplicates)) - 1))


def authorized_post(context, operation):
    ids, invoices, owners, account, cap, duplicates, (needs, quotas, headroom, _) = context
    args = operation["arguments"]
    index = invoices.index(args["inv_id"])
    amount = args["delta_cents"]
    return (args["ticket_id"] == ids[owners[index]] and args["acct_id"] == account
            and not any(i == index for _, i, _ in duplicates)
            and quotas[owners[index]] > 0 and abs(amount) <= min(abs(needs[index]), cap)
            and amount * needs[index] > 0 and headroom + amount >= 0)


def account_search(context, allow_post_before_void, max_states=None):
    ids, invoices, owners, account, cap, duplicates, start = context

    def successors(state):
        needs, quotas, headroom, live = state
        waiting = set()
        for bit, (entry, i, amount) in enumerate(duplicates):
            if live & (1 << bit):
                waiting.add(i)
                revised = needs[:i] + (needs[i] + amount,) + needs[i + 1:]
                yield ((revised, quotas, headroom - amount, live ^ (1 << bit)),
                       action("void_duplicate_entry", entry_id=entry))
        for i, need in enumerate(needs):
            pool = owners[i]
            if (not allow_post_before_void and i in waiting) or quotas[pool] <= 0:
                continue
            remaining = quotas[:pool] + (quotas[pool] - 1,) + quotas[pool + 1:]
            for magnitude in range(1, min(abs(need), cap) + 1):
                amount = magnitude if need > 0 else -magnitude
                if headroom + amount >= 0:
                    revised = needs[:i] + (need - amount,) + needs[i + 1:]
                    yield ((revised, remaining, headroom + amount, live),
                           action("post_adjustment", ticket_id=ids[pool], inv_id=invoices[i],
                                  acct_id=account, delta_cents=amount))

    return search(start, successors, lambda state: not any(state[0]) and not state[3], max_states)


def account_verdict(env, task, retain_witness=False):
    context = account_context(env, task)
    result = account_search(context, allow_post_before_void=False)
    if result["verdict"] == "unknown":
        return {"reachable": None, "search": result}
    if result["verdict"] == "unreachable":
        optimistic = account_search(context, allow_post_before_void=True)
        if optimistic["verdict"] != "unreachable":
            # A positive relaxation is not a native witness under the actual ordering constraint.
            return {"reachable": None, "strict_exhaustion": result, "optimistic": optimistic}
        return {"reachable": False, "exhaustion": optimistic}

    with closing(env_for(task)) as fork:
        env.conn.backup(fork.conn)
        operations = list(result["path"])
        ids, invoices, owners, *_ = context
        terminal = fork.snapshot_state()
        for pool, ticket in enumerate(ids):
            operations.append(action("run_audit_check", ticket_id=ticket))
            operations.extend(action("reconcile_invoice", inv_id=inv)
                              for inv, owner in zip(invoices, owners, strict=True)
                              if owner == pool and terminal["invoices"][inv] != "RECONCILED")
            if terminal["tickets"][ticket]["status"] != "CLOSED":
                operations.append(action("close_ticket", ticket_id=ticket))
        observations = execute_witness(fork, task, operations)
    payload = {"reachable": True, "native_scorer_passed": True, "expanded_states": result["expanded_states"],
               "tool_actions": len(operations)}
    if retain_witness:
        payload["witness"] = observations
    return payload


def check_pair(case, candidate, retain_witness=False):
    task = case["task"]
    is_service = task["domain"] == "ops_troubleshoot"
    with closing(env_for(task)) as env:
        if is_service:
            require_tool_coverage(env, SERVICE_TOOLS)
            contract = service_contract(task)
            before = service_verdict(env, task, contract, retain_witness)
            reply = execute(env, candidate, require_success=False)
            after = service_verdict(env, task, contract, retain_witness) if reply.ok else None
        else:
            require_tool_coverage(env, ACCOUNTING_TOOLS)
            before = account_verdict(env, task, retain_witness)
            if authorized_post(account_context(env, task), candidate):
                execute(env, candidate)
                after = account_verdict(env, task, retain_witness)
            else:
                after = None
    return before, after


def verify_labels(task_id, before, after, expected_before, expected_after):
    current_after = after["reachable"] if after else None
    if before["reachable"] != expected_before or current_after != expected_after:
        raise ValueError(f"Independent reachability differs at {task_id}: "
                         f"{(before['reachable'], current_after)} vs {(expected_before, expected_after)}")
    if before["reachable"] is None or (after and after["reachable"] is None):
        raise ValueError(f"Unknown result where a definitive label is expected: {task_id}")


def summarize_pair(buckets, before, after):
    for name, result in (("predecessor", before), ("successor", after)):
        state = "ineligible" if result is None else {True: "native_witness", False: "exhausted"}[result["reachable"]]
        buckets[f"{name}_{state}"] += 1
        if result and "exhaustion" in result:
            assert result["exhaustion"]["pending_states"] == 0
            buckets["negative_search_states_expanded"] += result["exhaustion"]["expanded_states"]


def fixture_checks():
    archived = {row["task"]: row for row in read("voidability_720.json")["rows"]}
    qualification = {"ops_troubleshoot": Counter(), "db_transaction": Counter()}
    for channel, case in offline_cases():
        task = case["task"]
        stored = archived.pop(task["task_id"])
        assert stored["evidence"] == channel and stored["candidate"] == case["candidate"]
        before, after = check_pair(case, stored["candidate"])
        verify_labels(task["task_id"], before, after, stored["truth_before"], stored["truth_after"])
        summary = qualification[task["domain"]]
        summary["records"] += 1
        summarize_pair(summary, before, after)
    assert not archived

    fixtures = {case["task"]["task_id"]: case for case in read("live/shared_allowance/cases.json")}
    fixed = Counter()
    for row in read("live/shared_allowance/fixed_candidates.json"):
        before, after = check_pair(fixtures[row["task"]], row["candidate"])
        verify_labels(row["task"], before, after, True, row["successor_reachable"])
        fixed["records"] += 1
        summarize_pair(fixed, before, after)
    return {"qualification_720": qualification, "fixed_accounting_52": fixed}


def natural_accounting_checks():
    fixtures = {case["task"]["task_id"]: case for case in read("live/shared_allowance/cases.json")}
    rows = read("live/shared_allowance/live/result.json")["rows"]
    summary = Counter()
    for row in rows:
        task = fixtures[row["task"]]["task"]
        ids = {}
        summary["episodes"] += 1
        with closing(env_for(task)) as env:
            for event in row["trace"]:
                oracle = event.get("oracle")
                candidate = event.get("candidate")
                if oracle is not None:
                    assert candidate["name"] == "post_adjustment" and event["guard"]["local"] == "pass"
                    assert authorized_post(account_context(env, task), candidate)
                    predecessor = account_verdict(env, task)
                    with closing(env_for(task)) as fork:
                        env.conn.backup(fork.conn)
                        execute(fork, candidate)
                        successor = account_verdict(fork, task)
                    verify_labels(f"{row['run_id']}:{event['step']}", predecessor, successor,
                                  oracle["before"], oracle["after_if_dispatched"])
                    summarize_pair(summary, predecessor, successor)
                    summary["archived_postings"] += 1
                if event["disposition"] == "dispatch":
                    saved = event["reply"]
                    if "entry_voidability" in saved.get("data", {}):
                        # This wrapper augments a read without changing the native state.
                        saved = {**saved, "data": {field: value for field, value in saved["data"].items()
                                                   if field != "entry_voidability"}}
                    replay_archived_call(env, {**event, "reply": saved}, ids)
            if env.verify(task).passed != row["terminal"]:
                raise ValueError(f"Archived accounting terminal state differs: {row['run_id']}")
    assert summary["episodes"] == 64 and summary["archived_postings"] == 177
    return summary


def archived_service_losses():
    fixtures = {case["task"]["task_id"]: case for case in read("formal_fixtures/cases.json")}
    historical = read("live/formal_completion/result.json")["rows"]
    recorded_forks = [*read("strict_replay.json")["withheld_dispatch_forks"],
                      *read("formal_completion_audit.json")["second_stage"]["replayed"]["withheld_dispatch_forks"]]
    forks = {(item["slot"], item["step_zero_based"]): item for item in recorded_forks}
    assert len(forks) == len(recorded_forks) == 11
    checked = []
    for row in historical:
        case = fixtures[row["task"]]
        task = case["task"]
        if task["domain"] != "ops_troubleshoot":
            continue
        targets = [i for i, event in enumerate(row["trace"]) if event.get("oracle", {}).get("new_loss")]
        for index in targets:
            source = row["trace"][index]
            assert source["event"] in ("dispatch", "future_block") and source["assessment"]["local"] == "pass"
            candidate = source["candidate"]
            assert candidate["name"] == "restart_service"
            sid = candidate["arguments"]["svc_id"]
            status = "observed_dispatch" if source["event"] == "dispatch" else "isolated_dispatch_of_hold"
            goal = task["goal"]["running_revisions"][sid]
            with closing(env_for(task)) as env:
                ids = {}
                public_receipts = []
                for event in row["trace"][:index]:
                    if event["event"] in ("dispatch", "auto_read"):
                        replay_archived_call(env, event, ids)
                    if event["event"] == "auto_read" and event["step"] == source["step"]:
                        public_receipts.append({"tool": event["candidate"], "reply": event["reply"]})
                contract = service_contract(task)
                pre = env.snapshot_state()["services"][sid]
                before = service_verdict(env, task, contract, retain_witness=True)
                if status == "observed_dispatch":
                    reply, _ = replay_archived_call(env, source, ids)
                else:
                    reply = execute(env, candidate)
                    archived = forks.pop((row["slot"], source["step"]))
                    assert archived["forked_response"] == reply.data and archived["new_loss_under_oracle"]
                post = env.snapshot_state()["services"][sid]
                after = service_verdict(env, task, contract)
                q_before = pre["max_restarts"] - pre["restarts"]
                q_after = post["max_restarts"] - post["restarts"]
                assert reply.ok and q_before == 1 and q_after == 0 and post["running_rev"] != goal
                assert before["reachable"] and after["reachable"] is False
                checked.append({
                    "slot": row["slot"], "arm": row["arm"], "provider": row["provider"], "task": row["task"],
                    "step_zero_based": source["step"], "record_type": status, "candidate": candidate,
                    "public_receipts": public_receipts, "recorded_assessment": source["assessment"],
                    "predecessor": {"service_state": pre, "allowance": q_before, **before},
                    "successor": {"service_state": post, "allowance": q_after, **after},
                    "required_running_revision": goal, "native_restart_response": reply.data,
                    "original_terminal_success": row["terminal"],
                })
    counts = Counter(row["record_type"] for row in checked)
    assert counts == {"observed_dispatch": 4, "isolated_dispatch_of_hold": 11} and not forks
    return {"candidate_counts": counts, "cases": checked}


def accounting_boundary_cases():
    hidden = _account_case("void-eligibility", (2, -2), (1,), 8, (0, 2), (0, 0), 2, 0, 2)
    for entry in hidden["task"]["initial_state"]["ledger_entries"]:
        if entry["is_duplicate"]:
            entry["ref_code"] = "migrated-ref"
    control = _account_case("void-proof", (2, 2), (2,), 8, (0, 0), (0, 0), 2, 0, 1)
    installment = _account_case("installment-witness", (10,), (3,), 8, (0,), (0,), 4, 0, 1)
    alternative = action("post_adjustment", **{**installment["candidate"]["arguments"], "delta_cents": 2})
    records = {}
    for label, case, candidate, expected in (
        ("hidden_free_void", hidden, hidden["candidate"], True),
        ("known_no_void", control, control["candidate"], False),
        ("installment_plus_one", installment, installment["candidate"], False),
        ("installment_plus_two", installment, alternative, True),
    ):
        before, after = check_pair(case, candidate, retain_witness=True)
        verify_labels(label, before, after, True, expected)
        records[label] = {"candidate": candidate, "predecessor": before, "successor": after}
    return records


def main(out):
    if not __debug__:
        raise RuntimeError("Run without Python optimization so evidence checks remain active.")
    cutoff = search((0,), lambda s: [((s[0] + 1,), None)], lambda s: s == (2,), max_states=1)
    assert cutoff["verdict"] == "unknown" and cutoff["pending_states"] == 1
    report = {
        "scope": "offline archived sandbox verification, no model calls",
        "contract": contract_checks(),
        "computational_cutoff_control": cutoff,
        "fixture_checks": fixture_checks(),
        "formal_service_losses": archived_service_losses(),
        "natural_accounting": natural_accounting_checks(),
        "accounting_boundaries": accounting_boundary_cases(),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "oracle_v50.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Verified 720 qualifications, 52 fixed and 177 natural accounting candidates, "
          "four boundary controls, and all 15 formal service losses.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args().out)
