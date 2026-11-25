"""Build and verify the research pilot's matched synthetic state pairs offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aroute.core.config import PROJECT_ROOT
from aroute.envs import BaseSandboxEnv, create_env
from aroute.envs.db_transaction_env import compute_expected_cents
from aroute.orchestrator.fsm_guard import FSMGuard
from aroute.orchestrator.memory_ledger import StateLedger


OUTPUT_DIR = PROJECT_ROOT / "data" / "research_pilot"
SEED = 20261123


def call(name: str, **arguments: Any) -> dict[str, Any]:
    return {"name": name, "arguments": arguments}


@dataclass
class Fixture:
    """A paired scenario and its legal and wrong-first offline witness paths."""

    initial_relaxed: dict[str, Any]
    initial_tight: dict[str, Any]
    changed_paths: tuple[str, ...]
    goal: dict[str, Any]
    prompt: str
    legal: list[dict[str, Any]]
    wrong_first: list[dict[str, Any]]
    expected_tight_error: str
    expected_tight_error_index: int


def build_supply(template: int, prefix: str, rng: random.Random) -> Fixture:
    # Vary the number of orders and whether fulfillment is split across warehouses.
    count = 2 + template // 2
    split = bool(template % 2)
    sku_ids = [f"SKU-{prefix}-{i + 1}" for i in range(count)]
    order_ids = [f"ORD-{prefix}-{i + 1}" for i in range(count)]
    warehouse_ids = [f"WH-{prefix}-{i + 1}" for i in range(count if split else 1)]
    supplier = f"SUP-{prefix}"
    inventory, suppliers, orders = [], [], []
    reads = [call("list_pending_orders")]
    purchases, shipping = [], []
    required_credit = 0
    fulfillment_cost = 0
    for i, (sku, order_id) in enumerate(zip(sku_ids, order_ids)):
        wh = warehouse_ids[i if split else 0]
        on_hand = rng.randint(2, 6)
        deficit = rng.randint(4, 9)
        qty = on_hand + deficit
        unit, fixed, moq = rng.randint(4, 11), rng.randint(2, 7), rng.randint(1, 4)
        required_credit += unit * deficit + fixed
        fulfillment_cost += qty * (i + 1 if split else 1)
        inventory.append({"wh_id": wh, "sku": sku, "on_hand": on_hand, "reserved": 0})
        suppliers.append(
            {
                "sup_id": supplier,
                "sku": sku,
                "unit_cost": unit,
                "fixed_fee": fixed,
                "min_order_qty": moq,
                "spent": 0,
                "active": 1,
                "allowed_whs": warehouse_ids[:],
            }
        )
        orders.append(
            {
                "order_id": order_id,
                "sku": sku,
                "qty": qty,
                "target_wh": wh,
                "status": "PENDING",
                "allowed_whs": [wh],
            }
        )
        reads += [call("query_inventory", sku=sku), call("query_suppliers", sku=sku)]
        purchases.append(call("create_restock_po", sup_id=supplier, wh_id=wh, sku=sku, qty=deficit))
        shipping += [
            call("reserve_stock", order_id=order_id, wh_id=wh, qty=qty),
            call("ship_order", order_id=order_id),
        ]

    extra_unit_cost = suppliers[0]["unit_cost"]
    shared_moq = suppliers[0]["min_order_qty"]
    for row in suppliers:
        # The shared guard stores one supplier rule across catalog rows.
        row["min_order_qty"] = shared_moq
        row["credit_limit"] = required_credit
    tight = {
        "warehouses": [
            {
                "wh_id": wh,
                "name": wh,
                "frozen": 0,
                "max_capacity": 500,
                "used_capacity": sum(row["on_hand"] for row in inventory if row["wh_id"] == wh),
                "fulfill_cost_per_unit": i + 1,
            }
            for i, wh in enumerate(warehouse_ids)
        ],
        "inventory": inventory,
        "suppliers": suppliers,
        "orders": orders,
        "reservations": [],
        "max_restock_budget": required_credit + fulfillment_cost + 3 * extra_unit_cost,
    }
    relaxed = deepcopy(tight)
    for row in relaxed["suppliers"]:
        row["credit_limit"] += 2 * extra_unit_cost

    wrong_purchases = deepcopy(purchases)
    wrong_purchases[0]["arguments"]["qty"] += 1
    return Fixture(
        initial_relaxed=relaxed,
        initial_tight=tight,
        changed_paths=tuple(f"/suppliers/{i}/credit_limit" for i in range(count)),
        goal={"shipped_orders": order_ids},
        prompt=(
            "Fulfill and ship the following pending orders: "
            + ", ".join(order_ids)
            + ". Inspect stock and supplier terms, and honor the purchasing and fulfillment constraints."
        ),
        legal=reads + purchases + shipping,
        wrong_first=reads + wrong_purchases + shipping,
        expected_tight_error="CREDIT_EXCEEDED",
        expected_tight_error_index=len(reads) + len(purchases) - 1,
    )


def build_db(template: int, prefix: str, rng: random.Random) -> Fixture:
    # Include both inflow and refund tasks and two- and three-ticket workflows.
    count = 2 + template // 2
    refund = bool(template % 2)
    account, tickets, invoices, entries = f"ACC-{prefix}", [], [], []
    reads = [call("list_open_tickets")]
    actions: list[list[dict[str, Any]]] = []
    corrections: list[int] = []
    for i in range(count):
        tid, inv = f"TKT-{prefix}-{i + 1}", f"INV-{prefix}-{i + 1}"
        subtotal, tax = rng.randint(1400, 3400), rng.choice((325, 575, 725))
        discount = rng.randint(15, 95)
        discount_pre_tax = (template + i) % 2
        late, days = (rng.choice((100, 150, 175)), i + 1) if i == 0 or template >= 2 else (0, 0)
        expected = compute_expected_cents(subtotal, tax, discount, discount_pre_tax, late, days)
        delta = rng.randint(110, 390) * (-1 if (i == 0 and refund) else 1)
        corrections.append(delta)
        tickets.append(
            {"ticket_id": tid, "title": f"Monthly reconciliation {i + 1}", "status": "OPEN", "max_adjustments": 2}
        )
        invoices.append(
            {
                "inv_id": inv,
                "ticket_id": tid,
                "acct_id": account,
                "subtotal_cents": subtotal,
                "tax_rate_bps": tax,
                "discount_cents": discount,
                "discount_pre_tax": discount_pre_tax,
                "late_fee_bps": late,
                "days_late": days,
                "status": "OPEN",
            }
        )
        entries.append(
            {
                "entry_id": f"ENTRY-{prefix}-{i + 1}",
                "ticket_id": tid,
                "acct_id": account,
                "inv_id": inv,
                "amount_cents": expected - delta,
                "ref_code": f"REF-{prefix}-{i + 1}",
                "is_duplicate": 0,
                "voided": 0,
            }
        )
        reads.append(call("get_invoice_bundle", ticket_id=tid))
        actions.append(
            [
                call("post_adjustment", ticket_id=tid, acct_id=account, inv_id=inv, delta_cents=delta),
                call("run_audit_check", ticket_id=tid),
                call("reconcile_invoice", inv_id=inv),
                call("close_ticket", ticket_id=tid),
            ]
        )
    reads.append(call("query_account", acct_id=account))
    relaxed = {
        "accounts": [
            {"acct_id": account, "owner": "Simulation clearing account", "frozen": 0, "balance_cents": 9000}
        ],
        "tickets": tickets,
        "invoices": invoices,
        "ledger_entries": entries,
    }
    tight = deepcopy(relaxed)
    tight["tickets"][0]["max_adjustments"] = 1
    target_ids = [row["ticket_id"] for row in tickets]
    legal = reads + [step for ticket_steps in actions for step in ticket_steps]
    wrong_adjustment = deepcopy(actions[0][0])
    wrong_adjustment["arguments"]["delta_cents"] = corrections[0] - 1
    correction = deepcopy(actions[0][0])
    correction["arguments"]["delta_cents"] = 1
    return Fixture(
        initial_relaxed=relaxed,
        initial_tight=tight,
        changed_paths=("/tickets/0/max_adjustments",),
        goal={"closed_tickets": target_ids},
        prompt=(
            "Reconcile all invoices in the following tickets, run their audit checks, and close them: "
            + ", ".join(target_ids)
            + ". Follow the adjustment limits and account rules."
        ),
        legal=legal,
        wrong_first=reads + [wrong_adjustment, actions[0][1], correction] + legal[len(reads) + 1 :],
        expected_tight_error="MAX_ADJUSTMENTS_EXCEEDED",
        expected_tight_error_index=len(reads) + 2,
    )


def build_ops(template: int, prefix: str, rng: random.Random) -> Fixture:
    dependency_shapes = (
        ((), (0,), (1,)),
        ((), (0,), (0,)),
        ((), (0,), (0,), (1, 2)),
        ((), (), (0, 1), (2,)),
    )
    dependencies = dependency_shapes[template]
    service_ids = [f"svc-{prefix}-{i + 1}".lower() for i in range(len(dependencies))]
    services, revisions, logs = [], [], []
    reads = [call("get_topology")]
    recovery: list[list[dict[str, Any]]] = []
    for i, (service, deps) in enumerate(zip(service_ids, dependencies)):
        # A plausible but incompatible revision must fail inside the environment
        # rather than being rejected before the restart quota can be consumed.
        policies = (
            ({"min_pool": 32, "max_pool": 96}, {"pool": 48}, {"pool": 16}),
            ({"required_auth": "mode-b"}, {"auth_mode": "mode-b"}, {"auth_mode": "mode-a"}),
            ({"min_schema": 4}, {"schema": 5}, {"schema": 3}),
            ({"min_tls": "1.3"}, {"tls": "1.3"}, {"tls": "1.2"}),
        )
        policy, good_values, bad_values = policies[(template + i) % len(policies)]
        mem_mb = rng.choice((80, 96, 128))
        good, bad = f"rev-{2 * i + 1:02d}", f"rev-{2 * i + 2:02d}"
        services.append(
            {
                "svc_id": service,
                "deps": [service_ids[dep] for dep in deps],
                "status": "DOWN",
                "active_rev": bad,
                "policy": policy,
                "max_restarts": 2,
            }
        )
        logs.append({"svc_id": service, "level": "WARN", "message": "Revision compatibility needs review."})
        revisions += [
            {
                "svc_id": service,
                "rev_id": good,
                "params": {"mem_mb": mem_mb, **good_values},
                "deployed_at": "2026-11-18",
                "note": "Archived candidate",
            },
            {
                "svc_id": service,
                "rev_id": bad,
                "params": {"mem_mb": mem_mb, **bad_values},
                "deployed_at": "2026-11-19",
                "note": "Current candidate",
            },
        ]
        reads.append(call("list_config_revisions", svc_id=service))
        recovery.append(
            [
                call("request_maintenance_window", svc_id=service),
                call("rollback_config", svc_id=service, rev_id=good),
                call("restart_service", svc_id=service),
                call("run_health_probe", svc_id=service),
            ]
        )

    relaxed = {"cluster_mem_cap_mb": 1024, "services": services, "logs": logs, "config_revisions": revisions}
    tight = deepcopy(relaxed)
    tight["services"][0]["max_restarts"] = 1
    legal = reads + [step for service_steps in recovery for step in service_steps]
    wrong = reads + [recovery[0][0], call("restart_service", svc_id=service_ids[0])] + legal[len(reads) + 1 :]
    return Fixture(
        initial_relaxed=relaxed,
        initial_tight=tight,
        changed_paths=("/services/0/max_restarts",),
        goal={"healthy_and_probed": service_ids},
        prompt=(
            "Return the listed services to healthy status and verify each with a health probe: "
            + ", ".join(service_ids)
            + ". Inspect policies and dependencies before applying recovery actions."
        ),
        legal=legal,
        wrong_first=wrong,
        expected_tight_error="MAX_RESTARTS_EXCEEDED",
        expected_tight_error_index=len(reads) + 3,
    )


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def state_differences(left: Any, right: Any, path: str = "") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict) and left.keys() == right.keys():
        return [found for key in sorted(left) for found in state_differences(left[key], right[key], f"{path}/{key}")]
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        return [found for i, (a, b) in enumerate(zip(left, right)) for found in state_differences(a, b, f"{path}/{i}")]
    return [path] if left != right else []


def tool_errors(
    env: BaseSandboxEnv,
    trace: list[dict[str, Any]],
    guard: FSMGuard | None = None,
) -> list[tuple[int, str]]:
    failed = []
    ledger = StateLedger()
    for i, step in enumerate(trace):
        name, args = step["name"], step["arguments"]
        if guard is not None:
            snapshot = env.snapshot_state()
            decision = guard.check(
                domain=env.domain, tool_name=name, args=args, snapshot=snapshot, ledger=ledger
            )
            if not decision.allowed:
                return failed + [(i, "FSM_BLOCKED")]
        result = env.execute_tool(name, args)
        if guard is not None:
            guard.record_outcome(name, args, result.ok, snapshot=snapshot)
            ledger.update(
                step_idx=i, tier="fast", tool_name=name, args=args, result=result, fsm_blocked=False
            )
        if not result.ok:
            failed.append((i, result.error_code or "UNKNOWN_ERROR"))
    return failed


def validate_pair(pair: list[dict[str, Any]], fixture: Fixture) -> dict[str, Any]:
    relaxed, tight = pair
    excluded = {"task_id", "state_label", "initial_state"}
    assert {k: v for k, v in relaxed.items() if k not in excluded} == {
        k: v for k, v in tight.items() if k not in excluded
    }, relaxed["family_id"]
    assert sorted(state_differences(relaxed["initial_state"], tight["initial_state"])) == sorted(
        fixture.changed_paths
    ), relaxed["family_id"]
    results: dict[str, Any] = {}
    for task in pair:
        label = task["state_label"]
        env = create_env(task["domain"])
        try:
            env.reset(task)
            schemas_before = env.tool_schemas()
            assert len(fixture.legal) <= task["max_steps"]
            failed = tool_errors(env, fixture.legal, FSMGuard())
            verdict = env.verify(task)
            assert not failed and verdict.passed, (task["task_id"], failed, verdict.violations)
            env.reset(task)
            # In the wrong-first path, downstream failures after the first
            # state-specific effect are unneeded. Stop at the witness on tight.
            assert len(fixture.wrong_first) <= task["max_steps"]
            if label == "tight":
                trace = fixture.wrong_first[: fixture.expected_tight_error_index + 1]
            else:
                trace = fixture.wrong_first
            errors = tool_errors(env, trace)
            verdict = env.verify(task)
            expected = [] if task["domain"] != "ops_troubleshoot" else [
                (
                    next(i for i, step in enumerate(trace) if step["name"] == "restart_service"),
                    "POLICY_VIOLATION",
                )
            ]
            if label == "tight":
                expected.append((fixture.expected_tight_error_index, fixture.expected_tight_error))
            assert errors == expected, (task["task_id"], errors, expected)
            assert verdict.passed == (label == "relaxed"), (task["task_id"], verdict.violations)
            env.reset(task)
            guarded_errors = tool_errors(env, trace, FSMGuard())
            guarded_expected = expected[:]
            if label == "tight" and task["domain"] in ("db_transaction", "ops_troubleshoot"):
                guarded_expected[-1] = (fixture.expected_tight_error_index, "FSM_BLOCKED")
            assert guarded_errors == guarded_expected, (task["task_id"], guarded_errors, guarded_expected)
            assert env.verify(task).passed == verdict.passed, task["task_id"]
            results[label] = {
                "initial_state_sha256": canonical_hash(task["initial_state"]),
                "tool_schemas_sha256": canonical_hash(schemas_before),
                "legal_trace_passed": True,
                "wrong_first_trace_passed": verdict.passed,
                "wrong_first_error_codes": [code for _, code in errors],
                "guarded_wrong_first_error_codes": [code for _, code in guarded_errors],
            }
        finally:
            env.close()
    assert results["tight"]["tool_schemas_sha256"] == results["relaxed"]["tool_schemas_sha256"]
    return {
        "family_id": relaxed["family_id"],
        "template_id": relaxed["template_id"],
        "prompt_sha256": hashlib.sha256(relaxed["prompt"].encode("utf-8")).hexdigest(),
        "assertions_sha256": canonical_hash(relaxed["assertions"]),
        "changed_initial_state_paths": list(fixture.changed_paths),
        "oracle_steps": len(fixture.legal),
        "first_write_preceding_read_steps": next(
            i
            for i, step in enumerate(fixture.legal)
            if step["name"] in ("create_restock_po", "post_adjustment", "request_maintenance_window")
        ),
        "states": results,
    }


def build() -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    builders = {
        "supply_chain": ("SC", build_supply),
        "db_transaction": ("DB", build_db),
        "ops_troubleshoot": ("OPS", build_ops),
    }
    rng = random.Random(SEED)
    tasks: list[dict[str, Any]] = []
    oracle: dict[str, list[dict[str, Any]]] = {}
    validation = []
    for domain, (domain_prefix, builder) in builders.items():
        for template in range(4):
            template_id = f"{domain}-pilot-{template + 1}"
            for variant in range(2):
                prefix = f"P{domain_prefix}{template + 1}{variant + 1}"
                family_id = prefix.lower()
                fixture = builder(template, prefix, rng)
                pair = []
                for label, state in (("relaxed", fixture.initial_relaxed), ("tight", fixture.initial_tight)):
                    task_id = f"{family_id}-{label}"
                    pair.append(
                        {
                            "task_id": task_id,
                            "family_id": family_id,
                            "template_id": template_id,
                            "state_label": label,
                            "domain": domain,
                            "difficulty": "Pilot",
                            "max_steps": 36,
                            "prompt": fixture.prompt,
                            "initial_state": state,
                            "assertions": fixture.goal,
                        }
                    )
                    oracle[task_id] = fixture.legal
                validation.append(validate_pair(pair, fixture))
                tasks.extend(pair)

    task_ids = {t["task_id"] for t in tasks}
    state_hashes = {
        canonical_hash({"domain": task["domain"], "initial_state": task["initial_state"]}) for task in tasks
    }
    assert len(tasks) == len(task_ids) == len(state_hashes) == len(oracle) == 48
    return tasks, oracle, {
        "version": "v1.0",
        "purpose": "Offline-verified pilot; two concrete variants share each structural template.",
        "generator_seed": SEED,
        "concrete_pair_count": len(validation),
        "structural_template_count": len({r["template_id"] for r in validation}),
        "state_count": len(tasks),
        "distinct_initial_state_count": len(state_hashes),
        "concrete_pairs_by_domain": dict(Counter(t["domain"] for t in tasks if t["state_label"] == "tight")),
        "legal_traces_verified": len(oracle),
        "wrong_first_recovery_verified_pairs": len(validation),
        "pairs": validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Rebuild offline and compare without writing.")
    args = parser.parse_args()
    tasks, oracle, manifest = build()
    outputs = {
        "tasks.json": tasks,
        "oracle_offline_only.json": oracle,
        "validation.json": manifest,
    }
    if not args.check:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, document in outputs.items():
        target: Path = OUTPUT_DIR / filename
        contents = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
        if args.check:
            assert target.read_text(encoding="utf-8") == contents, f"Generated file differs: {target}"
        else:
            target.write_text(contents, encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "pairs"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
