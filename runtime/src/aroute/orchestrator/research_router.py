"""Matched two-tier routing arms for the paired business-state experiment."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .memory_ledger import StateLedger
from .router import RouterParams, StepRouter, load_router_params


def _supply_signals(ledger: StateLedger, assess_risk: bool) -> tuple[bool, bool | None]:
    """Detect a shared supplier whose known purchase requirement exhausts available credit."""
    pending = {oid for oid, order in ledger.observed_orders.items() if order.get("status") == "PENDING"}
    if len(pending) < 2 or ledger.po_writes or not pending <= ledger.order_skus.keys():
        return False, None

    orders_by_sku: dict[str, list[str]] = defaultdict(list)
    for oid in sorted(pending):
        orders_by_sku[ledger.order_skus[oid]].append(oid)

    known_skus = (ledger.inventory_by_sku.keys() & ledger.supplier_offers_by_sku.keys()) - ledger.stale_inventory_skus
    if not orders_by_sku.keys() <= known_skus:
        return False, None
    if not assess_risk:
        return True, None

    costs_by_supplier: dict[str, list[int]] = defaultdict(list)
    credit_by_supplier: dict[str, int] = {}
    unknown_offer = False
    for sku, order_ids in orders_by_sku.items():
        warehouses = ledger.inventory_by_sku[sku]
        if any("active_reservations" not in wh for wh in warehouses):
            return True, None
        reserved: dict[str, int] = defaultdict(int)
        for wh in warehouses:
            for reservation in wh["active_reservations"]:
                reserved[str(reservation["order_id"])] += int(reservation["qty"])
        demand = sum(max(0, ledger.known_order_qty[oid] - reserved[oid]) for oid in order_ids)
        unrestricted = any(not ledger.order_allowed_whs[oid] for oid in order_ids)
        allowed = {wh for oid in order_ids for wh in ledger.order_allowed_whs[oid]}
        available = sum(
            max(0, int(wh["available"]))
            for wh in warehouses
            if not wh["frozen"] and (unrestricted or wh["wh_id"] in allowed)
        )
        deficit = demand - available
        if deficit <= 0:
            continue

        active_offers = [offer for offer in ledger.supplier_offers_by_sku[sku] if offer["active"]]
        if len(active_offers) != 1 or "remaining_credit" not in active_offers[0]:
            unknown_offer = True
            continue
        offer = active_offers[0]
        supplier = str(offer["sup_id"])
        cost = max(deficit, int(offer["min_order_qty"])) * int(offer["unit_cost"]) + int(offer["fixed_fee"])
        costs_by_supplier[supplier].append(cost)
        credit = int(offer["remaining_credit"])
        credit_by_supplier[supplier] = min(credit, credit_by_supplier.get(supplier, credit))

    tight = any(
        len(costs) > 1 and sum(costs) >= credit_by_supplier[supplier]
        for supplier, costs in costs_by_supplier.items()
    )
    return True, tight if tight or not unknown_offer else None


def _state_signals(ledger: StateLedger, domain: str, assess_risk: bool) -> tuple[bool, bool | None]:
    if domain == "supply_chain":
        return _supply_signals(ledger, assess_risk)
    if domain == "db_transaction":
        tickets = {tid: ticket for tid, ticket in ledger.observed_tickets.items() if ticket.get("status") != "CLOSED"}
        accounts = set(ledger.entry_to_acct.values())
        ready = (
            len(tickets) >= 2
            and tickets.keys() <= ledger.observed_invoice_bundles
            and bool(accounts)
            and accounts <= ledger.account_meta.keys()
            and ledger.adjustment_writes == 0
        )
        if not ready or not assess_risk or any(not {"max_adjustments", "adj_count"} <= ticket.keys() for ticket in tickets.values()):
            return ready, None
        return ready, any(int(ticket["max_adjustments"]) - int(ticket["adj_count"]) == 1 for ticket in tickets.values())
    if domain == "ops_troubleshoot":
        unhealthy = {sid: svc for sid, svc in ledger.observed_services.items() if svc.get("status") != "HEALTHY"}
        eligible = [svc for sid, svc in unhealthy.items() if svc.get("maint_approved") and sid in ledger.observed_rev_svcs]
        ready = bool(eligible) and len(unhealthy) >= 2 and ledger.rollback_writes == 0
        if not ready or not assess_risk or any(not {"max_restarts", "restart_count"} <= svc.keys() for svc in eligible):
            return ready, None
        return ready, any(int(svc["max_restarts"]) - int(svc["restart_count"]) == 1 for svc in eligible)
    return False, None


class ResearchStepRouter(StepRouter):
    """Hold recovery and guards fixed while changing the proactive eligibility rule."""

    def __init__(self, arm: str, params: RouterParams | None = None, *, use_guards: bool = True) -> None:
        if arm not in ("p0", "p1", "p2", "fast", "think"):
            raise ValueError(f"Unknown research arm: {arm}")
        base = params or load_router_params()
        super().__init__("bar_fsm", base.model_copy(update={"use_proactive_risk": False}))
        self.arm = arm
        self.use_guards = use_guards

    def fsm_enabled(self) -> bool:
        return self.use_guards

    def select_tier(
        self,
        *,
        ledger: StateLedger,
        remaining_budget_ratio: float,
        step_idx: int,
        difficulty: str = "",
        domain: str = "",
        snapshot: dict[str, Any] | None = None,
        active_pricing_period: str = "offpeak",
    ) -> str:
        # The shared router interface still accepts legacy inputs. Research routing reads neither.
        _ = (difficulty, snapshot)
        ready, high_risk = _state_signals(ledger, domain, assess_risk=self.arm == "p1")
        if self.arm in ("fast", "think"):
            tier, reason = self.arm, "fixed_tier"
        else:
            tier = super().select_tier(
                ledger=ledger, remaining_budget_ratio=remaining_budget_ratio, step_idx=step_idx,
                active_pricing_period=active_pricing_period,
            )
            reason = "routine"
            if self.params.use_budget_throttle and remaining_budget_ratio <= self.params.rho_min:
                reason = "budget_throttle"
            elif tier == "think":
                if ledger.just_rejected_stop:
                    reason = "reactive_stop"
                elif ledger.error_streak >= self.params.tau_err:
                    reason = "reactive_error"
                else:
                    reason = "reactive_repeat"
            elif ready and ledger.proactive_think_steps < self.params.max_proactive_think_steps:
                if self.arm == "p2" or (self.arm == "p1" and high_risk):
                    tier = "think"
                    reason = "proactive_fixed" if self.arm == "p2" else "proactive_state"
                    ledger.proactive_think_steps += 1

        self.last_route_event = {
            "arm": self.arm, "reason": reason, "ready": ready,
            "high_risk": high_risk if self.arm == "p1" else None,
            "state_observability": ("unknown" if high_risk is None else "observed") if self.arm == "p1" else None,
            "proactive_count": ledger.proactive_think_steps,
        }
        return tier
