"""Domain-generic deterministic FSM precondition guardrail V2 and loop circuit-breaker."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .memory_ledger import StateLedger


@dataclass
class GuardDecision:
    """Decision returned by FSMGuard before invoking backend sandbox."""

    allowed: bool
    reason: str | None = None
    suggestion: str | None = None
    loop_triggered: bool = False


def canonical_call_hash(tool_name: str, args: dict[str, Any]) -> str:
    """Stable hash of (tool_name, canonical JSON arguments)."""
    payload = json.dumps({"tool": tool_name, "args": args}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def canonical_snapshot_hash(snapshot: dict[str, Any]) -> str:
    """Stable short hash of canonical environment state snapshot."""
    payload = json.dumps(snapshot, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


class FSMGuard:
    """Deterministic API-contract precondition checker and state-aware duplicate-call breaker.

    Encodes strictly generic domain API contracts (zero task IDs or answers).
    """

    def __init__(self, *, max_repeat_same_call: int = 2) -> None:
        self.max_repeat_same_call = max(1, int(max_repeat_same_call))
        self.call_counts: dict[str, int] = {}
        self.fail_counts: dict[str, int] = {}

    def record_outcome(
        self,
        tool_name: str,
        args: dict[str, Any],
        ok: bool,
        *,
        snapshot: dict[str, Any] | None = None,
    ) -> None:
        skey = f"{canonical_snapshot_hash(snapshot or {})}:{canonical_call_hash(tool_name, args)}"
        self.call_counts[skey] = self.call_counts.get(skey, 0) + 1
        if not ok:
            self.fail_counts[skey] = self.fail_counts.get(skey, 0) + 1

    def check(
        self,
        *,
        domain: str,
        tool_name: str,
        args: dict[str, Any],
        snapshot: dict[str, Any],
        ledger: StateLedger,
    ) -> GuardDecision:
        skey = f"{canonical_snapshot_hash(snapshot)}:{canonical_call_hash(tool_name, args)}"
        if self.fail_counts.get(skey, 0) >= self.max_repeat_same_call:
            return GuardDecision(
                allowed=False,
                reason=f"NO_PROGRESS_LOOP: identical failing call {tool_name}({args}) repeated >= {self.max_repeat_same_call} times without state change",
                suggestion="Inspect missing preconditions or switch parameters/tool instead of retrying identically.",
                loop_triggered=True,
            )
        if self.call_counts.get(skey, 0) >= self.max_repeat_same_call + 2:
            return GuardDecision(
                allowed=False,
                reason=f"NO_PROGRESS_LOOP: redundant call {tool_name}({args}) repeated without state change",
                suggestion="Proceed to the next write/action.",
                loop_triggered=True,
            )
        if self.call_counts.get(skey, 0) >= self.max_repeat_same_call + 1:
            return GuardDecision(
                allowed=False,
                reason=f"REDUNDANT_CALL: identical call {tool_name}({args}) already succeeded in current state",
                suggestion="Do not re-fetch unchanged state; execute the next required write or check.",
                loop_triggered=False,
            )

        if domain == "supply_chain":
            return self._check_supply_chain(tool_name, args, snapshot, ledger)
        if domain == "db_transaction":
            return self._check_db_transaction(tool_name, args, snapshot, ledger)
        if domain == "ops_troubleshoot":
            return self._check_ops_troubleshoot(tool_name, args, snapshot, ledger)
        return GuardDecision(allowed=True)

    @staticmethod
    def _check_supply_chain(
        tool_name: str,
        args: dict[str, Any],
        snapshot: dict[str, Any],
        ledger: StateLedger,
    ) -> GuardDecision:
        if tool_name in ("reserve_stock", "create_restock_po"):
            wh_id = str(args.get("wh_id", ""))
            if wh_id and wh_id in ledger.frozen_warehouses:
                return GuardDecision(
                    allowed=False,
                    reason=f"FSM_PRECONDITION: warehouse '{wh_id}' is known to be frozen",
                    suggestion="Select an active non-frozen warehouse from query_inventory.",
                )
        if tool_name == "reserve_stock":
            oid = str(args.get("order_id", ""))
            wh_id = str(args.get("wh_id", ""))
            allowed_whs = ledger.order_allowed_whs.get(oid, [])
            if allowed_whs and wh_id not in allowed_whs:
                return GuardDecision(
                    allowed=False,
                    reason=f"FSM_PRECONDITION: order '{oid}' only permits warehouses {allowed_whs}, got '{wh_id}'",
                    suggestion=f"Reserve stock from an allowed warehouse in {allowed_whs}.",
                )
        if tool_name == "create_restock_po":
            sup_id = str(args.get("sup_id", ""))
            wh_id = str(args.get("wh_id", ""))
            qty = int(args.get("qty", 0))
            if sup_id and sup_id in ledger.inactive_suppliers:
                return GuardDecision(
                    allowed=False,
                    reason=f"FSM_PRECONDITION: supplier '{sup_id}' is inactive",
                    suggestion="Choose an active supplier from query_suppliers.",
                )
            rules = ledger.supplier_rules.get(sup_id)
            if rules:
                if qty < int(rules["min_order_qty"]):
                    return GuardDecision(
                        allowed=False,
                        reason=f"FSM_PRECONDITION: qty={qty} < supplier '{sup_id}' MOQ={rules['min_order_qty']}",
                        suggestion="Increase order quantity to at least MOQ or pick another supplier.",
                    )
                sup_whs = rules.get("allowed_whs", [])
                if sup_whs and wh_id not in sup_whs:
                    return GuardDecision(
                        allowed=False,
                        reason=f"FSM_PRECONDITION: supplier '{sup_id}' only ships to {sup_whs}, got '{wh_id}'",
                        suggestion=f"Send PO to one of {sup_whs} or choose a compatible supplier.",
                    )
        if tool_name == "ship_order":
            oid = str(args.get("order_id", ""))
            orders = snapshot.get("orders", {})
            if oid in orders and orders[oid] != "PENDING":
                return GuardDecision(
                    allowed=False,
                    reason=f"FSM_PRECONDITION: order '{oid}' status is {orders[oid]}",
                    suggestion="Ship only PENDING orders.",
                )
            req_qty = ledger.known_order_qty.get(oid)
            res_qty = int(snapshot.get("reserved", {}).get(oid, 0))
            if req_qty is not None and res_qty < req_qty:
                return GuardDecision(
                    allowed=False,
                    reason=f"FSM_PRECONDITION: order '{oid}' has {res_qty}/{req_qty} units reserved",
                    suggestion=f"Reserve the remaining {req_qty - res_qty} units before calling ship_order.",
                )
        return GuardDecision(allowed=True)

    @staticmethod
    def _check_db_transaction(
        tool_name: str,
        args: dict[str, Any],
        snapshot: dict[str, Any],
        ledger: StateLedger,
    ) -> GuardDecision:
        if tool_name == "post_adjustment":
            tid = str(args.get("ticket_id", ""))
            tstate = snapshot.get("tickets", {}).get(tid, {})
            if tstate and int(tstate.get("adj_count", 0)) >= int(tstate.get("max_adjustments", 999)):
                return GuardDecision(
                    allowed=False,
                    reason=f"FSM_PRECONDITION: ticket '{tid}' reached max_adjustments={tstate['max_adjustments']}",
                    suggestion="Verify invoice arithmetic first; do not exceed max_adjustments quota.",
                )
            aid = str(args.get("acct_id", ""))
            delta = int(args.get("delta_cents", 0))
            if aid and aid in ledger.frozen_accounts:
                fb = ledger.account_meta.get(aid, {}).get("fallback_acct_id")
                hint = f"Use fallback_acct_id '{fb}'." if fb else "Use a non-frozen settlement account."
                return GuardDecision(
                    allowed=False,
                    reason=f"FSM_PRECONDITION: account '{aid}' is frozen",
                    suggestion=hint,
                )
            meta = ledger.account_meta.get(aid)
            if meta and (int(meta["balance_cents"]) + delta < int(meta["min_reserve_cents"])):
                fb = meta.get("fallback_acct_id")
                hint = f"Use fallback_acct_id '{fb}'." if fb else "Use an account with sufficient reserve."
                return GuardDecision(
                    allowed=False,
                    reason=f"FSM_PRECONDITION: adjustment on '{aid}' would breach min_reserve_cents",
                    suggestion=hint,
                )
        if tool_name == "reconcile_invoice":
            inv_id = str(args.get("inv_id", ""))
            tid = ledger.invoice_to_ticket.get(inv_id)
            if tid:
                tstate = snapshot.get("tickets", {}).get(tid, {})
                if int(tstate.get("audit_passed", 0)) == 0:
                    return GuardDecision(
                        allowed=False,
                        reason=f"FSM_PRECONDITION: parent ticket '{tid}' has audit_passed=0",
                        suggestion=f"Call run_audit_check(ticket_id='{tid}') after fixing entries before reconciling.",
                    )
        if tool_name == "close_ticket":
            tid = str(args.get("ticket_id", ""))
            tstate = snapshot.get("tickets", {}).get(tid, {})
            if tstate and int(tstate.get("audit_passed", 0)) == 0:
                return GuardDecision(
                    allowed=False,
                    reason=f"FSM_PRECONDITION: ticket '{tid}' requires run_audit_check to pass first",
                    suggestion=f"Fix mismatches/duplicates and call run_audit_check(ticket_id='{tid}').",
                )
            invs = ledger.ticket_invoices.get(tid, [])
            inv_statuses = snapshot.get("invoices", {})
            unrec = [i for i in invs if inv_statuses.get(i) != "RECONCILED"]
            if unrec:
                return GuardDecision(
                    allowed=False,
                    reason=f"FSM_PRECONDITION: ticket '{tid}' has unreconciled invoices {unrec}",
                    suggestion="Call reconcile_invoice on every invoice in the ticket before closing.",
                )
        return GuardDecision(allowed=True)

    @staticmethod
    def _check_ops_troubleshoot(
        tool_name: str,
        args: dict[str, Any],
        snapshot: dict[str, Any],
        ledger: StateLedger,
    ) -> GuardDecision:
        svcs = snapshot.get("services", {})
        sid = str(args.get("svc_id", ""))
        if tool_name in ("rollback_config", "restart_service") and sid in svcs:
            if int(svcs[sid].get("maint", 0)) == 0:
                return GuardDecision(
                    allowed=False,
                    reason=f"FSM_PRECONDITION: service '{sid}' has no approved maintenance window",
                    suggestion=f"Call request_maintenance_window(svc_id='{sid}') before mutating it.",
                )
        if tool_name == "restart_service" and sid in svcs:
            if int(svcs[sid].get("restarts", 0)) >= int(svcs[sid].get("max_restarts", 2)):
                return GuardDecision(
                    allowed=False,
                    reason=f"FSM_PRECONDITION: service '{sid}' exhausted max_restarts quota",
                    suggestion="Do not retry restart_service on this service.",
                    loop_triggered=True,
                )
            deps = ledger.service_deps.get(sid, [])
            bad = [d for d in deps if svcs.get(d, {}).get("status") != "HEALTHY"]
            if bad:
                return GuardDecision(
                    allowed=False,
                    reason=f"FSM_PRECONDITION: upstream dependencies {bad} of '{sid}' are not HEALTHY",
                    suggestion=f"Recover and restart upstream services {bad} first.",
                )
            cap_mb = int(snapshot.get("cluster_mem_cap_mb", 0))
            tot_mb = int(snapshot.get("cluster_mem_mb", 0))
            if cap_mb > 0 and tot_mb > cap_mb:
                return GuardDecision(
                    allowed=False,
                    reason=f"FSM_PRECONDITION: active cluster mem {tot_mb}MB > cluster_mem_cap_mb {cap_mb}MB",
                    suggestion="Select a lower-memory config revision that fits policy & cluster memory cap before restarting.",
                )
        if tool_name == "run_health_probe" and sid in svcs:
            if svcs[sid].get("status") != "HEALTHY":
                return GuardDecision(
                    allowed=False,
                    reason=f"FSM_PRECONDITION: service '{sid}' status is {svcs[sid].get('status')}",
                    suggestion=f"Call restart_service(svc_id='{sid}') before run_health_probe.",
                )
        return GuardDecision(allowed=True)

    @staticmethod
    def check_stop(
        *,
        domain: str,
        snapshot: dict[str, Any],
        ledger: StateLedger,
    ) -> GuardDecision:
        """Prevent premature __STOP__ when observable domain goals remain incomplete."""
        if domain == "supply_chain":
            pending = [oid for oid, st in snapshot.get("orders", {}).items() if st == "PENDING"]
            if pending:
                return GuardDecision(
                    allowed=False,
                    reason=f"INCOMPLETE_STOP: pending orders not shipped: {pending}",
                    suggestion="Finish fulfilling and shipping all PENDING orders before stopping.",
                )
        elif domain == "db_transaction":
            open_t = [tid for tid, t in snapshot.get("tickets", {}).items() if t.get("status") != "CLOSED"]
            if open_t:
                return GuardDecision(
                    allowed=False,
                    reason=f"INCOMPLETE_STOP: open tickets remain unclosed: {open_t}",
                    suggestion="Reconcile all invoices, pass run_audit_check, and close all open tickets before stopping.",
                )
        elif domain == "ops_troubleshoot":
            svcs = snapshot.get("services", {})
            bad = [sid for sid, s in svcs.items() if s.get("status") != "HEALTHY"]
            unprobed = [
                sid
                for sid, s in svcs.items()
                if int(s.get("restarts", 0)) > 0 and int(s.get("probed", 0)) == 0
            ]
            if bad or unprobed:
                return GuardDecision(
                    allowed=False,
                    reason=f"INCOMPLETE_STOP: unhealthy={bad}, unprobed_restarted={unprobed}",
                    suggestion="Recover all non-HEALTHY services and run_health_probe on restarted services before stopping.",
                )
        return GuardDecision(allowed=True)
