"""Evidence contracts for candidate actions in the local research sandboxes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from aroute.envs.base import ToolResult


Risk = Literal["tight", "relaxed", "unknown", "non_consuming"]
READ_ONLY = frozenset({"inspect_service", "list_config_revisions", "get_invoice_bundle", "query_account",
                       "query_suppliers", "query_inventory", "list_pending_orders"})


@dataclass(frozen=True)
class Read:
    tool: str
    field: str = ""
    value: str = ""

    @property
    def arguments(self) -> dict[str, str]:
        return {self.field: self.value} if self.field else {}


@dataclass(frozen=True)
class Receipt:
    sequence: int
    data: dict[str, Any]


@dataclass(frozen=True)
class Assessment:
    target: tuple[str, ...]
    pools: tuple[str, ...]
    missing: tuple[Read, ...]
    sources: tuple[int, ...]
    risk: Risk
    before: dict[str, int]


class PrewriteControl:
    """Keep only current, explicitly observed facts. Each instance belongs to one episode."""

    def __init__(
        self, domain: str, *, supplement: bool, review: Literal["off", "targeted", "first"] = "off",
        max_reads: int = 24,
    ) -> None:
        self.domain = domain
        self.supplement = supplement
        self.review = review
        self.max_reads = max_reads
        self.receipts: dict[Read, Receipt] = {}
        self.failed_reads: set[Read] = set()
        self.sequence = 0
        self.reads_started = 0
        self.review_used = False

    def observe(self, tool: str, args: dict[str, Any], result: ToolResult) -> Receipt | None:
        """Use public reads and invalidate them after consequential tool outcomes."""
        self.sequence += 1
        if tool in READ_ONLY:
            read = Read(tool, *next(iter(args.items()))) if len(args) == 1 else Read(tool)
            if result.ok and not result.is_write and not result.state_changed and _matches(read, result.data):
                self.receipts[read] = receipt = Receipt(self.sequence, result.data)
                self.failed_reads.discard(read)
                return receipt
            return None

        if self.domain == "ops_troubleshoot":
            service = args.get("svc_id")
            if tool == "restart_service" or (result.ok and tool in ("rollback_config", "request_maintenance_window")):
                self.invalidate(Read("inspect_service", "svc_id", service))
            if result.ok and tool == "rollback_config":
                self.invalidate(Read("list_config_revisions", "svc_id", service))
        elif self.domain == "db_transaction" and result.ok:
            if tool == "post_adjustment":
                self.invalidate(Read("get_invoice_bundle", "ticket_id", args.get("ticket_id")))
                self.invalidate(Read("query_account", "acct_id", args.get("acct_id")))
            elif tool in ("void_duplicate_entry", "close_ticket", "reconcile_invoice", "run_audit_check"):
                # Some tool results omit the affected account or parent ticket.
                self.invalidate_tools("get_invoice_bundle", "query_account")
        elif self.domain == "supply_chain" and result.ok:
            if tool in ("create_restock_po", "ship_order"):
                self.invalidate_tools("query_suppliers", "query_inventory", "list_pending_orders")
            elif tool in ("reserve_stock", "release_reservation"):
                self.invalidate_tools("query_inventory")
        return None

    def invalidate(self, read: Read) -> None:
        self.receipts.pop(read, None)
        self.failed_reads.discard(read)

    def invalidate_tools(self, *tools: str) -> None:
        for read in self.receipts.keys() | self.failed_reads:
            if read.tool in tools:
                self.invalidate(read)

    def pending(self, assessment: Assessment) -> tuple[Read, ...]:
        if not self.supplement:
            return ()
        allowed = (read for read in assessment.missing if read not in self.failed_reads)
        return tuple(allowed)[:max(0, self.max_reads - self.reads_started)]

    def wants_review(self, assessment: Assessment, candidate_tier: str) -> bool:
        return (candidate_tier == "fast" and not self.review_used and not assessment.missing
                and (assessment.risk == "tight" and self.review == "targeted"
                     or assessment.risk in ("tight", "relaxed") and self.review == "first"))

    def assess(self, tool: str, args: dict[str, Any]) -> Assessment | None:
        if self.domain == "ops_troubleshoot" and tool in ("restart_service", "rollback_config"):
            keys = ("svc_id",) if tool == "restart_service" else ("svc_id", "rev_id")
            if not _valid_ids(args, keys):
                return None
            svc = args["svc_id"]
            reads = (Read("inspect_service", "svc_id", svc), Read("list_config_revisions", "svc_id", svc))
            return self._assess(tuple(args[key] for key in keys), (f"restart:{svc}",), reads, tool, args)
        if self.domain == "db_transaction" and tool == "post_adjustment":
            keys = ("ticket_id", "inv_id", "acct_id")
            if not _valid_ids(args, keys) or type(args.get("delta_cents")) is not int:
                return None
            ticket, invoice, account = (args[key] for key in keys)
            reads = (Read("get_invoice_bundle", "ticket_id", ticket), Read("query_account", "acct_id", account))
            return self._assess((ticket, invoice, account), (f"adjustments:{ticket}",), reads, tool, args)
        if self.domain == "supply_chain" and tool == "create_restock_po":
            keys = ("sku", "sup_id", "wh_id")
            if not _valid_ids(args, keys) or type(args.get("qty")) is not int:
                return None
            sku, supplier, warehouse = (args[key] for key in keys)
            reads = (Read("query_suppliers", "sku", sku), Read("query_inventory", "sku", sku),
                     Read("list_pending_orders"))
            return self._assess((sku, supplier, warehouse), (f"credit:{supplier}", "episode_budget"), reads, tool, args)
        return None

    def _assess(self, target, pools, reads, tool, args) -> Assessment:
        valid = [self.receipts.get(read) for read in reads]
        missing = tuple(read for read, receipt in zip(reads, valid) if receipt is None)
        sources = tuple(receipt.sequence for receipt in valid if receipt is not None)
        if missing:
            return Assessment(target, pools, missing, sources, "unknown", {})
        try:
            risk, before = _classify(tool, args, [r.data for r in valid if r is not None])
        except (KeyError, TypeError, ValueError):
            risk, before = "unknown", {}
        return Assessment(target, pools, (), sources, risk, before)


def _valid_ids(args: dict, keys: tuple[str, ...]) -> bool:
    return all(isinstance(args.get(key), str) and args[key].strip() for key in keys)


def _matches(read: Read, data: dict[str, Any]) -> bool:
    if read.tool == "inspect_service":
        svc = data.get("service", {})
        return (read.field == "svc_id" and svc.get("svc_id") == read.value
                and {"active_rev", "active_params", "policy", "maint_approved", "restart_count", "max_restarts"} <= svc.keys())
    if read.tool == "list_config_revisions":
        return (read.field == "svc_id" and data.get("svc_id") == read.value
                and {"active_rev", "policy", "revisions"} <= data.keys())
    if read.tool == "get_invoice_bundle":
        ticket = data.get("ticket", {})
        return (read.field == "ticket_id" and ticket.get("ticket_id") == read.value
                and {"adj_count", "max_adjustments"} <= ticket.keys() and "invoices" in data)
    if read.tool == "query_account":
        account = data.get("account", {})
        return (read.field == "acct_id" and account.get("acct_id") == read.value
                and {"frozen", "balance_cents", "min_reserve_cents"} <= account.keys())
    if read.tool == "list_pending_orders":
        return not read.field and "orders" in data
    key = "suppliers" if read.tool == "query_suppliers" else "warehouses"
    return read.field == "sku" and data.get("sku") == read.value and key in data


def _classify(tool: str, args: dict, facts: list[dict]) -> tuple[Risk, dict[str, int]]:
    if tool == "rollback_config":
        return "non_consuming", {}
    if tool == "restart_service":
        service = facts[0]["service"]
        sid = args["svc_id"]
        if service["svc_id"] != sid or facts[1]["svc_id"] != sid or service["active_rev"] != facts[1]["active_rev"]:
            return "unknown", {}
        used = service["restart_count"]
        remaining = service["max_restarts"] - used
        if not service["maint_approved"] or remaining <= 0:
            return "non_consuming", {"used": used, "remaining": remaining}
        return ("tight" if remaining == 1 else "relaxed"), {"used": used, "remaining": remaining}
    if tool == "post_adjustment":
        ticket, account = facts[0]["ticket"], facts[1]["account"]
        if ticket["ticket_id"] != args["ticket_id"] or account["acct_id"] != args["acct_id"]:
            return "unknown", {}
        used = ticket["adj_count"]
        remaining = ticket["max_adjustments"] - used
        before = {"used": used, "remaining": remaining, "balance": account["balance_cents"]}
        invoice_exists = any(inv["inv_id"] == args["inv_id"] for inv in facts[0]["invoices"])
        delta = args["delta_cents"]
        if (not invoice_exists or not delta or remaining <= 0 or account["frozen"]
                or before["balance"] + delta < account["min_reserve_cents"]):
            return "non_consuming", before
        if remaining == 1:
            return "tight", before
        if before["balance"] >= account["min_reserve_cents"]:
            return "relaxed", before
        return "unknown", before

    # Purchase cost is observable, but a valid replacement order requires a separate feasibility proof.
    quote = next((row for row in facts[0]["suppliers"] if row["sup_id"] == args["sup_id"]), None)
    if quote is None:
        return "non_consuming", {}
    cost = quote["fixed_fee"] + args["qty"] * quote["unit_cost"]
    return "unknown", {"candidate_cost": cost, "supplier_remaining_credit": quote["remaining_credit"],
                       "remaining_episode_budget": facts[0]["remaining_episode_budget"]}
