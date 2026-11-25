"""Deterministic SQLite sandbox V3 for multi-table financial reconciliation with 3-stage arithmetic and liquidity ordering."""

from __future__ import annotations

import uuid
from typing import Any

from .base import BaseSandboxEnv, ToolResult, VerifyResult

DB_READ_TOOLS = {
    "list_open_tickets",
    "get_invoice_bundle",
    "query_account",
    "run_audit_check",
}
DB_WRITE_TOOLS = {
    "void_duplicate_entry",
    "post_adjustment",
    "reconcile_invoice",
    "close_ticket",
}


def compute_expected_cents(
    subtotal_cents: int,
    tax_rate_bps: int,
    discount_cents: int,
    discount_pre_tax: int,
    late_fee_bps: int = 0,
    days_late: int = 0,
) -> int:
    """Exact 3-stage deterministic invoice formula in integer cents (half-up via +5000//10000)."""
    sub = int(subtotal_cents)
    tax = int(tax_rate_bps)
    disc = int(discount_cents)
    if int(discount_pre_tax) == 1:
        base = sub - disc
        taxed = base + (base * tax + 5000) // 10000
    else:
        taxed = sub + (sub * tax + 5000) // 10000 - disc
    if int(days_late) > 0 and int(late_fee_bps) > 0:
        late_fee = (taxed * int(late_fee_bps) + 5000) // 10000
        return taxed + late_fee
    return taxed


def _schema(name: str, desc: str, props: dict[str, Any], req: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req},
        },
    }


class DbTransactionEnv(BaseSandboxEnv):
    """V3 spoiler-free financial reconciliation with 3-stage arithmetic, noise entries, zero-slack quotas, and liquidity ordering."""

    domain = "db_transaction"

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [
            _schema(
                "list_open_tickets",
                "List reconciliation tickets with status, audit state, and max adjustment quota.",
                {},
                [],
            ),
            _schema(
                "get_invoice_bundle",
                (
                    "Get invoices and ledger entries for a ticket. Expected total is NOT precomputed; compute exact integer cents using: "
                    "if discount_pre_tax==1: base=subtotal_cents-discount_cents, taxed=base+(base*tax_rate_bps+5000)//10000; "
                    "else: taxed=subtotal_cents+(subtotal_cents*tax_rate_bps+5000)//10000-discount_cents. "
                    "Then if days_late>0 and late_fee_bps>0: expected=taxed+(taxed*late_fee_bps+5000)//10000; else: expected=taxed. "
                    "Ignore entries with voided==1 and include valid negative credit memos (amount_cents<0) in active sums. "
                    "Duplicate entries share identical (inv_id, acct_id, amount_cents, ref_code) with voided==0; distinct ref_codes are valid installments."
                ),
                {"ticket_id": {"type": "string", "description": "Ticket ID, e.g. TKT-101"}},
                ["ticket_id"],
            ),
            _schema(
                "query_account",
                "Query an account's frozen flag, balance_cents, min_reserve_cents, and fallback_acct_id.",
                {"acct_id": {"type": "string"}},
                ["acct_id"],
            ),
            _schema(
                "run_audit_check",
                "Verify zero active duplicates and invoice sum conservation on a ticket (returns audit_passed, has_active_duplicates, unbalanced_invoices).",
                {"ticket_id": {"type": "string"}},
                ["ticket_id"],
            ),
            _schema(
                "void_duplicate_entry",
                "Void an identified duplicate ledger entry and reverse its balance impact (rejects valid non-duplicate entries).",
                {"entry_id": {"type": "string"}},
                ["entry_id"],
            ),
            _schema(
                "post_adjustment",
                "Post an adjustment ledger entry in integer cents to a non-frozen account whose post-adjustment balance stays >= min_reserve_cents (strict max_adjustments quota).",
                {
                    "ticket_id": {"type": "string"},
                    "acct_id": {"type": "string"},
                    "inv_id": {"type": "string"},
                    "delta_cents": {"type": "integer"},
                },
                ["ticket_id", "acct_id", "inv_id", "delta_cents"],
            ),
            _schema(
                "reconcile_invoice",
                "Mark an invoice RECONCILED after the parent ticket passes run_audit_check.",
                {"inv_id": {"type": "string"}},
                ["inv_id"],
            ),
            _schema(
                "close_ticket",
                "Close a reconciliation ticket once audit_passed=1 and all its invoices are RECONCILED.",
                {"ticket_id": {"type": "string"}},
                ["ticket_id"],
            ),
        ]

    def is_write_tool(self, name: str) -> bool:
        return name in DB_WRITE_TOOLS

    def reset(self, task_spec: dict[str, Any]) -> None:
        self.task_spec = task_spec
        self.precondition_violations = 0
        self.write_count = 0
        self.read_count = 0
        conn = self._new_memory_conn()
        conn.executescript(
            """
            CREATE TABLE accounts (
                acct_id TEXT PRIMARY KEY,
                owner TEXT NOT NULL,
                frozen INTEGER NOT NULL,
                balance_cents INTEGER NOT NULL,
                min_reserve_cents INTEGER NOT NULL DEFAULT 0,
                fallback_acct_id TEXT
            );
            CREATE TABLE reconciliation_tickets (
                ticket_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                status TEXT NOT NULL,
                audit_passed INTEGER NOT NULL,
                max_adjustments INTEGER NOT NULL DEFAULT 5,
                adj_count INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE invoices (
                inv_id TEXT PRIMARY KEY,
                ticket_id TEXT NOT NULL,
                acct_id TEXT NOT NULL,
                subtotal_cents INTEGER NOT NULL,
                tax_rate_bps INTEGER NOT NULL,
                discount_cents INTEGER NOT NULL,
                discount_pre_tax INTEGER NOT NULL,
                late_fee_bps INTEGER NOT NULL DEFAULT 0,
                days_late INTEGER NOT NULL DEFAULT 0,
                expected_cents INTEGER NOT NULL,
                status TEXT NOT NULL
            );
            CREATE TABLE ledger_entries (
                entry_id TEXT PRIMARY KEY,
                ticket_id TEXT NOT NULL,
                acct_id TEXT NOT NULL,
                inv_id TEXT NOT NULL,
                amount_cents INTEGER NOT NULL,
                ref_code TEXT NOT NULL DEFAULT 'REF-0',
                posted_at TEXT NOT NULL DEFAULT '2026-11-01',
                memo TEXT NOT NULL DEFAULT '',
                is_duplicate INTEGER NOT NULL,
                voided INTEGER NOT NULL
            );
            """
        )
        init = task_spec["initial_state"]
        for a in init["accounts"]:
            conn.execute(
                "INSERT INTO accounts VALUES (?, ?, ?, ?, ?, ?)",
                (
                    a["acct_id"],
                    a["owner"],
                    int(a["frozen"]),
                    int(a["balance_cents"]),
                    int(a.get("min_reserve_cents", 0)),
                    a.get("fallback_acct_id"),
                ),
            )
        for t in init["tickets"]:
            conn.execute(
                "INSERT INTO reconciliation_tickets VALUES (?, ?, ?, ?, ?, ?)",
                (
                    t["ticket_id"],
                    t["title"],
                    t["status"],
                    int(t.get("audit_passed", 0)),
                    int(t.get("max_adjustments", 5)),
                    int(t.get("adj_count", 0)),
                ),
            )
        for inv in init["invoices"]:
            exp = compute_expected_cents(
                int(inv["subtotal_cents"]),
                int(inv["tax_rate_bps"]),
                int(inv["discount_cents"]),
                int(inv["discount_pre_tax"]),
                int(inv.get("late_fee_bps", 0)),
                int(inv.get("days_late", 0)),
            )
            if "expected_cents" in inv and int(inv["expected_cents"]) != exp:
                raise ValueError(f"Invoice {inv['inv_id']} expected_cents mismatch: {inv['expected_cents']} != {exp}")
            conn.execute(
                "INSERT INTO invoices VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    inv["inv_id"],
                    inv["ticket_id"],
                    inv["acct_id"],
                    int(inv["subtotal_cents"]),
                    int(inv["tax_rate_bps"]),
                    int(inv["discount_cents"]),
                    int(inv["discount_pre_tax"]),
                    int(inv.get("late_fee_bps", 0)),
                    int(inv.get("days_late", 0)),
                    exp,
                    inv["status"],
                ),
            )
        for le in init["ledger_entries"]:
            conn.execute(
                "INSERT INTO ledger_entries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    le["entry_id"],
                    le["ticket_id"],
                    le["acct_id"],
                    le["inv_id"],
                    int(le["amount_cents"]),
                    le.get("ref_code", "REF-0"),
                    le.get("posted_at", "2026-11-01"),
                    le.get("memo", ""),
                    int(le["is_duplicate"]),
                    int(le["voided"]),
                ),
            )
        conn.commit()

    def execute_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
        assert self.conn is not None
        if name in DB_READ_TOOLS:
            self.read_count += 1
        elif name in DB_WRITE_TOOLS:
            self.write_count += 1
        else:
            return ToolResult(ok=False, error_code="UNKNOWN_TOOL", error_msg=f"Unknown tool: {name}")

        c = self.conn
        try:
            if name == "list_open_tickets":
                rows = [
                    dict(r)
                    for r in c.execute("SELECT * FROM reconciliation_tickets ORDER BY ticket_id").fetchall()
                ]
                return ToolResult(ok=True, data={"tickets": rows})

            if name == "get_invoice_bundle":
                tid = str(args.get("ticket_id", ""))
                trow = c.execute("SELECT * FROM reconciliation_tickets WHERE ticket_id = ?", (tid,)).fetchone()
                if not trow:
                    return ToolResult(ok=False, error_code="NOT_FOUND", error_msg=f"Ticket '{tid}' not found")
                invs = [
                    dict(r)
                    for r in c.execute(
                        """
                        SELECT inv_id, ticket_id, acct_id, subtotal_cents, tax_rate_bps,
                               discount_cents, discount_pre_tax, late_fee_bps, days_late, status
                        FROM invoices WHERE ticket_id = ? ORDER BY inv_id
                        """,
                        (tid,),
                    ).fetchall()
                ]
                entries = [
                    dict(r)
                    for r in c.execute(
                        """
                        SELECT entry_id, ticket_id, acct_id, inv_id, amount_cents,
                               ref_code, posted_at, memo, voided
                        FROM ledger_entries WHERE ticket_id = ? ORDER BY entry_id
                        """,
                        (tid,),
                    ).fetchall()
                ]
                return ToolResult(ok=True, data={"ticket": dict(trow), "invoices": invs, "ledger_entries": entries})

            if name == "query_account":
                aid = str(args.get("acct_id", ""))
                row = c.execute("SELECT * FROM accounts WHERE acct_id = ?", (aid,)).fetchone()
                if not row:
                    return ToolResult(ok=False, error_code="NOT_FOUND", error_msg=f"Account '{aid}' not found")
                return ToolResult(ok=True, data={"account": dict(row)})

            if name == "run_audit_check":
                tid = str(args.get("ticket_id", ""))
                trow = c.execute("SELECT * FROM reconciliation_tickets WHERE ticket_id = ?", (tid,)).fetchone()
                if not trow:
                    return ToolResult(ok=False, error_code="NOT_FOUND", error_msg=f"Ticket '{tid}' not found")
                dup_count = int(
                    c.execute(
                        "SELECT COUNT(*) FROM ledger_entries WHERE ticket_id = ? AND is_duplicate = 1 AND voided = 0",
                        (tid,),
                    ).fetchone()[0]
                )
                unbalanced: list[str] = []
                for inv in c.execute("SELECT inv_id, expected_cents FROM invoices WHERE ticket_id = ? ORDER BY inv_id", (tid,)).fetchall():
                    actual = int(
                        c.execute(
                            "SELECT COALESCE(SUM(amount_cents), 0) FROM ledger_entries WHERE inv_id = ? AND voided = 0",
                            (inv["inv_id"],),
                        ).fetchone()[0]
                    )
                    if actual != int(inv["expected_cents"]):
                        unbalanced.append(str(inv["inv_id"]))
                has_dups = dup_count > 0
                passed = (not has_dups) and (len(unbalanced) == 0)
                c.execute(
                    "UPDATE reconciliation_tickets SET audit_passed = ? WHERE ticket_id = ?",
                    (1 if passed else 0, tid),
                )
                c.commit()
                return ToolResult(
                    ok=True,
                    data={
                        "ticket_id": tid,
                        "audit_passed": passed,
                        "has_active_duplicates": has_dups,
                        "unbalanced_invoices": unbalanced,
                    },
                    state_changed=True,
                )

            if name == "void_duplicate_entry":
                eid = str(args.get("entry_id", ""))
                row = c.execute("SELECT * FROM ledger_entries WHERE entry_id = ?", (eid,)).fetchone()
                if not row:
                    return ToolResult(ok=False, error_code="NOT_FOUND", error_msg=f"Entry '{eid}' not found", is_write=True)
                if int(row["voided"]) == 1:
                    return ToolResult(ok=False, error_code="ALREADY_VOIDED", error_msg=f"Entry '{eid}' already voided", is_write=True)
                if int(row["is_duplicate"]) == 0:
                    self.precondition_violations += 1
                    return ToolResult(ok=False, error_code="NOT_A_DUPLICATE", error_msg=f"Entry '{eid}' is a valid non-duplicate entry and cannot be voided", is_write=True)

                c.execute("UPDATE ledger_entries SET voided = 1 WHERE entry_id = ?", (eid,))
                c.execute("UPDATE accounts SET balance_cents = balance_cents - ? WHERE acct_id = ?", (int(row["amount_cents"]), row["acct_id"]))
                c.execute("UPDATE reconciliation_tickets SET audit_passed = 0 WHERE ticket_id = ?", (row["ticket_id"],))
                c.commit()
                return ToolResult(
                    ok=True,
                    data={"voided_entry_id": eid, "reversed_cents": int(row["amount_cents"])},
                    state_changed=True,
                    is_write=True,
                )

            if name == "post_adjustment":
                tid = str(args.get("ticket_id", ""))
                aid = str(args.get("acct_id", ""))
                inv_id = str(args.get("inv_id", ""))
                delta = int(args.get("delta_cents", 0))
                if delta == 0:
                    return ToolResult(ok=False, error_code="INVALID_ARG", error_msg="delta_cents cannot be 0", is_write=True)
                trow = c.execute("SELECT * FROM reconciliation_tickets WHERE ticket_id = ?", (tid,)).fetchone()
                if not trow:
                    return ToolResult(ok=False, error_code="NOT_FOUND", error_msg=f"Ticket '{tid}' not found", is_write=True)
                if int(trow["adj_count"]) >= int(trow["max_adjustments"]):
                    self.precondition_violations += 1
                    return ToolResult(
                        ok=False,
                        error_code="MAX_ADJUSTMENTS_EXCEEDED",
                        error_msg=f"Ticket '{tid}' reached max_adjustments={trow['max_adjustments']}",
                        is_write=True,
                    )
                inv = c.execute("SELECT * FROM invoices WHERE inv_id = ? AND ticket_id = ?", (inv_id, tid)).fetchone()
                if not inv:
                    return ToolResult(ok=False, error_code="NOT_FOUND", error_msg=f"Invoice '{inv_id}' not in ticket '{tid}'", is_write=True)
                acct = c.execute("SELECT * FROM accounts WHERE acct_id = ?", (aid,)).fetchone()
                if not acct:
                    return ToolResult(ok=False, error_code="NOT_FOUND", error_msg=f"Account '{aid}' not found", is_write=True)
                if int(acct["frozen"]) == 1:
                    self.precondition_violations += 1
                    return ToolResult(ok=False, error_code="ACCOUNT_FROZEN", error_msg=f"Account '{aid}' is frozen", is_write=True)
                new_bal = int(acct["balance_cents"]) + delta
                if new_bal < int(acct["min_reserve_cents"]):
                    self.precondition_violations += 1
                    return ToolResult(
                        ok=False,
                        error_code="BELOW_MIN_RESERVE",
                        error_msg=f"Account '{aid}' post-adjustment balance {new_bal} < min_reserve_cents {acct['min_reserve_cents']}",
                        is_write=True,
                    )

                new_id = f"ADJ-{uuid.uuid4().hex[:6].upper()}"
                c.execute(
                    "INSERT INTO ledger_entries VALUES (?, ?, ?, ?, ?, ?, '2026-11-20', 'adjustment', 0, 0)",
                    (new_id, tid, aid, inv_id, delta, f"ADJ-{new_id}"),
                )
                c.execute("UPDATE accounts SET balance_cents = ? WHERE acct_id = ?", (new_bal, aid))
                c.execute(
                    "UPDATE reconciliation_tickets SET audit_passed = 0, adj_count = adj_count + 1 WHERE ticket_id = ?",
                    (tid,),
                )
                c.commit()
                return ToolResult(
                    ok=True,
                    data={"entry_id": new_id, "inv_id": inv_id, "acct_id": aid, "delta_cents": delta},
                    state_changed=True,
                    is_write=True,
                )

            if name == "reconcile_invoice":
                inv_id = str(args.get("inv_id", ""))
                inv = c.execute("SELECT * FROM invoices WHERE inv_id = ?", (inv_id,)).fetchone()
                if not inv:
                    return ToolResult(ok=False, error_code="NOT_FOUND", error_msg=f"Invoice '{inv_id}' not found", is_write=True)
                if inv["status"] == "RECONCILED":
                    return ToolResult(ok=False, error_code="ALREADY_RECONCILED", error_msg=f"Invoice '{inv_id}' already reconciled", is_write=True)
                trow = c.execute("SELECT audit_passed FROM reconciliation_tickets WHERE ticket_id = ?", (inv["ticket_id"],)).fetchone()
                if not trow or int(trow["audit_passed"]) == 0:
                    self.precondition_violations += 1
                    return ToolResult(
                        ok=False,
                        error_code="AUDIT_REQUIRED",
                        error_msg=f"Ticket '{inv['ticket_id']}' must pass run_audit_check before reconciling invoices",
                        is_write=True,
                    )
                c.execute("UPDATE invoices SET status = 'RECONCILED' WHERE inv_id = ?", (inv_id,))
                c.commit()
                return ToolResult(ok=True, data={"inv_id": inv_id, "status": "RECONCILED"}, state_changed=True, is_write=True)

            if name == "close_ticket":
                tid = str(args.get("ticket_id", ""))
                trow = c.execute("SELECT * FROM reconciliation_tickets WHERE ticket_id = ?", (tid,)).fetchone()
                if not trow:
                    return ToolResult(ok=False, error_code="NOT_FOUND", error_msg=f"Ticket '{tid}' not found", is_write=True)
                if trow["status"] == "CLOSED":
                    return ToolResult(ok=False, error_code="ALREADY_CLOSED", error_msg=f"Ticket '{tid}' already closed", is_write=True)
                if int(trow["audit_passed"]) == 0:
                    self.precondition_violations += 1
                    return ToolResult(ok=False, error_code="AUDIT_REQUIRED", error_msg=f"Ticket '{tid}' audit_passed is 0", is_write=True)
                unrec = [
                    r["inv_id"]
                    for r in c.execute(
                        "SELECT inv_id FROM invoices WHERE ticket_id = ? AND status != 'RECONCILED'",
                        (tid,),
                    ).fetchall()
                ]
                if unrec:
                    self.precondition_violations += 1
                    return ToolResult(
                        ok=False,
                        error_code="INVOICES_NOT_RECONCILED",
                        error_msg=f"Cannot close ticket '{tid}'; unreconciled invoices: {unrec}",
                        is_write=True,
                    )
                c.execute("UPDATE reconciliation_tickets SET status = 'CLOSED' WHERE ticket_id = ?", (tid,))
                c.commit()
                return ToolResult(ok=True, data={"ticket_id": tid, "status": "CLOSED"}, state_changed=True, is_write=True)

        except (ValueError, TypeError) as exc:
            return ToolResult(ok=False, error_code="INVALID_ARG", error_msg=str(exc), is_write=self.is_write_tool(name))

        return ToolResult(ok=False, error_code="UNHANDLED", error_msg=name)

    def snapshot_state(self) -> dict[str, Any]:
        assert self.conn is not None
        tickets = {
            r["ticket_id"]: {
                "status": r["status"],
                "audit_passed": int(r["audit_passed"]),
                "adj_count": int(r["adj_count"]),
                "max_adjustments": int(r["max_adjustments"]),
            }
            for r in self.conn.execute("SELECT ticket_id, status, audit_passed, adj_count, max_adjustments FROM reconciliation_tickets").fetchall()
        }
        invs = {r["inv_id"]: r["status"] for r in self.conn.execute("SELECT inv_id, status FROM invoices").fetchall()}
        return {"tickets": tickets, "invoices": invs}

    def verify(self, task_spec: dict[str, Any]) -> VerifyResult:
        assert self.conn is not None
        violations: list[str] = []
        for tid in task_spec["assertions"]["closed_tickets"]:
            trow = self.conn.execute("SELECT status, audit_passed FROM reconciliation_tickets WHERE ticket_id = ?", (tid,)).fetchone()
            if not trow or trow["status"] != "CLOSED":
                violations.append(f"Ticket {tid} is not CLOSED")
            if trow and int(trow["audit_passed"]) != 1:
                violations.append(f"Ticket {tid} audit_passed != 1")
            active_dups = int(
                self.conn.execute(
                    "SELECT COUNT(*) FROM ledger_entries WHERE ticket_id = ? AND is_duplicate = 1 AND voided = 0",
                    (tid,),
                ).fetchone()[0]
            )
            if active_dups > 0:
                violations.append(f"Ticket {tid} still has {active_dups} unvoided duplicate entries")
            for inv in self.conn.execute("SELECT * FROM invoices WHERE ticket_id = ?", (tid,)).fetchall():
                if inv["status"] != "RECONCILED":
                    violations.append(f"Invoice {inv['inv_id']} is not RECONCILED")
                actual = int(
                    self.conn.execute(
                        "SELECT COALESCE(SUM(amount_cents), 0) FROM ledger_entries WHERE inv_id = ? AND voided = 0",
                        (inv["inv_id"],),
                    ).fetchone()[0]
                )
                if actual != int(inv["expected_cents"]):
                    violations.append(f"Invoice {inv['inv_id']} balance mismatch: {actual} != {inv['expected_cents']}")

        return VerifyResult(
            passed=len(violations) == 0,
            violations=violations,
            checks_summary={
                "closed_tickets_expected": len(task_spec["assertions"]["closed_tickets"]),
                "precondition_violations": self.precondition_violations,
            },
        )
