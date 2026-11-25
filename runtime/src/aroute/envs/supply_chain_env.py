"""Deterministic SQLite sandbox V3 for supply-chain CSP with shared supplier credit and fulfillment fees."""

from __future__ import annotations

import json
import uuid
from typing import Any

from .base import BaseSandboxEnv, ToolResult, VerifyResult

SUPPLY_CHAIN_READ_TOOLS = {
    "list_pending_orders",
    "get_order_detail",
    "query_inventory",
    "query_suppliers",
}
SUPPLY_CHAIN_WRITE_TOOLS = {
    "reserve_stock",
    "create_restock_po",
    "release_reservation",
    "ship_order",
}


def _schema(name: str, desc: str, props: dict[str, Any], req: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req},
        },
    }


class SupplyChainEnv(BaseSandboxEnv):
    """V3 supply-chain CSP: shared multi-SKU credit, warehouse fulfill_cost_per_unit, MOQ/fixed_fee, and tight global budget."""

    domain = "supply_chain"

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [
            _schema(
                "list_pending_orders",
                "List all orders with SKU, quantity, allowed_whs constraint (empty means any active warehouse), and status (PENDING/SHIPPED/CANCELLED).",
                {},
                [],
            ),
            _schema(
                "get_order_detail",
                "Get full details, allowed_whs, active stock reservations, and remaining total episode budget (PO + fulfillment fees) for an order.",
                {"order_id": {"type": "string", "description": "Order ID, e.g. ORD-101"}},
                ["order_id"],
            ),
            _schema(
                "query_inventory",
                "Query stock on hand, reserved, available, warehouse capacity/freeze status, fulfill_cost_per_unit, and active reservations for a SKU.",
                {"sku": {"type": "string", "description": "SKU code, e.g. SKU-ALPHA"}},
                ["sku"],
            ),
            _schema(
                "query_suppliers",
                "Query suppliers for a SKU including unit_cost, fixed_fee, min_order_qty, allowed_whs, shared credit remaining across SKUs, active flag, and remaining episode budget.",
                {"sku": {"type": "string", "description": "SKU code"}},
                ["sku"],
            ),
            _schema(
                "reserve_stock",
                "Reserve available stock in an active, order-allowed warehouse for a PENDING order.",
                {
                    "order_id": {"type": "string"},
                    "wh_id": {"type": "string"},
                    "qty": {"type": "integer", "minimum": 1},
                },
                ["order_id", "wh_id", "qty"],
            ),
            _schema(
                "create_restock_po",
                "Create an irreversible supplier purchase order (cost = qty * unit_cost + fixed_fee) honoring MOQ, supplier allowed_whs, warehouse capacity, shared credit, and total budget.",
                {
                    "sup_id": {"type": "string"},
                    "wh_id": {"type": "string"},
                    "sku": {"type": "string"},
                    "qty": {"type": "integer", "minimum": 1},
                },
                ["sup_id", "wh_id", "sku", "qty"],
            ),
            _schema(
                "release_reservation",
                "Release an active stock reservation (including stale reservations from CANCELLED orders) to free inventory.",
                {"res_id": {"type": "string"}},
                ["res_id"],
            ),
            _schema(
                "ship_order",
                "Irreversibly ship a PENDING order once reserved stock equals ordered quantity, incurring sum(qty * warehouse fulfill_cost_per_unit) against total budget.",
                {"order_id": {"type": "string"}},
                ["order_id"],
            ),
        ]

    def is_write_tool(self, name: str) -> bool:
        return name in SUPPLY_CHAIN_WRITE_TOOLS

    def reset(self, task_spec: dict[str, Any]) -> None:
        self.task_spec = task_spec
        self.precondition_violations = 0
        self.write_count = 0
        self.read_count = 0
        conn = self._new_memory_conn()
        conn.executescript(
            """
            CREATE TABLE warehouses (
                wh_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                frozen INTEGER NOT NULL,
                max_capacity INTEGER NOT NULL,
                used_capacity INTEGER NOT NULL,
                fulfill_cost_per_unit INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE inventory (
                wh_id TEXT NOT NULL,
                sku TEXT NOT NULL,
                on_hand INTEGER NOT NULL,
                reserved INTEGER NOT NULL,
                PRIMARY KEY (wh_id, sku)
            );
            CREATE TABLE supplier_credits (
                sup_id TEXT PRIMARY KEY,
                credit_limit INTEGER NOT NULL,
                spent INTEGER NOT NULL,
                active INTEGER NOT NULL
            );
            CREATE TABLE supplier_catalog (
                sup_id TEXT NOT NULL,
                sku TEXT NOT NULL,
                unit_cost INTEGER NOT NULL,
                fixed_fee INTEGER NOT NULL DEFAULT 0,
                min_order_qty INTEGER NOT NULL DEFAULT 1,
                allowed_whs_json TEXT NOT NULL DEFAULT '[]',
                PRIMARY KEY (sup_id, sku)
            );
            CREATE TABLE orders (
                order_id TEXT PRIMARY KEY,
                sku TEXT NOT NULL,
                qty INTEGER NOT NULL,
                target_wh TEXT NOT NULL,
                status TEXT NOT NULL,
                allowed_whs_json TEXT NOT NULL DEFAULT '[]',
                fulfill_cost INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE reservations (
                res_id TEXT PRIMARY KEY,
                order_id TEXT NOT NULL,
                wh_id TEXT NOT NULL,
                sku TEXT NOT NULL,
                qty INTEGER NOT NULL,
                active INTEGER NOT NULL
            );
            CREATE TABLE purchase_orders (
                po_id TEXT PRIMARY KEY,
                sup_id TEXT NOT NULL,
                wh_id TEXT NOT NULL,
                sku TEXT NOT NULL,
                qty INTEGER NOT NULL,
                total_cost INTEGER NOT NULL
            );
            CREATE TABLE audit_log (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                action TEXT NOT NULL,
                detail TEXT NOT NULL
            );
            """
        )
        init = task_spec["initial_state"]
        for w in init["warehouses"]:
            conn.execute(
                "INSERT INTO warehouses VALUES (?, ?, ?, ?, ?, ?)",
                (
                    w["wh_id"],
                    w["name"],
                    int(w["frozen"]),
                    int(w["max_capacity"]),
                    int(w["used_capacity"]),
                    int(w.get("fulfill_cost_per_unit", 0)),
                ),
            )
        for inv in init["inventory"]:
            conn.execute(
                "INSERT INTO inventory VALUES (?, ?, ?, ?)",
                (inv["wh_id"], inv["sku"], int(inv["on_hand"]), int(inv["reserved"])),
            )
        for s in init["suppliers"]:
            conn.execute(
                "INSERT OR IGNORE INTO supplier_credits VALUES (?, ?, ?, ?)",
                (s["sup_id"], int(s["credit_limit"]), int(s["spent"]), int(s["active"])),
            )
            conn.execute(
                "INSERT INTO supplier_catalog VALUES (?, ?, ?, ?, ?, ?)",
                (
                    s["sup_id"],
                    s["sku"],
                    int(s["unit_cost"]),
                    int(s.get("fixed_fee", 0)),
                    int(s.get("min_order_qty", 1)),
                    json.dumps(s.get("allowed_whs", [])),
                ),
            )
        for o in init["orders"]:
            conn.execute(
                "INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, 0)",
                (
                    o["order_id"],
                    o["sku"],
                    int(o["qty"]),
                    o.get("target_wh", ""),
                    o["status"],
                    json.dumps(o.get("allowed_whs", [])),
                ),
            )
        for res in init.get("reservations", []):
            conn.execute(
                "INSERT INTO reservations VALUES (?, ?, ?, ?, ?, ?)",
                (
                    res["res_id"],
                    res["order_id"],
                    res["wh_id"],
                    res["sku"],
                    int(res["qty"]),
                    int(res.get("active", 1)),
                ),
            )
        conn.commit()

    def _spend_totals(self) -> tuple[int, int, int, int]:
        assert self.conn is not None
        po_spend = int(self.conn.execute("SELECT COALESCE(SUM(total_cost), 0) FROM purchase_orders").fetchone()[0])
        fulfill_spend = int(self.conn.execute("SELECT COALESCE(SUM(fulfill_cost), 0) FROM orders WHERE status = 'SHIPPED'").fetchone()[0])
        max_budget = int(self.task_spec["initial_state"].get("max_restock_budget", 0))
        return po_spend, fulfill_spend, po_spend + fulfill_spend, max_budget

    def execute_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
        assert self.conn is not None
        if name in SUPPLY_CHAIN_READ_TOOLS:
            self.read_count += 1
        elif name in SUPPLY_CHAIN_WRITE_TOOLS:
            self.write_count += 1
        else:
            return ToolResult(ok=False, error_code="UNKNOWN_TOOL", error_msg=f"Unknown tool: {name}")

        c = self.conn
        try:
            if name == "list_pending_orders":
                rows = []
                for r in c.execute("SELECT * FROM orders ORDER BY order_id").fetchall():
                    d = dict(r)
                    d["allowed_whs"] = json.loads(d.pop("allowed_whs_json"))
                    rows.append(d)
                return ToolResult(ok=True, data={"orders": rows})

            if name == "get_order_detail":
                oid = str(args.get("order_id", ""))
                row = c.execute("SELECT * FROM orders WHERE order_id = ?", (oid,)).fetchone()
                if not row:
                    return ToolResult(ok=False, error_code="NOT_FOUND", error_msg=f"Order '{oid}' not found")
                od = dict(row)
                od["allowed_whs"] = json.loads(od.pop("allowed_whs_json"))
                res = [
                    dict(r)
                    for r in c.execute(
                        "SELECT res_id, wh_id, sku, qty FROM reservations WHERE order_id = ? AND active = 1",
                        (oid,),
                    ).fetchall()
                ]
                reserved_total = sum(int(r["qty"]) for r in res)
                _, _, total_spend, max_budget = self._spend_totals()
                return ToolResult(
                    ok=True,
                    data={
                        "order": od,
                        "active_reservations": res,
                        "reserved_total": reserved_total,
                        "max_total_budget": max_budget,
                        "remaining_episode_budget": max_budget - total_spend,
                    },
                )

            if name == "query_inventory":
                sku = str(args.get("sku", ""))
                rows = c.execute(
                    """
                    SELECT w.wh_id, w.name, w.frozen, w.max_capacity, w.used_capacity, w.fulfill_cost_per_unit,
                           COALESCE(i.on_hand, 0) AS on_hand, COALESCE(i.reserved, 0) AS reserved
                    FROM warehouses w
                    LEFT JOIN inventory i ON w.wh_id = i.wh_id AND i.sku = ?
                    ORDER BY w.wh_id
                    """,
                    (sku,),
                ).fetchall()
                items = []
                for r in rows:
                    d = dict(r)
                    d["available"] = int(d["on_hand"]) - int(d["reserved"])
                    d["remaining_capacity"] = int(d["max_capacity"]) - int(d["used_capacity"])
                    res_rows = c.execute(
                        """
                        SELECT r.res_id, r.order_id, COALESCE(o.status, 'UNKNOWN') AS order_status, r.qty
                        FROM reservations r
                        LEFT JOIN orders o ON r.order_id = o.order_id
                        WHERE r.wh_id = ? AND r.sku = ? AND r.active = 1
                        ORDER BY r.res_id
                        """,
                        (d["wh_id"], sku),
                    ).fetchall()
                    d["active_reservations"] = [dict(rr) for rr in res_rows]
                    items.append(d)
                return ToolResult(ok=True, data={"sku": sku, "warehouses": items})

            if name == "query_suppliers":
                sku = str(args.get("sku", ""))
                rows = c.execute(
                    """
                    SELECT sc.sup_id, sc.sku, sc.unit_cost, sc.fixed_fee, sc.min_order_qty, sc.allowed_whs_json,
                           cr.credit_limit, cr.spent, cr.active
                    FROM supplier_catalog sc
                    JOIN supplier_credits cr ON sc.sup_id = cr.sup_id
                    WHERE sc.sku = ?
                    ORDER BY sc.unit_cost, sc.sup_id
                    """,
                    (sku,),
                ).fetchall()
                items = []
                for r in rows:
                    d = dict(r)
                    d["allowed_whs"] = json.loads(d.pop("allowed_whs_json"))
                    d["remaining_credit"] = int(d["credit_limit"]) - int(d["spent"])
                    items.append(d)
                _, _, total_spend, max_budget = self._spend_totals()
                return ToolResult(
                    ok=True,
                    data={
                        "sku": sku,
                        "suppliers": items,
                        "remaining_episode_budget": max_budget - total_spend,
                    },
                )

            if name == "reserve_stock":
                oid = str(args.get("order_id", ""))
                wh_id = str(args.get("wh_id", ""))
                qty = int(args.get("qty", 0))
                if qty <= 0:
                    return ToolResult(ok=False, error_code="INVALID_ARG", error_msg="qty must be > 0", is_write=True)
                ord_row = c.execute("SELECT * FROM orders WHERE order_id = ?", (oid,)).fetchone()
                if not ord_row:
                    return ToolResult(ok=False, error_code="NOT_FOUND", error_msg=f"Order '{oid}' not found", is_write=True)
                if ord_row["status"] != "PENDING":
                    self.precondition_violations += 1
                    return ToolResult(ok=False, error_code="ORDER_NOT_PENDING", error_msg=f"Order '{oid}' status is {ord_row['status']}", is_write=True)
                allowed_whs = json.loads(ord_row["allowed_whs_json"])
                if allowed_whs and wh_id not in allowed_whs:
                    self.precondition_violations += 1
                    return ToolResult(
                        ok=False,
                        error_code="WAREHOUSE_NOT_ALLOWED",
                        error_msg=f"Order '{oid}' restricts fulfillment to warehouses {allowed_whs}, got '{wh_id}'",
                        is_write=True,
                    )
                wh_row = c.execute("SELECT * FROM warehouses WHERE wh_id = ?", (wh_id,)).fetchone()
                if not wh_row:
                    return ToolResult(ok=False, error_code="NOT_FOUND", error_msg=f"Warehouse '{wh_id}' not found", is_write=True)
                if int(wh_row["frozen"]) == 1:
                    self.precondition_violations += 1
                    return ToolResult(ok=False, error_code="WAREHOUSE_FROZEN", error_msg=f"Warehouse '{wh_id}' is frozen", is_write=True)

                sku = str(ord_row["sku"])
                inv_row = c.execute("SELECT * FROM inventory WHERE wh_id = ? AND sku = ?", (wh_id, sku)).fetchone()
                avail = (int(inv_row["on_hand"]) - int(inv_row["reserved"])) if inv_row else 0
                if avail < qty:
                    self.precondition_violations += 1
                    return ToolResult(
                        ok=False,
                        error_code="INSUFFICIENT_STOCK",
                        error_msg=f"Warehouse '{wh_id}' only has {avail} available for {sku}, need {qty}",
                        is_write=True,
                    )
                already = int(
                    c.execute(
                        "SELECT COALESCE(SUM(qty), 0) FROM reservations WHERE order_id = ? AND active = 1",
                        (oid,),
                    ).fetchone()[0]
                )
                if already + qty > int(ord_row["qty"]):
                    self.precondition_violations += 1
                    return ToolResult(
                        ok=False,
                        error_code="OVER_RESERVATION",
                        error_msg=f"Order '{oid}' requires {ord_row['qty']} total, already reserved {already}",
                        is_write=True,
                    )

                res_id = f"RES-{uuid.uuid4().hex[:6].upper()}"
                c.execute("UPDATE inventory SET reserved = reserved + ? WHERE wh_id = ? AND sku = ?", (qty, wh_id, sku))
                c.execute("INSERT INTO reservations VALUES (?, ?, ?, ?, ?, 1)", (res_id, oid, wh_id, sku, qty))
                c.execute("INSERT INTO audit_log(action, detail) VALUES ('reserve_stock', ?)", (json.dumps({"res_id": res_id, "order_id": oid, "wh_id": wh_id, "qty": qty}),))
                c.commit()
                return ToolResult(
                    ok=True,
                    data={"res_id": res_id, "order_id": oid, "wh_id": wh_id, "sku": sku, "qty": qty},
                    state_changed=True,
                    is_write=True,
                )

            if name == "create_restock_po":
                sup_id = str(args.get("sup_id", ""))
                wh_id = str(args.get("wh_id", ""))
                sku = str(args.get("sku", ""))
                qty = int(args.get("qty", 0))
                if qty <= 0:
                    return ToolResult(ok=False, error_code="INVALID_ARG", error_msg="qty must be > 0", is_write=True)
                cr = c.execute("SELECT * FROM supplier_credits WHERE sup_id = ?", (sup_id,)).fetchone()
                if not cr:
                    return ToolResult(ok=False, error_code="NOT_FOUND", error_msg=f"Supplier '{sup_id}' not found", is_write=True)
                if int(cr["active"]) == 0:
                    self.precondition_violations += 1
                    return ToolResult(ok=False, error_code="SUPPLIER_INACTIVE", error_msg=f"Supplier '{sup_id}' is inactive", is_write=True)
                cat = c.execute(
                    "SELECT * FROM supplier_catalog WHERE sup_id = ? AND sku = ?",
                    (sup_id, sku),
                ).fetchone()
                if not cat:
                    self.precondition_violations += 1
                    return ToolResult(ok=False, error_code="SKU_MISMATCH", error_msg=f"Supplier '{sup_id}' does not supply {sku}", is_write=True)
                if qty < int(cat["min_order_qty"]):
                    self.precondition_violations += 1
                    return ToolResult(
                        ok=False,
                        error_code="BELOW_MOQ",
                        error_msg=f"Supplier '{sup_id}' requires min_order_qty={cat['min_order_qty']} for {sku}, got {qty}",
                        is_write=True,
                    )
                sup_whs = json.loads(cat["allowed_whs_json"])
                if sup_whs and wh_id not in sup_whs:
                    self.precondition_violations += 1
                    return ToolResult(
                        ok=False,
                        error_code="SUPPLIER_WH_NOT_ALLOWED",
                        error_msg=f"Supplier '{sup_id}' only delivers {sku} to {sup_whs}, got '{wh_id}'",
                        is_write=True,
                    )
                wh = c.execute("SELECT * FROM warehouses WHERE wh_id = ?", (wh_id,)).fetchone()
                if not wh:
                    return ToolResult(ok=False, error_code="NOT_FOUND", error_msg=f"Warehouse '{wh_id}' not found", is_write=True)
                if int(wh["frozen"]) == 1:
                    self.precondition_violations += 1
                    return ToolResult(ok=False, error_code="WAREHOUSE_FROZEN", error_msg=f"Warehouse '{wh_id}' is frozen", is_write=True)
                if int(wh["used_capacity"]) + qty > int(wh["max_capacity"]):
                    self.precondition_violations += 1
                    rem_cap = int(wh["max_capacity"]) - int(wh["used_capacity"])
                    return ToolResult(ok=False, error_code="CAPACITY_EXCEEDED", error_msg=f"Warehouse '{wh_id}' remaining capacity is {rem_cap}", is_write=True)

                total_cost = qty * int(cat["unit_cost"]) + int(cat["fixed_fee"])
                if int(cr["spent"]) + total_cost > int(cr["credit_limit"]):
                    self.precondition_violations += 1
                    rem_cred = int(cr["credit_limit"]) - int(cr["spent"])
                    return ToolResult(ok=False, error_code="CREDIT_EXCEEDED", error_msg=f"Supplier '{sup_id}' shared remaining credit is {rem_cred}, cost={total_cost}", is_write=True)

                _, _, current_total_spend, max_budget = self._spend_totals()
                if current_total_spend + total_cost > max_budget:
                    self.precondition_violations += 1
                    return ToolResult(
                        ok=False,
                        error_code="BUDGET_EXCEEDED",
                        error_msg=f"Episode budget remaining={max_budget - current_total_spend}, PO cost={total_cost}",
                        is_write=True,
                    )

                po_id = f"PO-{uuid.uuid4().hex[:6].upper()}"
                c.execute("UPDATE warehouses SET used_capacity = used_capacity + ? WHERE wh_id = ?", (qty, wh_id))
                c.execute("UPDATE supplier_credits SET spent = spent + ? WHERE sup_id = ?", (total_cost, sup_id))
                inv = c.execute("SELECT * FROM inventory WHERE wh_id = ? AND sku = ?", (wh_id, sku)).fetchone()
                if inv:
                    c.execute("UPDATE inventory SET on_hand = on_hand + ? WHERE wh_id = ? AND sku = ?", (qty, wh_id, sku))
                else:
                    c.execute("INSERT INTO inventory VALUES (?, ?, ?, 0)", (wh_id, sku, qty))
                c.execute("INSERT INTO purchase_orders VALUES (?, ?, ?, ?, ?, ?)", (po_id, sup_id, wh_id, sku, qty, total_cost))
                c.execute("INSERT INTO audit_log(action, detail) VALUES ('create_restock_po', ?)", (json.dumps({"po_id": po_id, "sup_id": sup_id, "wh_id": wh_id, "qty": qty, "total_cost": total_cost}),))
                c.commit()
                return ToolResult(
                    ok=True,
                    data={"po_id": po_id, "sup_id": sup_id, "wh_id": wh_id, "sku": sku, "qty": qty, "total_cost": total_cost},
                    state_changed=True,
                    is_write=True,
                )

            if name == "release_reservation":
                res_id = str(args.get("res_id", ""))
                r = c.execute("SELECT * FROM reservations WHERE res_id = ? AND active = 1", (res_id,)).fetchone()
                if not r:
                    return ToolResult(ok=False, error_code="NOT_FOUND", error_msg=f"Active reservation '{res_id}' not found", is_write=True)
                ord_row = c.execute("SELECT status FROM orders WHERE order_id = ?", (r["order_id"],)).fetchone()
                if ord_row and ord_row["status"] == "SHIPPED":
                    self.precondition_violations += 1
                    return ToolResult(ok=False, error_code="ALREADY_SHIPPED", error_msg="Cannot release reservation of shipped order", is_write=True)
                c.execute("UPDATE reservations SET active = 0 WHERE res_id = ?", (res_id,))
                c.execute("UPDATE inventory SET reserved = MAX(0, reserved - ?) WHERE wh_id = ? AND sku = ?", (int(r["qty"]), r["wh_id"], r["sku"]))
                c.commit()
                return ToolResult(ok=True, data={"released_res_id": res_id, "order_id": r["order_id"], "wh_id": r["wh_id"], "qty": int(r["qty"])}, state_changed=True, is_write=True)

            if name == "ship_order":
                oid = str(args.get("order_id", ""))
                ord_row = c.execute("SELECT * FROM orders WHERE order_id = ?", (oid,)).fetchone()
                if not ord_row:
                    return ToolResult(ok=False, error_code="NOT_FOUND", error_msg=f"Order '{oid}' not found", is_write=True)
                if ord_row["status"] != "PENDING":
                    return ToolResult(ok=False, error_code="ORDER_NOT_PENDING", error_msg=f"Order '{oid}' status is {ord_row['status']}", is_write=True)
                allowed_whs = json.loads(ord_row["allowed_whs_json"])
                res_rows = c.execute(
                    "SELECT r.wh_id, r.qty, w.fulfill_cost_per_unit FROM reservations r JOIN warehouses w ON r.wh_id = w.wh_id WHERE r.order_id = ? AND r.active = 1",
                    (oid,),
                ).fetchall()
                reserved_total = sum(int(r["qty"]) for r in res_rows)
                if reserved_total != int(ord_row["qty"]):
                    self.precondition_violations += 1
                    return ToolResult(
                        ok=False,
                        error_code="UNRESERVED_ORDER",
                        error_msg=f"Order '{oid}' requires {ord_row['qty']} reserved units before shipping, currently {reserved_total}",
                        is_write=True,
                    )
                if allowed_whs:
                    bad_whs = [r["wh_id"] for r in res_rows if r["wh_id"] not in allowed_whs]
                    if bad_whs:
                        self.precondition_violations += 1
                        return ToolResult(
                            ok=False,
                            error_code="WAREHOUSE_NOT_ALLOWED",
                            error_msg=f"Reservations in disallowed warehouses {bad_whs}; order allows {allowed_whs}",
                            is_write=True,
                        )
                fulfill_cost = sum(int(r["qty"]) * int(r["fulfill_cost_per_unit"]) for r in res_rows)
                _, _, total_spend, max_budget = self._spend_totals()
                if total_spend + fulfill_cost > max_budget:
                    self.precondition_violations += 1
                    return ToolResult(
                        ok=False,
                        error_code="BUDGET_EXCEEDED",
                        error_msg=f"Shipping '{oid}' incurs fulfill_cost={fulfill_cost}, exceeding remaining budget {max_budget - total_spend}",
                        is_write=True,
                    )
                c.execute("UPDATE orders SET status = 'SHIPPED', fulfill_cost = ? WHERE order_id = ?", (fulfill_cost, oid))
                c.execute("INSERT INTO audit_log(action, detail) VALUES ('ship_order', ?)", (json.dumps({"order_id": oid, "fulfill_cost": fulfill_cost}),))
                c.commit()
                return ToolResult(ok=True, data={"order_id": oid, "status": "SHIPPED", "fulfill_cost": fulfill_cost}, state_changed=True, is_write=True)

        except (ValueError, TypeError) as exc:
            return ToolResult(ok=False, error_code="INVALID_ARG", error_msg=str(exc), is_write=self.is_write_tool(name))

        return ToolResult(ok=False, error_code="UNHANDLED", error_msg=name)

    def snapshot_state(self) -> dict[str, Any]:
        assert self.conn is not None
        orders = {r["order_id"]: r["status"] for r in self.conn.execute("SELECT order_id, status FROM orders").fetchall()}
        res_sum = {
            r["order_id"]: int(r["qty"])
            for r in self.conn.execute(
                "SELECT order_id, COALESCE(SUM(qty), 0) AS qty FROM reservations WHERE active = 1 GROUP BY order_id"
            ).fetchall()
        }
        po_spend, fulfill_spend, total_spend, _ = self._spend_totals()
        return {
            "orders": orders,
            "reserved": res_sum,
            "po_spend": po_spend,
            "fulfill_spend": fulfill_spend,
            "total_spend": total_spend,
        }

    def verify(self, task_spec: dict[str, Any]) -> VerifyResult:
        assert self.conn is not None
        violations: list[str] = []
        expected_orders = task_spec["assertions"]["shipped_orders"]
        for oid in expected_orders:
            row = self.conn.execute("SELECT * FROM orders WHERE order_id = ?", (oid,)).fetchone()
            if not row or row["status"] != "SHIPPED":
                violations.append(f"Order {oid} is not SHIPPED")
                continue
            allowed_whs = json.loads(row["allowed_whs_json"])
            res_rows = self.conn.execute(
                "SELECT wh_id, qty FROM reservations WHERE order_id = ? AND active = 1",
                (oid,),
            ).fetchall()
            res_qty = sum(int(r["qty"]) for r in res_rows)
            if res_qty != int(row["qty"]):
                violations.append(f"Order {oid} reservation sum mismatch: {res_qty} != {row['qty']}")
            if allowed_whs and any(r["wh_id"] not in allowed_whs for r in res_rows):
                violations.append(f"Order {oid} used disallowed warehouse outside {allowed_whs}")

        po_spend, fulfill_spend, total_spend, max_budget = self._spend_totals()
        if total_spend > max_budget:
            violations.append(f"Total spend {total_spend} (po={po_spend}+fulfill={fulfill_spend}) exceeds budget {max_budget}")

        stale = self.conn.execute(
            """
            SELECT r.res_id, r.order_id
            FROM reservations r
            JOIN orders o ON r.order_id = o.order_id
            WHERE o.status = 'CANCELLED' AND r.active = 1
            """
        ).fetchall()
        if stale:
            violations.append(f"Unreleased cancelled reservations remain: {[r['res_id'] for r in stale]}")

        return VerifyResult(
            passed=len(violations) == 0,
            violations=violations,
            checks_summary={
                "shipped_orders_expected": len(expected_orders),
                "po_spend": po_spend,
                "fulfill_spend": fulfill_spend,
                "total_spend": total_spend,
                "max_budget": max_budget,
                "precondition_violations": self.precondition_violations,
            },
        )
