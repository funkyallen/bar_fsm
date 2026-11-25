"""Lossless upsert structured state ledger V3 with observation-derived structural risk scoring."""

from __future__ import annotations

import json
from typing import Any

from aroute.envs.base import ToolResult


class StateLedger:
    """Maintains structured upserted facts, completed writes, errors, and structural risk state."""

    def __init__(self, *, window_k: int = 6) -> None:
        self.window_k = max(1, int(window_k))
        self.step_idx: int = 0
        self.completed_writes: list[str] = []
        self.open_errors: list[str] = []
        self.error_streak: int = 0
        self.repeat_count: int = 0
        self.last_action_sig: str | None = None
        self.prechecks_verified: bool = False
        self.pending_irreversible_write: bool = False
        self.recent_turns: list[dict[str, str]] = []
        self.raw_messages: list[dict[str, Any]] = []

        # Structured domain caches derived strictly from observed tool results
        self.known_order_qty: dict[str, int] = {}
        self.order_allowed_whs: dict[str, list[str]] = {}
        self.frozen_warehouses: set[str] = set()
        self.inactive_suppliers: set[str] = set()
        self.supplier_rules: dict[str, dict[str, Any]] = {}
        self.invoice_to_ticket: dict[str, str] = {}
        self.ticket_invoices: dict[str, list[str]] = {}
        self.entry_to_acct: dict[str, str] = {}
        self.frozen_accounts: set[str] = set()
        self.account_meta: dict[str, dict[str, Any]] = {}
        self.service_deps: dict[str, list[str]] = {}
        self.service_policies: dict[str, dict[str, Any]] = {}
        self.cluster_mem_cap_mb: int | None = None
        self.fact_map: dict[str, str] = {}

        # Purely observation-derived structural risk & control-flow counters
        self.pending_order_skus: set[str] = set()
        self.order_skus: dict[str, str] = {}
        self.inventory_by_sku: dict[str, list[dict[str, Any]]] = {}
        self.supplier_offers_by_sku: dict[str, list[dict[str, Any]]] = {}
        self.observed_supplier_skus: set[str] = set()
        self.has_fulfill_costs: bool = False
        self.has_cancelled_locks: bool = False
        self.po_writes: int = 0
        self.observed_orders: dict[str, dict[str, Any]] = {}
        self.observed_tickets: dict[str, dict[str, Any]] = {}
        self.observed_services: dict[str, dict[str, Any]] = {}
        self.stale_inventory_skus: set[str] = set()

        self.observed_invoice_bundles: set[str] = set()
        self.late_fee_invoices: set[str] = set()
        self.adjustment_writes: int = 0

        self.observed_rev_svcs: set[str] = set()
        self.rollback_writes: int = 0

        self.premature_stop_count: int = 0
        self.just_rejected_stop: bool = False
        self.proactive_think_steps: int = 0

        # Stage 5C: Sticky-until-write hold and episode crisis tracking
        self.escalation_hold_steps: int = 0
        self.escalation_hold_tier: str | None = None
        self.crisis_episodes: int = 0
        self._last_hold_step_idx: int = -1

        # Stage 5D: Track last tokenized prompt length for cache-hysteresis switching
        self.last_prompt_tokens: int = 0

        # Boundary-aware prefix-preserving hybrid segment buffers
        self.segment_messages: list[dict[str, Any]] = []
        self.segment_tier: str | None = None
        self.segment_turn_count: int = 0

    @property
    def key_facts(self) -> list[str]:
        return list(self.fact_map.values())

    def init_raw_conversation(self, system_prompt: str, task_prompt: str) -> None:
        self.raw_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_prompt},
        ]
        self.segment_messages = []
        self.segment_tier = None
        self.segment_turn_count = 0

    def record_assistant_message(self, assistant_msg: dict[str, Any]) -> None:
        self.raw_messages.append(dict(assistant_msg))
        if self.segment_messages:
            self.segment_messages.append(dict(assistant_msg))
            self.segment_turn_count += 1

    def record_tool_message(self, tool_call_id: str, content: str) -> None:
        tmsg = {"role": "tool", "tool_call_id": tool_call_id, "content": content}
        self.raw_messages.append(dict(tmsg))
        if self.segment_messages:
            self.segment_messages.append(dict(tmsg))

    def record_user_feedback(self, content: str, *, assistant_text: str = "") -> None:
        if assistant_text.strip():
            amsg = {"role": "assistant", "content": assistant_text}
            self.raw_messages.append(dict(amsg))
            if self.segment_messages:
                self.segment_messages.append(dict(amsg))
        umsg = {"role": "user", "content": content}
        self.raw_messages.append(dict(umsg))
        if self.segment_messages:
            self.segment_messages.append(dict(umsg))

    def append_raw_exchange(
        self,
        assistant_msg: dict[str, Any],
        tool_call_id: str,
        tool_output: str,
    ) -> None:
        self.raw_messages.append(assistant_msg)
        self.raw_messages.append(
            {"role": "tool", "tool_call_id": tool_call_id, "content": tool_output}
        )

    def _upsert_fact(self, key: str, line: str) -> None:
        if key in self.fact_map:
            del self.fact_map[key]
        self.fact_map[key] = line
        while len(self.fact_map) > 32:
            oldest = next(iter(self.fact_map))
            del self.fact_map[oldest]

    def _absorb_observation(self, tool_name: str, data: dict[str, Any]) -> None:
        if "orders" in data and isinstance(data["orders"], list):
            parts = []
            for o in data["orders"]:
                oid = str(o["order_id"])
                self.observed_orders[oid] = dict(o)
                self.known_order_qty[oid] = int(o["qty"])
                self.order_skus[oid] = str(o["sku"])
                allowed = list(o.get("allowed_whs", []))
                self.order_allowed_whs[oid] = allowed
                if str(o.get("status", "")) == "PENDING" and o.get("sku"):
                    self.pending_order_skus.add(str(o["sku"]))
                parts.append(f"{oid}({o['sku']}x{o['qty']}:{o['status']},allow={allowed})")
            self._upsert_fact("Orders", "Orders: " + ", ".join(parts))
            self.prechecks_verified = True

        if "order" in data and isinstance(data["order"], dict):
            o = data["order"]
            oid = str(o["order_id"])
            self.observed_orders[oid] = dict(o)
            self.known_order_qty[oid] = int(o["qty"])
            self.order_skus[oid] = str(o["sku"])
            allowed = list(o.get("allowed_whs", []))
            self.order_allowed_whs[oid] = allowed
            if str(o.get("status", "")) == "PENDING" and o.get("sku"):
                self.pending_order_skus.add(str(o["sku"]))
            self._upsert_fact(
                f"Order:{oid}",
                f"Order {oid}: sku={o['sku']}, qty={o['qty']}, status={o['status']}, "
                f"allow={allowed}, reserved={data.get('reserved_total', 0)}, "
                f"rem_budget={data.get('remaining_episode_budget', data.get('max_total_budget', 0))}",
            )

        if "warehouses" in data and isinstance(data["warehouses"], list):
            sku = str(data.get("sku", ""))
            self.inventory_by_sku[sku] = data["warehouses"]
            self.stale_inventory_skus.discard(sku)
            parts = []
            zombies = []
            for w in data["warehouses"]:
                if int(w.get("frozen", 0)) == 1:
                    self.frozen_warehouses.add(str(w["wh_id"]))
                if int(w.get("fulfill_cost_per_unit", 0)) > 0:
                    self.has_fulfill_costs = True
                parts.append(
                    f"{w['wh_id']}(frozen={w['frozen']},avail={w.get('available', 0)},"
                    f"rem_cap={w.get('remaining_capacity', 0)},fulfill={w.get('fulfill_cost_per_unit', 0)})"
                )
                for res in w.get("active_reservations", []):
                    if res.get("order_status") == "CANCELLED":
                        zombies.append(f"{res['res_id']}({res['order_id']},wh={w['wh_id']},qty={res['qty']})")
            self._upsert_fact(f"Inventory:{sku}", f"Inventory[{sku}]: " + ", ".join(parts))
            if zombies:
                self.has_cancelled_locks = True
                self._upsert_fact(f"CancelledLocks:{sku}", f"CancelledLocks[{sku}]: " + ", ".join(zombies))

        if "suppliers" in data and isinstance(data["suppliers"], list):
            sku = str(data.get("sku", ""))
            if sku:
                self.observed_supplier_skus.add(sku)
                self.supplier_offers_by_sku[sku] = data["suppliers"]
            parts = []
            for s in data["suppliers"]:
                sid = str(s["sup_id"])
                if int(s.get("active", 1)) == 0:
                    self.inactive_suppliers.add(sid)
                self.supplier_rules[sid] = {
                    "min_order_qty": int(s.get("min_order_qty", 1)),
                    "allowed_whs": list(s.get("allowed_whs", [])),
                    "fixed_fee": int(s.get("fixed_fee", 0)),
                    "unit_cost": int(s["unit_cost"]),
                }
                parts.append(
                    f"{sid}(cost={s['unit_cost']},fee={s.get('fixed_fee', 0)},moq={s.get('min_order_qty', 1)},"
                    f"whs={s.get('allowed_whs', [])},rem_cred={s.get('remaining_credit', 0)},act={s['active']})"
                )
            self._upsert_fact(
                f"Suppliers:{sku}",
                f"Suppliers[{sku}] rem_budget={data.get('remaining_episode_budget')}: " + ", ".join(parts),
            )

        if "tickets" in data and isinstance(data["tickets"], list):
            self.observed_tickets = {str(t["ticket_id"]): dict(t) for t in data["tickets"]}
            self._upsert_fact(
                "Tickets",
                "Tickets: "
                + ", ".join(
                    f"{t['ticket_id']}(st={t['status']},audit={t['audit_passed']},adj={t.get('adj_count', 0)}/{t.get('max_adjustments', 5)})"
                    for t in data["tickets"]
                ),
            )
            self.prechecks_verified = True

        if "invoices" in data and isinstance(data["invoices"], list) and "ticket" in data:
            t = data["ticket"]
            tid = str(t["ticket_id"])
            self.observed_tickets[tid] = dict(t)
            self.observed_invoice_bundles.add(tid)
            inv_ids = [str(i["inv_id"]) for i in data["invoices"]]
            self.ticket_invoices[tid] = inv_ids
            for iid in inv_ids:
                self.invoice_to_ticket[iid] = tid
            for i in data["invoices"]:
                if int(i.get("days_late", 0)) > 0 and int(i.get("late_fee_bps", 0)) > 0:
                    self.late_fee_invoices.add(str(i["inv_id"]))
            for e in data.get("ledger_entries", []):
                self.entry_to_acct[str(e["entry_id"])] = str(e["acct_id"])
            inv_desc = [
                f"{i['inv_id']}(st={i['status']},acct={i['acct_id']},sub={i['subtotal_cents']},tax={i['tax_rate_bps']},"
                f"disc={i['discount_cents']},pre={i['discount_pre_tax']},late_bps={i.get('late_fee_bps', 0)},days={i.get('days_late', 0)})"
                for i in data["invoices"]
            ]
            ent_desc = [
                f"{e['entry_id']}(inv={e['inv_id']},acct={e.get('acct_id')},amt={e['amount_cents']},ref={e.get('ref_code')},void={e['voided']})"
                for e in data.get("ledger_entries", [])
            ]
            self._upsert_fact(
                f"Invoices:{tid}",
                f"Invoices[{tid} adj={t.get('adj_count', 0)}/{t.get('max_adjustments', 5)}]: " + "; ".join(inv_desc),
            )
            self._upsert_fact(f"Entries:{tid}", f"Entries[{tid}]: " + "; ".join(ent_desc))

        if "account" in data and isinstance(data["account"], dict):
            a = data["account"]
            aid = str(a["acct_id"])
            if int(a.get("frozen", 0)) == 1:
                self.frozen_accounts.add(aid)
            self.account_meta[aid] = {
                "frozen": int(a.get("frozen", 0)),
                "balance_cents": int(a["balance_cents"]),
                "min_reserve_cents": int(a.get("min_reserve_cents", 0)),
                "fallback_acct_id": a.get("fallback_acct_id"),
            }
            self._upsert_fact(
                f"Account:{aid}",
                f"Account[{aid}]: frozen={a['frozen']}, bal={a['balance_cents']}, "
                f"min_res={a.get('min_reserve_cents', 0)}, fallback={a.get('fallback_acct_id')}",
            )

        if tool_name == "post_adjustment" and "acct_id" in data and "delta_cents" in data:
            aid = str(data["acct_id"])
            if aid in self.account_meta:
                meta = self.account_meta[aid]
                meta["balance_cents"] += int(data["delta_cents"])
                self._upsert_fact(
                    f"Account:{aid}",
                    f"Account[{aid}]: frozen={meta['frozen']}, bal={meta['balance_cents']}, "
                    f"min_res={meta['min_reserve_cents']}, fallback={meta['fallback_acct_id']}",
                )

        if tool_name == "void_duplicate_entry" and "voided_entry_id" in data and "reversed_cents" in data:
            aid = self.entry_to_acct.get(str(data["voided_entry_id"]))
            if aid and aid in self.account_meta:
                meta = self.account_meta[aid]
                meta["balance_cents"] -= int(data["reversed_cents"])
                self._upsert_fact(
                    f"Account:{aid}",
                    f"Account[{aid}]: frozen={meta['frozen']}, bal={meta['balance_cents']}, "
                    f"min_res={meta['min_reserve_cents']}, fallback={meta['fallback_acct_id']}",
                )

        if "audit_passed" in data:
            tid = str(data.get("ticket_id", ""))
            self._upsert_fact(
                f"Audit:{tid}",
                f"Audit[{tid}]: passed={data['audit_passed']}, "
                f"dups={data.get('has_active_duplicates')}, unbal={data.get('unbalanced_invoices')}",
            )

        if "services" in data and isinstance(data["services"], list):
            if "cluster_mem_cap_mb" in data:
                self.cluster_mem_cap_mb = int(data["cluster_mem_cap_mb"])
            parts = []
            for s in data["services"]:
                sid = str(s["svc_id"])
                self.observed_services[sid] = dict(s)
                self.service_deps[sid] = list(s.get("deps", []))
                if "policy" in s and isinstance(s["policy"], dict):
                    self.service_policies[sid] = dict(s["policy"])
                parts.append(
                    f"{sid}(st={s['status']},deps={s.get('deps', [])},rev={s['active_rev']},"
                    f"mem={s.get('active_mem_mb', 0)},rst={s.get('restart_count', 0)}/{s.get('max_restarts', 2)},"
                    f"maint={s.get('maint_approved', 0)},probed={s.get('health_probed', 0)})"
                )
            mem_prefix = ""
            if "cluster_mem_cap_mb" in data:
                mem_prefix = f"ClusterMem={data.get('current_total_mem_mb', 0)}/{data['cluster_mem_cap_mb']}MB | "
            self._upsert_fact("Topology", mem_prefix + "Topology: " + "; ".join(parts))
            self.prechecks_verified = True

        if "service" in data and isinstance(data["service"], dict):
            s = data["service"]
            sid = str(s["svc_id"])
            self.observed_services[sid] = dict(s)
            if "cluster_mem_cap_mb" in data:
                self.cluster_mem_cap_mb = int(data["cluster_mem_cap_mb"])
            self.service_deps[sid] = list(s.get("deps", []))
            if "policy" in s and isinstance(s["policy"], dict):
                self.service_policies[sid] = dict(s["policy"])
            self._upsert_fact(
                f"Service:{sid}",
                f"Service[{sid}]: st={s.get('status')}, rev={s.get('active_rev')}, "
                f"active={json.dumps(s.get('active_params', {}), ensure_ascii=False)}, "
                f"pol={json.dumps(s.get('policy', {}), ensure_ascii=False)}, "
                f"deps={s.get('deps', [])}, rst={s.get('restart_count', 0)}/{s.get('max_restarts', 2)}, "
                f"maint={s.get('maint_approved', 0)}, probed={s.get('health_probed', 0)}, "
                f"cluster_mem={data.get('current_total_mem_mb')}/{data.get('cluster_mem_cap_mb')}MB",
            )

        if "revisions" in data and isinstance(data["revisions"], list):
            sid = str(data.get("svc_id", ""))
            if sid:
                self.observed_rev_svcs.add(sid)
            if sid and isinstance(data.get("policy"), dict):
                self.service_policies[sid] = dict(data["policy"])
            rev_parts = [f"{r['rev_id']}:{json.dumps(r.get('params', {}), ensure_ascii=False)}" for r in data["revisions"]]
            self._upsert_fact(
                f"Revs:{sid}",
                f"Revs[{sid}] active={data.get('active_rev')} pol={data.get('policy')}: " + "; ".join(rev_parts),
            )

        if "logs" in data and isinstance(data["logs"], list):
            sid = str(data.get("svc_id", ""))
            errs = [lg["message"] for lg in data["logs"] if lg.get("level") == "ERROR"]
            if errs:
                self._upsert_fact(f"LogErrors:{sid}", f"LogErrors[{sid}]: " + " | ".join(errs))

        if tool_name == "request_maintenance_window" and data.get("svc_id") in self.observed_services:
            self.observed_services[data["svc_id"]]["maint_approved"] = data["maint_approved"]
        if tool_name == "ship_order" and data.get("order_id") in self.observed_orders:
            self.observed_orders[data["order_id"]]["status"] = data["status"]
        if tool_name == "close_ticket" and data.get("ticket_id") in self.observed_tickets:
            self.observed_tickets[data["ticket_id"]]["status"] = data["status"]

    def compute_structural_risk(
        self,
        *,
        domain: str,
        snapshot: dict[str, Any],
        w_quota: float = 0.35,
        w_couple: float = 0.35,
        w_math: float = 0.30,
    ) -> tuple[float, dict[str, bool]]:
        """Score structural complexity from purely observable runtime state (zero hidden labels)."""
        tight_quota = False
        multi_entity_coupling = False
        complex_constraint = False
        needs_planning = False

        if domain == "supply_chain":
            open_pending = [oid for oid, st in snapshot.get("orders", {}).items() if st == "PENDING"]
            tight_quota = len(open_pending) >= 2 and self.has_fulfill_costs and len(self.observed_supplier_skus) >= 1
            multi_entity_coupling = len(self.pending_order_skus) >= 2
            complex_constraint = len(self.observed_supplier_skus) >= 1 and (self.has_cancelled_locks or self.has_fulfill_costs)
            needs_planning = len(open_pending) >= 2 and len(self.observed_supplier_skus) >= 1 and self.po_writes == 0
        elif domain == "db_transaction":
            tkts = snapshot.get("tickets", {})
            open_tkts = [tid for tid, t in tkts.items() if t.get("status") != "CLOSED"]
            zero_adj_open = [
                tid
                for tid in open_tkts
                if int(tkts[tid].get("adj_count", 0)) == 0 and int(tkts[tid].get("max_adjustments", 5)) <= 1
            ]
            tight_quota = len(zero_adj_open) >= 1
            multi_entity_coupling = len(open_tkts) >= 2 and len(self.observed_invoice_bundles) >= 2
            complex_constraint = len(self.late_fee_invoices) >= 1
            needs_planning = (
                len(zero_adj_open) >= 1
                and len(self.observed_invoice_bundles) >= 2
                and self.adjustment_writes < len(open_tkts)
            )
        elif domain == "ops_troubleshoot":
            svcs = snapshot.get("services", {})
            unhealthy = [sid for sid, s in svcs.items() if s.get("status") != "HEALTHY"]
            tight_unhealthy = [
                sid
                for sid in unhealthy
                if int(svcs[sid].get("max_restarts", 2)) <= 1 and int(svcs[sid].get("restarts", 0)) == 0
            ]
            tight_quota = len(tight_unhealthy) >= 1
            multi_entity_coupling = len(svcs) >= 3 and len(unhealthy) >= 2
            complex_constraint = int(snapshot.get("cluster_mem_cap_mb", 0)) > 0
            needs_planning = (
                len(tight_unhealthy) >= 1
                and len(self.observed_rev_svcs) >= 1
                and self.rollback_writes < len(tight_unhealthy)
            )

        flags = {
            "tight_quota": tight_quota,
            "multi_entity_coupling": multi_entity_coupling,
            "complex_constraint": complex_constraint,
            "needs_planning": needs_planning,
        }
        if not needs_planning:
            return 0.0, flags
        score = (w_quota if tight_quota else 0.0) + (w_couple if multi_entity_coupling else 0.0) + (w_math if complex_constraint else 0.0)
        return round(score, 6), flags

    def arm_escalation_hold(self, tier: str, steps: int) -> None:
        """Hold escalated reasoning tier across follow-up reads until a write completes or steps expire."""
        self.escalation_hold_tier = tier
        self.escalation_hold_steps = max(self.escalation_hold_steps, int(steps))

    def update(
        self,
        *,
        step_idx: int,
        tier: str,
        tool_name: str,
        args: dict[str, Any],
        result: ToolResult,
        fsm_blocked: bool,
    ) -> None:
        self.step_idx = step_idx + 1
        sig = f"{tool_name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
        if sig == self.last_action_sig:
            self.repeat_count += 1
        else:
            self.repeat_count = 0
        self.last_action_sig = sig

        if result.ok:
            self.error_streak = 0
            self.open_errors.clear()
            self.just_rejected_stop = False
            self._absorb_observation(tool_name, result.data)
            if tool_name == "create_restock_po":
                self.po_writes += 1
            elif tool_name == "post_adjustment":
                self.adjustment_writes += 1
            elif tool_name == "rollback_config":
                self.rollback_writes += 1
            if tool_name in ("reserve_stock", "release_reservation"):
                # A mutation may change both the free stock and active reservations in a cached query.
                oid = str(result.data.get("order_id", args.get("order_id", "")))
                sku = self.order_skus.get(oid)
                if sku is None:
                    self.stale_inventory_skus.update(self.inventory_by_sku)
                else:
                    self.stale_inventory_skus.add(sku)

            if result.is_write:
                self.escalation_hold_steps = 0
                self.escalation_hold_tier = None
                res_snippet = json.dumps(result.data, ensure_ascii=False)
                if len(res_snippet) > 120:
                    res_snippet = res_snippet[:117] + "..."
                self.completed_writes.append(
                    f"s{step_idx}: {tool_name}({json.dumps(args, ensure_ascii=False)}) -> {res_snippet}"
                )
                self.pending_irreversible_write = False
            else:
                if self.escalation_hold_steps > 0 and step_idx != self._last_hold_step_idx:
                    self.escalation_hold_steps -= 1
                    self._last_hold_step_idx = step_idx
                    if self.escalation_hold_steps == 0:
                        self.escalation_hold_tier = None
                if tool_name in ("query_suppliers", "query_inventory", "get_invoice_bundle", "list_config_revisions"):
                    self.pending_irreversible_write = True
        else:
            self.error_streak += 1
            err_line = f"s{step_idx} {'[FSM]' if fsm_blocked else '[ENV]'} {tool_name}: {result.error_code} - {result.error_msg}"
            self.open_errors = [err_line]

        outcome = "OK" if result.ok else f"ERR:{result.error_code}"
        if result.ok:
            obs_str = json.dumps(result.data, ensure_ascii=False)
            if len(obs_str) > 420:
                obs_str = obs_str[:420] + "..."
        else:
            obs_str = str(result.error_msg or "")

        self.recent_turns.append(
            {
                "step": str(step_idx),
                "tier": tier,
                "action": f"{tool_name}({json.dumps(args, ensure_ascii=False)})",
                "outcome": f"{outcome} -> {obs_str}",
            }
        )
        if len(self.recent_turns) > self.window_k:
            self.recent_turns = self.recent_turns[-self.window_k:]

    def compact_summary(self) -> str:
        return (
            f"writes={len(self.completed_writes)} facts={len(self.fact_map)} "
            f"err_streak={self.error_streak} repeat={self.repeat_count} stops={self.premature_stop_count}"
        )

    def render_handoff_prompt(self, task_prompt: str) -> str:
        sections: list[str] = [f"TASK:\n{task_prompt}"]
        if self.key_facts:
            sections.append("VERIFIED FACTS:\n- " + "\n- ".join(self.key_facts))
        if self.completed_writes:
            sections.append("COMPLETED WRITES:\n- " + "\n- ".join(self.completed_writes[-16:]))
        if self.open_errors:
            sections.append("OPEN ERRORS TO RESOLVE:\n- " + "\n- ".join(self.open_errors))
        sections.append(
            "Continue directly from the verified state above without repeating unchanged reads. "
            "Compute exact constraints carefully and emit the next valid tool call(s)."
        )
        return "\n\n".join(sections)

    def build_messages(
        self,
        *,
        system_prompt: str,
        task_prompt: str,
        active_tier: str,
        use_state_ledger: bool,
        provider: str = "deepseek",
    ) -> list[dict[str, Any]]:
        if not use_state_ledger:
            if active_tier == "fast":
                cleaned: list[dict[str, Any]] = []
                for m in self.raw_messages:
                    if "reasoning_content" in m:
                        copy_m = {k: v for k, v in m.items() if k != "reasoning_content"}
                        cleaned.append(copy_m)
                    else:
                        cleaned.append(dict(m))
                return cleaned
            if provider != "deepseek":
                return [dict(message) for message in self.raw_messages]
            # DeepSeek's raw history requires reasoning_content for tool-call turns.
            filled: list[dict[str, Any]] = []
            for m in self.raw_messages:
                copy_m = dict(m)
                if copy_m.get("role") == "assistant" and copy_m.get("tool_calls") and not copy_m.get("reasoning_content"):
                    copy_m["reasoning_content"] = "Direct tool call executed."
                filled.append(copy_m)
            return filled

        need_handoff = (
            (not self.segment_messages)
            or (self.segment_tier == "fast" and active_tier in ("think", "pro"))
            or (active_tier == "fast" and self.segment_turn_count >= self.window_k)
        )
        if need_handoff:
            has_state = bool(self.fact_map or self.completed_writes or self.open_errors)
            user_content = self.render_handoff_prompt(task_prompt) if has_state else task_prompt
            self.segment_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
            self.segment_turn_count = 0
        elif active_tier == "fast":
            for m in self.segment_messages:
                if "reasoning_content" in m:
                    del m["reasoning_content"]

        self.segment_tier = active_tier
        return [dict(m) for m in self.segment_messages]
