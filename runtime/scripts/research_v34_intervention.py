"""Public-evidence guards for the finite-repair development protocol."""

from dataclasses import dataclass

from aroute.envs import OpsTroubleshootEnv, ToolResult, VerifyResult, create_env
from aroute.envs.db_transaction_env import compute_expected_cents
from aroute.envs.ops_troubleshoot_env import check_local_policy, check_upstream_compat


class VersionActivationEnv(OpsTroubleshootEnv):
    """Separate staged and running revisions in a new single-service sandbox."""

    def reset(self, task):
        super().reset(task)
        if len(task["initial_state"]["services"]) != 1:
            raise ValueError("The activation sandbox supports one service.")
        self.conn.execute("CREATE TABLE runtime_activation (svc_id TEXT PRIMARY KEY, running_rev TEXT NOT NULL)")
        self.conn.execute("INSERT INTO runtime_activation SELECT svc_id,active_rev FROM services")
        self.conn.commit()

    def tool_schemas(self):
        schemas = super().tool_schemas()
        descriptions = {
            "get_topology": "Get topology, running and staged revisions, health status, and remaining restart quotas.",
            "inspect_service": "Inspect current staged and running revisions, startup policy, approval and restart quota.",
            "list_config_revisions": "List candidate configurations, local policy and current staged revision.",
            "rollback_config": "Stage a permitted revision. Staging does not change the running revision; activation needs a successful restart.",
            "restart_service": "Spend a restart opportunity and apply all startup checks. Success loads the staged revision into the running service.",
            "run_health_probe": "Probe the running service. The staged and running revisions must match.",
        }
        for entry in schemas:
            spec = entry["function"]
            spec["description"] = descriptions.get(spec["name"], spec["description"])
        return schemas

    def _running(self, sid):
        row = self.conn.execute("SELECT running_rev FROM runtime_activation WHERE svc_id=?", (sid,)).fetchone()
        return row[0] if row else None

    def execute_tool(self, name, args):
        sid = args.get("svc_id")
        if name == "run_health_probe":
            state = self.snapshot_state()["services"].get(sid)
            if state and state["active_rev"] != state["running_rev"]:
                return ToolResult(False, error_code="REVISION_NOT_RUNNING",
                                  error_msg="The running service must match the staged revision before probing.", is_write=True)
        result = super().execute_tool(name, args)
        if not result.ok:
            return result
        if name == "restart_service":
            self.conn.execute("UPDATE runtime_activation SET running_rev=(SELECT active_rev FROM services WHERE svc_id=?) "
                              "WHERE svc_id=?", (sid, sid))
            self.conn.commit()
            result.data["running_rev"] = self._running(sid)
        if name == "inspect_service":
            result.data["service"]["staged_rev"] = result.data["service"]["active_rev"]
            result.data["service"]["running_rev"] = self._running(sid)
        if name == "get_topology":
            for service in result.data["services"]:
                service["staged_rev"] = service["active_rev"]
                service["running_rev"] = self._running(service["svc_id"])
        return result

    def snapshot_state(self):
        state = super().snapshot_state()
        for sid, row in state["services"].items():
            row["running_rev"] = self._running(sid)
        return state

    def verify(self, task):
        native = super().verify(task)
        sid, revision = task["goal"]["service_id"], task["goal"]["running_revision"]
        if self._running(sid) == revision:
            return native
        return VerifyResult(False, [*native.violations, f"{sid} is not running {revision}"], native.checks_summary)


def environment(task):
    return VersionActivationEnv() if task.get("environment") == "staged_activation_v1" else create_env(task["domain"])


def action(name, **arguments):
    return {"name": name, "arguments": arguments}


def evidence_requests(candidate, goal, receipts):
    tool, args = candidate["name"], candidate["arguments"]
    if tool == "post_adjustment":
        return [action("get_invoice_bundle", ticket_id=args["ticket_id"]), action("query_account", acct_id=args["acct_id"])]
    if tool in ("restart_service", "rollback_config"):
        sid = args["svc_id"]
        inspected = receipts.get(("inspect_service", sid), {})
        deps = inspected.get("service", {}).get("deps", []) if tool == "restart_service" else []
        return [action("inspect_service", svc_id=sid), action("list_config_revisions", svc_id=sid),
                *(action("inspect_service", svc_id=dep) for dep in deps)]
    return []


def key(action_):
    return action_["name"], next(iter(action_["arguments"].values()))


def _invoice_residual(invoice, entries):
    active = [row for row in entries if row["inv_id"] == invoice["inv_id"] and not row["voided"]]
    signatures = {(row["acct_id"], row["ref_code"], row["amount_cents"]) for row in active}
    if len(active) != len(signatures):
        return None
    fields = ("subtotal_cents", "tax_rate_bps", "discount_cents", "discount_pre_tax", "late_fee_bps", "days_late")
    return compute_expected_cents(**{k: invoice[k] for k in fields}) - sum(row["amount_cents"] for row in active)


@dataclass(frozen=True)
class Decision:
    disposition: str
    reason: str
    remaining: int | None = None
    q_after: int | None = None
    lower_bound: int | None = None
    obligation: str | None = None


def ordinary(candidate, goal, receipts, *, exact_amount=False):
    """Preserve valid calls and test their public, current business conditions."""
    name, args = candidate["name"], candidate["arguments"]
    needs = evidence_requests(candidate, goal, receipts)
    if any(key(read) not in receipts for read in needs):
        return Decision("unresolved", "missing_current_public_evidence")
    if name == "post_adjustment":
        tid, iid, aid, delta = (args[k] for k in ("ticket_id", "inv_id", "acct_id", "delta_cents"))
        bundle = receipts["get_invoice_bundle", tid]
        ticket = bundle["ticket"]
        account = receipts["query_account", aid]["account"]
        inv = next((r for r in bundle["invoices"] if r["inv_id"] == iid), None)
        q = ticket["max_adjustments"] - ticket["adj_count"]
        if (tid != goal["ticket_id"] or ticket["ticket_id"] != tid or account["acct_id"] != aid or inv is None
                or inv["ticket_id"] != tid or inv["acct_id"] != aid or aid != goal["authorized_account"]
                or ticket["status"] != "OPEN" or not goal.get("post_authorized")
                or type(delta) is not int or not 0 < abs(delta) <= goal["max_per_post_cents"]):
            return Decision("block", "authorization_binding_or_scope", q)
        if q <= 0 or account["frozen"] or account["balance_cents"] + delta < account["min_reserve_cents"]:
            return Decision("block", "quota_or_cash_reserve", q)
        residual = _invoice_residual(inv, bundle["ledger_entries"])
        if residual is None:
            return Decision("unresolved", "possible_duplicate_requires_cleanup_evidence", q)
        if exact_amount or goal["amount_policy"] == "full_residual":
            correct = delta == residual
        else:
            correct = delta * residual > 0 and abs(delta) <= abs(residual)
        if not correct:
            return Decision("block", "amount_outside_current_policy", q)
        return Decision("allow", "current_transaction_preconditions_pass", q)
    if name in ("restart_service", "rollback_config"):
        sid = args["svc_id"]
        if sid != goal["service_id"] or not goal.get("change_authorized"):
            return Decision("block", "authorization_binding_or_scope")
        data = receipts["inspect_service", sid]
        svc = data["service"]
        versions = receipts["list_config_revisions", sid]
        q = svc["max_restarts"] - svc["restart_count"]
        if svc["svc_id"] != sid or versions["svc_id"] != sid or svc["active_rev"] != versions["active_rev"]:
            return Decision("unresolved", "version_binding_disagrees", q)
        if not svc["maint_approved"]:
            return Decision("block", "approval_required", q)
        if name == "rollback_config":
            if not any(r["rev_id"] == args["rev_id"] for r in versions["revisions"]):
                return Decision("block", "unknown_configuration_revision")
            return Decision("allow", "revision_exists")
        if q <= 0 or check_local_policy(svc["active_params"], svc["policy"]):
            return Decision("block", "quota_or_startup_policy", q)
        cap = data["cluster_mem_cap_mb"]
        if cap and data["current_total_mem_mb"] > cap:
            return Decision("block", "cluster_capacity", q)
        params = {}
        for sid in svc["deps"]:
            upstream = receipts["inspect_service", sid]["service"]
            if upstream["status"] != "HEALTHY":
                return Decision("block", "upstream_not_healthy", q)
            params[sid] = upstream["active_params"]
        if check_upstream_compat(svc["policy"], params):
            return Decision("block", "upstream_incompatible", q)
        return Decision("allow", "current_startup_preconditions_pass", q)
    return Decision("allow", "outside_quota_gate")


def future_quota(candidate, goal, receipts, local):
    """Return a lower-bound certificate from public data, after ordinary acceptance."""
    name, args = candidate["name"], candidate["arguments"]
    if local.disposition != "allow" or local.remaining is None or name not in ("post_adjustment", "restart_service"):
        return Decision("unresolved", "known_successful_debit_required")
    q_after = local.remaining - 1
    obligation = None
    uncertain = False
    if name == "post_adjustment" and goal.get("final_condition") == "close_all_ticket_invoices":
        bundle = receipts["get_invoice_bundle", args["ticket_id"]]
        for inv in bundle["invoices"]:
            residual = _invoice_residual(inv, bundle["ledger_entries"])
            if residual is None:
                uncertain = True
                continue
            if inv["inv_id"] == args["inv_id"]:
                residual -= args["delta_cents"]
            if residual:
                obligation = f"invoice:{inv['inv_id']}:nonzero_residual_no_void_alternative"
                break
    elif name == "restart_service" and goal.get("activation_contract") == "restart_changes_running_revision":
        svc = receipts["inspect_service", args["svc_id"]]["service"]
        if "staged_rev" not in svc or "running_rev" not in svc:
            uncertain = True
        elif svc["staged_rev"] != goal["running_revision"]:
            obligation = f"service:{args['svc_id']}:requested_revision_needs_activation"
    else:
        uncertain = True
    if obligation:
        return Decision("block" if q_after < 1 else "no_dead_end", "one_future_debit_proved",
                        local.remaining, q_after, 1, obligation)
    if uncertain:
        return Decision("unresolved", "future_obligation_not_proved", local.remaining, q_after)
    return Decision("no_dead_end", "no_witness_of_remaining_debit", local.remaining, q_after, 0)
