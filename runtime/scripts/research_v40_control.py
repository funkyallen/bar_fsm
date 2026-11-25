"""Evidence-conditioned repair-budget control for the v4.0 local studies."""

from dataclasses import dataclass

from aroute.envs import OpsTroubleshootEnv
from aroute.envs.ops_troubleshoot_env import check_local_policy, check_upstream_compat

from .research_v34_intervention import VersionActivationEnv, _invoice_residual, action, key
from .research_v37_lookahead_stress import ordinary_generalized


class ActivationEnv(VersionActivationEnv):
    """Multi-service staged activation with upstream checks on the running revision."""

    def reset(self, task):
        OpsTroubleshootEnv.reset(self, task)
        self._loading = None
        self.conn.execute("CREATE TABLE runtime_activation (svc_id TEXT PRIMARY KEY, running_rev TEXT NOT NULL)")
        self.conn.execute("INSERT INTO runtime_activation SELECT svc_id,active_rev FROM services")
        self.conn.commit()

    def _active_params(self, sid):
        if sid == self._loading:
            return OpsTroubleshootEnv._active_params(self, sid)
        row = self.conn.execute("SELECT params_json FROM config_revisions WHERE svc_id=? AND rev_id=?",
                                (sid, self._running(sid))).fetchone()
        if not row:
            return None
        import json
        return json.loads(row[0])

    def execute_tool(self, name, args):
        if name == "restart_service":
            self._loading = args.get("svc_id")
        try:
            result = super().execute_tool(name, args)
        finally:
            self._loading = None
        if name == "inspect_service" and result.ok:
            result.data["service"]["staged_params"] = OpsTroubleshootEnv._active_params(self, args["svc_id"])
        return result

    def verify(self, task):
        from aroute.envs import VerifyResult
        base = OpsTroubleshootEnv.verify(self, task)
        violations = list(base.violations)
        for sid, required in task["goal"]["running_revisions"].items():
            if self._running(sid) != required:
                violations.append(f"{sid}: required running revision {required}")
        return VerifyResult(not violations, violations, base.checks_summary)


@dataclass(frozen=True)
class Assessment:
    local: str
    result: str
    reason: str
    q_after: int | None = None
    lower: int | None = None
    obligations: tuple = ()


def reads(candidate, goal, receipts):
    name, args = candidate["name"], candidate["arguments"]
    if name == "post_adjustment":
        return [action("get_invoice_bundle", ticket_id=args["ticket_id"]), action("query_account", acct_id=args["acct_id"])]
    if name in ("run_audit_check", "close_ticket"):
        return [action("get_invoice_bundle", ticket_id=args["ticket_id"])]
    if name in ("reconcile_invoice", "void_duplicate_entry"):
        return [action("get_invoice_bundle", ticket_id=tid) for tid in goal["authorized_tickets"]]
    if name in ("rollback_config", "restart_service", "run_health_probe", "request_maintenance_window"):
        sid = args["svc_id"]
        observed = receipts.get(("inspect_service", sid), {}).get("service", {})
        basereads = [action("inspect_service", svc_id=sid)]
        if name == "rollback_config":
            basereads.append(action("list_config_revisions", svc_id=sid))
        if name in ("restart_service", "run_health_probe"):
            basereads.extend(action("inspect_service", svc_id=d) for d in observed.get("deps", ()))
        return basereads
    return []


def _ledger_action(candidate, goal, receipts):
    name, args = candidate["name"], candidate["arguments"]
    if name in ("run_audit_check", "close_ticket"):
        if args["ticket_id"] not in goal["authorized_tickets"]:
            return "block", "ticket_outside_authorization"
        bundle = receipts["get_invoice_bundle", args["ticket_id"]]
        if name == "close_ticket" and (not bundle["ticket"]["audit_passed"] or
                                       any(i["status"] != "RECONCILED" for i in bundle["invoices"])):
            return "block", "audit_or_reconciliation_required"
    if name == "reconcile_invoice":
        bundles = (receipts["get_invoice_bundle", t] for t in goal["authorized_tickets"])
        for bundle in bundles:
            invoice = next((i for i in bundle["invoices"] if i["inv_id"] == args["inv_id"]), None)
            if invoice is not None:
                if not bundle["ticket"]["audit_passed"] or invoice["status"] == "RECONCILED":
                    return "block", "audit_required_or_already_reconciled"
                return "pass", "reconciliation_authorized"
        return "block", "invoice_outside_authorization"
    if name == "void_duplicate_entry":
        for tid in goal["authorized_tickets"]:
            entries = [r for r in receipts["get_invoice_bundle", tid]["ledger_entries"] if not r["voided"]]
            selected = next((r for r in entries if r["entry_id"] == args["entry_id"]), None)
            if selected:
                fields = ("inv_id", "acct_id", "ref_code", "amount_cents")
                match = [r for r in entries if all(r[k] == selected[k] for k in fields)]
                return ("pass", "duplicate_group_backend_checks_identity") if len(match) > 1 else ("block", "not_a_duplicate_group")
        return "block", "unknown_entry_or_unauthorized"
    return "pass", "current_operation_checks_pass"


def _service_local(candidate, goal, receipts):
    name, args = candidate["name"], candidate["arguments"]
    sid = args["svc_id"]
    svc = receipts["inspect_service", sid]["service"]
    if sid not in goal["authorized_services"]:
        return "block", "service_outside_authorization"
    if name == "request_maintenance_window":
        return "pass", "maintenance_request_allowed"
    if name == "rollback_config":
        revisions = receipts["list_config_revisions", sid]["revisions"]
        if not svc["maint_approved"] or not any(r["rev_id"] == args["rev_id"] for r in revisions):
            return "block", "maintenance_or_revision_required"
        return "pass", "revision_staging_allowed"
    if name == "restart_service":
        if not svc["maint_approved"] or svc["restart_count"] >= svc["max_restarts"]:
            return "block", "maintenance_or_quota_required"
        params = svc.get("staged_params")
        if params is None:
            return "unknown", "staged_configuration_not_proven"
    else:
        if svc["status"] != "HEALTHY" or svc["running_rev"] != svc["staged_rev"]:
            return "block", "running_service_not_probe_ready"
        params = svc["active_params"]
    upstream = {d: receipts["inspect_service", d]["service"] for d in svc["deps"]}
    if any(r["status"] != "HEALTHY" for r in upstream.values()):
        return "block", "upstream_not_healthy"
    if check_local_policy(params, svc["policy"]) or check_upstream_compat(svc["policy"], {d: r["active_params"] for d, r in upstream.items()}):
        return "block", "configuration_incompatible"
    limit = receipts["inspect_service", sid]
    cap = limit["cluster_mem_cap_mb"]
    return ("block", "cluster_capacity") if cap and limit["current_total_mem_mb"] > cap else ("pass", "startup_and_dependency_conditions_pass")


def assess(candidate, goal, receipts):
    """Use fresh public receipts only. Search and scorer state are unavailable here."""
    name, args = candidate["name"], candidate["arguments"]
    if any(key(request) not in receipts for request in reads(candidate, goal, receipts)):
        return Assessment("unknown", "unknown", "missing_or_stale_required_receipt")
    if name == "post_adjustment":
        ordinary = ordinary_generalized(candidate, goal, receipts)
        if ordinary.disposition != "allow":
            local = "block" if ordinary.disposition == "block" else "unknown"
            return Assessment(local, local, ordinary.reason)
        tid = args["ticket_id"]
        bundle = receipts["get_invoice_bundle", tid]
        left = ordinary.remaining - 1
        lower = 0
        obligations = []
        uncertain = False
        cap = goal["max_per_post_cents"]
        for invoice in bundle["invoices"]:
            residual = _invoice_residual(invoice, bundle["ledger_entries"])
            if residual is None:
                uncertain = True
                continue
            if invoice["inv_id"] == args["inv_id"]:
                residual -= args["delta_cents"]
            if residual:
                lower += (abs(residual) + cap - 1) // cap
                obligations.append((invoice["inv_id"], residual))
        if left < lower:
            return Assessment("pass", "block", "insufficient_future_adjustments", left, lower, tuple(obligations))
        return Assessment("pass", "unknown" if uncertain else "pass_checks",
                          "void_alternative_unresolved" if uncertain else "bound_within_budget", left, lower, tuple(obligations))
    if name in ("rollback_config", "request_maintenance_window", "restart_service", "run_health_probe"):
        local, reason = _service_local(candidate, goal, receipts)
        if local != "pass" or name != "restart_service":
            return Assessment(local, "pass_checks" if local == "pass" else local, reason)
        sid = args["svc_id"]
        svc = receipts["inspect_service", sid]["service"]
        target = goal["running_revisions"].get(sid)
        if target is None:
            return Assessment("pass", "unknown", "no_explicit_running_revision_obligation")
        left = svc["max_restarts"] - svc["restart_count"] - 1
        lower = int(svc["staged_rev"] != target)
        return Assessment("pass", "block" if left < lower else "pass_checks", "required_activation",
                          left, lower, ((sid, target),) if lower else ())
    local, reason = _ledger_action(candidate, goal, receipts)
    return Assessment(local, "pass_checks" if local == "pass" else local, reason)


def feedback(decision, arm):
    if decision.local != "pass":
        return f"The current operation is withheld: {decision.reason}. Use public tools or propose a different operation."
    if arm == "M1":
        return "The proposed operation is blocked because it would leave insufficient repair allowance."
    return ("The proposed operation is blocked. Public repair evidence: "
            f"post-action allowance={decision.q_after}, minimum remaining uses={decision.lower}, "
            f"outstanding obligations={decision.obligations}. Reconsider the prerequisites and shared allowance before proposing an action.")
