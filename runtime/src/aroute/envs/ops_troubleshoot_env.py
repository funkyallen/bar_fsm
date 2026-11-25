"""Deterministic SQLite sandbox V3 for microservice DAG troubleshooting with global memory cap and single-shot restarts."""

from __future__ import annotations

import json
import uuid
from typing import Any

from .base import BaseSandboxEnv, ToolResult, VerifyResult

OPS_READ_TOOLS = {
    "get_topology",
    "inspect_service",
    "read_service_logs",
    "list_config_revisions",
}
OPS_WRITE_TOOLS = {
    "request_maintenance_window",
    "rollback_config",
    "restart_service",
    "run_health_probe",
}


def _ver_tuple(ver: str) -> tuple[int, ...]:
    parts: list[int] = []
    for p in str(ver).strip().split("."):
        digits = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def check_local_policy(params: dict[str, Any], policy: dict[str, Any]) -> str | None:
    """Return violation reason if revision params violate local service policy, else None."""
    if "min_tls" in policy:
        actual = str(params.get("tls", "0"))
        req = str(policy["min_tls"])
        if _ver_tuple(actual) < _ver_tuple(req):
            return f"POLICY_TLS_MISMATCH: tls={actual} < min_tls={req}"

    if "min_pool" in policy or "max_pool" in policy:
        pool = int(params.get("pool", -1))
        lo = int(policy.get("min_pool", 0))
        hi = int(policy.get("max_pool", 10**9))
        if not (lo <= pool <= hi):
            return f"POLICY_POOL_OUT_OF_RANGE: pool={pool} not in [{lo}, {hi}]"

    if "required_auth" in policy:
        if params.get("auth_mode") != policy["required_auth"]:
            return f"POLICY_AUTH_MISMATCH: auth_mode={params.get('auth_mode')} != {policy['required_auth']}"

    if "min_schema" in policy:
        if int(params.get("schema", 0)) < int(policy["min_schema"]):
            return f"POLICY_SCHEMA_TOO_OLD: schema={params.get('schema')} < {policy['min_schema']}"

    return None


def check_upstream_compat(
    svc_policy: dict[str, Any],
    upstream_params: dict[str, dict[str, Any]],
) -> str | None:
    """Return incompatibility reason if any upstream active config breaks downstream requirements."""
    reqs = svc_policy.get("upstream_reqs", {})
    for up_id, req in reqs.items():
        up_p = upstream_params.get(up_id)
        if up_p is None:
            return f"UPSTREAM_INCOMPATIBLE: missing active config for '{up_id}'"
        if "min_schema" in req and int(up_p.get("schema", 0)) < int(req["min_schema"]):
            return (
                f"UPSTREAM_INCOMPATIBLE: upstream '{up_id}' schema={up_p.get('schema')} "
                f"< required {req['min_schema']}"
            )
        if "min_tls" in req and _ver_tuple(str(up_p.get("tls", "0"))) < _ver_tuple(str(req["min_tls"])):
            return (
                f"UPSTREAM_INCOMPATIBLE: upstream '{up_id}' tls={up_p.get('tls')} "
                f"< required {req['min_tls']}"
            )
        if "required_auth" in req and up_p.get("auth_mode") != req["required_auth"]:
            return (
                f"UPSTREAM_INCOMPATIBLE: upstream '{up_id}' auth_mode={up_p.get('auth_mode')} "
                f"!= required {req['required_auth']}"
            )
    return None


def _schema(name: str, desc: str, props: dict[str, Any], req: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req},
        },
    }


class OpsTroubleshootEnv(BaseSandboxEnv):
    """V3 DAG recovery with hidden per-node policy, global cluster_mem_cap_mb, cross-service compat, and max_restarts=1."""

    domain = "ops_troubleshoot"

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [
            _schema(
                "get_topology",
                "Get service dependency topology, statuses, active_rev, mem_mb, maintenance/probe flags, restart quotas, and global cluster_mem_cap_mb (call inspect_service/list_config_revisions for policies).",
                {},
                [],
            ),
            _schema(
                "inspect_service",
                "Inspect a single service's status, upstream dependencies, active_rev, local & upstream policy requirements, and remaining restart quota.",
                {"svc_id": {"type": "string"}},
                ["svc_id"],
            ),
            _schema(
                "read_service_logs",
                "Read diagnostic logs (INFO/WARN/ERROR) for a service.",
                {"svc_id": {"type": "string"}},
                ["svc_id"],
            ),
            _schema(
                "list_config_revisions",
                "List candidate config revisions (params including mem_mb, deployed_at, neutral note), service policy, and cluster_mem_cap_mb.",
                {"svc_id": {"type": "string"}},
                ["svc_id"],
            ),
            _schema(
                "request_maintenance_window",
                "Obtain an approved maintenance change request before rolling back or restarting a service.",
                {"svc_id": {"type": "string"}},
                ["svc_id"],
            ),
            _schema(
                "rollback_config",
                "Roll back a service's active config to an existing revision (requires maintenance approval).",
                {"svc_id": {"type": "string"}, "rev_id": {"type": "string"}},
                ["svc_id", "rev_id"],
            ),
            _schema(
                "restart_service",
                "Restart a service (consumes 1 restart attempt; enforces local policy, HEALTHY upstreams, upstream compatibility, and global cluster_mem_cap_mb).",
                {"svc_id": {"type": "string"}},
                ["svc_id"],
            ),
            _schema(
                "run_health_probe",
                "Run final health verification probe on a service (requires HEALTHY status, compliant local policy, compatible HEALTHY upstreams, and cluster_mem_cap_mb).",
                {"svc_id": {"type": "string"}},
                ["svc_id"],
            ),
        ]

    def is_write_tool(self, name: str) -> bool:
        return name in OPS_WRITE_TOOLS

    def reset(self, task_spec: dict[str, Any]) -> None:
        self.task_spec = task_spec
        self.precondition_violations = 0
        self.write_count = 0
        self.read_count = 0
        conn = self._new_memory_conn()
        conn.executescript(
            """
            CREATE TABLE cluster_meta (
                cluster_mem_cap_mb INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE services (
                svc_id TEXT PRIMARY KEY,
                deps_json TEXT NOT NULL,
                status TEXT NOT NULL,
                active_rev TEXT NOT NULL,
                policy_json TEXT NOT NULL DEFAULT '{}',
                maint_approved INTEGER NOT NULL,
                health_probed INTEGER NOT NULL,
                restart_count INTEGER NOT NULL DEFAULT 0,
                max_restarts INTEGER NOT NULL DEFAULT 2
            );
            CREATE TABLE logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                svc_id TEXT NOT NULL,
                level TEXT NOT NULL,
                message TEXT NOT NULL
            );
            CREATE TABLE config_revisions (
                svc_id TEXT NOT NULL,
                rev_id TEXT NOT NULL,
                params_json TEXT NOT NULL DEFAULT '{}',
                deployed_at TEXT NOT NULL DEFAULT '2026-11-10',
                note TEXT NOT NULL,
                PRIMARY KEY (svc_id, rev_id)
            );
            CREATE TABLE change_requests (
                req_id TEXT PRIMARY KEY,
                svc_id TEXT NOT NULL,
                status TEXT NOT NULL
            );
            """
        )
        init = task_spec["initial_state"]
        cap_mb = int(init.get("cluster_mem_cap_mb", 0))
        conn.execute("INSERT INTO cluster_meta VALUES (?)", (cap_mb,))
        for s in init["services"]:
            conn.execute(
                "INSERT INTO services VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    s["svc_id"],
                    json.dumps(s["deps"]),
                    s["status"],
                    s["active_rev"],
                    json.dumps(s.get("policy", {})),
                    int(s.get("maint_approved", 0)),
                    int(s.get("health_probed", 0)),
                    int(s.get("restart_count", 0)),
                    int(s.get("max_restarts", 2)),
                ),
            )
        for lg in init["logs"]:
            conn.execute(
                "INSERT INTO logs(svc_id, level, message) VALUES (?, ?, ?)",
                (lg["svc_id"], lg["level"], lg["message"]),
            )
        for rev in init["config_revisions"]:
            conn.execute(
                "INSERT INTO config_revisions VALUES (?, ?, ?, ?, ?)",
                (
                    rev["svc_id"],
                    rev["rev_id"],
                    json.dumps(rev.get("params", {})),
                    rev.get("deployed_at", "2026-11-10"),
                    rev.get("note", ""),
                ),
            )
        conn.commit()

    def _cluster_mem_info(self) -> tuple[int, int]:
        assert self.conn is not None
        cap_mb = int(self.conn.execute("SELECT cluster_mem_cap_mb FROM cluster_meta").fetchone()[0])
        total_mb = 0
        for srow in self.conn.execute("SELECT svc_id, active_rev FROM services").fetchall():
            r = self.conn.execute(
                "SELECT params_json FROM config_revisions WHERE svc_id = ? AND rev_id = ?",
                (srow["svc_id"], srow["active_rev"]),
            ).fetchone()
            if r:
                total_mb += int(json.loads(r["params_json"]).get("mem_mb", 0))
        return total_mb, cap_mb

    def _upstream_unhealthy(self, deps: list[str]) -> list[str]:
        assert self.conn is not None
        bad: list[str] = []
        for dep in deps:
            row = self.conn.execute("SELECT status FROM services WHERE svc_id = ?", (dep,)).fetchone()
            if not row or row["status"] != "HEALTHY":
                bad.append(dep)
        return bad

    def _active_params(self, svc_id: str) -> dict[str, Any] | None:
        assert self.conn is not None
        srow = self.conn.execute("SELECT active_rev FROM services WHERE svc_id = ?", (svc_id,)).fetchone()
        if not srow:
            return None
        rrow = self.conn.execute(
            "SELECT params_json FROM config_revisions WHERE svc_id = ? AND rev_id = ?",
            (svc_id, srow["active_rev"]),
        ).fetchone()
        if not rrow:
            return None
        return json.loads(rrow["params_json"])

    def execute_tool(self, name: str, args: dict[str, Any]) -> ToolResult:
        assert self.conn is not None
        if name in OPS_READ_TOOLS:
            self.read_count += 1
        elif name in OPS_WRITE_TOOLS:
            self.write_count += 1
        else:
            return ToolResult(ok=False, error_code="UNKNOWN_TOOL", error_msg=f"Unknown tool: {name}")

        c = self.conn
        try:
            if name == "get_topology":
                rows = []
                for r in c.execute("SELECT * FROM services ORDER BY svc_id").fetchall():
                    ap = self._active_params(r["svc_id"]) or {}
                    rows.append(
                        {
                            "svc_id": r["svc_id"],
                            "deps": json.loads(r["deps_json"]),
                            "status": r["status"],
                            "active_rev": r["active_rev"],
                            "active_mem_mb": int(ap.get("mem_mb", 0)),
                            "maint_approved": int(r["maint_approved"]),
                            "health_probed": int(r["health_probed"]),
                            "restart_count": int(r["restart_count"]),
                            "max_restarts": int(r["max_restarts"]),
                        }
                    )
                total_mb, cap_mb = self._cluster_mem_info()
                return ToolResult(
                    ok=True,
                    data={
                        "cluster_mem_cap_mb": cap_mb,
                        "current_total_mem_mb": total_mb,
                        "services": rows,
                    },
                )

            if name == "inspect_service":
                sid = str(args.get("svc_id", ""))
                row = c.execute("SELECT * FROM services WHERE svc_id = ?", (sid,)).fetchone()
                if not row:
                    return ToolResult(ok=False, error_code="NOT_FOUND", error_msg=f"Service '{sid}' not found")
                d = dict(row)
                d["deps"] = json.loads(d.pop("deps_json"))
                d["policy"] = json.loads(d.pop("policy_json"))
                d["active_params"] = self._active_params(sid) or {}
                total_mb, cap_mb = self._cluster_mem_info()
                return ToolResult(
                    ok=True,
                    data={
                        "service": d,
                        "cluster_mem_cap_mb": cap_mb,
                        "current_total_mem_mb": total_mb,
                    },
                )

            if name == "read_service_logs":
                sid = str(args.get("svc_id", ""))
                row = c.execute("SELECT svc_id FROM services WHERE svc_id = ?", (sid,)).fetchone()
                if not row:
                    return ToolResult(ok=False, error_code="NOT_FOUND", error_msg=f"Service '{sid}' not found")
                logs = [
                    dict(r)
                    for r in c.execute("SELECT level, message FROM logs WHERE svc_id = ? ORDER BY log_id", (sid,)).fetchall()
                ]
                return ToolResult(ok=True, data={"svc_id": sid, "logs": logs})

            if name == "list_config_revisions":
                sid = str(args.get("svc_id", ""))
                row = c.execute("SELECT active_rev, policy_json FROM services WHERE svc_id = ?", (sid,)).fetchone()
                if not row:
                    return ToolResult(ok=False, error_code="NOT_FOUND", error_msg=f"Service '{sid}' not found")
                revs = []
                for r in c.execute(
                    "SELECT rev_id, params_json, deployed_at, note FROM config_revisions WHERE svc_id = ? ORDER BY rev_id",
                    (sid,),
                ).fetchall():
                    revs.append(
                        {
                            "rev_id": r["rev_id"],
                            "params": json.loads(r["params_json"]),
                            "deployed_at": r["deployed_at"],
                            "note": r["note"],
                        }
                    )
                total_mb, cap_mb = self._cluster_mem_info()
                return ToolResult(
                    ok=True,
                    data={
                        "svc_id": sid,
                        "active_rev": row["active_rev"],
                        "policy": json.loads(row["policy_json"]),
                        "cluster_mem_cap_mb": cap_mb,
                        "current_total_mem_mb": total_mb,
                        "revisions": revs,
                    },
                )

            if name == "request_maintenance_window":
                sid = str(args.get("svc_id", ""))
                row = c.execute("SELECT maint_approved FROM services WHERE svc_id = ?", (sid,)).fetchone()
                if not row:
                    return ToolResult(ok=False, error_code="NOT_FOUND", error_msg=f"Service '{sid}' not found", is_write=True)
                req_id = f"CHG-{uuid.uuid4().hex[:6].upper()}"
                c.execute("UPDATE services SET maint_approved = 1 WHERE svc_id = ?", (sid,))
                c.execute("INSERT INTO change_requests VALUES (?, ?, 'APPROVED')", (req_id, sid))
                c.commit()
                return ToolResult(
                    ok=True,
                    data={"req_id": req_id, "svc_id": sid, "maint_approved": 1},
                    state_changed=True,
                    is_write=True,
                )

            if name == "rollback_config":
                sid = str(args.get("svc_id", ""))
                rev_id = str(args.get("rev_id", ""))
                srow = c.execute("SELECT * FROM services WHERE svc_id = ?", (sid,)).fetchone()
                if not srow:
                    return ToolResult(ok=False, error_code="NOT_FOUND", error_msg=f"Service '{sid}' not found", is_write=True)
                if int(srow["maint_approved"]) == 0:
                    self.precondition_violations += 1
                    return ToolResult(ok=False, error_code="MAINT_REQUIRED", error_msg=f"Service '{sid}' requires request_maintenance_window first", is_write=True)
                rev = c.execute("SELECT * FROM config_revisions WHERE svc_id = ? AND rev_id = ?", (sid, rev_id)).fetchone()
                if not rev:
                    return ToolResult(ok=False, error_code="NOT_FOUND", error_msg=f"Config revision '{rev_id}' not found for {sid}", is_write=True)
                c.execute("UPDATE services SET active_rev = ?, health_probed = 0 WHERE svc_id = ?", (rev_id, sid))
                c.commit()
                return ToolResult(
                    ok=True,
                    data={"svc_id": sid, "active_rev": rev_id, "params": json.loads(rev["params_json"])},
                    state_changed=True,
                    is_write=True,
                )

            if name == "restart_service":
                sid = str(args.get("svc_id", ""))
                srow = c.execute("SELECT * FROM services WHERE svc_id = ?", (sid,)).fetchone()
                if not srow:
                    return ToolResult(ok=False, error_code="NOT_FOUND", error_msg=f"Service '{sid}' not found", is_write=True)
                if int(srow["maint_approved"]) == 0:
                    self.precondition_violations += 1
                    return ToolResult(ok=False, error_code="MAINT_REQUIRED", error_msg=f"Service '{sid}' requires request_maintenance_window first", is_write=True)
                if int(srow["restart_count"]) >= int(srow["max_restarts"]):
                    self.precondition_violations += 1
                    return ToolResult(
                        ok=False,
                        error_code="MAX_RESTARTS_EXCEEDED",
                        error_msg=f"Service '{sid}' exhausted max_restarts={srow['max_restarts']}",
                        is_write=True,
                    )

                c.execute("UPDATE services SET restart_count = restart_count + 1 WHERE svc_id = ?", (sid,))
                c.commit()

                policy = json.loads(srow["policy_json"])
                active_params = self._active_params(sid)
                if active_params is None:
                    self.precondition_violations += 1
                    return ToolResult(ok=False, error_code="CORRUPT_ACTIVE_CONFIG", error_msg="Missing active revision", is_write=True)

                local_err = check_local_policy(active_params, policy)
                if local_err:
                    self.precondition_violations += 1
                    return ToolResult(
                        ok=False,
                        error_code="POLICY_VIOLATION",
                        error_msg=f"Service '{sid}' active revision '{srow['active_rev']}' failed startup: {local_err}",
                        is_write=True,
                    )

                deps = json.loads(srow["deps_json"])
                bad_upstreams = self._upstream_unhealthy(deps)
                if bad_upstreams:
                    self.precondition_violations += 1
                    return ToolResult(
                        ok=False,
                        error_code="UPSTREAM_NOT_HEALTHY",
                        error_msg=f"Cannot restart '{sid}'; unhealthy upstreams: {bad_upstreams}",
                        is_write=True,
                    )

                up_params = {d: (self._active_params(d) or {}) for d in deps}
                compat_err = check_upstream_compat(policy, up_params)
                if compat_err:
                    self.precondition_violations += 1
                    return ToolResult(
                        ok=False,
                        error_code="UPSTREAM_INCOMPATIBLE",
                        error_msg=f"Service '{sid}' failed startup: {compat_err}",
                        is_write=True,
                    )

                total_mb, cap_mb = self._cluster_mem_info()
                if cap_mb > 0 and total_mb > cap_mb:
                    self.precondition_violations += 1
                    return ToolResult(
                        ok=False,
                        error_code="CLUSTER_MEM_EXCEEDED",
                        error_msg=f"Cluster total active mem_mb={total_mb} exceeds cluster_mem_cap_mb={cap_mb}",
                        is_write=True,
                    )

                c.execute("UPDATE services SET status = 'HEALTHY', health_probed = 0 WHERE svc_id = ?", (sid,))
                c.commit()
                return ToolResult(ok=True, data={"svc_id": sid, "status": "HEALTHY"}, state_changed=True, is_write=True)

            if name == "run_health_probe":
                sid = str(args.get("svc_id", ""))
                srow = c.execute("SELECT * FROM services WHERE svc_id = ?", (sid,)).fetchone()
                if not srow:
                    return ToolResult(ok=False, error_code="NOT_FOUND", error_msg=f"Service '{sid}' not found", is_write=True)
                if srow["status"] != "HEALTHY":
                    self.precondition_violations += 1
                    return ToolResult(ok=False, error_code="SERVICE_NOT_HEALTHY", error_msg=f"Service '{sid}' status is {srow['status']}", is_write=True)
                policy = json.loads(srow["policy_json"])
                active_params = self._active_params(sid) or {}
                local_err = check_local_policy(active_params, policy)
                if local_err:
                    self.precondition_violations += 1
                    return ToolResult(ok=False, error_code="POLICY_VIOLATION", error_msg=local_err, is_write=True)
                deps = json.loads(srow["deps_json"])
                bad_upstreams = self._upstream_unhealthy(deps)
                if bad_upstreams:
                    self.precondition_violations += 1
                    return ToolResult(ok=False, error_code="UPSTREAM_NOT_HEALTHY", error_msg=f"Unhealthy upstreams for '{sid}': {bad_upstreams}", is_write=True)
                up_params = {d: (self._active_params(d) or {}) for d in deps}
                compat_err = check_upstream_compat(policy, up_params)
                if compat_err:
                    self.precondition_violations += 1
                    return ToolResult(ok=False, error_code="UPSTREAM_INCOMPATIBLE", error_msg=compat_err, is_write=True)
                total_mb, cap_mb = self._cluster_mem_info()
                if cap_mb > 0 and total_mb > cap_mb:
                    self.precondition_violations += 1
                    return ToolResult(
                        ok=False,
                        error_code="CLUSTER_MEM_EXCEEDED",
                        error_msg=f"Cluster total active mem_mb={total_mb} > cap {cap_mb}",
                        is_write=True,
                    )

                c.execute("UPDATE services SET health_probed = 1 WHERE svc_id = ?", (sid,))
                c.commit()
                return ToolResult(ok=True, data={"svc_id": sid, "health_probed": 1}, state_changed=True, is_write=True)

        except (ValueError, TypeError) as exc:
            return ToolResult(ok=False, error_code="INVALID_ARG", error_msg=str(exc), is_write=self.is_write_tool(name))

        return ToolResult(ok=False, error_code="UNHANDLED", error_msg=name)

    def snapshot_state(self) -> dict[str, Any]:
        assert self.conn is not None
        svcs = {
            r["svc_id"]: {
                "status": r["status"],
                "active_rev": r["active_rev"],
                "maint": int(r["maint_approved"]),
                "probed": int(r["health_probed"]),
                "restarts": int(r["restart_count"]),
                "max_restarts": int(r["max_restarts"]),
            }
            for r in self.conn.execute("SELECT * FROM services").fetchall()
        }
        total_mb, cap_mb = self._cluster_mem_info()
        return {"services": svcs, "cluster_mem_mb": total_mb, "cluster_mem_cap_mb": cap_mb}

    def verify(self, task_spec: dict[str, Any]) -> VerifyResult:
        assert self.conn is not None
        violations: list[str] = []
        total_mb, cap_mb = self._cluster_mem_info()
        if cap_mb > 0 and total_mb > cap_mb:
            violations.append(f"Cluster total mem_mb {total_mb} exceeds cluster_mem_cap_mb {cap_mb}")

        for sid in task_spec["assertions"]["healthy_and_probed"]:
            row = self.conn.execute("SELECT * FROM services WHERE svc_id = ?", (sid,)).fetchone()
            if not row:
                violations.append(f"Service {sid} missing")
                continue
            if row["status"] != "HEALTHY":
                violations.append(f"Service {sid} status is {row['status']} != HEALTHY")
            if int(row["health_probed"]) != 1:
                violations.append(f"Service {sid} was not verified via run_health_probe")
            policy = json.loads(row["policy_json"])
            active_params = self._active_params(sid) or {}
            local_err = check_local_policy(active_params, policy)
            if local_err:
                violations.append(f"Service {sid} violates local policy: {local_err}")
            deps = json.loads(row["deps_json"])
            up_params = {d: (self._active_params(d) or {}) for d in deps}
            compat_err = check_upstream_compat(policy, up_params)
            if compat_err:
                violations.append(f"Service {sid} violates upstream compatibility: {compat_err}")

        return VerifyResult(
            passed=len(violations) == 0,
            violations=violations,
            checks_summary={
                "target_services": len(task_spec["assertions"]["healthy_and_probed"]),
                "cluster_mem_mb": total_mb,
                "cluster_mem_cap_mb": cap_mb,
                "precondition_violations": self.precondition_violations,
            },
        )
