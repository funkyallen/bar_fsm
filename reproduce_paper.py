"""Regenerate the manuscript's audits, tables, and figures without model access."""

import argparse
import ast
import csv
import json
import os
import subprocess
import sys
from decimal import Decimal
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ARM_IDS = ("M0", "M1", "M2")
CONTRASTS = (("M1", "M0"), ("M2", "M1"), ("M2", "M0"))
OFFLINE_LAUNCHER = """
import runpy, socket, sys
def deny(*args, **kwargs):
    raise RuntimeError('Network access is disabled during reproduction.')
socket.socket.connect = deny
socket.socket.connect_ex = deny
socket.create_connection = deny
path = sys.argv.pop(1)
sys.argv[0] = path
runpy.run_path(path, run_name='__main__')
"""


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit_accounting_contract():
    source = ROOT / "runtime/src/aroute/envs/db_transaction_env.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    executor = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == "execute_tool")
    public_tools = set()
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in ("DB_READ_TOOLS", "DB_WRITE_TOOLS")
            for target in statement.targets
        ):
            public_tools.update(ast.literal_eval(statement.value))

    handlers = {}
    for statement in ast.walk(executor):
        if (isinstance(statement, ast.If) and isinstance(statement.test, ast.Compare)
                and ast.unparse(statement.test.left) == "name"
                and len(statement.test.comparators) == 1
                and isinstance(statement.test.ops[0], ast.Eq)):
            tool = ast.literal_eval(statement.test.comparators[0])
            handlers[tool] = statement
    if set(handlers) != public_tools:
        raise ValueError("The accounting tool set and the audited handlers differ.")

    def writes(node):
        result = []
        for call in ast.walk(node):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "execute":
                sql = " ".join(ast.literal_eval(call.args[0]).split())
                if not sql.upper().startswith("SELECT "):
                    result.append(sql)
        return sorted(result)

    expected = {
        "list_open_tickets": [], "get_invoice_bundle": [], "query_account": [],
        "run_audit_check": ["UPDATE reconciliation_tickets SET audit_passed = ? WHERE ticket_id = ?"],
        "void_duplicate_entry": [
            "UPDATE ledger_entries SET voided = 1 WHERE entry_id = ?",
            "UPDATE accounts SET balance_cents = balance_cents - ? WHERE acct_id = ?",
            "UPDATE reconciliation_tickets SET audit_passed = 0 WHERE ticket_id = ?",
        ],
        "post_adjustment": [
            "INSERT INTO ledger_entries VALUES (?, ?, ?, ?, ?, ?, '2026-11-20', 'adjustment', 0, 0)",
            "UPDATE accounts SET balance_cents = ? WHERE acct_id = ?",
            "UPDATE reconciliation_tickets SET audit_passed = 0, adj_count = adj_count + 1 WHERE ticket_id = ?",
        ],
        "reconcile_invoice": ["UPDATE invoices SET status = 'RECONCILED' WHERE inv_id = ?"],
        "close_ticket": ["UPDATE reconciliation_tickets SET status = 'CLOSED' WHERE ticket_id = ?"],
    }
    for tool, handler in handlers.items():
        if writes(handler) != sorted(expected[tool]):
            raise ValueError(f"Accounting writes differ for {tool}.")
    if writes(executor) != sorted(sql for each in expected.values() for sql in each):
        raise ValueError("The accounting dispatcher contains an unclassified write.")

    void_steps = handlers["void_duplicate_entry"].body
    guards = {ast.unparse(step.test): step for step in void_steps if isinstance(step, ast.If)}
    for gate, error in (("int(row['is_duplicate']) == 0", "NOT_A_DUPLICATE"),
                        ("int(row['voided']) == 1", "ALREADY_VOIDED")):
        if error not in ast.unparse(guards[gate]) or not any(isinstance(x, ast.Return) for x in guards[gate].body):
            raise ValueError(f"The void rejection guard changed: {gate}.")
    ddl = next(" ".join(node.value.split()) for node in ast.walk(tree)
               if isinstance(node, ast.Constant) and isinstance(node.value, str)
               and "CREATE TABLE ledger_entries" in node.value)
    if "is_duplicate INTEGER NOT NULL, voided INTEGER NOT NULL" not in ddl:
        raise ValueError("The layout of the ledger eligibility flags changed.")
    receipt = ROOT / "runtime/scripts/research_voidability_qualification.py"
    if "bool(row['is_duplicate'] and (not row['voided']))" not in ast.unparse(ast.parse(receipt.read_text(encoding="utf-8"))):
        raise ValueError("The public eligibility receipt no longer mirrors the void condition.")
    return {"source": source.relative_to(ROOT).as_posix(), "public_tools_checked": sorted(handlers),
            "result": "Authorized continuations cannot introduce new free voids for included invoices."}


def export_tables(output):
    pooled = load(output / "formal_completion_audit.json")["pooled"]
    points = load(output / "editorial_checks.json")["formal_cost_points"]["arms"]
    with (output / "table1.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["policy", "completed", "assigned", "total_charge_cny", "cny_per_completed_episode"])
        for arm in ARM_IDS:
            row = points[arm]
            writer.writerow([arm, row["successes"], row["assigned"], row["cost_cny"], row["cny_per_success"]])
    with (output / "family_effects.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["family", "matched_groups", *ARM_IDS, *(f"{a}_minus_{b}_pp" for a, b in CONTRASTS)])
        for family, counts in pooled["family_terminal_counts"].items():
            n = sum(counts["M0"].values())
            if n != 12 or any(sum(counts[arm].values()) != n for arm in ARM_IDS):
                raise ValueError(f"Unexpected per-family denominator: {family}.")
            successes = {arm: counts[arm].get("success", 0) for arm in ARM_IDS}
            differences = [Decimal(100 * (successes[a] - successes[b])) / n for a, b in CONTRASTS]
            writer.writerow([family, n, *(successes[x] for x in ARM_IDS), *(f"{d:+.2f}" for d in differences)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    out = parser.parse_args().out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    environment = {k: v for k, v in os.environ.items()
                   if not any(key in k.upper() for key in ("API_KEY", "TOKEN", "SECRET", "PYTHONPATH"))}

    def run(script, *args):
        subprocess.run([sys.executable, "-B", "-E", "-c", OFFLINE_LAUNCHER, script, *map(str, args)],
                       cwd=ROOT, env=environment, check=True)

    # Later audits read the replay files produced by the first three scripts.
    for script in ("reproduce_revision.py", "reproduce_update.py", "reproduce_final.py",
                   "audit_trace_forensics.py", "reproduce_frozen_hold.py", "audit_control_versions_v50.py",
                   "audit_oracle_v50.py", "audit_protocol_v50.py"):
        run(script, "--out", out)
    for name in ("trace_forensics.json", "frozen_hold_audit.json"):
        if load(out / name) != load(ROOT / "source_data" / name):
            raise ValueError(f"Reproduced audit differs from release: {name}.")
    run("reproduce_evidence_v50.py", "--audit-dir", out, "--out", out)
    for name in ("control_versions_v50.json", "oracle_v50.json", "protocol_v50.json", "request_call_map_v50.jsonl",
                 "evidence_v50.json", "formal_episode_exposure.csv", "formal_candidate_exposure.csv",
                 "natural_accounting_exposure.csv"):
        if (out / name).read_bytes() != (ROOT / "source_data" / name).read_bytes():
            raise ValueError(f"Recomputed study evidence differs from release: {name}.")
    export_tables(out)
    stages_file = out / "qualification_stages_v51.json"
    run("audit_qualification_stages.py", "--qualification", out / "voidability_720.json", "--out", stages_file)
    if stages_file.read_bytes() != (ROOT / "source_data" / stages_file.name).read_bytes():
        raise ValueError("Reproduced qualification guard-stage counts differ from the release.")
    report = json.dumps(audit_accounting_contract(), indent=2) + "\n"
    (out / "accounting_contract_audit.json").write_text(report, encoding="utf-8")
    figures = out / "figures"
    run("plot_manuscript_figures.py", "--audit-dir", out, "--out", figures)
    if load(figures / "figure_data.json") != load(ROOT / "figures/figure_data.json"):
        raise ValueError("Reproduced figure data differ from the released source data.")
    run("plot_candidate_chain_v50.py", "--audit-dir", out, "--out", figures)
    if load(figures / "candidate_chain_data.json") != load(ROOT / "figures/candidate_chain_data.json"):
        raise ValueError("Reproduced candidate-chain data differ from the released source data.")
    print("Rebuilt the qualification, runtime, and scoring audits, version and oracle checks, exposure tables, and figures.")


if __name__ == "__main__":
    main()
