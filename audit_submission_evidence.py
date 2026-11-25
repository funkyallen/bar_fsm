"""Recheck formal write outcomes, visible messages, and the qualified accounting bounds."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import json
import sys


def read(root, name):
    return json.loads((root / name).read_text(encoding="utf-8"))


def formal_writes(root):
    import audit_oracle_v50 as oracle
    from scripts.research_accounting_void_contract import assess_authoritative_voidability
    from scripts.research_v34_intervention import key
    from scripts.research_v40_control import assess, feedback
    from scripts.research_voidability_qualification import assess_with_receipt

    fixtures = {case["task"]["task_id"]: case["task"] for case in read(root, "formal_fixtures/cases.json")}
    formal = read(root, "live/formal_completion/result.json")["rows"]
    coverage = defaultdict(Counter)
    candidates = []
    service_holds = []
    for episode in formal:
        task = fixtures[episode["task"]]
        service = task["domain"] == "ops_troubleshoot"
        contract = oracle.service_contract(task) if service else None

        def verdict(env, witness=False):
            try:
                if service:
                    return oracle.service_verdict(env, task, contract, retain_witness=witness)
                return oracle.account_verdict(env, task, retain_witness=witness)
            except ValueError as error:
                # A missing native witness is an unresolved label, even after a positive relaxed search.
                return {"reachable": None, "reason": str(error)}

        counts = coverage[episode["arm"]]
        ids = {}
        step, receipts = None, {}
        with closing(oracle.env_for(task)) as env:
            for event in episode["trace"]:
                kind = event["event"]
                call = event.get("candidate", {})
                name = call.get("name")
                if event["step"] != step:
                    step, receipts = event["step"], {}
                if service and kind in ("dispatch", "local_hold", "future_block"):
                    goal = task["goal"]
                    checks = {
                        "A": assess(call, goal, receipts),
                        "B": assess_authoritative_voidability(call, goal, receipts, {}),
                        "C": assess_with_receipt(call, goal, receipts),
                    }
                    assert all(json.loads(json.dumps(asdict(c))) == event["assessment"] for c in checks.values())
                    counts["service_decisions_checked_across_versions"] += 1
                    if kind != "dispatch":
                        service_holds.append({"slot": episode["slot"], "step": step, "kind": kind, "arm": episode["arm"],
                                              "messages": {v: feedback(c, episode["arm"]) for v, c in checks.items()}})
                measured = (name in ("restart_service", "post_adjustment")
                            and event.get("assessment", {}).get("local") == "pass"
                            and kind in ("dispatch", "future_block"))
                if measured:
                    before = verdict(env)
                    with closing(oracle.env_for(task)) as fork:
                        env.conn.backup(fork.conn)
                        outcome = oracle.execute(fork, call, require_success=False)
                        after = verdict(fork)
                        b, a = before["reachable"], after["reachable"]
                        if not outcome.ok:
                            label = "backend_rejected"
                        elif b is None or a is None:
                            label = "unknown"
                        elif b and not a:
                            label = "new_loss"
                        elif b:
                            label = "preserving"
                        else:
                            label = "already_unreachable"
                        record = {"slot": episode["slot"], "step": event["step"], "provider": episode["provider"],
                                  "arm": episode["arm"], "domain": task["domain"], "event": kind, "tool": name,
                                  "candidate": call, "before": b, "after": a, "label": label,
                                  "enumerator_new_loss": event.get("oracle", {}).get("new_loss", False)}
                        if label == "new_loss" and service:
                            sid = call["arguments"]["svc_id"]
                            expected = fork.snapshot_state()
                            original = expected["services"][sid]
                            assert original["max_restarts"] - original["restarts"] == 0
                            fork.conn.execute("UPDATE services SET max_restarts = max_restarts + 1 WHERE svc_id = ?", (sid,))
                            fork.conn.commit()
                            expected["services"][sid] = {**original, "max_restarts": original["max_restarts"] + 1}
                            assert fork.snapshot_state() == expected
                            record["restore_one_restart"] = verdict(fork, witness=True)
                    counts[f"{name}:{kind}:{label}"] += 1
                    candidates.append(record)
                elif kind in ("dispatch", "future_block") and name and env.is_write_tool(name):
                    counts[f"other_write:{name}:{kind}"] += 1
                if kind in ("auto_read", "dispatch"):
                    oracle.replay_archived_call(env, event, ids)
                if kind == "auto_read" and event["reply"]["ok"]:
                    receipts[key(call)] = event["reply"]["data"]
            assert env.verify(task).passed == episode["terminal"]
    summary = {}
    for arm in ("M0", "M1", "M2"):
        arm_rows = [r for r in candidates if r["arm"] == arm]
        restored = sum(
            bool(r.get("restore_one_restart", {}).get("reachable"))
            for r in arm_rows
            if r["label"] == "new_loss"
        )
        summary[arm] = {
            "locally_cleared": len(arm_rows),
            "preserving": sum(r["label"] == "preserving" for r in arm_rows),
            "new_loss": sum(r["label"] == "new_loss" for r in arm_rows),
            "already_unreachable": sum(r["label"] == "already_unreachable" for r in arm_rows),
            "restored_by_one_restart": restored,
        }
    total = {k: sum(summary[arm][k] for arm in ("M0", "M1", "M2")) for k in summary["M0"]}
    total["restarts"] = sum(r["tool"] == "restart_service" for r in candidates)
    total["postings"] = sum(r["tool"] == "post_adjustment" for r in candidates)
    assert total["locally_cleared"] == 686
    assert total["restarts"] == 378 and total["postings"] == 308
    assert total["preserving"] == 667
    assert total["new_loss"] == 15
    assert total["already_unreachable"] == 4
    assert total["restored_by_one_restart"] == 15
    summary["total"] = total
    return {"episodes": len(formal), "counts_by_arm": dict(coverage), "formal_686_summary": summary,
            "candidates": candidates, "service_hold_comparisons": service_holds}


def messages_and_omissions(root, service_holds):
    from scripts.research_v40_control import feedback
    from scripts.research_v40_cases import env_for

    version = read(root, "source_data/control_versions_v50.json")["candidates"]
    matrices = {(f'formal_{"original" if row["stage"] == "original" else "completion"}/{row["run_id"]}',
                 row["step"]): row for row in version}
    formal_rows = read(root, "live/formal_completion/result.json")["rows"]
    fixtures = {case["task"]["task_id"]: case["task"] for case in read(root, "formal_fixtures/cases.json")}
    counts = Counter()
    holds = []
    bundles = []
    omitted = defaultdict(Counter)
    tool_types = {}
    for domain in ("ops_troubleshoot", "db_transaction"):
        with closing(env_for(next(t for t in fixtures.values() if t["domain"] == domain))) as env:
            tool_types[domain] = {schema["function"]["name"]: env.is_write_tool(schema["function"]["name"])
                                  for schema in env.tool_schemas()}
    scored = {(r["run_id"], r["slot"]) for r in formal_rows}
    service_messages = {(r["slot"], r["step"]): r for r in service_holds}
    for line in (root / "source_data/request_call_map_v50.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        run, step = row["run"], row["step"]
        stage, short_run = run.split("/", 1)
        if stage not in ("formal_original", "formal_completion"):
            continue
        if row["terminal"] is None or (short_run, row["slot"]) not in scored:
            continue
        domain = fixtures[row["source_task"]]["domain"]
        arm = row["arm"]
        counts["scored_model_replies"] += 1
        suffix = row["context_messages_appended"]
        old = matrices.get((run, step))
        if old is not None:
            counts["accounting_condition_rows_matched_to_messages"] += 1
            if old["A"]["resource"] != old["B"]["resource"]:
                assert old["A"]["disposition"] == "DISPATCH"
                counts["accounting_changed_resource_verdicts_on_dispatch"] += 1
            if old["A"]["disposition"] != "DISPATCH":
                messages = {v: feedback(SimpleNamespace(**old[v]), arm) for v in "ABC"}
                observed = next(m["content"] for m in reversed(suffix) if m["role"] == "user")
                assert observed == messages["A"]
                holds.append({"cohort": "accounting", "run": run, "step": step, "arm": arm,
                              "messages": messages, "recorded": observed})
        elif domain == "ops_troubleshoot" and row["disposition"] in ("local_hold", "future_block"):
            comparison = service_messages.pop((row["slot"], step))
            observed = next(m["content"] for m in reversed(suffix) if m["role"] == "user")
            assert comparison["messages"]["A"] == observed
            holds.append({"cohort": "service", "run": run, **comparison, "recorded": observed})
        if domain == "db_transaction":
            tool_data = (json.loads(m["content"]).get("data") for m in suffix if m["role"] == "tool")
            visible_bundles = sum(isinstance(data, dict) and {"ledger_entries", "invoices"} <= data.keys()
                                  for data in tool_data)
            if visible_bundles:
                bundles.append({"run": run, "slot": row["slot"], "step": step,
                                "visible_successful_bundles": visible_bundles,
                                "next_request_exists": row["next_request_id"] is not None})
        if row["omitted_indices"]:
            key = "/".join((row["provider"], arm, domain, "completed" if row["terminal"] else "unfinished"))
            omitted[key]["replies_with_omissions"] += 1
            for index in row["omitted_indices"]:
                tool = row["proposed_calls"][index]["name"]
                # The native audit check updates audit_passed, despite its read designation.
                mutates = tool_types[domain][tool] or tool == "run_audit_check"
                omitted[key]["omitted_mutating" if mutates else "omitted_read_only"] += 1
                omitted[key][tool] += 1
    assert not service_messages
    counts["hold_messages_compared"] = len(holds)
    counts["different_hold_messages"] = sum(len(set(row["messages"].values())) > 1 for row in holds)
    counts["visible_bundle_replies"] = sum(row["visible_successful_bundles"] for row in bundles)
    counts["visible_bundle_replies_with_next_request"] = sum(row["visible_successful_bundles"] for row in bundles
                                                              if row["next_request_exists"])
    return {"counts": counts, "hold_messages": holds, "bundle_message_blocks": bundles,
            "omitted_by_provider_arm_domain_outcome": dict(omitted)}


def accounting_and_fixtures(root):
    rows = read(root, "live/shared_allowance/fixed_candidates.json")
    fixtures = {c["task"]["task_id"]: c["task"] for c in read(root, "live/shared_allowance/cases.json")}
    agg = Counter()
    bounds = Counter()
    overlap = Counter()
    for row in rows:
        guard = row["guard"]
        cap = fixtures[row["task"]]["goal"]["max_per_post_cents"]
        obs = guard["obligations"]
        l_count = len(obs)
        total = sum(abs(residual) for _, residual in obs)
        l_agg = (total + cap - 1) // cap
        l_max = max(l_count, l_agg)
        l_acc = guard["lower"]
        assert l_acc >= l_max
        assert (guard["q_after"] < l_count) == row["blocked"]["B1"]
        assert (guard["q_after"] < l_acc) == row["blocked"]["certificate"]
        h_count = guard["q_after"] < l_count
        h_agg = guard["q_after"] < l_agg
        h_max = guard["q_after"] < l_max
        h_acc = guard["q_after"] < l_acc
        label = "preserving" if row["successor_reachable"] else "loss"
        agg[f"{label}:{'hold' if h_agg else 'no_hold'}"] += 1
        agg[f"per_invoice_only:{label}"] += h_acc and not h_agg
        for name, flag in (("invoice_count", h_count), ("aggregate_capacity", h_agg),
                           ("max_count_and_aggregate", h_max), ("per_invoice_capacity", h_acc)):
            bounds[f"{name}:{'losses_held' if label == 'loss' else 'preserving_held' if flag else 'preserving_allowed'}"] += (
                flag if label == "loss" else 1
            )
        if label == "loss":
            assert h_acc
            if h_count and h_agg:
                overlap["both_count_and_aggregate"] += 1
            elif h_count and not h_agg:
                overlap["count_only"] += 1
            elif h_agg and not h_count:
                overlap["aggregate_only"] += 1
            else:
                overlap["per_invoice_only_beyond_max"] += 1
    overlap["per_invoice_only_beyond_aggregate"] = overlap["count_only"] + overlap["per_invoice_only_beyond_max"]
    assert overlap == {
        "both_count_and_aggregate": 4,
        "count_only": 1,
        "aggregate_only": 6,
        "per_invoice_only_beyond_max": 4,
        "per_invoice_only_beyond_aggregate": 5,
    }

    nat_episodes = read(root, "live/shared_allowance/live/result.json")["rows"]
    nat = Counter()
    for episode in nat_episodes:
        cap = fixtures[episode["task"]]["goal"]["max_per_post_cents"]
        for event in episode["trace"]:
            if event.get("oracle") is None:
                continue
            guard = event["guard"]
            obs = guard["obligations"]
            l_count = len(obs)
            l_agg = (sum(abs(residual) for _, residual in obs) + cap - 1) // cap
            l_max = max(l_count, l_agg)
            l_acc = guard["lower"]
            q = guard["q_after"]
            assert event["oracle"]["after_if_dispatched"] is True
            assert not (q < l_count or q < l_agg or q < l_max or q < l_acc)
            nat["preserving"] += 1
            nat["invoice_count"] += int(q < l_count)
            nat["aggregate_capacity"] += int(q < l_agg)
            nat["max_count_and_aggregate"] += int(q < l_max)
            nat["per_invoice_capacity"] += int(q < l_acc)
    assert nat["preserving"] == 177

    constructed = read(root, "oracle_results.json")["cases"]
    generation = {}
    for domain in ("ops_troubleshoot", "db_transaction"):
        domain_rows = [row for row in constructed if row["domain"] == domain]
        # Fixture fields include the candidate object index. Amount is the remaining identifier-free argument.
        keys = Counter(json.dumps([row["fixture"], row["candidate"]["arguments"].get("delta_cents")], sort_keys=True)
                       for row in domain_rows)
        generation[domain] = {"rows": len(domain_rows), "evidence": Counter(r["evidence"] for r in domain_rows),
                              "unique_state_candidate_parameter_tuples": len(keys),
                              "maximum_tuple_multiplicity": max(keys.values())}
    return {"aggregate_capacity": agg,
            "accounting_bound_comparison": {"fixed_52": dict(bounds), "loss_overlap": dict(overlap),
                                            "natural_177_holds": dict(nat)},
            "constructed_design": generation}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.archive.resolve()))
    coverage = formal_writes(args.archive)
    report = {"formal_write_coverage": coverage,
              "context_and_omissions": messages_and_omissions(args.archive, coverage["service_hold_comparisons"]),
              **accounting_and_fixtures(args.archive)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
