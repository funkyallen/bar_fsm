"""Freeze and compare generic and structured feedback at archived first holds."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from decimal import Decimal
from math import comb
from pathlib import Path
from random import Random
import sqlite3

from aroute.core.billing_meter import BillingMeter
from aroute.core.config import PROJECT_ROOT, get_api_key, get_base_url, load_models_catalog
from aroute.core.immutable import sha256_file
from aroute.core.model_pool import LiveModelPool

from .research_complete_frozen_trial import bill, write
from .research_feedback_and_transfer import PrefixPool, replace_obligations
from .research_v40_cases import env_for
from .research_v40_control import Assessment, feedback
from .research_v40_trial import DATABASE, PROVIDERS, _json_digest, _read_json, _script, audit, run_branch


ARMS = ("M1", "M2")
PROVIDER = "dashscope"
CAP_CNY = 6.0
REPEATS = 5
WORKERS = 4


def snapshot(case, trace):
    with closing(env_for(case["task"])) as env:
        for event in trace:
            if event["event"] == "dispatch":
                candidate = event["candidate"]
                env.execute_tool(candidate["name"], candidate["arguments"])
        return env.snapshot_state()


def register(archive):
    manifest = _read_json(archive / "sha256.json")
    paths = ("live/formal/protocol.json", "live/formal_completion/result.json", "formal_fixtures/cases.json")
    for path in paths:
        if sha256_file(archive / path) != manifest[path]:
            raise ValueError(f"Source archive integrity failure: {path}.")
    parent, merged, cases = (_read_json(archive / path) for path in paths)
    registered = (
        "configs/models.yaml", "scripts/research_v40_trial.py", "scripts/research_v40_control.py",
        "scripts/research_v40_cases.py", "src/aroute/envs/ops_troubleshoot_env.py",
        "src/aroute/core/model_pool.py", "src/aroute/core/prewrite_execution.py",
    )
    for path in registered:
        if sha256_file(PROJECT_ROOT / path) != parent["source_hashes"][path]:
            raise ValueError(f"Historical execution source mismatch: {path}.")
    case_by_task = {case["task"]["task_id"]: case for case in cases}
    groups, excluded = {}, []
    for row in merged["rows"]:
        if row["terminal"] is None or row["arm"] not in ARMS:
            continue
        holds = [event for event in row["trace"] if event["event"] == "future_block"]
        if not holds:
            continue
        first = holds[0]
        if row["provider"] != PROVIDER or not first["oracle"]["new_loss"]:
            raise ValueError(f"Unexpected first hold in slot {row['slot']}.")
        trace = [event for event in row["trace"] if event["step"] <= first["step"]]
        stable_id = _json_digest([row["task"], row["provider"], trace])
        if stable_id not in groups:
            case = case_by_task[row["task"]]
            state = snapshot(case, trace)
            groups[stable_id] = {
                "id": stable_id, "task": row["task"], "family": row["family"], "trace": trace,
                "state": state, "state_sha256": _json_digest(state), "source_slots": [],
                "remaining_decisions": case["task"]["max_steps"] - first["step"] - 1,
            }
        groups[stable_id]["source_slots"].append(row["slot"])
        excluded.extend({"slot": row["slot"], "step": h["step"], "prior_feedback": row["arm"]} for h in holds[1:])
    if len(groups) != 4 or sum(len(g["source_slots"]) for g in groups.values()) != 10 or len(excluded) != 1:
        raise ValueError("First-hold cohort differs from the audited v4.8 archive.")
    rng = Random(20261125)
    allocation = []
    for repeat in range(1, REPEATS + 1):
        for checkpoint in rng.sample(list(groups.values()), len(groups)):
            for arm in rng.sample(ARMS, len(ARMS)):
                allocation.append({
                    "slot": len(allocation) + 1, "task": checkpoint["task"], "provider": PROVIDER,
                    "prefix": checkpoint["id"], "arm": arm, "repeat": repeat,
                })
    return {
        "estimand": "Difference between persistent feedback policies on these four frozen first holds.",
        "source_sha256": {path: manifest[path] for path in paths},
        "runner_sha256": sha256_file(Path(__file__)),
        "prefix_pool_sha256": sha256_file(Path(__file__).with_name("research_feedback_and_transfer.py")),
        "models": {PROVIDER: parent["models"][PROVIDER]},
        "max_reply_tokens": parent["max_output_tokens"],
        "max_decisions_including_prefix": parent["max_decisions"],
        "transport": parent["transport"], "budget_cny": CAP_CNY, "worker_limit": WORKERS,
        "repeats_per_arm_per_checkpoint": REPEATS, "order_seed": 20261125,
        "later_holds_excluded": excluded, "checkpoints": list(groups.values()),
        "cases": [case_by_task[task] for task in sorted({g["task"] for g in groups.values()})],
        "allocation": allocation,
    }


def summary(protocol, rows):
    cell = {(row["prefix"], row["repeat"], row["arm"]): row["terminal"] for row in rows}
    checkpoints, discordance = [], Counter()
    lower = upper = 0
    totals = {arm: 0 for arm in ARMS}
    for prefix in protocol["checkpoints"]:
        counts, nulls = Counter(), Counter()
        for repeat in range(1, REPEATS + 1):
            a, b = (cell.get((prefix["id"], repeat, arm)) for arm in ARMS)
            for arm, value in zip(ARMS, (a, b), strict=True):
                counts[arm] += value is True
                nulls[arm] += value is None
            lower += (b is True) - (a is not False)
            upper += (b is not False) - (a is True)
            if a is None or b is None:
                discordance["unresolved"] += 1
            else:
                discordance["both_success" if a and b else
                            "M2_only" if b else "M1_only" if a else "both_fail"] += 1
        for arm in ARMS:
            totals[arm] += counts[arm]
        checkpoints.append({"id": prefix["id"], "task": prefix["task"], "family": prefix["family"],
                            "successes": dict(counts), "unresolved": dict(nulls)})
    n = len(protocol["allocation"]) // 2
    wins, losses = discordance["M2_only"], discordance["M1_only"]
    pairs = wins + losses
    exact = min(1, 2 * sum(comb(pairs, i) for i in range(min(wins, losses) + 1)) / 2**pairs)
    return {
        "checkpoints": checkpoints,
        "pairs": {name: discordance[name] for name in ("M2_only", "M1_only", "both_success", "both_fail", "unresolved")},
        "successes": totals, "assessed": sum(row["terminal"] is not None for row in rows),
        "assigned": len(protocol["allocation"]), "assigned_pairs": n,
        "difference_pp": (wins - losses) * 100 / n if not discordance["unresolved"] else None,
        "missing_outcome_bounds_pp": [lower * 100 / n, upper * 100 / n],
        "exact_paired_sign_p": exact if not discordance["unresolved"] else None,
    }


def execute(archive, out, live):
    protocol = register(archive)
    signature = _json_digest(protocol)
    out.mkdir(parents=True, exist_ok=True)
    registration = out / "registration.json"
    if registration.exists() and _json_digest(_read_json(registration)) != signature:
        raise ValueError("Frozen registration changed.")
    if live:
        acceptance = _read_json(out / "offline/result.json")
        if (acceptance["scope"] != "scripted" or acceptance["registration_sha256"] != signature
                or acceptance["assessed"] != len(protocol["allocation"])):
            raise ValueError("Matching scripted acceptance is required.")
    write(registration, protocol)
    catalog = load_models_catalog()
    provider = catalog.providers[PROVIDER]
    if live and (not get_api_key(provider) or get_base_url(provider).rstrip("/") != PROVIDERS[PROVIDER]
                 or provider.tiers["fast"].resolve_model_id() != protocol["models"][PROVIDER]):
        raise ValueError("The requested DashScope deployment, endpoint or credential is unavailable.")
    batch = out / ("live" if live else "offline")
    batch.mkdir()
    meter = BillingMeter(batch / DATABASE, project_cap_cny=CAP_CNY)
    delegate = LiveModelPool(catalog, meter, timeout_sec=120, max_retries=0) if live else None
    cases = {case["task"]["task_id"]: case for case in protocol["cases"]}
    prefixes = {p["id"]: p for p in protocol["checkpoints"]}

    def run(job):
        case = cases[job["task"]]
        prefix = prefixes[job["prefix"]]
        expected = prefix["trace"]
        actions = [event for event in expected if event["event"] != "auto_read"]
        pool = PrefixPool(actions, delegate or _script(case, catalog, meter))
        row = run_branch(case, job, pool, meter, "formal")
        audit(case, row)
        actual = row["trace"][:len(expected)]
        if _json_digest(actual) != _json_digest(expected) or _json_digest(snapshot(case, actual)) != prefix["state_sha256"]:
            raise ValueError(f"Archived prefix, read receipts, or database state changed for slot {job['slot']}.")
        decision = Assessment(**replace_obligations(expected[-1]["assessment"]))
        if not pool.first_prompt_hash or pool.feedback_text != feedback(decision, job["arm"]):
            raise ValueError(f"Prompt binding failed for slot {job['slot']}.")
        with closing(sqlite3.connect(batch / DATABASE, timeout=30)) as db:
            db.row_factory = sqlite3.Row
            attempts = list(db.execute("SELECT * FROM api_attempts WHERE run_id = ?", (row["run_id"],)))
        if not attempts or min(attempt["step_idx"] for attempt in attempts) != len(actions):
            raise ValueError("The frozen prefix generated an unregistered model call.")
        row.update(bill(attempts), shared_prompt_sha256=pool.first_prompt_hash, feedback_text=pool.feedback_text,
                   checkpoint_state_sha256=prefix["state_sha256"], remaining_decisions=prefix["remaining_decisions"])
        return row

    rows = []
    allocation = protocol["allocation"]
    waves = [allocation[:2], *(allocation[i:i + WORKERS] for i in range(2, len(allocation), WORKERS))]
    with ThreadPoolExecutor(max_workers=WORKERS) as worker:
        for jobs in waves:
            wave = list(worker.map(run, jobs))
            for left, right in zip(wave[::2], wave[1::2], strict=True):
                if left["shared_prompt_sha256"] != right["shared_prompt_sha256"]:
                    raise ValueError("Matched continuations received different pre-feedback prompts.")
            rows.extend(wave)
            write(batch / "checkpoint.json", rows)
            print(f"{len(rows)}/{len(allocation)} continuations, {sum(r['terminal'] is not None for r in rows)} assessed",
                  flush=True)
            if live and all(r["terminal"] is None for r in wave):
                break
    with closing(sqlite3.connect(batch / DATABASE)) as db:
        db.row_factory = sqlite3.Row
        ledger = bill(list(db.execute("SELECT * FROM api_attempts")))
    result = {"scope": "live" if live else "scripted", "registration_sha256": signature,
              **summary(protocol, rows), "rows": rows, "ledger": ledger}
    if not live and (result["assessed"] != len(allocation) or sum(result["successes"].values()) != len(allocation)):
        raise ValueError("Scripted acceptance failed.")
    if Decimal(ledger["paid_cny"]) + Decimal(ledger["reserve_cny"]) > Decimal(str(CAP_CNY)):
        raise ValueError("The batch ledger exceeded the cap.")
    write(batch / "result.json", result)
    return {key: result[key] for key in ("scope", "assessed", "assigned", "pairs", "ledger")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    print(execute(args.archive, args.out, args.live))
