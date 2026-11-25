"""Freeze and run a same-prefix feedback experiment or a new-family transfer pilot."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from decimal import Decimal
from pathlib import Path
from random import Random

from aroute.core.billing_meter import BillingMeter
from aroute.core.immutable import sha256_file
from aroute.core.model_pool import ModelResponse

from .research_complete_frozen_trial import bill, load_registry, model_pool, write
from .research_v33_checkpoint_trial import _spec
from .research_v40_cases import _account_case, _service_case
from .research_v40_control import Assessment, feedback
from .research_v40_oracle import candidate_truth
from .research_v40_trial import (
    DATABASE, PROVIDERS, _json_digest, _read_database, _read_json, _script, audit,
    load_models_catalog, run_branch, summarize,
)


CAPS = {"feedback": 0.80, "transfer": 1.50}


def feedback_prefixes(rows):
    selected = {}
    for row in rows:
        prefix = []
        for event in row["trace"]:
            if event["event"] == "auto_read":
                continue
            if event["event"] not in ("dispatch", "future_block"):
                raise ValueError("Cannot reproduce a held or invalid proposal by injecting a valid call.")
            prefix.append(event)
            if event["event"] == "future_block":
                key = _json_digest([row["task"], row["provider"], prefix[:-1], event["candidate"]])
                if key not in selected:
                    selected[key] = {"id": key, "task": row["task"], "provider": row["provider"],
                                     "source_slots": [], "events": prefix}
                selected[key]["source_slots"].append(row["slot"])
                break
    return list(selected.values())


def independent_cases():
    account = (
        ("two-refunds", (-7, -5), (3,), 12, (0, 0), 5, 0, -1),
        ("cross-ticket-funding", (8, -6, -6), (3, 1), 4, (0, 0, 1), 6, 0, 1),
        ("credit-instalments", (-4, 8, -4), (4,), 3, (0, 0, 0), 4, 1, 2),
        ("paired-ticket-caps", (-3, 3, -4, 2), (2, 2), 7, (0, 0, 1, 1), 4, 0, -1),
    )
    cases = []
    for index, (tag, residual, quotas, headroom, owners, cap, idx, delta) in enumerate(account, start=1):
        case = _account_case(f"independent-{index:02d}", residual, quotas, headroom, (0,) * len(residual),
                             owners, cap, idx, delta)
        case["family"] = tag
        cases.append(case)
    service = (
        ("diamond", (0, 0, 0, 0), (True, False, False, False), (1, 1, 1, 1),
         ((), ((0, 1),), ((0, 1),), ((1, 1), (2, 1))), (1, 1, 1, 1), (0, 0, 0, 0), 0),
        ("parallel-handoffs", (0, 0, 0, 0), (True, False, True, False), (1, 1, 1, 1),
         ((), ((0, 1),), (), ((2, 1),)), (1, 1, 1, 1), (0, 0, 0, 0), 0),
        ("existing-upstream", (1, 0, 0), (True, True, False), (1, 1, 1),
         ((), ((0, 0),), ((0, 0), (1, 1))), (1, 1, 1), (1, 0, 0), 1),
        ("mixed-quota-paths", (0, 0, 0, 0), (True, False, True, False), (2, 1, 1, 1),
         ((), ((0, 1),), (), ((1, 1), (2, 1))), (1, 1, 1, 1), (0, 0, 0, 0), 2),
    )
    for index, (tag, *args) in enumerate(service, start=len(cases) + 1):
        case = _service_case(f"independent-{index:02d}", *args)
        case["family"] = tag
        cases.append(case)
    for case in cases:
        if candidate_truth(case) != (True, False):
            raise ValueError(f"Invalid prespecified transfer fixture {case['task']['task_id']}.")
        case["quota_group"] = "tight"
    return cases


class PrefixPool:
    """Supply archived actions before the checkpoint, then delegate to the identical model."""

    def __init__(self, prefix, delegate):
        self.prefix = prefix
        self.delegate = delegate
        self.first_prompt_hash = None
        self.feedback_text = None

    def call_with_tools(self, *, step_idx, provider, tier, messages, **kwargs):
        if step_idx < len(self.prefix):
            call = _spec(self.prefix[step_idx]["candidate"], f"checkpoint_{step_idx}")
            return ModelResponse(provider=provider, tier=tier, model_requested="archived", model_actual="archived",
                                 content="", reasoning_content="", tool_calls=[call], prompt_tokens=0,
                                 cache_hit_tokens=0, completion_tokens=0, reasoning_tokens=0, latency_ms=0,
                                 cost_cny=0.0, pricing_period="offline", finish_reason="tool_calls")
        if step_idx == len(self.prefix):
            self.first_prompt_hash = _json_digest(messages[:-1])
            self.feedback_text = messages[-1]["content"]
        return self.delegate.call_with_tools(step_idx=step_idx, provider=provider, tier=tier, messages=messages, **kwargs)


def register(mode, frozen, prior):
    parent, source_cases, archive, _, _ = load_registry(frozen, prior)
    if mode == "feedback":
        intervened = [row for row in archive["rows"] if any(event["event"] == "future_block" for event in row["trace"])]
        prefixes = feedback_prefixes(intervened)
        cases = [source_cases[task] for task in sorted({prefix["task"] for prefix in prefixes})]
        groups = [(prefix["task"], prefix["provider"], repeat, prefix["id"]) for prefix in prefixes for repeat in (1, 2)]
        arms = ("M1", "M2")
    else:
        prefixes = []
        cases = independent_cases()
        groups = [(case["task"]["task_id"], provider, 1, None) for case in cases for provider in PROVIDERS]
        arms = ("M0", "M1")
    rng = Random(20261124)
    rng.shuffle(groups)
    jobs = []
    for task, provider, repeat, prefix in groups:
        for arm in rng.sample(arms, len(arms)):
            jobs.append({"slot": len(jobs) + 1, "task": task, "provider": provider, "repeat": repeat,
                         "arm": arm, "prefix": prefix})
    return {
        "phase": mode, "frozen_parent": _json_digest(parent), "source_log_sha256": sha256_file(prior / "result.json"),
        "runner_sha256": sha256_file(Path(__file__)), "models": parent["models"], "budget_cny": CAPS[mode],
        "arms": arms, "cases": cases, "prefixes": prefixes, "allocation": jobs,
        "result_unit": "prefix" if mode == "feedback" else "independent family",
    }


def compare(rows, mode):
    blocks = {}
    for row in rows:
        key = (row["task"], row["provider"], row["repeat"], row["prefix"])
        blocks.setdefault(key, {})[row["arm"]] = row["terminal"]
    before, after = ("M1", "M2") if mode == "feedback" else ("M0", "M1")
    return Counter(f"{block[before]} to {block[after]}" for block in blocks.values())


def execute(mode, frozen, prior, out, budget=None):
    protocol = register(mode, frozen, prior)
    digest = _json_digest(protocol)
    out.mkdir(parents=True, exist_ok=True)
    registration = out / "registration.json"
    if registration.exists() and _json_digest(_read_json(registration)) != digest:
        raise ValueError("The registered experiment changed.")
    write(registration, protocol)
    offline = budget is None
    if not offline:
        acceptance = _read_json(out / "offline" / "result.json")
        if (not 0 < budget <= CAPS[mode] or acceptance["registration_sha256"] != digest or
                acceptance["scope"] != "scripted" or acceptance["assessed"] != len(protocol["allocation"])):
            raise ValueError("A matching scripted acceptance and approved cap are required.")
    batch = out / ("offline" if offline else "live")
    batch.mkdir(exist_ok=False)
    catalog = load_models_catalog()
    meter = BillingMeter(batch / DATABASE, project_cap_cny=CAPS[mode] if offline else budget)
    live = None if offline else model_pool(catalog, protocol["models"], meter)
    cases = {case["task"]["task_id"]: case for case in protocol["cases"]}
    prefixes = {p["id"]: p["events"] for p in protocol["prefixes"]}

    def run(job):
        case = cases[job["task"]]
        delegate = live or _script(case, catalog, meter)
        prefix = prefixes.get(job["prefix"])
        pool = PrefixPool(prefix, delegate) if prefix else delegate
        row = run_branch(case, job, pool, meter, "formal")
        audit(case, row)
        if prefix:
            observed = [e for e in row["trace"] if e["event"] != "auto_read"][:len(prefix)]
            fields = ("step", "event", "candidate", "assessment", "reply", "oracle")
            actual = [[event.get(k) for k in fields] for event in observed]
            expected = [[event.get(k) for k in fields] for event in prefix]
            if _json_digest(actual) != _json_digest(expected):
                raise ValueError(f"Checkpoint mismatch in slot {job['slot']}.")
            if offline and pool.feedback_text is None:
                raise ValueError(f"No continuation from the checkpoint in slot {job['slot']}.")
            decision = prefix[-1]["assessment"]
            if pool.feedback_text != feedback(Assessment(**replace_obligations(decision)), job["arm"]):
                raise ValueError(f"Unexpected feedback in slot {job['slot']}.")
            row.update(shared_prompt_sha256=pool.first_prompt_hash, remaining_model_decisions=28-len(prefix))
        with closing(_read_database(batch / DATABASE)) as db:
            row.update(bill(list(db.execute("SELECT * FROM api_attempts WHERE run_id=?", (row["run_id"],)))))
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=2) as workers:
        for index in range(0, len(protocol["allocation"]), 2):
            pair = list(workers.map(run, protocol["allocation"][index:index+2]))
            if mode == "feedback" and pair[0]["shared_prompt_sha256"] != pair[1]["shared_prompt_sha256"]:
                raise ValueError("Matched checkpoint prompts differ before delivering feedback.")
            rows.extend(pair)
            write(batch / "checkpoint.json", rows)
    with closing(_read_database(batch / DATABASE)) as db:
        ledger = bill(list(db.execute("SELECT * FROM api_attempts")))
    if not offline and Decimal(ledger["paid_cny"]) + Decimal(ledger["reserve_cny"]) > Decimal(str(budget)):
        raise ValueError("Experiment ledger exceeded its cap.")
    result = {
        "scope": "scripted" if offline else "live", "registration_sha256": digest,
        "assessed": sum(r["terminal"] is not None for r in rows), "assigned": len(rows), "ledger": ledger,
        "summary": summarize(rows), "pairs": compare(rows, mode), "rows": rows,
    }
    if offline and not all(r["terminal"] for r in rows):
        raise ValueError("Not all scripted paths finished.")
    write(batch / "result.json", result)
    return result


def replace_obligations(decision):
    # JSON converts tuple-valued witnesses to lists. Runtime feedback uses tuples.
    return {**decision, "obligations": tuple(tuple(item) for item in decision["obligations"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=CAPS)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget", type=float)
    args = parser.parse_args()
    result = execute(args.mode, args.frozen, args.prior, args.out, args.budget)
    print({key: val for key, val in result.items() if key not in ("rows", "summary")})


if __name__ == "__main__":
    main()
