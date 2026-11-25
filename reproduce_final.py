"""Recompute the final manuscript's shared-candidate, cost, and native scoring checks."""

import argparse
from collections import Counter
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parent
RETAIL_COMMIT = "59a200c6d575d595120f1cb70fea53cef0632f6b"
RETAIL_SHA256 = "f99f4c96270825edf0bd2da36157d4d183cc8f1a9925158c344fa5cec0a60919"
POLICIES = ("B0", "B1", "B2")
SHADOW_LABEL = {
    (False, False): "both_allow",
    (True, True): "both_hold",
    (True, False): "baseline_only_holds",
    (False, True): "certificate_only_holds",
}


def load(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def common_candidates(raw):
    pairs = {policy: dict.fromkeys(SHADOW_LABEL.values(), 0) for policy in POLICIES}
    scope = Counter()
    by_origin = Counter()
    separating_losses = 0
    for episode in raw["rows"]:
        for event in episode["trace"]:
            if event.get("candidate", {}).get("name") != "post_adjustment" or event.get("guard", {}).get("local") != "pass":
                continue
            scope["locally_cleared"] += 1
            by_origin[episode["arm"]] += 1
            flags = event["shadow_blocks"]
            if set(flags) != {*POLICIES, "certificate"} or any(type(flag) is not bool for flag in flags.values()):
                raise ValueError("A locally cleared posting is missing a comparable policy decision.")
            for policy in POLICIES:
                pairs[policy][SHADOW_LABEL[flags[policy], flags["certificate"]]] += 1
            truth = event["oracle"]
            scope["feasible_prefix"] += truth["before"] is True
            scope["preserving"] += truth["before"] is True and truth["after_if_dispatched"] is True
            new_loss = truth["before"] is True and truth["after_if_dispatched"] is False
            scope["new_loss"] += new_loss
            separating_losses += new_loss and flags["certificate"] and not flags["B0"]
    if scope["locally_cleared"] != 177 or any(sum(table.values()) != 177 for table in pairs.values()):
        raise ValueError("The common-candidate denominator changed.")
    return {"episodes": len(raw["rows"]), "scope": dict(scope), "counts_by_trajectory_policy": dict(by_origin),
            "paired_shadow_decisions": pairs, "new_losses_proved_beyond_B0": separating_losses}


def costs(formal):
    cohorts = formal["pooled"]["summary"]["all"]
    rows = {}
    for arm, cell in cohorts.items():
        charge = Decimal(cell["cost_cny"])
        rows[arm] = {"successes": cell["success"], "assigned": cell["assigned"], "cost_cny": str(charge),
                     "completion_rate_pct": str(Decimal(cell["success"]) * 100 / cell["assigned"]),
                     "cny_per_success": str(charge / cell["success"])}
    base, treatment = rows["M0"], rows["M2"]
    return {"arms": rows, "M2_minus_M0": {
        "completion_pp": str(Decimal(treatment["completion_rate_pct"]) - Decimal(base["completion_rate_pct"])),
        "relative_charge_pct": str((Decimal(treatment["cost_cny"]) / Decimal(base["cost_cny"]) - 1) * 100),
        "relative_cny_per_success_pct": str((Decimal(treatment["cny_per_success"]) / Decimal(base["cny_per_success"]) - 1) * 100),
    }}


def classify_fixtures(oracle):
    totals = Counter()
    for row in oracle["cases"]:
        before, after = row["truth_before"], row["truth_after"]
        if after is None:
            totals["oracle_ineligible"] += 1
        elif before is False:
            totals["initially_unreachable"] += 1
        elif after is False:
            totals["new_loss"] += 1
        else:
            totals["preserving"] += 1
    labels = Counter(row["label"] for row in oracle["cases"])
    return {"classification": dict(totals), "resolution": {
        "certified_new_losses": labels["true_block"],
        "unresolved_new_losses": totals["new_loss"] - labels["true_block"],
        "explicit_preserving_passes": labels["true_pass"],
        "unresolved_preserving": totals["preserving"] - labels["true_pass"] - labels["false_block"],
        "preserving_false_blocks": labels["false_block"],
    }}


def native_scores(archive):
    if sha256(archive.read_bytes()).hexdigest() != RETAIL_SHA256:
        raise ValueError("The packaged upstream archive differs from the pinned download.")
    with TemporaryDirectory(prefix="bar-fsm-retail-") as directory:
        with ZipFile(archive) as z:
            z.extractall(directory)
        source = Path(directory) / ("tau-bench-" + RETAIL_COMMIT)
        sys.path.insert(0, str(source))
        shim = ModuleType("litellm")

        def forbid_model_call(*_args, **_kwargs):
            raise RuntimeError("This replay does not use a user simulator or any model request.")

        shim.completion = forbid_model_call
        sys.modules["litellm"] = shim
        try:
            from tau_bench.envs.retail import MockRetailDomainEnv

            rows = []
            for task, skip in ((64, None), (64, "exchange_delivered_order_items"), (106, None), (106, "all_actions")):
                env = MockRetailDomainEnv(task_index=task, user_strategy="human")
                initial = env.get_data_hash()
                if env.task.outputs:
                    raise ValueError("These controls require tasks without text scoring.")
                writes = []
                for action in env.task.actions:
                    if skip == "all_actions" or action.name in ("respond", skip):
                        continue
                    reply = env.step(action).observation
                    if action.name in ("exchange_delivered_order_items", "modify_pending_order_items"):
                        rejected = reply.startswith("Error:")
                        writes.append({"tool": action.name, "backend_ok": not rejected, "error": reply if rejected else None})
                changed = env.get_data_hash() != initial
                rows.append({"task": task, "omitted": skip, "writes": writes,
                             "data_changed": changed, "native_score": env.calculate_reward().reward})
        finally:
            sys.path.remove(str(source))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    source = ROOT / "upstream" / (RETAIL_COMMIT + ".zip")
    report = {
        "common_candidate_comparison": common_candidates(load("live/shared_allowance/live/result.json")),
        "formal_cost_points": costs(load("formal_completion_audit.json")),
        "constructed_sample_classification": classify_fixtures(load("oracle_results.json")),
    }
    scores = native_scores(source)
    if scores != load("retail_scoring_replay.json"):
        raise ValueError("The native scores or physical mutations differ from the published replay.")
    published = ROOT / "editorial_checks.json"
    if published.exists() and report != load(published.name):
        raise ValueError("The common-candidate, cost or fixture results differ from the published checks.")
    for name, payload in (("editorial_checks.json", report), ("retail_scoring_replay.json", scores)):
        (args.out / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Recomputed the 177 matched decisions, formal cost estimates, the fixture counts, and four native Retail scores.")


if __name__ == "__main__":
    main()
