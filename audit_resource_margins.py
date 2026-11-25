"""Recompute the qualification transitions and accounting margins from archived candidates."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path


CONDITIONS = ("archived", "conservative_without_authority", "conservative_with_public_receipt")
DISPOSITION = {
    ("unknown", "unknown"): "ED", ("block", "block"): "LR", ("pass", "block"): "CD",
    ("pass", "pass_checks"): "ND", ("pass", "unknown"): "UR",
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def qualification_transitions(rows):
    transitions = {"A_to_B": defaultdict(Counter), "B_to_C": defaultdict(Counter)}
    for row in rows:
        before, after = row["truth_before"], row["truth_after"]
        label = ("inadmissible" if after is None else "already_unreachable" if not before
                 else "preserving" if after else "new_loss")
        a, b, c = (DISPOSITION[row["decisions"][key]["local"], row["decisions"][key]["result"]]
                   for key in CONDITIONS)
        transitions["A_to_B"][label][f"{a} -> {b}"] += 1
        transitions["B_to_C"][label][f"{b} -> {c}"] += 1
    return transitions


def margin_record(cohort, task, run_id, step, candidate, guard, blocked, reachable, quota, cap):
    args = candidate["arguments"]
    obligations = guard["obligations"]
    demand = sum((abs(residual) + cap - 1) // cap for _, residual in obligations)
    count_margin = quota - len(obligations)
    capacity_margin = quota - demand
    if (guard["local"] != "pass" or guard["q_after"] != quota or guard["lower"] != demand
            or blocked["B1"] != (count_margin < 0) or blocked["certificate"] != (capacity_margin < 0)):
        raise ValueError(f"Archived margins or verdicts differ for {task}, {run_id}, {step}, {args}.")
    region = "both_hold" if count_margin < 0 else "capacity_only_hold" if capacity_margin < 0 else "neither_holds"
    return {
        "cohort": cohort, "task": task, "run_id": run_id, "step": step, "invoice": args["inv_id"],
        "posting_cents": args["delta_cents"], "cap_cents": cap, "successor_allowance": quota,
        "count_bound": len(obligations), "capacity_bound": demand,
        "count_margin": count_margin, "capacity_margin": capacity_margin, "region": region,
        "resource_verdict": guard["result"], "successor_reachable": reachable,
        "largest_posting_hold": blocked["B2"],
    }


def collect(root):
    archive = root / "live/shared_allowance"
    fixtures = {row["task"]["task_id"]: row for row in read(archive / "cases.json")}
    independent = read(root / "shared_allowance_audit.json")["relaxed_demand_checks"]
    fixed = read(archive / "fixed_candidates.json")
    rows = []
    for saved, audit in zip(fixed, independent, strict=True):
        task = saved["task"]
        quota, = fixtures[task]["truth"]["quotas"]
        cap = fixtures[task]["task"]["goal"]["max_per_post_cents"]
        row = margin_record("fixed", task, "", "", saved["candidate"], saved["guard"], saved["blocked"],
                            saved["successor_reachable"], quota - 1, cap)
        if (audit["task"], audit["candidate"], audit["actual_successor_quota"], audit["proved_lower_bound"]) != (
                task, saved["candidate"], row["successor_allowance"], row["capacity_bound"]):
            raise ValueError(f"Independent accounting audit differs for {task}.")
        rows.append(row)

    episodes = read(archive / "live/result.json")["rows"]
    for episode in episodes:
        task = episode["task"]
        cap = fixtures[task]["task"]["goal"]["max_per_post_cents"]
        for event in episode["trace"]:
            if event.get("oracle") is None:
                continue
            bundle = next(item["receipt"]["data"]["ticket"] for item in event["receipts"]
                          if item["tool"]["name"] == "get_invoice_bundle")
            quota = bundle["max_adjustments"] - bundle["adj_count"] - 1
            row = margin_record("model_generated", task, episode["run_id"], event["step"], event["candidate"],
                                event["guard"], event["shadow_blocks"], event["oracle"]["after_if_dispatched"],
                                quota, cap)
            rows.append(row)
    return rows


def summarize(rows):
    result = {}
    for cohort in ("fixed", "model_generated"):
        grouped = defaultdict(Counter)
        for row in rows:
            if row["cohort"] != cohort:
                continue
            for counts in (grouped[row["task"]], grouped["total"]):
                counts["candidates"] += 1
                counts[row["region"]] += 1
                counts["new_losses"] += not row["successor_reachable"]
                counts["largest_posting_holds_preserving"] += row["successor_reachable"] and row["largest_posting_hold"]
                counts["unresolved"] += row["resource_verdict"] == "unknown"
        result[cohort] = dict(sorted(grouped.items()))
    return result


def draw(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 9, "pdf.fonttype": 42, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(7.0, 2.8))
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.25, top=0.83, wspace=0.23)
    for ax, cohort, title, letter in zip(axes, ("fixed", "model_generated"),
                                        ("Fixed candidates", "Model-generated candidates"), ("a", "b"), strict=True):
        counts = Counter((r["count_margin"], r["capacity_margin"], r["successor_reachable"])
                         for r in rows if r["cohort"] == cohort)
        # Cell edges separate the integer margins at which the verdict changes.
        ax.add_patch(Rectangle((-0.5, -1.54), 4.06, 1.04, color="#fae6df", lw=0, zorder=0))
        ax.axhline(-0.5, color="#aebac2", lw=0.7)
        ax.axvline(-0.5, color="#aebac2", lw=0.7)
        for (x, y, reachable), n in sorted(counts.items()):
            ax.scatter(x, y, s=280, color="#007782" if reachable else "#a04c37", zorder=3)
            ax.text(x, y, str(n), color="white", ha="center", va="center", fontsize=8, zorder=4)
        ax.set_title(f"{title} (n = {sum(counts.values())})", fontsize=10, pad=10)
        ax.text(-0.10, 1.11, letter, transform=ax.transAxes, fontweight="bold", fontsize=12)
        ax.set(xlim=(-1.55, 3.55), ylim=(-1.55, 1.55), xticks=range(-1, 4), yticks=range(-1, 2))
        ax.set_xlabel("Invoice-count margin (uses)", labelpad=4)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(labelleft=True, length=3)
    axes[0].set_ylabel("Capacity margin (uses)")
    fig.text(0.12, 0.055, "Reachability preserved", color="#007782", fontsize=9)
    fig.text(0.38, 0.055, "Reachability lost", color="#a04c37", fontsize=9)
    fig.text(0.59, 0.055, "Shaded area: capacity-only holds", color="#61473f", fontsize=9)
    return fig


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--figures", type=Path)
    parser.add_argument("--check-alignment", action="store_true",
                        help="Run the rendered panel audit when audit_panel_alignment is installed.")
    args = parser.parse_args()
    rows = collect(args.archive)
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "accounting_resource_margins.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    data = {
        "accounting": summarize(rows),
        "qualification_transitions": qualification_transitions(read(args.archive / "voidability_720.json")["rows"]),
    }
    (args.out / "resource_margin_audit.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    if args.figures:
        args.figures.mkdir(parents=True, exist_ok=True)
        figure = draw(rows)
        base = args.figures / "figS3_resource_margins"
        if args.check_alignment:
            from audit_panel_alignment import require_matplotlib_panel_alignment
            require_matplotlib_panel_alignment(figure, json_out=base.with_suffix(".alignment.json"))
        figure.savefig(base.with_suffix(".pdf"))
        figure.savefig(base.with_suffix(".svg"))
        figure.savefig(base.with_suffix(".png"), dpi=600)
        figure.savefig(base.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})


if __name__ == "__main__":
    main()
