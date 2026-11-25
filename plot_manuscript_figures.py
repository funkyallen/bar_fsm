"""Draw the three manuscript figures from the released audit results."""

import argparse
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from audit_experimental_results import qualification as audit_qualification


ROOT = Path(__file__).resolve().parent
WIDTH_MM = 181.864
WIDTH = WIDTH_MM / 25.4
INK = "#202A33"
NEUTRAL = "#5E6C7A"
PALE = "#DCE3EA"
TEAL = "#007782"
RUST = "#A04C37"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def collect_data(audit_dir):
    editorial = load(audit_dir / "editorial_checks.json")
    formal = load(audit_dir / "formal_completion_audit.json")["pooled"]
    accounting = load(audit_dir / "shared_allowance_audit.json")
    accounts = accounting["relaxed_demand_checks"]
    lost = sum(row["exact_relaxed_demand"] > row["actual_successor_quota"] for row in accounts)
    summary = formal["summary"]["all"]
    contrasts = []
    for name, contrast in formal["paired"].items():
        counts = contrast["pair_counts"]
        n = sum(counts.values())
        contrasts.append({"contrast": name, "matched_groups": n, "families": contrast["families"],
                          "estimate_pp": (counts["treatment_only"] - counts["baseline_only"]) / n * 100,
                          "interval_95_pp": contrast["family_equal_bootstrap_95_pp"], "pair_counts": counts})
    return {
        "classification": editorial["constructed_sample_classification"],
        "formal": {"restarts_by_policy": [
            {"arm": "M0", "path": "Actual dispatch", "losses": summary["M0"]["actual_reachability_losses"]},
            {"arm": "M1", "path": "Replay of hold", "losses": summary["M1"]["true_future_blocks"]},
            {"arm": "M2", "path": "Replay of hold", "losses": summary["M2"]["true_future_blocks"]},
        ], "contrasts": contrasts},
        "accounting": {"fixed_new_losses": lost, "fixed_preserving": len(accounts) - lost,
                       "fixed_policy_counts": accounting["fixed_candidate_counts"],
                       "natural": editorial["common_candidate_comparison"]},
    }


def configure_style():
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 9, "text.color": INK, "axes.labelcolor": INK,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
                         "axes.linewidth": 0.7, "axes.edgecolor": NEUTRAL,
                         "xtick.color": INK, "ytick.color": INK, "xtick.major.size": 3,
                         "ytick.major.size": 0, "svg.fonttype": "none", "pdf.fonttype": 42})


def case_classification(data, qualification):
    c = data["classification"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(WIDTH, 2.12))
    fig.subplots_adjust(left=0.21, right=0.97, bottom=0.25, top=0.73, wspace=0.56)
    panel_heading(left, "a", "Backend classification")
    panel_heading(right, "b", "Effect of authoritative receipts")
    labels = ("Locally inadmissible", "Already unreachable", "New loss", "Preserving")
    counts = (c["oracle_ineligible"], c["initially_unreachable"],
              c["new_loss"], c["preserving"])
    left.barh(range(4), counts, color=(NEUTRAL, NEUTRAL, TEAL, TEAL), height=0.56)
    for y, count in enumerate(counts):
        left.annotate(str(count), (count, y), (4, 0), textcoords="offset points", va="center")
    left.set(yticks=range(4), yticklabels=labels, ylim=(3.62, -0.62),
             xlim=(0, 428), xticks=(0, 100, 200, 300, 400), xlabel="State–action pairs")

    transitions = qualification["same_logic_receipt_transitions"]
    eligible = (transitions["new_loss"], transitions["preserved"])
    dispositions = ("certificate_hold", "no_certified_deficit", "resource_unresolved")
    changes = [sum(row.get(f"resource_unresolved -> {d}", 0) for row in eligible)
               for d in dispositions]
    outcomes = ("Certified deficit", "No certified deficit", "Still unresolved")
    right.barh(range(3), changes, color=(TEAL, "#A4D3D4", PALE), height=.49,
               edgecolor=(TEAL, "#A4D3D4", NEUTRAL), linewidth=.7)
    for y, count in enumerate(changes):
        right.annotate(str(count), (count, y), (4, 0), textcoords="offset points", va="center")
    right.set(yticks=range(3), yticklabels=outcomes, ylim=(2.55, -.55),
              xlim=(0, 40), xticks=(0, 10, 20, 30, 40), xlabel="Previously unresolved candidates")
    return fig


def panel_heading(ax, letter, title):
    ax.text(0, 1.13, letter, transform=ax.transAxes, va="baseline", fontweight="bold", fontsize=11)
    ax.annotate(title, xy=(0, 1.13), xycoords="axes fraction", xytext=(16, 0),
                textcoords="offset points", va="baseline", fontsize=10, fontweight="bold")


def formal_study(data):
    fig, (left, right) = plt.subplots(1, 2, figsize=(WIDTH, 2.60))
    fig.subplots_adjust(left=0.17, right=0.96, bottom=0.25, top=0.79, wspace=0.48)
    panel_heading(left, "a", "Confirmed reachability-loss candidates")
    positions = [0, 1, 2]
    restarts = data["restarts_by_policy"]
    for y, row in zip(positions, restarts, strict=True):
        value = row["losses"]
        style = {"color": RUST} if row["arm"] == "M0" else {"color": PALE, "edgecolor": TEAL, "linewidth": 1.0}
        left.barh(y, value, height=0.54, **style)
        left.annotate(str(value), xy=(value, y), xytext=(5, 0), textcoords="offset points", va="center")
    left.set_yticks(positions, [f"{row['arm']}\n{row['path']}" for row in restarts])
    left.set_ylim(2.57, -0.57)
    left.set_xlim(0, 7)
    left.set_xticks([0, 2, 4, 6])
    left.set_xlabel("Restart candidates")

    panel_heading(right, "b", "Completion contrasts")
    right.axvline(0, color=NEUTRAL, linewidth=0.75, linestyle=(0, (3, 3)), zorder=0)
    contrasts = data["contrasts"]
    for y, row in zip(positions, contrasts, strict=True):
        estimate = row["estimate_pp"]
        low, high = row["interval_95_pp"]
        shade = TEAL if row["contrast"] == "M2-M0" else NEUTRAL
        right.errorbar(estimate, y, xerr=[[estimate - low], [high - estimate]], fmt="o", color=shade,
                       capsize=3, elinewidth=1.35, markersize=4.8, zorder=2)
    right.set_yticks(positions, [r["contrast"].replace("-", " − ") for r in contrasts])
    right.set_ylim(2.57, -0.57)
    right.set_xlim(-8, 18)
    right.set_xticks([-5, 0, 5, 10, 15])
    right.set_xlabel("Paired difference (percentage points)")
    return fig


def accounting_study(data):
    fig, (left, right) = plt.subplots(1, 2, figsize=(WIDTH, 2.65), sharey=True)
    fig.subplots_adjust(left=0.23, right=0.97, bottom=0.25, top=0.79, wspace=0.27)
    policies = {"B0": "Local checks (B0)", "B1": "Invoice count (B1)",
                "B2": "Largest posting (B2)", "certificate": "Capacity-aware certificate"}
    panels = [(left, "a", "Loses reachability", "true_blocks", data["fixed_new_losses"]),
              (right, "b", "Preserves reachability", "preserving_blocks", data["fixed_preserving"])]
    for ax, letter, heading, metric, n in panels:
        panel_heading(ax, letter, heading)
        for y, policy in enumerate(policies):
            count = data["fixed_policy_counts"][policy][metric]
            rate = count / n
            shade = TEAL if policy == "certificate" else RUST if policy == "B2" else NEUTRAL
            marker = "D" if policy == "certificate" else "o"
            if count:
                ax.hlines(y, 0, rate, color=shade, linewidth=1.15)
            ax.plot(rate, y, marker=marker, markersize=5, markeredgecolor=shade,
                    markerfacecolor=shade if count else "white", linestyle="none")
            ax.annotate(f"{count}/{n}", xy=(rate, y), xytext=(6, 0), textcoords="offset points", va="center")
        ax.set_xlim(-0.04, 1.25)
        ax.set_xticks([0, 0.5, 1])
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.set_xlabel("Candidates held")
    left.set_yticks(range(4), list(policies.values()))
    left.set_ylim(3.59, -0.59)
    right.tick_params(labelleft=False)
    return fig


def audit_alignment(fig, base, external_tool):
    fig.canvas.draw()
    if len(fig.axes) == 1:
        return
    if external_tool:
        spec = spec_from_file_location("audit_panel_alignment", external_tool)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        module.require_matplotlib_panel_alignment(fig, json_out=base.with_suffix(".alignment.json"),
                                                  tolerance_pt=1.5, gutter_tolerance_pt=1.5,
                                                  require_panel_labels=True, strict=True)
        return
    # Portable check of the rendered plot rectangles in each two-panel row.
    px_to_pt = 72 / fig.dpi
    left, right = (ax.get_window_extent() for ax in fig.axes)
    differences = [abs(a - b) * px_to_pt for a, b in [(left.width, right.width),
                                                      (left.y0, right.y0), (left.y1, right.y1)]]
    if max(differences) > 1.5 or not left.x1 < right.x0:
        raise ValueError("Rendered plot areas do not align within 1.5 pt.")


def save_figure(fig, base, alignment_tool):
    audit_alignment(fig, base, alignment_tool)
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".svg"))
    fig.savefig(base.with_suffix(".png"), dpi=600)
    fig.savefig(base.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=ROOT / "figures", type=Path)
    parser.add_argument("--audit-dir", default=ROOT, type=Path)
    parser.add_argument("--alignment-tool", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    configure_style()
    data = collect_data(args.audit_dir)
    (args.out / "figure_data.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    qualification = audit_qualification(load(args.audit_dir / "voidability_720.json"))
    save_figure(case_classification(data["classification"], qualification),
                args.out / "fig1_classification", args.alignment_tool)
    save_figure(formal_study(data["formal"]), args.out / "fig2_formal_study", args.alignment_tool)
    save_figure(accounting_study(data["accounting"]), args.out / "fig3_accounting", args.alignment_tool)


if __name__ == "__main__":
    main()
