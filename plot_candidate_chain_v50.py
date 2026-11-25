"""Visualize the shared recorded proposal and separate paths in formal slots 10–12."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from plot_manuscript_figures import INK, NEUTRAL, RUST, TEAL, load


ROOT = Path(__file__).resolve().parent


def collect(audits):
    rows = {r["slot"]: r for r in load(ROOT / "live/formal_completion/result.json")["rows"]}
    cases = {r["task"]["task_id"]: r["task"] for r in load(ROOT / "formal_fixtures/cases.json")}
    forks = {(r["slot"], r["step_zero_based"]): r
             for r in load(audits / "strict_replay.json")["withheld_dispatch_forks"]}
    witnesses = {(r["slot"], r["step_zero_based"]): r
                 for r in load(audits / "oracle_v50.json")["formal_service_losses"]["cases"]}
    paths = []
    for slot in (11, 12, 10):
        row = rows[slot]
        topology, read, event = row["trace"][:3]
        if topology["candidate"]["name"] != "get_topology" or read["event"] != "auto_read":
            raise ValueError(f"Unexpected public prefix at slot {slot}.")
        service = read["reply"]["data"]["service"]
        target = cases[row["task"]]["goal"]["running_revisions"][service["svc_id"]]
        shared = {"candidate": event["candidate"], "staged": service["staged_rev"], "goal": target,
                  "initial_allowance": service["max_restarts"] - service["restart_count"],
                  "successor_allowance": event["assessment"]["q_after"], "demand": event["assessment"]["lower"]}
        if paths and shared != previous:
            raise ValueError("The three displayed paths do not share the recorded candidate and decisive public facts.")
        previous = shared
        verified = witnesses[(slot, event["step"])]
        if not (verified["predecessor"]["native_scorer_passed"] and not verified["successor"]["reachable"]):
            raise ValueError("The audited predecessor witness or successor label changed.")
        if event["event"] == "dispatch":
            observed = row["trace"][3]["reply"]["data"]["service"]
            running = event["reply"]["data"]["running_rev"]
            after = observed["max_restarts"] - observed["restart_count"]
            actual = after
            path = "Observed dispatch"
        else:
            fork = forks[(slot, event["step"])]
            running = fork["forked_response"]["running_rev"]
            after = fork["forked_allowance_after_dispatch"]
            actual = fork["actual_allowance_preserved"]
            path = "Isolated replay"
        paths.append({"slot": slot, "policy": row["arm"], "online_disposition": event["event"],
                      "path": path, "execution_running_revision": running, "execution_allowance_after": after,
                      "original_allowance_after": actual, "original_completed": row["terminal"],
                      "reachability_after_execution": verified["successor"]["reachable"]})
    return {"common_candidate_and_facts": shared, "paths": paths}


def draw(data):
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "text.color": INK,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(7.16, 2.90))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    shared = data["common_candidate_and_facts"]
    ax.add_patch(Rectangle((.025, .758), .95, .216, facecolor="#EFF5F6", edgecolor="none"))
    ax.text(.048, .92, "Same recorded first restart proposal and decisive receipts",
            weight="bold", fontsize=10, va="center")
    ax.text(.048, .855, f"Upstream service staged on {shared['staged']}, target {shared['goal']}, "
            f"{shared['initial_allowance']} restart left. Projected allowance {shared['successor_allowance']} "
            f"< required {shared['demand']}.", fontsize=8.7, va="center")
    ax.text(.048, .795, "Offline audit: each predecessor had a native completion witness.", fontsize=8.7, va="center")
    x = (.048, .235, .440, .752)
    for pos, label in zip(x, ("Policy and slot", "Online decision", "Candidate execution and audit", "Original episode")):
        ax.text(pos, .712, label, color=NEUTRAL, size=8.1, weight="bold", va="center")
    for y, row in zip((.63, .435, .24), data["paths"], strict=True):
        ax.plot((.025, .975), (y+.04, y+.04), lw=.55, color="#C8D1D6")
        replay = row["path"] == "Isolated replay"
        result = "Completed" if row["original_completed"] else "Did not complete"
        ax.text(x[0], y, f"{row['policy']}  ·  {row['slot']}", weight="bold", size=9, va="top")
        ax.text(x[1], y, "Hold" if replay else "Dispatch (shadow)", size=9, va="top")
        ax.text(x[2], y, row["path"], color=TEAL if replay else RUST, weight="bold", size=9, va="top")
        ax.text(x[2], y-.061, f"{row['execution_running_revision'].capitalize()} running, "
                f"{row['execution_allowance_after']} left", size=8.35, va="top")
        ax.text(x[2], y-.115, "Offline: goal unreachable" if not row["reachability_after_execution"] else "Offline: goal reachable",
                size=8.35, va="top")
        disposition = "hold" if replay else "dispatch"
        ax.text(x[3], y, f"After {disposition}: {row['original_allowance_after']} left", size=9, va="top")
        ax.text(x[3], y-.061, result, weight="bold" if row["original_completed"] else "normal", size=8.35, va="top")
    ax.text(.048, .037, "The slots use independently sampled model replies.",
            color=NEUTRAL, size=8.1, va="center")
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    options = parser.parse_args()
    options.out.mkdir(parents=True, exist_ok=True)
    data = collect(options.audit_dir)
    (options.out / "candidate_chain_data.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    figure = draw(data)
    figure.savefig(options.out / "fig2_candidate_chain.pdf")
    figure.savefig(options.out / "fig2_candidate_chain.svg")
    figure.savefig(options.out / "fig2_candidate_chain.png", dpi=600)
    figure.savefig(options.out / "fig2_candidate_chain.tiff", dpi=600)
    plt.close(figure)
