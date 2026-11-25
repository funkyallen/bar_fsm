"""Visualize the shared recorded proposal and separate paths in formal slots 10–12."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_manuscript_figures import INK, NEUTRAL, RUST, TEAL, WIDTH, configure_style, load


ROOT = Path(__file__).resolve().parent


def collect(audits, root=ROOT):
    rows = {r["slot"]: r for r in load(root / "live/formal_completion/result.json")["rows"]}
    cases = {r["task"]["task_id"]: r["task"] for r in load(root / "formal_fixtures/cases.json")}
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
        points = [
            {"decision": 1, "allowance": shared["initial_allowance"], "demand": shared["demand"]},
            {"decision": event["step"] + 1, "allowance": actual, "demand": shared["demand"]},
        ]
        if row["arm"] == "M2":
            service_calls = [e for e in row["trace"] if e["step"] > event["step"] and
                             e["candidate"]["arguments"].get("svc_id") == service["svc_id"]
                             and e["event"] == "dispatch" and e["reply"]["ok"]]
            staged = next(e for e in service_calls if e["candidate"]["name"] == "rollback_config")
            restart = next(e for e in service_calls if e["candidate"]["name"] == "restart_service")
            assert staged["reply"]["data"]["active_rev"] == target
            assert restart["reply"]["data"]["running_rev"] == target
            assert restart["assessment"]["lower"] == 0 and restart["assessment"]["q_after"] == 0
            assert staged["step"] < restart["step"]
            points.extend([
                {"decision": staged["step"] + 1, "allowance": actual, "demand": shared["demand"]},
                {"decision": restart["step"] + 1, "allowance": restart["assessment"]["q_after"],
                 "demand": restart["assessment"]["lower"]},
            ])
        paths.append({"slot": slot, "policy": row["arm"], "online_disposition": event["event"],
                      "path": path, "execution_running_revision": running, "execution_allowance_after": after,
                      "original_allowance_after": actual, "original_completed": row["terminal"],
                      "reachability_after_execution": verified["successor"]["reachable"], "original_points": points})
    return {"common_candidate_and_facts": shared, "paths": paths}


def draw(data):
    configure_style()
    demand_color = "#354A6C"
    fig, axes = plt.subplots(2, 3, figsize=(WIDTH, 2.87), sharex=True, sharey=True)
    fig.subplots_adjust(left=.105, right=.982, bottom=.31, top=.77, wspace=.28, hspace=.26)
    fig.text(.105, .947, "Same proposal at decision 2: restart with the legacy revision staged",
             weight="bold", fontsize=9.8)
    titles = ("M0  Shadow", "M1  Generic feedback", "M2  Evidence and action cue")
    for column, (title, row) in enumerate(zip(titles, data["paths"], strict=True)):
        points = row["original_points"]
        x, quota, demand = ([point[key] for point in points] for key in ("decision", "allowance", "demand"))
        option, obligation = axes[:, column]
        for ax, values, color, marker, line in (
            (option, quota, TEAL, "o", "-"),
            (obligation, demand, demand_color, "s", "--"),
        ):
            ax.set(xlim=(.78, 7.32), ylim=(-.24, 1.55), yticks=(0, 1), xticks=(1, 2, 4, 6, 7))
            ax.grid(axis="y", linewidth=.6, color="#DCE3EA")
            ax.axvline(x[1], color="#A8B3BA", linewidth=.7, zorder=0)
            ax.step(x, values, where="post", color=color, linewidth=1.65, linestyle=line, zorder=2)
            ax.scatter(x[1:], values[1:], color=color, marker=marker, s=14, zorder=3)
        option.tick_params(axis="x", bottom=False)
        option.spines["bottom"].set_visible(False)
        option.set_title(title, loc="left", fontsize=9.1, fontweight="bold", pad=5)
        replay = row["path"] == "Isolated replay"
        if replay:
            option.plot([x[1]] * 2, [quota[1], row["execution_allowance_after"]], ":",
                        color=RUST, linewidth=1.35, zorder=2)
            option.scatter([x[1]], [row["execution_allowance_after"]], facecolors="white",
                           edgecolors=RUST, s=23, linewidths=1, zorder=4)
        option.text(x[1]+.24, 1.23, "Hold" if replay else "Dispatch legacy", fontsize=8, color=INK)
        if row["original_completed"]:
            option.text(x[-2]+.10, 1.23, "Stage target", ha="center", fontsize=8)
            obligation.text(x[-1]-.24, .09, "Target activated", ha="right", fontsize=8)
        end = "Completed" if row["original_completed"] else "Did not complete"
        fig.text(option.get_position().x0, .109, f"Original episode: {end}", color=INK, fontsize=8.0)
    axes[0, 0].set_ylabel("Allowance", fontsize=8.5)
    axes[1, 0].set_ylabel("Activation\ndemand", fontsize=8.5)
    fig.text(.53, .18, "Model decision (one-based)", ha="center", fontsize=8.7)
    fig.legend(handles=[
        Line2D([], [], color=RUST, linestyle=":", marker="o", markerfacecolor="white", markersize=4.5,
               label="Isolated replay: 0 left, legacy running"),
    ], loc="lower center", frameon=False, fontsize=8.0,
       bbox_to_anchor=(.52, .014), handletextpad=.6)
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    options = parser.parse_args()
    options.out.mkdir(parents=True, exist_ok=True)
    data = collect(options.audit_dir, options.root)
    (options.out / "candidate_chain_data.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    figure = draw(data)
    figure.savefig(options.out / "fig2_candidate_chain.pdf")
    figure.savefig(options.out / "fig2_candidate_chain.svg")
    figure.savefig(options.out / "fig2_candidate_chain.png", dpi=600)
    figure.savefig(options.out / "fig2_candidate_chain.tiff", dpi=600)
    plt.close(figure)
