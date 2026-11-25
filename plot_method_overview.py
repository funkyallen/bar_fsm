"""Draw the information boundary between write verification and offline auditing."""

from argparse import ArgumentParser
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ONLINE = "#106763"
OFFLINE = "#3C5475"


def draw(out):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "bar-fsm-method-overview",
    })
    fig = plt.figure(figsize=(7.15, 1.91), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1), xlim=(0, 7.15), ylim=(0, 1.91))
    ax.axis("off")
    width, height = 2.15, 0.37
    left, middle, right = 0.10, 2.50, 4.90

    def box(x, y, title, color, fill):
        ax.add_patch(FancyBboxPatch((x, y), width, height,
                                   boxstyle="round,pad=0.025,rounding_size=0.06",
                                   facecolor=fill, edgecolor=color, linewidth=0.75))
        ax.text(x + width / 2, y + height / 2, title, fontsize=8.6, ha="center", weight="bold", color=color,
                va="center")

    def arrow(start, end, color):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                     mutation_scale=8.5, linewidth=0.85, color=color,
                                     shrinkA=1.0, shrinkB=1.0))

    top, bottom = 1.25, 0.10
    ax.text(left, 1.75, "ONLINE   WRITE VERIFICATION", weight="bold",
            color=ONLINE, fontsize=9)
    ax.plot((left, right + width), (0.94, 0.94), color="#9DABB7",
            linewidth=0.65, dashes=(3, 2))
    ax.text(left, 0.62, "OFFLINE   REACHABILITY AUDIT", weight="bold",
            color=OFFLINE, fontsize=9)
    for y, color, fill, labels in (
        (top, ONLINE, "#EFF8F6", ("Candidate and evidence", "Local check and certificate", "Dispatch or hold")),
        (bottom, OFFLINE, "#F1F5F9", ("Restored backend state", "Native outcome or replay",
                                      "Reachability and completion")),
    ):
        for x, label in zip((left, middle, right), labels, strict=True):
            box(x, y, label, color, fill)
        for x in (left, middle):
            arrow((x + width + 0.025, y + height / 2), (x + 2.375, y + height / 2), color)
    record_x = right + width / 2
    audit_x = middle + width / 2
    ax.plot((record_x, record_x, audit_x), (top - 0.025, 0.78, 0.78), color=OFFLINE, linewidth=.8)
    arrow((audit_x, 0.78), (audit_x, bottom + height + 0.025), OFFLINE)
    ax.text((record_x + audit_x) / 2, 0.835, "Decision record", fontsize=8.0, color=OFFLINE, ha="center")

    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig_method_overview.pdf", metadata={"CreationDate": None})
    svg = out / "fig_method_overview.svg"
    fig.savefig(svg, metadata={"Date": None})
    xml = re.sub(r"<!DOCTYPE svg PUBLIC[^>]+>\n", "", svg.read_text(encoding="utf-8"), count=1)
    svg.write_text(xml, encoding="utf-8")
    fig.savefig(out / "fig_method_overview.png", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("figures"))
    draw(parser.parse_args().out)
