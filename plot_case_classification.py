"""Render the complete oracle classification and evidence resolution for Figure 1."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


ROOT = Path(__file__).resolve().parent


def main():
    data = json.loads((ROOT / "editorial_checks.json").read_text(encoding="utf-8"))["constructed_sample_classification"]
    count = data["classification"]
    resolved = data["resolution"]
    if sum(count.values()) != 720 or resolved["preserving_false_blocks"]:
        raise ValueError("The plot expects the reviewed 720-pair classification and zero preserving false blocks.")

    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 8.5, "pdf.fonttype": 42, "svg.fonttype": "none", "axes.linewidth": 0.7})
    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    fig.subplots_adjust(left=0.255, right=0.965, bottom=0.23, top=0.95)
    names = ["Oracle-ineligible", "Already unreachable", "New loss of reachability", "Preserves reachability"]
    values = [count["oracle_ineligible"], count["initially_unreachable"], count["new_loss"], count["preserving"]]
    parts = [values[0], values[1], resolved["certified_new_losses"], resolved["explicit_preserving_passes"]]
    suffixes = [str(values[0]), str(values[1]),
                f'{parts[2]} certified, {resolved["unresolved_new_losses"]} unresolved',
                f'{parts[3]} passed, {resolved["unresolved_preserving"]} unresolved']

    for i, (total, visible, suffix) in enumerate(zip(values, parts, suffixes, strict=True)):
        ax.barh(i, visible, color="#4B5058" if i < 2 else "#226E80", height=0.55)
        if total > visible:
            ax.barh(i, total - visible, left=visible, height=0.55, color="#DDE2E6",
                    edgecolor="#687581", linewidth=0.65)
        ax.text(total + 8, i, suffix, va="center", ha="left", fontsize=8.3)

    ax.set_yticks(range(len(names)), labels=names)
    ax.invert_yaxis()
    ax.set_xlim(0, 430)
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.set_xlabel("Constructed state–action pairs")
    ax.tick_params(axis="y", length=0, pad=6)
    ax.tick_params(axis="x", length=3, pad=4)
    for edge in ("top", "right", "left"):
        ax.spines[edge].set_visible(False)
    output = ROOT / "figures" / "case_classification"
    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".svg"))
    fig.savefig(output.with_suffix(".png"), dpi=600)
    fig.savefig(output.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


if __name__ == "__main__":
    main()
