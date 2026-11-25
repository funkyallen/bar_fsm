"""Produce the manuscript's evidence-channel figure from saved counts."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


root = Path(__file__).resolve().parent
data = json.loads((root / "oracle_results.json").read_text(encoding="utf-8"))["summary"]["by_evidence"]
channels = ["complete", "missing", "stale", "ambiguous"]
labels = ["Complete", "Missing", "Stale", "Ambiguous"]
groups = [
    ("true_block", "Certified dead end", "#ae4b39"),
    ("true_pass", "Preserving transition passed", "#28727a"),
    ("unknown", "Unresolved", "#dba645"),
    ("initially_unreachable", "Initially unreachable", "#7d8191"),
    ("local_precondition_failed", "Outside local contract", "#d6dde4"),
]
fig, ax = plt.subplots(figsize=(7.0, 3.05), layout="constrained")
offset = [0] * len(channels)
for field, name, color in groups:
    lengths = [data[k].get(field, 0) for k in channels]
    ax.barh(labels, lengths, left=offset, color=color, height=.57, label=name)
    for i, (start, size) in enumerate(zip(offset, lengths)):
        if size >= 13:
            ax.text(start+size/2, i, str(size), va="center", ha="center", fontsize=8,
                    color="white" if field in ("true_block", "true_pass", "initially_unreachable") else "#17212a")
    offset = [a+b for a,b in zip(offset,lengths)]
ax.set_xlim(0, 370)
ax.set_xlabel("Candidate state–action pairs")
ax.invert_yaxis()
ax.spines[["right","top","left"]].set_visible(False)
ax.legend(loc="upper left", bbox_to_anchor=(0, -.26), frameon=False, fontsize=7.8, ncol=3)
ax.tick_params(axis="y", length=0)
(root / "figures").mkdir(exist_ok=True)
fig.savefig(root / "figures/oracle_evidence.png", dpi=300)
fig.savefig(root / "figures/oracle_evidence.pdf")
