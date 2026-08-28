# plot_heatmap.py -- Experiment 2 heatmap
# Input: experiment1_<model>.json (output of analyze_experiment, with accuracy for all configs)
# Output: data/figures/heatmap_<model>.png
# Elements:
#   - x axis = N (number of candidates), y axis = M (number of verifications), cell color = accuracy
#   - Each cell is labeled with accuracy + actual cost N x (1+M)
#   - Border highlight for the 4 legal combinations (5,9)(10,4)(25,1)(50,0) (equal cost at B=50)
#   - Over-budget reference points (cost > 50) are labeled separately
# Usage: python scripts/plot_heatmap.py --inputs data/results/experiment1_qwen3-4b.json

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 4 legal combinations (B=50 identity N x (1+M) = 50)
LEGAL_COMBOS = [(5, 9), (10, 4), (25, 1), (50, 0)]
BUDGET = 50


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment 2 heatmap ((N, M) accuracy + cost annotations)")
    ap.add_argument("--inputs", nargs="+", required=True, help="path(s) to experiment1_*.json (multiple allowed)")
    ap.add_argument("--out-dir", default=os.path.join(PROJECT_ROOT, "data", "figures"))
    args = ap.parse_args()

    for path in args.inputs:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        model = data["model"]
        configs = data["configs"]

        # Cell data: sparse N x M grid
        ns = sorted({c["config"]["N"] for c in configs})
        ms = sorted({c["config"]["M"] for c in configs})
        acc = {}
        cost = {}
        for c in configs:
            n, m = c["config"]["N"], c["config"]["M"]
            acc[(n, m)] = c["accuracy"]
            cost[(n, m)] = c["cost"]

        # Heatmap (pcolormesh sparse cells + annotations)
        fig, ax = plt.subplots(figsize=(10, 7))
        for (n, m), a in acc.items():
            color = plt.cm.viridis(a)
            rect = plt.Rectangle((n - 1, m - 1), 1.8, 1.8, color=color, alpha=0.9)
            ax.add_patch(rect)
            legal = (n, m) in LEGAL_COMBOS
            # Annotations: accuracy + cost (legal combinations get a highlighted border)
            ax.text(n, m + 0.35, f"{a:.2f}", ha="center", va="center",
                    fontsize=11, fontweight="bold" if legal else "normal",
                    color="white" if a < 0.5 else "black")
            ax.text(n, m - 0.35, f"cost {int(cost[(n, m)])}", ha="center", va="center", fontsize=8, color="gray")
            if legal:
                ax.add_patch(plt.Rectangle((n - 1, m - 1), 1.8, 1.8, fill=False,
                                           edgecolor="red", linewidth=2.5))

        ax.set_xlim(0, max(ns) + 1)
        ax.set_ylim(0, max(ms) + 1)
        ax.set_xticks(ns)
        ax.set_yticks(ms)
        ax.set_xlabel("Number of candidates N")
        ax.set_ylabel("Verifications per candidate M")
        ax.set_title(f"Experiment 2: optimal mix-ratio heatmap ({model}, B={BUDGET})\nRed box = legal combination N x (1+M) = {BUDGET}; each cell shows accuracy and actual cost")
        ax.grid(True, alpha=0.3)

        # Color bar
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 1))
        fig.colorbar(sm, ax=ax, label="Accuracy")

        out = os.path.join(args.out_dir, f"heatmap_{os.path.basename(path).replace('experiment1_', '').replace('.json', '.png')}")
        os.makedirs(args.out_dir, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Heatmap saved: {out} ({len(configs)} config points)")


if __name__ == "__main__":
    main()
