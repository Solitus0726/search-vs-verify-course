# plot_curves.py -- Experiment 1 budget-accuracy curves
# Input: experiment1_<model>.json (one or more, to draw a family of curves)
# Output: data/figures/curve_<model>.png
# Elements: dual-scale x axis (call count + token actual measurement), B=50 reference line,
#   equal-candidate comparison points (N=10 search vs N=10/M=4 verify, connected by dashed lines with annotations)
# Usage: python scripts/plot_curves.py --inputs data/results/experiment1_qwen3-4b.json \
#        --tokens-per-call 608 --out data/figures/curve_qwen3-4b.png

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
# Font fallback (rendered on the author's Windows side; students only view the rendered PNGs, no font dependency)
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment 1 curves (dual-scale x axis + B=50 + equal-candidate comparison)")
    ap.add_argument("--inputs", nargs="+", required=True, help="path(s) to experiment1_*.json (multiple allowed)")
    ap.add_argument("--tokens-per-call", type=float, default=608.0,
                    help="token secondary-scale conversion: average tokens per call (calibrated value 608)")
    ap.add_argument("--out", default=None, help="output PNG path (default data/figures/curve_<first model>.png)")
    args = ap.parse_args()

    datas = []
    for path in args.inputs:
        with open(path, encoding="utf-8") as f:
            datas.append(json.load(f))

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for i, data in enumerate(datas):
        color = colors[i % len(colors)]
        model = data["model"]
        search = [c for c in data["configs"] if c["config"]["strategy"] == "search"]
        verify = [c for c in data["configs"] if c["config"]["strategy"] == "verify"]
        # Search line (sorted by cost)
        sx = [c["cost"] for c in search]
        sy = [c["accuracy"] for c in search]
        if sx:
            ax.plot(sx, sy, "o-", color=color, label=f"{model} search (voting)")
        # Verify line
        vx = [c["cost"] for c in verify]
        vy = [c["accuracy"] for c in verify]
        if vx:
            ax.plot(vx, vy, "s--", color=color, label=f"{model} verify (mean-score selection)")

        # Equal-candidate comparison: N=10 search vs N=10/M=4 verify (dashed link, removing the candidate-count effect)
        s10 = next((c for c in search if c["config"]["N"] == 10), None)
        v10m4 = next((c for c in verify if c["config"]["N"] == 10 and c["config"]["M"] == 4), None)
        if s10 is not None and v10m4 is not None:
            ax.plot([s10["cost"], v10m4["cost"]], [s10["accuracy"], v10m4["accuracy"]],
                    "k:", linewidth=1.2)
            ax.annotate("Equal candidates\n(N=10)", xy=((s10["cost"] + v10m4["cost"]) / 2,
                        (s10["accuracy"] + v10m4["accuracy"]) / 2),
                        fontsize=8, ha="center", va="bottom")

    # B=50 reference line
    ax.axvline(x=50, color="gray", linestyle="--", linewidth=1.0)
    ax.text(50, ax.get_ylim()[0] if ax.get_ylim()[0] >= 0 else 0, "  B=50",
            fontsize=9, color="gray", va="bottom")

    ax.set_xlabel("Budget cost (call count)")
    ax.set_ylabel("Accuracy (MATH-500 subset)")
    ax.set_title(f"Budget-accuracy curves (dual-scale x axis: call count / token x {args.tokens_per_call:.0f})")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)

    # Secondary x axis: token scale (actual mean = call count x average tokens per call)
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim()[0] * args.tokens_per_call, ax.get_xlim()[1] * args.tokens_per_call)
    ax2.set_xlabel(f"token (actual scale approx. call count x {args.tokens_per_call:.0f})")

    out = args.out or os.path.join(PROJECT_ROOT, "data", "figures",
                                   f"curve_{os.path.basename(args.inputs[0]).replace('experiment1_', '').replace('.json', '.png')}")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Curves saved: {out}")


if __name__ == "__main__":
    main()
